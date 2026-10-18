use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use chrono::Local;
use eframe::egui;
use serde_json::{Value, json};

use super::snapshot::{
    self, EngineChange, MaintenanceSettings, PendingRestore, ProgressFn, SnapshotInfo, SnapshotKind, StampState, human, log_event,
    snapshots_dir, store_has_data, store_in_use, store_name,
};
use super::surql::{self, ExportOptions, VerifyOptions, VerifyTarget};
use super::thumb_blobs::{self, QuickStats};
use super::thumb_migration::MigrationState;

pub use super::snapshot::override_snapshot_once;

const REFRESH_EVERY: Duration = Duration::from_secs(15);

fn store_path() -> PathBuf {
    PathBuf::from(super::get_db_path())
}

/// Engine, stamp, store and snapshot state as JSON; reads the filesystem only and never opens the store.
pub fn info_json(store: &Path) -> Value {
    let versions = snapshot::engine_versions();
    let has_data = store_has_data(store);
    let stamp = snapshot::read_stamp(store);
    let change = snapshot::classify(store, has_data, &stamp, versions);
    let stamp_json = match &stamp {
        StampState::Missing => Value::Null,
        StampState::Unreadable(e) => json!({ "unreadable": e }),
        StampState::Present(s) => serde_json::to_value(s).unwrap_or(Value::Null),
    };
    json!({
        "engine": { "surrealdb": versions.surrealdb, "surrealkv": versions.surrealkv, "app": env!("CARGO_PKG_VERSION") },
        "store": {
            "path": store,
            "exists": store.exists(),
            "has_data": has_data,
            "size_bytes": snapshot::dir_size(store),
            "in_use": store_in_use(store),
            "snapshots_dir": snapshots_dir(store),
        },
        "stamp": stamp_json,
        "engine_change": { "state": change.label(), "detail": change },
        "snapshots": snapshot::list_snapshots_for(store),
        "thumb_migration": MigrationState::load(store),
        "settings": MaintenanceSettings::load(store),
        "pending_restore": snapshot::pending_restore(store),
    })
}

#[derive(Clone)]
struct StoreInfo {
    path: PathBuf,
    exists: bool,
    size: u64,
    in_use: bool,
    stamp: StampState,
    change: EngineChange,
    pending: Option<PendingRestore>,
    migration: MigrationState,
}

fn read_store_info(store: &Path) -> StoreInfo {
    let has_data = store_has_data(store);
    let stamp = snapshot::read_stamp(store);
    StoreInfo {
        path: store.to_path_buf(),
        exists: store.exists(),
        size: snapshot::dir_size(store),
        in_use: store_in_use(store),
        change: snapshot::classify(store, has_data, &stamp, snapshot::engine_versions()),
        stamp,
        pending: snapshot::pending_restore(store),
        migration: MigrationState::load(store),
    }
}

#[derive(Clone)]
struct Op {
    id: u64,
    label: String,
    cur: u64,
    total: u64,
    detail: String,
    started: Instant,
}

#[derive(Clone)]
enum Confirm {
    Restore(SnapshotInfo),
    Delete(SnapshotInfo),
    ImportCurrent(PathBuf),
}

#[derive(Default)]
struct Panel {
    settings: Option<MaintenanceSettings>,
    snapshots: Vec<SnapshotInfo>,
    store: Option<StoreInfo>,
    last_refresh: Option<Instant>,
    refreshing: bool,
    stale: bool,
    ops: Vec<Op>,
    next_op: u64,
    confirm: Option<Confirm>,
    row_results: HashMap<PathBuf, (bool, String)>,
    blobs: Option<QuickStats>,
    ctx: Option<egui::Context>,
    hooked: bool,
}

static PANEL: LazyLock<Mutex<Panel>> = LazyLock::new(|| Mutex::new(Panel::default()));

fn with<R>(f: impl FnOnce(&mut Panel) -> R) -> R {
    let mut g = PANEL.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut g)
}

fn repaint() {
    if let Some(ctx) = with(|p| p.ctx.clone()) {
        ctx.request_repaint();
    }
}

fn spawn<F: Future<Output = ()> + Send + 'static>(fut: F) {
    match tokio::runtime::Handle::try_current() {
        Ok(h) => {
            h.spawn(fut);
        }
        Err(_) => {
            std::thread::spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    rt.block_on(fut);
                }
            });
        }
    }
}

fn mark_stale() {
    with(|p| p.stale = true);
    repaint();
}

fn begin_op(label: &str) -> u64 {
    let id = with(|p| {
        p.next_op += 1;
        let id = p.next_op;
        p.ops.push(Op { id, label: label.to_string(), cur: 0, total: 0, detail: String::new(), started: Instant::now() });
        id
    });
    repaint();
    id
}

fn end_op(id: u64) {
    with(|p| p.ops.retain(|o| o.id != id));
    mark_stale();
}

fn op_progress(id: u64, prefix: &'static str) -> ProgressFn {
    let status = snapshot::status_progress(prefix);
    let last = Mutex::new(Instant::now() - Duration::from_secs(1));
    Arc::new(move |cur, total, what| {
        status(cur, total, what);
        let Ok(mut l) = last.lock() else { return };
        if cur < total && l.elapsed() < Duration::from_millis(80) {
            return;
        }
        *l = Instant::now();
        with(|p| {
            if let Some(o) = p.ops.iter_mut().find(|o| o.id == id) {
                o.cur = cur;
                o.total = total;
                o.detail = what.to_string();
            }
        });
        repaint();
    })
}

fn set_result(path: &Path, ok: bool, text: impl Into<String>) {
    let text = text.into();
    with(|p| {
        p.row_results.insert(path.to_path_buf(), (ok, text));
    });
    repaint();
}

fn fresh_export_path(store: &Path, label: &str) -> PathBuf {
    let base = snapshot::snapshot_file_name(&store_name(store), label, &snapshot::engine_versions().surrealdb, Local::now());
    snapshot::unique_path(&snapshots_dir(store), &base, ".surql")
}

/// Directory snapshot of the configured store, labelled "manual" (hot copies are flagged in the manifest).
pub async fn op_snapshot_now() {
    let id = begin_op("Snapshot now");
    match snapshot::snapshot_now_with(op_progress(id, "Snapshot")).await {
        Ok(info) => log_event(log::Level::Info, format!("Snapshot created: {}", info.path.display())),
        Err(e) => log_event(log::Level::Error, format!("Snapshot failed: {}", e.user_message())),
    }
    end_op(id);
}

/// Export the open database to a .surql in the snapshots folder, optionally verifying the file afterwards.
pub async fn op_export(verify: bool) {
    let label = if verify { "Export + Verify" } else { "Export .surql" };
    let id = begin_op(label);
    let store = store_path();
    if !super::is_connected() {
        log_event(log::Level::Error, "Export needs an open database; none is connected in this process");
        end_op(id);
        return;
    }
    let path = fresh_export_path(&store, "export");
    let progress = op_progress(id, "Export");
    let include_thumb_blobs = MaintenanceSettings::load(&store).export_include_thumb_blobs;
    let _ga = super::db_activity("Export DB");
    match surql::export_atomic_with(&super::DB, &path, &progress, ExportOptions { include_thumb_blobs }).await {
        Ok(rep) => {
            log_event(log::Level::Info, format!("Exported {} in {:.1}s: {}", human(rep.bytes), rep.millis as f32 / 1000.0, path.display()));
            if verify {
                run_verify_file(&path, &progress, include_thumb_blobs).await;
            }
            let settings = snapshot::MaintenanceSettings::load(&store);
            snapshot::prune_snapshots(&store, &settings, Local::now());
        }
        Err(e) => log_event(log::Level::Error, format!("Export failed: {e}")),
    }
    end_op(id);
}

async fn run_verify_file(path: &Path, progress: &ProgressFn, include_thumb_blobs: bool) {
    let target = if super::is_connected() { VerifyTarget::Live(&super::DB) } else { VerifyTarget::FileOnly };
    match surql::verify_export(path, target, &VerifyOptions { include_thumb_blobs, ..Default::default() }, progress).await {
        Ok(r) if r.ok => set_result(path, true, format!("OK: {} records, {} tables ({})", r.imported.total_records(), r.imported.tables.len(), r.backend)),
        Ok(r) => set_result(path, false, format!("FAILED: {}", r.problems.first().cloned().unwrap_or_else(|| "see log".into()))),
        Err(e) => set_result(path, false, format!("FAILED: {e}")),
    }
}

/// Verify a snapshot row: hash check for directories, scratch import for .surql files.
pub async fn op_verify(info: SnapshotInfo) {
    let id = begin_op(&format!("Verify {}", info.name));
    let progress = op_progress(id, "Verify");
    match info.kind {
        SnapshotKind::Surql => run_verify_file(&info.path, &progress, false).await,
        SnapshotKind::Dir => {
            let dir = info.path.clone();
            let p = progress.clone();
            let res = tokio::task::spawn_blocking(move || snapshot::verify_snapshot_dir(&dir, &p)).await;
            match res {
                Ok(Ok(c)) => {
                    let mut text = format!("{}: {} files, {}", if c.ok { "OK" } else { "FAILED" }, c.files_checked, human(c.bytes_checked));
                    if !c.has_manifest {
                        text.push_str(" (legacy: no manifest)");
                    }
                    let mut ok = c.ok;
                    if ok && MaintenanceSettings::load(&store_path()).deep_verify_snapshots {
                        match surql::digest_snapshot_copy(&info.path, &progress).await {
                            Ok(m) => text.push_str(&format!("; opens, {} records", m.total_records())),
                            Err(e) => {
                                ok = false;
                                text = format!("FAILED: copy does not open: {e}");
                            }
                        }
                    }
                    log_event(if ok { log::Level::Info } else { log::Level::Warn }, format!("Verify {}: {text}", info.path.display()));
                    set_result(&info.path, ok, text);
                }
                Ok(Err(e)) => set_result(&info.path, false, format!("FAILED: {}", e.user_message())),
                Err(e) => set_result(&info.path, false, format!("FAILED: {e}")),
            }
        }
    }
    end_op(id);
}

/// Restore a directory snapshot (staged until restart while connected) or load a .surql into a new store.
pub async fn op_restore(info: SnapshotInfo) {
    let store = store_path();
    let id = begin_op(&format!("Restore {}", info.name));
    let progress = op_progress(id, "Restore");
    match info.kind {
        SnapshotKind::Dir if super::is_connected() => match snapshot::stage_restore(&store, &info.path) {
            Ok(()) => set_result(&info.path, true, "Restore staged: restart the app to apply"),
            Err(e) => log_event(log::Level::Error, format!("Could not stage restore: {e}")),
        },
        SnapshotKind::Dir => match snapshot::restore_snapshot_to(info.path.clone(), store.clone(), progress).await {
            Ok(r) => {
                set_result(&info.path, true, format!("Restored; previous store kept at {}", r.replaced_to.map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into())));
                super::reopen_with_last_tx();
            }
            Err(e) => log_event(log::Level::Error, format!("Restore failed: {}", e.user_message())),
        },
        SnapshotKind::Surql => {
            let target = snapshot::unique_path(&snapshot::db_parent(&store), &format!("{}-from-{}", store_name(&store), info.name.trim_end_matches(".surql")), "");
            op_import_new_inner(&info.path, &target, &progress).await;
        }
    }
    end_op(id);
}

async fn op_import_new_inner(file: &Path, dir: &Path, progress: &ProgressFn) {
    match surql::import_into_new_store(file, dir, progress).await {
        Ok(rep) => log_event(
            log::Level::Info,
            format!("Imported {} records into new store {}; choose it with 'Choose DB Folder' and restart to use it", rep.total_records, rep.dir.display()),
        ),
        Err(e) => log_event(log::Level::Error, format!("Import into new store failed: {e}")),
    }
}

/// Import a .surql file into a freshly created SurrealKV folder.
pub async fn op_import_new(file: PathBuf, dir: PathBuf) {
    let id = begin_op("Import into new store");
    op_import_new_inner(&file, &dir, &op_progress(id, "Import")).await;
    end_op(id);
}

/// Import a .surql file into the open database.
pub async fn op_import_current(file: PathBuf) {
    let id = begin_op("Import into current DB");
    if !super::is_connected() {
        log_event(log::Level::Error, "Import needs an open database; none is connected in this process");
    } else {
        match super::import_from(&file).await {
            Ok(()) => log_event(log::Level::Info, format!("Imported {} into the current database", file.display())),
            Err(e) => log_event(log::Level::Error, format!("Import failed: {e}")),
        }
    }
    end_op(id);
}

/// Delete one snapshot or export file.
pub async fn op_delete(info: SnapshotInfo) {
    let store = store_path();
    let res = tokio::task::spawn_blocking({
        let path = info.path.clone();
        move || snapshot::delete_snapshot(&store, &path)
    })
    .await;
    match res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => log_event(log::Level::Error, format!("Delete failed: {}", e.user_message())),
        Err(e) => log_event(log::Level::Error, format!("Delete failed: {e}")),
    }
    mark_stale();
}

fn refresh_now() {
    let started = with(|p| {
        if p.refreshing {
            false
        } else {
            p.refreshing = true;
            true
        }
    });
    if !started {
        return;
    }
    let store = store_path();
    spawn(async move {
        let s2 = store.clone();
        let result = tokio::task::spawn_blocking(move || (snapshot::list_snapshots_for(&s2), read_store_info(&s2), MaintenanceSettings::load(&s2))).await;
        let blobs = if super::is_connected() { thumb_blobs::quick_stats().await.ok() } else { None };
        with(|p| {
            p.blobs = blobs;
            if let Ok((list, info, settings)) = result {
                p.snapshots = list;
                p.store = Some(info);
                if p.settings.is_none() {
                    p.settings = Some(settings);
                }
            }
            p.refreshing = false;
            p.stale = false;
            p.last_refresh = Some(Instant::now());
        });
        repaint();
    });
}

fn pick_file_then(confirm: bool) {
    spawn(async move {
        let picked = tokio::task::spawn_blocking(|| rfd::FileDialog::new().set_title("Choose a .surql file").add_filter("SurrealQL", &["surql"]).pick_file()).await.ok().flatten();
        let Some(file) = picked else { return };
        if confirm {
            with(|p| p.confirm = Some(Confirm::ImportCurrent(file)));
            repaint();
            return;
        }
        let dir = tokio::task::spawn_blocking(|| rfd::FileDialog::new().set_title("Choose an empty folder for the new store").pick_folder()).await.ok().flatten();
        if let Some(dir) = dir {
            op_import_new(file, dir).await;
        }
    });
}

enum Action {
    Refresh,
    SnapshotNow,
    Export(bool),
    ImportCurrent,
    ImportNew,
    OpenFolder(PathBuf),
    Verify(SnapshotInfo),
    AskRestore(SnapshotInfo),
    AskDelete(SnapshotInfo),
    ConfirmYes,
    ConfirmNo,
    SaveSettings(MaintenanceSettings),
    OverrideAndRetry,
    CancelPending,
}

fn apply(action: Action) {
    match action {
        Action::Refresh => mark_stale(),
        Action::SnapshotNow => spawn(op_snapshot_now()),
        Action::Export(verify) => spawn(op_export(verify)),
        Action::ImportCurrent => pick_file_then(true),
        Action::ImportNew => pick_file_then(false),
        Action::OpenFolder(p) => {
            if let Err(e) = open::that_detached(&p) {
                log_event(log::Level::Warn, format!("Could not open {}: {e}", p.display()));
            }
        }
        Action::Verify(info) => spawn(op_verify(info)),
        Action::AskRestore(info) => with(|p| p.confirm = Some(Confirm::Restore(info))),
        Action::AskDelete(info) => with(|p| p.confirm = Some(Confirm::Delete(info))),
        Action::ConfirmNo => with(|p| p.confirm = None),
        Action::ConfirmYes => {
            if let Some(c) = with(|p| p.confirm.take()) {
                match c {
                    Confirm::Restore(info) => spawn(op_restore(info)),
                    Confirm::Delete(info) => spawn(op_delete(info)),
                    Confirm::ImportCurrent(file) => spawn(op_import_current(file)),
                }
            }
        }
        Action::SaveSettings(s) => {
            if let Err(e) = s.save(&store_path()) {
                log_event(log::Level::Error, format!("Could not save maintenance settings: {e}"));
            }
            with(|p| p.settings = Some(s));
        }
        Action::OverrideAndRetry => {
            override_snapshot_once();
            if !super::reopen_with_last_tx() {
                log_event(log::Level::Warn, "No earlier open attempt to retry; choose the database folder again");
            }
        }
        Action::CancelPending => {
            snapshot::clear_pending_restore(&store_path());
            with(|p| p.row_results.retain(|_, (_, t)| !t.starts_with("Restore staged")));
            log_event(log::Level::Info, "Staged restore cancelled");
            mark_stale();
        }
    }
    repaint();
}

struct View {
    settings: MaintenanceSettings,
    snapshots: Vec<SnapshotInfo>,
    store: Option<StoreInfo>,
    ops: Vec<Op>,
    confirm: Option<Confirm>,
    results: HashMap<PathBuf, (bool, String)>,
    blobs: Option<QuickStats>,
    refreshing: bool,
}

fn view(ctx: &egui::Context) -> View {
    let (need_refresh, v) = with(|p| {
        if !p.hooked {
            p.hooked = true;
            let c = ctx.clone();
            snapshot::set_event_hook(Some(Arc::new(move || c.request_repaint())));
        }
        p.ctx = Some(ctx.clone());
        let due = p.stale || p.last_refresh.map(|t| t.elapsed() > REFRESH_EVERY).unwrap_or(true);
        (
            due && !p.refreshing,
            View {
                settings: p.settings.clone().unwrap_or_default(),
                snapshots: p.snapshots.clone(),
                store: p.store.clone(),
                ops: p.ops.clone(),
                confirm: p.confirm.clone(),
                results: p.row_results.clone(),
                blobs: p.blobs,
                refreshing: p.refreshing,
            },
        )
    });
    if need_refresh {
        refresh_now();
    }
    v
}

fn fmt_time(t: Option<chrono::DateTime<Local>>) -> String {
    t.map(|t| t.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_else(|| "-".into())
}

fn cell(ui: &mut egui::Ui, width: f32, text: egui::RichText, hover: &str) {
    ui.allocate_ui_with_layout(egui::vec2(width, ui.spacing().interact_size.y), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.add(egui::Label::new(text).truncate()).on_hover_text(hover);
    });
}

fn kv(ui: &mut egui::Ui, key: &str, value: impl Into<egui::WidgetText>) {
    ui.label(egui::RichText::new(key).strong());
    ui.label(value);
    ui.end_row();
}

fn blob_summary(blobs: Option<QuickStats>, store_size: Option<u64>) -> String {
    let Some(q) = blobs else { return "unknown (database not connected)".into() };
    if q.stats.count == 0 {
        return "none stored".into();
    }
    let approx = if q.estimated { "about " } else { "" };
    let share = store_size.filter(|s| *s > 0).map(|s| format!(", {approx}{:.0}% of the store folder", q.stats.total_bytes as f64 * 100.0 / s as f64)).unwrap_or_default();
    format!("{} blobs, {approx}{}{share}", q.stats.count, human(q.stats.total_bytes))
}

fn migration_summary(m: &MigrationState) -> String {
    if let Some(done) = &m.finished {
        return format!("migrated to blobs and cleared ({} rows, finished {})", m.cleared, done.get(..10).unwrap_or(done));
    }
    if m.copy_done {
        return format!("copied to blobs ({} migrated, {} failed); finish step pending", m.migrated, m.failed);
    }
    if m.started.is_some() {
        return format!("copy in progress or interrupted ({} migrated, {} failed)", m.migrated, m.failed);
    }
    "not migrated (open Devtools > Thumbnail storage)".into()
}

fn status_text(ui: &egui::Ui, ok: bool, text: &str) -> egui::RichText {
    let v = ui.visuals();
    egui::RichText::new(text).color(if ok { v.text_color() } else { v.error_fg_color })
}

/// Self-contained database maintenance panel (engine stamp, snapshots, exports, verify/restore, settings, log).
pub fn show(ui: &mut egui::Ui) {
    let v = view(ui.ctx());
    let mut actions: Vec<Action> = Vec::new();
    let warn = ui.visuals().warn_fg_color;
    let err = ui.visuals().error_fg_color;
    let versions = snapshot::engine_versions();
    let connected = super::is_connected();

    ui.heading("Database maintenance");
    ui.add_space(4.0);

    if let Some(b) = snapshot::blocked() {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new("Database open was blocked").color(err).strong());
            ui.label(b.message);
            if b.overridable && ui.button("Open without snapshot (this time only)").clicked() {
                actions.push(Action::OverrideAndRetry);
            }
        });
        ui.add_space(4.0);
    }

    if let Some(p) = v.store.as_ref().and_then(|s| s.pending.as_ref()) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new(format!("Restore staged from {}", p.snapshot.display())).color(warn));
            ui.label("Applied the next time the database opens (restart the app). The current store is moved aside, never deleted.");
            if ui.button("Cancel staged restore").clicked() {
                actions.push(Action::CancelPending);
            }
        });
        ui.add_space(4.0);
    }

    egui::CollapsingHeader::new("Engine and store").default_open(true).show(ui, |ui| {
        egui::Grid::new("maint_engine_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            kv(ui, "SurrealDB (compiled)", versions.surrealdb.clone());
            kv(ui, "SurrealKV (compiled)", versions.surrealkv.clone().unwrap_or_else(|| "unknown".into()));
            kv(ui, "App version", env!("CARGO_PKG_VERSION"));
            kv(ui, "Connected in this process", if connected { "yes" } else { "no" });
            if let Some(s) = &v.store {
                kv(ui, "Store path", s.path.display().to_string());
                kv(ui, "Store size", if s.exists { human(s.size) } else { "folder does not exist".into() });
                kv(ui, "Store lock held", if s.in_use { "yes" } else { "no" });
                match &s.stamp {
                    StampState::Present(st) => {
                        kv(ui, "Engine stamp", format!("surrealdb {} / surrealkv {}", st.surrealdb_version, st.surrealkv_version.clone().unwrap_or_else(|| "?".into())));
                        kv(ui, "Last opened", st.last_opened_rfc3339.clone());
                        kv(ui, "Stamped store", st.store_path.clone());
                        kv(ui, "Stamped app version", st.app_version.clone());
                    }
                    StampState::Missing => kv(ui, "Engine stamp", "none (engine that last wrote this store is unknown)"),
                    StampState::Unreadable(e) => kv(ui, "Engine stamp", format!("unreadable: {e}")),
                }
                let change = match &s.change {
                    EngineChange::NoStore => "no existing store".to_string(),
                    EngineChange::Unchanged => "unchanged since last open".to_string(),
                    EngineChange::Changed { reason, .. } => format!("CHANGED: {reason}"),
                };
                ui.label(egui::RichText::new("Engine status").strong());
                ui.label(egui::RichText::new(change).color(if s.change.is_changed() { warn } else { ui.visuals().text_color() }));
                ui.end_row();
            } else {
                kv(ui, "Store", "loading...");
            }
            if let Some(o) = snapshot::last_open_summary() {
                kv(ui, "Last open: engine", o.change.label());
                kv(ui, "Last open: pre-snapshot", o.snapshot.map(|p| p.display().to_string()).or(o.note).unwrap_or_else(|| "none needed".into()));
                if let Some(p) = o.auto_export {
                    kv(ui, "Last open: auto export", p.display().to_string());
                }
                if let Some(p) = o.restored {
                    kv(ui, "Last open: restored from", p.display().to_string());
                }
            }
            kv(ui, "Thumbnail blobs", blob_summary(v.blobs, v.store.as_ref().map(|s| s.size)));
            if let Some(s) = &v.store {
                kv(ui, "Inline thumbnails", migration_summary(&s.migration));
            }
            kv(ui, "Snapshots folder", v.store.as_ref().map(|s| snapshots_dir(&s.path).display().to_string()).unwrap_or_default());
        });
    });

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        let idle = v.ops.is_empty();
        if ui.add_enabled(idle, egui::Button::new("Snapshot now")).on_hover_text("Copy the store folder into the snapshots folder").clicked() {
            actions.push(Action::SnapshotNow);
        }
        if ui.add_enabled(idle && connected, egui::Button::new("Export .surql now")).on_hover_text("Write a SurrealQL export of the open database").clicked() {
            actions.push(Action::Export(false));
        }
        if ui.add_enabled(idle && connected, egui::Button::new("Export + Verify")).on_hover_text("Export, then import the file into a scratch store and compare record counts and ids").clicked() {
            actions.push(Action::Export(true));
        }
        if ui.add_enabled(idle && connected, egui::Button::new("Import .surql into current DB")).on_hover_text("Runs the file against the open database").clicked() {
            actions.push(Action::ImportCurrent);
        }
        if ui.add_enabled(idle, egui::Button::new("Import .surql into NEW store")).on_hover_text("Create a fresh store folder on the running engine from a .surql file").clicked() {
            actions.push(Action::ImportNew);
        }
        if ui.button("Open snapshots folder").clicked() {
            let dir = v.store.as_ref().map(|s| snapshots_dir(&s.path)).unwrap_or_else(|| snapshots_dir(&store_path()));
            let _ = std::fs::create_dir_all(&dir);
            actions.push(Action::OpenFolder(dir));
        }
        if ui.button("Refresh").clicked() {
            actions.push(Action::Refresh);
        }
        if v.refreshing {
            ui.spinner();
        }
    });

    if let Some(c) = &v.confirm {
        let text = match c {
            Confirm::Restore(i) if connected && i.kind == SnapshotKind::Dir => format!(
                "Stage restore of {}? On the next start the current store is moved to a .replaced folder (not deleted) and this snapshot is copied into place.",
                i.name
            ),
            Confirm::Restore(i) if i.kind == SnapshotKind::Dir => format!("Restore {} now? The current store is moved to a .replaced folder (not deleted).", i.name),
            Confirm::Restore(i) => format!("Import {} into a new store folder next to the current one?", i.name),
            Confirm::Delete(i) => format!("Permanently delete {}?", i.name),
            Confirm::ImportCurrent(p) => format!("Run {} against the open database? Existing definitions and records may be changed. Take a snapshot first.", p.display()),
        };
        egui::Frame::group(ui.style()).stroke(egui::Stroke::new(1.0, warn)).show(ui, |ui| {
            ui.label(egui::RichText::new(text).color(warn));
            ui.horizontal(|ui| {
                if ui.button("Confirm").clicked() {
                    actions.push(Action::ConfirmYes);
                }
                if ui.button("Cancel").clicked() {
                    actions.push(Action::ConfirmNo);
                }
            });
        });
    }

    for op in &v.ops {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("{} ({}s)", op.label, op.started.elapsed().as_secs()));
            if op.total > 0 {
                ui.add(egui::ProgressBar::new((op.cur as f32 / op.total as f32).clamp(0.0, 1.0)).desired_width(220.0).show_percentage());
            } else if op.cur > 0 {
                ui.label(human(op.cur));
            }
            if !op.detail.is_empty() {
                ui.label(egui::RichText::new(&op.detail).weak());
            }
        });
        ui.ctx().request_repaint_after(Duration::from_millis(200));
    }

    ui.separator();
    egui::CollapsingHeader::new("Settings").default_open(true).show(ui, |ui| {
        let mut s = v.settings.clone();
        let mut gib = s.export_size_threshold_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        ui.checkbox(&mut s.snapshot_on_engine_change, "Snapshot the store folder before opening it with a different engine version");
        ui.checkbox(&mut s.auto_export_on_engine_change, "Write a .surql export after the first successful open on a new engine");
        ui.checkbox(&mut s.deep_verify_snapshots, "Deep verify: also open a temporary copy when verifying a snapshot folder");
        ui.checkbox(&mut s.export_include_thumb_blobs, "Include thumbnail blobs in .surql exports (large; thumbnails can be regenerated)");
        ui.horizontal(|ui| {
            ui.label("Skip auto export when the store is larger than");
            ui.add(egui::DragValue::new(&mut gib).range(0.1..=4096.0).speed(0.1).suffix(" GiB"));
        });
        ui.horizontal(|ui| {
            ui.label("Keep newest");
            ui.add(egui::DragValue::new(&mut s.retention_keep).range(1..=200));
            ui.label("snapshots per kind; never delete ones younger than");
            ui.add(egui::DragValue::new(&mut s.retention_min_age_days).range(0..=365).suffix(" days"));
        });
        s.export_size_threshold_bytes = (gib * 1024.0 * 1024.0 * 1024.0) as u64;
        if s != v.settings {
            actions.push(Action::SaveSettings(s));
        }
    });

    ui.separator();
    ui.label(egui::RichText::new(format!("Snapshots and exports ({})", v.snapshots.len())).strong());
    egui::ScrollArea::both().id_salt("maint_snapshots").max_height(250.0).auto_shrink([false, true]).show(ui, |ui| {
        egui::Grid::new("maint_snapshot_grid").striped(true).num_columns(7).spacing([10.0, 4.0]).show(ui, |ui| {
            for h in ["Name", "Kind", "Engine", "Size", "Created", "Check", "Actions"] {
                ui.label(egui::RichText::new(h).strong());
            }
            ui.end_row();
            let busy = !v.ops.is_empty();
            for info in &v.snapshots {
                cell(ui, 270.0, egui::RichText::new(&info.name), &info.path.display().to_string());
                ui.label(info.kind.label());
                ui.label(info.engine_version.clone().unwrap_or_else(|| "?".into()));
                ui.label(human(info.size_bytes));
                ui.label(fmt_time(info.created));
                match v.results.get(&info.path) {
                    Some((ok, text)) => cell(ui, 150.0, status_text(ui, *ok, text), text),
                    None => cell(ui, 150.0, egui::RichText::new(if info.kind == SnapshotKind::Dir && !info.has_manifest { "legacy (no manifest)" } else { "-" }), ""),
                }
                ui.horizontal(|ui| {
                    if ui.small_button("Reveal").clicked() {
                        let target = if info.kind == SnapshotKind::Dir { info.path.clone() } else { info.path.parent().map(Path::to_path_buf).unwrap_or_else(|| info.path.clone()) };
                        actions.push(Action::OpenFolder(target));
                    }
                    if ui.add_enabled(!busy, egui::Button::new("Verify").small()).clicked() {
                        actions.push(Action::Verify(info.clone()));
                    }
                    let restore_label = if info.kind == SnapshotKind::Dir { "Restore" } else { "Import new" };
                    if ui.add_enabled(!busy, egui::Button::new(restore_label).small()).clicked() {
                        actions.push(Action::AskRestore(info.clone()));
                    }
                    if ui.add_enabled(!busy, egui::Button::new("Delete").small()).clicked() {
                        actions.push(Action::AskDelete(info.clone()));
                    }
                });
                ui.end_row();
            }
        });
        if v.snapshots.is_empty() {
            ui.label(egui::RichText::new("No snapshots yet.").weak());
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Maintenance log").strong());
    egui::ScrollArea::vertical().id_salt("maint_log").max_height(180.0).auto_shrink([false, true]).stick_to_bottom(true).show(ui, |ui| {
        for e in snapshot::recent_events() {
            let color = match e.level {
                log::Level::Error => err,
                log::Level::Warn => warn,
                _ => ui.visuals().text_color(),
            };
            ui.label(egui::RichText::new(format!("{}  {}", e.at.format("%H:%M:%S"), e.msg)).monospace().color(color));
        }
    });

    for a in actions {
        apply(a);
    }
}
