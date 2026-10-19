//! Devtools window for collections: list with counts, manual membership edits, migration controls, orphan check and timing probe.

use std::future::Future;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, Ui, Window};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::RecordId;

use crate::database::DB;
use crate::database::collections::{self, CollectionCount, ProbeStep};
use crate::database::collections_migration::{CopyOptions, FinishOptions, Migrator, Phase, Plan, RollbackOptions, RollbackReport, Status, VerifyReport};
use crate::icons::{self, svg};

const POLL: Duration = Duration::from_secs(5);
const UNINDEXED_SCAN_LIMIT: usize = 200_000;
const LISTED: usize = 200;

fn open_id() -> Id {
    Id::unique("ame_devtools_collections_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

struct Panel {
    ctx: Option<Context>,
    list: Vec<CollectionCount>,
    status: Option<Status>,
    polled: Option<Instant>,
    poll_busy: bool,
    note: Option<(bool, String)>,
    busy: Vec<&'static str>,
    target: Option<String>,
    paths_text: String,
    new_name: String,
    confirm_delete: Option<RecordId>,
    plan: Option<Plan>,
    verify: Option<VerifyReport>,
    rollback: Option<RollbackReport>,
    confirm_rollback: bool,
    force_rollback: bool,
    require_snapshot: bool,
    orphans: Option<(u64, u64)>,
    probe_n: usize,
    probe: Option<Vec<ProbeStep>>,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            ctx: None,
            list: Vec::new(),
            status: None,
            polled: None,
            poll_busy: false,
            note: None,
            busy: Vec::new(),
            target: None,
            paths_text: String::new(),
            new_name: String::new(),
            confirm_delete: None,
            plan: None,
            verify: None,
            rollback: None,
            confirm_rollback: false,
            force_rollback: false,
            require_snapshot: true,
            orphans: None,
            probe_n: 10_000,
            probe: None,
        }
    }
}

static PANEL: LazyLock<Mutex<Panel>> = LazyLock::new(|| Mutex::new(Panel::default()));

fn with<R>(f: impl FnOnce(&mut Panel) -> R) -> R {
    f(&mut PANEL.lock().unwrap_or_else(|e| e.into_inner()))
}

fn repaint() {
    if let Some(ctx) = with(|p| p.ctx.clone()) {
        ctx.request_repaint();
    }
}

fn note(ok: bool, text: impl Into<String>) {
    let text = text.into();
    with(|p| p.note = Some((ok, text)));
    repaint();
}

fn run<F: Future<Output = ()> + Send + 'static>(label: &'static str, fut: F) {
    with(|p| p.busy.push(label));
    repaint();
    tokio::spawn(async move {
        fut.await;
        with(|p| {
            if let Some(i) = p.busy.iter().position(|l| *l == label) {
                p.busy.remove(i);
            }
            p.polled = None;
        });
        repaint();
    });
}

fn busy(p: &Panel, label: &str) -> bool {
    p.busy.iter().any(|l| *l == label)
}

/// Paths from pasted text: one per line, blanks skipped, matching quotes removed.
pub fn parse_paths(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    text.lines()
        .map(|l| l.trim())
        .map(|l| {
            let quoted = l.len() >= 2 && ((l.starts_with('"') && l.ends_with('"')) || (l.starts_with('\'') && l.ends_with('\'')));
            if quoted { &l[1..l.len() - 1] } else { l }
        })
        .filter(|l| !l.is_empty())
        .filter(|l| seen.insert(l.to_string()))
        .map(str::to_string)
        .collect()
}

/// Adds or removes the pasted paths; the result is a one-line summary.
pub async fn apply_paths(db: &Surreal<Db>, id: &RecordId, text: &str, add: bool) -> anyhow::Result<String> {
    let paths = parse_paths(text);
    anyhow::ensure!(!paths.is_empty(), "no paths to use");
    if add {
        let r = crate::database::journal::add_to_collection_with(db, id, &paths, "Add to collection (devtools)").await?;
        Ok(format!("{} added, {} already in the collection", r.added, r.already))
    } else {
        let n = crate::database::journal::remove_from_collection_with(db, id, &paths, "Remove from collection (devtools)").await?;
        Ok(format!("{n} removed, {} were not in the collection", paths.len() - n))
    }
}

fn refresh() {
    let started = with(|p| !std::mem::replace(&mut p.poll_busy, true));
    if !started {
        return;
    }
    tokio::spawn(async move {
        let list = collections::list_with_counts(&DB).await;
        let status = Migrator::global().status().await;
        with(|p| {
            match list {
                Ok(l) => p.list = l,
                Err(e) => p.note = Some((false, format!("Listing collections failed: {e:#}"))),
            }
            p.status = status.ok();
            p.polled = Some(Instant::now());
            p.poll_busy = false;
        });
        repaint();
    });
}

fn maybe_refresh(ctx: &Context) {
    let due = with(|p| {
        p.ctx = Some(ctx.clone());
        p.polled.is_none_or(|t| t.elapsed() > POLL) && !p.poll_busy
    });
    if due && crate::database::is_connected() {
        refresh();
    }
}

fn create_collection(name: String) {
    run("create", async move {
        match collections::create(&DB, &name).await {
            Ok(c) => note(true, format!("Created \"{}\"", c.name)),
            Err(e) => note(false, format!("Create failed: {e}")),
        }
    });
}

fn delete_collection(id: RecordId) {
    run("delete", async move {
        match collections::delete(&DB, &id).await {
            Ok(n) => note(true, format!("Deleted the collection and {n} memberships; files were not touched")),
            Err(e) => note(false, format!("Delete failed: {e:#}")),
        }
    });
}

fn edit_paths(id: RecordId, text: String, add: bool) {
    run("paths", async move {
        match apply_paths(&DB, &id, &text, add).await {
            Ok(msg) => note(true, msg),
            Err(e) => note(false, format!("{} failed: {e:#}", if add { "Add" } else { "Remove" })),
        }
    });
}

fn dry_run() {
    run("plan", async {
        match Migrator::global().plan().await {
            Ok(p) => with(|s| s.plan = Some(p)),
            Err(e) => note(false, format!("Dry run failed: {e:#}")),
        }
    });
}

fn verify_now() {
    run("verify", async {
        match Migrator::global().verify().await {
            Ok(v) => with(|s| s.verify = Some(v)),
            Err(e) => note(false, format!("Verify failed: {e:#}")),
        }
    });
}

fn check_orphans() {
    run("orphans", async {
        let result = async { anyhow::Ok((collections::orphan_count(&DB).await?, collections::unindexed_count(&DB, UNINDEXED_SCAN_LIMIT).await?)) }.await;
        match result {
            Ok(v) => with(|s| s.orphans = Some(v)),
            Err(e) => note(false, format!("Orphan check failed: {e:#}")),
        }
    });
}

fn prune_orphans() {
    run("prune", async {
        match collections::prune_orphans(&DB).await {
            Ok(n) => {
                with(|s| s.orphans = None);
                note(true, format!("Removed {n} memberships whose collection no longer exists"));
            }
            Err(e) => note(false, format!("Prune failed: {e:#}")),
        }
    });
}

fn run_probe(n: usize) {
    run("probe", async move {
        match collections::probe(&DB, n).await {
            Ok(steps) => with(|s| s.probe = Some(steps)),
            Err(e) => note(false, format!("Probe failed: {e:#}")),
        }
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Collections")
        .open(&mut open)
        .default_width(700.0)
        .default_height(640.0)
        .show(ctx, |ui| {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, body);
        });
    set_open(ctx, open);
}

fn kv(ui: &mut Ui, key: &str, value: impl Into<egui::WidgetText>) {
    ui.label(RichText::new(key).strong());
    ui.label(value);
    ui.end_row();
}

fn body(ui: &mut Ui) {
    maybe_refresh(ui.ctx());
    let connected = crate::database::is_connected();
    let err = crate::theme::error(ui);
    let ok_color = crate::theme::success(ui);
    let warn = crate::theme::warn(ui);
    if !connected {
        ui.label(RichText::new("The database is not connected in this process; actions are disabled.").color(warn));
    }
    if let Some((ok, text)) = with(|p| p.note.clone()) {
        ui.label(RichText::new(text).color(if ok { ok_color } else { err }));
    }
    ui.horizontal(|ui| {
        if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh").clicked() {
            with(|p| p.polled = None);
            if connected {
                refresh();
            }
        }
        let n = with(|p| p.busy.len());
        if n > 0 {
            ui.spinner();
            ui.weak(with(|p| p.busy.join(", ")));
        }
    });
    list_section(ui, connected);
    paths_section(ui, connected);
    migration_section(ui, connected);
    health_section(ui, connected);
    probe_section(ui, connected);
}

fn list_section(ui: &mut Ui, connected: bool) {
    let (list, confirm) = with(|p| (p.list.clone(), p.confirm_delete.clone()));
    CollapsingHeader::new(format!("Collections ({})", list.len())).default_open(true).show(ui, |ui| {
        if list.is_empty() {
            ui.weak("No collections yet.");
        }
        Grid::new("collections_dev_list").num_columns(5).spacing([12.0, 4.0]).striped(true).show(ui, |ui| {
            for h in ["Name", "Members", "Key", "Legacy key", ""] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for c in list.iter().take(LISTED) {
                let sel = with(|p| p.target.clone());
                if ui.selectable_label(sel.as_deref() == Some(c.collection.key().as_str()), &c.collection.name).clicked() {
                    with(|p| p.target = Some(c.collection.key()));
                }
                ui.label(c.count.to_string());
                ui.label(RichText::new(c.collection.key()).monospace());
                ui.label(RichText::new(c.collection.legacy_key.clone().unwrap_or_default()).monospace());
                if ui.add_enabled(connected, egui::Button::new("Delete")).on_hover_text("Removes the collection and its memberships; files are untouched").clicked() {
                    with(|p| p.confirm_delete = Some(c.collection.id.clone()));
                }
                ui.end_row();
            }
        });
        if list.len() > LISTED {
            ui.weak(format!("{} more not shown", list.len() - LISTED));
        }
        if let Some(id) = confirm {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                let name = list.iter().find(|c| c.collection.id == id).map(|c| c.collection.name.clone()).unwrap_or_default();
                ui.label(RichText::new(format!("Delete \"{name}\"? Its memberships are removed; files and thumbnails stay.")).color(crate::theme::warn(ui)));
                ui.horizontal(|ui| {
                    if ui.button("Delete").clicked() {
                        with(|p| p.confirm_delete = None);
                        delete_collection(id.clone());
                    }
                    if ui.button("Cancel").clicked() {
                        with(|p| p.confirm_delete = None);
                    }
                });
            });
        }
        ui.horizontal(|ui| {
            let mut name = with(|p| p.new_name.clone());
            let edit = ui.add(egui::TextEdit::singleline(&mut name).hint_text("New collection name").desired_width(220.0));
            with(|p| p.new_name = name.clone());
            let go = ui.add_enabled(connected && !name.trim().is_empty(), egui::Button::new("Create"));
            if go.clicked() || (edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !name.trim().is_empty() && connected) {
                with(|p| p.new_name.clear());
                create_collection(name);
            }
        });
    });
}

fn paths_section(ui: &mut Ui, connected: bool) {
    CollapsingHeader::new("Add or remove paths").default_open(true).show(ui, |ui| {
        let list = with(|p| p.list.clone());
        let mut target = with(|p| p.target.clone());
        let label = list.iter().find(|c| Some(c.collection.key()) == target).map_or("Choose a collection".to_string(), |c| c.collection.name.clone());
        egui::ComboBox::from_id_salt("collections_dev_target").selected_text(label).width(260.0).show_ui(ui, |ui| {
            for c in &list {
                ui.selectable_value(&mut target, Some(c.collection.key()), &c.collection.name);
            }
        });
        with(|p| p.target = target.clone());
        let mut text = with(|p| p.paths_text.clone());
        ui.add(egui::TextEdit::multiline(&mut text).hint_text("One path per line").desired_rows(4).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace));
        let count = parse_paths(&text).len();
        with(|p| p.paths_text = text.clone());
        let id = target.as_deref().map(collections::collection_id);
        ui.horizontal(|ui| {
            ui.label(format!("{count} paths"));
            let ready = connected && count > 0 && id.is_some();
            if ui.add_enabled(ready, egui::Button::new("Add")).clicked() {
                if let Some(id) = id.clone() {
                    edit_paths(id, text.clone(), true);
                }
            }
            if ui.add_enabled(ready, egui::Button::new("Remove")).clicked() {
                if let Some(id) = id.clone() {
                    edit_paths(id, text.clone(), false);
                }
            }
            if ui.button("Clear").clicked() {
                with(|p| p.paths_text.clear());
            }
        });
    });
}

fn phase_color(ui: &Ui, phase: Phase) -> egui::Color32 {
    match phase {
        Phase::Finished => crate::theme::success(ui),
        Phase::Copying | Phase::Copied => crate::theme::warn(ui),
        Phase::NotStarted | Phase::RolledBack => ui.visuals().text_color(),
    }
}

fn migration_section(ui: &mut Ui, connected: bool) {
    let migrator = Migrator::global();
    let progress = migrator.progress();
    let running = migrator.is_running();
    let (status, plan, verify, rollback, mut confirm_rollback, mut force, mut snapshot, busy_plan, busy_verify) =
        with(|p| (p.status.clone(), p.plan.clone(), p.verify.clone(), p.rollback.clone(), p.confirm_rollback, p.force_rollback, p.require_snapshot, busy(p, "plan"), busy(p, "verify")));
    let err = crate::theme::error(ui);
    let warn = crate::theme::warn(ui);
    CollapsingHeader::new("Migration from logical groups").default_open(true).show(ui, |ui| {
        ui.label("The copy turns each logical group except Default into a collection and its files into memberships. The legacy groups and links are never changed, so rollback only removes collections. Default means the file is in no collection.");
        ui.weak("Large libraries: close the app and run smart-media --db-collections migrate, then finish.");
        ui.add_space(4.0);
        let state = status.as_ref().map(|s| s.state.clone()).unwrap_or_default();
        let phase = status.as_ref().map_or(Phase::NotStarted, |s| s.phase);
        Grid::new("collections_dev_migration").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label(RichText::new("Phase").strong());
            ui.label(RichText::new(phase.label()).color(phase_color(ui, phase)));
            ui.end_row();
            if state.needs_resync {
                kv(ui, "Stale", "legacy rows were imported after the last copy; run the copy again");
            }
            if let Some(s) = &status {
                kv(ui, "Legacy groups (not Default)", s.legacy_groups.to_string());
                kv(ui, "Collections", s.collections.to_string());
                kv(ui, "Memberships", s.memberships.to_string());
            }
            kv(ui, "This run", if progress.message.is_empty() { progress.step.label().to_string() } else { format!("{}: {}", progress.step.label(), progress.message) });
            kv(ui, "Progress", format!("{} of {} memberships, {} of {} groups, {} added", progress.processed, progress.memberships_total, progress.groups_done, progress.groups_total, progress.added));
            kv(ui, "Saved state", format!("{} groups mapped, {} collections created, {} memberships added, {} already present", state.groups_mapped, state.collections_created, state.memberships_added, state.memberships_existing));
            kv(ui, "Unfiled / dangling links", format!("{} rows in Default (not materialized), {} rows linked to a missing group", state.unfiled_rows, state.dangling_links));
            kv(ui, "Passes", state.runs.to_string());
            kv(ui, "Copied", state.copied_at.clone().unwrap_or_else(|| "no".to_string()));
            kv(ui, "Finished", state.finished.clone().unwrap_or_else(|| "no".to_string()));
            kv(ui, "Rolled back", state.rolled_back.clone().unwrap_or_else(|| "no".to_string()));
            kv(ui, "Snapshot", state.snapshot.clone().unwrap_or_else(|| "none yet (taken before the first write)".to_string()));
        });
        for w in &state.warnings {
            ui.label(RichText::new(w).color(warn));
        }
        if !state.renamed.is_empty() {
            CollapsingHeader::new(format!("Groups renamed to fit collection names ({})", state.renamed.len())).show(ui, |ui| {
                for r in state.renamed.iter().take(LISTED) {
                    ui.label(format!("{:?} -> {:?}", r.original, r.assigned));
                }
            });
        }
        if let Some(f) = progress.fraction().filter(|_| running) {
            ui.add(egui::ProgressBar::new(f).show_percentage());
        }
        if let Some(e) = &progress.error {
            ui.label(RichText::new(e).color(err));
        }
        ui.add_space(4.0);
        ui.checkbox(&mut snapshot, "Take a store snapshot first (reused when one from the last hour exists)");
        ui.horizontal_wrapped(|ui| {
            let idle = connected && !running;
            if ui.add_enabled(idle && !busy_plan, egui::Button::new("Dry run")).on_hover_text("Lists what the copy would create without writing anything").clicked() {
                dry_run();
            }
            let label = if state.copy_done || phase == Phase::Finished { "Migrate again" } else if state.started.is_some() { "Resume migrate" } else { "Migrate" };
            let start = ui.add_enabled_ui(idle, |ui| icons::button(ui, svg::PLAY, label).on_hover_text("Creates collections and memberships; legacy data is not changed")).inner;
            if start.clicked() {
                migrator.spawn_copy(CopyOptions { require_snapshot: snapshot, ..Default::default() });
            }
            if ui.add_enabled(idle && state.copy_done && phase != Phase::Finished, egui::Button::new("Finish")).on_hover_text("Verifies every legacy membership is present, copies any that are missing, then marks the migration finished").clicked() {
                migrator.spawn_finish(FinishOptions { require_snapshot: snapshot, repair: true });
            }
            if ui.add_enabled(idle && !busy_verify, egui::Button::new("Verify")).on_hover_text("Compares legacy links with collections without changing anything").clicked() {
                verify_now();
            }
            if ui.add_enabled(running, egui::Button::new("Cancel")).on_hover_text("Stops after the current group; progress is kept").clicked() {
                migrator.cancel();
            }
            if ui.add_enabled(idle && state.started.is_some(), egui::Button::new("Rollback...")).on_hover_text("Removes all collections and memberships; legacy groups stay").clicked() {
                confirm_rollback = true;
            }
        });
        if confirm_rollback {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new("Rollback deletes every collection and membership. It refuses when collections or memberships were made after the migration unless discarding them is allowed below.").color(warn));
                ui.checkbox(&mut force, "Discard collections and memberships made after the migration");
                ui.horizontal(|ui| {
                    if ui.button("Roll back").clicked() {
                        confirm_rollback = false;
                        with(|p| p.rollback = None);
                        let m = Migrator::global();
                        run("rollback", async move {
                            match m.run_rollback(RollbackOptions { require_snapshot: snapshot, force }).await {
                                Ok(r) => with(|s| s.rollback = Some(r)),
                                Err(e) => note(false, format!("Rollback failed: {e:#}")),
                            }
                        });
                    }
                    if ui.button("Cancel").clicked() {
                        confirm_rollback = false;
                    }
                });
            });
        }
        if let Some(r) = &rollback {
            ui.label(RichText::new(&r.summary).color(if r.ok { crate::theme::success(ui) } else { warn }));
        }
        if let Some(v) = &verify {
            ui.label(RichText::new(&v.summary).color(if v.ok { crate::theme::success(ui) } else { warn }));
            for g in v.mismatched.iter().take(LISTED) {
                ui.label(RichText::new(format!("{}: {} legacy, {} present, collection {}", g.legacy_key, g.legacy_count, g.present, g.collection.clone().unwrap_or_else(|| "missing".into()))).monospace());
            }
        }
        if let Some(p) = &plan {
            CollapsingHeader::new(format!("Dry run: {} groups, {} memberships, {} to create", p.groups.len(), p.memberships, p.to_create)).default_open(true).show(ui, |ui| {
                ui.label(format!("{} rows are in Default and stay unfiled, {} rows link to a group that does not exist, {} names change", p.unfiled_rows, p.dangling_links, p.renamed));
                for w in &p.warnings {
                    ui.label(RichText::new(w).color(warn));
                }
                ScrollArea::vertical().id_salt("collections_dev_plan").max_height(160.0).show(ui, |ui| {
                    for g in p.groups.iter().take(LISTED) {
                        let a = &g.assignment;
                        let action = if a.reuse { "reuse" } else { "create" };
                        ui.label(RichText::new(format!("{action}  {:?} -> {:?}  ({} files)", a.legacy_name, a.name, g.memberships)).monospace());
                    }
                });
            });
        }
    });
    with(|p| {
        p.confirm_rollback = confirm_rollback;
        p.force_rollback = force;
        p.require_snapshot = snapshot;
    });
}

fn health_section(ui: &mut Ui, connected: bool) {
    let (orphans, busy_orphans, busy_prune) = with(|p| (p.orphans, busy(p, "orphans"), busy(p, "prune")));
    CollapsingHeader::new("Orphan memberships").default_open(false).show(ui, |ui| {
        ui.label("Orphans point at a collection that no longer exists. Unindexed members are files with no thumbnail row; they are valid and appear once the file is indexed.");
        match orphans {
            Some((o, u)) => ui.label(format!("{o} orphan memberships, {u} members of files that are not indexed (first {UNINDEXED_SCAN_LIMIT} scanned)")),
            None => ui.weak("Not checked yet."),
        };
        ui.horizontal(|ui| {
            if ui.add_enabled(connected && !busy_orphans, egui::Button::new("Check")).on_hover_text("Reads every membership; slow on large libraries").clicked() {
                check_orphans();
            }
            if ui.add_enabled(connected && !busy_prune && orphans.is_some_and(|(o, _)| o > 0), egui::Button::new("Remove orphans")).clicked() {
                prune_orphans();
            }
        });
    });
}

fn probe_section(ui: &mut Ui, connected: bool) {
    let (mut n, probe, busy_probe) = with(|p| (p.probe_n, p.probe.clone(), busy(p, "probe")));
    CollapsingHeader::new("Timing probe").default_open(false).show(ui, |ui| {
        ui.label("Creates a temporary collection of synthetic paths, times each API call, then deletes it.");
        ui.horizontal(|ui| {
            ui.label("Paths");
            ui.add(egui::DragValue::new(&mut n).range(100..=200_000).speed(100.0));
            if ui.add_enabled(connected && !busy_probe, egui::Button::new("Run probe")).clicked() {
                run_probe(n);
            }
        });
        if let Some(steps) = &probe {
            Grid::new("collections_dev_probe").num_columns(3).spacing([12.0, 2.0]).show(ui, |ui| {
                for s in steps {
                    ui.label(&s.step);
                    ui.label(format!("{} ms", s.millis));
                    ui.weak(&s.detail);
                    ui.end_row();
                }
            });
        }
    });
    with(|p| p.probe_n = n);
}

#[cfg(test)]
mod tests;
