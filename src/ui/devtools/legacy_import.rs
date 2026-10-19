//! Devtools window for importing a legacy SurrealDB 2.x library (for example a Windows-side store).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use eframe::egui::{self, CollapsingHeader, ComboBox, Context, Grid, Id, RichText, ScrollArea, Ui, Window};

use crate::database::legacy::convert::{ConvertReport, is_secret_field};
use crate::database::legacy::engine::{self, EngineOptions};
use crate::database::legacy::job::{EngineStatus, Job, PHASES, PhaseState, PhaseView};
use crate::database::legacy::pathmap::{Config, Policy, Rule};
use crate::database::legacy::{self, ImportRequest, LegacyReport, Prepared, Target};
use crate::database::snapshot::human;
use crate::icons::{self, svg};

fn open_id() -> Id {
    Id::unique("ame_devtools_legacy_import_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModeSel {
    Merge,
    NewStore,
}

struct Panel {
    old_store: String,
    layout: (String, Option<engine::StoreKind>),
    mode: ModeSel,
    new_store: String,
    work_dir: String,
    surreal_bin: String,
    old_ns: String,
    old_db: String,
    cfg: Config,
    map_probed: bool,
    map_file: Option<PathBuf>,
    map_generated: bool,
    map_warnings: Vec<String>,
    map_unresolved: Vec<String>,
    settings_sel: BTreeSet<String>,
    drop_b64: bool,
    keep_export: bool,
    job: Option<Arc<Job>>,
    prepared: Option<Arc<Prepared>>,
    dry: Option<Arc<ConvertReport>>,
    busy: Option<&'static str>,
    report: Option<Arc<LegacyReport>>,
    report_path: Option<PathBuf>,
    error: Option<String>,
    note: Option<(bool, String)>,
    confirm_import: bool,
    snapshot_note: Option<(bool, String)>,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            old_store: String::new(),
            layout: (String::new(), None),
            mode: ModeSel::Merge,
            new_store: String::new(),
            work_dir: String::new(),
            surreal_bin: String::new(),
            old_ns: String::new(),
            old_db: String::new(),
            cfg: Config::default(),
            map_probed: false,
            map_file: None,
            map_generated: false,
            map_warnings: Vec::new(),
            map_unresolved: Vec::new(),
            settings_sel: BTreeSet::new(),
            drop_b64: false,
            keep_export: false,
            job: None,
            prepared: None,
            dry: None,
            busy: None,
            report: None,
            report_path: None,
            error: None,
            note: None,
            confirm_import: false,
            snapshot_note: None,
        }
    }
}

static PANEL: LazyLock<Mutex<Panel>> = LazyLock::new(|| Mutex::new(Panel::default()));
static UI_CTX: Mutex<Option<Context>> = Mutex::new(None);

fn ui_ctx() -> Option<Context> {
    UI_CTX.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn with<R>(f: impl FnOnce(&mut Panel) -> R) -> R {
    f(&mut PANEL.lock().unwrap_or_else(|e| e.into_inner()))
}

fn repaint() {
    if let Some(ctx) = ui_ctx() {
        ctx.request_repaint();
    }
}

fn note(ok: bool, text: impl Into<String>) {
    let text = text.into();
    with(|p| p.note = Some((ok, text)));
    repaint();
}

/// The job of the import running or last run from the window.
pub fn current_job() -> Option<Arc<Job>> {
    with(|p| p.job.clone())
}

fn new_job() -> Arc<Job> {
    let job = Job::new();
    if let Some(ctx) = ui_ctx() {
        job.set_repaint(Some(Arc::new(move || ctx.request_repaint())));
    }
    job
}

fn some(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

fn request_from(p: &Panel) -> ImportRequest {
    let target = match p.mode {
        ModeSel::Merge => Target::Live,
        ModeSel::NewStore => Target::NewStore(PathBuf::from(p.new_store.trim())),
    };
    let mut req = ImportRequest::new(p.old_store.trim(), target);
    req.work_dir = some(&p.work_dir).map(PathBuf::from);
    req.surreal_bin = some(&p.surreal_bin).map(PathBuf::from);
    req.old_ns = some(&p.old_ns);
    req.old_db = some(&p.old_db);
    req.drop_b64 = p.drop_b64;
    req.keep_export = p.keep_export;
    req.settings_fields = p.settings_sel.iter().cloned().collect();
    req
}

fn load_default_map_once() {
    if with(|p| std::mem::replace(&mut p.map_probed, true)) {
        return;
    }
    let path = crate::database::legacy::pathmap::default_config_path(&Target::Live.store_path());
    if let Ok(c) = Config::load(&path) {
        with(|p| {
            if p.cfg.rules.is_empty() {
                p.cfg = c;
                p.map_file = Some(path);
            }
        });
    }
}

fn finish_busy() {
    with(|p| p.busy = None);
    repaint();
}

fn start_engine_check() {
    let job = new_job();
    let opts = with(|p| {
        p.job = Some(Arc::clone(&job));
        p.busy = Some("engine");
        p.error = None;
        EngineOptions { explicit: some(&p.surreal_bin).map(PathBuf::from), allow_download: true, cache_dir: None }
    });
    tokio::spawn(async move {
        match engine::ensure_engine(&opts, &job).await {
            Ok(e) => note(true, format!("SurrealDB {} CLI ready ({})", e.version, e.source.label())),
            Err(e) => note(false, format!("Engine unavailable: {e:#}")),
        }
        finish_busy();
    });
}

fn start_analyze() {
    let job = new_job();
    let req = with(|p| {
        let mut req = request_from(p);
        req.dry_run = true;
        p.job = Some(Arc::clone(&job));
        p.busy = Some("analyze");
        p.error = None;
        p.report = None;
        p.dry = None;
        p.note = None;
        p.prepared = None;
        req
    });
    tokio::spawn(async move {
        let res = async {
            legacy::validate(&req)?;
            let prep = legacy::prepare(&req, &job).await?;
            let store = req.target.store_path();
            let choice = {
                let (explicit, p2, s2) = (None::<PathBuf>, prep.clone(), store.clone());
                let current = with(|p| p.cfg.clone());
                if current.rules.is_empty() {
                    tokio::task::spawn_blocking(move || legacy::resolve_map(explicit.as_deref(), &p2, &s2)).await.map_err(legacy::LegacyError::from)??
                } else {
                    legacy::MapChoice { config: current, file: crate::database::legacy::pathmap::default_config_path(&store), generated: false, suggestion: None }
                }
            };
            let dry = legacy::dry_run(&prep, &choice.config, &req, &job).await?;
            Ok::<_, legacy::LegacyError>((prep, choice, dry))
        }
        .await;
        match res {
            Ok((prep, choice, dry)) => with(|p| {
                let (w, u) = choice.suggestion.as_ref().map(|s| (s.warnings.clone(), s.unresolved.clone())).unwrap_or_default();
                p.settings_sel.retain(|f| prep.analysis.settings_fields.contains(f));
                p.cfg = choice.config;
                p.map_file = Some(choice.file);
                p.map_generated = choice.generated;
                p.map_warnings = w;
                p.map_unresolved = u;
                p.prepared = Some(Arc::new(prep));
                p.dry = Some(Arc::new(dry));
            }),
            Err(e) => with(|p| p.error = Some(e.message())),
        }
        finish_busy();
    });
}

fn start_dry_run() {
    let Some((prep, cfg, req)) = with(|p| {
        let prep = p.prepared.clone()?;
        let mut req = request_from(p);
        req.dry_run = true;
        Some((prep, p.cfg.clone(), req))
    }) else {
        return;
    };
    let job = new_job();
    with(|p| {
        p.job = Some(Arc::clone(&job));
        p.busy = Some("dry");
        p.error = None;
    });
    tokio::spawn(async move {
        match legacy::dry_run(&prep, &cfg, &req, &job).await {
            Ok(d) => with(|p| p.dry = Some(Arc::new(d))),
            Err(e) => with(|p| p.error = Some(e.message())),
        }
        finish_busy();
    });
}

fn start_import() {
    let Some((prep, cfg, req)) = with(|p| {
        let prep = p.prepared.clone()?;
        let mut req = request_from(p);
        let work = prep.work.clone();
        req.report = Some(work.join("legacy-import-report.json"));
        Some((prep, p.cfg.clone(), req))
    }) else {
        return;
    };
    let job = new_job();
    let report_path = req.report.clone();
    with(|p| {
        p.job = Some(Arc::clone(&job));
        p.busy = Some("import");
        p.error = None;
        p.report = None;
        p.report_path = report_path;
        p.confirm_import = false;
    });
    tokio::spawn(async move {
        let keep = req.keep_export;
        let outcome = legacy::run_prepared(req, (*prep).clone(), cfg, job).await;
        let err = outcome.result.as_ref().err().map(|e| e.message());
        with(|p| {
            p.report = Some(Arc::new(outcome.report));
            p.error = err;
            if !keep {
                p.prepared = None;
                p.dry = None;
            }
        });
        finish_busy();
    });
}

fn take_snapshot() {
    with(|p| p.busy = Some("snapshot"));
    tokio::spawn(async move {
        let res = crate::database::snapshot::snapshot_now().await;
        with(|p| {
            p.snapshot_note = Some(match res {
                Ok(info) => (true, format!("Snapshot created: {} ({}). Restoring it needs an app restart (Database maintenance).", info.path.display(), human(info.size_bytes))),
                Err(e) => (false, format!("Snapshot failed: {}", e.user_message())),
            })
        });
        finish_busy();
    });
}

fn discard_analysis() {
    if let Some(prep) = with(|p| p.prepared.take()) {
        legacy::discard_prepared(&prep);
    }
    with(|p| {
        p.dry = None;
        p.note = Some((true, "Analysis discarded and the export file removed".into()));
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Legacy import").open(&mut open).default_width(820.0).default_height(700.0).max_width(1100.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, body);
    });
    set_open(ctx, open);
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h {}m", s / 3600, s % 3600 / 60) } else if s >= 60 { format!("{}m {}s", s / 60, s % 60) } else { format!("{s}s") }
}

fn phase_text(v: &PhaseView) -> String {
    let bytes = !matches!(v.phase, crate::database::legacy::job::PhaseId::Verify | crate::database::legacy::job::PhaseId::Thumbs);
    let n = |x: u64| if bytes { human(x) } else { x.to_string() };
    match v.state {
        PhaseState::Pending => "waiting".into(),
        PhaseState::Skipped => format!("skipped: {}", v.what),
        PhaseState::Failed => format!("failed: {}", v.what),
        PhaseState::Done => format!("done in {}", fmt_dur(v.elapsed())),
        PhaseState::Running => {
            let of = if v.total > 0 { format!(" of {}", n(v.total)) } else { String::new() };
            let eta = v.eta().map(|d| format!(", about {} left", fmt_dur(d))).unwrap_or_default();
            format!("{}: {}{of}{eta}", v.what, n(v.done))
        }
    }
}

fn kv(ui: &mut Ui, key: &str, value: impl Into<egui::WidgetText>) {
    ui.label(RichText::new(key).strong());
    ui.label(value);
    ui.end_row();
}

fn engine_section(ui: &mut Ui, job: Option<&Arc<Job>>, busy: Option<&'static str>) {
    let ok = crate::theme::success(ui);
    let err = crate::theme::error(ui);
    let status = job.map(|j| j.engine_status()).unwrap_or_default();
    match status {
        EngineStatus::Found { path, version, source, checksum_ok } => {
            ui.label(RichText::new(format!("SurrealDB {version} CLI {} at {path}", source.label())).color(ok));
            ui.label(if checksum_ok { "SHA-256 matches the pinned release digest." } else { "Not checked against the pinned digest (user supplied binary)." });
        }
        EngineStatus::Downloading { done, total } => {
            let f = if total > 0 { done as f32 / total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(f).text(format!("downloading {} of {}", human(done), if total > 0 { human(total) } else { "?".into() })));
        }
        EngineStatus::Verifying => {
            ui.label("Verifying the download (SHA-256) and unpacking it.");
        }
        EngineStatus::Checking => {
            ui.label("Looking for the engine.");
        }
        EngineStatus::Failed(e) => {
            ui.label(RichText::new(e).color(err));
        }
        EngineStatus::Unknown => {
            let cached = engine::cached_binary();
            if cached.is_file() {
                ui.label(format!("A cached SurrealDB {} CLI exists at {}; it is verified before use.", engine::PINNED_VERSION, cached.display()));
            } else {
                ui.label(format!("The pinned SurrealDB {} CLI (about 25 MB) is downloaded from the official GitHub release on first use, checked against its SHA-256 and cached in {}.", engine::PINNED_VERSION, engine::cache_dir().display()));
            }
        }
    }
    ui.horizontal(|ui| {
        if ui.add_enabled(busy.is_none(), egui::Button::new("Check engine now")).on_hover_text("Finds or downloads the CLI without touching any store").clicked() {
            start_engine_check();
        }
    });
}

fn phases_section(ui: &mut Ui, job: &Arc<Job>) {
    Grid::new("legacy_phases").num_columns(3).spacing([10.0, 4.0]).show(ui, |ui| {
        for p in PHASES {
            let v = job.view(p);
            ui.label(RichText::new(p.label()).strong());
            let frac = match v.state {
                PhaseState::Done => Some(1.0),
                PhaseState::Pending | PhaseState::Skipped => Some(0.0),
                _ => v.fraction(),
            };
            let bar = match frac {
                Some(f) => egui::ProgressBar::new(f).desired_width(260.0),
                None => egui::ProgressBar::new(0.0).animate(true).desired_width(260.0),
            };
            ui.add(bar);
            ui.label(phase_text(&v));
            ui.end_row();
        }
    });
}

fn policy_combo(ui: &mut Ui, id: &str, label: &str, value: &mut Policy, hint: &str) {
    ui.label(label);
    ComboBox::from_id_salt(id).selected_text(value.label()).show_ui(ui, |ui| {
        for p in [Policy::Keep, Policy::Drop, Policy::Tag] {
            ui.selectable_value(value, p, p.label());
        }
    });
    ui.weak(hint);
    ui.end_row();
}

fn map_section(ui: &mut Ui, busy: bool, has_prep: bool) {
    let warn = crate::theme::warn(ui);
    let (mut cfg, file, generated, warnings, unresolved) = with(|p| (p.cfg.clone(), p.map_file.clone(), p.map_generated, p.map_warnings.clone(), p.map_unresolved.clone()));
    let before = cfg.clone();
    ui.label("Rules match path components case-insensitively; the longest matching prefix wins. An empty target leaves matching rows unmapped.");
    if let Some(f) = &file {
        ui.label(format!("Map file: {}{}", f.display(), if generated { " (suggested from this machine's volumes, saved for review)" } else { "" }));
    }
    for w in &warnings {
        ui.label(RichText::new(w).color(warn));
    }
    for u in &unresolved {
        ui.label(RichText::new(format!("No location found for {u}")).color(warn));
    }
    Grid::new("legacy_policies").num_columns(3).spacing([10.0, 4.0]).show(ui, |ui| {
        policy_combo(ui, "legacy_unmapped", "No rule matches", &mut cfg.options.unmapped, "keep the Windows path, drop the row, or keep it with a tag");
        policy_combo(ui, "legacy_missing", "Mapped file is missing", &mut cfg.options.missing, "keep, drop, or tag the row");
    });
    ui.checkbox(&mut cfg.options.case_fix, "Repair letter case differences against the real folder names");
    let mut remove = None;
    Grid::new("legacy_rules").num_columns(4).spacing([8.0, 4.0]).striped(true).show(ui, |ui| {
        ui.label(RichText::new("From (Windows)").strong());
        ui.label(RichText::new("To (this machine)").strong());
        ui.label(RichText::new("Note").strong());
        ui.label("");
        ui.end_row();
        for (i, r) in cfg.rules.iter_mut().enumerate() {
            ui.add_sized([170.0, 22.0], egui::TextEdit::singleline(&mut r.from));
            ui.add_sized([340.0, 22.0], egui::TextEdit::singleline(&mut r.to));
            ui.add_sized([230.0, 22.0], egui::TextEdit::singleline(&mut r.note));
            if icons::icon_button(ui, svg::X, "Remove rule").clicked() {
                remove = Some(i);
            }
            ui.end_row();
        }
    });
    if let Some(i) = remove {
        cfg.rules.remove(i);
    }
    if cfg.rules.iter().any(|r| crate::database::legacy::suggest::is_automount(std::path::Path::new(&r.to))) {
        ui.label(RichText::new("A target under /run/media or /media is an automount name that can change. Mount the volume at a fixed place (fstab) or use a symlink and put that path here.").color(warn));
    }
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::PLUS, "Add rule").clicked() {
            cfg.rules.push(Rule::default());
        }
        if ui.add_enabled(has_prep && !busy, egui::Button::new("Suggest from this machine")).on_hover_text("Replaces the rules with a suggestion built from mounted volumes and size-matched sample files").clicked() {
            if let Some(prep) = with(|p| p.prepared.clone()) {
                with(|p| p.busy = Some("suggest"));
                tokio::spawn(async move {
                    let s = tokio::task::spawn_blocking(move || legacy::suggest_map(&prep)).await;
                    with(|p| {
                        if let Ok(s) = s {
                            p.cfg = s.config;
                            p.map_warnings = s.warnings;
                            p.map_unresolved = s.unresolved;
                            p.map_generated = true;
                        }
                    });
                    finish_busy();
                });
            }
        }
        if ui.button("Load file...").clicked() {
            let ctx = ui.ctx().clone();
            crate::ui::dialogs::pick_file(&ctx, "Choose a path map (.toml)", |path| match Config::load(&path) {
                Ok(c) => with(|p| {
                    p.cfg = c;
                    p.map_file = Some(path);
                    p.map_generated = false;
                    p.note = Some((true, "Path map loaded".into()));
                }),
                Err(e) => note(false, format!("{e:#}")),
            });
        }
        let can_save = file.is_some() || with(|p| p.mode == ModeSel::Merge);
        if ui.add_enabled(can_save, egui::Button::new("Save")).on_hover_text("Writes legacy-pathmap.toml next to the database folder").clicked() {
            let target = file.clone().unwrap_or_else(|| crate::database::legacy::pathmap::default_config_path(&Target::Live.store_path()));
            match cfg.validate().and_then(|_| cfg.to_toml()).and_then(|t| std::fs::write(&target, t).map_err(Into::into)) {
                Ok(()) => {
                    note(true, format!("Saved {}", target.display()));
                    with(|p| p.map_file = Some(target));
                }
                Err(e) => note(false, format!("Not saved: {e:#}")),
            }
        }
    });
    if cfg != before {
        with(|p| p.cfg = cfg);
    }
}

fn dry_table(ui: &mut Ui, d: &ConvertReport) {
    let err = crate::theme::error(ui);
    ui.label(format!("{} thumbnail rows, {:.1}% map to an existing file ({} case-repaired paths).", d.thumbnail_rows, d.exists_rate() * 100.0, d.path_rules.iter().map(|r| r.case_fixed_rows).sum::<u64>()));
    Grid::new("legacy_dry").num_columns(5).spacing([10.0, 4.0]).striped(true).show(ui, |ui| {
        for h in ["Rule", "Rows", "Exist", "Case-fixed", "Missing prefixes"] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for r in &d.path_rules {
            let rule = if r.rule.is_none() { "(no rule matched)".to_string() } else if r.to.is_empty() { format!("{} (unmapped)", r.from) } else { format!("{} -> {}", r.from, r.to) };
            ui.label(rule);
            ui.label(r.rows.to_string());
            ui.label(format!("{:.1}%", r.exists_rate * 100.0));
            ui.label(r.case_fixed_rows.to_string());
            let text = r.missing_top_prefixes.iter().take(3).map(|(p, n)| format!("{p} ({n})")).collect::<Vec<_>>().join("; ");
            let l = ui.add(egui::Label::new(RichText::new(text).monospace()).truncate());
            if !r.missing_examples.is_empty() {
                l.on_hover_text(r.missing_examples.join("\n"));
            }
            ui.end_row();
        }
    });
    for (t, c) in &d.tables {
        if c.rows_in > 0 && t != "thumbnails" {
            ui.weak(format!("{t}: {} rows read, {} would be imported", c.rows_in, c.rows_out));
        }
    }
    if !d.unmapped_prefixes.is_empty() {
        ui.label(RichText::new(format!("Unmapped prefixes: {}", d.unmapped_prefixes.iter().map(|(k, v)| format!("{k} ({v})")).collect::<Vec<_>>().join("; "))).color(err));
    }
}

fn report_section(ui: &mut Ui, r: &LegacyReport, report_path: Option<&PathBuf>) {
    let ok = crate::theme::success(ui);
    let err = crate::theme::error(ui);
    let warn = crate::theme::warn(ui);
    ui.label(RichText::new(if r.ok { "Import finished and verified" } else { "Import did not complete cleanly" }).strong().color(if r.ok { ok } else { err }));
    if let Some(e) = &r.error {
        ui.label(RichText::new(e).color(err));
    }
    Grid::new("legacy_report").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        kv(ui, "Total time", fmt_dur(Duration::from_millis(r.elapsed_ms as u64)));
        if let Some(rss) = r.peak_rss_mb {
            kv(ui, "Peak memory of this process", format!("{rss} MB"));
        }
        if let Some(s) = &r.snapshot {
            kv(ui, "Snapshot before the import", s.clone());
        }
        if let Some(v) = &r.verify {
            for (t, c) in &v.counts {
                if c.before > 0 || c.after > 0 || c.expected_added > 0 {
                    kv(ui, t, format!("{} before + {} added = {} ({})", c.before, c.expected_added, c.after, if c.ok { "ok" } else { "MISMATCH" }));
                }
            }
            kv(ui, "Typed read-back", format!("{} thumbnails, {} embeddings, {} groups", v.typecheck.thumbnails, v.typecheck.clip_embeddings, v.typecheck.logical_groups));
            kv(ui, "Similarity queries", format!("{} ok", v.knn.iter().filter(|k| k.top1_is_self).count()));
            kv(ui, "Second run adds nothing", v.idempotent.as_ref().map_or("not checked".into(), |i| if i.ok { "yes".to_string() } else { "no".to_string() }));
        }
        if r.thumb_migration_state_reset {
            kv(ui, "Thumbnail migration state", "reset because imported rows carry inline thumbnails");
        }
    });
    for p in r.verify.iter().flat_map(|v| v.problems.iter()) {
        ui.label(RichText::new(p).color(err));
    }
    for w in &r.warnings {
        ui.label(RichText::new(w).color(warn));
    }
    for n in &r.next_steps {
        ui.label(format!("Next: {n}"));
    }
    ui.horizontal(|ui| {
        if let Some(path) = report_path {
            if ui.button("Open report").clicked() {
                let _ = open::that_detached(path);
            }
        }
        if icons::button(ui, svg::DATABASE, "Run thumbnail migration").on_hover_text("Opens the Thumbnail storage window; run the copy phase, then finish").clicked() {
            super::thumb_storage::set_open(ui.ctx(), true);
        }
    });
}

fn body(ui: &mut Ui) {
    *UI_CTX.lock().unwrap_or_else(|e| e.into_inner()) = Some(ui.ctx().clone());
    load_default_map_once();
    let err = crate::theme::error(ui);
    let ok = crate::theme::success(ui);
    let warn = crate::theme::warn(ui);
    let connected = crate::database::is_connected();
    let (busy, job, prepared, dry, report, report_path, error, note_text, snapshot_note) = with(|p| (p.busy, p.job.clone(), p.prepared.clone(), p.dry.clone(), p.report.clone(), p.report_path.clone(), p.error.clone(), p.note.clone(), p.snapshot_note.clone()));
    let idle = busy.is_none();

    ui.label("Imports a SurrealDB 2.x library, such as a Windows-side database, into this app. A pinned official 2.x CLI exports a copy of the old store; the original is never opened or modified.");
    ui.weak("Order matters: finish the import first, then migrate thumbnails (blobs are keyed by the stored path).");
    if let Some((good, text)) = &note_text {
        ui.label(RichText::new(text).color(if *good { ok } else { err }));
    }
    if let Some(e) = &error {
        ui.label(RichText::new(e).color(err));
    }

    CollapsingHeader::new("1. Old store and engine").default_open(true).show(ui, |ui| {
        let mut old = with(|p| p.old_store.clone());
        ui.horizontal(|ui| {
            ui.label("Old store folder");
            if ui.add(egui::TextEdit::singleline(&mut old).desired_width(430.0).hint_text("folder containing clog/ and manifest/")).changed() {
                with(|p| p.old_store = old.clone());
            }
            if icons::button(ui, svg::FOLDER_OPEN, "Browse...").clicked() {
                let ctx = ui.ctx().clone();
                crate::ui::dialogs::pick_folder(&ctx, "Choose the old database folder", |path| with(|p| p.old_store = path.display().to_string()));
            }
        });
        if !old.trim().is_empty() {
            let cached = with(|p| (p.layout.0 == old).then_some(p.layout.1).flatten());
            let kind = cached.unwrap_or_else(|| {
                let k = engine::detect_kind(std::path::Path::new(old.trim()));
                with(|p| p.layout = (old.clone(), Some(k)));
                k
            });
            ui.weak(format!("Detected layout: {}", kind.label()));
        }
        ui.add_space(4.0);
        engine_section(ui, job.as_ref(), busy);
        let mut bin = with(|p| p.surreal_bin.clone());
        ui.horizontal(|ui| {
            ui.label("Use this SurrealDB 2.x binary instead (optional)");
            if ui.add(egui::TextEdit::singleline(&mut bin).desired_width(300.0)).changed() {
                with(|p| p.surreal_bin = bin.clone());
            }
        });
    });

    CollapsingHeader::new("2. Where the rows go").default_open(true).show(ui, |ui| {
        let mut mode = with(|p| p.mode);
        ui.radio_value(&mut mode, ModeSel::Merge, "Merge into the open database (existing rows win)");
        ui.radio_value(&mut mode, ModeSel::NewStore, "Create a new store folder (fast route; contains only the imported rows)");
        if mode == ModeSel::NewStore {
            let mut dir = with(|p| p.new_store.clone());
            ui.horizontal(|ui| {
                if ui.add(egui::TextEdit::singleline(&mut dir).desired_width(430.0).hint_text("empty folder for the new store")).changed() {
                    with(|p| p.new_store = dir.clone());
                }
                if icons::button(ui, svg::FOLDER_OPEN, "Browse...").clicked() {
                    let ctx = ui.ctx().clone();
                    crate::ui::dialogs::pick_folder(&ctx, "Choose an empty folder for the new store", |path| with(|p| p.new_store = path.display().to_string()));
                }
            });
        } else if !connected {
            ui.label(RichText::new("The database is not connected; merging needs the open database.").color(warn));
        }
        if mode != with(|p| p.mode) {
            with(|p| p.mode = mode);
        }
        let mut work = with(|p| p.work_dir.clone());
        ui.horizontal(|ui| {
            ui.label("Work folder (copy, export, scratch)");
            if ui.add(egui::TextEdit::singleline(&mut work).desired_width(360.0).hint_text("default: legacy-import-work next to the database")).changed() {
                with(|p| p.work_dir = work.clone());
            }
        });
        ui.horizontal(|ui| {
            let (mut ns, mut db) = with(|p| (p.old_ns.clone(), p.old_db.clone()));
            ui.label("Old namespace / database");
            let a = ui.add(egui::TextEdit::singleline(&mut ns).desired_width(130.0).hint_text("file_explorer")).changed();
            let b = ui.add(egui::TextEdit::singleline(&mut db).desired_width(130.0).hint_text("ai_search")).changed();
            if a || b {
                with(|p| {
                    p.old_ns = ns.clone();
                    p.old_db = db.clone();
                });
            }
            ui.weak("left empty, the defaults are tried and the store is searched for a thumbnails table");
        });
    });

    ui.horizontal(|ui| {
        let can = idle && !with(|p| p.old_store.trim().is_empty());
        if icons::button(ui, svg::PLAY, "Analyze old store").on_hover_text("Copies the old store, exports it with the 2.x CLI and shows how paths would map").clicked() && can {
            start_analyze();
        }
        if let Some(j) = &job {
            if busy.is_some() && ui.button("Cancel").clicked() {
                j.cancel();
            }
        }
        if busy.is_some() {
            ui.spinner();
        }
        if prepared.is_some() && idle && ui.button("Discard analysis").on_hover_text("Deletes the export file").clicked() {
            discard_analysis();
        }
    });
    if let Some(j) = &job {
        phases_section(ui, j);
    }

    if let Some(prep) = &prepared {
        CollapsingHeader::new("3. What was found").default_open(true).show(ui, |ui| {
            Grid::new("legacy_found").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                kv(ui, "Namespace / database", format!("{} / {}", prep.ns, prep.db));
                for (t, n) in &prep.analysis.tables {
                    kv(ui, t, n.to_string());
                }
                if let Some(e) = &prep.engine {
                    kv(ui, "Engine", format!("SurrealDB {} ({})", e.version, e.source));
                }
                if let Some(x) = &prep.export_info {
                    kv(ui, "Export", format!("{} in {}", human(x.bytes), fmt_dur(Duration::from_millis(x.millis as u64))));
                }
            });
            if !prep.namespaces.is_empty() {
                ui.weak(format!("Other namespaces seen: {}", prep.namespaces.iter().map(|n| format!("{} [{}]", n.ns, n.dbs.iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join("; ")));
            }
        });
    }

    CollapsingHeader::new("4. Path map").default_open(true).show(ui, |ui| map_section(ui, !idle, prepared.is_some()));

    if prepared.is_some() {
        CollapsingHeader::new("5. Dry run").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.add_enabled(idle, egui::Button::new("Run dry run")).on_hover_text("Maps every stored path with the rules above; writes nothing").clicked() {
                    start_dry_run();
                }
            });
            if let Some(d) = &dry {
                dry_table(ui, d);
            }
        });
        CollapsingHeader::new("6. Options").default_open(true).show(ui, |ui| {
            let (mut drop_b64, mut keep) = with(|p| (p.drop_b64, p.keep_export));
            ui.checkbox(&mut drop_b64, "Leave out the stored thumbnail images (they are regenerated by scanning; smaller store)");
            ui.checkbox(&mut keep, "Keep the export file afterwards (it contains the old settings including any API keys)");
            with(|p| {
                p.drop_b64 = drop_b64;
                p.keep_export = keep;
            });
            ui.separator();
            ui.label(RichText::new("Settings to merge into the current settings").strong());
            ui.weak("Nothing is merged by default. Existing settings and API keys of this machine stay untouched unless a field is ticked.");
            let fields = prepared.as_ref().map(|p| p.analysis.settings_fields.clone()).unwrap_or_default();
            if fields.is_empty() {
                ui.weak("The old store has no settings record.");
            }
            let mut sel = with(|p| p.settings_sel.clone());
            ui.horizontal_wrapped(|ui| {
                for f in &fields {
                    let mut on = sel.contains(f);
                    let label = if is_secret_field(f) { RichText::new(format!("{f} (secret)")).color(warn) } else { RichText::new(f.clone()) };
                    if ui.checkbox(&mut on, label).changed() {
                        if on { sel.insert(f.clone()); } else { sel.remove(f); }
                    }
                }
            });
            with(|p| p.settings_sel = sel);
        });
        CollapsingHeader::new("7. Import").default_open(true).show(ui, |ui| {
            let mode = with(|p| p.mode);
            let target_ready = match mode {
                ModeSel::Merge => connected,
                ModeSel::NewStore => !with(|p| p.new_store.trim().is_empty()),
            };
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(idle && connected, egui::Button::new("Take snapshot first")).on_hover_text("Copies the current store so there is a rollback; restoring needs an app restart").clicked() {
                    take_snapshot();
                }
                if icons::button(ui, svg::DOWNLOAD_SIMPLE, "Import...").clicked() && idle && target_ready {
                    with(|p| p.confirm_import = true);
                }
            });
            if let Some((good, text)) = &snapshot_note {
                ui.label(RichText::new(text).color(if *good { ok } else { err }));
            }
            if with(|p| p.confirm_import) {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    let what = match mode {
                        ModeSel::Merge => "into the open database. Existing rows win; a snapshot of the current store is taken first and restoring it needs an app restart.",
                        ModeSel::NewStore => "into a new store folder. The open database is not touched.",
                    };
                    ui.label(RichText::new(format!("Import the converted rows {what} The import can use about 2.6 GB of memory for a 2 GB library.")).color(warn));
                    ui.horizontal(|ui| {
                        if ui.button("Start import").clicked() {
                            start_import();
                        }
                        if ui.button("Cancel").clicked() {
                            with(|p| p.confirm_import = false);
                        }
                    });
                });
            }
        });
    }

    if let Some(r) = &report {
        CollapsingHeader::new("Result").default_open(true).show(ui, |ui| report_section(ui, r, report_path.as_ref()));
    }

    if let Some(j) = &job {
        CollapsingHeader::new("Log").default_open(false).show(ui, |ui| {
            ScrollArea::vertical().id_salt("legacy_log").max_height(160.0).auto_shrink([false, true]).stick_to_bottom(true).show(ui, |ui| {
                for l in j.log_lines() {
                    ui.label(RichText::new(l).monospace());
                }
            });
        });
    }
    if busy.is_some() {
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }
}
