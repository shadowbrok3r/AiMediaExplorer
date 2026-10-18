//! Devtools window for thumbnail blob storage: stats, migration controls, rebuild list, sample viewer and integrity check.

use std::future::Future;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, Ui, Window};

use crate::database::snapshot::human;
use crate::database::thumb_blobs::{self, BlobStats, QuickStats, ThumbBlob};
use crate::database::thumb_migration::{self, CopyOptions, FailedRow, FinishOptions, FinishReport, MigrationState, Migrator, MissingReport, Phase};
use crate::icons::{self, svg};

const STATE_POLL: Duration = Duration::from_millis(800);
const QUICK_POLL: Duration = Duration::from_secs(20);
const QUICK_POLL_RUNNING: Duration = Duration::from_secs(3);
const LIST_SHOWN: usize = 200;
const INTEGRITY_WINDOW: usize = 50;

fn open_id() -> Id {
    Id::new("ame_devtools_thumb_storage_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

struct SampleView {
    blob: ThumbBlob,
    texture: Option<egui::TextureHandle>,
    error: Option<String>,
}

#[derive(Clone, Default)]
struct IntegrityReport {
    requested: usize,
    checked: usize,
    ok: usize,
    failures: Vec<FailedRow>,
    millis: u128,
}

struct Panel {
    was_running: bool,
    ctx: Option<Context>,
    quick: Option<QuickStats>,
    exact: Option<BlobStats>,
    thumbnail_rows: Option<u64>,
    legacy_rows: Option<u64>,
    store_size: Option<u64>,
    last_quick: Option<Instant>,
    quick_busy: bool,
    state: MigrationState,
    state_at: Option<Instant>,
    note: Option<(bool, String)>,
    busy: Vec<&'static str>,
    finish_report: Option<FinishReport>,
    confirm_finish: bool,
    confirm_reset: bool,
    clear_undecodable: bool,
    finish_sample: usize,
    missing: Option<Arc<MissingReport>>,
    sample: Option<SampleView>,
    lookup_path: String,
    integrity: Option<IntegrityReport>,
    integrity_running: Option<(usize, usize)>,
    integrity_n: usize,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            was_running: false,
            ctx: None,
            quick: None,
            exact: None,
            thumbnail_rows: None,
            legacy_rows: None,
            store_size: None,
            last_quick: None,
            quick_busy: false,
            state: MigrationState::default(),
            state_at: None,
            note: None,
            busy: Vec::new(),
            finish_report: None,
            confirm_finish: false,
            confirm_reset: false,
            clear_undecodable: false,
            finish_sample: 200,
            missing: None,
            sample: None,
            lookup_path: String::new(),
            integrity: None,
            integrity_running: None,
            integrity_n: 100,
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
            p.state_at = None;
        });
        repaint();
    });
}

fn busy(p: &Panel, label: &str) -> bool {
    p.busy.iter().any(|l| *l == label)
}

fn refresh_quick() {
    let started = with(|p| !std::mem::replace(&mut p.quick_busy, true));
    if !started {
        return;
    }
    tokio::spawn(async move {
        let quick = thumb_blobs::quick_stats().await;
        let rows = crate::database::surql::count_rows(&crate::database::DB, crate::database::THUMBNAILS).await;
        let store = std::path::PathBuf::from(crate::database::get_db_path());
        let size = tokio::task::spawn_blocking(move || crate::database::snapshot::dir_size(&store)).await.ok();
        with(|p| {
            p.quick = quick.ok();
            p.thumbnail_rows = rows.ok();
            p.store_size = size;
            p.last_quick = Some(Instant::now());
            p.quick_busy = false;
        });
        repaint();
    });
}

fn pick_random_sample() {
    run("sample", async {
        let result = async {
            let count = thumb_blobs::quick_stats().await?.stats.count;
            anyhow::ensure!(count > 0, "no blobs are stored yet");
            let start = thumb_migration::random_below(count);
            let mut blobs = thumb_blobs::sample_with(&crate::database::DB, 1, start).await?;
            blobs.pop().ok_or_else(|| anyhow::anyhow!("no blob at offset {start}"))
        }
        .await;
        show_sample(result).await;
    });
}

fn lookup_sample(path: String) {
    run("sample", async move {
        let key = thumb_blobs::key_for_path(&path);
        let result = thumb_blobs::get_many(std::slice::from_ref(&key))
            .await
            .and_then(|mut m| m.remove(&key).ok_or_else(|| anyhow::anyhow!("no blob stored for key {key}")));
        show_sample(result).await;
    });
}

async fn show_sample(result: anyhow::Result<ThumbBlob>) {
    match result {
        Ok(blob) => {
            let ctx = with(|p| p.ctx.clone());
            let bytes = blob.bytes.clone();
            let decoded = tokio::task::spawn_blocking(move || {
                image::load_from_memory(&bytes).map(|img| {
                    let rgba = img.to_rgba8();
                    egui::ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw())
                })
            })
            .await;
            let (texture, error) = match (decoded, ctx) {
                (Ok(Ok(image)), Some(ctx)) => (Some(ctx.load_texture("thumb_storage_sample", image, egui::TextureOptions::LINEAR)), None),
                (Ok(Err(e)), _) => (None, Some(format!("decode failed: {e}"))),
                (Err(e), _) => (None, Some(format!("decode task failed: {e}"))),
                _ => (None, Some("no UI context".to_string())),
            };
            with(|p| p.sample = Some(SampleView { blob, texture, error }));
        }
        Err(e) => {
            with(|p| p.sample = None);
            note(false, format!("Sample failed: {e:#}"));
        }
    }
}

fn run_integrity(n: usize) {
    run("integrity", async move {
        let started = Instant::now();
        let mut report = IntegrityReport { requested: n, ..Default::default() };
        with(|p| p.integrity_running = Some((0, n)));
        let count = match thumb_blobs::quick_stats().await {
            Ok(q) => q.stats.count,
            Err(e) => {
                note(false, format!("Integrity check failed: {e:#}"));
                with(|p| p.integrity_running = None);
                return;
            }
        };
        while report.checked < n && count > 0 {
            let take = INTEGRITY_WINDOW.min(n - report.checked);
            let start = thumb_migration::random_below(count.saturating_sub(take as u64).max(1));
            let blobs = match thumb_blobs::sample_with(&crate::database::DB, take, start).await {
                Ok(b) => b,
                Err(e) => {
                    note(false, format!("Integrity check failed: {e:#}"));
                    break;
                }
            };
            if blobs.is_empty() {
                break;
            }
            let results = tokio::task::spawn_blocking(move || blobs.into_iter().map(|b| (b.path.clone(), thumb_blobs::verify_blob(&b))).collect::<Vec<_>>())
                .await
                .unwrap_or_default();
            for (path, res) in results {
                report.checked += 1;
                match res {
                    Ok(_) => report.ok += 1,
                    Err(e) => report.failures.push(FailedRow { path, reason: format!("{e:#}") }),
                }
            }
            let (done, total) = (report.checked, n);
            with(|p| p.integrity_running = Some((done, total)));
            repaint();
        }
        report.millis = started.elapsed().as_millis();
        with(|p| {
            p.integrity_running = None;
            p.integrity = Some(report);
        });
    });
}

fn count_inline_rows() {
    run("count", async {
        match Migrator::global().count_legacy().await {
            Ok(n) => with(|p| p.legacy_rows = Some(n)),
            Err(e) => note(false, format!("Counting inline thumbnails failed: {e:#}")),
        }
    });
}

fn exact_stats() {
    run("stats", async {
        match thumb_blobs::stats().await {
            Ok(s) => with(|p| p.exact = Some(s)),
            Err(e) => note(false, format!("Reading blob statistics failed: {e:#}")),
        }
    });
}

fn find_missing() {
    run("missing", async {
        match Migrator::global().missing_paths(100_000).await {
            Ok(r) => with(|p| p.missing = Some(Arc::new(r))),
            Err(e) => note(false, format!("Scanning for rows without a blob failed: {e:#}")),
        }
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Thumbnail storage")
        .open(&mut open)
        .default_width(640.0)
        .default_height(620.0)
        .show(ctx, |ui| {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, body);
        });
    set_open(ctx, open);
}

fn maybe_refresh(ctx: &Context) {
    let running = Migrator::global().is_running();
    let (poll_state, refresh_stats) = with(|p| {
        p.ctx = Some(ctx.clone());
        let ended = std::mem::replace(&mut p.was_running, running) && !running;
        let every = if running { QUICK_POLL_RUNNING } else { QUICK_POLL };
        (p.state_at.is_none_or(|t| t.elapsed() > STATE_POLL) || ended, (ended || p.last_quick.is_none_or(|t| t.elapsed() > every)) && !p.quick_busy)
    });
    if poll_state {
        let state = Migrator::global().state();
        with(|p| {
            p.state = state;
            p.state_at = Some(Instant::now());
        });
    }
    if refresh_stats && crate::database::is_connected() {
        refresh_quick();
    }
}

fn kv(ui: &mut Ui, key: &str, value: impl Into<egui::WidgetText>) {
    ui.label(RichText::new(key).strong());
    ui.label(value);
    ui.end_row();
}

fn failure_list(ui: &mut Ui, id: &str, rows: &[FailedRow]) {
    ScrollArea::vertical().id_salt(id).max_height(120.0).auto_shrink([false, true]).show(ui, |ui| {
        for r in rows.iter().take(LIST_SHOWN) {
            ui.label(RichText::new(format!("{}  ({})", r.path, r.reason)).monospace());
        }
    });
}

fn body(ui: &mut Ui) {
    maybe_refresh(ui.ctx());
    let connected = crate::database::is_connected();
    let migrator = Migrator::global();
    let progress = migrator.progress();
    let running = migrator.is_running();
    let (quick, exact, rows, legacy, store_size, state, note_text, mut confirm_finish, mut confirm_reset, mut clear_undecodable, mut finish_sample) = with(|p| {
        (p.quick, p.exact, p.thumbnail_rows, p.legacy_rows, p.store_size, p.state.clone(), p.note.clone(), p.confirm_finish, p.confirm_reset, p.clear_undecodable, p.finish_sample)
    });
    let err = crate::theme::error(ui);
    let ok_color = crate::theme::success(ui);
    let warn = crate::theme::warn(ui);

    if !connected {
        ui.label(RichText::new("The database is not connected in this process; actions are disabled.").color(warn));
    }
    if let Some((ok, text)) = &note_text {
        ui.label(RichText::new(text).color(if *ok { ok_color } else { err }));
    }

    CollapsingHeader::new("Storage").default_open(true).show(ui, |ui| {
        Grid::new("thumb_storage_stats").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            kv(ui, "Thumbnail rows", rows.map_or("unknown".to_string(), |n| n.to_string()));
            match (exact, quick) {
                (Some(s), _) => {
                    kv(ui, "Blobs (exact)", format!("{} blobs, {} total, {} average", s.count, human(s.total_bytes), human(s.avg_bytes())));
                    if let Some(sz) = store_size.filter(|s| *s > 0) {
                        kv(ui, "Share of store folder", format!("{:.1}% of {}", s.total_bytes as f64 * 100.0 / sz as f64, human(sz)));
                    }
                }
                (None, Some(q)) => {
                    let approx = if q.estimated { "about " } else { "" };
                    kv(ui, "Blobs", format!("{} blobs, {approx}{} total, {approx}{} average", q.stats.count, human(q.stats.total_bytes), human(q.stats.avg_bytes())));
                    if let Some(sz) = store_size.filter(|s| *s > 0) {
                        kv(ui, "Share of store folder", format!("{approx}{:.1}% of {}", q.stats.total_bytes as f64 * 100.0 / sz as f64, human(sz)));
                    }
                }
                (None, None) => kv(ui, "Blobs", "unknown"),
            }
            kv(ui, "Rows with inline thumbnail", legacy.map_or("not counted yet".to_string(), |n| n.to_string()));
            kv(ui, "List queries select inline data", if thumb_migration::inline_may_exist() { "yes (until migration finishes)" } else { "no" });
        });
        ui.horizontal_wrapped(|ui| {
            if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh").on_hover_text("Re-read the blob count and a sampled size estimate").clicked() {
                refresh_quick();
            }
            let stats_busy = with(|p| busy(p, "stats"));
            if ui.add_enabled(connected && !stats_busy, egui::Button::new("Exact statistics")).on_hover_text("Reads every blob record; slow on large stores").clicked() {
                exact_stats();
            }
            let count_busy = with(|p| busy(p, "count"));
            if ui.add_enabled(connected && !count_busy, egui::Button::new("Count inline rows")).on_hover_text("Reads every thumbnail row; slow before migration").clicked() {
                count_inline_rows();
            }
        });
    });

    CollapsingHeader::new("Migration to blobs").default_open(true).show(ui, |ui| {
        ui.label("The copy phase writes blobs only and never touches the original rows. Finish verifies coverage and decodes a sample before clearing the inline data. Rollback is the snapshot named below (restore it from Database maintenance).");
        ui.weak("Very large libraries: close the app and run smart-media --db-thumbs migrate, then --db-thumbs finish. SurrealKV keeps what it reads in memory until the app restarts.");
        ui.add_space(4.0);
        Grid::new("thumb_storage_migration").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            kv(ui, "Phase", if progress.message.is_empty() { progress.phase.label().to_string() } else { format!("{}: {}", progress.phase.label(), progress.message) });
            kv(ui, "This run", format!("{} processed, {} migrated, {} already present, {} failed, {} cleared", progress.processed, progress.migrated, progress.skipped_existing, progress.failed, progress.cleared));
            if progress.bytes_in > 0 {
                kv(ui, "Bytes", format!("{} read as base64, {} written as blobs", human(progress.bytes_in), human(progress.bytes_out)));
            }
            if running || progress.processed > 0 {
                let eta = progress.eta().map_or(String::new(), |d| format!(", about {} s left", d.as_secs()));
                kv(ui, "Rate", format!("{:.0} rows/s{eta}", progress.rows_per_sec()));
            }
            kv(ui, "Saved state", format!("{} migrated, {} skipped, {} failed, {} cleared", state.migrated, state.skipped_existing, state.failed, state.cleared));
            kv(ui, "Thumbnail rows at start", state.rows_total.map_or("not counted".to_string(), |n| n.to_string()));
            kv(ui, "Copy complete", if state.copy_done { "yes" } else { "no" });
            kv(ui, "Finished", state.finished.clone().unwrap_or_else(|| "no".to_string()));
            kv(ui, "Last position", state.last_key.clone().unwrap_or_else(|| "-".to_string()));
            kv(ui, "Snapshot", state.snapshot.clone().unwrap_or_else(|| "none yet (taken before the first write)".to_string()));
        });
        if let Some(f) = progress.fraction().filter(|_| running) {
            ui.add(egui::ProgressBar::new(f).show_percentage());
        }
        if state.is_finished() {
            ui.label(RichText::new("Inline data is cleared, but the store folder keeps its size until the store is rewritten: in Database maintenance enable thumbnail blobs for exports, Export .surql now, Import .surql into NEW store, then select that folder and restart.").color(warn));
        }
        if let Some(e) = &progress.error {
            ui.label(RichText::new(e).color(err));
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let can_start = connected && !running && !state.is_finished();
            let start_label = if state.copy_done { "Run copy again" } else if state.started.is_some() { "Resume copy" } else { "Start copy" };
            let start = ui.add_enabled_ui(can_start, |ui| icons::button(ui, svg::PLAY, start_label).on_hover_text("Takes a snapshot first unless one from the last hour exists")).inner;
            if start.clicked() {
                migrator.spawn_copy(CopyOptions::default());
            }
            if running && !migrator.is_paused() {
                if icons::button(ui, svg::PAUSE, "Pause").clicked() {
                    migrator.pause();
                }
            } else if running && icons::button(ui, svg::PLAY, "Resume").clicked() {
                migrator.resume();
            }
            if ui.add_enabled(running, egui::Button::new("Cancel")).on_hover_text("Stops after the current batch; progress is kept").clicked() {
                migrator.cancel();
            }
            if ui.add_enabled(connected && !running && !state.is_finished(), egui::Button::new("Restart from beginning")).on_hover_text("Re-scans every row; blobs that already exist are skipped").clicked() {
                migrator.spawn_copy(CopyOptions { restart: true, ..Default::default() });
            }
            if ui.add_enabled(connected && !running && state.copy_done && !state.is_finished(), egui::Button::new("Finish migration...")).clicked() {
                confirm_finish = true;
            }
            if ui.add_enabled(!running, egui::Button::new("Forget migration state")).on_hover_text("Deletes thumb-migration.json; blobs and rows are untouched").clicked() {
                confirm_reset = true;
            }
        });
        if confirm_finish {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new("Finish verifies that every row with inline data has a blob whose key matches, decodes a random sample, and only then clears the inline base64 from the rows. This cannot be undone except by restoring the snapshot.").color(warn));
                ui.checkbox(&mut clear_undecodable, "Also clear inline data that cannot be decoded (those rows need a rebuild)");
                ui.horizontal(|ui| {
                    ui.label("Blobs to decode as a sample");
                    ui.add(egui::DragValue::new(&mut finish_sample).range(0..=4096));
                });
                ui.horizontal(|ui| {
                    if ui.button("Verify and clear").clicked() {
                        confirm_finish = false;
                        let m = Migrator::global();
                        with(|p| p.finish_report = None);
                        let opts = FinishOptions { clear_undecodable, sample: finish_sample, ..Default::default() };
                        run("finish", async move {
                            match m.run_finish(opts).await {
                                Ok(r) => with(|p| p.finish_report = Some(r)),
                                Err(e) => note(false, format!("Finish failed: {e:#}")),
                            }
                        });
                    }
                    if ui.button("Cancel").clicked() {
                        confirm_finish = false;
                    }
                });
            });
        }
        if confirm_reset {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new("Forget the saved position and counters? The next copy re-scans every row (existing blobs are skipped).").color(warn));
                ui.horizontal(|ui| {
                    if ui.button("Forget").clicked() {
                        confirm_reset = false;
                        thumb_migration::reset_state(migrator.store());
                        with(|p| p.state_at = None);
                    }
                    if ui.button("Cancel").clicked() {
                        confirm_reset = false;
                    }
                });
            });
        }
        if let Some(r) = with(|p| p.finish_report.clone()) {
            ui.add_space(4.0);
            ui.label(RichText::new(&r.summary).strong().color(if r.ok { ok_color } else { warn }));
            ui.label(format!(
                "{} rows checked, {} covered ({} repaired), {} undecodable, {} sampled ({} failed), {} cleared, {} remaining",
                r.legacy_rows, r.covered, r.repaired, r.undecodable_count, r.sample_checked, r.sample_failures.len(), r.cleared, r.remaining
            ));
            if !r.undecodable.is_empty() {
                CollapsingHeader::new(format!("Undecodable rows ({})", r.undecodable_count)).show(ui, |ui| failure_list(ui, "thumb_storage_undecodable", &r.undecodable));
            }
            if !r.sample_failures.is_empty() {
                CollapsingHeader::new(format!("Sample failures ({})", r.sample_failures.len())).default_open(true).show(ui, |ui| failure_list(ui, "thumb_storage_sample_fail", &r.sample_failures));
            }
        }
        if !state.failures.is_empty() {
            CollapsingHeader::new(format!("Rows that could not be converted ({})", state.failed)).show(ui, |ui| failure_list(ui, "thumb_storage_failures", &state.failures));
        }
    });

    CollapsingHeader::new("Rows without a blob").default_open(false).show(ui, |ui| {
        ui.label("Thumbnail rows that have no blob; the scan pipeline can regenerate these from the source files.");
        let missing_busy = with(|p| busy(p, "missing"));
        if ui.add_enabled(connected && !missing_busy, egui::Button::new("Find rows without a blob")).clicked() {
            find_missing();
        }
        if missing_busy {
            ui.spinner();
        }
        if let Some(m) = with(|p| p.missing.clone()) {
            ui.label(format!("{} rows scanned, {} without a blob{}", m.scanned, m.missing, if m.truncated { " (list truncated)" } else { "" }));
            if ui.add_enabled(!m.paths.is_empty(), egui::Button::new("Copy paths to clipboard")).clicked() {
                ui.ctx().copy_text(m.paths.join("\n"));
            }
            ScrollArea::vertical().id_salt("thumb_storage_missing").max_height(160.0).auto_shrink([false, true]).show(ui, |ui| {
                for p in m.paths.iter().take(LIST_SHOWN) {
                    ui.label(RichText::new(p).monospace());
                }
                if m.paths.len() > LIST_SHOWN {
                    ui.weak(format!("... and {} more", m.paths.len() - LIST_SHOWN));
                }
            });
        }
    });

    CollapsingHeader::new("Sample viewer").default_open(true).show(ui, |ui| {
        let sample_busy = with(|p| busy(p, "sample"));
        ui.horizontal_wrapped(|ui| {
            if icons::button(ui, svg::IMAGE, "Random blob").clicked() && connected && !sample_busy {
                pick_random_sample();
            }
            let mut path = with(|p| p.lookup_path.clone());
            ui.label("Path");
            if ui.add(egui::TextEdit::singleline(&mut path).desired_width(260.0).hint_text("exact thumbnails.path value")).changed() {
                with(|p| p.lookup_path = path.clone());
            }
            if ui.add_enabled(connected && !sample_busy && !path.is_empty(), egui::Button::new("Look up")).clicked() {
                lookup_sample(path.clone());
            }
            if sample_busy {
                ui.spinner();
            }
        });
        with(|p| {
            if let Some(s) = &p.sample {
                Grid::new("thumb_storage_sample").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    kv(ui, "Path", s.blob.path.clone());
                    kv(ui, "Key", s.blob.key.clone());
                    kv(ui, "Format", format!("{}, {}x{}, {}", s.blob.format, s.blob.w, s.blob.h, human(s.blob.bytes.len() as u64)));
                });
                if let Some(e) = &s.error {
                    ui.label(RichText::new(e).color(err));
                }
                if let Some(t) = &s.texture {
                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::from_handle(t)).max_width(320.0).max_height(320.0));
                }
            }
        });
    });

    CollapsingHeader::new("Integrity check").default_open(true).show(ui, |ui| {
        let running_check = with(|p| p.integrity_running);
        ui.horizontal_wrapped(|ui| {
            let mut n = with(|p| p.integrity_n);
            ui.label("Decode");
            if ui.add(egui::DragValue::new(&mut n).range(1..=5000)).changed() {
                with(|p| p.integrity_n = n);
            }
            ui.label("random blobs");
            if ui.add_enabled(connected && running_check.is_none(), egui::Button::new("Run integrity check")).clicked() {
                run_integrity(n);
            }
            if let Some((done, total)) = running_check {
                ui.spinner();
                ui.label(format!("{done} of {total}"));
            }
        });
        if let Some(r) = with(|p| p.integrity.clone()) {
            let all_ok = r.failures.is_empty() && r.checked > 0;
            ui.label(RichText::new(format!("{} of {} requested blobs checked: {} decoded, {} failed ({} ms)", r.checked, r.requested, r.ok, r.failures.len(), r.millis)).color(if all_ok { ok_color } else { err }));
            if !r.failures.is_empty() {
                failure_list(ui, "thumb_storage_integrity", &r.failures);
            }
        }
    });

    CollapsingHeader::new("Recent events").default_open(false).show(ui, |ui| {
        ScrollArea::vertical().id_salt("thumb_storage_events").max_height(140.0).auto_shrink([false, true]).stick_to_bottom(true).show(ui, |ui| {
            for e in crate::database::snapshot::recent_events().iter().rev().take(40).rev() {
                ui.label(RichText::new(format!("{}  {}", e.at.format("%H:%M:%S"), e.msg)).monospace());
            }
        });
    });

    with(|p| {
        p.confirm_finish = confirm_finish;
        p.confirm_reset = confirm_reset;
        p.clear_undecodable = clear_undecodable;
        p.finish_sample = finish_sample;
    });
    let active = running || !with(|p| p.busy.is_empty()) || progress.phase == Phase::Paused;
    ui.ctx().request_repaint_after(Duration::from_millis(if active { 200 } else { 1000 }));
}
