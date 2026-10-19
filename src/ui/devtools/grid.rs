//! Devtools window for the thumbnail grid: tile metrics, the texture cache, the decode pipeline, frame times and a stress scroller.

use std::ops::Range;
use std::sync::Arc;

use eframe::egui::{Context, DragValue, Grid, Id, ProgressBar, RichText, ScrollArea, Ui, Window};
use humansize::{BINARY, format_size};

use super::row_store::{frame_summary, sparkline};
use crate::icons::{self, svg};
use crate::ui::file_table::table::grid::GridStats;
use crate::ui::file_table::table::view_mode::{TileFit, ViewMode};
use crate::ui::file_table::thumbs::{self, ThumbService, service::Snapshot as ServiceSnapshot};

/// Rows per second the stress scroller starts with.
pub const DEFAULT_STRESS_ROWS_PER_S: f32 = 40.0;

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub owner: Id,
    pub label: String,
    pub mode: ViewMode,
    pub total: usize,
    pub selected: usize,
    pub cols: usize,
    pub tile_target: f32,
    pub tile_px: f32,
    pub row_h: f32,
    pub caption_h: f32,
    pub fit: TileFit,
    pub offset: f32,
    pub viewport_h: f32,
    pub content_h: f32,
    pub visible: Range<usize>,
    pub ahead: (Range<usize>, Range<usize>),
    pub stats: GridStats,
    pub frame_ms: Vec<f32>,
    pub stress: Option<f32>,
    pub facts_queued: usize,
    pub facts_inflight: usize,
    pub facts_known: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    SetStress(Option<f32>),
    SetMode(ViewMode),
    SetTile(f32),
    ScrollTop,
    ScrollEnd,
    ResetFrameStats,
    ResetFacts,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_grid_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_grid_snapshot")
}

fn commands_id(owner: Id) -> Id {
    Id::unique(("ame_devtools_grid_commands", owner))
}

fn stress_rate_id() -> Id {
    Id::unique("ame_devtools_grid_stress_rate")
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

/// Makes `snapshot` the one the window shows.
pub fn publish(ctx: &Context, snapshot: Snapshot) {
    ctx.data_mut(|d| d.insert_temp(snapshot_id(), snapshot));
}

/// The snapshot of the explorer that drew last.
pub fn latest(ctx: &Context) -> Option<Snapshot> {
    ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id()))
}

/// Queues `command` for the explorer identified by `owner`.
pub fn push_command(ctx: &Context, owner: Id, command: Command) {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Vec<Command>>(commands_id(owner)).push(command));
}

/// Takes the commands queued for `owner`.
pub fn take_commands(ctx: &Context, owner: Id) -> Vec<Command> {
    ctx.data_mut(|d| d.remove_temp::<Vec<Command>>(commands_id(owner))).unwrap_or_default()
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Grid").open(&mut open).default_width(500.0).default_height(660.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let svc = thumbs::service();
            match latest(ctx) {
                Some(snapshot) => body(ui, &snapshot, &svc),
                None => {
                    ui.weak("No grid has drawn since the window opened.");
                    cache_section(ui, &svc);
                }
            }
        });
    });
    set_open(ctx, open);
}

fn mb(bytes: usize) -> String {
    format_size(bytes as u64, BINARY)
}

fn range_text(r: &Range<usize>) -> String {
    if r.is_empty() { "none".to_string() } else { format!("{}..{} ({})", r.start, r.end, r.len()) }
}

fn body(ui: &mut Ui, s: &Snapshot, svc: &Arc<ThumbService>) {
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    ui.label(RichText::new(&s.label).weak());
    tile_section(ui, s);
    ui.separator();
    cache_section(ui, svc);
    ui.separator();
    pipeline_section(ui, svc);
    ui.separator();
    frame_section(ui, s);
    ui.separator();
    control_section(ui, s);
}

fn tile_section(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::AME_THUMBNAIL_GRID, "Tiles");
    Grid::new("grid_tiles").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("View mode");
        ui.monospace(s.mode.label());
        ui.label("Rows in view / selected");
        ui.monospace(format!("{} / {}", s.total, s.selected));
        ui.end_row();
        ui.label("Columns");
        ui.monospace(s.cols.to_string());
        ui.label("Tile target / actual");
        ui.monospace(format!("{:.0} / {:.1} px", s.tile_target, s.tile_px));
        ui.end_row();
        ui.label("Row height / caption");
        ui.monospace(format!("{:.1} / {:.1} px", s.row_h, s.caption_h));
        ui.label("Fit");
        ui.monospace(s.fit.label());
        ui.end_row();
        ui.label("Scroll offset / viewport");
        ui.monospace(format!("{:.0} / {:.0} px", s.offset, s.viewport_h));
        ui.label("Content height");
        ui.monospace(format!("{:.0} px", s.content_h));
        ui.end_row();
        ui.label("Visible positions");
        ui.monospace(range_text(&s.visible));
        ui.label("Tiles drawn");
        ui.monospace(s.stats.tiles_drawn.to_string());
        ui.end_row();
        ui.label("Requested ahead (above)");
        ui.monospace(range_text(&s.ahead.0));
        ui.label("Requested ahead (below)");
        ui.monospace(range_text(&s.ahead.1));
        ui.end_row();
        ui.label("Thumbnail requests / ahead");
        ui.monospace(format!("{} / {}", s.stats.thumbs_requested, s.stats.ahead_requested));
        ui.label("Tiles waiting for an image");
        ui.monospace(s.stats.pending.to_string());
        ui.end_row();
        ui.label("Facts queued / running / known");
        ui.monospace(format!("{} / {} / {}", s.facts_queued, s.facts_inflight, s.facts_known));
        ui.label("Frames drawn");
        ui.monospace(s.stats.frames.to_string());
        ui.end_row();
    });
}

fn cache_section(ui: &mut Ui, svc: &Arc<ThumbService>) {
    let snap = svc.snapshot();
    icons::heading(ui, svg::MEMORY, "Texture cache");
    let fraction = if snap.budget == 0 { 0.0 } else { (snap.used as f32 / snap.budget as f32).min(1.0) };
    ui.add(ProgressBar::new(fraction).text(format!("{} of {} (approximate GPU bytes)", mb(snap.used), mb(snap.budget))));
    ui.horizontal(|ui| {
        ui.label("Budget (MB)");
        let mut budget_mb = (snap.budget / (1024 * 1024)).max(1);
        if ui.add(DragValue::new(&mut budget_mb).range(8..=8192).speed(8)).changed() {
            svc.set_budget_bytes(budget_mb * 1024 * 1024);
        }
        if crate::ui::danger_icon_button(ui, svg::TRASH, "Purge cache").on_hover_text("Drops every texture and the encoded thumbnails cached in memory; visible tiles load again").clicked() {
            svc.flush();
        }
    });
    let c = &snap.cache;
    Grid::new("grid_cache").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Entries / on screen");
        ui.monospace(format!("{} / {}", snap.entries, snap.on_screen));
        ui.label("Hits / misses (fetches)");
        ui.monospace(format!("{} / {}", c.hits, c.claims_granted));
        ui.end_row();
        ui.label("Hit rate / reuse rate");
        ui.monospace(format!("{:.1}% / {:.1}%", c.hit_rate() * 100.0, c.reuse_rate() * 100.0));
        ui.label("Evictions / second chances");
        ui.monospace(format!("{} / {}", c.evictions, c.second_chances));
        ui.end_row();
        ui.label("Re-fetched after eviction");
        ui.monospace(c.thrash.to_string());
        ui.label("Requests / ahead requests");
        ui.monospace(format!("{} / {}", snap.core.requests, snap.core.ahead_requests));
        ui.end_row();
    });
}

fn pipeline_section(ui: &mut Ui, svc: &Arc<ThumbService>) {
    let snap = svc.snapshot();
    icons::heading(ui, svg::STACK, "Decode pipeline");
    queue_grid(ui, &snap);
    ui.horizontal(|ui| {
        ui.label("Uploads per frame");
        let mut n = snap.upload_per_frame;
        if ui.add(DragValue::new(&mut n).range(1..=64)).changed() {
            svc.set_upload_per_frame(n);
        }
        ui.label("Fetch claims per frame");
        let mut claims = snap.claim_budget;
        if ui.add(DragValue::new(&mut claims).range(1..=256)).changed() {
            svc.set_claim_budget(claims);
        }
    });
}

fn queue_grid(ui: &mut Ui, snap: &ServiceSnapshot) {
    let d = &snap.depths;
    Grid::new("grid_queue").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Decode queue / decoding");
        ui.monospace(format!("{} / {}", d.decode_ready, snap.decode_inflight));
        ui.label("Awaiting upload");
        ui.monospace(snap.staged.to_string());
        ui.end_row();
        ui.label("Waiting for the store");
        ui.monospace(d.queued.to_string());
        ui.label("Waiting for the generator");
        ui.monospace(d.gen_wait.to_string());
        ui.end_row();
        ui.label("Uploaded (new / in place)");
        ui.monospace(format!("{} / {}", snap.core.uploads_new, snap.core.uploads_in_place));
        ui.label("Last upload (ms)");
        ui.monospace(format!("{:.2}", snap.timings.upload.last));
        ui.end_row();
        ui.label("Cancelled (left the screen / scope)");
        ui.monospace(format!("{} / {}", snap.core.cancelled_stale, snap.core.cancelled_scope));
        ui.label("Decoded ok / failed / dropped");
        ui.monospace(format!("{} / {} / {}", snap.core.decode_ok, snap.core.decode_failed, snap.core.decode_dropped));
        ui.end_row();
    });
}

fn frame_section(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::GAUGE, "Grid frame time");
    match frame_summary(&s.frame_ms) {
        Some((mean, p95, max)) => {
            ui.label(format!("mean {mean:.2} ms, p95 {p95:.2} ms, max {max:.2} ms over {} frames", s.frame_ms.len()));
            sparkline(ui, &s.frame_ms);
        }
        None => {
            ui.weak("no frames recorded");
        }
    }
}

fn control_section(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::SLIDERS, "Controls");
    let ctx = ui.ctx().clone();
    ui.horizontal_wrapped(|ui| {
        for mode in ViewMode::ALL {
            if icons::selectable_button(ui, if mode == ViewMode::Grid { svg::SQUARES_FOUR } else { svg::LIST }, s.mode == mode, mode.label()).clicked() {
                push_command(&ctx, s.owner, Command::SetMode(mode));
            }
        }
        for size in [64.0, 96.0, 160.0, 256.0, 320.0] {
            if ui.selectable_label((s.tile_target - size).abs() < 0.5, format!("{size:.0}")).clicked() {
                push_command(&ctx, s.owner, Command::SetTile(size));
            }
        }
    });
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::ARROW_UP, "Top").clicked() {
            push_command(&ctx, s.owner, Command::ScrollTop);
        }
        if icons::button(ui, svg::ARROW_DOWN, "End").clicked() {
            push_command(&ctx, s.owner, Command::ScrollEnd);
        }
        if icons::button(ui, svg::ARROW_COUNTER_CLOCKWISE, "Reset frame stats").clicked() {
            push_command(&ctx, s.owner, Command::ResetFrameStats);
        }
        if icons::button(ui, svg::ARROWS_CLOCKWISE, "Probe again").on_hover_text("Forget offline, folder count, collection and video duration facts so visible tiles ask again").clicked() {
            push_command(&ctx, s.owner, Command::ResetFacts);
        }
    });
    let mut rate = ctx.data(|d| d.get_temp::<f32>(stress_rate_id())).unwrap_or(DEFAULT_STRESS_ROWS_PER_S);
    ui.horizontal_wrapped(|ui| {
        ui.label("Stress scroller");
        if ui.add(DragValue::new(&mut rate).range(1.0..=2000.0).speed(1.0).suffix(" rows/s")).changed() {
            ctx.data_mut(|d| d.insert_temp(stress_rate_id(), rate));
            if s.stress.is_some() {
                push_command(&ctx, s.owner, Command::SetStress(Some(rate)));
            }
        }
        match s.stress {
            None => {
                if icons::button(ui, svg::PLAY, "Start").clicked() {
                    push_command(&ctx, s.owner, Command::SetStress(Some(rate)));
                }
            }
            Some(_) => {
                if icons::button(ui, svg::STOP, "Stop").clicked() {
                    push_command(&ctx, s.owner, Command::SetStress(None));
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{RawInput, Rect, pos2, vec2};

    #[test]
    fn commands_are_delivered_once_to_their_owner() {
        let ctx = Context::default();
        let (a, b) = (Id::unique("a"), Id::unique("b"));
        push_command(&ctx, a, Command::ScrollTop);
        push_command(&ctx, a, Command::SetStress(Some(10.0)));
        push_command(&ctx, b, Command::ResetFacts);
        assert_eq!(take_commands(&ctx, a), [Command::ScrollTop, Command::SetStress(Some(10.0))]);
        assert!(take_commands(&ctx, a).is_empty());
        assert_eq!(take_commands(&ctx, b), [Command::ResetFacts]);
    }

    #[test]
    fn ranges_read_as_text() {
        assert_eq!(range_text(&(0..0)), "none");
        assert_eq!(range_text(&(12..30)), "12..30 (18)");
    }

    #[test]
    fn the_devtools_flag_and_the_command_name_the_window() {
        let args = ["--devtools".to_string(), "grid".to_string()];
        let cmds = crate::command::cli::startup_commands(&args).unwrap();
        assert!(matches!(cmds.as_slice(), [crate::command::AppCommand::OpenDevtools { window: crate::command::DevtoolsWindow::Grid }]));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_window_toggles_and_shows_without_a_snapshot() {
        let ctx = crate::ui::test_support::test_ctx();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
        let input = RawInput { screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 900.0))), ..Default::default() };
        ctx.run_ui(input, |ui| {
            eframe::egui::CentralPanel::default().show(ui, |_| {});
            show(ui.ctx());
        })
        .drop_without_applying_deltas();
        toggle(&ctx);
        assert!(!is_open(&ctx));
    }
}
