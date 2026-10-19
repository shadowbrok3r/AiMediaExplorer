//! The Scale window: live counters and the buttons that start scripted runs.

use std::time::{Duration, Instant};

use eframe::egui::{Context, Grid, Id, ProgressBar, RichText, ScrollArea, Ui, Window, pos2};
use humansize::{BINARY, format_size};

use super::bench::{Report, Suite};
use super::frames::{self, Spread};
use super::{Facts, is_open, last_report, latest, marks, proc, progress, request_cancel, set_open};
use crate::command::AppCommand;
use crate::database::scope_queries::counter::{self, Counts, Site};
use crate::icons::{self, svg};
use crate::ui::devtools::row_store::sparkline;
use crate::ui::file_table::thumbs;

/// Frames the live statistics cover.
const LIVE_FRAMES: usize = 240;
/// Time between the readings the rates are computed from.
const RATE_INTERVAL: Duration = Duration::from_secs(2);
/// Repaint causes listed.
const CAUSES_SHOWN: usize = 10;
/// Time between repaints while the window is open and nothing runs.
const REFRESH: Duration = Duration::from_millis(500);

#[derive(Clone)]
struct Reading {
    at: Instant,
    queries: Counts,
    cpu_secs: Option<f64>,
    frames: u64,
}

impl Reading {
    fn take() -> Self {
        Self { at: Instant::now(), queries: counter::snapshot(), cpu_secs: proc::cpu_seconds(), frames: frames::mark() }
    }
}

/// Rates over the last full interval.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Rates {
    queries_per_s: f32,
    frames_per_s: f32,
    cpu_percent: Option<f32>,
}

fn rates_between(old: &Reading, new: &Reading) -> Rates {
    let secs = new.at.saturating_duration_since(old.at).as_secs_f32();
    if secs <= 0.0 {
        return Rates::default();
    }
    Rates {
        queries_per_s: new.queries.since(&old.queries).total() as f32 / secs,
        frames_per_s: new.frames.saturating_sub(old.frames) as f32 / secs,
        cpu_percent: match (old.cpu_secs, new.cpu_secs) {
            (Some(a), Some(b)) => Some(((b - a) / secs as f64 * 100.0) as f32),
            _ => None,
        },
    }
}

fn readings_id() -> Id {
    Id::new("ame_devtools_scale_readings")
}

// Rolls the stored readings forward and returns the rates between the two newest.
fn rates(ctx: &Context) -> Rates {
    let now = Reading::take();
    let (mut prev, mut last) = ctx.data(|d| d.get_temp::<(Reading, Reading)>(readings_id())).unwrap_or_else(|| (now.clone(), now.clone()));
    if now.at.saturating_duration_since(last.at) >= RATE_INTERVAL {
        prev = last;
        last = now;
    }
    ctx.data_mut(|d| d.insert_temp(readings_id(), (prev.clone(), last.clone())));
    rates_between(&prev, &last)
}

fn bytes(n: u64) -> String {
    format_size(n, BINARY)
}

fn opt_bytes(n: Option<u64>) -> String {
    n.map_or("not provided on this platform".to_string(), bytes)
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    if !progress().is_some_and(|p| p.quiet) {
        ctx.request_repaint_after(REFRESH);
    }
    Window::new("Devtools: Scale").open(&mut open).default_pos(pos2(440.0, 40.0)).default_width(540.0).default_height(720.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| body(ctx, ui));
    });
    set_open(ctx, open);
}

fn body(ctx: &Context, ui: &mut Ui) {
    let rates = rates(ctx);
    process_section(ui, &rates);
    ui.separator();
    rows_section(ui, latest(ctx).as_ref());
    ui.separator();
    frames_section(ui, &rates);
    ui.separator();
    thumbs_section(ui);
    ui.separator();
    database_section(ui, &rates);
    ui.separator();
    causes_section(ui);
    ui.separator();
    bench_section(ui);
}

fn process_section(ui: &mut Ui, rates: &Rates) {
    icons::heading(ui, svg::MEMORY, "Process");
    Grid::new("scale_process").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Resident memory (peak)");
        ui.monospace(format!("{} ({})", opt_bytes(proc::rss_bytes()), opt_bytes(proc::peak_rss_bytes())));
        ui.end_row();
        ui.label("Heap and stack / mapped files");
        ui.monospace(proc::rss_split().map_or("not provided on this platform".to_string(), |(a, f)| format!("{} / {}", bytes(a), bytes(f))));
        ui.end_row();
        ui.label("Allocated by malloc");
        ui.monospace(opt_bytes(proc::heap_in_use()));
        ui.end_row();
        ui.label("CPU over the last interval");
        ui.monospace(rates.cpu_percent.map_or("not provided".to_string(), |c| format!("{c:.0}% of one core")));
        ui.end_row();
        ui.label("Time since the process started");
        ui.monospace(proc::process_age().map_or("-".to_string(), |a| format!("{:.1} s", a.as_secs_f32())));
        ui.end_row();
        for (name, secs) in marks::all() {
            ui.label(format!("Start-up: {name}"));
            ui.monospace(format!("{secs:.2} s"));
            ui.end_row();
        }
    });
}

fn rows_section(ui: &mut Ui, facts: Option<&Facts>) {
    icons::heading(ui, svg::STACK, "Rows of the focused tab");
    let Some(f) = facts else {
        ui.weak("No explorer has drawn since the window opened.");
        return;
    };
    ui.label(RichText::new(format!("{} ({})", f.label, f.scope)).weak());
    Grid::new("scale_rows").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Rows in store");
        ui.monospace(f.store_len.to_string());
        ui.label("View index size");
        ui.monospace(f.view_len.to_string());
        ui.end_row();
        ui.label("Hidden by filters");
        ui.monospace(f.stats.hidden_by_filter.to_string());
        ui.label("Selected");
        ui.monospace(f.selected.to_string());
        ui.end_row();
        ui.label("Store / view revision");
        ui.monospace(format!("{} / {}", f.store_revision, f.view_revision));
        ui.label("View refreshes");
        ui.monospace(format!("{} same, {} appended, {} rebuilt", f.refreshes.unchanged, f.refreshes.appended, f.refreshes.rebuilt));
        ui.end_row();
        ui.label("Mode");
        ui.monospace(if f.details { "details" } else { "grid" });
        ui.label("Loading");
        ui.monospace(f.loading.to_string());
        ui.end_row();
        ui.label("Row memory");
        ui.monospace(format!("{} ({} per row)", bytes(f.row_bytes as u64), bytes((f.row_bytes / f.store_len.max(1)) as u64)));
        ui.end_row();
    });
}

fn frames_section(ui: &mut Ui, rates: &Rates) {
    icons::heading(ui, svg::GAUGE, "Frames");
    let recent = frames::latest(LIVE_FRAMES);
    let dt: Vec<f32> = recent.iter().map(|f| f.dt_ms).collect();
    let ui_ms: Vec<f32> = recent.iter().map(|f| f.ui_ms).collect();
    let cpu: Vec<f32> = recent.iter().map(|f| f.cpu_ms).collect();
    Grid::new("scale_frames").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        for (name, series) in [("Frame time", &dt), ("Interface build", &ui_ms), ("Frame CPU", &cpu)] {
            ui.label(name);
            ui.monospace(spread_text(Spread::of(series)));
            ui.end_row();
        }
        ui.label("Frames per second");
        ui.monospace(format!("{:.1}", rates.frames_per_s));
        ui.end_row();
        ui.label("Allocations per frame");
        if super::alloc::enabled() {
            let allocs: Vec<f32> = recent.iter().map(|f| f.allocs as f32).collect();
            ui.monospace(allocs_text(Spread::of(&allocs)));
        } else {
            ui.weak("build with --features alloc-stats");
        }
        ui.end_row();
    });
    sparkline(ui, &dt);
}

fn allocs_text(spread: Option<Spread>) -> String {
    spread.map_or("no frames".to_string(), |s| format!("min {:.0}  avg {:.0}  p95 {:.0}  max {:.0}", s.min, s.avg, s.p95, s.max))
}

fn spread_text(spread: Option<Spread>) -> String {
    spread.map_or("no frames".to_string(), |s| format!("min {:.1}  avg {:.1}  p95 {:.1}  max {:.1} ms", s.min, s.avg, s.p95, s.max))
}

fn thumbs_section(ui: &mut Ui) {
    icons::heading(ui, svg::AME_THUMBNAIL_GRID, "Thumbnails");
    let snap = thumbs::service().snapshot();
    let fraction = if snap.budget == 0 { 0.0 } else { (snap.used as f32 / snap.budget as f32).min(1.0) };
    ui.add(ProgressBar::new(fraction).text(format!("{} of {} in {} textures", bytes(snap.used as u64), bytes(snap.budget as u64), snap.entries)));
    Grid::new("scale_thumbs").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Cache hits / rate");
        ui.monospace(format!("{} / {:.0}%", snap.cache.hits, snap.cache.hit_rate() * 100.0));
        ui.label("Evictions / thrash");
        ui.monospace(format!("{} / {}", snap.cache.evictions, snap.cache.thrash));
        ui.end_row();
        ui.label("Decode queue");
        ui.monospace(format!("{} queued, {} decoding", snap.depths.queued, snap.decode_inflight));
        ui.label("Fetching / staged");
        ui.monospace(format!("{} / {}", snap.fetch_inflight, snap.staged));
        ui.end_row();
        ui.label("Encoded cache");
        ui.monospace(format!("{} in {} blobs", bytes(snap.recent_bytes as u64), snap.recent_entries));
        ui.label("Known stored thumbnails");
        ui.monospace(snap.presence.to_string());
        ui.end_row();
        ui.label("Writes pending");
        ui.monospace(snap.pending_writes.to_string());
        ui.label("On screen");
        ui.monospace(snap.on_screen.to_string());
        ui.end_row();
    });
}

fn database_section(ui: &mut Ui, rates: &Rates) {
    icons::heading(ui, svg::DATABASE, "Database queries of the browser");
    let counts = counter::snapshot();
    ui.label(format!("{} queries so far, {:.1} per second over the last interval", counts.total(), rates.queries_per_s));
    Grid::new("scale_queries").num_columns(4).spacing([16.0, 3.0]).striped(true).show(ui, |ui| {
        for (i, site) in Site::ALL.iter().enumerate() {
            ui.label(site.label());
            ui.monospace(counts.of(*site).to_string());
            if i % 2 == 1 {
                ui.end_row();
            }
        }
    });
}

fn causes_section(ui: &mut Ui) {
    icons::heading(ui, svg::ARROWS_CLOCKWISE, "Who asked for frames");
    let mut causes: Vec<(String, u64)> = super::repaint_causes().into_iter().collect();
    causes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if causes.is_empty() {
        ui.weak("Counted while this window is open or a run goes.");
        return;
    }
    Grid::new("scale_causes").num_columns(2).spacing([16.0, 3.0]).striped(true).show(ui, |ui| {
        for (cause, frames) in causes.into_iter().take(CAUSES_SHOWN) {
            ui.monospace(frames.to_string());
            ui.label(cause);
            ui.end_row();
        }
    });
}

fn bench_section(ui: &mut Ui) {
    icons::heading(ui, svg::PULSE, "Scripted runs");
    let running = progress();
    ui.horizontal_wrapped(|ui| {
        let idle = running.is_none();
        for (suite, label, tip) in [
            (Suite::Scroll, "Run scroll benchmark", "Scrolls the focused tab from the first row to the last in the grid and in the details view"),
            (Suite::Ui, "Run UI suite", "Resizes, sorts, switches view, toggles filters and selects all in the focused tab"),
            (Suite::Scopes, "Run scope loads", "Opens the library, recent, smart views and the biggest tag, category and collection"),
            (Suite::Record, "Record 10 s", "Measures the application for ten seconds without touching it"),
            (Suite::All, "Run everything", "Every suite in order, ending with 30 idle seconds"),
        ] {
            if ui.add_enabled(idle, icons::button_widget(ui, svg::PLAY, label)).on_hover_text(tip).clicked() {
                let seconds = (suite == Suite::Record).then_some(10.0);
                crate::command::send(AppCommand::ScaleBench { suite, seconds, scopes: Vec::new(), report: None });
            }
        }
        if ui.add_enabled(!idle, icons::button_widget(ui, svg::STOP, "Stop")).clicked() {
            request_cancel();
        }
    });
    if let Some(p) = &running {
        let fraction = if p.steps == 0 { 0.0 } else { (p.step.saturating_sub(1)) as f32 / p.steps as f32 };
        ui.add(ProgressBar::new(fraction).text(format!("step {} of {}: {} ({:.0} s)", p.step, p.steps, p.label, p.elapsed)));
    }
    match last_report() {
        Some(report) => report_table(ui, &report),
        None => {
            ui.weak("No run has finished yet.");
        }
    }
}

fn report_table(ui: &mut Ui, report: &Report) {
    ui.label(format!("{} run, {:.1} s, memory {} to {} (peak {})", report.suite.label(), report.seconds, mb(report.rss_start_mb), mb(report.rss_end_mb), mb(report.peak_rss_mb)));
    Grid::new("scale_report").num_columns(7).spacing([10.0, 3.0]).striped(true).show(ui, |ui| {
        for head in ["Step", "Frames", "Min ms", "Avg ms", "P95 ms", "Max ms", "Queries"] {
            ui.label(RichText::new(head).strong());
        }
        ui.end_row();
        for s in &report.steps {
            ui.label(&s.label);
            ui.monospace(s.frames.to_string());
            match s.spread {
                Some(f) => {
                    for v in [f.dt.min, f.dt.avg, f.dt.p95, f.dt.max] {
                        ui.monospace(format!("{v:.1}"));
                    }
                }
                None => {
                    for _ in 0..4 {
                        ui.monospace("-");
                    }
                }
            }
            ui.monospace(s.queries.to_string());
            ui.end_row();
        }
    });
    if ui.button("Copy report as JSON").clicked() {
        ui.ctx().copy_text(report.to_json());
    }
}

fn mb(v: Option<f32>) -> String {
    v.map_or("-".to_string(), |v| format!("{v:.0} MB"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(at: Instant, queries: u64, cpu: f64, frames: u64) -> Reading {
        let mut counts = Counts::default();
        counts.by_site[Site::ScopePage as usize] = queries;
        Reading { at, queries: counts, cpu_secs: Some(cpu), frames }
    }

    #[test]
    fn rates_are_the_difference_over_the_interval() {
        let t0 = Instant::now();
        let r = rates_between(&reading(t0, 10, 1.0, 100), &reading(t0 + Duration::from_secs(2), 50, 1.5, 220));
        assert!((r.queries_per_s - 20.0).abs() < 1e-3);
        assert!((r.frames_per_s - 60.0).abs() < 1e-3);
        assert!((r.cpu_percent.unwrap() - 25.0).abs() < 1e-3);
    }

    #[test]
    fn no_time_between_readings_gives_no_rates() {
        let t0 = Instant::now();
        assert_eq!(rates_between(&reading(t0, 0, 0.0, 0), &reading(t0, 9, 9.0, 9)), Rates::default());
    }

    #[test]
    fn the_rates_roll_forward_once_an_interval_has_passed() {
        let ctx = crate::ui::test_support::test_ctx();
        let first = rates(&ctx);
        assert_eq!(first.queries_per_s, 0.0);
        let stored = ctx.data(|d| d.get_temp::<(Reading, Reading)>(readings_id())).unwrap();
        assert!(stored.1.at.saturating_duration_since(stored.0.at) < RATE_INTERVAL);
    }

    #[test]
    fn the_window_draws_without_a_tab_and_lists_the_counters() {
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(900.0, 1200.0))), ..Default::default() };
        ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |_| {});
            show(ui.ctx());
        })
        .drop_without_applying_deltas();
        assert!(is_open(&ctx));
    }
}
