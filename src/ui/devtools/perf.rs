//! Devtools window with frame timing, the texture cache, task counts and the platform meters.

use std::collections::VecDeque;

use eframe::egui::{Context, Grid, Id, RichText, Ui, Window};
use humansize::{BINARY, format_size};

use crate::tasks;
use crate::ui::taskbar::SystemMetrics;

/// Frames kept for the timing statistics.
pub const FRAME_WINDOW: usize = 240;

fn open_id() -> Id {
    Id::new("ame_devtools_perf_open")
}

fn frames_id() -> Id {
    Id::new("ame_devtools_perf_frames")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Recent frame durations in seconds, oldest first.
#[derive(Clone, Debug, Default)]
pub struct FrameTimes {
    frames: VecDeque<f32>,
}

/// Summary of a [`FrameTimes`] window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    pub count: usize,
    pub avg_ms: f32,
    pub p95_ms: f32,
    pub max_ms: f32,
    pub fps: f32,
}

impl FrameTimes {
    pub fn push(&mut self, seconds: f32) {
        if self.frames.len() == FRAME_WINDOW {
            self.frames.pop_front();
        }
        self.frames.push_back(seconds.max(0.0));
    }

    pub fn stats(&self) -> FrameStats {
        if self.frames.is_empty() {
            return FrameStats::default();
        }
        let mut sorted: Vec<f32> = self.frames.iter().copied().collect();
        sorted.sort_by(f32::total_cmp);
        let avg = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let p95 = sorted[((sorted.len() as f32 * 0.95).ceil() as usize).clamp(1, sorted.len()) - 1];
        FrameStats {
            count: sorted.len(),
            avg_ms: avg * 1000.0,
            p95_ms: p95 * 1000.0,
            max_ms: sorted[sorted.len() - 1] * 1000.0,
            fps: if avg > 0.0 { 1.0 / avg } else { 0.0 },
        }
    }
}

/// Records this frame's duration while the window is open.
pub fn record_frame(ctx: &Context) {
    if !is_open(ctx) {
        return;
    }
    let dt = ctx.input(|i| i.unstable_dt);
    ctx.data_mut(|d| d.get_temp_mut_or_insert_with(frames_id(), FrameTimes::default).push(dt));
}

fn meter_row(ui: &mut Ui, label: &str, reading: Option<String>) {
    ui.label(label);
    match reading {
        Some(text) => ui.label(text),
        None => ui.label(RichText::new("not provided on this platform").weak()),
    };
    ui.end_row();
}

fn body(ctx: &Context, ui: &mut Ui) {
    let stats = ctx.data(|d| d.get_temp::<FrameTimes>(frames_id())).unwrap_or_default().stats();
    let counts = tasks::with(|r| r.counts());
    let metrics = SystemMetrics::sample();
    let cache = crate::ui::file_table::thumbs::service().snapshot();
    Grid::new("ame_devtools_perf_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Frames sampled");
        ui.label(stats.count.to_string());
        ui.end_row();
        ui.label("Frame time (avg / p95 / max)");
        ui.label(format!("{:.1} / {:.1} / {:.1} ms", stats.avg_ms, stats.p95_ms, stats.max_ms));
        ui.end_row();
        ui.label("FPS while repainting");
        ui.label(format!("{:.0}", stats.fps));
        ui.end_row();
        ui.label("Texture cache");
        ui.label(format!("{} of {} in {} textures", format_size(cache.used as u64, BINARY), format_size(cache.budget as u64, BINARY), cache.entries));
        ui.end_row();
        ui.label("Thumbnail queue");
        ui.label(format!("{} queued, {} decoding, {} writes pending", cache.depths.queued, cache.decode_inflight, cache.pending_writes));
        ui.end_row();
        ui.label("Tasks");
        ui.label(format!("{} running, {} queued, {} finished, {} failed", counts.running, counts.queued, counts.finished, counts.failed));
        ui.end_row();
        ui.label("Needs attention");
        ui.label(counts.attention.to_string());
        ui.end_row();
        ui.label("Commands waiting");
        ui.label(crate::command::global().len().to_string());
        ui.end_row();
        ui.label("Commands logged");
        ui.label(crate::command::with_log(|l| l.len()).to_string());
        ui.end_row();
        meter_row(ui, "CPU", metrics.cpu01.map(|v| format!("{:.0}%", v * 100.0)));
        meter_row(ui, "RAM", metrics.ram.map(|(u, t)| format!("{u:.0} of {t:.0} MiB")));
        meter_row(ui, "VRAM", metrics.vram.map(|(u, t)| format!("{u:.0} of {t:.0} MiB")));
    });
    super::scan_perf::section(ctx, ui);
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    record_frame(ctx);
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(500));
    Window::new("Devtools: Perf").open(&mut open).default_width(460.0).show(ctx, |ui| body(ctx, ui));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times(ms: &[f32]) -> FrameTimes {
        let mut t = FrameTimes::default();
        ms.iter().for_each(|m| t.push(m / 1000.0));
        t
    }

    #[test]
    fn no_frames_means_zero_stats() {
        assert_eq!(FrameTimes::default().stats(), FrameStats::default());
    }

    #[test]
    fn stats_summarise_the_window() {
        let s = times(&[10.0, 20.0, 30.0, 40.0]).stats();
        assert_eq!(s.count, 4);
        assert!((s.avg_ms - 25.0).abs() < 0.01);
        assert!((s.max_ms - 40.0).abs() < 0.01);
        assert!((s.p95_ms - 40.0).abs() < 0.01);
        assert!((s.fps - 40.0).abs() < 0.01);
    }

    #[test]
    fn the_window_keeps_only_the_latest_frames() {
        let mut t = FrameTimes::default();
        for i in 0..(FRAME_WINDOW + 50) {
            t.push(i as f32 / 1000.0);
        }
        let s = t.stats();
        assert_eq!(s.count, FRAME_WINDOW);
        assert!((s.max_ms - (FRAME_WINDOW + 49) as f32).abs() < 0.01);
    }

    #[test]
    fn frames_are_recorded_only_while_the_window_is_open() {
        let ctx = crate::ui::test_support::test_ctx();
        ctx.run_ui(Default::default(), |ui| record_frame(ui.ctx())).drop_without_applying_deltas();
        assert!(ctx.data(|d| d.get_temp::<FrameTimes>(frames_id())).is_none());
        set_open(&ctx, true);
        ctx.run_ui(Default::default(), |ui| record_frame(ui.ctx())).drop_without_applying_deltas();
        assert_eq!(ctx.data(|d| d.get_temp::<FrameTimes>(frames_id())).unwrap().frames.len(), 1);
    }
}
