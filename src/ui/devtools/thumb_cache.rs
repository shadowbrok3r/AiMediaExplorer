//! Devtools window showing the thumbnail service live: cache budget and hit rate, queues, per-stage timings, frame times and controls.

use std::sync::Arc;

use eframe::egui::{Align2, Area, Context, DragValue, Frame, Grid, Id, Order, ProgressBar, RichText, ScrollArea, Sense, Shape, Stroke, Ui, Window, pos2, vec2};
use humansize::{BINARY, format_size};

use crate::icons::{self, svg};
use crate::ui::file_table::thumbs::{self, ThumbService, service::Snapshot, stats::StageStat};

fn open_id() -> Id {
    Id::new("ame_devtools_thumb_cache_open")
}

fn overlay_id() -> Id {
    Id::new("ame_devtools_thumb_cache_overlay")
}

fn force_n_id() -> Id {
    Id::new("ame_devtools_thumb_cache_force_n")
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

fn overlay_on(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(overlay_id())).unwrap_or(false)
}

/// Draws the window and the optional overlay when enabled.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if open {
        Window::new("Devtools: Thumbnail cache")
            .open(&mut open)
            .default_width(520.0)
            .default_height(640.0)
            .show(ctx, |ui| {
                ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| body(ui, &thumbs::service()));
            });
        set_open(ctx, open);
    }
    if overlay_on(ctx) {
        overlay(ctx, &thumbs::service());
    }
}

fn mb(bytes: usize) -> String {
    format_size(bytes as u64, BINARY)
}

fn ms(v: f32) -> String {
    format!("{v:.2}")
}

fn stage_row(ui: &mut Ui, name: &str, s: &StageStat) {
    ui.label(name);
    ui.monospace(s.samples.to_string());
    ui.monospace(ms(s.last));
    ui.monospace(ms(s.avg));
    ui.monospace(ms(s.p95));
    ui.monospace(ms(s.max_window));
    ui.end_row();
}

/// Window contents for `svc`.
pub fn body(ui: &mut Ui, svc: &Arc<ThumbService>) {
    let snap = svc.snapshot();
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    cache_section(ui, svc, &snap);
    ui.separator();
    queue_section(ui, &snap);
    ui.separator();
    store_section(ui, svc, &snap);
    ui.separator();
    timing_section(ui, &snap);
    ui.separator();
    control_section(ui, svc, &snap);
}

fn cache_section(ui: &mut Ui, svc: &Arc<ThumbService>, snap: &Snapshot) {
    icons::heading(ui, svg::MEMORY, "Texture cache");
    let fraction = if snap.budget == 0 { 0.0 } else { (snap.used as f32 / snap.budget as f32).min(1.0) };
    ui.add(ProgressBar::new(fraction).text(format!("{} of {}", mb(snap.used), mb(snap.budget))));
    ui.horizontal(|ui| {
        ui.label("Budget (MB)");
        let mut budget_mb = (snap.budget / (1024 * 1024)).max(1);
        if ui.add(DragValue::new(&mut budget_mb).range(8..=8192).speed(8)).changed() {
            svc.set_budget_bytes(budget_mb * 1024 * 1024);
        }
    });
    Grid::new("thumb_cache_numbers").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        let c = &snap.cache;
        ui.label("Entries");
        ui.monospace(snap.entries.to_string());
        ui.label("On screen");
        ui.monospace(snap.on_screen.to_string());
        ui.end_row();
        ui.label("Hit rate / reuse rate");
        ui.monospace(format!("{:.1}% / {:.1}%", c.hit_rate() * 100.0, c.reuse_rate() * 100.0));
        ui.label("Hits / fetches started");
        ui.monospace(format!("{} / {}", c.hits, c.claims_granted));
        ui.end_row();
        ui.label("Evictions");
        ui.monospace(c.evictions.to_string());
        ui.label("Second chances");
        ui.monospace(c.second_chances.to_string());
        ui.end_row();
        ui.label("Re-fetched after eviction");
        ui.monospace(c.thrash.to_string());
        ui.label("In-place updates");
        ui.monospace(c.in_place_updates.to_string());
        ui.end_row();
        ui.label("Claims granted");
        ui.monospace(c.claims_granted.to_string());
        ui.label("Claims over budget");
        ui.monospace(c.claims_over_budget.to_string());
        ui.end_row();
        ui.label("Requests");
        ui.monospace(snap.core.requests.to_string());
        ui.label("Negative hits");
        ui.monospace(snap.core.negative_hits.to_string());
        ui.end_row();
    });
}

fn queue_section(ui: &mut Ui, snap: &Snapshot) {
    icons::heading(ui, svg::STACK, "Queues");
    let d = &snap.depths;
    Grid::new("thumb_cache_queues").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Waiting for DB batch");
        ui.monospace(d.queued.to_string());
        ui.label("DB batches in flight");
        ui.monospace(format!("{} ({} keys total)", snap.fetch_inflight, snap.core.fetch_keys));
        ui.end_row();
        ui.label("Decode queue");
        ui.monospace(d.decode_ready.to_string());
        ui.label("Decoding / awaiting upload");
        ui.monospace(format!("{} / {}", snap.decode_inflight, snap.staged));
        ui.end_row();
        ui.label("Waiting for generator");
        ui.monospace(d.gen_wait.to_string());
        ui.label("Generating");
        ui.monospace(snap.generate_inflight.to_string());
        ui.end_row();
        ui.label("In-flight claims");
        ui.monospace(format!("{} (cap {} per frame)", snap.claims_in_flight, snap.claim_budget));
        ui.label("Negative entries");
        ui.monospace(d.negative.to_string());
        ui.end_row();
        ui.label("Cancelled (stale / scope)");
        ui.monospace(format!("{} / {}", snap.core.cancelled_stale, snap.core.cancelled_scope));
        ui.label("Decode ok / failed / dropped");
        ui.monospace(format!("{} / {} / {}", snap.core.decode_ok, snap.core.decode_failed, snap.core.decode_dropped));
        ui.end_row();
    });
}

fn store_section(ui: &mut Ui, svc: &Arc<ThumbService>, snap: &Snapshot) {
    icons::heading(ui, svg::DATABASE, "Blob store");
    Grid::new("thumb_cache_store").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Encoded cache");
        ui.monospace(format!("{} in {} entries ({} unwritten, {} waiting for a sweep)", mb(snap.recent_bytes), snap.recent_entries, snap.recent_pinned, snap.recent_unqueued));
        ui.label("Known thumbnails");
        ui.monospace(snap.presence.to_string());
        ui.end_row();
        ui.label("Writes queued");
        ui.monospace(snap.pending_writes.to_string());
        ui.label("Written / errors / deferred");
        ui.monospace(format!("{} / {} / {}", snap.counters.writes_ok, snap.counters.write_errors, snap.counters.write_deferred));
        ui.end_row();
        ui.label("Produced");
        ui.monospace(snap.counters.produced.to_string());
        ui.label("Reused (fresh / stale)");
        ui.monospace(format!("{} / {}", snap.counters.fresh_hits, snap.counters.fresh_misses));
        ui.end_row();
        ui.label("Legacy converted / skipped / bad");
        ui.monospace(format!("{} / {} / {}", snap.counters.legacy_ingested, snap.counters.legacy_skipped, snap.counters.legacy_failed));
        ui.label("Stored blobs");
        match snap.store_stats {
            Some(s) => ui.monospace(format!("{} ({})", s.count, mb(s.total_bytes as usize))),
            None => ui.weak("not read"),
        };
        ui.end_row();
    });
    if icons::button(ui, svg::ARROWS_CLOCKWISE, "Read stored blob count").clicked() {
        svc.request_store_stats();
    }
}

fn timing_section(ui: &mut Ui, snap: &Snapshot) {
    icons::heading(ui, svg::GAUGE, "Timings (ms)");
    let t = &snap.timings;
    Grid::new("thumb_cache_timings").num_columns(7).spacing([14.0, 3.0]).striped(true).show(ui, |ui| {
        for h in ["Stage", "n", "last", "avg", "p95", "max"] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        stage_row(ui, "DB batch", &t.db_batch);
        stage_row(ui, "Decode", &t.decode);
        stage_row(ui, "Upload", &t.upload);
        stage_row(ui, "Store write", &t.store_put);
        stage_row(ui, "Frame interval", &t.frame);
        stage_row(ui, "Explorer UI (CPU)", &t.ui);
    });
    ui.label(format!("Explorer UI p95 {} ms, max {} ms; frame interval p50 {} ms", ms(t.ui.p95), ms(t.ui.max_window), ms(t.frame_p50)));
    sparkline(ui, &t.ui_series);
}

fn sparkline(ui: &mut Ui, series: &[f32]) {
    let width = ui.available_width().max(120.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, 56.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 2.0, ui.visuals().widgets.noninteractive.bg_stroke, egui::StrokeKind::Inside);
    if series.len() < 2 {
        return;
    }
    let top = series.iter().copied().fold(1.0f32, f32::max) * 1.25;
    let y_of = |v: f32| rect.bottom() - (v / top).clamp(0.0, 1.0) * rect.height();
    if top > 16.7 {
        let guide = y_of(16.7);
        painter.line_segment([pos2(rect.left(), guide), pos2(rect.right(), guide)], Stroke::new(1.0, ui.visuals().weak_text_color()));
    }
    let step = rect.width() / (series.len() - 1) as f32;
    let points: Vec<_> = series.iter().enumerate().map(|(i, v)| pos2(rect.left() + step * i as f32, y_of(*v))).collect();
    painter.add(Shape::line(points, Stroke::new(1.5, crate::theme::info(ui))));
}

fn control_section(ui: &mut Ui, svc: &Arc<ThumbService>, snap: &Snapshot) {
    icons::heading(ui, svg::SLIDERS, "Controls");
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::BROOM, "Flush cache").on_hover_text("Drop every texture, the encoded cache and negative entries").clicked() {
            svc.flush();
        }
        let mut n = ui.ctx().data(|d| d.get_temp::<usize>(force_n_id())).unwrap_or(32);
        if icons::button(ui, svg::LIGHTNING, "Force decode").on_hover_text("Drop the visible textures so they are fetched and decoded again").clicked() {
            svc.force_decode(n);
        }
        if ui.add(DragValue::new(&mut n).range(1..=2000).suffix(" visible")).changed() {
            ui.ctx().data_mut(|d| d.insert_temp(force_n_id(), n));
        }
        if ui.button("Clear negative cache").clicked() {
            svc.clear_negative();
        }
    });
    let gate = svc.gate();
    let mut slow = gate.delay_ms() > 0;
    let mut delay = gate.delay_ms().max(250);
    ui.horizontal(|ui| {
        if ui.checkbox(&mut slow, "Simulate slow DB").changed() {
            gate.set_delay_ms(if slow { delay } else { 0 });
        }
        if ui.add_enabled(slow, DragValue::new(&mut delay).range(10..=5000).suffix(" ms")).changed() {
            gate.set_delay_ms(delay);
        }
        let mut failing = snap.store_failing;
        if ui.checkbox(&mut failing, "Fail DB calls").changed() {
            gate.set_failing(failing);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Claims per frame");
        let mut claims = snap.claim_budget;
        if ui.add(DragValue::new(&mut claims).range(1..=256)).changed() {
            svc.set_claim_budget(claims);
        }
        ui.label("Uploads per frame");
        let mut uploads = snap.upload_per_frame;
        if ui.add(DragValue::new(&mut uploads).range(1..=64)).changed() {
            svc.set_upload_per_frame(uploads);
        }
    });
    let mut on = overlay_on(ui.ctx());
    if ui.checkbox(&mut on, "Frame overlay").changed() {
        ui.ctx().data_mut(|d| d.insert_temp(overlay_id(), on));
    }
}

fn overlay(ctx: &Context, svc: &Arc<ThumbService>) {
    let snap = svc.snapshot();
    let t = &snap.timings;
    Area::new(Id::new("ame_thumb_overlay_area")).order(Order::Foreground).anchor(Align2::RIGHT_BOTTOM, vec2(-12.0, -12.0)).interactable(false).show(ctx, |ui| {
        Frame::popup(ui.style()).show(ui, |ui| {
            let fps = if t.frame.avg > 0.0 { 1000.0 / t.frame.avg } else { 0.0 };
            ui.monospace(format!("{fps:.0} fps  ui avg {} p95 {} ms", ms(t.ui.avg), ms(t.ui.p95)));
            ui.monospace(format!("cache {} / {}  hit {:.0}%", mb(snap.used), mb(snap.budget), snap.cache.hit_rate() * 100.0));
            ui.monospace(format!("queue {}  decode {}  uploads {}", snap.depths.queued + snap.depths.decode_ready, snap.decode_inflight, snap.staged));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ctx: &Context) {
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 900.0))), ..Default::default() };
        ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |_| {});
            show(ui.ctx());
        })
        .drop_without_applying_deltas();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_window_and_overlay_draw_with_live_data() {
        let ctx = Context::default();
        let svc = thumbs::service();
        svc.poll_now(&ctx);
        frame(&ctx);
        assert!(!is_open(&ctx));
        toggle(&ctx);
        ctx.data_mut(|d| d.insert_temp(overlay_id(), true));
        frame(&ctx);
        frame(&ctx);
        assert!(is_open(&ctx));
        toggle(&ctx);
        assert!(!is_open(&ctx));
    }

    #[test]
    fn byte_and_ms_formatting() {
        assert_eq!(mb(1024 * 1024), "1 MiB");
        assert_eq!(ms(1.234), "1.23");
    }
}
