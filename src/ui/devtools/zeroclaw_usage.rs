//! Devtools window for the gateway usage: totals, tables per session and per day, and a reset.

use eframe::egui::{Context, Grid, Id, RichText, ScrollArea, Ui, Window};

use crate::zc::events::now_ms;
use crate::zc::usage::{self, UsageData, UsageStore, dollars, grouped};

/// Rows of the session table shown.
const SESSION_ROWS: usize = 60;
/// Rows of the day table shown.
const DAY_ROWS: usize = 60;

fn open_id() -> Id {
    Id::new("ame_devtools_zeroclaw_usage_open")
}

fn confirm_id() -> Id {
    Id::new("ame_devtools_zeroclaw_usage_confirm")
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

/// Local date and time of a Unix-millisecond stamp.
fn stamp(at_ms: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64).map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default()
}

/// A session id cut to a width the table keeps.
fn short(id: &str) -> String {
    if id.chars().count() <= 20 { id.to_string() } else { format!("{}...", id.chars().take(17).collect::<String>()) }
}

fn totals_row(ui: &mut Ui, data: &UsageData) {
    let t = &data.total;
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("{} turns", grouped(t.turns))).strong());
        ui.label(format!("{} tokens in", grouped(t.input_tokens)));
        ui.label(format!("{} tokens out", grouped(t.output_tokens)));
        ui.label(RichText::new(dollars(t.cost_usd)).strong());
    });
    if data.since_ms > 0 {
        ui.weak(format!("Counting since {}", stamp(data.since_ms)));
    } else {
        ui.weak("No finished turn has been counted yet.");
    }
}

/// Draws the window content for the usage in `store`; the reset button clears it.
pub fn body(ui: &mut Ui, store: &UsageStore) {
    let data = store.snapshot();
    totals_row(ui, &data);
    ui.horizontal_wrapped(|ui| {
        let confirm = ui.ctx().data(|d| d.get_temp::<bool>(confirm_id())).unwrap_or(false);
        if confirm {
            if ui.button("Reset for good").clicked() {
                store.reset();
                ui.ctx().data_mut(|d| d.insert_temp(confirm_id(), false));
            }
            if ui.button("Keep").clicked() {
                ui.ctx().data_mut(|d| d.insert_temp(confirm_id(), false));
            }
        } else if ui.button("Reset").on_hover_text("Forget every counted turn").clicked() {
            ui.ctx().data_mut(|d| d.insert_temp(confirm_id(), true));
        }
        if let Some(path) = store.path() {
            ui.weak(path.display().to_string());
        }
    });
    ui.separator();
    ui.label(RichText::new(format!("Per session ({})", data.sessions.len())).strong());
    ScrollArea::vertical().id_salt("zc_usage_sessions").max_height(220.0).auto_shrink([false, true]).show(ui, |ui| {
        Grid::new("zc_usage_session_grid").striped(true).num_columns(6).show(ui, |ui| {
            for head in ["Session", "Turns", "In", "Out", "Cost", "Last"] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for s in data.sessions.iter().take(SESSION_ROWS) {
                ui.label(short(&s.session)).on_hover_text(&s.session);
                ui.label(grouped(s.totals.turns));
                ui.label(grouped(s.totals.input_tokens));
                ui.label(grouped(s.totals.output_tokens));
                ui.label(dollars(s.totals.cost_usd));
                ui.label(stamp(s.last_ms)).on_hover_text(s.model.clone().unwrap_or_default());
                ui.end_row();
            }
        });
    });
    ui.separator();
    ui.label(RichText::new(format!("Per day ({})", data.days.len())).strong());
    ScrollArea::vertical().id_salt("zc_usage_days").max_height(220.0).auto_shrink([false, true]).show(ui, |ui| {
        Grid::new("zc_usage_day_grid").striped(true).num_columns(5).show(ui, |ui| {
            for head in ["Day", "Turns", "In", "Out", "Cost"] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for d in data.days.iter().take(DAY_ROWS) {
                ui.label(&d.day);
                ui.label(grouped(d.totals.turns));
                ui.label(grouped(d.totals.input_tokens));
                ui.label(grouped(d.totals.output_tokens));
                ui.label(dollars(d.totals.cost_usd));
                ui.end_row();
            }
        });
    });
    ui.weak(format!("Updated {}", stamp(now_ms())));
}

pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: ZeroClaw usage").open(&mut open).default_width(560.0).default_height(520.0).show(ctx, |ui| body(ui, usage::store()));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::{press_release, test_ctx, texts};
    use egui::{CentralPanel, Pos2, RawInput, Rect, vec2};
    use zeroclaw_client::Usage;

    fn turn(model: &str, i: u64, o: u64, c: f64) -> Usage {
        Usage { input_tokens: Some(i), output_tokens: Some(o), cost_usd: Some(c), model: Some(model.into()), ..Default::default() }
    }

    fn frame(ctx: &Context, store: &UsageStore, width: f32, events: Vec<egui::Event>) -> Vec<(String, Pos2)> {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 900.0))), events, ..Default::default() };
        let out = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| body(ui, store));
        });
        let drawn = texts(&out.shapes);
        out.drop_without_applying_deltas();
        drawn
    }

    fn rects(ctx: &Context, store: &UsageStore, width: f32) -> Vec<(String, Rect)> {
        fn walk(shape: &egui::Shape, out: &mut Vec<(String, Rect)>) {
            match shape {
                egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 900.0))), ..Default::default() };
        let full = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| body(ui, store));
        });
        let mut out = Vec::new();
        full.shapes.iter().for_each(|s| walk(&s.shape, &mut out));
        full.drop_without_applying_deltas();
        out
    }

    fn seeded() -> UsageStore {
        let store = UsageStore::in_memory();
        let t0 = 1_760_000_000_000;
        store.record_at(t0, "chat-one", &turn("qwen", 1200, 300, 0.0123));
        store.record_at(t0 + 1000, "chat-one", &turn("qwen", 2400, 100, 0.002));
        store.record_at(t0 + 48 * 3600 * 1000, "chat-two-with-a-very-long-session-identifier", &turn("qwen", 10, 5, 0.0));
        store
    }

    #[test]
    fn the_window_shows_totals_and_both_tables() {
        let ctx = test_ctx();
        let store = seeded();
        frame(&ctx, &store, 900.0, vec![]);
        let drawn: Vec<String> = frame(&ctx, &store, 900.0, vec![]).into_iter().map(|t| t.0).collect();
        for expected in ["3 turns", "3,610 tokens in", "405 tokens out", "$0.01", "Per session (2)", "Per day (2)", "Session", "Day", "chat-one", "Reset"] {
            assert!(drawn.iter().any(|t| t.contains(expected)), "{expected}: {drawn:?}");
        }
        assert!(drawn.iter().any(|t| t == "2" || t == "1"), "turn counts per row");
        assert!(drawn.iter().any(|t| t.ends_with("...")), "a long session id is cut: {drawn:?}");
        assert!(drawn.iter().any(|t| t.starts_with("Counting since")));
    }

    #[test]
    fn an_empty_store_says_so_and_reset_asks_first() {
        let ctx = test_ctx();
        let store = UsageStore::in_memory();
        let drawn: Vec<String> = frame(&ctx, &store, 700.0, vec![]).into_iter().map(|t| t.0).collect();
        assert!(drawn.iter().any(|t| t.contains("No finished turn")), "{drawn:?}");
        let store = seeded();
        let first = frame(&ctx, &store, 700.0, vec![]);
        let reset = first.iter().find(|(t, _)| t == "Reset").unwrap().1;
        frame(&ctx, &store, 700.0, press_release(reset + vec2(12.0, 8.0)));
        let asked: Vec<String> = frame(&ctx, &store, 700.0, vec![]).into_iter().map(|t| t.0).collect();
        assert!(asked.iter().any(|t| t == "Reset for good") && asked.iter().any(|t| t == "Keep"), "{asked:?}");
        assert_eq!(store.snapshot().total.turns, 3, "nothing is forgotten before the confirmation");
        let confirm = frame(&ctx, &store, 700.0, vec![]).into_iter().find(|(t, _)| t == "Reset for good").unwrap().1;
        frame(&ctx, &store, 700.0, press_release(confirm + vec2(12.0, 8.0)));
        frame(&ctx, &store, 700.0, vec![]);
        assert_eq!(store.snapshot(), UsageData::default());
    }

    #[test]
    fn the_tables_fit_every_width() {
        let ctx = test_ctx();
        let store = seeded();
        for width in [700.0, 1000.0, 1600.0] {
            rects(&ctx, &store, width);
            for (text, rect) in rects(&ctx, &store, width) {
                assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at {width}: {rect:?}");
            }
        }
    }

    #[test]
    fn the_toggle_opens_and_closes_the_window() {
        let ctx = test_ctx();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
        toggle(&ctx);
        assert!(!is_open(&ctx));
    }
}
