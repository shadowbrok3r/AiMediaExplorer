//! Bottom bar: view statistics on the left, activity chips and the Task Center in the middle, logs and platform meters on the right.

pub mod adapter;
pub mod center;
pub mod items;
pub mod layout;
pub mod view_stats;

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use eframe::egui::*;

use self::center::{CenterAction, TaskView};
use self::items::{BarAction, Item, entry_of};
use self::layout::{Mode, plan};
use self::view_stats::{ViewStats, ViewStatsCache};
use crate::command::{AppCommand, Tool};
use crate::icons::{self, IconSize, svg};
use crate::tasks;
use crate::ui::file_table::FileExplorer;

/// Narrowest window width the bar guarantees not to overlap at.
pub const MIN_SUPPORTED_WIDTH: f32 = 640.0;
/// Repaint interval while a task is visible, for progress and elapsed time.
const ACTIVE_REPAINT: Duration = Duration::from_millis(250);
const METER_REFRESH: Duration = Duration::from_millis(500);

/* ------------------------------------------- shared state ------------------------------------------ */

fn center_id() -> Id {
    Id::new("ame_task_center_open")
}

pub fn is_center_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(center_id())).unwrap_or(false)
}

/// Opens or closes the Task Center popover.
pub fn set_center_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(center_id(), open));
    ctx.request_repaint();
}

fn view_id() -> Id {
    Id::new("ame_task_view")
}

/// The registry as of now, rebuilt only when its revision moved.
pub fn task_view(ctx: &Context) -> TaskView {
    let revision = tasks::with(|r| r.revision());
    if let Some(view) = ctx.data(|d| d.get_temp::<TaskView>(view_id())).filter(|v| v.revision == revision) {
        return view;
    }
    let view = tasks::with(|r| TaskView {
        revision: r.revision(),
        tasks: std::sync::Arc::new(r.tasks().to_vec()),
        attention: std::sync::Arc::new(r.attention()),
    });
    ctx.data_mut(|d| d.insert_temp(view_id(), view.clone()));
    view
}

/// Rectangles the last drawn bar occupied, for tests and devtools.
#[derive(Clone, Copy, Debug)]
pub struct BarLayout {
    pub left: Rect,
    pub right: Rect,
    pub panel: Rect,
    pub overflowed: usize,
    pub shown: usize,
}

pub fn last_layout(ctx: &Context, owner: &str) -> Option<BarLayout> {
    ctx.data(|d| d.get_temp::<BarLayout>(Id::new(("ame_bar_layout", owner))))
}

/// Number of times the stats of `owner` were recomputed.
pub fn stats_computes(ctx: &Context, owner: &str) -> u64 {
    ctx.data(|d| d.get_temp::<ViewStatsCache>(Id::new(("ame_view_stats", owner)))).map_or(0, |c| c.computes)
}

/* ------------------------------------------- platform meters ---------------------------------------- */

/// Live system meter readings; `None` when the platform provides no data.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMetrics {
    pub cpu01: Option<f32>,
    pub ram: Option<(f32, f32)>,
    pub vram: Option<(f32, f32)>,
}

impl SystemMetrics {
    /// Readings from the platform helpers; unsupported meters stay `None`.
    pub fn sample() -> Self {
        use crate::utilities::windows::{gpu_mem_mb, smoothed_cpu01, system_mem_mb};
        Self {
            cpu01: cfg!(windows).then(smoothed_cpu01),
            ram: system_mem_mb().filter(|(_, total)| *total > 0.0),
            vram: gpu_mem_mb().filter(|(_, total)| *total > 0.0),
        }
    }

    pub fn any(&self) -> bool {
        self.cpu01.is_some() || self.ram.is_some() || self.vram.is_some()
    }

    fn items(&self) -> Vec<Item> {
        let mut items = Vec::new();
        if let Some(cpu) = self.cpu01 {
            items.push(Item::Meter { icon: svg::CPU, label: "CPU", pct: cpu.clamp(0.0, 1.0), detail: format!("{:.0}%", cpu * 100.0) });
        }
        if let Some((used, total)) = self.ram {
            items.push(Item::Meter { icon: svg::MEMORY, label: "RAM", pct: (used / total).clamp(0.0, 1.0), detail: format!("{used:.0}/{total:.0} MiB") });
        }
        if let Some((used, total)) = self.vram {
            items.push(Item::Meter { icon: svg::AME_GPU, label: "VRAM", pct: (used / total).clamp(0.0, 1.0), detail: format!("{used:.0}/{total:.0} MiB") });
        }
        items
    }
}

/* ----------------------------------------------- drawing -------------------------------------------- */

fn text_button_width(ui: &Ui, text: &str) -> f32 {
    let font = TextStyle::Button.resolve(ui.style());
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, Color32::WHITE);
    galley.size().x + ui.spacing().button_padding.x * 2.0 + 2.0
}

fn separator(ui: &mut Ui, width: f32) {
    let height = ui.available_height().min(18.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let x = rect.center().x;
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter().line_segment([pos2(x, rect.top() + 2.0), pos2(x, rect.bottom() - 2.0)], stroke);
}

/// Width the Logs button takes.
fn logs_width(ui: &Ui) -> f32 {
    text_button_width(ui, "Logs") + IconSize::Md.px(ui) + ui.spacing().item_spacing.x
}

/// Menus of the explorer that stay reachable from the bar until the selection actions move.
fn explorer_menus(ui: &mut Ui, ex: &mut FileExplorer, owner: &str) {
    ex.logical_group_menu(ui);
    ex.selection_menu(ui, owner);
}

fn apply_center_action(ctx: &Context, action: CenterAction) {
    match action {
        CenterAction::Cancel(id) => {
            tasks::with(|r| r.request_cancel(id));
        }
        CenterAction::Command(cmd) => crate::command::send(cmd),
        CenterAction::Dismiss(id) => {
            tasks::with(|r| r.dismiss(id));
        }
        CenterAction::ClearFinished => {
            tasks::with(|r| r.clear_finished());
        }
    }
    ctx.request_repaint();
}

/// Draws the bar for `ex` with the live registry and the platform's meters.
pub fn show(ui: &mut Ui, ex: &mut FileExplorer, owner: &str) {
    let view = task_view(ui.ctx());
    show_with(ui, ex, owner, &view, SystemMetrics::sample(), Instant::now());
}

/// Draws the bar with an explicit task view, meter readings and time.
pub fn show_with(ui: &mut Ui, ex: &mut FileExplorer, owner: &str, view: &TaskView, metrics: SystemMetrics, now: Instant) {
    let ctx = ui.ctx().clone();
    let m = crate::theme::metrics(&ctx);
    let mut actions: Vec<BarAction> = Vec::new();
    let mut anchor: Option<Response> = None;
    Panel::bottom("FileExplorer Bottom Panel").exact_size(m.bar_h).show(ui, |ui| {
        let stats = cached_stats(&ctx, ex, owner, now);
        let chips = tasks::activity_chips(&view.tasks, now);
        let items = items::build(&stats, chips, view.visible_active(now), view.attention.len(), metrics.items());

        let gap = m.gap.x;
        let inner = gap / 2.0;
        let step = inner * 2.0 + 1.0;
        let rect = ui.available_rect_before_wrap();
        let select_w = text_button_width(ui, &ex.selection_label());
        let groups_w = text_button_width(ui, &ex.logical_group_label());
        let right_w = (select_w + gap + groups_w + gap + logs_width(ui)).min(rect.width());
        let right_rect = Rect::from_min_max(pos2(rect.right() - right_w, rect.top()), rect.right_bottom());
        let left_rect = Rect::from_min_max(rect.left_top(), pos2((right_rect.left() - gap).max(rect.left()), rect.bottom()));

        let overflow_w = ui.spacing().interact_size.y.max(IconSize::Md.px(ui) + 4.0);
        let entries: Vec<_> = items.iter().map(|it| entry_of(ui, it, inner)).collect();
        let plan = plan(&entries, step, overflow_w, left_rect.width());

        let right = ui.scope_builder(UiBuilder::new().max_rect(right_rect).layout(Layout::right_to_left(Align::Center)), |ui| {
            ui.spacing_mut().item_spacing.x = gap;
            if icons::button(ui, svg::TERMINAL_WINDOW, "Logs").on_hover_text("Open the log tab").clicked() {
                crate::command::send(AppCommand::OpenTool { tool: Tool::Logs, path: None, target: crate::command::Target::NewTab });
            }
            explorer_menus(ui, ex, owner);
        });

        let left = ui.scope_builder(UiBuilder::new().max_rect(left_rect).layout(Layout::left_to_right(Align::Center)), |ui| {
            ui.set_clip_rect(left_rect.intersect(ui.clip_rect()));
            ui.spacing_mut().item_spacing.x = inner;
            let mut first = true;
            for (item, mode) in items.iter().zip(&plan.modes) {
                let Some(mode) = mode else { continue };
                if !first {
                    separator(ui, 1.0);
                }
                first = false;
                let drawn = item.draw(ui, *mode);
                actions.extend(drawn.action);
                if drawn.anchor.is_some() {
                    anchor = drawn.anchor;
                }
            }
            if plan.overflow {
                if !first {
                    separator(ui, 1.0);
                }
                icons::icon_menu_button(ui, svg::DOTS_THREE, "More status items", |ui| {
                    for (item, mode) in items.iter().zip(&plan.modes) {
                        if mode.is_none() {
                            ui.horizontal(|ui| actions.extend(item.draw(ui, Mode::Full).action));
                        }
                    }
                });
            }
        });

        if items.iter().zip(&plan.modes).any(|(it, mode)| it.is_meter() && mode.is_some()) {
            ctx.request_repaint_after(METER_REFRESH);
        }
        if view.visible_active(now) > 0 {
            ctx.request_repaint_after(ACTIVE_REPAINT);
        }
        if let Some(after) = view.next_reveal(now) {
            ctx.request_repaint_after(after);
        }
        let layout = BarLayout {
            left: left.response.rect,
            right: right.response.rect,
            panel: rect,
            overflowed: plan.modes.iter().filter(|m| m.is_none()).count(),
            shown: plan.modes.iter().filter(|m| m.is_some()).count(),
        };
        ctx.data_mut(|d| d.insert_temp(Id::new(("ame_bar_layout", owner)), layout));
    });

    let mut open = is_center_open(&ctx);
    for action in actions {
        match action {
            BarAction::ClearFilters => adapter::clear_filters(ex),
            BarAction::ToggleCenter => open = !open,
        }
    }
    if let Some(anchor) = anchor {
        let before = open;
        for action in center::show(&anchor, &mut open, view, now) {
            apply_center_action(&ctx, action);
        }
        if open != before || open != is_center_open(&ctx) {
            set_center_open(&ctx, open);
        }
    } else if open != is_center_open(&ctx) {
        set_center_open(&ctx, open);
    }
}

/// Stats of the explorer for the bar, recomputed only when its signature changes.
fn cached_stats(ctx: &Context, ex: &FileExplorer, owner: &str, now: Instant) -> std::sync::Arc<ViewStats> {
    let id = Id::new(("ame_view_stats", owner));
    let mut cache = ctx.data_mut(|d| d.remove_temp::<ViewStatsCache>(id)).unwrap_or_default();
    let answer = cache.get(now, adapter::view_signature(ex), adapter::is_streaming(ex), || adapter::compute_view_stats(ex));
    ctx.data_mut(|d| d.insert_temp(id, cache));
    if let Some(after) = answer.repaint_after {
        ctx.request_repaint_after(after);
    }
    answer.stats
}
