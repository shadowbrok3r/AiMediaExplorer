//! The Task Center popover: needs-attention items, running tasks and the recent history.

use std::sync::Arc;
use std::time::Instant;

use eframe::egui::*;

use super::items::icon_of;
use crate::command::AppCommand;
use crate::icons::{self, svg};
use crate::tasks::{AttentionItem, AttentionKind, Task, TaskId, TaskState, format_elapsed};

/// Width of the popover.
pub const WIDTH: f32 = 400.0;
const LIST_MAX_HEIGHT: f32 = 320.0;

/// What the registry looked like when the frame began.
#[derive(Clone, Default)]
pub struct TaskView {
    pub revision: u64,
    pub tasks: Arc<Vec<Task>>,
    pub attention: Arc<Vec<AttentionItem>>,
}

impl TaskView {
    pub fn active(&self) -> usize {
        self.tasks.iter().filter(|t| t.state.is_active()).count()
    }

    pub fn visible_active(&self, now: Instant) -> usize {
        self.tasks.iter().filter(|t| t.is_visible(now)).count()
    }

    pub fn finished(&self) -> impl DoubleEndedIterator<Item = &Task> {
        self.tasks.iter().filter(|t| !t.state.is_active())
    }

    /// Time until the first active task that is still too young to show becomes visible.
    pub fn next_reveal(&self, now: Instant) -> Option<std::time::Duration> {
        self.tasks.iter().filter(|t| t.state.is_active() && !t.is_visible(now)).map(|t| t.visible_after.saturating_sub(t.elapsed(now))).min()
    }
}

/// What a click in the popover asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum CenterAction {
    Cancel(TaskId),
    Command(AppCommand),
    Dismiss(TaskId),
    ClearFinished,
}

fn section(ui: &mut Ui, title: &str, count: usize) {
    ui.add_space(4.0);
    ui.label(RichText::new(format!("{title} ({count})")).strong());
    ui.separator();
}

fn attention_row(ui: &mut Ui, item: &AttentionItem, actions: &mut Vec<CenterAction>) {
    ui.horizontal(|ui| {
        icons::label(ui, svg::WARNING, RichText::new(&item.title).strong());
        if let Some(command) = &item.action {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let (label, icon) = match item.kind {
                    AttentionKind::FailedThumbs => ("Show failures", svg::FUNNEL),
                    AttentionKind::Database => ("Retry", svg::ARROW_CLOCKWISE),
                    _ => ("Open", svg::FUNNEL),
                };
                if icons::small_button(ui, icon, label).clicked() {
                    actions.push(CenterAction::Command(command.clone()));
                }
            });
        }
    });
    if !item.detail.is_empty() {
        ui.label(RichText::new(&item.detail).weak());
    }
}

fn active_row(ui: &mut Ui, task: &Task, now: Instant, actions: &mut Vec<CenterAction>) {
    ui.horizontal(|ui| {
        icons::label(ui, icon_of(task.kind), RichText::new(&task.title).strong());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(format_elapsed(task.elapsed(now))).weak());
            if task.state == TaskState::Queued {
                ui.label(RichText::new("waiting").weak());
            }
        });
    });
    ui.horizontal(|ui| {
        let button_w = if task.cancel.is_some() { 84.0 } else { 0.0 };
        let width = (ui.available_width() - button_w - ui.spacing().item_spacing.x).max(60.0);
        match task.fraction() {
            Some(f) => ui.add(ProgressBar::new(f).desired_width(width).show_percentage()),
            None => ui.add(ProgressBar::new(0.0).desired_width(width).text("working")),
        };
        if task.cancel.is_some() {
            let cancelling = task.cancel_requested;
            let text = if cancelling { "Stopping" } else { "Cancel" };
            ui.add_enabled_ui(!cancelling, |ui| {
                if crate::ui::danger_icon_button(ui, svg::STOP, text).clicked() {
                    actions.push(CenterAction::Cancel(task.id));
                }
            });
        }
    });
    if !task.detail.is_empty() {
        ui.label(RichText::new(&task.detail).weak());
    }
    if let Some(scope) = &task.failure_scope {
        if icons::small_button(ui, svg::FUNNEL, "Show failures").clicked() {
            actions.push(CenterAction::Command(AppCommand::open(scope.clone())));
        }
    }
}

fn finished_row(ui: &mut Ui, task: &Task, now: Instant, actions: &mut Vec<CenterAction>) {
    let (icon, tint) = match task.state {
        TaskState::Completed => (svg::CHECK_CIRCLE, crate::theme::success(ui)),
        TaskState::Failed => (svg::X_CIRCLE, crate::theme::error(ui)),
        _ => (svg::PROHIBIT, ui.visuals().weak_text_color()),
    };
    ui.horizontal(|ui| {
        ui.add(icon.image(14.0).tint(tint));
        ui.label(&task.title);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if icons::icon_button(ui, svg::X, "Remove from the list").clicked() {
                actions.push(CenterAction::Dismiss(task.id));
            }
            let ago = task.ended.map(|e| now.saturating_duration_since(e)).unwrap_or_default();
            ui.label(RichText::new(format!("{} ago", format_elapsed(ago))).weak());
        });
    });
    if let Some(error) = &task.error {
        ui.label(RichText::new(error).color(crate::theme::error(ui)));
    }
    ui.horizontal(|ui| {
        if let (TaskState::Failed, Some(retry)) = (task.state, &task.retry) {
            if icons::small_button(ui, svg::ARROW_CLOCKWISE, "Retry").clicked() {
                actions.push(CenterAction::Command(retry.clone()));
            }
        }
        if let (true, Some(scope)) = (task.state == TaskState::Failed, &task.failure_scope) {
            if icons::small_button(ui, svg::FUNNEL, "Show failures").clicked() {
                actions.push(CenterAction::Command(AppCommand::open(scope.clone())));
            }
        }
    });
}

/// The popover body; returns what the user asked for.
pub fn body(ui: &mut Ui, view: &TaskView, now: Instant) -> Vec<CenterAction> {
    let mut actions = Vec::new();
    let active: Vec<&Task> = view.tasks.iter().filter(|t| t.state.is_active()).collect();
    let finished: Vec<&Task> = view.finished().rev().collect();
    ui.horizontal(|ui| {
        ui.heading("Tasks");
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.add_enabled(!finished.is_empty(), Button::new("Clear finished")).clicked() {
                actions.push(CenterAction::ClearFinished);
            }
        });
    });
    ScrollArea::vertical().max_height(LIST_MAX_HEIGHT).auto_shrink([true, true]).show(ui, |ui| {
        if view.attention.is_empty() && active.is_empty() && finished.is_empty() {
            ui.label(RichText::new("Nothing is running.").weak());
            return;
        }
        if !view.attention.is_empty() {
            section(ui, "Needs attention", view.attention.len());
            for item in view.attention.iter() {
                attention_row(ui, item, &mut actions);
                ui.add_space(2.0);
            }
        }
        if !active.is_empty() {
            section(ui, "Running", active.len());
            for task in active {
                active_row(ui, task, now, &mut actions);
                ui.add_space(4.0);
            }
        }
        if !finished.is_empty() {
            section(ui, "Recent", finished.len());
            for task in finished {
                finished_row(ui, task, now, &mut actions);
                ui.add_space(2.0);
            }
        }
    });
    actions
}

/// Shows the popover above `anchor` while `open`; clicking outside closes it.
pub fn show(anchor: &Response, open: &mut bool, view: &TaskView, now: Instant) -> Vec<CenterAction> {
    let mut actions = Vec::new();
    Popup::from_response(anchor)
        .open_bool(open)
        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
        .align(RectAlign::TOP_START)
        .gap(6.0)
        .width(WIDTH)
        .show(|ui| actions = body(ui, view, now));
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{Outcome, TaskKind, TaskRegistry, TaskSpec};

    fn view_of(r: &TaskRegistry) -> TaskView {
        TaskView { revision: r.revision(), tasks: Arc::new(r.tasks().to_vec()), attention: Arc::new(r.attention()) }
    }

    #[test]
    fn the_view_counts_active_visible_and_finished_tasks() {
        let mut r = TaskRegistry::new();
        let now = Instant::now();
        r.begin_at(TaskSpec::new(TaskKind::Scan, "a"), now);
        r.begin_at(TaskSpec::new(TaskKind::Database, "db").visible_after(std::time::Duration::from_secs(2)), now);
        let done = r.begin_at(TaskSpec::new(TaskKind::Clip, "c"), now);
        r.finish_at(done, Outcome::Completed, now);
        let view = view_of(&r);
        assert_eq!(view.active(), 2);
        assert_eq!(view.visible_active(now), 1);
        assert_eq!(view.finished().count(), 1);
        assert_eq!(view.next_reveal(now), Some(std::time::Duration::from_secs(2)));
        assert_eq!(view.next_reveal(now + std::time::Duration::from_millis(1500)), Some(std::time::Duration::from_millis(500)));
        assert_eq!(view.next_reveal(now + std::time::Duration::from_secs(3)), None);
    }
}
