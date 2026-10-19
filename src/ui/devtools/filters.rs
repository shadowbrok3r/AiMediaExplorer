//! Devtools window for the filter of the focused tab: its spec as JSON, why rows are hidden and what each condition removes.

use std::time::Duration;

use eframe::egui::{Color32, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::icons::{self, svg};
use crate::ui::file_table::rows::{HiddenReport, Refresh, RefreshCounts, Verdict};

/// Outcome of asking why one path is listed or hidden.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplainReport {
    pub path: String,
    /// The tab holds a row for the path.
    pub found: bool,
    pub listed: bool,
    pub verdicts: Vec<Verdict>,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub owner: Id,
    pub label: String,
    pub spec_json: String,
    pub base_json: String,
    pub signature: u64,
    pub chips: usize,
    pub rows_total: usize,
    pub rows_visible: usize,
    pub view_revision: u64,
    pub refresh: RefreshCounts,
    pub last_refresh: Refresh,
    pub changes: u64,
    pub since_change: Option<Duration>,
    pub report: HiddenReport,
    pub probe_entries: usize,
    pub probe_pending: usize,
    pub probe_generation: u64,
    pub indexed: String,
    pub explain: Option<ExplainReport>,
    pub views: Vec<String>,
    pub active_view: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    ApplyJson(String),
    Clear,
    Reset,
    MakeDefault,
    Explain(String),
}

#[derive(Clone, Default)]
struct Editor {
    text: String,
    synced: Option<(Id, u64)>,
    dirty: bool,
    path: String,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_filters_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_filters_snapshot")
}

fn commands_id(owner: Id) -> Id {
    Id::unique(("ame_devtools_filters_commands", owner))
}

fn editor_id() -> Id {
    Id::unique("ame_devtools_filters_editor")
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
    Window::new("Devtools: Filters").open(&mut open).default_width(540.0).default_height(660.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match latest(ctx) {
            Some(snapshot) => body(ui, &snapshot),
            None => {
                ui.weak("No explorer has drawn since the window opened.");
            }
        });
    });
    set_open(ctx, open);
}

fn age_text(since: Option<Duration>) -> String {
    match since {
        None => "never".to_string(),
        Some(d) if d.as_secs() < 1 => format!("{} ms ago", d.as_millis()),
        Some(d) => format!("{} s ago", d.as_secs()),
    }
}

fn body(ui: &mut Ui, s: &Snapshot) {
    ui.ctx().request_repaint_after(Duration::from_millis(250));
    ui.label(RichText::new(&s.label).weak());
    summary(ui, s);
    ui.separator();
    conditions(ui, &s.report);
    ui.separator();
    editor(ui, s);
    ui.separator();
    why_hidden(ui, s);
}

fn summary(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::FUNNEL, "Filter");
    Grid::new("filters_summary").num_columns(4).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Signature");
        ui.monospace(format!("{:016x}", s.signature));
        ui.label("Active filters");
        ui.monospace(s.chips.to_string());
        ui.end_row();
        ui.label("Rows in tab");
        ui.monospace(s.rows_total.to_string());
        ui.label("Listed / hidden");
        ui.monospace(format!("{} / {}", s.rows_visible, s.rows_total.saturating_sub(s.rows_visible)));
        ui.end_row();
        ui.label("View revision");
        ui.monospace(s.view_revision.to_string());
        ui.label("Last refresh");
        ui.monospace(format!("{:?}", s.last_refresh));
        ui.end_row();
        ui.label("Refreshes");
        ui.monospace(format!("{} unchanged, {} appended, {} rebuilt", s.refresh.unchanged, s.refresh.appended, s.refresh.rebuilt));
        ui.label("Filter changes");
        ui.monospace(format!("{} (last {})", s.changes, age_text(s.since_change)));
        ui.end_row();
        ui.label("File probes");
        ui.monospace(format!("{} cached, {} pending, generation {}", s.probe_entries, s.probe_pending, s.probe_generation));
        ui.label("Indexed lookup");
        ui.monospace(&s.indexed);
        ui.end_row();
        ui.label("Saved views");
        ui.monospace(if s.views.is_empty() { "none loaded".to_string() } else { s.views.join(", ") });
        ui.label("Active view");
        ui.monospace(s.active_view.as_deref().unwrap_or("-"));
        ui.end_row();
    });
}

fn conditions(ui: &mut Ui, report: &HiddenReport) {
    icons::heading(ui, svg::LIST_CHECKS, "Hidden by condition");
    if report.per_condition.is_empty() {
        ui.weak("No condition is active.");
        return;
    }
    Grid::new("filters_conditions").num_columns(3).spacing([16.0, 3.0]).striped(true).show(ui, |ui| {
        ui.strong("Condition");
        ui.strong("Hides");
        ui.strong("Hides only this");
        ui.end_row();
        for c in &report.per_condition {
            ui.label(c.condition.label());
            ui.monospace(c.failing.to_string());
            ui.monospace(c.only.to_string());
            ui.end_row();
        }
    });
    ui.label(format!("{} of {} rows listed", report.visible, report.total));
}

fn editor(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::BRACKETS_CURLY, "Spec");
    let ctx = ui.ctx().clone();
    let mut ed: Editor = ctx.data(|d| d.get_temp(editor_id())).unwrap_or_default();
    if !ed.dirty && ed.synced != Some((s.owner, s.signature)) {
        ed.text = s.spec_json.clone();
        ed.synced = Some((s.owner, s.signature));
    }
    let resp = ui.add(TextEdit::multiline(&mut ed.text).code_editor().desired_rows(10).desired_width(f32::INFINITY));
    if resp.changed() {
        ed.dirty = true;
    }
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::CHECK, "Apply").on_hover_text("Replace the tab's filters with this JSON").clicked() {
            push_command(&ctx, s.owner, Command::ApplyJson(ed.text.clone()));
            ed.dirty = false;
            ed.synced = None;
        }
        if icons::button(ui, svg::ARROW_COUNTER_CLOCKWISE, "Reload").on_hover_text("Discard edits and show the current filters").clicked() {
            ed.dirty = false;
            ed.synced = None;
        }
        if icons::button(ui, svg::X, "Clear all").on_hover_text("Remove every condition, the defaults included").clicked() {
            push_command(&ctx, s.owner, Command::Clear);
            ed.dirty = false;
        }
        if icons::button(ui, svg::HOUSE_SIMPLE, "Reset to defaults").on_hover_text("Go back to the conditions the settings define").clicked() {
            push_command(&ctx, s.owner, Command::Reset);
            ed.dirty = false;
        }
        if icons::button(ui, svg::FLOPPY_DISK, "Make default").on_hover_text("Write the size, date, icon, extension and folder conditions to the settings").clicked() {
            push_command(&ctx, s.owner, Command::MakeDefault);
        }
    });
    if let Some(e) = &s.error {
        ui.colored_label(crate::theme::error(ui), e);
    }
    ui.collapsing("Defaults from the settings", |ui| {
        ui.add(TextEdit::multiline(&mut s.base_json.as_str()).code_editor().desired_rows(6).desired_width(f32::INFINITY));
    });
    ctx.data_mut(|d| d.insert_temp(editor_id(), ed));
}

fn why_hidden(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::MAGNIFYING_GLASS, "Why is this row hidden");
    let ctx = ui.ctx().clone();
    let mut ed: Editor = ctx.data(|d| d.get_temp(editor_id())).unwrap_or_default();
    ui.horizontal(|ui| {
        let resp = ui.add(TextEdit::singleline(&mut ed.path).hint_text("path of a row in this tab").desired_width(ui.available_width() - 90.0));
        let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(eframe::egui::Key::Enter));
        if (icons::button(ui, svg::MAGNIFYING_GLASS, "Explain").clicked() || enter) && !ed.path.trim().is_empty() {
            push_command(&ctx, s.owner, Command::Explain(ed.path.clone()));
        }
    });
    if let Some(report) = &s.explain {
        if !report.found {
            ui.colored_label(crate::theme::warn(ui), format!("{} is not a row of this tab", report.path));
        } else {
            let (text, color) = if report.listed { ("Listed", crate::theme::success(ui)) } else { ("Hidden", crate::theme::error(ui)) };
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(color, RichText::new(text).strong());
                ui.monospace(&report.path);
            });
            if report.verdicts.is_empty() {
                ui.weak("No condition is active.");
            }
            for v in &report.verdicts {
                let (mark, color): (&str, Color32) = if v.passed { ("pass", crate::theme::success(ui)) } else { ("FAIL", crate::theme::error(ui)) };
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(color, mark);
                    ui.label(RichText::new(v.condition.label()).strong());
                    ui.label(&v.detail);
                });
            }
        }
    }
    ctx.data_mut(|d| d.insert_temp(editor_id(), ed));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_reach_only_their_owner_and_only_once() {
        let ctx = Context::default();
        let (a, b) = (Id::unique("a"), Id::unique("b"));
        push_command(&ctx, a, Command::Clear);
        push_command(&ctx, a, Command::Explain("/x".into()));
        push_command(&ctx, b, Command::Reset);
        assert_eq!(take_commands(&ctx, a), [Command::Clear, Command::Explain("/x".into())]);
        assert!(take_commands(&ctx, a).is_empty());
        assert_eq!(take_commands(&ctx, b), [Command::Reset]);
    }

    #[test]
    fn the_window_opens_and_closes() {
        let ctx = Context::default();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
        set_open(&ctx, false);
        assert!(!is_open(&ctx));
    }

    #[test]
    fn ages_read_in_milliseconds_then_seconds() {
        assert_eq!(age_text(None), "never");
        assert_eq!(age_text(Some(Duration::from_millis(40))), "40 ms ago");
        assert_eq!(age_text(Some(Duration::from_secs(3))), "3 s ago");
    }
}
