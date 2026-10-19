//! Devtools window for the action registry of the focused tab: every action, whether it can run on the current selection and why not, and a Run button.

use std::time::Duration;

use eframe::egui::{Button, Context, Grid, Id, RichText, ScrollArea, Ui, Window};

use crate::command::ActionId;
use crate::command::actions::Facts;
use crate::icons::{self, svg};

/// One action of the registry as the window shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub id: ActionId,
    pub menu_id: &'static str,
    pub group: &'static str,
    pub shortcut: Option<&'static str>,
    pub in_bar: bool,
    pub confirm: &'static str,
    pub needs: &'static str,
    /// Files the action would act on now.
    pub count: usize,
    pub enabled: Result<(), String>,
}

/// One run in the history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRow {
    pub age: Duration,
    pub action: ActionId,
    pub origin: &'static str,
    pub count: usize,
    pub ok: bool,
    pub outcome: String,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub owner: Id,
    pub label: String,
    pub facts: Facts,
    pub popover: String,
    pub rows: Vec<Row>,
    pub history: Vec<RunRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Runs the action on the selection as the bar does.
    Run(ActionId),
}

fn open_id() -> Id {
    Id::unique("ame_devtools_browser_actions_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_browser_actions_snapshot")
}

fn commands_id(owner: Id) -> Id {
    Id::unique(("ame_devtools_browser_actions_commands", owner))
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

/// The JSON name of an action, as `run` takes it.
pub fn json_name(id: ActionId) -> String {
    serde_json::to_value(id).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Browser actions").open(&mut open).default_width(900.0).default_height(560.0).show(ctx, |ui| {
        ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| match latest(ctx) {
            Some(snapshot) => body(ui, &snapshot),
            None => {
                ui.weak("No explorer has drawn since the window opened.");
            }
        });
    });
    set_open(ctx, open);
}

fn body(ui: &mut Ui, s: &Snapshot) {
    ui.ctx().request_repaint_after(Duration::from_millis(250));
    ui.label(RichText::new(&s.label).weak());
    let f = &s.facts;
    ui.label(format!("{} selected: {} files, {} folders, {} on disk, {} images, {} bytes; library {}, vision model {}; popover: {}", f.selected, f.files, f.dirs, f.local, f.images, f.size, if f.db_ready { "connected" } else { "not connected" }, if f.ai_ready { "ready" } else { "not set up" }, s.popover));
    ui.separator();
    actions(ui, s);
    ui.separator();
    history(ui, s);
}

fn actions(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::LIST_BULLETS, "Actions");
    let ctx = ui.ctx().clone();
    Grid::new("browser_actions_grid").num_columns(10).striped(true).spacing([14.0, 3.0]).show(ui, |ui| {
        for head in ["Action", "Menu id", "Group", "Key", "Bar", "Asks", "Needs", "Acts on", "Enabled"] {
            ui.strong(head);
        }
        ui.strong("");
        ui.end_row();
        for row in &s.rows {
            ui.monospace(json_name(row.id));
            ui.monospace(row.menu_id);
            ui.label(row.group);
            ui.label(row.shortcut.unwrap_or(""));
            ui.label(if row.in_bar { "yes" } else { "menu only" });
            ui.label(row.confirm);
            ui.label(row.needs);
            ui.monospace(row.count.to_string());
            match &row.enabled {
                Ok(()) => ui.label("yes"),
                Err(reason) => ui.label(RichText::new(format!("no: {reason}")).color(ui.visuals().warn_fg_color)),
            };
            if ui.add_enabled(row.enabled.is_ok(), Button::new("Run").small()).clicked() {
                push_command(&ctx, s.owner, Command::Run(row.id));
            }
            ui.end_row();
        }
    });
}

fn history(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::CLOCK_COUNTER_CLOCKWISE, "Runs");
    if s.history.is_empty() {
        ui.weak("No action has run yet.");
        return;
    }
    Grid::new("browser_actions_history").num_columns(5).striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
        for head in ["When", "Action", "From", "Files", "Result"] {
            ui.strong(head);
        }
        ui.end_row();
        for run in s.history.iter().rev() {
            ui.label(format!("{} s ago", run.age.as_secs()));
            ui.monospace(json_name(run.action));
            ui.label(run.origin);
            ui.monospace(run.count.to_string());
            let text = RichText::new(&run.outcome);
            ui.label(if run.ok { text } else { text.color(ui.visuals().error_fg_color) });
            ui.end_row();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_reach_only_their_owner_and_only_once() {
        let ctx = Context::default();
        let (a, b) = (Id::unique("a"), Id::unique("b"));
        push_command(&ctx, a, Command::Run(ActionId::Tag));
        push_command(&ctx, b, Command::Run(ActionId::Open));
        assert_eq!(take_commands(&ctx, a), [Command::Run(ActionId::Tag)]);
        assert!(take_commands(&ctx, a).is_empty());
        assert_eq!(take_commands(&ctx, b), [Command::Run(ActionId::Open)]);
    }

    #[test]
    fn action_names_are_the_json_names_of_the_run_command() {
        assert_eq!(json_name(ActionId::RemoveFromLibrary), "remove_from_library");
        assert_eq!(json_name(ActionId::Tag), "tag");
        for id in ActionId::ALL {
            assert!(!json_name(id).is_empty());
        }
    }

    use crate::command::DevtoolsWindow;
    use crate::ui::file_table::action_bar::{Origin, Popover, pin_environment};
    use crate::ui::file_table::test_rig::Rig;

    #[test]
    fn the_window_lists_every_action_with_its_state_for_the_current_selection() {
        pin_environment(Some(true), Some(true));
        let mut rig = Rig::new(1000.0, 900.0);
        crate::ui::devtools::open(&rig.ctx, DevtoolsWindow::BrowserActions);
        rig.devtools = true;
        rig.select_images(0..2);
        rig.settle(3);
        for id in [ActionId::Open, ActionId::Reveal, ActionId::CopyPath, ActionId::OpenInNewTab] {
            assert!(rig.has_text(&json_name(id)), "{id:?} is not listed");
        }
        let snapshot = latest(&rig.ctx).expect("published");
        assert_eq!(snapshot.rows.len(), ActionId::ALL.len());
        assert_eq!(snapshot.facts.selected, 2);
        let tag = snapshot.rows.iter().find(|r| r.id == ActionId::Tag).unwrap();
        assert!(tag.enabled.is_ok() && tag.count == 2 && tag.shortcut == Some("T"));
        let exclude = snapshot.rows.iter().find(|r| r.id == ActionId::ExcludeFolder).unwrap();
        assert_eq!(exclude.enabled, Err("Select a folder first".to_string()), "the reason a button is off is shown");
    }

    #[test]
    fn a_run_request_runs_the_action_on_the_selection_with_the_devtools_origin() {
        pin_environment(Some(true), Some(true));
        let mut rig = Rig::new(1000.0, 900.0);
        crate::ui::devtools::open(&rig.ctx, DevtoolsWindow::BrowserActions);
        rig.devtools = true;
        rig.select_images(0..2);
        rig.settle(3);
        let owner = latest(&rig.ctx).unwrap().owner;
        push_command(&rig.ctx, owner, Command::Run(ActionId::CopyPath));
        rig.settle(2);
        assert_eq!(rig.copied.len(), 1);
        let run = rig.ex.toolbar.bar.runs.back().unwrap();
        assert_eq!((run.action, run.origin), (ActionId::CopyPath, Origin::Devtools));
        rig.settle(3);
        assert_eq!(latest(&rig.ctx).unwrap().history.len(), 1, "the run is in the history the window shows");
        push_command(&rig.ctx, owner, Command::Run(ActionId::Tag));
        rig.settle(3);
        assert_eq!(rig.ex.toolbar.bar.popover, Some(Popover::Tag), "the button opens the same popover as the bar");
    }

    #[test]
    fn the_window_opens_and_closes() {
        let ctx = Context::default();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
    }
}
