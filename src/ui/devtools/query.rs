//! Devtools window for the query bar of the focused tab: its text and mode, what they parse to, what Enter runs and the history of runs.

use std::time::Duration;

use eframe::egui::{Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::command::QueryMode;
use crate::icons::{self, svg};

/// One run of the bar in the history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRow {
    pub age: Duration,
    pub mode: QueryMode,
    pub text: String,
    pub plan: String,
    pub ok: bool,
    pub outcome: String,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub owner: Id,
    pub label: String,
    pub scope: String,
    pub mode: QueryMode,
    pub text: String,
    /// What typing does with the text.
    pub typing: String,
    /// What Enter does with the text.
    pub enter: String,
    /// The name and search text now in the tab's filter.
    pub filter_name: String,
    pub filter_query: String,
    pub focus_file: Option<String>,
    pub listed: usize,
    pub total: usize,
    pub runs: u64,
    pub history: Vec<RunRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Sets the bar; with `run` Enter is pressed.
    Set { mode: QueryMode, text: String, run: bool },
    Clear,
    FocusField,
}

#[derive(Clone, Default)]
struct Editor {
    mode: QueryMode,
    text: String,
    synced: Option<(Id, QueryMode, String)>,
    dirty: bool,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_query_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_query_snapshot")
}

fn commands_id(owner: Id) -> Id {
    Id::unique(("ame_devtools_query_commands", owner))
}

fn editor_id() -> Id {
    Id::unique("ame_devtools_query_editor")
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
    Window::new("Devtools: Query").open(&mut open).default_width(560.0).default_height(520.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match latest(ctx) {
            Some(snapshot) => body(ui, &snapshot),
            None => {
                ui.weak("No explorer has drawn since the window opened.");
            }
        });
    });
    set_open(ctx, open);
}

fn age_text(age: Duration) -> String {
    if age.as_secs() < 1 { format!("{} ms ago", age.as_millis()) } else { format!("{} s ago", age.as_secs()) }
}

fn body(ui: &mut Ui, s: &Snapshot) {
    ui.ctx().request_repaint_after(Duration::from_millis(250));
    ui.label(RichText::new(&s.label).weak());
    summary(ui, s);
    ui.separator();
    editor(ui, s);
    ui.separator();
    history(ui, s);
}

fn summary(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::MAGNIFYING_GLASS, "Query bar");
    Grid::new("query_summary").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Scope");
        ui.monospace(&s.scope);
        ui.end_row();
        ui.label("Mode and text");
        ui.monospace(format!("{} \"{}\"", s.mode.label(), s.text));
        ui.end_row();
        ui.label("Typing runs");
        ui.monospace(&s.typing);
        ui.end_row();
        ui.label("Enter runs");
        ui.monospace(&s.enter);
        ui.end_row();
        ui.label("Filter now holds");
        ui.monospace(format!("name \"{}\", search text \"{}\"", s.filter_name, s.filter_query));
        ui.end_row();
        ui.label("Focused file");
        ui.monospace(s.focus_file.as_deref().unwrap_or("-"));
        ui.end_row();
        ui.label("Listed / rows");
        ui.monospace(format!("{} / {}", s.listed, s.total));
        ui.end_row();
        ui.label("Runs");
        ui.monospace(s.runs.to_string());
        ui.end_row();
    });
}

fn editor(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::PENCIL_SIMPLE, "Drive the bar");
    let ctx = ui.ctx().clone();
    let mut ed: Editor = ctx.data(|d| d.get_temp(editor_id())).unwrap_or_default();
    let key = (s.owner, s.mode, s.text.clone());
    if !ed.dirty && ed.synced.as_ref() != Some(&key) {
        ed.mode = s.mode;
        ed.text = s.text.clone();
        ed.synced = Some(key);
    }
    ui.horizontal(|ui| {
        for mode in QueryMode::ALL {
            if ui.selectable_label(ed.mode == mode, mode.label()).clicked() {
                ed.mode = mode;
                ed.dirty = true;
            }
        }
    });
    let width = ui.available_width();
    if ui.add(TextEdit::singleline(&mut ed.text).hint_text(ed.mode.hint()).desired_width(width)).changed() {
        ed.dirty = true;
    }
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::CHECK, "Set").on_hover_text("Put this mode and text in the bar and apply the live filter").clicked() {
            push_command(&ctx, s.owner, Command::Set { mode: ed.mode, text: ed.text.clone(), run: false });
            ed.dirty = false;
            ed.synced = None;
        }
        if icons::button(ui, svg::PLAY, "Run").on_hover_text("Set, then press Enter").clicked() {
            push_command(&ctx, s.owner, Command::Set { mode: ed.mode, text: ed.text.clone(), run: true });
            ed.dirty = false;
            ed.synced = None;
        }
        if icons::button(ui, svg::X, "Clear").clicked() {
            push_command(&ctx, s.owner, Command::Clear);
            ed.dirty = false;
            ed.synced = None;
        }
        if icons::button(ui, svg::CURSOR_CLICK, "Focus the field").on_hover_text("Same as Ctrl+F").clicked() {
            push_command(&ctx, s.owner, Command::FocusField);
        }
    });
    ui.weak("Command line: --run '{\"cmd\":\"query\",\"mode\":\"semantic\",\"text\":\"a red car\",\"run\":true}' or --query semantic:'a red car'");
    ctx.data_mut(|d| d.insert_temp(editor_id(), ed));
}

fn history(ui: &mut Ui, s: &Snapshot) {
    icons::heading(ui, svg::CLOCK_COUNTER_CLOCKWISE, "History of runs");
    if s.history.is_empty() {
        ui.weak("Nothing has been run yet.");
        return;
    }
    Grid::new("query_history").num_columns(5).striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
        for head in ["When", "Mode", "Text", "Ran", "Result"] {
            ui.strong(head);
        }
        ui.end_row();
        for run in s.history.iter().rev() {
            ui.label(age_text(run.age));
            ui.label(run.mode.label());
            ui.monospace(&run.text);
            ui.monospace(&run.plan);
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
        push_command(&ctx, a, Command::Clear);
        push_command(&ctx, a, Command::Set { mode: QueryMode::Text, text: "x".into(), run: true });
        push_command(&ctx, b, Command::FocusField);
        assert_eq!(take_commands(&ctx, a), [Command::Clear, Command::Set { mode: QueryMode::Text, text: "x".into(), run: true }]);
        assert!(take_commands(&ctx, a).is_empty());
        assert_eq!(take_commands(&ctx, b), [Command::FocusField]);
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

    use crate::command::DevtoolsWindow;
    use crate::ui::file_table::test_rig::Rig;

    #[test]
    fn the_window_shows_the_bar_what_it_runs_and_the_history() {
        let mut rig = Rig::new(1000.0, 800.0);
        crate::ui::devtools::open(&rig.ctx, DevtoolsWindow::Query);
        rig.devtools = true;
        rig.ex.apply_query_command(QueryMode::Name, "img_03", false).unwrap();
        rig.ex.apply_query_command(QueryMode::Semantic, "a red car", true).unwrap();
        rig.settle(3);
        for text in ["Devtools: Query", "Scope", "Typing runs", "Enter runs", "Filter now holds", "History of runs", "Drive the bar"] {
            assert!(rig.has_text(text), "{text} is missing");
        }
        assert!(rig.has_text("filter: no name or text condition"), "typing in a search mode runs no filter");
        assert!(rig.has_text("open q:sem:\"a red car\""), "Enter opens the search scope, and the history lists it too");
        assert!(rig.has_text("Semantic \"a red car\""));
    }

    #[test]
    fn a_command_from_the_window_sets_and_runs_the_bar_of_its_explorer() {
        let mut rig = Rig::new(1000.0, 800.0);
        crate::ui::devtools::open(&rig.ctx, DevtoolsWindow::Query);
        rig.devtools = true;
        rig.settle(3);
        let owner = latest(&rig.ctx).expect("the explorer published").owner;
        push_command(&rig.ctx, owner, Command::Set { mode: QueryMode::Name, text: "img_04".into(), run: false });
        rig.settle(3);
        assert_eq!(rig.ex.filter_spec().name, "img_04");
        assert_eq!(rig.ex.view.len(), 1);
        push_command(&rig.ctx, owner, Command::Clear);
        rig.settle(3);
        assert_eq!(rig.ex.filter_spec().name, "");
        push_command(&rig.ctx, owner, Command::FocusField);
        rig.settle(3);
        assert!(rig.ctx.memory(|m| m.has_focus(crate::ui::file_table::query::text_id(&rig.ex))));
    }

    #[test]
    fn ages_read_in_milliseconds_then_seconds() {
        assert_eq!(age_text(Duration::from_millis(40)), "40 ms ago");
        assert_eq!(age_text(Duration::from_secs(3)), "3 s ago");
    }
}
