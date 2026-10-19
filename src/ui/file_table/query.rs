//! The query bar: its text and mode, how they become filter conditions or a scope, and the widget.

use std::collections::VecDeque;
use std::time::Instant;

use eframe::egui::*;

use super::FileExplorer;
use crate::command::{CommandError, CommandResult, QueryMode, Scope};
use crate::icons::{self, Icon, svg};
use crate::theme::controls::{Segment, segmented};
use crate::ui::titlebar::kit_icon;

/// Runs kept for the Devtools window.
pub const HISTORY: usize = 40;

/// What the text of the bar asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Conditions of the tab's filter: file name and search text.
    Filter { name: String, query: String },
    /// A scope the tab shows.
    Open(Scope),
    /// Nothing to run, with the reason.
    Idle(String),
}

impl Plan {
    /// No filter conditions from the bar.
    fn clear() -> Plan {
        Plan::Filter { name: String::new(), query: String::new() }
    }
}

/// The plan for `text` in `mode`; `enter` is set when Enter was pressed. Typing alone never leaves the tab's scope: the searches wait for Enter and clear the live filter meanwhile.
pub fn plan(mode: QueryMode, text: &str, focus: Option<&str>, enter: bool) -> Plan {
    let text = text.trim();
    match mode {
        QueryMode::Name => Plan::Filter { name: text.to_string(), query: String::new() },
        QueryMode::Text => Plan::Filter { name: String::new(), query: text.to_string() },
        QueryMode::Semantic | QueryMode::Similar if !enter => Plan::clear(),
        QueryMode::Semantic if text.is_empty() => Plan::Idle("Type what to look for first".to_string()),
        QueryMode::Semantic => Plan::Open(Scope::Semantic(text.to_string())),
        QueryMode::Similar => match (text.is_empty(), focus) {
            (false, _) => Plan::Open(Scope::Similar(text.to_string())),
            (true, Some(path)) => Plan::Open(Scope::Similar(path.to_string())),
            (true, None) => Plan::Idle("Select a file or type a path first".to_string()),
        },
    }
}

/// One line telling what `plan` does.
pub fn describe(plan: &Plan) -> String {
    match plan {
        Plan::Filter { name, query } if name.is_empty() && query.is_empty() => "filter: no name or text condition".to_string(),
        Plan::Filter { name, query } if query.is_empty() => format!("filter: file name contains \"{name}\""),
        Plan::Filter { name, query } if name.is_empty() => format!("filter: search text \"{query}\" in name, type, path, tags, category, description"),
        Plan::Filter { name, query } => format!("filter: name \"{name}\" and text \"{query}\""),
        Plan::Open(scope) => format!("open {}", scope.format()),
        Plan::Idle(reason) => format!("nothing to run: {reason}"),
    }
}

/// The icon of a query mode.
pub fn mode_icon(mode: QueryMode) -> Icon {
    match mode {
        QueryMode::Name => svg::FILE,
        QueryMode::Text => svg::LIST_MAGNIFYING_GLASS,
        QueryMode::Semantic => svg::AME_SEMANTIC_SEARCH,
        QueryMode::Similar => svg::AME_FIND_SIMILAR,
    }
}

/// One run of the bar, for the Devtools history.
#[derive(Clone, Debug)]
pub struct Run {
    pub at: Instant,
    pub mode: QueryMode,
    pub text: String,
    pub plan: String,
    pub outcome: Result<String, String>,
}

/// State of the query bar of one tab.
#[derive(Debug, Default)]
pub struct QueryBar {
    pub mode: QueryMode,
    pub text: String,
    /// The name and search text last written to the filter, to tell changes made elsewhere.
    applied: Option<(String, String)>,
    scope_seen: Option<Scope>,
    /// The field takes the keyboard focus on the next frame.
    pub focus: bool,
    pub history: VecDeque<Run>,
    pub runs: u64,
}

impl QueryBar {
    pub fn record(&mut self, mode: QueryMode, text: String, plan: String, outcome: Result<String, String>) {
        if self.history.len() >= HISTORY {
            self.history.pop_front();
        }
        self.runs += 1;
        self.history.push_back(Run { at: Instant::now(), mode, text, plan, outcome });
    }

    /// Takes in a filter that changed outside the bar: a chip removed, a saved view, a command.
    fn reflect(&mut self, now: &(String, String)) {
        if self.applied.as_ref() == Some(now) {
            return;
        }
        let (name, query) = now;
        let pick = match self.mode {
            QueryMode::Text => (!query.is_empty()).then_some((QueryMode::Text, query)).or((!name.is_empty()).then_some((QueryMode::Name, name))),
            _ => (!name.is_empty()).then_some((QueryMode::Name, name)).or((!query.is_empty()).then_some((QueryMode::Text, query))),
        };
        match pick {
            Some((mode, text)) => {
                self.mode = mode;
                self.text = text.clone();
            }
            None if self.mode.is_live() => self.text.clear(),
            None => {}
        }
        self.applied = Some(now.clone());
    }

    /// Follows the scope the tab shows: a search scope puts its text in the bar, any other scope leaves a search mode.
    fn follow_scope(&mut self, scope: &Scope) {
        if self.scope_seen.as_ref() == Some(scope) {
            return;
        }
        match scope {
            Scope::Semantic(text) => {
                self.mode = QueryMode::Semantic;
                self.text = text.clone();
            }
            Scope::Similar(path) => {
                self.mode = QueryMode::Similar;
                self.text = path.clone();
            }
            _ if !self.mode.is_live() && self.scope_seen.is_some() => {
                self.mode = QueryMode::Name;
                self.text.clear();
            }
            _ => {}
        }
        self.scope_seen = Some(scope.clone());
    }
}

impl FileExplorer {
    /// The file the query and Find similar start from: the previewed row, else the last previewed file.
    pub fn focus_file(&self) -> Option<String> {
        let focused = self.selection.preview_target().and_then(|k| self.table.get_by_key(k)).filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone());
        focused.or_else(|| Some(self.current_thumb.path.clone()).filter(|p| !p.is_empty()))
    }

    /// Brings the bar in step with filters and scope changes made elsewhere; call once per frame before drawing it.
    pub fn sync_query(&mut self) {
        let scope = self.nav.scope.clone();
        self.toolbar.query.follow_scope(&scope);
        let spec = self.filter_spec();
        self.toolbar.query.reflect(&(spec.name, spec.query));
    }

    /// Writes what the bar holds to the tab's filter: the conditions of Name and Text, none while a search mode is chosen.
    pub fn apply_query_live(&mut self) {
        let q = &self.toolbar.query;
        let (name, query) = match plan(q.mode, &q.text, None, false) {
            Plan::Filter { name, query } => (name, query),
            _ => (String::new(), String::new()),
        };
        self.edit_filters(|s| {
            s.name = name;
            s.query = query;
        });
        let spec = self.filter_spec();
        self.toolbar.query.applied = Some((spec.name, spec.query));
    }

    /// Presses Enter in the bar: Name and Text keep their filter, Semantic and Similar open their scope.
    pub fn run_query(&mut self) -> CommandResult {
        let (mode, text) = (self.toolbar.query.mode, self.toolbar.query.text.clone());
        let planned = plan(mode, &text, self.focus_file().as_deref(), true);
        let result = match &planned {
            Plan::Filter { .. } => {
                self.apply_query_live();
                Ok(describe(&planned))
            }
            Plan::Open(scope) => self.open_scope(scope.clone()).map(|()| describe(&planned)).map_err(CommandError::not_found),
            Plan::Idle(reason) => Err(CommandError::invalid(reason.clone())),
        };
        let outcome = result.as_ref().map(String::clone).map_err(|e| e.message.clone());
        self.toolbar.query.record(mode, text, describe(&planned), outcome);
        result
    }

    /// Sets the bar from a command: mode and text, then the live filter, and with `run` Enter.
    pub fn apply_query_command(&mut self, mode: QueryMode, text: &str, run: bool) -> CommandResult {
        self.toolbar.query.mode = mode;
        self.toolbar.query.text = text.to_string();
        self.apply_query_live();
        if run {
            return self.run_query();
        }
        Ok(format!("query set to {} \"{}\"", mode.label(), text))
    }
}

fn dev_owner(ex: &FileExplorer) -> Id {
    Id::unique(("ame_query_dev_owner", ex.viewer.scope))
}

impl FileExplorer {
    /// Runs what the Query devtools window asked for and publishes what it shows while it is open.
    pub(crate) fn run_query_dev(&mut self, ctx: &Context) {
        use crate::ui::devtools::query as dev;
        let owner = dev_owner(self);
        for command in dev::take_commands(ctx, owner) {
            match command {
                dev::Command::Set { mode, text, run } => {
                    let _ = self.apply_query_command(mode, &text, run);
                }
                dev::Command::Clear => {
                    self.toolbar.query.text.clear();
                    self.apply_query_live();
                }
                dev::Command::FocusField => self.toolbar.query.focus = true,
            }
        }
        if !dev::is_open(ctx) {
            return;
        }
        self.sync_view();
        let spec = self.filter_spec();
        let q = &self.toolbar.query;
        let focus = self.focus_file();
        let now = Instant::now();
        let history = q
            .history
            .iter()
            .map(|r| dev::RunRow { age: now.saturating_duration_since(r.at), mode: r.mode, text: r.text.clone(), plan: r.plan.clone(), ok: r.outcome.is_ok(), outcome: r.outcome.clone().unwrap_or_else(|e| e) })
            .collect();
        dev::publish(
            ctx,
            dev::Snapshot {
                owner,
                label: self.tab_title(),
                scope: self.nav.scope.format(),
                mode: q.mode,
                text: q.text.clone(),
                typing: describe(&plan(q.mode, &q.text, focus.as_deref(), false)),
                enter: describe(&plan(q.mode, &q.text, focus.as_deref(), true)),
                filter_name: spec.name,
                filter_query: spec.query,
                focus_file: focus,
                listed: self.view.len(),
                total: self.table.len(),
                runs: q.runs,
                history,
            },
        );
    }
}

/// How the mode control is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeStyle {
    /// Four joined segments with icon and name.
    Labels,
    /// Four joined icons.
    Icons,
    /// One icon opening a menu.
    Menu,
}

/// Where the parts of the bar were drawn.
#[derive(Clone, Copy, Debug)]
pub struct QueryRects {
    pub mode: Rect,
    pub text: Rect,
}

pub(crate) fn text_id(ex: &FileExplorer) -> Id {
    Id::unique(("ame_query_text", ex.viewer.scope))
}

fn mode_control(ui: &mut Ui, mode: &mut QueryMode, style: ModeStyle) -> Response {
    match style {
        ModeStyle::Labels | ModeStyle::Icons => {
            let labels = style == ModeStyle::Labels;
            let segments: Vec<Segment<'_, QueryMode>> = QueryMode::ALL.into_iter().map(|m| Segment::new(m, if labels { m.label() } else { "" }).icon(kit_icon(mode_icon(m)))).collect();
            let response = segmented(ui, mode, &segments);
            response.on_hover_text("Name and Text filter as you type; Semantic and Similar search when you press Enter")
        }
        ModeStyle::Menu => {
            let current = *mode;
            icons::icon_menu_button(ui, mode_icon(current), &format!("Search mode: {}", current.label()), |ui| {
                for m in QueryMode::ALL {
                    if icons::selectable_button(ui, mode_icon(m), current == m, m.label()).clicked() {
                        *mode = m;
                        ui.close();
                    }
                }
            })
            .response
        }
    }
}

/// Draws the bar into `ui`, left to right, and applies what the user did in it.
pub fn show(ui: &mut Ui, ex: &mut FileExplorer, style: ModeStyle) -> QueryRects {
    ex.sync_query();
    let id = text_id(ex);
    let before_mode = ex.toolbar.query.mode;
    let mut mode = before_mode;
    let mode_response = mode_control(ui, &mut mode, style);
    let mut apply = false;
    let mut focus_text = false;
    if mode != before_mode {
        ex.toolbar.query.mode = mode;
        apply = true;
        focus_text = true;
    }
    if std::mem::take(&mut ex.toolbar.query.focus) {
        focus_text = true;
    }
    let had_focus = ui.memory(|m| m.had_focus_last_frame(id));
    let escape = ui.input(|i| i.key_pressed(Key::Escape));
    let spacing = ui.spacing().item_spacing.x;
    let clear_w = ui.spacing().interact_size.y.max(icons::IconSize::Md.px(ui) + 4.0);
    let width = (ui.available_width() - spacing - clear_w - spacing).max(40.0);
    let hint = ex.toolbar.query.mode.hint();
    let mut enter = false;
    let response = TextEdit::singleline(&mut ex.toolbar.query.text).id(id).hint_text(hint).desired_width(width).show(ui).response;
    if response.changed() {
        apply = true;
    }
    if response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
        enter = true;
    }
    if had_focus && escape && response.lost_focus() {
        ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));
        if !ex.toolbar.query.text.is_empty() {
            ex.toolbar.query.text.clear();
            apply = true;
            focus_text = true;
        }
    }
    if focus_text {
        ui.memory_mut(|m| m.request_focus(id));
    }
    if !ex.toolbar.query.text.is_empty() && icons::icon_button(ui, svg::X, "Clear the query (Esc)").clicked() {
        ex.toolbar.query.text.clear();
        apply = true;
    }
    if apply {
        ex.apply_query_live();
    }
    if enter {
        if let Err(e) = ex.run_query() {
            crate::ui::dialogs::toast(egui_toast::ToastKind::Warning, e.message);
        }
        if ex.toolbar.query.mode.is_live() {
            ui.memory_mut(|m| m.request_focus(id));
        }
    }
    QueryRects { mode: mode_response.rect, text: response.rect }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(name: &str, query: &str) -> Plan {
        Plan::Filter { name: name.into(), query: query.into() }
    }

    #[test]
    fn name_and_text_filter_live_and_trim() {
        assert_eq!(plan(QueryMode::Name, "  cat ", None, false), filter("cat", ""));
        assert_eq!(plan(QueryMode::Text, "red, fox", None, false), filter("", "red, fox"));
        assert_eq!(plan(QueryMode::Name, "cat", None, true), filter("cat", ""));
        assert_eq!(plan(QueryMode::Text, "", None, true), filter("", ""));
    }

    #[test]
    fn semantic_and_similar_wait_for_enter_and_clear_the_live_filter_meanwhile() {
        assert_eq!(plan(QueryMode::Semantic, "red car", None, false), filter("", ""));
        assert_eq!(plan(QueryMode::Similar, "/a.jpg", None, false), filter("", ""));
        assert_eq!(plan(QueryMode::Semantic, " red car ", None, true), Plan::Open(Scope::Semantic("red car".into())));
        assert_eq!(plan(QueryMode::Similar, " /p/a.jpg ", Some("/p/b.jpg"), true), Plan::Open(Scope::Similar("/p/a.jpg".into())));
    }

    #[test]
    fn similar_without_text_uses_the_focused_file() {
        assert_eq!(plan(QueryMode::Similar, "  ", Some("/p/b.jpg"), true), Plan::Open(Scope::Similar("/p/b.jpg".into())));
        assert!(matches!(plan(QueryMode::Similar, "", None, true), Plan::Idle(_)));
        assert!(matches!(plan(QueryMode::Semantic, "  ", Some("/p/b.jpg"), true), Plan::Idle(_)));
    }

    #[test]
    fn every_plan_reads_back_as_a_sentence() {
        assert!(describe(&filter("cat", "")).contains("file name contains \"cat\""));
        assert!(describe(&filter("", "fox")).contains("search text \"fox\""));
        assert!(describe(&filter("a", "b")).contains("name \"a\" and text \"b\""));
        assert!(describe(&filter("", "")).contains("no name or text"));
        assert_eq!(describe(&Plan::Open(Scope::Semantic("red car".into()))), "open q:sem:\"red car\"");
        assert!(describe(&Plan::Idle("why".into())).contains("why"));
    }

    #[test]
    fn a_scope_string_round_trips_through_the_plan() {
        for (mode, text) in [(QueryMode::Semantic, "a \"quoted\" car"), (QueryMode::Similar, "/p/a b.jpg")] {
            let Plan::Open(scope) = plan(mode, text, None, true) else { panic!("{mode:?}") };
            assert_eq!(Scope::parse(&scope.format()).unwrap(), scope);
        }
    }

    #[test]
    fn a_filter_set_elsewhere_shows_in_the_bar_and_removing_it_empties_the_bar() {
        let mut q = QueryBar::default();
        q.reflect(&(String::new(), String::new()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Name, ""));
        q.reflect(&(String::new(), "fox".into()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Text, "fox"), "a search text found in the filter switches to Text");
        q.reflect(&("dog".into(), String::new()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Name, "dog"));
        q.reflect(&(String::new(), String::new()));
        assert_eq!(q.text, "", "the chip was removed");
    }

    #[test]
    fn the_filter_the_bar_wrote_itself_is_not_taken_in_again() {
        let mut q = QueryBar { mode: QueryMode::Text, text: "fox ".into(), ..Default::default() };
        q.applied = Some((String::new(), "fox".into()));
        q.reflect(&(String::new(), "fox".into()));
        assert_eq!(q.text, "fox ", "the typed trailing space stays");
    }

    #[test]
    fn a_search_mode_keeps_its_text_while_no_filter_is_set() {
        let mut q = QueryBar { mode: QueryMode::Semantic, text: "red car".into(), ..Default::default() };
        q.applied = Some((String::new(), String::new()));
        q.reflect(&(String::new(), String::new()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Semantic, "red car"));
        q.reflect(&("x".into(), String::new()));
        assert_eq!(q.mode, QueryMode::Name, "a filter chosen while a search mode is on takes the bar over");
    }

    #[test]
    fn a_search_scope_fills_the_bar_and_leaving_it_resets_the_mode() {
        let mut q = QueryBar::default();
        q.follow_scope(&Scope::Library);
        q.text = "typed".into();
        q.follow_scope(&Scope::Semantic("sunset".into()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Semantic, "sunset"));
        q.follow_scope(&Scope::Similar("/p/a.jpg".into()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Similar, "/p/a.jpg"));
        q.follow_scope(&Scope::Folder("/p".into()));
        assert_eq!((q.mode, q.text.as_str()), (QueryMode::Name, ""));
        q.text = "typed".into();
        q.follow_scope(&Scope::Folder("/p".into()));
        assert_eq!(q.text, "typed", "the same scope changes nothing");
    }

    #[test]
    fn the_history_keeps_the_newest_runs() {
        let mut q = QueryBar::default();
        for i in 0..HISTORY + 5 {
            q.record(QueryMode::Name, format!("t{i}"), "plan".into(), Ok("ok".into()));
        }
        assert_eq!(q.history.len(), HISTORY);
        assert_eq!(q.history.front().unwrap().text, "t5");
        assert_eq!(q.runs as usize, HISTORY + 5);
    }
}
