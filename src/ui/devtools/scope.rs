//! Devtools window for scopes: parse a scope string, open it, move through the history of the focused tab and read the loader timings.

use std::time::Duration;

use eframe::egui::{Color32, Context, FontId, Id, Label, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::command::{self, AppCommand, NavOp, Scope, Target};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::scope::LoadRecord;
use crate::ui::file_table::scope::nav::check_openable;

fn open_id() -> Id {
    Id::new("ame_devtools_scope_open")
}

fn state_id() -> Id {
    Id::new("ame_devtools_scope_state")
}

fn snapshot_id() -> Id {
    Id::new("ame_devtools_scope_snapshot")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

#[derive(Clone, Default)]
struct State {
    input: String,
    message: Option<(bool, String)>,
}

/// One entry of the history stack.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryLine {
    pub scope: String,
    pub scroll_row: usize,
    pub anchor: Option<String>,
}

/// The load a tab is waiting for.
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveLine {
    pub id: u64,
    pub scope: String,
    pub rows: usize,
    pub batches: usize,
    pub elapsed: Duration,
    pub cancelled: bool,
}

/// What the window shows about the focused tab.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub tab: String,
    pub scope: String,
    pub kind: &'static str,
    pub rows: usize,
    pub loading: bool,
    pub more: bool,
    pub rows_only: bool,
    pub error: Option<String>,
    pub position: usize,
    pub history: Vec<HistoryLine>,
    pub active: Option<ActiveLine>,
    pub log: Vec<LoadRecord>,
    /// Rows known to be in the library, disk checks started and in flight.
    pub badges: (usize, usize, usize),
}

/// Reads the facts the window shows from an explorer.
pub fn snapshot_of(ex: &FileExplorer) -> Snapshot {
    let nav = &ex.nav;
    Snapshot {
        tab: ex.tab_title(),
        scope: nav.scope.format(),
        kind: nav.scope.kind(),
        rows: ex.table.len(),
        loading: ex.is_loading(),
        more: nav.more.is_some(),
        rows_only: nav.rows_only,
        error: nav.error.clone(),
        position: nav.history.position(),
        history: nav.history.entries().iter().map(|e| HistoryLine { scope: e.scope.format(), scroll_row: e.scroll_row, anchor: e.anchor.clone() }).collect(),
        active: nav.active.as_ref().filter(|a| a.is_running()).map(|a| ActiveLine { id: a.id, scope: a.scope.format(), rows: a.rows, batches: a.batches, elapsed: a.started.elapsed(), cancelled: a.cancelled }),
        log: nav.log.iter().cloned().collect(),
        badges: (ex.viewer.badges.indexed_len(), ex.viewer.badges.requested(), ex.viewer.badges.pending()),
    }
}

/// Makes the focused tab the one the window describes; does nothing while the window is closed.
pub fn publish(ctx: &Context, ex: &FileExplorer) {
    if is_open(ctx) {
        ctx.data_mut(|d| d.insert_temp(snapshot_id(), snapshot_of(ex)));
    }
}

/// What a scope string means, for the parsed-form line.
#[derive(Clone, Debug, PartialEq)]
pub struct Parsed {
    pub scope: Scope,
    pub formatted: String,
    pub kind: &'static str,
    pub title: String,
    pub parent: Option<String>,
    /// Why the tab could not show it, if it could not.
    pub problem: Option<String>,
}

/// Parses `text` the way `--scope` does.
pub fn parse_text(text: &str) -> Result<Parsed, String> {
    let scope = Scope::parse(text.trim()).map_err(|e| e.to_string())?;
    Ok(Parsed {
        formatted: scope.format(),
        kind: scope.kind(),
        title: scope.default_title(),
        parent: scope.parent().map(|p| p.format()),
        problem: check_openable(&scope).err(),
        scope,
    })
}

/// Queues the command that opens the scope in `text`; the message says what was queued or why not.
pub fn open_text(text: &str, target: Target) -> (bool, String) {
    match parse_text(text) {
        Ok(parsed) => {
            command::send(AppCommand::OpenScope { scope: parsed.scope, target, title: None, recursive: false });
            (true, format!("Queued {} ({:?})", parsed.formatted, target))
        }
        Err(e) => (false, e),
    }
}

fn millis(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1000.0)
}

fn record_row(ui: &mut Ui, r: &LoadRecord) {
    ui.horizontal(|ui| {
        let (status, color) = match (&r.error, r.cancelled) {
            (Some(_), _) => ("failed", crate::theme::error(ui)),
            (None, true) => ("cancelled", ui.visuals().weak_text_color()),
            (None, false) => ("ok", ui.visuals().text_color()),
        };
        ui.label(RichText::new(status).color(color));
        let total = r.total.map_or(String::new(), |t| format!(" of {t}"));
        let first = r.first_batch.map_or("-".to_string(), millis);
        let line = format!("{}  {} rows{total}  {} batches  first {first}  total {}", r.scope, r.rows, r.batches, millis(r.elapsed));
        ui.label(RichText::new(line).font(FontId::monospace(12.0)));
    });
    if let Some(error) = &r.error {
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.add(Label::new(RichText::new(error).color(crate::theme::error(ui))).truncate()).on_hover_text(error);
        });
    }
}

fn body(ctx: &Context, ui: &mut Ui) {
    let mut state = ctx.data(|d| d.get_temp::<State>(state_id())).unwrap_or_default();
    let snap = ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id())).unwrap_or_default();
    if state.input.is_empty() {
        state.input = snap.scope.clone();
    }
    ui.label("Scope string in, parsed form out. Examples: fs:/path, lib:all, col:Name, tag:x, cat:x, sim:/path, scan:42, smart:no-clip, q:sem:\"red car\".");
    let edit = ui.add(TextEdit::singleline(&mut state.input).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY).hint_text("lib:all"));
    match parse_text(&state.input) {
        Ok(p) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("{}  kind: {}  title: {}", p.formatted, p.kind, p.title)).font(FontId::monospace(12.0)));
                if let Some(parent) = &p.parent {
                    ui.label(RichText::new(format!("up: {parent}")).weak());
                }
            });
            if let Some(problem) = &p.problem {
                ui.label(RichText::new(problem).color(crate::theme::warn(ui)));
            }
        }
        Err(e) => {
            ui.label(RichText::new(e).color(crate::theme::error(ui)));
        }
    }
    ui.horizontal(|ui| {
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if icons::button(ui, svg::PLAY, "Open").clicked() || enter {
            state.message = Some(open_text(&state.input, Target::Current));
        }
        if icons::button(ui, svg::AME_NEW_TAB, "Open in new tab").clicked() {
            state.message = Some(open_text(&state.input, Target::NewTab));
        }
        if ui.add_enabled(snap.position > 0, egui::Button::new("Back")).clicked() {
            command::send(AppCommand::Navigate { op: NavOp::Back });
        }
        if ui.add_enabled(snap.position + 1 < snap.history.len(), egui::Button::new("Forward")).clicked() {
            command::send(AppCommand::Navigate { op: NavOp::Forward });
        }
        if let Some((ok, text)) = &state.message {
            let color = if *ok { Color32::GRAY } else { crate::theme::error(ui) };
            ui.label(RichText::new(text).color(color));
        }
    });
    ui.separator();
    ui.label(RichText::new(format!("Focused tab: {}", if snap.tab.is_empty() { "-" } else { &snap.tab })).strong());
    ui.label(RichText::new(format!(
        "scope {}  ({})  rows {}  loading {}  more {}{}",
        snap.scope,
        snap.kind,
        snap.rows,
        snap.loading,
        snap.more,
        if snap.rows_only { "  rows held in memory" } else { "" }
    ))
    .font(FontId::monospace(12.0)));
    if let Some(error) = &snap.error {
        ui.add(Label::new(RichText::new(format!("last error: {error}")).color(crate::theme::error(ui))).truncate()).on_hover_text(error);
    }
    ui.label(RichText::new(format!("badges: {} indexed, {} disk checks started, {} in flight", snap.badges.0, snap.badges.1, snap.badges.2)).weak());
    if let Some(a) = &snap.active {
        ui.label(RichText::new(format!("load #{} {}  {} rows in {} batches  {}{}", a.id, a.scope, a.rows, a.batches, millis(a.elapsed), if a.cancelled { "  cancelling" } else { "" })).font(FontId::monospace(12.0)));
    }
    ui.separator();
    ui.label(RichText::new(format!("History ({} entries, showing #{})", snap.history.len(), snap.position + 1)).strong());
    ScrollArea::vertical().id_salt("scope_history").max_height(130.0).auto_shrink([false, true]).show(ui, |ui| {
        for (i, line) in snap.history.iter().enumerate() {
            let mark = if i == snap.position { ">" } else { " " };
            let anchor = line.anchor.as_deref().map_or(String::new(), |a| format!("  anchor {a}"));
            ui.label(RichText::new(format!("{mark} {i:>2}  {}  row {}{anchor}", line.scope, line.scroll_row)).font(FontId::monospace(12.0)));
        }
    });
    ui.separator();
    ui.label(RichText::new(format!("Loads of this tab, newest last ({})", snap.log.len())).strong());
    ScrollArea::vertical().id_salt("scope_log").auto_shrink([false, false]).show(ui, |ui| {
        for record in &snap.log {
            record_row(ui, record);
        }
    });
    ctx.data_mut(|d| d.insert_temp(state_id(), state));
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Scope").open(&mut open).default_width(620.0).default_height(520.0).show(ctx, |ui| body(ctx, ui));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::SmartView;
    use crate::command::bus::testing::{serial, take_matching};

    #[test]
    fn a_scope_string_is_parsed_into_its_form_kind_title_and_parent() {
        let p = parse_text("  tag:sunset ").unwrap();
        assert_eq!((p.formatted.as_str(), p.kind, p.title.as_str()), ("tag:sunset", "tag", "Tag: sunset"));
        assert_eq!(p.parent.as_deref(), Some("lib:all"));
        assert_eq!(p.problem, None);
        let q = parse_text("q:sem:red car").unwrap();
        assert_eq!(q.formatted, "q:sem:\"red car\"");
        assert_eq!(parse_text("smart:duplicates").unwrap().scope, Scope::Smart(SmartView::Duplicates));
        assert_eq!(parse_text("lib:all").unwrap().parent, None);
    }

    #[test]
    fn a_folder_that_does_not_exist_is_flagged_as_a_problem() {
        let p = parse_text("fs:/definitely/not/here").unwrap();
        assert!(p.problem.is_some());
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(parse_text(&format!("fs:{}", dir.path().display())).unwrap().problem, None);
    }

    #[test]
    fn bad_text_comes_back_as_the_parse_error() {
        for bad in ["", "plain", "wat:1", "smart:nope", "scan:x"] {
            assert!(parse_text(bad).is_err(), "{bad:?}");
        }
        assert!(parse_text("wat:1").unwrap_err().contains("unknown prefix"));
    }

    #[test]
    fn opening_queues_the_open_scope_command_with_the_target() {
        let _serial = serial();
        take_matching(|c| matches!(c, AppCommand::OpenScope { .. }));
        assert!(open_text("col:Trip", Target::Current).0);
        assert!(open_text("lib:all", Target::NewTab).0);
        assert!(!open_text("nonsense", Target::NewTab).0);
        let found = take_matching(|c| matches!(c, AppCommand::OpenScope { .. }));
        assert_eq!(found.len(), 2);
        assert!(matches!(&found[0], AppCommand::OpenScope { scope: Scope::Collection(n), target: Target::Current, .. } if n == "Trip"));
        assert!(matches!(&found[1], AppCommand::OpenScope { scope: Scope::Library, target: Target::NewTab, .. }));
    }

    #[tokio::test]
    async fn the_snapshot_follows_the_tab_and_the_window_draws_it() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let mut ex = FileExplorer::new(true);
        ex.start_at(Scope::Folder(dir.path().display().to_string()), false).unwrap();
        ex.open_scope(Scope::Folder(sub.display().to_string())).unwrap();
        ex.nav.push_log(LoadRecord { scope: "lib:all".into(), kind: "library", rows: 1203, batches: 3, first_batch: Some(Duration::from_millis(12)), elapsed: Duration::from_millis(480), error: None, cancelled: false, total: Some(1203) });
        let snap = snapshot_of(&ex);
        assert_eq!((snap.kind, snap.position, snap.history.len()), ("folder", 1, 2));
        assert_eq!(snap.log.len(), 1);

        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        publish(&ctx, &ex);
        let mut texts = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(Default::default(), |ui| show(ui.ctx()));
            texts = crate::ui::test_support::texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
            out.drop_without_applying_deltas();
        }
        let all = texts.join("\n");
        for needle in ["Open in new tab", "History (2 entries", "lib:all", "1203 rows of 1203", "first 12 ms"] {
            assert!(all.contains(needle), "missing {needle:?} in\n{all}");
        }
    }

    #[test]
    fn nothing_is_published_while_the_window_is_closed() {
        let ctx = crate::ui::test_support::test_ctx();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();
        let ex = FileExplorer::new(true);
        publish(&ctx, &ex);
        assert!(ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id())).is_none());
        set_open(&ctx, true);
        publish(&ctx, &ex);
        assert!(ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id())).is_some());
    }
}
