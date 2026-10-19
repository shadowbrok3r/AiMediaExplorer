//! Restored tabs: a scope that is read when its tab is first shown, and the placeholder of one that cannot be read now.

use std::time::{Duration, Instant};

use eframe::egui::{Context, RichText, Ui};

use super::history::History;
use super::nav::check_openable;
use super::Restore;
use crate::command::Scope;
use crate::icons::{self, svg};
use crate::settings::workspace::{BrowserState, MAX_SELECTION, QueryState};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::table::ExplorerMode;

/// Longest wait for the library before a restored scope that needs it shows its placeholder.
pub const LIBRARY_WAIT: Duration = Duration::from_secs(20);

/// Where a restored scope stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PendingKind {
    /// Read when the tab is first shown.
    Lazy,
    /// Needs the library, which has not connected yet.
    Waiting,
    /// Cannot be read now; the reason is shown with a Retry.
    Offline(String),
    /// A search; it is not run again until the user asks.
    Search,
}

/// A restored scope that is not loaded.
#[derive(Clone, Debug)]
pub struct Pending {
    pub kind: PendingKind,
    /// When the wait for the library began.
    pub since: Instant,
    /// Paths to select once the rows are in.
    pub select: Vec<String>,
}

/// What a restored scope does when its tab is shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Load,
    Wait,
    Offline(String),
    Search,
}

/// True for scopes that read the library.
pub fn needs_library(scope: &Scope) -> bool {
    !matches!(scope, Scope::Folder(_) | Scope::Archive(_))
}

/// True for searches, which need an engine and are not run by themselves.
pub fn is_search(scope: &Scope) -> bool {
    matches!(scope, Scope::Semantic(_) | Scope::Similar(_))
}

/// What to do with `scope` after `waited` for the library, which is `library_ready` or not.
pub fn verdict(scope: &Scope, waited: Duration, library_ready: bool) -> Verdict {
    if is_search(scope) {
        return Verdict::Search;
    }
    if let Err(reason) = check_openable(scope) {
        return Verdict::Offline(reason);
    }
    match (needs_library(scope) && !library_ready, waited < LIBRARY_WAIT) {
        (false, _) => Verdict::Load,
        (true, true) => Verdict::Wait,
        (true, false) => Verdict::Offline("The library is not connected".to_string()),
    }
}

impl FileExplorer {
    /// Brings a saved tab back without reading it: the view settings are applied and the scope is read when the tab is first shown.
    pub fn restore_state(&mut self, state: &BrowserState) {
        let Some(scope) = state.parsed_scope() else { return };
        self.stop_work();
        self.reset_rows();
        self.nav.history = History::new(scope.clone());
        self.nav.input = scope.format();
        self.nav.title = None;
        self.nav.error = None;
        self.nav.rows_only = false;
        self.nav.more = None;
        self.nav.restore = None;
        self.nav.restore_select.clear();
        if let Scope::Folder(path) | Scope::Archive(path) = &scope {
            self.current_path = path.clone();
        }
        self.viewer.mode = if scope.is_filesystem() { ExplorerMode::FileSystem } else { ExplorerMode::Database };
        self.viewer.showing_similarity = matches!(scope, Scope::Semantic(_) | Scope::Similar(_));
        self.nav.scope = scope;
        self.browse.mode = state.view.mode;
        self.browse.tile = state.view.tile;
        self.browse.fit = state.view.fit;
        self.browse.captions = state.view.captions;
        self.browse.badges = state.view.badges;
        self.view.set_sort(state.sort.clone());
        self.row_sync.restored_sort = Some(state.sort.clone());
        self.toolbar.query.mode = state.query.mode;
        self.toolbar.query.text = state.query.text.clone();
        if let Some(spec) = &state.filter {
            self.set_filter_spec(spec.clone());
        }
        self.defer_load(state.selection.clone());
    }

    /// Leaves the scope of the tab unread until the tab is first shown.
    pub fn defer_load(&mut self, select: Vec<String>) {
        self.nav.pending = Some(Pending { kind: PendingKind::Lazy, since: Instant::now(), select });
    }

    /// The saved form of this tab under `id`; `None` for a tab that shows rows held in memory, which no scope can load again.
    pub fn workspace_state(&self, id: u64) -> Option<BrowserState> {
        if self.nav.rows_only {
            return None;
        }
        let selection: Vec<String> = match &self.nav.pending {
            Some(pending) => pending.select.clone(),
            None if self.selection.len() <= MAX_SELECTION => self.selection.keys().iter().map(|k| k.as_str().to_string()).collect(),
            None => Vec::new(),
        };
        let spec = self.filter_spec();
        let filter = (spec != *self.viewer.filters.base()).then_some(spec);
        Some(BrowserState {
            id,
            scope: self.nav.scope.format(),
            view: self.browse.prefs(),
            sort: self.view.sort().clone(),
            query: QueryState { mode: self.toolbar.query.mode, text: self.toolbar.query.text.clone() },
            filter,
            selection,
        })
    }

    /// Reads a restored scope when the tab is drawn, or when the library it waited for has connected.
    pub(crate) fn run_pending(&mut self, ctx: &Context) {
        let Some(pending) = &self.nav.pending else { return };
        if !matches!(pending.kind, PendingKind::Lazy | PendingKind::Waiting) {
            return;
        }
        let waited = pending.since.elapsed();
        match verdict(&self.nav.scope, waited, crate::database::is_ready()) {
            Verdict::Load => self.start_pending(),
            Verdict::Wait => {
                self.set_pending_kind(PendingKind::Waiting);
                ctx.request_repaint_after(Duration::from_millis(250));
            }
            Verdict::Offline(reason) => self.set_pending_kind(PendingKind::Offline(reason)),
            Verdict::Search => self.set_pending_kind(PendingKind::Search),
        }
    }

    fn set_pending_kind(&mut self, kind: PendingKind) {
        if let Some(pending) = self.nav.pending.as_mut() {
            pending.kind = kind;
        }
    }

    /// Reads the scope of the tab now, with the selection it was saved with.
    fn start_pending(&mut self) {
        let Some(pending) = self.nav.pending.take() else { return };
        let scope = self.nav.scope.clone();
        let restore = pending.select.split_first().map(|(first, _)| Restore { scroll_row: None, anchor: Some(first.clone()) });
        let rest: Vec<String> = pending.select.iter().skip(1).cloned().collect();
        self.show_scope(scope, false, restore);
        self.nav.restore_select = rest;
    }

    /// The Retry of the placeholder: reads the scope when it can be read now, else says why it cannot.
    pub fn retry_pending(&mut self) {
        let Some(pending) = self.nav.pending.as_ref() else { return };
        let force = matches!(pending.kind, PendingKind::Search);
        match verdict(&self.nav.scope, Duration::ZERO, crate::database::is_ready()) {
            Verdict::Load => self.start_pending(),
            Verdict::Search if force => self.start_pending(),
            Verdict::Search => self.set_pending_kind(PendingKind::Search),
            Verdict::Wait => {
                if let Some(pending) = self.nav.pending.as_mut() {
                    pending.kind = PendingKind::Waiting;
                    pending.since = Instant::now();
                }
            }
            Verdict::Offline(reason) => self.set_pending_kind(PendingKind::Offline(reason)),
        }
    }

    /// Draws the placeholder of a tab whose scope is not read; false when there is none.
    pub(crate) fn pending_panel(&mut self, ui: &mut Ui) -> bool {
        let Some(pending) = self.nav.pending.clone() else { return false };
        let scope = self.nav.scope.format();
        let (title, detail, retry) = match &pending.kind {
            PendingKind::Lazy | PendingKind::Waiting => ("Waiting for the library", format!("{scope}  The files appear when the library has connected."), None),
            PendingKind::Offline(reason) => ("Not available right now", reason.clone(), Some("Retry")),
            PendingKind::Search => ("Search not run", format!("{scope}  Searches are not run again when the application starts."), Some("Run search")),
        };
        let top = (ui.available_height() / 3.0).max(16.0);
        ui.vertical_centered(|ui| {
            ui.add_space(top);
            ui.label(RichText::new(title).strong().size(16.0));
            ui.add_space(4.0);
            ui.weak(detail);
            if let Some(label) = retry {
                ui.add_space(8.0);
                if icons::button(ui, svg::ARROW_CLOCKWISE, label).clicked() {
                    self.retry_pending();
                }
            }
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::SmartView;

    #[test]
    fn a_folder_loads_at_once_whether_or_not_the_library_is_up() {
        let dir = tempfile::tempdir().unwrap();
        let folder = Scope::Folder(dir.path().display().to_string());
        assert_eq!(verdict(&folder, Duration::ZERO, false), Verdict::Load);
        assert_eq!(verdict(&folder, Duration::ZERO, true), Verdict::Load);
    }

    #[test]
    fn a_folder_that_is_gone_is_offline_with_the_reason() {
        match verdict(&Scope::Folder("/definitely/not/here".into()), Duration::ZERO, true) {
            Verdict::Offline(reason) => assert!(reason.contains("/definitely/not/here"), "{reason}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn library_scopes_wait_for_the_library_and_then_give_up() {
        for scope in [Scope::Library, Scope::Tag("sun".into()), Scope::Smart(SmartView::Untagged), Scope::Recent] {
            assert_eq!(verdict(&scope, Duration::from_secs(1), false), Verdict::Wait, "{scope}");
            assert_eq!(verdict(&scope, LIBRARY_WAIT, false), Verdict::Offline("The library is not connected".into()), "{scope}");
            assert_eq!(verdict(&scope, LIBRARY_WAIT * 5, true), Verdict::Load, "a library that came up late still loads");
        }
    }

    #[test]
    fn searches_wait_for_the_user() {
        assert_eq!(verdict(&Scope::Semantic("a cat".into()), Duration::ZERO, true), Verdict::Search);
        assert_eq!(verdict(&Scope::Similar("/a.jpg".into()), Duration::ZERO, false), Verdict::Search);
        assert!(is_search(&Scope::Semantic("x".into())) && !is_search(&Scope::Library));
        assert!(needs_library(&Scope::Library) && !needs_library(&Scope::Folder("/".into())));
    }
}
