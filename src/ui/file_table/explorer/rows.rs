//! Glue between the explorer and the row model: view refresh, selection effects, opening rows and the details panel.

use std::time::{Duration, Instant};

use egui::{Context, Ui};

use crate::Thumbnail;
use crate::ui::devtools::row_store::{self, Command, Snapshot};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::thumbnail::{DIR_TYPE, heap_bytes, synthetic};
use crate::ui::file_table::rows::{PathKey, Refresh, SelectionStats, Step, ViewInputs, ViewRef, ViewStats};
use crate::ui::file_table::table::columns::Col;
use crate::ui::file_table::table::details::{DetailsEvent, DetailsInputs};
use crate::ui::file_table::table::keys::KeyAction;
use crate::ui::file_table::table::{ExplorerMode, TabAction, cells};

/// How long a row keeps the focus before its on-demand work (archive thumbnail, CLIP check) starts.
const FOCUS_SETTLE: Duration = Duration::from_millis(150);

/// Bookkeeping between the selection, the view and the rest of the explorer.
#[derive(Default)]
pub struct RowSync {
    preview: Option<PathKey>,
    settle: Option<(PathKey, Instant)>,
    stats_key: Option<(u64, u64)>,
    selection_stats: SelectionStats,
    prefs_loaded: bool,
    /// Bumped whenever CLIP presence changes.
    pub clip_gen: u64,
    pub last_check: Option<Result<(), String>>,
    #[cfg(test)]
    pub opened: Vec<String>,
}

impl RowSync {
    /// The row whose on-demand work is waiting for the focus to settle.
    pub fn pending_focus(&self) -> Option<&PathKey> {
        self.settle.as_ref().map(|(key, _)| key)
    }
}

/// Cached numbers of the current view, for the status bar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplorerViewStats {
    /// Revision of the view the numbers belong to; changes whenever any of them may have.
    pub revision: u64,
    pub rows: ViewStats,
    pub selection_revision: u64,
    pub selection: SelectionStats,
}

/// The cached stats of `ex`'s view and selection as of its last [`FileExplorer::sync_view`]; reads no rows.
pub fn view_stats(ex: &FileExplorer) -> ExplorerViewStats {
    ExplorerViewStats { revision: ex.view.revision(), rows: *ex.view.stats(), selection_revision: ex.selection.revision(), selection: ex.row_sync.selection_stats }
}

impl FileExplorer {
    pub fn view_ref(&self) -> ViewRef<'_, Thumbnail> {
        ViewRef::new(&self.table, &self.view)
    }

    /// Selected rows in view order; empty when nothing is selected.
    pub fn selected_rows(&self) -> Vec<&Thumbnail> {
        self.selection.rows(&self.view_ref())
    }

    /// Brings the view up to date with the rows, the filters and the sort, then settles the selection.
    pub fn sync_view(&mut self) -> Refresh {
        let Self { table, view, selection, viewer, details, row_sync, .. } = self;
        let available = Col::available(viewer.mode, viewer.showing_similarity);
        if view.sort().keys().iter().any(|k| !available.iter().any(|c| c.id() == k.column)) {
            let mut spec = view.sort().clone();
            spec.retain_columns(|id| available.iter().any(|c| c.id() == id));
            view.set_sort(spec);
        }
        let mut data_sig = 0u64;
        if view.sort().uses(Col::Clip.id()) {
            data_sig = data_sig.wrapping_mul(31).wrapping_add(row_sync.clip_gen + 1);
        }
        if view.sort().uses(Col::Similarity.id()) {
            data_sig = data_sig.wrapping_mul(31).wrapping_add(viewer.similar_scores.len() as u64 + 1);
        }
        let filter = viewer.row_filter();
        let filter_sig = viewer.filter_signature();
        let viewer_ref: &crate::ui::file_table::table::FileTableViewer = viewer;
        let inputs = ViewInputs { filter_sig, data_sig, filter: &|r| filter.matches(r), sort_key: &|r, c| cells::sort_value(viewer_ref, r, c) };
        let outcome = view.refresh(table, &inputs);
        if outcome == Refresh::Rebuilt && !selection.is_empty() {
            selection.prune(&ViewRef::new(table, view));
        }
        if table.is_empty() {
            details.scroll_to_top();
        }
        self.settle_selection();
        outcome
    }

    /// Follows the preview target and refreshes the selection counts.
    fn settle_selection(&mut self) {
        let revision = self.selection.revision();
        let target = self.selection.preview_target().cloned();
        if target != self.row_sync.preview {
            if let Some(row) = target.as_ref().and_then(|k| self.table.get_by_key(k)) {
                if row.file_type != DIR_TYPE && row.path != self.current_thumb.path {
                    self.current_thumb = row.clone();
                    self.row_sync.settle = target.clone().map(|k| (k, Instant::now()));
                }
            }
            self.row_sync.preview = target;
        }
        let key = (self.view.revision(), revision);
        if self.row_sync.stats_key != Some(key) {
            self.row_sync.selection_stats = self.selection.stats(&self.table);
            self.row_sync.stats_key = Some(key);
        }
    }

    /// Starts the on-demand work of a row that kept the focus for a moment.
    fn fire_settled_focus(&mut self, ctx: &Context) {
        let Some((key, since)) = self.row_sync.settle.clone() else { return };
        let waited = since.elapsed();
        if waited < FOCUS_SETTLE {
            ctx.request_repaint_after(FOCUS_SETTLE - waited);
            return;
        }
        self.row_sync.settle = None;
        if let Some(row) = self.table.get_by_key(&key) {
            self.viewer.on_row_focused(row);
        }
    }

    /// Selects exactly the visible rows among `paths`.
    pub fn select_paths<'a>(&mut self, paths: impl IntoIterator<Item = &'a str>) {
        let keys: Vec<PathKey> = paths.into_iter().map(|p| self.table.key_for(p)).collect();
        let view = ViewRef::new(&self.table, &self.view);
        self.selection.select_keys(&view, keys);
        self.settle_selection();
    }

    pub fn clear_selection(&mut self) {
        self.selection.clear();
        self.settle_selection();
    }

    /// Opens a row the way a double click does: folders navigate, archives are browsed, files open externally.
    pub fn open_row(&mut self, key: &PathKey) {
        let Some(row) = self.table.get_by_key(key).cloned() else { return };
        if row.file_type == DIR_TYPE {
            if let Some(rest) = row.path.strip_prefix("cat://") {
                self.viewer.requested_tabs.push(TabAction::OpenCategory(rest.trim_end_matches('/').to_string()));
            } else if let Some(rest) = row.path.strip_prefix("tag://") {
                self.viewer.requested_tabs.push(TabAction::OpenTag(rest.trim_end_matches('/').to_string()));
            } else if let Err(reason) = self.open_scope(super::super::scope::nav::scope_for_path(&row.path)) {
                crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
            }
            return;
        }
        let is_zip = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("zip"));
        if row.file_type == "<ARCHIVE>" || is_zip {
            self.viewer.requested_tabs.push(TabAction::OpenArchive(row.path));
        } else if !row.path.contains("://") {
            self.open_external(&row.path);
        }
    }

    fn open_external(&mut self, path: &str) {
        #[cfg(test)]
        self.row_sync.opened.push(path.to_string());
        #[cfg(not(test))]
        if let Err(e) = open::that_detached(path) {
            log::warn!("Could not open {path}: {e}");
        }
    }

    /// Draws the details view and applies what the user did in it.
    pub fn details_panel(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        self.load_details_prefs_once(&ctx);
        self.run_row_commands(&ctx);
        self.sync_view();
        let events = {
            let input = DetailsInputs { store: &self.table, view: &self.view, selection: &self.selection, viewer: &self.viewer };
            self.details.show(ui, &input)
        };
        if !events.is_empty() {
            self.apply_details_events(&ctx, events);
        }
        self.fire_settled_focus(&ctx);
        self.publish_row_debug(&ctx);
    }

    fn load_details_prefs_once(&mut self, ctx: &Context) {
        if std::mem::replace(&mut self.row_sync.prefs_loaded, true) {
            return;
        }
        if let Some(sort) = self.details.load_prefs(ctx) {
            self.view.set_sort(sort);
        }
    }

    pub(crate) fn apply_details_events(&mut self, ctx: &Context, events: Vec<DetailsEvent>) {
        for event in events {
            match event {
                DetailsEvent::Click { key, ctrl, shift } => {
                    let view = ViewRef::new(&self.table, &self.view);
                    match (shift, ctrl) {
                        (true, additive) => {
                            self.selection.select_range(&view, &key, additive);
                        }
                        (false, true) => {
                            self.selection.toggle(key);
                        }
                        (false, false) => {
                            self.selection.click(key);
                        }
                    }
                    self.details.set_keyboard_active(true);
                }
                DetailsEvent::ContextOn { key } => {
                    if !self.selection.contains(&key) {
                        self.selection.click(key);
                    }
                }
                DetailsEvent::Open { key } => self.open_row(&key),
                DetailsEvent::Sort { column, additive } => {
                    let mut spec = self.view.sort().clone();
                    spec.click(column.id(), additive);
                    self.view.set_sort(spec);
                    self.details.save_prefs(ctx, self.view.sort());
                    self.details.set_keyboard_active(true);
                }
                DetailsEvent::ToggleColumn { column, hidden } => {
                    if self.details.prefs.set_hidden(column, hidden) {
                        self.details.save_prefs(ctx, self.view.sort());
                    }
                }
                DetailsEvent::Tab(action) => self.viewer.requested_tabs.push(action),
                DetailsEvent::Menu(action) => {
                    self.sync_view();
                    let view = ViewRef::new(&self.table, &self.view);
                    let rows = self.selection.rows(&view);
                    self.viewer.run_action(action, &rows);
                }
                DetailsEvent::Key(key) => self.apply_key(ctx, key),
            }
        }
        self.settle_selection();
        ctx.request_repaint();
    }

    fn apply_key(&mut self, ctx: &Context, key: KeyAction) {
        let view = ViewRef::new(&self.table, &self.view);
        match key {
            KeyAction::Move { step, extend } => {
                if let Some(focus) = self.selection.step_focus(&view, step, extend) {
                    if let Some(pos) = view.pos_of(&focus) {
                        self.details.scroll_to(pos);
                    }
                }
            }
            KeyAction::SelectAll => {
                self.selection.select_all(&view);
            }
            KeyAction::Open => {
                if let Some(focus) = self.selection.focus().cloned() {
                    self.open_row(&focus);
                }
            }
            KeyAction::TogglePreview => self.open_preview_pane = !self.open_preview_pane,
            KeyAction::CopyPath => {
                let mut paths: Vec<&str> = self.selection.rows(&view).iter().map(|r| r.path.as_str()).collect();
                if paths.is_empty() {
                    paths.extend(self.selection.focus().and_then(|k| self.table.get_by_key(k)).map(|r| r.path.as_str()));
                }
                if !paths.is_empty() {
                    ctx.copy_text(paths.join("\n"));
                }
            }
            KeyAction::ClearSelection => {
                self.selection.clear();
            }
        }
    }

    /// Moves the keyboard focus the way an arrow key does.
    pub fn step_focus(&mut self, step: Step, extend: bool) -> Option<PathKey> {
        self.sync_view();
        let view = ViewRef::new(&self.table, &self.view);
        let focus = self.selection.step_focus(&view, step, extend);
        self.settle_selection();
        focus
    }

    /* ---------------------------------------- developer tools ---------------------------------------- */

    /// Verifies the store, the view and the selection against each other.
    pub fn check_row_invariants(&self) -> Result<(), String> {
        self.table.check_invariants().map_err(|e| format!("store: {e}"))?;
        let filter = self.viewer.row_filter();
        let viewer = &self.viewer;
        let inputs = ViewInputs {
            filter_sig: self.viewer.filter_signature(),
            data_sig: self.current_data_sig(),
            filter: &|r| filter.matches(r),
            sort_key: &|r, c| cells::sort_value(viewer, r, c),
        };
        self.view.check_invariants(&self.table, &inputs).map_err(|e| format!("view: {e}"))?;
        if let Some(missing) = self.selection.keys().iter().find(|k| !self.table.contains_path(k.as_str())) {
            return Err(format!("selection: {missing} is not in the store"));
        }
        let view = self.view_ref();
        if let Some(hidden) = self.selection.keys().iter().find(|k| !view.is_visible(k)) {
            return Err(format!("selection: {hidden} is not visible"));
        }
        Ok(())
    }

    fn current_data_sig(&self) -> u64 {
        let mut data_sig = 0u64;
        if self.view.sort().uses(Col::Clip.id()) {
            data_sig = data_sig.wrapping_mul(31).wrapping_add(self.row_sync.clip_gen + 1);
        }
        if self.view.sort().uses(Col::Similarity.id()) {
            data_sig = data_sig.wrapping_mul(31).wrapping_add(self.viewer.similar_scores.len() as u64 + 1);
        }
        data_sig
    }

    /// Replaces the rows with `count` generated ones for profiling.
    pub fn fill_synthetic_rows(&mut self, count: usize) {
        self.table.replace_all(synthetic(count));
        self.selection.clear();
        self.sync_view();
    }

    fn run_row_commands(&mut self, ctx: &Context) {
        for command in row_store::take_commands(ctx, self.debug_id()) {
            match command {
                Command::CheckInvariants => {
                    self.sync_view();
                    self.row_sync.last_check = Some(self.check_row_invariants());
                }
                Command::Synthetic(count) => self.fill_synthetic_rows(count),
                Command::ClearRows => {
                    self.table.clear();
                    self.sync_view();
                }
                Command::ResetFrameStats => self.details.clear_frame_times(),
            }
        }
    }

    fn debug_id(&self) -> egui::Id {
        egui::Id::new(("ame_row_store_owner", self.viewer.scope))
    }

    fn publish_row_debug(&self, ctx: &Context) {
        if !row_store::is_open(ctx) {
            return;
        }
        let estimate = self.table.estimate_bytes(heap_bytes);
        let view_bytes = self.view.order().len() * 4 + self.table.len() * 4;
        let selection_bytes = self.selection.len() * (std::mem::size_of::<PathKey>() * 2 + 16);
        let key_text = |k: Option<&PathKey>| k.map(|k| k.as_str().to_string());
        row_store::publish(
            ctx,
            Snapshot {
                owner: self.debug_id(),
                label: self.current_path.clone(),
                mode: self.viewer.mode.to_str().to_string(),
                versions: self.table.versions(),
                store_len: self.table.len(),
                view_len: self.view.len(),
                view_revision: self.view.revision(),
                stats: *self.view.stats(),
                refresh_counts: self.view.counts(),
                last_refresh: self.view.last_refresh(),
                sort: self.view.sort().clone(),
                selection_len: self.selection.len(),
                selection_revision: self.selection.revision(),
                anchor: key_text(self.selection.anchor()),
                focus: key_text(self.selection.focus()),
                preview: key_text(self.selection.preview_target()),
                selection_stats: self.row_sync.selection_stats,
                memory: estimate,
                view_bytes,
                selection_bytes,
                frame_ms: self.details.frame_times().iter().copied().collect(),
                drawn_rows: self.details.drawn_rows(),
                page_rows: self.details.page_rows,
                last_check: self.row_sync.last_check.clone(),
            },
        );
    }
}
