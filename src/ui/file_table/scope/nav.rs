//! Navigation of a tab: opening scopes, going back and forward, loading rows and applying the batches that arrive.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use eframe::egui::{Context, RichText, Ui};

use super::history::History;
use super::{ActiveLoad, LoadBody, LoadMsg, LoadRecord, More, Request, Restore, RowBatch, runner};
use crate::command::Scope;
use crate::ui::file_table::table::ExplorerMode;
use crate::ui::file_table::{FileExplorer, SimilarResult};

/// Longest time one frame spends applying loaded rows.
const FRAME_BUDGET: Duration = Duration::from_millis(6);
/// Most rows one frame applies.
const FRAME_ROWS: usize = 4_000;
/// Most rows an engine hydration covers after a load.
const HYDRATE_MAX: usize = 5_000;
/// Longest tab title before it is cut.
const TITLE_MAX: usize = 36;

/// Why `scope` cannot be shown, if it cannot.
pub fn check_openable(scope: &Scope) -> Result<(), String> {
    match scope {
        Scope::Folder(path) => {
            let normalized = crate::utilities::windows::normalize_wsl_unc(path);
            if Path::new(&normalized).is_dir() { Ok(()) } else { Err(format!("{path} is not a folder that exists")) }
        }
        Scope::Archive(path) => {
            let (_, archive, _) = crate::command::scope::split_virtual(path).ok_or_else(|| format!("{path} is not an archive path"))?;
            if Path::new(&archive).is_file() { Ok(()) } else { Err(format!("{archive} is not a file that exists")) }
        }
        _ => Ok(()),
    }
}

/// The folder or archive scope a filesystem path names.
pub fn scope_for_path(path: &str) -> Scope {
    if crate::command::scope::split_virtual(path).is_some() { Scope::Archive(path.to_string()) } else { Scope::Folder(path.to_string()) }
}

fn short_title(title: &str) -> String {
    if title.chars().count() <= TITLE_MAX { title.to_string() } else { title.chars().take(TITLE_MAX - 3).collect::<String>() + "..." }
}

impl FileExplorer {
    /// What the tab shows.
    pub fn scope(&self) -> &Scope {
        &self.nav.scope
    }

    /// Title of the tab: the one it was opened with until it navigates, then the scope's.
    pub fn tab_title(&self) -> String {
        short_title(&self.nav.title.clone().unwrap_or_else(|| self.nav.scope.default_title()))
    }

    pub fn can_go_back(&self) -> bool {
        self.nav.history.can_back()
    }

    pub fn can_go_forward(&self) -> bool {
        self.nav.history.can_forward()
    }

    pub fn can_go_up(&self) -> bool {
        self.nav.scope.parent().is_some()
    }

    /// True while rows of the current scope are still being read.
    pub fn is_loading(&self) -> bool {
        self.nav.is_loading() || self.current_scan_id.is_some()
    }

    /// Id of the scan feeding the tab, if one is running or was the last.
    pub fn scan_id(&self) -> Option<u64> {
        self.owning_scan_id
    }

    /// Encrypted archives waiting for a password.
    pub fn archive_prompt_count(&self) -> usize {
        self.pending_zip_passwords.len()
    }

    /// Shows `scope` in this tab and remembers where it was; a scope that cannot be shown leaves the tab alone.
    pub fn open_scope(&mut self, scope: Scope) -> Result<(), String> {
        self.open_scope_recursive(scope, false)
    }

    /// [`open_scope`](Self::open_scope) with a recursive listing for folders.
    pub fn open_scope_recursive(&mut self, scope: Scope, recursive: bool) -> Result<(), String> {
        check_openable(&scope)?;
        self.remember_view();
        if let Scope::Folder(path) = &scope {
            if crate::settings::store().is_loaded() {
                let recent = path.clone();
                crate::settings::update("explorer:recent-path", |s| s.push_recent_path(recent));
            }
        }
        self.nav.history.push(scope.clone());
        self.show_scope(scope, recursive, None);
        Ok(())
    }

    /// Makes `scope` the first entry of a fresh tab.
    pub fn start_at(&mut self, scope: Scope, recursive: bool) -> Result<(), String> {
        check_openable(&scope)?;
        self.nav.history = History::new(scope.clone());
        self.show_scope(scope, recursive, None);
        Ok(())
    }

    /// Goes to the previous scope of the history, with the view it was left in.
    pub fn go_back(&mut self) -> bool {
        self.step_history(true)
    }

    pub fn go_forward(&mut self) -> bool {
        self.step_history(false)
    }

    fn step_history(&mut self, back: bool) -> bool {
        let allowed = if back { self.nav.history.can_back() } else { self.nav.history.can_forward() };
        if !allowed {
            return false;
        }
        self.remember_view();
        let entry = if back { self.nav.history.back().cloned() } else { self.nav.history.forward().cloned() };
        let Some(entry) = entry else { return false };
        if let Err(reason) = check_openable(&entry.scope) {
            if back {
                self.nav.history.forward();
            } else {
                self.nav.history.back();
            }
            self.nav.error = Some(reason);
            return false;
        }
        let restore = Restore { scroll_row: Some(entry.scroll_row), anchor: entry.anchor };
        self.show_scope(entry.scope, false, Some(restore));
        true
    }

    /// Goes to the parent of the scope: the folder around, or the library.
    pub fn go_up(&mut self) -> bool {
        match self.nav.scope.parent() {
            Some(parent) => self.open_scope(parent).is_ok(),
            None => false,
        }
    }

    /// Opens the home folder.
    pub fn go_home(&mut self) -> bool {
        let home = crate::settings::dirs::home_dir().to_string_lossy().to_string();
        self.open_scope(Scope::Folder(home)).is_ok()
    }

    /// Reads the current scope again, keeping the history.
    pub fn reload_scope(&mut self) {
        if self.nav.rows_only {
            return;
        }
        let scope = self.nav.scope.clone();
        let recursive = self.recursive_scan;
        let restore = Restore { scroll_row: self.first_visible_row(), anchor: self.anchor_path() };
        self.show_scope(scope, recursive, Some(restore));
    }

    /// Opens what the address box holds: a scope string or a path.
    pub fn submit_address(&mut self) -> Result<(), String> {
        let scope = super::parse_address(&self.nav.input)?;
        self.open_scope(scope)
    }

    /// Loads the next page of a search.
    pub fn load_more(&mut self) -> bool {
        let Some(more) = self.nav.more.clone() else { return false };
        if self.nav.is_loading() {
            return false;
        }
        self.nav.more = None;
        self.start_load(Request { scope: self.nav.scope.clone(), start: more.start, vector: more.vector }, true);
        true
    }

    /// Stops the load in flight.
    pub fn cancel_load(&mut self) {
        self.nav.cancel_active();
    }

    /// Makes the view show rows that are held in memory, such as a list of selected files.
    pub fn show_rows(&mut self, title: Option<String>, rows: Vec<crate::database::Thumbnail>) {
        self.stop_work();
        self.reset_rows();
        self.viewer.mode = ExplorerMode::Database;
        self.nav.rows_only = true;
        self.nav.error = None;
        self.nav.title = title;
        self.nav.scope = Scope::Library;
        self.nav.history = History::new(Scope::Library);
        self.nav.input.clear();
        self.apply_rows(RowBatch { rows, ..Default::default() });
    }

    /// Shows results a search produced elsewhere as the scope they belong to.
    pub fn show_similarity_results(&mut self, origin: &str, results: Vec<SimilarResult>) {
        if results.is_empty() {
            return;
        }
        let scope = match origin.strip_prefix("query:") {
            Some(text) => Scope::Semantic(text.to_string()),
            None => Scope::Similar(origin.to_string()),
        };
        let same = self.nav.scope == scope && self.viewer.showing_similarity;
        if !same {
            self.remember_view();
            self.nav.history.push(scope.clone());
            self.stop_work();
            self.reset_rows();
            self.viewer.mode = ExplorerMode::Database;
            self.viewer.showing_similarity = true;
            self.nav.rows_only = false;
            self.nav.error = None;
            self.nav.title = None;
            self.nav.input = scope.format();
            self.nav.scope = scope;
        }
        let scores = results.iter().filter_map(|r| r.clip_similarity_score.or(r.similarity_score).map(|s| (r.thumb.path.clone(), s))).collect();
        let rows = results.into_iter().map(|r| r.thumb).collect();
        self.apply_rows(RowBatch { rows, scores, ..Default::default() });
    }

    /// Records the scroll position and selection in the history entry being left.
    fn remember_view(&mut self) {
        let (row, anchor) = (self.first_visible_row().unwrap_or(0), self.anchor_path());
        self.nav.history.save_view(row, anchor);
    }

    fn first_visible_row(&self) -> Option<usize> {
        self.details.first_visible()
    }

    fn anchor_path(&self) -> Option<String> {
        self.selection.anchor().map(|k| k.as_str().to_string())
    }

    /// Shows `scope`: stops what the tab was doing, empties the view and starts the load or the scan.
    pub(super) fn show_scope(&mut self, scope: Scope, recursive: bool, restore: Option<Restore>) {
        self.stop_work();
        self.reset_rows();
        self.viewer.mode = if scope.is_filesystem() { ExplorerMode::FileSystem } else { ExplorerMode::Database };
        self.viewer.showing_similarity = matches!(scope, Scope::Semantic(_) | Scope::Similar(_));
        self.nav.rows_only = false;
        self.nav.error = None;
        self.nav.title = None;
        self.nav.input = scope.format();
        self.nav.restore = restore;
        self.nav.scope = scope.clone();
        self.nav.fs_started = None;
        match scope {
            Scope::Folder(path) => {
                self.current_path = path;
                if recursive {
                    self.start_recursive_scan();
                } else {
                    self.populate_current_directory();
                }
            }
            Scope::Archive(path) => {
                self.current_path = path;
                self.populate_current_directory();
            }
            other => self.start_load(Request::first(other), false),
        }
    }

    /// Stops the load and the scan of the tab; whatever they still deliver is ignored.
    pub(crate) fn stop_work(&mut self) {
        self.nav.pending = None;
        self.nav.cancel_active();
        if let Some(id) = self.current_scan_id.take() {
            crate::utilities::scan::cancel_scan(id);
        }
        self.owning_scan_id = None;
    }

    /// Empties the view and every cache that belongs to the rows in it.
    pub(super) fn reset_rows(&mut self) {
        self.table.clear();
        self.selection.clear();
        self.viewer.thumbs.new_scope(self.viewer.scope);
        self.viewer.similar_scores.clear();
        self.viewer.clip_presence.clear();
        self.viewer.clip_presence_hashes.clear();
        self.viewer.badges.clear();
        self.clip_presence.clear();
        self.pending_thumb_rows.clear();
        self.files.clear();
        self.last_scan_rows.clear();
        self.last_scan_paths.clear();
        self.last_scan_root = None;
        self.recursive_scan = false;
        self.recursive_filtered_rows.clear();
        self.recursive_filter_sig = None;
        self.recursive_current_page = 0;
        self.recursive_total_pages = 0;
        self.scan_done = false;
        self.file_scan_progress = 0.0;
        self.db_lookup.clear();
        self.nav.more = None;
        self.nav.hydrate.clear();
    }

    fn start_load(&mut self, req: Request, page: bool) {
        self.nav.loads_started += 1;
        let id = self.nav.loads_started;
        let cancel = Arc::new(AtomicBool::new(false));
        self.nav.active = Some(ActiveLoad { id, scope: req.scope.clone(), cancel: cancel.clone(), started: Instant::now(), rows: 0, batches: 0, finished: false, cancelled: false, page });
        runner::spawn(id, req, self.nav.tx.clone(), cancel);
    }

    /// Takes in what the loader sent since the last frame, within a time budget.
    pub fn receive_scope(&mut self, ctx: &Context) {
        if self.viewer.badges.poll() > 0 {
            ctx.request_repaint();
        }
        let started = Instant::now();
        let mut rows = 0;
        while let Ok(msg) = self.nav.rx.try_recv() {
            rows += self.apply_load_msg(msg);
            if rows >= FRAME_ROWS || started.elapsed() >= FRAME_BUDGET {
                if !self.nav.rx.is_empty() {
                    ctx.request_repaint();
                }
                break;
            }
        }
        self.apply_restore();
    }

    fn apply_load_msg(&mut self, msg: LoadMsg) -> usize {
        if let LoadBody::Embedded(paths) = msg.body {
            self.viewer.clip_presence.extend(paths);
            self.row_sync.clip_gen += 1;
            return 0;
        }
        let Some(active) = self.nav.active.as_ref().filter(|a| a.id == msg.load) else { return 0 };
        let cancelled = active.cancelled;
        match msg.body {
            LoadBody::Rows(batch) if !cancelled => self.apply_rows(batch),
            LoadBody::Done(outcome) => {
                self.finish_load(outcome);
                0
            }
            LoadBody::Failed(error) => {
                self.fail_load(error);
                0
            }
            _ => 0,
        }
    }

    /// Adds rows to the view and everything that tracks them.
    pub(crate) fn apply_rows(&mut self, mut batch: RowBatch) -> usize {
        let count = batch.rows.len();
        crate::ui::file_table::thumbs::adopt_legacy(&self.viewer.thumbs, &mut batch.rows);
        self.viewer.thumbs.note_rows(batch.rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect());
        let unindexed: HashSet<&str> = batch.unindexed.iter().map(String::as_str).collect();
        self.viewer.badges.mark_indexed(batch.rows.iter().map(|r| r.path.as_str()).filter(|p| !unindexed.contains(p)));
        if !batch.embedded.is_empty() {
            self.viewer.clip_presence.extend(batch.embedded);
            self.row_sync.clip_gen += 1;
        }
        if !batch.scores.is_empty() {
            self.viewer.showing_similarity = true;
            self.viewer.similar_scores.extend(batch.scores);
        }
        let room = HYDRATE_MAX.saturating_sub(self.nav.hydrate.len());
        self.nav.hydrate.extend(batch.rows.iter().filter(|r| r.file_type != "<DIR>").take(room).map(|r| r.path.clone()));
        self.table.bulk_load(batch.rows);
        if let Some(active) = self.nav.active.as_mut() {
            active.rows += count;
            active.batches += 1;
        }
        count
    }

    fn finish_load(&mut self, outcome: super::Outcome) {
        let Some(active) = self.nav.active.as_mut() else { return };
        active.finished = true;
        let record = LoadRecord {
            scope: active.scope.format(),
            kind: active.scope.kind(),
            rows: outcome.rows,
            batches: outcome.batches,
            first_batch: outcome.first_batch,
            elapsed: outcome.elapsed,
            error: None,
            cancelled: outcome.cancelled || active.cancelled,
            total: outcome.total,
        };
        let cancelled = record.cancelled;
        self.nav.push_log(record);
        if cancelled {
            return;
        }
        if let Some(title) = outcome.title.filter(|_| self.nav.title.is_none()) {
            self.nav.title = Some(title);
        }
        self.nav.more = outcome.more.then(|| More { start: outcome.next_start, vector: outcome.vector });
        let paths = std::mem::take(&mut self.nav.hydrate);
        if !paths.is_empty() && paths.len() <= HYDRATE_MAX && !self.viewer.showing_similarity {
            tokio::spawn(async move {
                let _ = crate::ai::GLOBAL_AI_ENGINE.hydrate_directory_paths(&paths).await;
            });
        }
    }

    fn fail_load(&mut self, error: String) {
        let Some(active) = self.nav.active.as_mut() else { return };
        active.finished = true;
        let record = LoadRecord {
            scope: active.scope.format(),
            kind: active.scope.kind(),
            rows: active.rows,
            batches: active.batches,
            first_batch: None,
            elapsed: active.started.elapsed(),
            error: Some(error.clone()),
            cancelled: active.cancelled,
            total: None,
        };
        let cancelled = active.cancelled;
        self.nav.push_log(record);
        if !cancelled {
            log::warn!("[scope] {} failed: {error}", self.nav.scope);
            self.nav.error = Some(error);
        }
    }

    /// Brings back the scroll position and selection of a history entry once its rows are in.
    pub(crate) fn apply_restore(&mut self) {
        let Some(mut restore) = self.nav.restore.take() else { return };
        let loading = self.is_loading();
        self.sync_view();
        let have = self.view.len();
        if let Some(row) = restore.scroll_row {
            if have > row || !loading {
                if have > 0 {
                    self.details.scroll_to_top_row(row.min(have - 1));
                }
                restore.scroll_row = None;
            }
        }
        if let Some(anchor) = restore.anchor.clone() {
            if self.table.contains_path(&anchor) {
                let more = std::mem::take(&mut self.nav.restore_select);
                self.select_paths(std::iter::once(anchor.as_str()).chain(more.iter().map(String::as_str)));
                restore.anchor = None;
            } else if !loading {
                restore.anchor = None;
                self.nav.restore_select.clear();
            }
        }
        if restore.scroll_row.is_some() || restore.anchor.is_some() {
            self.nav.restore = Some(restore);
        }
    }

    /// Notes a finished filesystem listing in the load log.
    pub(crate) fn note_listing_done(&mut self) {
        let Some(started) = self.nav.fs_started.take() else { return };
        let rows = self.table.len();
        let record = LoadRecord {
            scope: self.nav.scope.format(),
            kind: self.nav.scope.kind(),
            rows,
            batches: 0,
            first_batch: None,
            elapsed: started.elapsed(),
            error: None,
            cancelled: false,
            total: Some(rows as u64),
        };
        self.nav.push_log(record);
    }

    /// Keeps the scope in step with `current_path` after code outside navigation changed the path.
    pub(crate) fn sync_fs_scope(&mut self) {
        self.nav.rows_only = false;
        self.nav.error = None;
        let scope = scope_for_path(&self.current_path);
        if self.nav.scope != scope {
            self.nav.scope = scope;
            self.nav.input = self.nav.scope.format();
        }
        self.viewer.mode = ExplorerMode::FileSystem;
        self.nav.fs_started = Some(Instant::now());
    }

    /// The footer under the rows: a failed load with a retry, and the button for the next page of a search.
    pub fn scope_footer(&mut self, ui: &mut Ui) {
        let has_error = self.nav.error.is_some();
        let has_more = self.nav.more.is_some();
        if !has_error && !has_more {
            return;
        }
        eframe::egui::Panel::bottom("scope_footer").exact_size(28.).show_separator_line(false).show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(ui.spacing().item_spacing.x);
                if let Some(error) = self.nav.error.clone() {
                    if ui.button("Retry").clicked() {
                        self.nav.error = None;
                        self.reload_scope();
                    }
                    let text = RichText::new(format!("Could not load: {}", super::one_line(&error))).color(crate::theme::error(ui));
                    ui.add(eframe::egui::Label::new(text).truncate()).on_hover_text(error);
                }
                if has_more {
                    ui.label(format!("{} results", self.table.len()));
                    let idle = !self.nav.is_loading();
                    if ui.add_enabled(idle, eframe::egui::Button::new("Load more")).clicked() {
                        self.load_more();
                    }
                }
            });
        });
    }

    /// Loads the next page of a search when the last row has scrolled into view.
    pub fn auto_load_more(&mut self) {
        if self.nav.more.is_none() || self.nav.is_loading() {
            return;
        }
        let last = self.view.len();
        if last > 0 && self.details.row_rects().iter().any(|(pos, _)| pos + 1 >= last) {
            self.load_more();
        }
    }
}
