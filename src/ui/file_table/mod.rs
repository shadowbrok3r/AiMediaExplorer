use crate::{ScanEnvelope, Thumbnail};
use crossbeam::channel::{Receiver, Sender};
use eframe::egui::*;
use egui::style::StyleModifier;
use serde::Serialize;
use table::FileTableViewer;

pub mod chips;
pub mod explorer;
pub mod filters;
pub mod load;
pub mod menus;
pub mod preview_pane;
pub mod quick_access_pane;
pub mod receive;
#[path = "../../rows/mod.rs"]
pub mod rows;
pub mod scope;
pub mod table;
#[path = "../../thumbs/mod.rs"]
pub mod thumbs;

pub use explorer::rows::{ExplorerViewStats, view_stats};

// Request to open a new filtered tab in the dock
#[derive(Clone, Debug)]
pub enum FilterRequest {
    NewTab {
        title: String,
        rows: Vec<crate::database::Thumbnail>,
        // Optional: mark this tab as a similarity-results view and attach scores
        showing_similarity: bool,
        similar_scores: Option<std::collections::HashMap<String, f32>>,
        // For similarity tabs, the origin that generated results (e.g., image path or "query:<text>")
        origin_path: Option<String>,
        // Open without focusing the new tab
        background: bool,
    },
    OpenPath {
        title: String,
        path: String,
        recursive: bool,
        // Open without focusing the new tab
        background: bool,
    },
}

// AI streaming update messages (interim + final)
#[derive(Debug, Clone)]
pub enum AIUpdate {
    Interim {
        path: String,
        text: String,
    },
    Final {
        path: String,
        description: String,
        caption: Option<String>,
        category: Option<String>,
        tags: Vec<String>,
    },
    SimilarResults {
        origin_path: String,
        results: Vec<SimilarResult>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimilarResult {
    pub thumb: crate::database::Thumbnail,
    pub created: Option<surrealdb::types::Datetime>,
    pub updated: Option<surrealdb::types::Datetime>,
    pub similarity_score: Option<f32>,
    pub clip_similarity_score: Option<f32>,
}

#[derive(Serialize)]
pub struct FileExplorer {
    #[serde(skip)]
    pub table: rows::RowStore<Thumbnail>,
    #[serde(skip)]
    pub view: rows::ViewIndex,
    #[serde(skip)]
    pub selection: rows::Selection,
    #[serde(skip)]
    pub details: table::details::DetailsView,
    #[serde(skip)]
    pub browse: table::browse::Browse,
    #[serde(skip)]
    row_sync: explorer::rows::RowSync,
    pub viewer: FileTableViewer,
    batch_size: usize,
    files: Vec<Thumbnail>,
    pub current_path: String,
    open_preview_pane: bool,
    open_quick_access: bool,
    pub file_scan_progress: f32,
    recursive_scan: bool,
    pub scan_done: bool,
    excluded_terms: Vec<String>,
    pub current_thumb: Thumbnail,
    #[serde(skip)]
    thumbnail_tx: Sender<Thumbnail>,
    #[serde(skip)]
    thumbnail_rx: Receiver<Thumbnail>,
    #[serde(skip)]
    scan_tx: Sender<ScanEnvelope>,
    #[serde(skip)]
    scan_rx: Receiver<ScanEnvelope>,
    #[serde(skip)]
    pending_thumb_rows: Vec<Thumbnail>,
    #[serde(skip)]
    ai_update_rx: Receiver<AIUpdate>,
    #[serde(skip)]
    pub streaming_interim: std::collections::HashMap<String, String>,
    // Auto-follow active vision generation updates
    #[serde(skip)]
    pub follow_active_vision: bool,
    // Cache of clip embedding presence per path to avoid frequent async queries in UI thread
    #[serde(skip)]
    clip_presence: std::collections::HashSet<String>,
    #[serde(skip)]
    clip_embedding_rx: Receiver<crate::ClipEmbeddingRow>,
    // Vision description generation tracking (bulk vision progress separate from indexing)
    #[serde(skip)]
    pub vision_started: usize,
    #[serde(skip)]
    pub vision_completed: usize,
    #[serde(skip)]
    pub vision_pending: usize, // scheduled but not yet final
    // Track current scan id for cancellation
    #[serde(skip)]
    current_scan_id: Option<u64>,
    // DB rows for current path (by path). Used to merge scan results and reuse existing IDs.
    #[serde(skip)]
    db_lookup: std::collections::HashMap<String, crate::database::Thumbnail>,
    // Channel to receive preloaded DB rows for current path
    #[serde(skip)]
    db_preload_rx: Receiver<Vec<crate::database::Thumbnail>>,
    #[serde(skip)]
    db_preload_tx: Sender<Vec<crate::database::Thumbnail>>,
    // Snapshot of the most recent recursive scan so we can restore its view without rescanning
    #[serde(skip)]
    last_scan_rows: Vec<Thumbnail>,
    #[serde(skip)]
    last_scan_paths: std::collections::HashSet<String>,
    #[serde(skip)]
    last_scan_root: Option<String>,
    pub active_filter_group: Option<String>,
    #[serde(skip)]
    filter_groups: Vec<crate::database::FilterGroup>,
    #[serde(skip)]
    filter_groups_tx: Sender<Vec<crate::database::FilterGroup>>,
    #[serde(skip)]
    filter_groups_rx: Receiver<Vec<crate::database::FilterGroup>>,
    #[serde(skip)]
    logical_groups: Vec<crate::database::LogicalGroup>,
    #[serde(skip)]
    pub logical_groups_tx: Sender<Vec<crate::database::LogicalGroup>>,
    #[serde(skip)]
    logical_groups_rx: Receiver<Vec<crate::database::LogicalGroup>>,
    // UI state: group management
    #[serde(skip)]
    group_create_name_input: String,
    #[serde(skip)]
    group_rename_target: Option<String>,
    #[serde(skip)]
    group_rename_input: String,
    // UI fields for group operations
    #[serde(skip)]
    group_add_target: String,
    #[serde(skip)]
    group_add_view_target: String,
    #[serde(skip)]
    group_copy_src: String,
    #[serde(skip)]
    group_copy_dst: String,
    #[serde(skip)]
    group_add_unassigned_target: String,
    // Encrypted archive support: prompt queue shown as egui modal after scan completes
    #[serde(skip)]
    pending_zip_passwords: std::collections::VecDeque<String>,
    #[serde(skip)]
    active_zip_prompt: Option<String>, // archive path currently asking password for
    #[serde(skip)]
    zip_password_input: String,
    #[serde(skip)]
    show_zip_modal: bool,
    // Track which scan_id owns this explorer so updates don't leak across tabs
    #[serde(skip)]
    owning_scan_id: Option<u64>,
    // Cached WSL data to avoid recomputing every frame
    #[serde(skip)]
    cached_wsl_distros: Option<Vec<String>>,
    #[serde(skip)]
    cached_physical_drives: Option<Vec<crate::utilities::explorer::PhysicalDrive>>,
    // Scan performance timings
    #[serde(skip)]
    perf_scan_started: Option<std::time::Instant>,
    #[serde(skip)]
    perf_last_batch_at: Option<std::time::Instant>,
    #[serde(skip)]
    perf_batches: Vec<(usize, std::time::Duration, std::time::Duration)>, // (batch_len, recv_gap, ui_processing)
    #[serde(skip)]
    perf_last_total: Option<std::time::Duration>,
    #[serde(skip)]
    perf_show_per_1k: bool,
    #[serde(skip)]
    pub recursive_page_size: usize,
    #[serde(skip)]
    pub recursive_current_page: usize,
    #[serde(skip)]
    pub recursive_total_pages: usize,
    #[serde(skip)]
    pub recursive_filtered_rows: Vec<Thumbnail>, // cached filtered+sorted snapshot
    #[serde(skip)]
    pub recursive_sort_key: Option<&'static str>, // simplistic sort key id
    #[serde(skip)]
    pub recursive_filter_sig: Option<String>, // hash/signature of last applied filter+type toggles
    // Collect failed thumbnail generations for a dedicated tab after scan completes
    #[serde(skip)]
    pub failed_thumbs: Vec<(String, String)>, // (path, error)
    #[serde(skip)]
    repaint_next: bool,
    // Where the tab is, where it came from and the loader feeding it
    #[serde(skip)]
    pub nav: scope::Nav,
}

impl FileExplorer {
    pub fn is_recursive_scan(&self) -> bool {
        self.recursive_scan
    }

    // New constructor that optionally skips initial directory scan/population
    pub fn new(skip_initial_scan: bool) -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
        let (ai_update_tx, ai_update_rx) = crossbeam::channel::unbounded();
        let (clip_embedding_tx, clip_embedding_rx) = crossbeam::channel::unbounded();
        let (db_preload_tx, db_preload_rx) = crossbeam::channel::unbounded();
        let (logical_groups_tx, logical_groups_rx) = crossbeam::channel::unbounded();
        let (filter_groups_tx, filter_groups_rx) = crossbeam::channel::unbounded();
        let current_path = crate::settings::dirs::start_dir().to_string_lossy().to_string();

        let mut this = Self {
            table: Default::default(),
            view: Default::default(),
            selection: Default::default(),
            details: Default::default(),
            browse: Default::default(),
            row_sync: Default::default(),
            viewer: FileTableViewer::new(
                thumbnail_tx.clone(),
                ai_update_tx,
                clip_embedding_tx.clone(),
            ),
            files: Default::default(),
            open_preview_pane: false,
            open_quick_access: false,
            current_path: current_path.clone(),
            file_scan_progress: 0.0,
            recursive_scan: false,
            scan_done: false,
            excluded_terms: Vec::new(),
            current_thumb: Thumbnail::new(""),
            thumbnail_tx,
            thumbnail_rx,
            scan_tx,
            scan_rx,
            pending_thumb_rows: Vec::new(),
            ai_update_rx,
            streaming_interim: std::collections::HashMap::new(),
            follow_active_vision: false,
            clip_presence: std::collections::HashSet::new(),
            clip_embedding_rx,
            vision_started: 0,
            vision_completed: 0,
            vision_pending: 0,
            current_scan_id: None,
            db_lookup: std::collections::HashMap::new(),
            db_preload_rx,
            db_preload_tx,
            last_scan_rows: Vec::new(),
            last_scan_paths: std::collections::HashSet::new(),
            last_scan_root: None,
            active_filter_group: None,
            filter_groups: Vec::new(),
            filter_groups_tx,
            filter_groups_rx,
            logical_groups: Vec::new(),
            logical_groups_tx,
            logical_groups_rx,
            group_create_name_input: String::new(),
            group_rename_target: None,
            group_rename_input: String::new(),
            group_add_target: String::new(),
            group_add_view_target: String::new(),
            group_copy_src: String::new(),
            group_copy_dst: String::new(),
            group_add_unassigned_target: String::new(),
            pending_zip_passwords: std::collections::VecDeque::new(),
            active_zip_prompt: None,
            zip_password_input: String::new(),
            show_zip_modal: false,
            owning_scan_id: None,
            cached_wsl_distros: Some(crate::utilities::explorer::list_wsl_distros()),
            cached_physical_drives: Some(crate::utilities::explorer::list_physical_drives()),
            batch_size: 128,
            perf_scan_started: None,
            perf_last_batch_at: None,
            perf_batches: Vec::new(),
            perf_last_total: None,
            perf_show_per_1k: false,
            recursive_page_size: 5_000,
            recursive_current_page: 0,
            recursive_total_pages: 0,
            recursive_filtered_rows: Vec::new(),
            recursive_sort_key: None,
            recursive_filter_sig: None,
            failed_thumbs: Vec::new(),
            repaint_next: false,
            nav: scope::Nav::new(crate::command::Scope::Folder(current_path.clone())),
        };
        // Preload saved filter groups
        {
            let tx = this.filter_groups_tx.clone();
            tokio::spawn(async move {
                match crate::database::list_filter_groups().await {
                    Ok(groups) => {
                        let _ = tx.try_send(groups);
                    }
                    Err(e) => log::warn!("No filter groups yet or failed to load: {e:?}"),
                }
            });
        }
        if !skip_initial_scan {
            this.populate_current_directory();
        }
        this
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        let started = std::time::Instant::now();
        self.viewer.sync_settings();
        self.receive(ui.ctx());
        self.viewer.thumbs.poll(ui.ctx());
        if self.repaint_next {
            ui.ctx().request_repaint();
            self.repaint_next = false;
        }
        self.preview_pane(ui);
        self.quick_access_pane(ui);
        self.navbar(ui);
        self.breadcrumbs(ui);
        self.scope_footer(ui);

        // For recursive scan pagination: ensure filtered snapshot stays in sync with current user filter
        if self.recursive_scan {
            // Force refresh of filtered snapshot if signature changed (apply_recursive will short-circuit if unchanged)
            self.apply_recursive_filter_and_sort();
        }

        let style = StyleModifier::default();
        style.apply(ui.style_mut());

        CentralPanel::default().show(ui, |ui| {
            // Recursive scan pagination controls (only when snapshot large enough)
            if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                self.update_recursive_total_pages();
                ui.horizontal(|ui| {
                    let total = self.recursive_total_pages.max(1);
                    let first_enabled = self.recursive_current_page > 0;
                    if ui
                        .add_enabled(first_enabled, egui::Button::new("First"))
                        .clicked()
                    {
                        log::info!("Pagination: First clicked");
                        self.recursive_current_page = 0;
                        self.rebuild_recursive_page();
                    }
                    let prev_enabled = self.recursive_current_page > 0;
                    if ui
                        .add_enabled(prev_enabled, egui::Button::new("Prev"))
                        .clicked()
                    {
                        log::info!(
                            "Pagination: Prev clicked (from {})",
                            self.recursive_current_page
                        );
                        if self.recursive_current_page > 0 {
                            self.recursive_current_page -= 1;
                            self.rebuild_recursive_page();
                        }
                    }
                    ui.label(format!(
                        "Page {} / {}",
                        self.recursive_current_page + 1,
                        total
                    ));
                    let next_enabled = self.recursive_current_page + 1 < total;
                    if ui
                        .add_enabled(next_enabled, egui::Button::new("Next"))
                        .clicked()
                    {
                        log::info!(
                            "Pagination: Next clicked (from {})",
                            self.recursive_current_page
                        );
                        if self.recursive_current_page + 1 < total {
                            self.recursive_current_page += 1;
                            self.rebuild_recursive_page();
                        }
                    }
                    let last_enabled = self.recursive_current_page + 1 < total;
                    if ui
                        .add_enabled(last_enabled, egui::Button::new("Last"))
                        .clicked()
                    {
                        log::info!("Pagination: Last clicked (target {}-1)", total);
                        if total > 0 {
                            self.recursive_current_page = total - 1;
                            self.rebuild_recursive_page();
                        }
                    }
                    ui.separator();
                    let filtered_total = self.recursive_filtered_rows.len();
                    let unfiltered_total = self.last_scan_rows.len();
                    if filtered_total == unfiltered_total {
                        ui.label(format!(
                            "Showing {} of {} (page size {})",
                            self.table.len(),
                            unfiltered_total,
                            self.recursive_page_size
                        ));
                    } else {
                        ui.label(format!(
                            "Showing {} of {} filtered ({} total, page size {})",
                            self.table.len(),
                            filtered_total,
                            unfiltered_total,
                            self.recursive_page_size
                        ));
                    }
                    // Display the current row range (1-based) for clarity
                    if self.recursive_page_size > 0 && !self.recursive_filtered_rows.is_empty() {
                        let start_idx = self.recursive_current_page * self.recursive_page_size;
                        let end_idx = (start_idx + self.recursive_page_size)
                            .min(self.recursive_filtered_rows.len());
                        ui.separator();
                        ui.label(format!("Rows {}-{}", start_idx + 1, end_idx));
                    }
                });
                ui.separator();
            }
            chips::show(ui, self);
            self.browse_panel(ui);

            // Modal password prompt (appears at end of scans or when browsing encrypted archives)
            if self.show_zip_modal {
                egui::Window::new("Archive Password Required")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .show(ui.ctx(), |ui| {
                        let current = self.active_zip_prompt.clone().unwrap_or_default();
                        ui.label(
                            "One or more encrypted archives were found. Please enter the password.",
                        );
                        ui.separator();
                        ui.label(format!("Archive: {}", current));
                        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let resp = egui::TextEdit::singleline(&mut self.zip_password_input)
                            .password(true)
                            .hint_text("Password")
                            .desired_width(240.)
                            .ui(ui);
                        ui.horizontal(|ui| {
                            if (resp.lost_focus() && enter_pressed) || ui.button("Submit").clicked()
                            {
                                // Store in-memory for this session and refresh any open view for that archive
                                let pw = std::mem::take(&mut self.zip_password_input);
                                if !current.is_empty() {
                                    self.viewer.archive_passwords.insert(current.clone(), pw);
                                }
                                // Refresh current view (filesystem or virtual) so thumbnail tasks are scheduled with the new password
                                self.populate_current_directory();
                                // Advance the queue
                                if let Some(_) = self.pending_zip_passwords.pop_front() {}
                                self.active_zip_prompt =
                                    self.pending_zip_passwords.front().cloned();
                                if self.active_zip_prompt.is_none() {
                                    self.show_zip_modal = false;
                                }
                            }
                            if ui
                                .button("Skip")
                                .on_hover_text("Skip this archive")
                                .clicked()
                            {
                                self.zip_password_input.clear();
                                if let Some(_) = self.pending_zip_passwords.pop_front() {}
                                self.active_zip_prompt =
                                    self.pending_zip_passwords.front().cloned();
                                if self.active_zip_prompt.is_none() {
                                    self.show_zip_modal = false;
                                }
                            }
                            if ui.button("Cancel All").clicked() {
                                self.pending_zip_passwords.clear();
                                self.active_zip_prompt = None;
                                self.zip_password_input.clear();
                                self.show_zip_modal = false;
                            }
                        });
                    });
            }

            self.apply_tab_actions(ui);

            // Summary inline (counts) if selection active
            if !self.selection.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.selection.len()));
            }

            // Similarity button relocated to navbar
        });
        self.auto_load_more();
        self.viewer.thumbs.record_ui_time(started.elapsed());
    }

    /// Opens what a click on a category, tag or archive in the table asked for; Ctrl opens a new tab.
    fn apply_tab_actions(&mut self, ui: &Ui) {
        if self.viewer.requested_tabs.is_empty() {
            return;
        }
        let new_tab = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
        for action in std::mem::take(&mut self.viewer.requested_tabs) {
            let scope = match action {
                table::TabAction::OpenCategory(category) => crate::command::Scope::Category(category),
                table::TabAction::OpenTag(tag) => crate::command::Scope::Tag(tag),
                table::TabAction::OpenSimilar(path) => crate::command::Scope::Similar(path),
                table::TabAction::OpenArchive(path) => explorer::archive_scope(&path),
            };
            if new_tab {
                crate::command::send(crate::command::AppCommand::OpenScope { scope, target: crate::command::Target::NewTab, title: None, recursive: false });
            } else if let Err(reason) = self.open_scope(scope) {
                crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
            }
        }
    }

    /// Number of selected rows in the current table.
    pub fn selection_count(&self) -> usize {
        self.selection.len()
    }

    // Recompute total pages for current recursive snapshot (uses filtered rows for accurate pagination)
    fn update_recursive_total_pages(&mut self) {
        if self.recursive_page_size == 0 {
            self.recursive_total_pages = 0;
            return;
        }
        // Use filtered rows count for pagination, not the unfiltered last_scan_rows
        let total_rows = self.recursive_filtered_rows.len();
        self.recursive_total_pages =
            (total_rows + self.recursive_page_size - 1) / self.recursive_page_size;
        if self.recursive_current_page >= self.recursive_total_pages
            && self.recursive_total_pages > 0
        {
            self.recursive_current_page = self.recursive_total_pages - 1;
        }
    }

    // Build table contents for the active recursive page from snapshot rows
    fn rebuild_recursive_page(&mut self) {
        // Build filtered+sorted snapshot first
        self.apply_recursive_filter_and_sort();
        self.table.clear();
        let total_rows = self.recursive_filtered_rows.len();
        // Defensive recompute & clamp.
        if self.recursive_page_size == 0 {
            self.recursive_current_page = 0;
        } else {
            let max_pages = if total_rows == 0 {
                0
            } else {
                (total_rows + self.recursive_page_size - 1) / self.recursive_page_size
            };
            if max_pages == 0 {
                self.recursive_current_page = 0;
            } else if self.recursive_current_page >= max_pages {
                self.recursive_current_page = max_pages - 1;
            }
        }
        let start = self
            .recursive_current_page
            .saturating_mul(self.recursive_page_size);
        if start >= total_rows {
            return;
        }
        let end = (start + self.recursive_page_size).min(total_rows);
        log::warn!(
            "rebuild_recursive_page: total_rows={}, page_size={}, current_page={}, slice={}..{}",
            total_rows,
            self.recursive_page_size,
            self.recursive_current_page,
            start,
            end
        );
        self.viewer.thumbs.new_scope(self.viewer.scope);
        self.table.bulk_load(self.recursive_filtered_rows[start..end].iter().cloned());
        // Ask for a repaint on next frame (we don't hold a ctx here)
        self.repaint_next = true;
    }

    fn apply_recursive_filter_and_sort(&mut self) {
        // Build signature (filter text + type toggles + source row count) to avoid unnecessary rebuild work
        // Include last_scan_rows.len() so we recompute when new rows arrive during a scan
        let sig = format!(
            "{:x}|i{}|v{}|d{}|n{}",
            self.viewer.filter_signature(),
            self.viewer.types_show_images as u8,
            self.viewer.types_show_videos as u8,
            self.viewer.types_show_dirs as u8,
            self.last_scan_rows.len()
        );
        let changed = self
            .recursive_filter_sig
            .as_ref()
            .map(|p| p != &sig)
            .unwrap_or(true);
        // If unchanged (including row count) and we already have filtered rows, skip recompute
        if !changed && !self.recursive_filtered_rows.is_empty() {
            return;
        }
        // Only reset page if filter settings changed (not just row count)
        let filter_changed = self
            .recursive_filter_sig
            .as_ref()
            .map(|p| {
                // Compare without the row count suffix
                let old_parts: Vec<&str> = p.splitn(5, '|').collect();
                let new_parts: Vec<&str> = sig.splitn(5, '|').collect();
                old_parts.get(..4) != new_parts.get(..4)
            })
            .unwrap_or(true);
        if filter_changed {
            self.recursive_current_page = 0;
        }
        self.recursive_filter_sig = Some(sig.clone());
        let filter = self.viewer.row_filter();
        let mut out: Vec<Thumbnail> = self
            .last_scan_rows
            .iter()
            .filter(|r| filter.matches(r))
            .cloned()
            .collect();
        drop(filter);
        // Optional sort (basic keys; can expand)
        if let Some(key) = self.recursive_sort_key {
            match key {
                "name" => out.sort_by(|a, b| a.filename.cmp(&b.filename)),
                "modified" => out.sort_by(|a, b| a.modified.cmp(&b.modified)),
                "size" => out.sort_by(|a, b| a.size.cmp(&b.size)),
                _ => {}
            }
        }
        self.recursive_filtered_rows = out;
        self.update_recursive_total_pages();
    }

    // Public helpers used by navbar for global counts
    pub fn recursive_total_unfiltered(&self) -> usize {
        self.last_scan_rows.len()
    }
    pub fn recursive_total_filtered(&self) -> usize {
        if self.recursive_filtered_rows.is_empty() {
            self.last_scan_rows.len()
        } else {
            self.recursive_filtered_rows.len()
        }
    }
}
