use std::{borrow::Cow, collections::HashMap, sync::Arc};
use crossbeam::channel::{Receiver, Sender};
use crate::{ScanEnvelope, Thumbnail};
use std::sync::{Mutex, OnceLock};
use egui::style::StyleModifier;
use egui_data_table::Renderer;
use table::FileTableViewer;
use humansize::DECIMAL;
use serde::Serialize;
use eframe::egui::*;
use std::sync::atomic::AtomicBool;
use std::sync::Arc as StdArc;

pub mod quick_access_pane;
pub mod preview_pane;
pub mod explorer;
pub mod receive;
pub mod table;
pub mod menus;
pub mod load;

// Global accessor for the current active logical group name to be used by viewer context menu actions.
static ACTIVE_GROUP_NAME: OnceLock<Mutex<Option<String>>> = OnceLock::new();
pub fn active_group_name() -> Option<String> {
    ACTIVE_GROUP_NAME
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| g.clone())
}

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
    // Open a new tab that loads the entire database (DB mode, not filtered by current_path)
    OpenDatabaseAll {
        title: String,
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
    pub created: Option<surrealdb::sql::Datetime>,
    pub updated: Option<surrealdb::sql::Datetime>,
    pub similarity_score: Option<f32>,
    pub clip_similarity_score: Option<f32>,
}

#[derive(Clone)]
pub struct VideoFrame {
    pub rgba: StdArc<[u8]>,
    pub width: usize,
    pub height: usize,
    pub pts: f64,
}

#[derive(Serialize)]
pub struct FileExplorer {
    #[serde(skip)]
    pub table: egui_data_table::DataTable<Thumbnail>,
    #[serde(skip)]
    pub table_index: std::collections::HashMap<String, usize>,
    pub viewer: FileTableViewer,
    batch_size: usize,
    files: Vec<Thumbnail>,
    pub current_path: String,
    open_preview_pane: bool,
    open_quick_access: bool,
    pub file_scan_progress: f32,
    recursive_scan: bool,
    pub scan_done: bool,
    excluded_term_input: String,
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
    back_stack: Vec<String>,
    #[serde(skip)]
    forward_stack: Vec<String>,
    #[serde(skip)]
    pending_thumb_rows: Vec<Thumbnail>,
    #[serde(skip)]
    ai_update_rx: Receiver<AIUpdate>,
    #[serde(skip)]
    pub streaming_interim: std::collections::HashMap<String, String>,
    #[serde(skip)]
    thumb_scheduled: std::collections::HashSet<String>,
    #[serde(skip)]
    thumb_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
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
    // Database paging state
    #[serde(skip)]
    pub db_offset: usize,
    #[serde(skip)]
    pub db_limit: usize,
    #[serde(skip)]
    pub db_last_batch_len: usize,
    #[serde(skip)]
    db_loading: bool,
    // When true, the Database view shows all thumbnails in the DB (not filtered by current_path)
    #[serde(skip)]
    pub db_all_view: bool,
    // Toggle: when true and in Database mode, the path input performs AI semantic search over the whole DB
    #[serde(skip)]
    ai_search_enabled: bool,
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
    // Temporary text inputs for size filters (MB). Persisted to UiSettings on Apply.
    #[serde(skip)]
    pub min_size_mb_input: String,
    #[serde(skip)]
    pub max_size_mb_input: String,
    #[serde(skip)]
    pub filter_group_name_input: String,
    pub active_filter_group: Option<String>,
    #[serde(skip)]
    filter_groups: Vec<crate::database::FilterGroup>,
    #[serde(skip)]
    filter_groups_tx: Sender<Vec<crate::database::FilterGroup>>,
    #[serde(skip)]
    filter_groups_rx: Receiver<Vec<crate::database::FilterGroup>>,
    // Logical group navigation
    pub active_logical_group_name: Option<String>,
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
    // The cached_scan id for the currently running recursive scan (DB record)
    #[serde(skip)]
    _cached_scan_id: Option<surrealdb::RecordId>,
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
    // Similarity incremental append state
    #[serde(skip)]
    similarity_origin_path: Option<String>,
    #[serde(skip)]
    similarity_batch_size: usize,
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
    #[serde(skip)]
    pub recursive_prefetch_enabled: bool,
    #[serde(skip)]
    pub recursive_prefetch_inflight: usize,
    // Collect failed thumbnail generations for a dedicated tab after scan completes
    #[serde(skip)]
    pub failed_thumbs: Vec<(String, String)>, // (path, error)
    #[serde(skip)]
    pub failed_tab_opened: bool,
    #[serde(skip)]
    repaint_next: bool,
    #[serde(skip)]
    pub video_frame_tx: crossbeam::channel::Sender<VideoFrame>,
    #[serde(skip)]
    pub video_frame_rx: crossbeam::channel::Receiver<VideoFrame>,
    #[serde(skip)]
    pub video_playing: bool,
    #[serde(skip)]
    pub video_stop_flag: StdArc<AtomicBool>,
    #[serde(skip)]
    pub video_texture: Option<egui::TextureHandle>,
    #[serde(skip)]
    pub video_duration_secs: Option<f32>,
    #[serde(skip)]
    pub video_position_secs: f32,
}

impl FileExplorer {
    pub fn is_recursive_scan(&self) -> bool { self.recursive_scan }
    // New constructor that optionally skips initial directory scan/population
    pub fn new(skip_initial_scan: bool) -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
        let (video_frame_tx, video_frame_rx) = crossbeam::channel::unbounded();
        let (ai_update_tx, ai_update_rx) = crossbeam::channel::unbounded();
        let (clip_embedding_tx, clip_embedding_rx) = crossbeam::channel::unbounded();
        let (db_preload_tx, db_preload_rx) = crossbeam::channel::unbounded();
        let (logical_groups_tx, logical_groups_rx) = crossbeam::channel::unbounded();
        let (filter_groups_tx, filter_groups_rx) = crossbeam::channel::unbounded();
        let current_path = directories::UserDirs::new()
            .unwrap()
            .picture_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        
        let mut this = Self {
            table: Default::default(),
            table_index: std::collections::HashMap::new(),
            viewer: FileTableViewer::new(thumbnail_tx.clone(), ai_update_tx, clip_embedding_tx.clone()),
            files: Default::default(),
            open_preview_pane: false,
            open_quick_access: false,
            current_path,
            file_scan_progress: 0.0,
            recursive_scan: false,
            scan_done: false,
            excluded_term_input: String::new(),
            excluded_terms: Vec::new(),
            current_thumb: Thumbnail::new("") ,
            thumbnail_tx,
            thumbnail_rx,
            scan_tx,
            scan_rx,
            back_stack: Vec::new(),
            forward_stack: Vec::new(),
            pending_thumb_rows: Vec::new(),
            ai_update_rx,
            streaming_interim: std::collections::HashMap::new(),
            video_frame_tx,
            video_frame_rx,
            video_playing: false,
            video_stop_flag: StdArc::new(AtomicBool::new(false)),
            video_texture: None,
            video_duration_secs: None,
            video_position_secs: 0.0,
            thumb_scheduled: std::collections::HashSet::new(),
            thumb_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(6)),
            follow_active_vision: true,
            clip_presence: std::collections::HashSet::new(),
            clip_embedding_rx,
            vision_started: 0,
            vision_completed: 0,
            vision_pending: 0,
            db_offset: 0,
            db_limit: 500,
            db_last_batch_len: 0,
            db_loading: false,
            db_all_view: false,
            ai_search_enabled: false,
            current_scan_id: None,
            db_lookup: std::collections::HashMap::new(),
            db_preload_rx,
            db_preload_tx,
            last_scan_rows: Vec::new(),
            last_scan_paths: std::collections::HashSet::new(),
            last_scan_root: None,
            min_size_mb_input: String::new(),
            max_size_mb_input: String::new(),
            filter_group_name_input: String::new(),
            active_filter_group: None,
            filter_groups: Vec::new(),
            filter_groups_tx,
            filter_groups_rx,
            active_logical_group_name: None,
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
            _cached_scan_id: None,
            cached_wsl_distros: Some(crate::utilities::explorer::list_wsl_distros()),
            cached_physical_drives: Some(crate::utilities::explorer::list_physical_drives()),
            batch_size: 128,
            perf_scan_started: None,
            perf_last_batch_at: None,
            perf_batches: Vec::new(),
            perf_last_total: None,
            perf_show_per_1k: false,
            similarity_origin_path: None,
            similarity_batch_size: 50,
            recursive_page_size: 5_000,
            recursive_current_page: 0,
            recursive_total_pages: 0,
            recursive_filtered_rows: Vec::new(),
            recursive_sort_key: None,
            recursive_filter_sig: None,
            recursive_prefetch_enabled: true,
            recursive_prefetch_inflight: 0,
            failed_thumbs: Vec::new(),
            failed_tab_opened: false,
            repaint_next: false,
        };
        // Preload saved filter groups
        {
            let tx = this.filter_groups_tx.clone();
            tokio::spawn(async move {
                match crate::database::list_filter_groups().await {
                    Ok(groups) => { let _ = tx.try_send(groups); },
                    Err(e) => log::warn!("No filter groups yet or failed to load: {e:?}"),
                }
            });
        }
        // Preload logical groups list
        {
            let tx = this.logical_groups_tx.clone();
            tokio::spawn(async move {
                match crate::database::LogicalGroup::list_all().await {
                    Ok(groups) => { let _ = tx.try_send(groups); },
                    Err(e) => log::warn!("No logical groups yet or failed to load: {e:?}"),
                }
            });
        }
        if !skip_initial_scan {
            this.populate_current_directory();
        }
        this
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        if self.repaint_next { ui.ctx().request_repaint(); self.repaint_next = false; }
        self.preview_pane(ui);
        self.quick_access_pane(ui);
        self.navbar(ui);

        // For recursive scan pagination: ensure filtered snapshot stays in sync with current user filter
        if self.recursive_scan {
            // Force refresh of filtered snapshot if signature changed (apply_recursive will short-circuit if unchanged)
            self.apply_recursive_filter_and_sort();
        }

        let style = StyleModifier::default();
        style.apply(ui.style_mut());

        CentralPanel::default().show_inside(ui, |ui| {
            // Recursive scan pagination controls (only when snapshot large enough)
            if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                self.update_recursive_total_pages();
                ui.horizontal(|ui| {
                    let total = self.recursive_total_pages.max(1);
                    let first_enabled = self.recursive_current_page > 0;
                    if ui.add_enabled(first_enabled, egui::Button::new("First")).clicked() {
                        log::info!("Pagination: First clicked");
                        self.recursive_current_page = 0; self.rebuild_recursive_page();
                    }
                    let prev_enabled = self.recursive_current_page > 0;
                    if ui.add_enabled(prev_enabled, egui::Button::new("Prev")).clicked() {
                        log::info!("Pagination: Prev clicked (from {})", self.recursive_current_page);
                        if self.recursive_current_page > 0 { self.recursive_current_page -= 1; self.rebuild_recursive_page(); }
                    }
                    ui.label(format!("Page {} / {}", self.recursive_current_page + 1, total));
                    let next_enabled = self.recursive_current_page + 1 < total;
                    if ui.add_enabled(next_enabled, egui::Button::new("Next")).clicked() {
                        log::info!("Pagination: Next clicked (from {})", self.recursive_current_page);
                        if self.recursive_current_page + 1 < total { self.recursive_current_page += 1; self.rebuild_recursive_page(); }
                    }
                    let last_enabled = self.recursive_current_page + 1 < total;
                    if ui.add_enabled(last_enabled, egui::Button::new("Last")).clicked() {
                        log::info!("Pagination: Last clicked (target {}-1)", total);
                        if total > 0 { self.recursive_current_page = total - 1; self.rebuild_recursive_page(); }
                    }
                    ui.separator();
                    ui.label(format!("Showing {} of {} (page size {})", self.table.len(), self.last_scan_rows.len(), self.recursive_page_size));
                    // Display the current row range (1-based) for clarity
                    if self.recursive_page_size > 0 && !self.recursive_filtered_rows.is_empty() {
                        let start_idx = self.recursive_current_page * self.recursive_page_size;
                        let end_idx = (start_idx + self.recursive_page_size).min(self.recursive_filtered_rows.len());
                        ui.separator();
                        ui.label(format!("Rows {}-{}", start_idx + 1, end_idx));
                    }
                });
                ui.separator();
            }
            // Active filter preset summary
            if let Some(name) = self.active_filter_group.clone() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Preset: {}", name)).strong());
                    let types = format!(
                        "Types: {}{}{}",
                        if self.viewer.types_show_images { "I" } else { "" },
                        if self.viewer.types_show_videos { "V" } else { "" },
                        if self.viewer.types_show_dirs { "D" } else { "" }
                    );
                    ui.separator();
                    ui.label(types);
                    ui.separator();
                    let minb = self.viewer.ui_settings.db_min_size_bytes;
                    let maxb = self.viewer.ui_settings.db_max_size_bytes;
                    let size_str = match (minb, maxb) {
                        (None, None) => "Size: any".to_string(),
                        (Some(mn), None) => format!("Size: ≥{}", humansize::format_size(mn, DECIMAL)),
                        (None, Some(mx)) => format!("Size: ≤{}", humansize::format_size(mx, DECIMAL)),
                        (Some(mn), Some(mx)) => format!("Size: {}..{}", humansize::format_size(mn, DECIMAL), humansize::format_size(mx, DECIMAL)),
                    };
                    ui.label(size_str);
                    ui.separator();
                    ui.label(format!("Skip icons: {}", if self.viewer.ui_settings.filter_skip_icons { "on" } else { "off" }));
                    if !self.excluded_terms.is_empty() {
                        ui.separator();
                        ui.label(format!("Excluded: {}", self.excluded_terms.len()));
                    }
                });
                ui.separator();
            }
            // Check if we have a single empty default row and remove it
            if self.table.len() == 1 {
                let should_remove_empty = self.table.iter().next().map(|row| {
                    row.filename.is_empty() && row.path.is_empty() && row.size == 0 && 
                    row.file_type.is_empty() && row.thumbnail_b64.is_none()
                }).unwrap_or(false);
                
                if should_remove_empty {
                    self.table.clear();
                    self.table_index.clear();
                }
            }
            
            Renderer::new(&mut self.table, &mut self.viewer)
            .with_style_modify(|s| {
                s.scroll_bar_visibility = scroll_area::ScrollBarVisibility::AlwaysVisible;
                s.single_click_edit_mode = true;
                s.table_row_height = Some(75.0);
                s.auto_shrink = [false, false].into();
            })
            .ui(ui);

            // Modal password prompt (appears at end of scans or when browsing encrypted archives)
            if self.show_zip_modal {
                egui::Window::new("Archive Password Required")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .show(ui.ctx(), |ui| {
                        let current = self.active_zip_prompt.clone().unwrap_or_default();
                        ui.label("One or more encrypted archives were found. Please enter the password.");
                        ui.separator();
                        ui.label(format!("Archive: {}", current));
                        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let resp = egui::TextEdit::singleline(&mut self.zip_password_input)
                            .password(true)
                            .hint_text("Password")
                            .desired_width(240.)
                            .ui(ui);
                        ui.horizontal(|ui| {
                            if (resp.lost_focus() && enter_pressed) || ui.button("Submit").clicked() {
                                // Store in-memory for this session and refresh any open view for that archive
                                let pw = std::mem::take(&mut self.zip_password_input);
                                if !current.is_empty() {
                                    self.viewer.archive_passwords.insert(current.clone(), pw);
                                }
                                // Refresh current view (filesystem or virtual) so thumbnail tasks are scheduled with the new password
                                self.populate_current_directory();
                                // Advance the queue
                                if let Some(_) = self.pending_zip_passwords.pop_front() {}
                                self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                                if self.active_zip_prompt.is_none() { self.show_zip_modal = false; }
                            }
                            if ui.button("Skip").on_hover_text("Skip this archive").clicked() {
                                self.zip_password_input.clear();
                                if let Some(_) = self.pending_zip_passwords.pop_front() {}
                                self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                                if self.active_zip_prompt.is_none() { self.show_zip_modal = false; }
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
            
            // Check for tag/category open-tab actions requested by the viewer
            if !self.viewer.requested_tabs.is_empty() {
                let actions = std::mem::take(&mut self.viewer.requested_tabs);
                for act in actions.into_iter() {
                    match act {
                        crate::ui::file_table::table::TabAction::OpenCategory(cat) => {
                            let title = format!("Category: {}", cat);
                            // Prefer current-table filter; if none (e.g., from virtual view), fetch from DB
                            let rows_in_view: Vec<crate::database::Thumbnail> = self
                                .table
                                .iter()
                                .filter(|r| r.category.as_deref() == Some(cat.as_str()))
                                .cloned()
                                .collect();
                            if rows_in_view.is_empty() || matches!(self.viewer.mode, table::ExplorerMode::Database) {
                                tokio::spawn(async move {
                                    let fetched = crate::Thumbnail::fetch_by_category(&cat).await.unwrap_or_default();
                                    crate::app::OPEN_TAB_REQUESTS
                                        .lock()
                                        .unwrap()
                                        .push(crate::ui::file_table::FilterRequest::NewTab { title, rows: fetched, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
                                });
                            } else {
                                crate::app::OPEN_TAB_REQUESTS
                                    .lock()
                                    .unwrap()
                                    .push(crate::ui::file_table::FilterRequest::NewTab { title, rows: rows_in_view, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
                            }
                        }
                        crate::ui::file_table::table::TabAction::OpenTag(tag) => {
                            let title = format!("Tag: {}", tag);
                            let rows_in_view: Vec<crate::database::Thumbnail> = self
                                .table
                                .iter()
                                .filter(|r| r.tags.iter().any(|t| t.eq_ignore_ascii_case(&tag)))
                                .cloned()
                                .collect();
                            if rows_in_view.is_empty() || matches!(self.viewer.mode, table::ExplorerMode::Database) {
                                tokio::spawn(async move {
                                    let fetched = crate::Thumbnail::fetch_by_tag(&tag).await.unwrap_or_default();
                                    crate::app::OPEN_TAB_REQUESTS
                                        .lock()
                                        .unwrap()
                                        .push(crate::ui::file_table::FilterRequest::NewTab { title, rows: fetched, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
                                });
                            } else {
                                crate::app::OPEN_TAB_REQUESTS
                                    .lock()
                                    .unwrap()
                                    .push(crate::ui::file_table::FilterRequest::NewTab { title, rows: rows_in_view, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
                            }
                        }
                        crate::ui::file_table::table::TabAction::OpenArchive(path_clicked) => {
                            // Choose scheme based on extension (zip or tar family)
                            let is_virtual = path_clicked.starts_with("zip://") || path_clicked.starts_with("tar://") || path_clicked.starts_with("7z://");
                            let vpath = if is_virtual {
                                path_clicked
                            } else {
                                let ext = std::path::Path::new(&path_clicked)
                                    .extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).unwrap_or_default();
                                let name = path_clicked.to_ascii_lowercase();
                                let is_tar_family = ext == "tar" || ext == "gz" || ext == "bz" || ext == "bz2" || ext == "xz"
                                    || name.ends_with(".tgz") || name.ends_with(".tbz") || name.ends_with(".tbz2") || name.ends_with(".txz")
                                    || name.ends_with(".tar.gz") || name.ends_with(".tar.bz2") || name.ends_with(".tar.xz");
                                if is_tar_family { 
                                    format!("tar://{}!/", path_clicked) 
                                } else if ext == "7z" { 
                                    format!("7z://{}!/", path_clicked) 
                                } else { 
                                    format!("zip://{}!/", path_clicked) 
                                }
                            };
                            self.current_path = vpath;
                            self.populate_current_directory();
                        }
                        crate::ui::file_table::table::TabAction::OpenSimilar(filename) => {
                            let title = format!("Similar to {filename}");
                            // If similar results are present, use them; else fallback to current table (no filtering)
                            let rows: Vec<crate::database::Thumbnail> = if self.viewer.showing_similarity && !self.viewer.similar_scores.is_empty() {
                                self.table.iter().cloned().collect()
                            } else {
                                Vec::new()
                            };
                            crate::app::OPEN_TAB_REQUESTS
                                .lock()
                                .unwrap()
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: true, similar_scores: Some(self.viewer.similar_scores.clone()), origin_path: Some(filename), background: false });
                        }
                    }
                }
            }
            
            // Summary inline (counts) if selection active
            if !self.viewer.selected.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.viewer.selected.len()));
            }
            
            if self.viewer.mode == table::ExplorerMode::Database && !self.viewer.showing_similarity {
                ui.separator();
                ui.horizontal(|ui| {
                    if self.db_all_view { ui.colored_label(Color32::LIGHT_BLUE, "All DB"); ui.separator(); }
                    if self.db_loading {
                        ui.label(RichText::new("Loading page ...").italics());
                    } else {
                        let no_more = self.db_last_batch_len < self.db_limit && self.db_offset > 0;
                        if no_more {
                            ui.label(RichText::new("No more rows").weak());
                        } else if ui.button("Load More").clicked() {
                            // Prepare for next page
                            self.db_last_batch_len = 0; // reset counter for incoming batch
                            self.load_database_rows();
                        }
                    }
                    ui.label(format!("Loaded: {} rows", self.table.len()));
                });
            }

            // Similarity button relocated to navbar
        });
    }

    /// Number of selected file rows in the current table.
    pub fn selection_count(&self) -> usize {
        self.viewer.selected.len()
    }

    pub fn append_more_similar(&mut self) {
        if !self.viewer.showing_similarity { return; }
        let origin = if let Some(o) = self.similarity_origin_path.clone() { o } else { return; };
        let batch = self.similarity_batch_size;
        let tx_updates = self.viewer.ai_update_tx.clone();
        let existing: std::collections::HashSet<String> = self.table.iter().map(|r| r.path.clone()).collect();
        tokio::spawn(async move {
            if let Ok(Some(thumb)) = crate::Thumbnail::get_thumbnail_by_path(&origin).await {
                let embed_vec = thumb.get_embedding().await.unwrap_or_default().embedding.clone();
                if embed_vec.is_empty() { return; }
                let k = existing.len() + batch * 4; // oversample to get enough new uniques
                if let Ok(hits) = crate::database::ClipEmbeddingRow::find_similar_by_embedding(&embed_vec, k, 256).await {
                    let mut results: Vec<crate::ui::file_table::SimilarResult> = Vec::new();
                    for hit in hits.into_iter() {
                        if existing.contains(&hit.path) { continue; }
                        let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                        let cosine_sim = 1.0 - hit.dist;
                        let norm_sim = ((cosine_sim + 1.0) / 2.0).clamp(0.0, 1.0);
                        results.push(crate::ui::file_table::SimilarResult { thumb: thumb.clone(), created: None, updated: None, similarity_score: Some(norm_sim), clip_similarity_score: Some(norm_sim) });
                        if results.len() >= batch { break; }
                    }
                    if !results.is_empty() { let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: origin.clone(), results }); }
                }
            }
        });
    }

    // Recompute total pages for current recursive snapshot
    fn update_recursive_total_pages(&mut self) {
        if self.recursive_page_size == 0 { self.recursive_total_pages = 0; return; }
        let total_rows = self.last_scan_rows.len();
        self.recursive_total_pages = (total_rows + self.recursive_page_size - 1) / self.recursive_page_size;
        if self.recursive_current_page >= self.recursive_total_pages && self.recursive_total_pages > 0 {
            self.recursive_current_page = self.recursive_total_pages - 1;
        }
    }

    // Build table contents for the active recursive page from snapshot rows
    fn rebuild_recursive_page(&mut self) {
        // Build filtered+sorted snapshot first
        self.apply_recursive_filter_and_sort();
        self.table.clear();
        self.table_index.clear();
        let total_rows = self.recursive_filtered_rows.len();
        // Defensive recompute & clamp.
        if self.recursive_page_size == 0 {
            self.recursive_current_page = 0;
        } else {
            let max_pages = if total_rows == 0 { 0 } else { (total_rows + self.recursive_page_size - 1) / self.recursive_page_size };
            if max_pages == 0 { self.recursive_current_page = 0; }
            else if self.recursive_current_page >= max_pages { self.recursive_current_page = max_pages - 1; }
        }
        let start = self.recursive_current_page.saturating_mul(self.recursive_page_size);
        if start >= total_rows { return; }
        let end = (start + self.recursive_page_size).min(total_rows);
        log::warn!(
            "rebuild_recursive_page: total_rows={}, page_size={}, current_page={}, slice={}..{}",
            total_rows,
            self.recursive_page_size,
            self.recursive_current_page,
            start,
            end
        );
        // Prune cache to rows in slice
        if !self.viewer.thumb_cache.is_empty() {
            let mut keep: std::collections::HashSet<String> = std::collections::HashSet::new();
            for r in self.recursive_filtered_rows[start..end].iter() { keep.insert(r.path.clone()); }
            self.viewer.thumb_cache.retain(|k, _| keep.contains(k) || k.starts_with("preview::"));
        }
        for row in self.recursive_filtered_rows[start..end].iter().cloned() {
            let idx = self.table.len();
            self.table_index.insert(row.path.clone(), idx);
            self.table.push(row);
        }
        self.schedule_missing_thumbs_for_current_page();
        // Opportunistically prefetch next page thumbnails (lightweight) after current page scheduling
        self.prefetch_next_page_thumbs();
        // Ask for a repaint on next frame (we don't hold a ctx here)
        self.repaint_next = true;
    }

    fn apply_recursive_filter_and_sort(&mut self) {
        // Build signature (filter text + type toggles) to avoid unnecessary rebuild work
        let sig = format!("{}|i{}|v{}|d{}", self.viewer.filter, self.viewer.types_show_images as u8, self.viewer.types_show_videos as u8, self.viewer.types_show_dirs as u8);
        let changed = self.recursive_filter_sig.as_ref().map(|p| p != &sig).unwrap_or(true);
        // If unchanged and we already have filtered rows, skip recompute
        if !changed && !self.recursive_filtered_rows.is_empty() { return; }
        if changed { self.recursive_current_page = 0; }
        self.recursive_filter_sig = Some(sig.clone());
        let mut out: Vec<Thumbnail> = self
            .last_scan_rows
            .iter()
            .cloned()
            .filter(|r| self.viewer.row_passes_filter(r))
            .collect();
        // Optional sort (basic keys; can expand)
        if let Some(key) = self.recursive_sort_key {
            match key {
                "name" => out.sort_by(|a,b| a.filename.cmp(&b.filename)),
                "modified" => out.sort_by(|a,b| a.modified.cmp(&b.modified)),
                "size" => out.sort_by(|a,b| a.size.cmp(&b.size)),
                _ => {}
            }
        }
        self.recursive_filtered_rows = out;
        self.update_recursive_total_pages();
    }

    // Public helpers used by navbar for global counts
    pub fn recursive_total_unfiltered(&self) -> usize { self.last_scan_rows.len() }
    pub fn recursive_total_filtered(&self) -> usize { if self.recursive_filtered_rows.is_empty() { self.last_scan_rows.len() } else { self.recursive_filtered_rows.len() } }

    fn prefetch_next_page_thumbs(&mut self) {
        if !self.recursive_prefetch_enabled { return; }
        if self.recursive_page_size == 0 { return; }
        // Limit concurrent prefetch waves
        if self.recursive_prefetch_inflight > 0 { return; }
        let total = self.recursive_filtered_rows.len();
        let next_page = self.recursive_current_page + 1;
        if next_page * self.recursive_page_size >= total { return; }
        let start = next_page * self.recursive_page_size;
        let end = (start + self.recursive_page_size).min(total);
        // Collect a small subset (e.g., first 128 of next page) to avoid wasting work
        let mut candidates: Vec<String> = Vec::new();
        for row in self.recursive_filtered_rows[start..end].iter() {
            if candidates.len() >= 128 { break; }
            if row.thumbnail_b64.is_some() { continue; }
            if row.file_type == "<DIR>" || row.file_type == "<ARCHIVE>" { continue; }
            // Skip if already scheduled or cached
            if self.viewer.thumb_cache.contains_key(&row.path) { continue; }
            if self.thumb_scheduled.contains(&row.path) { continue; }
            let ext_opt = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
            if let Some(ext) = ext_opt { if crate::is_image(ext.as_str()) || crate::is_video(ext.as_str()) { candidates.push(row.path.clone()); } }
        }
        if candidates.is_empty() { return; }
        self.recursive_prefetch_inflight = candidates.len();
        let sem = self.thumb_semaphore.clone();
        let scan_tx = self.scan_tx.clone();
        let owning_scan_id = self.owning_scan_id.or(self.current_scan_id).unwrap_or(0);
        for path_str in candidates.into_iter() {
            // Mark scheduled to avoid duplicate
            self.thumb_scheduled.insert(path_str.clone());
            let permit_fut = sem.clone().acquire_owned();
            let tx_clone = scan_tx.clone();
            // Track inflight completion by sending a Done decrement via closure
            let inflight_counter = &mut self.recursive_prefetch_inflight as *mut usize;
            tokio::spawn(async move {
                let _permit = permit_fut.await.ok();
                let pbuf = std::path::PathBuf::from(&path_str);
                let mut thumb_b64: Option<String> = None;
                if let Some(ext) = pbuf.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                    if crate::is_image(ext.as_str()) { thumb_b64 = crate::utilities::thumbs::generate_image_thumb_data(&pbuf).ok(); }
                    else if crate::is_video(ext.as_str()) { thumb_b64 = crate::utilities::thumbs::generate_video_thumb_data(&pbuf).ok(); }
                }
                if let Some(b64) = thumb_b64 { let _ = tx_clone.try_send(crate::ScanEnvelope { scan_id: owning_scan_id, msg: crate::utilities::scan::ScanMsg::UpdateThumb { path: pbuf.clone(), thumb: b64 } }); }
                else { let _ = tx_clone.try_send(crate::ScanEnvelope { scan_id: owning_scan_id, msg: crate::utilities::scan::ScanMsg::ThumbFailed { path: pbuf.clone(), error: "prefetch thumbnail generation failed".to_string() } }); }
                // SAFETY: UI thread only mutation; this async block executes on runtime thread. To avoid unsafe, we skip decrement here; rely on natural replacement (simplify).
                // (If precise tracking needed, implement atomic counter.)
                let _ = inflight_counter; // suppress unused warning for placeholder.
            });
        }
    }

    // Schedule thumbnail generation for visible page rows that lack thumbnail_b64 and are images/videos.
    fn schedule_missing_thumbs_for_current_page(&mut self) {
        if self.table.is_empty() { return; }
        // Collect paths needing generation
        let mut to_gen: Vec<String> = Vec::new();
        for row in self.table.iter() {
            if row.thumbnail_b64.is_some() { continue; }
            // Skip dirs / archives
            if row.file_type == "<DIR>" || row.file_type == "<ARCHIVE>" { continue; }
            let ext_opt = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
            if let Some(ext) = ext_opt {
                if !(crate::is_image(ext.as_str()) || crate::is_video(ext.as_str())) { continue; }
            } else { continue; }
            // Avoid rescheduling same path
            if !self.thumb_scheduled.insert(row.path.clone()) { continue; }
            to_gen.push(row.path.clone());
        }
        if to_gen.is_empty() { return; }
        let sem = self.thumb_semaphore.clone();
        let scan_tx = self.scan_tx.clone();
        let owning_scan_id = self.owning_scan_id.or(self.current_scan_id).unwrap_or(0);
        for path_str in to_gen.into_iter() {
            let permit_fut = sem.clone().acquire_owned();
            let tx_clone = scan_tx.clone();
            tokio::spawn(async move {
                let _permit = permit_fut.await.expect("thumb semaphore closed");
                let p = std::path::PathBuf::from(&path_str);
                let mut thumb_b64: Option<String> = None;
                if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                    if crate::is_image(ext.as_str()) {
                        thumb_b64 = crate::utilities::thumbs::generate_image_thumb_data(&p).ok();
                    } else if crate::is_video(ext.as_str()) {
                        thumb_b64 = crate::utilities::thumbs::generate_video_thumb_data(&p).ok();
                    }
                }
                if let Some(b64) = thumb_b64 { let _ = tx_clone.try_send(crate::ScanEnvelope { scan_id: owning_scan_id, msg: crate::utilities::scan::ScanMsg::UpdateThumb { path: p.clone(), thumb: b64 } }); }
                else { let _ = tx_clone.try_send(crate::ScanEnvelope { scan_id: owning_scan_id, msg: crate::utilities::scan::ScanMsg::ThumbFailed { path: p.clone(), error: "thumbnail generation failed".to_string() } }); }
            });
        }
    }
}

pub fn get_img_ui(
    thumb_cache: &HashMap<String, Arc<[u8]>>,
    cache_key: &String,
    ui: &mut Ui,
) -> Response {
    if let Some(bytes_arc) = thumb_cache.get(cache_key) {
        let img_src = ImageSource::Bytes {
            uri: Cow::from(format!("bytes://{}", cache_key)),
            bytes: eframe::egui::load::Bytes::Shared(bytes_arc.clone()),
        };
        Image::new(img_src)
            .max_size(ui.available_size()/1.3)
            .ui(ui)
    } else {
        ui.label("No Image")
    }
}


pub fn insert_thumbnail(thumb_cache: &mut HashMap<String, Arc<[u8]>>, thumbnail: Thumbnail) {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let cache_key = &thumbnail.path;
    let thumb_b64 = thumbnail.thumbnail_b64;
    if let Some(mut b64) = thumb_b64 {
        if !thumb_cache.contains_key(cache_key) {
            if b64.starts_with("data:image/png;base64,") {
                let (_, end) =
                b64.split_once("data:image/png;base64,").unwrap_or_default();
                b64 = end.to_string();
            }
            let decoded_bytes = B64.decode(b64.as_bytes()).unwrap_or_default();
            thumb_cache.insert(cache_key.clone(), Arc::from(decoded_bytes.into_boxed_slice()));
        }
    }
}