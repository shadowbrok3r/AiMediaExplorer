use crate::{ScanEnvelope, Thumbnail, utilities::windows::{gpu_mem_mb, system_mem_mb, smoothed_cpu01, smoothed_ram01, smoothed_vram01}};
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};
use std::{borrow::Cow, collections::HashMap, sync::Arc};
use crossbeam::channel::{Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use egui_data_table::Renderer;
use table::FileTableViewer;
use humansize::DECIMAL;
use serde::Serialize;
use eframe::egui::*;

pub mod quick_access_pane;
pub mod preview_pane;
pub mod explorer;
pub mod receive;
pub mod table;

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

#[derive(Serialize)]
pub struct FileExplorer {
    #[serde(skip)]
    table: egui_data_table::DataTable<Thumbnail>,
    pub viewer: FileTableViewer,
    files: Vec<Thumbnail>,
    pub current_path: String,
    open_preview_pane: bool,
    open_quick_access: bool,
    file_scan_progress: f32,
    recursive_scan: bool,
    scan_done: bool,
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
    selected: std::collections::HashSet<String>,
    #[serde(skip)]
    ai_update_rx: Receiver<AIUpdate>,
    #[serde(skip)]
    streaming_interim: std::collections::HashMap<String, String>,
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
    vision_started: usize,
    #[serde(skip)]
    vision_completed: usize,
    #[serde(skip)]
    vision_pending: usize, // scheduled but not yet final
    // Database paging state
    #[serde(skip)]
    pub db_offset: usize,
    #[serde(skip)]
    pub db_limit: usize,
    #[serde(skip)]
    pub db_last_batch_len: usize,
    #[serde(skip)]
    db_loading: bool,
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
    logical_groups_tx: Sender<Vec<crate::database::LogicalGroup>>,
    #[serde(skip)]
    logical_groups_rx: Receiver<Vec<crate::database::LogicalGroup>>,
    // Prefetch throttle for logical groups when DB becomes ready after construction
    #[serde(skip)]
    last_groups_fetch_attempt: Option<std::time::Instant>,
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
    // In-memory archive passwords for current session (keyed by archive filesystem path)
    #[serde(skip)]
    archive_passwords: std::collections::HashMap<String, String>,
    // Track which scan_id owns this explorer so updates don't leak across tabs
    #[serde(skip)]
    owning_scan_id: Option<u64>,
}

impl FileExplorer {
    // New constructor that optionally skips initial directory scan/population
    pub fn new(skip_initial_scan: bool) -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
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
            current_thumb: Thumbnail::default(),
            thumbnail_tx,
            thumbnail_rx,
            scan_tx,
            scan_rx,
            back_stack: Vec::new(),
            forward_stack: Vec::new(),
            pending_thumb_rows: Vec::new(),
            selected: std::collections::HashSet::new(),
            ai_update_rx,
            streaming_interim: std::collections::HashMap::new(),
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
            last_groups_fetch_attempt: None,
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
            archive_passwords: std::collections::HashMap::new(),
            owning_scan_id: None,
        };
        // Preload saved filter groups
        {
            let tx = this.filter_groups_tx.clone();
            tokio::spawn(async move {
                match crate::database::list_filter_groups().await {
                    Ok(groups) => { let _ = tx.try_send(groups); },
                    Err(e) => log::debug!("No filter groups yet or failed to load: {e:?}"),
                }
            });
        }
        // Preload logical groups list
        {
            let tx = this.logical_groups_tx.clone();
            tokio::spawn(async move {
                match crate::database::LogicalGroup::list_all().await {
                    Ok(groups) => { let _ = tx.try_send(groups); },
                    Err(e) => log::debug!("No logical groups yet or failed to load: {e:?}"),
                }
            });
        }
        if !skip_initial_scan {
            this.populate_current_directory();
        }
        this
    }

    /// Initialize this explorer to open a specific path in a new tab, optionally performing a recursive scan.
    /// This encapsulates internal state and avoids external code touching private fields.
    pub fn init_open_path(&mut self, path: &str, recursive: bool) {
        self.current_path = path.to_string();
        // reset view state
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.table.clear();
        if recursive {
            self.recursive_scan = true;
            self.scan_done = false;
            self.file_scan_progress = 0.0;
            // Reset previous scan snapshot
            self.last_scan_rows.clear();
            self.last_scan_paths.clear();
            self.last_scan_root = Some(self.current_path.clone());
            // Spawn recursive scan with current filters
            let scan_id = crate::next_scan_id();
            self.owning_scan_id = Some(scan_id);
            let tx = self.scan_tx.clone();
            let mut filters = crate::Filters::default();
            filters.root = std::path::PathBuf::from(self.current_path.clone());
            filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
            filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
            filters.include_images = self.viewer.types_show_images;
            filters.include_videos = self.viewer.types_show_videos;
            filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
            filters.excluded_terms = self.excluded_terms.clone();
            self.current_scan_id = Some(scan_id);
            tokio::spawn(async move { crate::spawn_scan(filters, tx, true, scan_id).await; });
        } else {
            // Shallow scan / directory populate
            self.populate_current_directory();
        }
    }
    
    pub fn set_rows(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = table::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
    }
    
    // Set the table rows from DB results and switch to Database viewing mode
    pub fn set_rows_from_db(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = table::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        // Keep global active group name snapshot in sync for viewer context menu
        {
            let guard = ACTIVE_GROUP_NAME.get_or_init(|| Mutex::new(None));
            if let Ok(mut g) = guard.lock() { *g = self.active_logical_group_name.clone(); }
        }
        // Integrate any freshly loaded filter groups
        while let Ok(groups) = self.filter_groups_rx.try_recv() {
            self.filter_groups = groups;
        }
        // Integrate any freshly loaded logical groups
        while let Ok(groups) = self.logical_groups_rx.try_recv() {
            self.logical_groups = groups;
            if self.logical_groups.is_empty() {
                // Ensure at least Default exists, then refresh list
                let tx = self.logical_groups_tx.clone();
                tokio::spawn(async move {
                    let _ = crate::database::LogicalGroup::create("Default").await;
                    if let Ok(groups) = crate::database::LogicalGroup::list_all().await { let _ = tx.try_send(groups); }
                });
            } else if self.active_logical_group_name.is_none() {
                // Choose a sensible default: previously used from settings or first available
                let preferred = crate::ui::file_table::active_group_name();
                if let Some(name) = preferred {
                    if self.logical_groups.iter().any(|g| g.name == name) { self.active_logical_group_name = Some(name); }
                }
                if self.active_logical_group_name.is_none() {
                    self.active_logical_group_name = Some(self.logical_groups[0].name.clone());
                }
            }
        }
        // If DB became ready after construction and list is empty, try fetching groups automatically (no manual refresh needed)
        if self.logical_groups.is_empty() {
            let now = std::time::Instant::now();
            let should = match self.last_groups_fetch_attempt {
                None => true,
                Some(t) => now.duration_since(t) > std::time::Duration::from_secs(2),
            };
            if should {
                self.last_groups_fetch_attempt = Some(now);
                let tx = self.logical_groups_tx.clone();
                tokio::spawn(async move {
                    if let Ok(groups) = crate::database::LogicalGroup::list_all().await { let _ = tx.try_send(groups); }
                });
            }
        }
        self.preview_pane(ui);
        self.quick_access_pane(ui);

        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        MenuBar::new()
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .style(style)
        .ui(ui, |ui| {
            ui.set_height(25.);
            ui.horizontal_top(|ui| {
                let err_color = ui.style().visuals.error_fg_color;
                let font = FontId::proportional(15.);
                let size = vec2(25., 25.);
                ui.add_space(5.);
                if Button::new(
                    RichText::new("⚙").font(font.clone()).color(err_color)
                )
                .min_size(size)
                .ui(ui)
                .on_hover_text("Open Options side panel")
                .clicked() {
                    self.open_quick_access = !self.open_quick_access;
                }
                
                ui.add_space(5.);
                ui.separator();
                ui.add_space(5.);
                if Button::new(RichText::new("⬆").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_up(); }
                if Button::new(RichText::new("⬅").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_back(); }
                if Button::new(RichText::new("➡").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_forward(); }
                if Button::new(RichText::new("⟲").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.refresh(); }
                if Button::new(RichText::new("🏠").font(font).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_home(); }
                ui.separator();
                let path_edit = TextEdit::singleline(&mut self.current_path)
                .hint_text(if self.viewer.mode == table::ExplorerMode::Database {
                    if self.ai_search_enabled { "AI Search (text prompt)" } else { "Path Prefix Filter" }
                } else {
                    "Current Directory"
                })
                .desired_width(300.)
                .ui(ui);

                // If browsing inside a zip archive, show a password button
                // Zip password button removed; we show a modal automatically when needed

                match self.viewer.mode {
                    table::ExplorerMode::FileSystem => {
                        if path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                            self.refresh();
                        }
                    },
                    table::ExplorerMode::Database => {
                        let mut ai_box = self.ai_search_enabled;
                        if ui.checkbox(&mut ai_box, "AI").on_hover_text("Use AI semantic search across the database (text prompt)").clicked() {
                            self.ai_search_enabled = ai_box;
                        }
                        // Apply filter on Enter
                        if (path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                            || ui.button("Apply Filter").clicked()
                        {
                            if self.ai_search_enabled {
                                // Run AI semantic search over the whole DB using text prompt in current_path
                                let query = self.current_path.trim().to_string();
                                if !query.is_empty() {
                                    let tx_updates = self.viewer.ai_update_tx.clone();
                                    let types_show_images = self.viewer.types_show_images;
                                    let types_show_videos = self.viewer.types_show_videos;
                                    let minb = self.viewer.ui_settings.db_min_size_bytes;
                                    let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                    tokio::spawn(async move {
                                        // Ensure engine ready
                                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                        // Embed query text
                                        let q_vec_opt = {
                                            let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
                                            if let Some(engine) = guard.as_mut() {
                                                engine.embed_text(&query).ok()
                                            } else { None }
                                        };
                                        if let Some(q) = q_vec_opt {
                                            let mut results: Vec<crate::ui::file_table::SimilarResult> = Vec::new();
                                            match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, 48, 96).await {
                                                Ok(hits) => {
                                                    for hit in hits.into_iter() {
                                                        // Get thumbnail record (prefer embedded thumb_ref on hit)
                                                        let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                                        // Optional: filter by type and size
                                                        let ext_ok = std::path::Path::new(&thumb.path)
                                                            .extension()
                                                            .and_then(|e| e.to_str())
                                                            .map(|s| s.to_ascii_lowercase());
                                                        let mut type_ok = true;
                                                        if let Some(ext) = ext_ok.as_ref() {
                                                            if crate::is_image(ext.as_str()) { type_ok = types_show_images; }
                                                            else if crate::is_video(ext.as_str()) { type_ok = types_show_videos; }
                                                            else { type_ok = false; }
                                                        }
                                                        if !type_ok { continue; }
                                                        if let Some(mn) = minb { if thumb.size < mn { continue; } }
                                                        if let Some(mx) = maxb { if thumb.size > mx { continue; } }

                                                        // Compute similarity vs stored embedding (prefer clip_similarity_score)
                                                        let (mut created, mut updated, mut stored_sim, mut clip_sim) = (None, None, None, None);
                                                        if let Ok(rows) = crate::database::ClipEmbeddingRow::load_clip_embeddings_for_path(&thumb.path).await {
                                                            for row in rows.iter() {
                                                                created = row.created.clone();
                                                                updated = row.updated.clone();
                                                                stored_sim = row.similarity_score.or(row.clip_similarity_score);
                                                                if clip_sim.is_none() && !row.embedding.is_empty() { clip_sim = Some(crate::ai::clip::dot(&q, &row.embedding)); }
                                                            }
                                                        }
                                                        results.push(crate::ui::file_table::SimilarResult { thumb, created, updated, similarity_score: stored_sim, clip_similarity_score: clip_sim });
                                                    }
                                                }
                                                Err(e) => log::error!("[AI] text knn failed: {e:?}"),
                                            }
                                            let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: format!("query:{query}"), results });
                                        }
                                    });
                                }
                            } else {
                                // Reset paging and reload
                                self.db_offset = 0;
                                self.db_last_batch_len = 0;
                                self.load_database_rows();
                            }
                        }
                    },
                }
                
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    MenuButton::new("🔻")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
                    .ui(ui, |ui| {
                        ui.label(RichText::new("Excluded Terms (substring, case-insensitive)").italics());
                        ui.horizontal(|ui| {
                            let resp = TextEdit::singleline(&mut self.excluded_term_input)
                                .hint_text("term")
                                .desired_width(140.)
                                .ui(ui);
                            let add_clicked = ui.button("Add").clicked();
                            if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || add_clicked {
                                resp.request_focus();
                                let term = self.excluded_term_input.trim().to_ascii_lowercase();
                                if !term.is_empty() && !self.excluded_terms.iter().any(|t| t == &term) { self.excluded_terms.push(term); }
                                // Manual change invalidates active preset label
                                self.active_filter_group = None;
                                self.excluded_term_input.clear();
                                // Apply excluded terms immediately to current table (keep directories)
                                if !self.excluded_terms.is_empty() {
                                    let terms = self.excluded_terms.clone();
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        let lp = r.path.to_ascii_lowercase();
                                        !terms.iter().any(|t| lp.contains(t))
                                    });
                                }
                            }
                            if ui.button("Clear All").clicked() { 
                                self.excluded_terms.clear();
                                self.active_filter_group = None;
                                // Note: clearing does not restore previously filtered rows to keep UX consistent with size filters
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            let mut remove_idx: Option<usize> = None;
                            for (i, term) in self.excluded_terms.iter().enumerate() {
                                if ui.add(Button::new(format!("{} ✖", term)).small()).clicked() { remove_idx = Some(i); }
                            }
                            if let Some(i) = remove_idx { 
                                self.excluded_terms.remove(i);
                                // Re-apply exclusion after removal (cannot restore already removed rows)
                                if !self.excluded_terms.is_empty() {
                                    let terms = self.excluded_terms.clone();
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        let lp = r.path.to_ascii_lowercase();
                                        !terms.iter().any(|t| lp.contains(t))
                                    });
                                }
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);
                        ui.label(RichText::new("Size Filters (KB)").italics());
                        ui.horizontal(|ui| {
                            ui.label("Min:");
                            let mut min_kb: i64 = self.viewer.ui_settings.db_min_size_bytes
                                .map(|b| (b / 1024) as i64)
                                .unwrap_or(0);
                            let mut max_kb: i64 = self.viewer.ui_settings.db_max_size_bytes
                                .map(|b| (b / 1024) as i64)
                                .unwrap_or(0);
                            let min_resp = egui::DragValue::new(&mut min_kb).speed(1).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Minimum file size in KiB");
                            ui.label("Max:");
                            let max_resp = egui::DragValue::new(&mut max_kb).speed(10).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Maximum file size in KiB (0 = no max)");
                            let changed = min_resp.changed() || max_resp.changed();
                            if changed || ui.button("Apply").clicked() {
                                let min_b = if min_kb <= 0 { None } else { Some(min_kb as u64 * 1024) };
                                let max_b = if max_kb <= 0 { None } else { Some(max_kb as u64 * 1024) };
                                // Enforce min <= max if both set
                                let (min_b, max_b) = match (min_b, max_b) {
                                    (Some(a), Some(b)) if a > b => (Some(b), Some(a)),
                                    other => other,
                                };
                                self.viewer.ui_settings.db_min_size_bytes = min_b;
                                self.viewer.ui_settings.db_max_size_bytes = max_b;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                                // Apply to current table view
                                let minb = self.viewer.ui_settings.db_min_size_bytes;
                                let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                self.table.retain(|r| {
                                    if r.file_type == "<DIR>" { return true; }
                                    let ok_min = minb.map(|m| r.size >= m).unwrap_or(true);
                                    let ok_max = maxb.map(|m| r.size <= m).unwrap_or(true);
                                    ok_min && ok_max
                                });
                            }
                            if ui.button("Clear").on_hover_text("Clear size constraints").clicked() {
                                self.min_size_mb_input.clear();
                                self.max_size_mb_input.clear();
                                self.viewer.ui_settings.db_min_size_bytes = None;
                                self.viewer.ui_settings.db_max_size_bytes = None;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);
                        ui.label(RichText::new("Extension filters").italics());
                        ui.horizontal(|ui| {
                            let changed_i = ui.checkbox(&mut self.viewer.types_show_images, "Images").changed();
                            let changed_v = ui.checkbox(&mut self.viewer.types_show_videos, "Videos").changed();
                            let changed_d = ui.checkbox(&mut self.viewer.types_show_dirs, "Folders").changed();
                            if changed_i || changed_v || changed_d { self.active_filter_group = None; }
                        });
                        ui.horizontal(|ui| {
                            let mut skip_icons = self.viewer.ui_settings.filter_skip_icons;
                            if ui.checkbox(&mut skip_icons, "Skip likely icons").on_hover_text("Filter out tiny images (.ico, <= 16–64px, and small files)").changed() {
                                self.viewer.ui_settings.filter_skip_icons = skip_icons;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                                if skip_icons {
                                    // Apply to current table using same heuristic as scanner
                                    let tiny_thresh = self.viewer.ui_settings.db_min_size_bytes.unwrap_or(10 * 1024);
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        // ico extension
                                        let is_ico = std::path::Path::new(&r.path)
                                            .extension()
                                            .and_then(|e| e.to_str())
                                            .map(|s| s.eq_ignore_ascii_case("ico"))
                                            .unwrap_or(false);
                                        if is_ico { return false; }
                                        // Evaluate size (use row.size, fall back to fs)
                                        let mut size_val = r.size;
                                        if size_val == 0 { if let Ok(md) = std::fs::metadata(&r.path) { size_val = md.len(); } }
                                        if size_val > tiny_thresh { return true; }
                                        // Only images: check tiny dims
                                        let is_img = std::path::Path::new(&r.path)
                                            .extension()
                                            .and_then(|e| e.to_str())
                                            .map(|s| s.to_ascii_lowercase())
                                            .map(|ext| crate::is_image(ext.as_str()))
                                            .unwrap_or(false);
                                        if !is_img { return true; }
                                        let tiny_dims = image::ImageReader::open(&r.path)
                                            .ok()
                                            .and_then(|r| r.with_guessed_format().ok())
                                            .and_then(|r| r.into_dimensions().ok())
                                            .map(|(w,h)| w <= 64 && h <= 64)
                                            .unwrap_or(false);
                                        !(size_val <= tiny_thresh && tiny_dims)
                                    });
                                }
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);
                        ui.label(RichText::new("Save filters as group").italics());
                        ui.horizontal(|ui| {
                            ui.add_sized([180., 22.], TextEdit::singleline(&mut self.filter_group_name_input).hint_text("Group name"));
                            let save_clicked = ui.button("Save").clicked();
                            if save_clicked && !self.filter_group_name_input.trim().is_empty() {
                                let name = self.filter_group_name_input.trim().to_string();
                                let include_images = self.viewer.types_show_images;
                                let include_videos = self.viewer.types_show_videos;
                                let include_dirs = self.viewer.types_show_dirs;
                                let skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                let minb = self.viewer.ui_settings.db_min_size_bytes;
                                let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                let excluded_terms = self.excluded_terms.clone();
                                let tx_groups = self.filter_groups_tx.clone();
                                tokio::spawn(async move {
                                    let g = crate::database::FilterGroup {
                                        id: None,
                                        name,
                                        include_images,
                                        include_videos,
                                        include_dirs,
                                        skip_icons,
                                        min_size_bytes: minb,
                                        max_size_bytes: maxb,
                                        excluded_terms,
                                        created: None,
                                        updated: None,
                                    };
                                    match crate::database::save_filter_group(g).await {
                                        Ok(_) => {
                                            // After saving, refresh list
                                            match crate::database::list_filter_groups().await {
                                                Ok(groups) => { let _ = tx_groups.try_send(groups); },
                                                Err(e) => log::error!("list_filter_groups after save failed: {e:?}"),
                                            }
                                        }
                                        Err(e) => log::error!("save filter group failed: {e:?}"),
                                    }
                                });
                                self.filter_group_name_input.clear();
                            }
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Saved filter groups").italics());
                            if ui.button("⟲ Refresh").clicked() {
                                let tx = self.filter_groups_tx.clone();
                                tokio::spawn(async move {
                                    match crate::database::list_filter_groups().await {
                                        Ok(groups) => { let _ = tx.try_send(groups); },
                                        Err(e) => log::error!("list_filter_groups failed: {e:?}"),
                                    }
                                });
                            }
                        });
                        if self.filter_groups.is_empty() {
                            ui.label(RichText::new("No saved groups yet.").weak());
                        } else {
                            egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                                let groups = self.filter_groups.clone();
                                for g in groups.iter() {
                                    ui.horizontal(|ui| {
                                        ui.label(&g.name);
                                        if ui.button("Apply").on_hover_text("Apply this filter group to settings and current table").clicked() {
                                            // Apply toggles and settings
                                            self.viewer.types_show_images = g.include_images;
                                            self.viewer.types_show_videos = g.include_videos;
                                            self.viewer.types_show_dirs = g.include_dirs;
                                            self.viewer.ui_settings.filter_skip_icons = g.skip_icons;
                                            self.viewer.ui_settings.db_min_size_bytes = g.min_size_bytes;
                                            self.viewer.ui_settings.db_max_size_bytes = g.max_size_bytes;
                                            crate::database::settings::save_settings(&self.viewer.ui_settings);
                                            self.excluded_terms = g.excluded_terms.clone();
                                            self.excluded_term_input.clear();
                                            self.active_filter_group = Some(g.name.clone());
                                            // Prune current table with the newly applied filters
                                            self.apply_filters_to_current_table();
                                        }
                                        if ui.button("Delete").on_hover_text("Delete this saved group").clicked() {
                                            let name = g.name.clone();
                                            let name_for_spawn = name.clone();
                                            let tx = self.filter_groups_tx.clone();
                                            let active = self.active_filter_group.clone();
                                            tokio::spawn(async move {
                                                match crate::database::delete_filter_group_by_name(&name_for_spawn).await {
                                                    Ok(_) => {
                                                        match crate::database::list_filter_groups().await {
                                                            Ok(groups) => { let _ = tx.try_send(groups); },
                                                            Err(e) => log::error!("refresh groups after delete failed: {e:?}"),
                                                        }
                                                    }
                                                    Err(e) => log::error!("delete filter group failed: {e:?}"),
                                                }
                                            });
                                            if active.as_deref() == Some(&name) { self.active_filter_group = None; }
                                        }
                                    });
                                }
                            });
                        }
                        ui.separator();
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            if ui.button("Reload Rows").on_hover_text("Reload current view rows").clicked() {
                                if self.viewer.mode == table::ExplorerMode::Database {
                                    self.db_offset = 0;
                                    self.db_last_batch_len = 0;
                                    self.load_database_rows();
                                } else {
                                    self.table.clear();
                                    self.populate_current_directory();
                                }
                            }
                            if !self.last_scan_rows.is_empty() {
                                if ui.button("Restore Last Scan").on_hover_text("Restore last recursive scan results").clicked() {
                                    self.table.clear();
                                    for r in self.last_scan_rows.clone().into_iter() {
                                        self.table.push(r);
                                    }
                                    self.viewer.showing_similarity = false;
                                    self.viewer.similar_scores.clear();
                                    self.scan_done = true;
                                    self.file_scan_progress = 1.0;
                                }
                            }
                        });
                    });
                    ui.menu_button("👁", |ui| {
                        ui.checkbox(&mut self.open_preview_pane, "Show Preview Pane");
                        ui.checkbox(&mut self.open_quick_access, "Show Quick Access");
                        ui.separator();
                        ui.checkbox(&mut self.follow_active_vision, "Follow active vision (auto-select)")
                            .on_hover_text("When enabled, the preview auto-selects the image currently being described.");
                        ui.separator();
                        // Auto-save to DB toggle (persisted)
                        let mut auto_save = self.viewer.ui_settings.auto_save_to_database;
                        if ui.checkbox(&mut auto_save, "Auto save to database").on_hover_text("When enabled, newly discovered files will be saved to the database automatically (requires an active logical group)").changed() {
                            self.viewer.ui_settings.auto_save_to_database = auto_save;
                            crate::database::settings::save_settings(&self.viewer.ui_settings);
                        }
                        if self.active_logical_group_name.is_none() {
                            ui.colored_label(Color32::YELLOW, "No active logical group. Auto save and indexing are gated.");
                        } else {
                            ui.colored_label(Color32::LIGHT_GREEN, format!("Active Group: {}", self.active_logical_group_name.clone().unwrap_or_default()));
                        }

                    });
                    TextEdit::singleline(&mut self.viewer.filter)
                    .desired_width(200.0)
                    .hint_text("Search for files")
                    .ui(ui);

                    ui.separator();

                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    
                    MenuButton::new("Scan")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
                    .ui(ui, |ui| {
                        ui.vertical_centered_justified(|ui| { 
                            ui.set_width(400.);
                            ui.heading("Recursive Scanning");
                            if Button::new("Recursive Scan").right_text("💡").ui(ui).clicked() {
                                let title = format!("Scan: {}", self.current_path);
                                let path = self.current_path.clone();
                                crate::app::OPEN_TAB_REQUESTS
                                    .lock()
                                    .unwrap()
                                    .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive: true, background: false });
                            }
                            
                            if Button::new("Re-scan Current Folder").right_text("🔄").ui(ui).on_hover_text("Shallow scan with current filters (applies 'Skip likely icons')").clicked() {
                                self.recursive_scan = false;
                                self.scan_done = false;
                                self.table.clear();
                                self.last_scan_rows.clear();
                                self.last_scan_paths.clear();
                                self.last_scan_root = Some(self.current_path.clone());
                                self.file_scan_progress = 0.0;
                                let scan_id = crate::next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = false;
                                let mut filters = crate::Filters::default();
                                filters.root = std::path::PathBuf::from(self.current_path.clone());
                                filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                                filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                                filters.include_images = self.viewer.types_show_images;
                                filters.include_videos = self.viewer.types_show_videos;
                                filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                filters.excluded_terms = self.excluded_terms.clone();
                                self.current_scan_id = Some(scan_id);
                                tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                            }
                
                            if Button::new("Cancel Scan").right_text(RichText::new("■").color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text("Cancel active recursive scan").clicked() {
                                if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
                            }
                            ui.separator();
                            ui.heading("Database Save");
                            if ui.button("Save Current View to DB").on_hover_text("Upsert all currently visible rows into the database and add them to the active logical group").clicked() {
                                if let Some(group_name) = self.active_logical_group_name.clone() {
                                    let rows: Vec<crate::database::Thumbnail> = self
                                        .table
                                        .iter()
                                        .filter(|r| r.file_type != "<DIR>")
                                        .cloned()
                                        .collect();
                                    if !rows.is_empty() {
                                        tokio::spawn(async move {
                                            // Upsert rows and collect ids
                                            match crate::database::upsert_rows_and_get_ids(rows).await {
                                                Ok(ids) => {
                                                    // Find the group and add ids
                                                    match crate::database::LogicalGroup::get_by_name(&group_name).await {
                                                        Ok(Some(g)) => {
                                                            if let Some(gid) = g.id.as_ref() {
                                                                if let Err(e) = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await {
                                                                    log::error!("Add thumbs to group failed: {e:?}");
                                                                } else {
                                                                    log::info!("Saved {} rows to DB and associated with group '{}'", ids.len(), group_name);
                                                                }
                                                            }
                                                        }
                                                        Ok(None) => log::warn!("Active group '{}' not found during save", group_name),
                                                        Err(e) => log::error!("get_by_name failed: {e:?}"),
                                                    }
                                                }
                                                Err(e) => log::error!("Upsert rows failed: {e:?}"),
                                            }
                                        });
                                    }
                                } else {
                                    log::warn!("Save skipped: no active logical group selected");
                                }
                            }
                            
                            if !self.last_scan_rows.is_empty() {
                                if Button::new("Return to Active Scan").right_text("↩").ui(ui).on_hover_text("Restore the last recursive scan results without rescanning").clicked() {
                                
                                    self.table.clear();
                                    for r in self.last_scan_rows.clone().into_iter() {
                                        self.table.push(r);
                                    }
                                    self.viewer.showing_similarity = false;
                                    self.viewer.similar_scores.clear();
                                    self.scan_done = true;
                                    self.file_scan_progress = 1.0;
                                }
                            }
                        
                            ui.separator();

                            ui.heading("Bulk Operations");
                            if Button::new("Bulk Generate Descriptions").right_text("🖩").ui(ui).on_hover_text("Generate AI descriptions for images missing caption/description").clicked() {
                                let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                let prompt = self.viewer.ui_settings.ai_prompt_template.clone();
                                let mut scheduled = 0usize;
                                for row in self.table.iter() {
                                    if self.viewer.bulk_cancel_requested { break; }
                                    if row.file_type == "<DIR>" { continue; }
                                    if let Some(ext) = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) { if !crate::is_image(ext.as_str()) { continue; } } else { continue; }
                                    // Respect size bounds
                                    if let Some(minb) = self.viewer.ui_settings.db_min_size_bytes { if row.size < minb { continue; } }
                                    if let Some(maxb) = self.viewer.ui_settings.db_max_size_bytes { if row.size > maxb { continue; } }
                                    if row.caption.is_some() || row.description.is_some() { continue; }
                                    let path_str = row.path.clone();
                                    let path_str_clone = path_str.clone();
                                    let tx_updates = self.viewer.ai_update_tx.clone();
                                    let prompt_clone = prompt.clone();
                                    let eng = engine.clone();
                                    tokio::spawn(async move {
                                        eng.stream_vision_description(std::path::Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
                                            if let Some(vd) = final_opt {
                                                let _ = tx_updates.try_send(AIUpdate::Final {
                                                    path: path_str.clone(),
                                                    description: vd.description.clone(),
                                                    caption: Some(vd.caption.clone()),
                                                    category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                                    tags: vd.tags.clone(),
                                                });
                                            } else {
                                                let _ = tx_updates.try_send(AIUpdate::Interim { path: path_str.clone(), text: interim.to_string() });
                                            }
                                        }).await;
                                    });
                                    self.vision_started += 1;
                                    self.vision_pending += 1;
                                    scheduled += 1;
                                }
                                if scheduled == 0 { log::info!("[AI] Bulk Generate: nothing to schedule"); } else { log::info!("[AI] Bulk Generate scheduled {scheduled} items"); }
                                // reset flags after start to allow next run triggers later
                                self.viewer.bulk_cancel_requested = false;
                                crate::ai::GLOBAL_AI_ENGINE.reset_bulk_cancel();
                            }
                            
                            if Button::new("Generate Missing CLIP Embeddings").right_text("⚡").ui(ui).clicked() {
                                // Collect all image paths currently visible in the table (current directory / DB page)
                                let minb = self.viewer.ui_settings.db_min_size_bytes;
                                let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                let paths: Vec<String> = self
                                    .table
                                    .iter()
                                    .filter(|r| {
                                        if r.file_type == "<DIR>" { return false; }
                                        if let Some(ext) = std::path::Path::new(&r.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                            if !crate::is_image(ext.as_str()) { return false; }
                                        } else { return false; }
                                        let ok_min = minb.map(|m| r.size >= m).unwrap_or(true);
                                        let ok_max = maxb.map(|m| r.size <= m).unwrap_or(true);
                                        ok_min && ok_max
                                    })
                                    .map(|r| r.path.clone())
                                    .collect();
                                tokio::spawn(async move {
                                    // Ensure engine and model are ready
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                    match crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&paths).await {
                                        Ok(added) => log::info!("[CLIP] Manual generation completed. Added {added} new embeddings from {} images", paths.len()),
                                        Err(e) => log::error!("[CLIP] Bulk generate failed: {e:?}")
                                    }
                                    Ok::<(), anyhow::Error>(())
                                });
                            }

                            if Button::new("Cancel Bulk Descriptions").right_text(RichText::new("■").color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text("Stop scheduling/streaming new vision descriptions").clicked() {
                                crate::ai::GLOBAL_AI_ENGINE.cancel_bulk_descriptions();
                                crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
                                self.viewer.bulk_cancel_requested = true;
                            }
                            
                        });
                    });
                    
                    MenuButton::new("Table")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style))
                    .ui(ui, |ui| {
                        ui.vertical_centered_justified(|ui| {
                            ui.selectable_value(&mut self.viewer.mode, table::ExplorerMode::Database, "Database");
                            ui.selectable_value(&mut self.viewer.mode, table::ExplorerMode::FileSystem, "FileSystem");

                            // if response.changed() {
                            //     match self.viewer.mode {
                            //         table::ExplorerMode::Database => self.load_database_rows(),
                            //         table::ExplorerMode::FileSystem => self.populate_current_directory()
                            //     }
                            // }
                            if ui.button("Reload Page").clicked() { self.load_database_rows(); }
                            if ui.button("Clear Table").clicked() { self.table.clear(); }
                        });

                        ui.add_space(4.0);
                        if matches!(self.viewer.mode, table::ExplorerMode::Database) {
                            ui.label(format!("Loaded Rows: {} (offset {})", self.table.len(), self.db_offset));
                            if self.db_loading { ui.colored_label(Color32::YELLOW, "Loading..."); }
                        }
                    });

                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    MenuButton::new("Logical Groups")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style))
                    .ui(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.set_width(470.);
                            ui.heading("Logical Groups");
                            ui.horizontal(|ui| {
                                if let Some(name) = &self.active_logical_group_name {
                                    ui.label(format!("Active: {}", name));
                                } else {
                                    ui.label("Active: (none)");
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("⟲ Refresh").clicked() {
                                        let tx = self.logical_groups_tx.clone();
                                        tokio::spawn(async move {
                                            match crate::database::LogicalGroup::list_all().await {
                                                Ok(groups) => { let _ = tx.try_send(groups); },
                                                Err(e) => log::error!("refresh logical groups failed: {e:?}"),
                                            }
                                        });
                                    }
                                });
                            });
                            ui.separator();
                            ui.label(RichText::new("Operations").italics());
                            ui.horizontal(|ui| {
                                ui.label("Add selection to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_target.trim().to_string();
                                        if !target.is_empty() && !self.selected.is_empty() {
                                            // Gather selected rows and upsert to get ids
                                            let rows: Vec<crate::database::Thumbnail> = self
                                                .table
                                                .iter()
                                                .filter(|r| r.file_type != "<DIR>" && self.selected.contains(&r.path))
                                                .cloned()
                                                .collect();
                                            if !rows.is_empty() {
                                                tokio::spawn(async move {
                                                    // Ensure rows exist in DB and get their ids
                                                    match crate::database::upsert_rows_and_get_ids(rows).await {
                                                        Ok(ids) => {
                                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                if let Some(gid) = g.id.as_ref() {
                                                                    let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                                }
                                                            } else {
                                                                if let Ok(_) = crate::database::LogicalGroup::create(&target).await {
                                                                    if let Ok(Some(g2)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                        if let Some(gid) = g2.id.as_ref() { let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await; }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Err(e) => log::error!("Add selection: upsert failed: {e:?}"),
                                                    }
                                                });
                                            }
                                        }
                                    }
                                });
                            });

                            ui.add_space(5.);

                            ui.horizontal(|ui| {
                                ui.label("Add current view to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_view_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_view_target.trim().to_string();
                                        if !target.is_empty() {
                                            let rows: Vec<crate::database::Thumbnail> = self
                                            .table
                                            .iter()
                                            .filter(|r| r.file_type != "<DIR>")
                                            .cloned()
                                            .collect();

                                            if !rows.is_empty() {
                                                tokio::spawn(async move {
                                                    match crate::database::upsert_rows_and_get_ids(rows).await {
                                                        Ok(ids) => {
                                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                if let Some(gid) = g.id.as_ref() {
                                                                    let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                                }
                                                            } else {
                                                                if let Ok(_) = crate::database::LogicalGroup::create(&target).await {
                                                                    if let Ok(Some(g2)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                        if let Some(gid) = g2.id.as_ref() { let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await; }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Err(e) => log::error!("Add view: upsert failed: {e:?}"),
                                                    }
                                                });
                                            }
                                        }
                                    }
                                });
                            });

                            ui.add_space(5.);

                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Copy From: "));
                                TextEdit::singleline(&mut self.group_copy_src).hint_text("Group to Copy From").desired_width(150.).ui(ui);
                                ui.label(RichText::new(" -> "));
                                TextEdit::singleline(&mut self.group_copy_dst).hint_text("Destination group").desired_width(150.).ui(ui);

                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("Copy").clicked() {
                                        let src = self.group_copy_src.trim().to_string();
                                        let dst = self.group_copy_dst.trim().to_string();
                                        if !src.is_empty() && !dst.is_empty() && src != dst {
                                            tokio::spawn(async move {
                                                if let Ok(Some(src_g)) = crate::database::LogicalGroup::get_by_name(&src).await {
                                                    let ids = src_g.thumbnails.clone();
                                                    if !ids.is_empty() {
                                                        // Ensure destination exists
                                                        let dst_gid = match crate::database::LogicalGroup::get_by_name(&dst).await {
                                                            Ok(Some(g)) => g.id,
                                                            _ => {
                                                                let _ = crate::database::LogicalGroup::create(&dst).await;
                                                                crate::database::LogicalGroup::get_by_name(&dst).await.ok().flatten().and_then(|g| g.id)
                                                            }
                                                        };
                                                        if let Some(gid) = dst_gid.as_ref() {
                                                            let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                        }
                                                    }
                                                }
                                            });
                                        }
                                    }
                                });
                            });
                            
                            ui.add_space(5.);
                            
                            ui.horizontal(|ui| {
                                ui.label("Add unassigned to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_unassigned_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_unassigned_target.trim().to_string();
                                        if !target.is_empty() {
                                            tokio::spawn(async move {
                                                match crate::database::LogicalGroup::list_unassigned_thumbnail_ids().await {
                                                    Ok(ids) => {
                                                        log::info!("Thumbnails without a group: {}", ids.len());
                                                        if !ids.is_empty() {
                                                            // ensure destination group exists
                                                            let gid_opt = match crate::database::LogicalGroup::get_by_name(&target).await {
                                                                Ok(Some(g)) => g.id,
                                                                _ => {
                                                                    let _ = crate::database::LogicalGroup::create(&target).await;
                                                                    crate::database::LogicalGroup::get_by_name(&target).await.ok().flatten().and_then(|g| g.id)
                                                                }
                                                            };
                                                            if let Some(gid) = gid_opt.as_ref() {
                                                                let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => log::error!("List unassigned failed: {e:?}"),
                                                }
                                            });
                                        }
                                    }
                                    if let Some(gname) = &self.active_logical_group_name {
                                        if ui.button("To Current Group").on_hover_text("Add all unassigned thumbnails to the active group").clicked() {
                                            let target = gname.clone();
                                            tokio::spawn(async move {
                                                match crate::database::LogicalGroup::list_unassigned_thumbnail_ids().await {
                                                    Ok(ids) => {
                                                        if ids.is_empty() { return; }
                                                        // Ensure group exists then add
                                                        let gid_opt = match crate::database::LogicalGroup::get_by_name(&target).await {
                                                            Ok(Some(g)) => g.id,
                                                            _ => {
                                                                let _ = crate::database::LogicalGroup::create(&target).await;
                                                                crate::database::LogicalGroup::get_by_name(&target).await.ok().flatten().and_then(|g| g.id)
                                                            }
                                                        };
                                                        if let Some(gid) = gid_opt.as_ref() {
                                                            if let Err(e) = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await {
                                                                log::error!("Add unassigned to current group failed: {e:?}");
                                                            }
                                                        }
                                                    }
                                                    Err(e) => log::error!("List unassigned failed: {e:?}"),
                                                }
                                            });
                                        }
                                    }
                                });
                            });

                            ui.separator();

                            ui.label(RichText::new("Create new group").italics());
                            
                            ui.horizontal(|ui| {
                                let response = TextEdit::singleline(&mut self.group_create_name_input).hint_text("Group name").desired_width(150.).ui(ui);
                                let btn = ui.with_layout(Layout::right_to_left(Align::Center), |ui| ui.button("Create")).inner;
                                if ( response.lost_focus() && response.changed() ) || btn.clicked() {
                                    let name = self.group_create_name_input.trim().to_string();
                                    if !name.is_empty() {
                                        let tx = self.logical_groups_tx.clone();
                                        tokio::spawn(async move {
                                            match crate::database::LogicalGroup::create(&name).await {
                                                Ok(_) => {
                                                    match crate::database::LogicalGroup::list_all().await {
                                                        Ok(groups) => { let _ = tx.try_send(groups); },
                                                        Err(e) => log::error!("list groups after create failed: {e:?}"),
                                                    }
                                                }
                                                Err(e) => log::error!("create group failed: {e:?}"),
                                            }
                                        });
                                        self.group_create_name_input.clear();
                                    }
                                }
                            });
                            ui.separator();
                            if self.logical_groups.is_empty() {
                                ui.label(RichText::new("No groups defined yet.").weak());
                            } else {
                                egui::ScrollArea::vertical().max_height(180.).show(ui, |ui| {
                                    for g in self.logical_groups.clone().into_iter() {
                                        ui.horizontal(|ui| {
                                            // Name or rename editor
                                            if self.group_rename_target.as_deref() == Some(g.name.as_str()) {
                                                ui.add_sized([180., 22.], TextEdit::singleline(&mut self.group_rename_input));
                                                if ui.button("Save").on_hover_text("Rename group").clicked() {
                                                    if let Some(gid) = g.id.clone() {
                                                        let new_name = self.group_rename_input.trim().to_string();
                                                        if !new_name.is_empty() {
                                                            let tx = self.logical_groups_tx.clone();
                                                            tokio::spawn(async move {
                                                                match crate::database::LogicalGroup::rename(&gid, &new_name).await {
                                                                    Ok(_) => {
                                                                        match crate::database::LogicalGroup::list_all().await {
                                                                            Ok(groups) => { let _ = tx.try_send(groups); },
                                                                            Err(e) => log::error!("list groups after rename failed: {e:?}"),
                                                                        }
                                                                    }
                                                                    Err(e) => log::error!("rename group failed: {e:?}"),
                                                                }
                                                            });
                                                        }
                                                    }
                                                    self.group_rename_target = None;
                                                    self.group_rename_input.clear();
                                                }
                                                if ui.button("Cancel").clicked() {
                                                    self.group_rename_target = None;
                                                    self.group_rename_input.clear();
                                                }
                                            } else {
                                                ui.label(&g.name);
                                            }

                                            if ui.button("Select").on_hover_text("Use this as the active group (no mode switch)").clicked() {
                                                self.active_logical_group_name = Some(g.name.clone());
                                            }
                                            if ui.button("Open").on_hover_text("Load this group's thumbnails").clicked() {
                                                self.active_logical_group_name = Some(g.name.clone());
                                                self.viewer.mode = table::ExplorerMode::Database;
                                                self.table.clear();
                                                self.db_offset = 0;
                                                self.db_last_batch_len = 0;
                                                self.db_loading = true;
                                                self.load_logical_group_by_name(g.name.clone());
                                            }
                                            if self.group_rename_target.as_deref() != Some(g.name.as_str()) {
                                                if ui.button("Rename").clicked() {
                                                    self.group_rename_target = Some(g.name.clone());
                                                    self.group_rename_input = g.name.clone();
                                                }
                                            }
                                            if ui.button("Delete").on_hover_text("Delete this group").clicked() {
                                                if let Some(gid) = g.id.clone() {
                                                    let tx = self.logical_groups_tx.clone();
                                                    let active = self.active_logical_group_name.clone();
                                                    let name = g.name.clone();
                                                    tokio::spawn(async move {
                                                        match crate::database::LogicalGroup::delete(&gid).await {
                                                            Ok(_) => {
                                                                match crate::database::LogicalGroup::list_all().await {
                                                                    Ok(groups) => { let _ = tx.try_send(groups); },
                                                                    Err(e) => log::error!("list groups after delete failed: {e:?}"),
                                                                }
                                                            }
                                                            Err(e) => log::error!("delete group failed: {e:?}"),
                                                        }
                                                    });
                                                    if active.as_deref() == Some(name.as_str()) {
                                                        self.active_logical_group_name = None;
                                                    }
                                                }
                                            }
                                        });
                                    }
                                });
                            }
                        });
                    });
                });
            });
        });

        TopBottomPanel::bottom("FileExplorer Bottom Panel")
        .exact_height(22.)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if self.file_scan_progress > 0.0 {
                    let mut bar = ProgressBar::new(self.file_scan_progress)
                        .animate(true)
                        .desired_width(100.)
                        .show_percentage();

                    if self.scan_done {
                        self.file_scan_progress = 0.;
                        bar = bar.text(RichText::new("Scan Complete").color(Color32::LIGHT_GREEN));
                    }
                    bar.ui(ui);
                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                }

                // AI status & progress
                // Determine AI readiness via index worker activity (vision model removed)
                let ai_ready = {
                    use std::sync::atomic::Ordering;
                    let queued = crate::ai::GLOBAL_AI_ENGINE
                        .index_queue_len
                        .load(Ordering::Relaxed);
                    let active = crate::ai::GLOBAL_AI_ENGINE
                        .index_active
                        .load(Ordering::Relaxed);
                    let completed = crate::ai::GLOBAL_AI_ENGINE
                        .index_completed
                        .load(Ordering::Relaxed);
                    // If we've performed any indexing or have worker activity, treat as 'ready'
                    (active + completed) > 0 || queued > 0
                };
                use std::sync::atomic::Ordering;
                let q = crate::ai::GLOBAL_AI_ENGINE
                    .index_queue_len
                    .load(Ordering::Relaxed);
                let active = crate::ai::GLOBAL_AI_ENGINE
                    .index_active
                    .load(Ordering::Relaxed);
                let completed = crate::ai::GLOBAL_AI_ENGINE
                    .index_completed
                    .load(Ordering::Relaxed);
                let total_for_ratio = q + active + completed;
                // Vision generation state
                let vision_active = !self.streaming_interim.is_empty();
                let vision_pending = self.vision_pending;
                let vision_started = self.vision_started;
                let vision_completed = self.vision_completed;
                // Build status strings
                if total_for_ratio > 0 {
                    let ratio = (completed as f32) / (total_for_ratio as f32);
                    ProgressBar::new(ratio.clamp(0.0, 1.0))
                        .desired_width(140.)
                        .show_percentage()
                        .text(format!(
                            "Index {} ({} / {})",
                            if ai_ready { "ing" } else { "load" },
                            completed,
                            total_for_ratio
                        ))
                        .ui(ui);

                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                }
                if vision_started > 0 {
                    let done_ratio = if vision_started == 0 {
                        0.0
                    } else {
                        (vision_completed as f32) / (vision_started as f32)
                    };
                    let bar = ProgressBar::new(done_ratio.clamp(0.0, 1.0))
                        .desired_width(140.)
                        .show_percentage()
                        .text(format!(
                            "Vision {} act:{} pend:{}",
                            vision_completed,
                            self.streaming_interim.len(),
                            vision_pending.saturating_sub(self.streaming_interim.len())
                        ));
                    bar.ui(ui);
                } else if vision_active || vision_pending > 0 {
                    ui.label("Vision: starting...");
                } else {
                    ui.label("AI: Idle");
                }

                ui.add_space(5.);
                ui.separator();
                ui.add_space(5.);
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
                let vram01 = smoothed_vram01();
                let (v_used, v_total) = gpu_mem_mb().unwrap_or_default();
                ui.label(format!("VRAM: {:.0}/{:.0} MiB", v_used, v_total)); 
                ProgressBar::new(vram01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);

                ui.separator();

                // System metrics (CPU, RAM, VRAM)
                let cpu01 = smoothed_cpu01();
                ui.label(format!("CPU: {:.2}%", cpu01 * 100.0));
                ProgressBar::new(cpu01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);

                ui.separator(); 

                let ram01 = smoothed_ram01();
                if let Some((used_mb, total_mb)) = system_mem_mb() {
                    ui.label(format!("RAM: {:.0}/{:.0} MiB", used_mb, total_mb));
                } else {
                    ui.label("RAM: n/a");
                }
                ProgressBar::new(ram01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);
                    
                ui.separator(); 

                ui.add_space(10.);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let mut img_cnt = 0usize;
                    let mut vid_cnt = 0usize;
                    let mut dir_cnt = 0usize;
                    let mut total_size = 0u64;
                    for r in self.table.iter() {
                        if r.file_type == "<DIR>" {
                            dir_cnt += 1;
                            continue;
                        }
                        if let Some(ext) = std::path::Path::new(&r.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                        {
                            if crate::is_image(ext.as_str()) {
                                img_cnt += 1;
                            }
                            if crate::is_video(ext.as_str()) {
                                vid_cnt += 1;
                            }
                        }
                        total_size += r.size;
                    }
                    ui.label(format!("Dirs: {dir_cnt}"));
                    ui.separator();
                    ui.label(format!("Images: {img_cnt}"));
                    ui.separator();
                    ui.label(format!("Videos: {vid_cnt}"));
                    ui.separator();
                    ui.label(format!(
                        "Total Size: {}",
                        humansize::format_size(total_size, DECIMAL)
                    ));
                });
            });
        });

        CentralPanel::default().show_inside(ui, |ui| {
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
            Renderer::new(&mut self.table, &mut self.viewer)
                .with_style_modify(|s| {
                    s.single_click_edit_mode = true;
                    s.table_row_height = Some(60.0);
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
                        let resp = egui::TextEdit::singleline(&mut self.zip_password_input)
                            .password(true)
                            .hint_text("Password")
                            .desired_width(240.)
                            .ui(ui);
                        ui.horizontal(|ui| {
                            if ui.button("Submit").clicked() || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                                // Store in-memory for this session and refresh any open view for that archive
                                let pw = std::mem::take(&mut self.zip_password_input);
                                if !current.is_empty() {
                                    self.archive_passwords.insert(current.clone(), pw);
                                }
                                // If currently browsing this archive, refresh to re-attempt list with password
                                if self.current_path.starts_with(&format!("zip://{}!", current))
                                    || self.current_path.starts_with(&format!("7z://{}!", current))
                                    || self.current_path.starts_with(&format!("tar://{}!", current)) // tar typically not encrypted, but harmless
                                {
                                    self.populate_current_directory();
                                }
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
                            let rows: Vec<crate::database::Thumbnail> = self
                                .table
                                .iter()
                                .filter(|r| r.category.as_deref() == Some(cat.as_str()))
                                .cloned()
                                .collect();
                            crate::app::OPEN_TAB_REQUESTS
                                .lock()
                                .unwrap()
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: false, similar_scores: None, background: false });
                        }
                        crate::ui::file_table::table::TabAction::OpenTag(tag) => {
                            let title = format!("Tag: {}", tag);
                            let rows: Vec<crate::database::Thumbnail> = self
                                .table
                                .iter()
                                .filter(|r| r.tags.iter().any(|t| t.eq_ignore_ascii_case(&tag)))
                                .cloned()
                                .collect();
                            crate::app::OPEN_TAB_REQUESTS
                                .lock()
                                .unwrap()
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: false, similar_scores: None, background: false });
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
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: true, similar_scores: Some(self.viewer.similar_scores.clone()), background: false });
                        }
                    }
                }
            }
            // Summary inline (counts) if selection active
            if !self.selected.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.selected.len()));
            }
            if self.viewer.mode == table::ExplorerMode::Database && !self.viewer.showing_similarity {
                ui.separator();
                ui.horizontal(|ui| {
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
        });

        // Removed modal; results shown inline in table
    }

    // Load and display rows from a logical group by name
    pub fn load_logical_group_by_name(&mut self, name: String) {
        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            match crate::database::LogicalGroup::get_by_name(&name).await {
                Ok(Some(group)) => {
                    match crate::Thumbnail::fetch_by_ids(group.thumbnails.clone()).await {
                        Ok(rows) => {
                            let count = rows.len();
                            for r in rows.into_iter() { let _ = tx.try_send(r); }
                            log::info!("[Groups] Loaded '{}' with {} rows", name, count);
                        }
                        Err(e) => log::error!("Failed to fetch rows for group '{}': {e:?}", name),
                    }
                }
                Ok(None) => log::warn!("Logical group '{}' not found", name),
                Err(e) => log::error!("get_by_name '{}' failed: {e:?}", name),
            }
        });
    }

    fn apply_filters_to_current_table(&mut self) {
        // Type toggles: retain allowed types (always keep directories if include_dirs true)
        let include_images = self.viewer.types_show_images;
        let include_videos = self.viewer.types_show_videos;
        let include_dirs = self.viewer.types_show_dirs;
        let minb = self.viewer.ui_settings.db_min_size_bytes;
        let maxb = self.viewer.ui_settings.db_max_size_bytes;
        let skip_icons = self.viewer.ui_settings.filter_skip_icons;
        let excluded_terms = self.excluded_terms.clone();
        self.table.retain(|r| {
            // Dir handling
            if r.file_type == "<DIR>" { return include_dirs; }
            // Extension type
            let mut type_ok = true;
            if let Some(ext) = std::path::Path::new(&r.path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
            {
                if crate::is_image(ext.as_str()) { type_ok = include_images; }
                else if crate::is_video(ext.as_str()) { type_ok = include_videos; }
                else { type_ok = false; }
            }
            if !type_ok { return false; }
            // Size bounds
            if let Some(mn) = minb { if r.size < mn { return false; } }
            if let Some(mx) = maxb { if r.size > mx { return false; } }
            // Excluded terms
            if !excluded_terms.is_empty() {
                let lp = r.path.to_ascii_lowercase();
                if excluded_terms.iter().any(|t| lp.contains(t)) { return false; }
            }
            // Skip icons heuristic
            if skip_icons {
                let tiny_thresh = minb.unwrap_or(10 * 1024);
                // ico extension
                let is_ico = std::path::Path::new(&r.path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case("ico"))
                    .unwrap_or(false);
                if is_ico { return false; }
                // Size check
                let mut size_val = r.size;
                if size_val == 0 { if let Ok(md) = std::fs::metadata(&r.path) { size_val = md.len(); } }
                if size_val <= tiny_thresh {
                    // Only images: check dims
                    let is_img = std::path::Path::new(&r.path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                        .map(|ext| crate::is_image(ext.as_str()))
                        .unwrap_or(false);
                    if is_img {
                        let tiny_dims = image::ImageReader::open(&r.path)
                            .ok()
                            .and_then(|r| r.with_guessed_format().ok())
                            .and_then(|r| r.into_dimensions().ok())
                            .map(|(w,h)| w <= 64 && h <= 64)
                            .unwrap_or(false);
                        if tiny_dims { return false; }
                    }
                }
            }
            true
        });
    }

    pub fn load_database_rows(&mut self) {
        if self.db_loading {
            return;
        }
        // Reset similarity state when (re)loading DB pages
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.db_loading = true;
        // When offset is zero we are (re)loading fresh page; clear existing rows
        if self.db_offset == 0 {
            self.table.clear();
            self.thumb_scheduled.clear();
            self.pending_thumb_rows.clear();
        }
        let tx = self.thumbnail_tx.clone();
        let offset = self.db_offset;
    let _limit = self.db_limit; // reserved for future paging reuse
        let path = self.current_path.clone();
        tokio::spawn(async move {
            match crate::Thumbnail::get_all_thumbnails_from_directory(&path).await {
                Ok(rows) => {
                    for r in rows.iter() {
                        let _ = tx.try_send(r.clone());
                    }
                    // Send a synthetic zero-size row? Not needed; we will update state after join via channel? We'll rely on UI polling.
                    log::info!("[DB] Loaded page offset={} count={}", offset, rows.len());
                }
                Err(e) => {
                    log::error!("DB page load failed: {e}");
                }
            }
        });
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
            .fit_to_original_size(1.0)
            .max_size(vec2(ui.available_width(), 600.))
            .ui(ui) // .paint_at(ui, rect);
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