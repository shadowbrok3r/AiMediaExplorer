use std::{borrow::Cow, collections::HashMap, sync::Arc};
use crossbeam::channel::{Receiver, Sender};
use crate::{ScanEnvelope, Thumbnail};
use std::sync::{Mutex, OnceLock};
use egui::style::StyleModifier;
use chrono::{DateTime, Local};
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
pub mod navbar;
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
    pub table: egui_data_table::DataTable<Thumbnail>,
    pub viewer: FileTableViewer,
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
    selected: std::collections::HashSet<String>,
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
    // Cached WSL data to avoid recomputing every frame
    #[serde(skip)]
    cached_wsl_distros: Option<Vec<String>>,
    #[serde(skip)]
    cached_physical_drives: Option<Vec<crate::utilities::explorer::PhysicalDrive>>,
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
            current_thumb: Thumbnail::new("") ,
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
            cached_wsl_distros: Some(crate::utilities::explorer::list_wsl_distros()),
            cached_physical_drives: Some(crate::utilities::explorer::list_physical_drives()),
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

    pub fn ui(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        self.preview_pane(ui);
        self.quick_access_pane(ui);
        self.navbar(ui);

        let style = StyleModifier::default();
        style.apply(ui.style_mut());

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
            // Check if we have a single empty default row and remove it
            if self.table.len() == 1 {
                let should_remove_empty = self.table.iter().next().map(|row| {
                    row.filename.is_empty() && row.path.is_empty() && row.size == 0 && 
                    row.file_type.is_empty() && row.thumbnail_b64.is_none()
                }).unwrap_or(false);
                
                if should_remove_empty {
                    self.table.clear();
                }
            }
            
            Renderer::new(&mut self.table, &mut self.viewer)
            .with_style_modify(|s| {
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
        });
    }

    /// Number of selected file rows in the current table.
    pub fn selection_count(&self) -> usize {
        self.selected.len()
    }

    fn apply_filters_to_current_table(&mut self) {
        use crate::utilities::filtering::FiltersExt;
        // Build a Filters value from UI settings for consistent evaluation
        let mut filters = crate::Filters::default();
        filters.include_images = self.viewer.types_show_images;
        filters.include_videos = self.viewer.types_show_videos;
        // Archives are not shown in DB table rows normally; keep false
        filters.include_archives = false;
        filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
        filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
        filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
        // Date filters from UI
        filters.modified_after = self.viewer.ui_settings.filter_modified_after.clone();
        filters.modified_before = self.viewer.ui_settings.filter_modified_before.clone();
        let include_dirs = self.viewer.types_show_dirs;
        let excluded_terms = self.excluded_terms.clone();
        let have_date_filters = filters.modified_after.is_some() || filters.modified_before.is_some();

        self.table.retain(|r| {
            // Keep directories independent of FiltersExt logic
            if r.file_type == "<DIR>" { return include_dirs; }

            // Excluded terms (path/filename contains any term)
            if !excluded_terms.is_empty() {
                let lp = r.path.to_ascii_lowercase();
                if excluded_terms.iter().any(|t| lp.contains(t)) { return false; }
            }

            let path = std::path::Path::new(&r.path);
            // Quick kind check from extension to mirror scanner behavior
            let is_archive_file = path.extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .map(|e| crate::is_archive(e.as_str()))
                .unwrap_or(false);
            let kind = if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                if crate::is_image(ext.as_str()) { crate::utilities::types::MediaKind::Image }
                else if crate::is_video(ext.as_str()) { crate::utilities::types::MediaKind::Video }
                else if crate::is_archive(ext.as_str()) { crate::utilities::types::MediaKind::Archive }
                else { crate::utilities::types::MediaKind::Other }
            } else { crate::utilities::types::MediaKind::Other };
            if !filters.kind_allowed(&kind, is_archive_file) { return false; }

            // Use FiltersExt to evaluate size and skip_icons heuristics (dates are not applied to DB view here)
            let size_val = if r.size == 0 {
                std::fs::metadata(&r.path).ok().map(|m| m.len()).unwrap_or(0)
            } else { r.size };
            // Use strict UI variant: still allows small non-image files, but filters tiny icon-like images
            if !filters.skip_icons_strict_allows(path, size_val) { return false; }
            if !filters.size_ok(size_val) { return false; }
            // Apply date filter if active (use filesystem metadata; DB may not have timezone-local timestamps)
            if have_date_filters {
                let modified: Option<DateTime<Local>> = std::fs::metadata(&r.path)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|st| DateTime::<Local>::from(st));
                if !filters.date_ok(modified, None, /*recursive*/ false) { return false; }
            }
            true
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