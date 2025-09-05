use std::{borrow::Cow, collections::HashMap, path::PathBuf, sync::Arc, path::Path};
use crossbeam::channel::{Receiver, Sender};
use egui_data_table::Renderer;
use humansize::DECIMAL;
use serde::Serialize;
use viewer::FileTableViewer;
use crate::{next_scan_id, spawn_scan, Filters, ScanEnvelope, Thumbnail};
use eframe::egui::*;
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use jwalk::WalkDirGeneric;

// Temporary global size filter placeholders (to be replaced by settings persistence)
static mut MIN_SIZE_MB: Option<u64> = None; // value stored in bytes (converted from MB input)
static mut MAX_SIZE_MB: Option<u64> = None; // value stored in bytes (converted from MB input)

pub mod viewer;
pub mod codec;
pub mod preview_pane;
pub mod quick_access_pane;

// AI streaming update messages (interim + final)
#[derive(Debug, Clone)]
enum AIUpdate {
    Interim { path: String, text: String },
    Final { path: String, description: String, caption: Option<String>, category: Option<String>, tags: Vec<String> },
    SimilarResults { origin_path: String, results: Vec<crate::database::FileMetadata> },
}

#[derive(Debug, Clone)]
struct AIMetadataUpdate { path: String, description: Option<String>, caption: Option<String>, category: Option<String>, tags: Vec<String> }

#[derive(Serialize)]
pub struct FileExplorer {
    #[serde(skip)]
    table: egui_data_table::DataTable<Thumbnail>,
    viewer: FileTableViewer,
    files: Vec<Thumbnail>,
    current_path: String,
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
    ai_update_tx: Sender<AIUpdate>,
    #[serde(skip)]
    ai_update_rx: Receiver<AIUpdate>,
    #[serde(skip)]
    streaming_interim: std::collections::HashMap<String, String>,
    #[serde(skip)]
    thumb_scheduled: std::collections::HashSet<String>,
    #[serde(skip)]
    thumb_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    // Similar image search UI state
    #[serde(skip)]
    show_similar_modal: bool,
    #[serde(skip)]
    similar_origin: Option<String>,
    #[serde(skip)]
    similar_results: Vec<crate::database::FileMetadata>,
    // Auto-follow active vision generation updates
    #[serde(skip)]
    pub follow_active_vision: bool,
    // Cache of clip embedding presence per path to avoid frequent async queries in UI thread
    #[serde(skip)]
    clip_presence: std::collections::HashMap<String, bool>,
    // Channel for async metadata merges
    #[serde(skip)]
    meta_tx: Sender<AIMetadataUpdate>,
    #[serde(skip)]
    meta_rx: Receiver<AIMetadataUpdate>,
        // Vision description generation tracking (bulk vision progress separate from indexing)
        #[serde(skip)]
        vision_started: usize,
        #[serde(skip)]
        vision_completed: usize,
        #[serde(skip)]
        vision_pending: usize, // scheduled but not yet final
}

impl Default for FileExplorer {
    fn default() -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
        let (ai_update_tx, ai_update_rx) = crossbeam::channel::unbounded();
        let (meta_tx, meta_rx) = crossbeam::channel::unbounded();
        let current_path = directories::UserDirs::new().unwrap().picture_dir().unwrap().to_string_lossy().to_string();
        let mut this = Self {
            table: Default::default(), 
            viewer: FileTableViewer::new(thumbnail_tx.clone()), 
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
            thumbnail_tx, thumbnail_rx,
            scan_tx, scan_rx,
            back_stack: Vec::new(),
            forward_stack: Vec::new(),
            pending_thumb_rows: Vec::new(),
            selected: std::collections::HashSet::new(),
            ai_update_tx, ai_update_rx,
            streaming_interim: std::collections::HashMap::new(),
            thumb_scheduled: std::collections::HashSet::new(),
            thumb_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(6)),
            show_similar_modal: false,
            similar_origin: None,
            similar_results: Vec::new(),
            follow_active_vision: true,
            clip_presence: std::collections::HashMap::new(),
            meta_tx, meta_rx,
            vision_started: 0,
            vision_completed: 0,
            vision_pending: 0,
        };
        // Initial shallow directory population (non-recursive)
        this.populate_current_directory();
        this
    }
}

impl FileExplorer {
    pub fn ui(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        self.preview_pane(ui);
        self.quick_access_pane(ui);

        TopBottomPanel::top("FileExplorerTopPanel").exact_height(25.).show_inside(ui, |ui| {
            ui.horizontal_top(|ui| {
                let fs_mode = self.viewer.mode == viewer::ExplorerMode::FileSystem;
                if ui.add_enabled(fs_mode, Button::new("⬅")).clicked() { self.nav_back(); }
                if ui.add_enabled(fs_mode, Button::new("⬆")).clicked() { self.nav_up(); }
                if ui.add_enabled(fs_mode, Button::new("➡")).clicked() { self.nav_forward(); }
                if ui.add_enabled(fs_mode, Button::new("⟲")).clicked() { self.refresh(); }
                if ui.add_enabled(fs_mode, Button::new("🏠")).clicked() { self.nav_home(); }
                ui.separator();
                egui::ComboBox::from_label("Mode")
                    .selected_text(match self.viewer.mode { viewer::ExplorerMode::FileSystem => "File Explorer", viewer::ExplorerMode::Database => "Database Explorer" })
                    .show_ui(ui, |ui| {
                        let prev = self.viewer.mode;
                        if ui.selectable_value(&mut self.viewer.mode, viewer::ExplorerMode::FileSystem, "File Explorer").clicked() && prev != viewer::ExplorerMode::FileSystem {
                            self.populate_current_directory();
                        }
                        if ui.selectable_value(&mut self.viewer.mode, viewer::ExplorerMode::Database, "Database Explorer").clicked() && prev != viewer::ExplorerMode::Database {
                            self.load_database_rows();
                        }
                    });
                if self.viewer.mode == viewer::ExplorerMode::Database {
                    if ui.button("🔄 Reload DB").clicked() { self.load_database_rows(); }
                }
                
                // Breadcrumbs (avoid borrow conflicts by cloning path first)
                let current_path_clone = self.current_path.clone();
                let parts: Vec<String> = current_path_clone.split(['\\','/']).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
                let root_has_slash = current_path_clone.starts_with('/');
                let mut accum = if root_has_slash { String::from("/") } else { String::new() };
                ui.horizontal(|ui| {
                    for (i, part) in parts.iter().enumerate() {
                        if !accum.ends_with(std::path::MAIN_SEPARATOR) && !accum.is_empty() { accum.push(std::path::MAIN_SEPARATOR); }
                        accum.push_str(part);
                        let display = if part.is_empty() { std::path::MAIN_SEPARATOR.to_string() } else { part.clone() };
                        if ui.selectable_label(false, RichText::new(display).underline()).clicked() {
                            self.push_history(accum.clone());
                            self.populate_current_directory();
                        }
                        if i < parts.len()-1 { ui.label(RichText::new("›").weak()); }
                        // Remove trailing segment for next iteration accumulation clone safety
                    }
                });
                TextEdit::singleline(&mut self.current_path).hint_text("Current Directory").desired_width(300.).ui(ui);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.menu_button("⚙", |ui| {
                        ui.menu_button("Scan", |ui| {
                            // Size filter inputs (temporary simple fields in bytes; could be enhanced with suffix parsing later)
                            ui.horizontal(|ui| {
                                ui.label("Min Size (MB):");
                                static mut MIN_SIZE_MB: Option<u64> = None; // unsafe avoided elsewhere; quick placeholder (will refactor to settings struct)
                                static mut MAX_SIZE_MB: Option<u64> = None;
                                let mut min_txt = unsafe { MIN_SIZE_MB.map(|v| (v/1_000_000).to_string()).unwrap_or_default() };
                                let mut max_txt = unsafe { MAX_SIZE_MB.map(|v| (v/1_000_000).to_string()).unwrap_or_default() };
                                if ui.add(TextEdit::singleline(&mut min_txt).desired_width(60.)).lost_focus() { /* no-op */ }
                                if ui.add(TextEdit::singleline(&mut max_txt).desired_width(60.)).lost_focus() { /* no-op */ }
                                if ui.button("✔ Apply Size").clicked() {
                                    unsafe {
                                        MIN_SIZE_MB = min_txt.trim().parse::<u64>().ok().map(|m| m*1_000_000);
                                        MAX_SIZE_MB = max_txt.trim().parse::<u64>().ok().map(|m| m*1_000_000);
                                    }
                                }
                            });
                            if ui.button("💡 Recursive Scan").clicked() {
                                self.recursive_scan = true;
                                self.scan_done = false; // reset completion flag
                                self.table.clear();
                                self.file_scan_progress = 0.0;
                                let scan_id = next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = self.recursive_scan.clone();
                                let mut filters = Filters::default();
                                filters.root = PathBuf::from(self.current_path.clone());
                                // Apply temporary size filters (if set)
                                unsafe {
                                    extern crate core; // silence unused warning potential
                                    filters.min_size_bytes = MIN_SIZE_MB;
                                    filters.max_size_bytes = MAX_SIZE_MB;
                                }
                                // Attach excluded terms list (already lowercase)
                                filters.excluded_terms = self.excluded_terms.clone();
                                tokio::spawn(async move {
                                    spawn_scan(
                                        filters,
                                        tx,
                                        recurse,
                                        scan_id
                                    ).await;
                                });
                            }
                            if ui.button("🖩 Bulk Generate").clicked() {
                                // Bulk generate AI descriptions for images without description.
                                let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                // Pull prompt template from current loaded settings (fallback to default if not yet loaded)
                                let prompt = crate::database::settings::load_settings().ai_prompt_template;
                                let mut scheduled = 0usize;
                                for row in self.table.iter() {
                                    if row.file_type == "<DIR>" { continue; }
                                    // crude image extension test
                                    if let Some(ext) = Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                        let is_img = crate::is_image(ext.as_str());
                                        if !is_img { continue; }
                                    } else { continue; }
                                    if row.caption.is_some() || row.description.is_some() { continue; }
                                    let path_str = row.path.clone();
                                    let path_str_clone = path_str.clone();
                                    let tx_updates = self.ai_update_tx.clone();
                                    let prompt_clone = prompt.clone();
                                    let eng = engine.clone();
                                    tokio::spawn(async move {
                                        eng.stream_vision_description(Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
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
                                    // Update vision generation counters
                                    self.vision_started += 1;
                                    self.vision_pending += 1;
                                    scheduled += 1;
                                }
                                if scheduled == 0 { log::info!("[AI] Bulk Generate: nothing to schedule"); } else { log::info!("[AI] Bulk Generate scheduled {scheduled} items"); }
                            }
                            if ui.button("🗙 Cancel Scan").clicked() { /* TODO cancel scan token */ }
                        });
                        ui.menu_button("View", |ui| {
                            ui.checkbox(&mut self.open_preview_pane, "Show Preview");
                            ui.checkbox(&mut self.open_quick_access, "Show Quick Access");
                            if ui.button("Group by Category").clicked() { /* TODO group feature */ }
                        });
                        ui.menu_button("Filters", |ui| {
                            ui.label(RichText::new("Excluded Terms").strong());
                            ui.label("(substring match, case-insensitive)");
                            ui.horizontal(|ui| {
                                let resp = TextEdit::singleline(&mut self.excluded_term_input)
                                    .hint_text("term")
                                    .desired_width(120.)
                                    .ui(ui);
                                let add_clicked = ui.button("Add").clicked();
                                if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || add_clicked {
                                    let term = self.excluded_term_input.trim().to_ascii_lowercase();
                                    if !term.is_empty() && !self.excluded_terms.iter().any(|t| t == &term) { self.excluded_terms.push(term); }
                                    self.excluded_term_input.clear();
                                }
                                if ui.button("Clear All").clicked() { self.excluded_terms.clear(); }
                            });
                            ui.horizontal_wrapped(|ui| {
                                let mut remove_idx: Option<usize> = None;
                                for (i, term) in self.excluded_terms.iter().enumerate() {
                                    if ui.add(Button::new(format!("{} ✕", term)).small()).clicked() { remove_idx = Some(i); }
                                }
                                if let Some(i) = remove_idx { self.excluded_terms.remove(i); }
                            });
                        });
                        ui.menu_button("Database", |ui| {
                            if ui.button("Reload Thumbnails Table").clicked() { self.load_database_rows(); }
                            if ui.button("Clear Table View").clicked() { self.table.clear(); }
                            if ui.button("Switch To DB Mode").clicked() { if self.viewer.mode != viewer::ExplorerMode::Database { self.viewer.mode = viewer::ExplorerMode::Database; self.load_database_rows(); } }
                            if ui.button("Switch To FS Mode").clicked() { if self.viewer.mode != viewer::ExplorerMode::FileSystem { self.viewer.mode = viewer::ExplorerMode::FileSystem; self.populate_current_directory(); } }
                        });
                    });
                    TextEdit::singleline(&mut self.viewer.filter).desired_width(200.0).hint_text("Search for files").ui(ui);
                });
            });
        });

        TopBottomPanel::bottom("FileExplorer Bottom Panel")
            .default_height(25.)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.file_scan_progress > 0.0 {
                        let mut bar = ProgressBar::new(self.file_scan_progress)
                            .animate(true)
                            .desired_width(200.)
                            .show_percentage();
                        if self.scan_done { bar = bar.text(RichText::new("Scan Complete").color(Color32::LIGHT_GREEN)); }
                        bar.ui(ui);
                    }
                    // AI status & progress
                    // Determine AI readiness via index worker activity (vision model removed)
                    let ai_ready = {
                        use std::sync::atomic::Ordering;
                        let queued = crate::ai::GLOBAL_AI_ENGINE.index_queue_len.load(Ordering::Relaxed);
                        let active = crate::ai::GLOBAL_AI_ENGINE.index_active.load(Ordering::Relaxed);
                        let completed = crate::ai::GLOBAL_AI_ENGINE.index_completed.load(Ordering::Relaxed);
                        // If we've performed any indexing or have worker activity, treat as 'ready'
                        (active + completed) > 0 || queued > 0
                    };
                    use std::sync::atomic::Ordering;
                    let q = crate::ai::GLOBAL_AI_ENGINE.index_queue_len.load(Ordering::Relaxed);
                    let active = crate::ai::GLOBAL_AI_ENGINE.index_active.load(Ordering::Relaxed);
                    let completed = crate::ai::GLOBAL_AI_ENGINE.index_completed.load(Ordering::Relaxed);
                    let total_for_ratio = q + active + completed;
                    // Vision generation state
                    let vision_active = !self.streaming_interim.is_empty();
                    let vision_pending = self.vision_pending;
                    let vision_started = self.vision_started;
                    let vision_completed = self.vision_completed;
                    // Build status strings
                    if total_for_ratio > 0 {
                        let ratio = (completed as f32) / (total_for_ratio as f32);
                        ProgressBar::new(ratio.clamp(0.0,1.0))
                            .desired_width(140.)
                            .show_percentage()
                            .text(format!("Index {} ({} / {})", if ai_ready {"ing"} else {"load"}, completed, total_for_ratio))
                            .ui(ui);
                    }
                    if vision_started > 0 {
                        let done_ratio = if vision_started == 0 {0.0} else {(vision_completed as f32)/(vision_started as f32)};
                        let bar = ProgressBar::new(done_ratio.clamp(0.0,1.0))
                            .desired_width(140.)
                            .show_percentage()
                            .text(format!("Vision {} act:{} pend:{}", vision_completed, self.streaming_interim.len(), vision_pending.saturating_sub(self.streaming_interim.len())));
                        bar.ui(ui);
                    } else if vision_active || vision_pending > 0 {
                        ui.label("Vision: starting...");
                    } else if !ai_ready {
                        ui.label("AI: Loading");
                    } else {
                        ui.label("AI: Idle");
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {                
                        let mut img_cnt=0usize; let mut vid_cnt=0usize; let mut dir_cnt=0usize; let mut total_size=0u64;
                        for r in self.table.iter() {
                            if r.file_type == "<DIR>" { dir_cnt+=1; continue; }
                            if let Some(ext) = std::path::Path::new(&r.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                if crate::is_image(ext.as_str()) { img_cnt+=1; }
                                if crate::is_video(ext.as_str()) { vid_cnt+=1; }
                            }
                            total_size += r.size;
                        }
                        ui.label(format!("Dirs: {dir_cnt}"));
                        ui.separator();
                        ui.label(format!("Images: {img_cnt}"));
                        ui.separator();
                        ui.label(format!("Videos: {vid_cnt}"));
                        ui.separator();
                        ui.label(format!("Total Size: {}", humansize::format_size(total_size, DECIMAL)));
                    });
                });
            });

        CentralPanel::default().show_inside(ui, |ui| {
            Renderer::new(
                &mut self.table, 
                &mut self.viewer
            ).with_style_modify(|s| {
                s.single_click_edit_mode = true;
                s.table_row_height = Some(50.0);
                s.auto_shrink = [false, false].into();
            }).ui(ui);
            // Summary inline (counts) if selection active
            if !self.selected.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.selected.len()));
            }
        });

        // Similar images modal (rendered each frame if active)
        if self.show_similar_modal {
            let mut open_flag = true; // represent window open state locally
            let title = if let Some(orig) = &self.similar_origin { format!("Similar To: {}", std::path::Path::new(orig).file_name().and_then(|n| n.to_str()).unwrap_or(orig)) } else { "Similar Images".to_string() };
            egui::Window::new(title)
                .open(&mut open_flag)
                .resizable(true)
                .vscroll(true)
                .show(ui.ctx(), |ui| {
                    if ui.button("Close").clicked() { /* set after closure */ }
                    ui.separator();
                    if self.similar_results.is_empty() {
                        ui.label("Searching / No results yet...");
                    } else {
                        ui.horizontal(|ui| { ui.label(format!("{} results", self.similar_results.len())); });
                        ui.separator();
                        for meta in self.similar_results.iter() {
                            ui.horizontal(|ui| {
                                if ui.button("Open").clicked() { let _ = open::that(meta.path.clone()); }
                                ui.label(RichText::new(&meta.filename).strong());
                                if let Some(score) = meta.clip_similarity_score.or(meta.similarity_score) {
                                    ui.label(format!("score: {:.4}", score));
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("Preview").clicked() {
                                        // Attempt to load thumbnail row for this path into current_thumb (FileSystem mode only currently)
                                        if let Some(row) = self.table.iter().find(|r| r.path == meta.path) {
                                            self.current_thumb = row.clone();
                                            self.open_preview_pane = true;
                                        }
                                    }
                                });
                            });
                        }
                    }
                });
            if !open_flag { self.show_similar_modal = false; }
        }
    }

    pub fn set_path(&mut self, path: impl Into<String>) {
        self.current_path = path.into();
        self.populate_current_directory();
    }

    pub fn receive(&mut self, ctx: &Context) {
        while let Ok(thumbnail) =  self.thumbnail_rx.try_recv() {
            ctx.request_repaint();
            if self.viewer.mode == viewer::ExplorerMode::FileSystem {
                if thumbnail.file_type == "<DIR>" {
                    // Navigate into directory
                    self.current_path = thumbnail.path.clone();
                    self.table.clear();
                    self.populate_current_directory();
                } else {
                    self.current_thumb = thumbnail;
                    if !self.current_thumb.filename.is_empty() {
                        self.open_preview_pane = true;
                    }
                }
            } else {
                // Database mode:
                // If this path already exists in the table, treat as a selection (open preview)
                if self.table.iter().any(|r| r.path == thumbnail.path) {
                    self.current_thumb = thumbnail.clone();
                    if !self.current_thumb.filename.is_empty() { self.open_preview_pane = true; }
                } else {
                    // New row arriving from async DB load
                    self.table.push(thumbnail.clone());
                }
            }
        }

        while let Ok(env) = self.scan_rx.try_recv() {
            match env.msg {
                crate::utilities::scan::ScanMsg::Found(item) => {
                    log::info!("Found");
                    // (extension filtering handled during scan)
                    if let Some(row) = crate::file_to_thumbnail(&item) { 
                        let mut row = row;
                        // If archive, tag type for UI (file_to_thumbnail currently classifies only image/video/other)
                        if let Some(ext) = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                            if crate::utilities::types::is_archive(&ext) { row.file_type = "<ARCHIVE>".into(); }
                        }
                        self.table.push(row); 
                        ctx.request_repaint();
                        // Enqueue for AI indexing (images/videos). Use lightweight FileMetadata conversion.
                        let path_clone = item.path.to_string_lossy().to_string();
                        let ftype = {
                            if let Some(ext) = item.path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                if crate::is_image(ext.as_str()) { "image".to_string() }
                                else if crate::is_video(ext.as_str()) { "video".to_string() }
                                else { "other".to_string() }
                            } else { "other".to_string() }
                        };
                        let meta = crate::database::FileMetadata {
                            id: None,
                            path: path_clone.clone(),
                            filename: item.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
                            file_type: ftype,
                            size: item.size.unwrap_or(0),
                            modified: None,
                            created: None,
                            thumbnail_path: None,
                            thumb_b64: item.thumb_data.clone(),
                            hash: None,
                            description: None,
                            caption: None,
                            tags: Vec::new(),
                            category: None,
                            embedding: None,
                            similarity_score: None,
                            clip_embedding: None,
                            clip_similarity_score: None,
                        };
                        tokio::spawn(async move { let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await; });
                    }
                    
                }
                crate::utilities::scan::ScanMsg::FoundBatch(batch) => {
                    log::info!("FoundBatch");
                    let mut newly_enqueued: Vec<String> = Vec::new();
                    let mut to_index: Vec<crate::database::FileMetadata> = Vec::new();
                    let excluded = if self.excluded_terms.is_empty() { None } else { Some(self.excluded_terms.iter().map(|s| s.to_ascii_lowercase()).collect::<Vec<_>>()) };
                    for item in batch.into_iter() {
                        // (extension filtering handled during scan)
                        if let Some(ref ex) = excluded {
                            let lp = item.path.to_string_lossy().to_ascii_lowercase();
                            if ex.iter().any(|t| lp.contains(t)) { continue; }
                        }
                        if let Some(row) = crate::file_to_thumbnail(&item) { 
                            let mut row = row;
                            if let Some(ext) = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                if crate::utilities::types::is_archive(&ext) { row.file_type = "<ARCHIVE>".into(); }
                            }
                            self.table.push(row);
                            ctx.request_repaint();
                            // Schedule thumbnail generation if none yet
                            if item.thumb_data.is_none() {
                                let p_str = item.path.to_string_lossy().to_string();
                                if self.thumb_scheduled.insert(p_str.clone()) {
                                    newly_enqueued.push(p_str);
                                }
                            }
                            // Prepare indexing metadata
                            let ftype = {
                                if let Some(ext) = item.path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                                    if crate::is_image(ext.as_str()) { "image".to_string() }
                                    else if crate::is_video(ext.as_str()) { "video".to_string() }
                                    else { "other".to_string() }
                                } else { "other".to_string() }
                            };
                            to_index.push(crate::database::FileMetadata {
                                id: None,
                                path: item.path.to_string_lossy().to_string(),
                                filename: item.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
                                file_type: ftype,
                                size: item.size.unwrap_or(0),
                                modified: None,
                                created: None,
                                thumbnail_path: None,
                                thumb_b64: item.thumb_data.clone(),
                                hash: None,
                                description: None,
                                caption: None,
                                tags: Vec::new(),
                                category: None,
                                embedding: None,
                                similarity_score: None,
                                clip_embedding: None,
                                clip_similarity_score: None,
                            });
                        }
                        if self.pending_thumb_rows.len() >= 32 {
                            let batch = std::mem::take(&mut self.pending_thumb_rows);
                            tokio::spawn(async move { 
                                if let Err(e) = crate::database::save_thumbnail_batch(batch).await { 
                                    log::error!("scan batch persistence failed: {e}"); 
                                }
                            });
                        }
                    }
                    if !to_index.is_empty() {
                        tokio::spawn(async move {
                            for meta in to_index.into_iter() { let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await; }
                        });
                    }
                    if !newly_enqueued.is_empty() {
                        let scan_tx = self.scan_tx.clone();
                        let sem = self.thumb_semaphore.clone();
                        // Use a dedicated scan id for these thumb updates
                        let scan_id_for_updates = next_scan_id();
                        for path_str in newly_enqueued.into_iter() {
                            let permit_fut = sem.clone().acquire_owned();
                            let tx_clone = scan_tx.clone();
                            tokio::spawn(async move {
                                let _permit = permit_fut.await.expect("semaphore closed");
                                let path_buf = std::path::PathBuf::from(&path_str);
                                let ext = path_buf.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                                let mut thumb: Option<String> = None;
                                if let Some(ext) = ext {
                                    let is_img = crate::is_image(ext.as_str());
                                    let is_vid = crate::is_video(ext.as_str());
                                    if is_img {
                                        thumb = crate::utilities::thumbs::generate_image_thumb_data(&path_buf).ok();
                                    } else if is_vid {
                                        #[cfg(windows)]
                                        { thumb = crate::utilities::thumbs::generate_video_thumb_data(&path_buf).ok(); }
                                    }
                                }
                                if let Some(t) = thumb {
                                    let _ = tx_clone.try_send(ScanEnvelope { scan_id: scan_id_for_updates, msg: crate::utilities::scan::ScanMsg::UpdateThumb { path: path_buf, thumb: t } });
                                }
                            });
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::UpdateThumb { path, thumb } => { 
                    if let Some(found) = self.table.iter_mut().find(|f| PathBuf::from(f.path.clone()) == path) { 
                        found.thumbnail_b64 = Some(thumb.clone()); 
                        self.pending_thumb_rows.push(found.clone());
                    } else {
                        log::error!("Not found");
                    }
                    if let Ok(decoded) = B64.decode(thumb.as_bytes()) {
                        let key = path.to_string_lossy().to_string();
                        self.viewer.thumb_cache.entry(key).or_insert_with(|| Arc::from(decoded.into_boxed_slice()));
                    }
                    if self.pending_thumb_rows.len() >= 32 {
                        let batch = std::mem::take(&mut self.pending_thumb_rows);
                        tokio::spawn(async move { 
                            if let Err(e) = crate::database::save_thumbnail_batch(batch).await { 
                                log::error!("thumb update persistence failed: {e}"); 
                            }
                        });
                    }
                 }
                crate::utilities::scan::ScanMsg::Progress { scanned, total } => if scanned > 0 && total > 0 {
                    self.file_scan_progress = (scanned as f32) / (total as f32);
                }
                crate::utilities::scan::ScanMsg::Error(e) => { log::error!("ScanMsg Error: {e:?}"); }
                crate::utilities::scan::ScanMsg::Done => {
                    log::info!("Done");
                    self.file_scan_progress = 1.0;
                    self.scan_done = true;
                    if !self.pending_thumb_rows.is_empty() {
                        let batch = std::mem::take(&mut self.pending_thumb_rows);
                        tokio::spawn(async move { 
                            if let Err(e) = crate::database::save_thumbnail_batch(batch).await { 
                                log::error!("final scan batch persistence failed: {e}"); 
                            }
                        });
                    }
                }
            }
            ctx.request_repaint();
        }

        // Process AI streaming updates
        while let Ok(update) = self.ai_update_rx.try_recv() {
            ctx.request_repaint();
            match update {
                AIUpdate::Interim { path, text } => {
                    // Auto-follow: always advance to the currently streaming image if enabled.
                    if self.follow_active_vision && !path.is_empty() {
                        if self.current_thumb.path != path {
                            if let Some(row) = self.table.iter().find(|r| r.path == path) {
                                self.current_thumb = row.clone();
                                self.open_preview_pane = true; // ensure visible
                            }
                        }
                    }
                    self.streaming_interim.insert(path, text);
                }
                AIUpdate::Final { path, description, caption, category, tags } => {
                    if self.follow_active_vision && !path.is_empty() {
                        if self.current_thumb.path != path {
                            if let Some(row) = self.table.iter().find(|r| r.path == path) {
                                self.current_thumb = row.clone();
                                self.open_preview_pane = true;
                            }
                        }
                    }
                    self.streaming_interim.remove(&path);
                    // Update counters: a pending item finished.
                    if self.vision_pending > 0 { self.vision_pending -= 1; }
                    self.vision_completed += 1;
                    let desc_clone_for_row = description.clone();
                    let caption_clone_for_row = caption.clone();
                    let category_clone_for_row = category.clone();
                    let tags_clone_for_row = tags.clone();
                    if let Some(row) = self.table.iter_mut().find(|r| r.path == path) {
                        if !desc_clone_for_row.trim().is_empty() { row.description = Some(desc_clone_for_row.clone()); }
                        if let Some(c) = caption_clone_for_row.clone() { if !c.trim().is_empty() { row.caption = Some(c); } }
                        if let Some(cat) = category_clone_for_row.clone() { if !cat.trim().is_empty() { row.category = Some(cat); } }
                        if !tags_clone_for_row.is_empty() { row.tags = tags_clone_for_row.clone(); }
                    }
                    if self.current_thumb.path == path {
                        if !description.trim().is_empty() { self.current_thumb.description = Some(description.clone()); }
                        if let Some(c) = caption.clone() { if !c.trim().is_empty() { self.current_thumb.caption = Some(c); } }
                        if let Some(cat) = category.clone() { if !cat.trim().is_empty() { self.current_thumb.category = Some(cat); } }
                        if !tags.is_empty() { self.current_thumb.tags = tags.clone(); }
                    }
                    // Defensive persistence: ensure final AI metadata is saved to DB (idempotent if already saved by engine)
                    {
                        let persist_path = path.clone();
                        let description_clone = description.clone();
                        let caption_clone = caption.clone();
                        let category_clone = category.clone();
                        let tags_clone = tags.clone();
                        // Prefer current_thumb if active, else table row, else the cloned incoming values.
                        let (desc_final, cap_final, cat_final, tags_final) = if self.current_thumb.path == persist_path {
                            (
                                self.current_thumb.description.clone().unwrap_or_default(),
                                self.current_thumb.caption.clone().unwrap_or_default(),
                                self.current_thumb.category.clone().unwrap_or_else(|| "general".into()),
                                self.current_thumb.tags.clone(),
                            )
                        } else if let Some(row) = self.table.iter().find(|r| r.path == persist_path) {
                            (
                                row.description.clone().unwrap_or_default(),
                                row.caption.clone().unwrap_or_default(),
                                row.category.clone().unwrap_or_else(|| "general".into()),
                                row.tags.clone(),
                            )
                        } else { (description_clone, caption_clone.unwrap_or_default(), category_clone.unwrap_or_else(|| "general".into()), tags_clone) };
                        tokio::spawn(async move {
                            let vd = crate::ai::generate::VisionDescription { description: desc_final, caption: cap_final, category: cat_final, tags: tags_final };
                            if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.apply_vision_description(&persist_path, &vd).await {
                                log::warn!("[AI] UI final persist failed for {}: {}", persist_path, e);
                            }
                        });
                    }
                }
                AIUpdate::SimilarResults { origin_path, results } => {
                    self.similar_origin = Some(origin_path.clone());
                    self.similar_results = results;
                    if self.similar_results.is_empty() {
                        // keep modal open showing placeholder
                        self.show_similar_modal = true;
                    } else {
                        self.show_similar_modal = true; // ensure visible
                    }
                }
            }
        }

        // Process metadata update messages
        while let Ok(update) = self.meta_rx.try_recv() {
            ctx.request_repaint();
            let path = update.path.clone();
            let description = update.description.clone();
            let caption = update.caption.clone();
            let category = update.category.clone();
            let tags = update.tags.clone();
            // Update in-memory table row(s)
            if let Some(row) = self.table.iter_mut().find(|r| r.path == path) {
                if let Some(desc) = description.clone() { row.description = Some(desc); }
                if let Some(cap) = caption.clone() { row.caption = Some(cap); }
                if let Some(cat) = category.clone() { row.category = Some(cat); }
                row.tags = tags.clone();
            }
            // Update current_thumb if it matches
            if self.current_thumb.path == path {
                if let Some(desc) = description { self.current_thumb.description = Some(desc); }
                if let Some(cap) = caption { self.current_thumb.caption = Some(cap); }
                if let Some(cat) = category { self.current_thumb.category = Some(cat); }
                self.current_thumb.tags = tags.clone();
            }
        }
    }

    fn fetch_directory_metadata(&self) {
        // Optimized: only fetch rows for the currently listed directory entries instead of selecting whole table.
        // We build a list of file paths present in `self.table` that are real media files (or directories) and query by chunks.
        // This leverages the UNIQUE index on thumbnails.path.
        let paths: Vec<String> = self
            .table
            .iter()
            .map(|t| t.path.clone())
            .collect();
        if paths.is_empty() { return; }
        let tx = self.meta_tx.clone();
        // Chunk to keep query parameter size reasonable (Surreal has internal limits; 512 chosen conservatively)
        const CHUNK: usize = 400; // adjustable; each row only binds path strings
        for chunk in paths.chunks(CHUNK) {
            let chunk_vec: Vec<String> = chunk.iter().cloned().collect();
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                // SurrealDB v2 membership test: array::find($paths, path) returns the value or NONE.
                // Filter rows whose path is in the bound array.
                let primary_sql = "SELECT path, description, db_created, caption, category, tags, size, filename, file_type FROM thumbnails WHERE array::find($paths, path) != NONE";
                let mut resp = match crate::database::DB
                    .query(primary_sql)
                    .bind(("paths", chunk_vec.clone()))
                    .await {
                        Ok(r) => r,
                        Err(e) => {
                            log::warn!("Primary directory metadata query failed: {e}");
                            return;
                        }
                    };
                let rows: Result<Vec<crate::Thumbnail>, _> = resp.take(0);
                match rows {
                    Ok(list) => {
                        for row in list.into_iter() {
                            let _ = tx_clone.try_send(AIMetadataUpdate { path: row.path.clone(), description: row.description.clone(), caption: row.caption.clone(), category: row.category.clone(), tags: row.tags.clone() });
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to take directory metadata rows: {e}");
                    }
                }
            });
        }
    }
}    


pub fn get_img_ui(thumb_cache: &HashMap<String, Arc<[u8]>>, cache_key: &String, ui: &mut Ui) -> Response {
    if let Some(bytes_arc) = thumb_cache.get(cache_key) {
        let img_src = ImageSource::Bytes { 
            uri: Cow::from(format!("bytes://{}", cache_key)), 
            bytes: eframe::egui::load::Bytes::Shared(bytes_arc.clone()) 
        };
        Image::new(img_src)
        .fit_to_original_size(1.0)
        .max_size(vec2(ui.available_width(), 600.))

        .ui(ui) // .paint_at(ui, rect);
    } else {
        ui.label("No Image")
    }
}

impl FileExplorer {
    fn populate_current_directory(&mut self) {
        let path = PathBuf::from(&self.current_path);
        if !path.exists() { return; }
        self.table.clear();
        self.files.clear();
        let walker = WalkDirGeneric::<((), Option<u64>)>::new(&path)
            .parallelism(jwalk::Parallelism::RayonNewPool(16))
            .skip_hidden(false)
            .follow_links(false)
            .max_depth(1);
        let mut media_paths: Vec<PathBuf> = Vec::new();
        let excluded: Vec<String> = self.excluded_terms.iter().map(|s| s.to_ascii_lowercase()).collect();
        for entry in walker {
            if let Ok(e) = entry {
                let p = e.path();
                // Skip the root itself (depth 0) - jwalk root yields itself sometimes
                if p == path { continue; }
                if e.file_type().is_dir() {
                    if let Some(row) = Self::directory_to_thumbnail(&p) { self.table.push(row); }
                    continue;
                }
                if e.file_type().is_file() {
                    if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                        if !crate::is_supported_media_ext(ext.as_str()) { continue; }
                        // Mark archives differently
                        if crate::utilities::types::is_archive(&ext) {
                            // Represent as a pseudo-directory row to allow future expansion (opening contents)
                            if let Some(mut row) = Self::file_to_minimal_thumbnail(&p) {
                                row.file_type = "<ARCHIVE>".into();
                                self.table.push(row);
                            }
                            continue; // don't schedule as media thumbnail
                        }
                    } else { continue; }
                    if !excluded.is_empty() {
                        let lower_path = p.to_string_lossy().to_ascii_lowercase();
                        if excluded.iter().any(|term| lower_path.contains(term)) { continue; }
                    }
                    // Reuse existing conversion helper if available
                    if let Some(row) = Self::file_to_minimal_thumbnail(&p) { self.table.push(row); }
                    media_paths.push(p.clone());
                }
            }
        }
        // Spawn shallow thumbnail generation for displayed media
        if !media_paths.is_empty() {
            let tx = self.scan_tx.clone();
            let scan_id = next_scan_id();
            std::thread::spawn(move || {
                for p in media_paths.into_iter() {
                    if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                        let is_img = crate::is_image(ext.as_str());
                        let is_vid = crate::is_video(ext.as_str());
                        if is_img {
                            if let Ok(thumb) = crate::utilities::thumbs::generate_image_thumb_data(&p) {
                                let _ = tx.try_send(ScanEnvelope { scan_id, msg: crate::utilities::scan::ScanMsg::UpdateThumb { path: p.clone(), thumb } });
                            }
                        } else if is_vid {
                            #[cfg(windows)]
                            if let Ok(thumb) = crate::utilities::thumbs::generate_video_thumb_data(&p) {
                                let _ = tx.try_send(ScanEnvelope { scan_id, msg: crate::utilities::scan::ScanMsg::UpdateThumb { path: p.clone(), thumb } });
                            }
                        }
                    }
                }
                let _ = tx.try_send(ScanEnvelope { scan_id, msg: crate::utilities::scan::ScanMsg::Done });
            });
        }
        // After listing entries, fetch any existing AI metadata cached in DB and merge.
        // New per-directory minimal AI metadata hydration (path-only set + minimal fields).
        {
            let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
            let dir_file_paths: Vec<String> = self.table.iter().filter(|t| t.file_type != "<DIR>").map(|t| t.path.clone()).collect();
            if !dir_file_paths.is_empty() {
                tokio::spawn(async move {
                    let _ = engine.hydrate_directory_paths(&dir_file_paths).await; // count logged inside
                });
            }
        }
        // Legacy metadata fetch pathway retained for now (can be removed once UI fully reads from AI engine state if desired)
        self.fetch_directory_metadata();
    }

    fn directory_to_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        t.file_type = "<DIR>".into();
        Some(t)
    }

    fn file_to_minimal_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) { t.file_type = ext.to_ascii_lowercase(); }
        if let Ok(meta) = std::fs::metadata(p) { t.size = meta.len(); }
        Some(t)
    }

    fn push_history(&mut self, new_path: String) {
        if self.current_path != new_path {
            self.back_stack.push(self.current_path.clone());
            self.forward_stack.clear();
            self.current_path = new_path;
        }
    }

    fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            self.forward_stack.push(self.current_path.clone());
            self.current_path = prev;
            self.populate_current_directory();
        }
    }

    fn nav_forward(&mut self) {
        if let Some(next) = self.forward_stack.pop() {
            self.back_stack.push(self.current_path.clone());
            self.current_path = next;
            self.populate_current_directory();
        }
    }

    fn nav_up(&mut self) {
        if let Some(parent) = Path::new(&self.current_path).parent() {
            let p = parent.to_string_lossy().to_string();
            self.push_history(p);
            self.populate_current_directory();
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    fn nav_home(&mut self) {
        if let Some(ud) = directories::UserDirs::new() {
            let home = ud.home_dir();
            let hp = home.to_string_lossy().to_string();
            self.push_history(hp);
            self.populate_current_directory();
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    fn refresh(&mut self) { self.populate_current_directory(); }

    fn load_database_rows(&mut self) {
        self.table.clear();
        self.thumb_scheduled.clear();
        self.pending_thumb_rows.clear();
        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            // Page size for debug/DB view; can make user-configurable later.
            let page_limit = 500usize;
            match crate::database::load_thumbnails_page(0, page_limit, None, None, None, None, None).await {
                Ok(rows) => {
                    for r in rows.into_iter() { let _ = tx.try_send(r); }
                },
                Err(e) => { log::error!("DB page load failed: {e}"); }
            }
        });
    }
}