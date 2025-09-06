use std::{borrow::Cow, collections::HashMap, sync::Arc};
use crossbeam::channel::{Receiver, Sender};
use egui::containers::menu::MenuConfig;
use crate::{ScanEnvelope, Thumbnail};
use egui_data_table::Renderer;
use viewer::FileTableViewer;
use humansize::DECIMAL;
use serde::Serialize;
use eframe::egui::*;

// Temporary global size filter placeholders (to be replaced by settings persistence)
pub(super) static mut MIN_SIZE_MB: Option<u64> = None; // value stored in bytes (converted from MB input)
pub(super) static mut MAX_SIZE_MB: Option<u64> = None; // value stored in bytes (converted from MB input)

pub mod quick_access_pane;
pub mod preview_pane;
pub mod explorer;
pub mod receive;
pub mod viewer;
pub mod codec;

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
        results: Vec<crate::database::FileMetadata>,
    },
}

#[derive(Debug, Clone)]
struct AIMetadataUpdate {
    path: String,
    description: Option<String>,
    caption: Option<String>,
    category: Option<String>,
    tags: Vec<String>,
}

#[derive(Serialize)]
pub struct FileExplorer {
    #[serde(skip)]
    table: egui_data_table::DataTable<Thumbnail>,
    pub viewer: FileTableViewer,
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
    // Channel for clip presence updates (path -> has_clip)
    #[serde(skip)]
    clip_presence_tx: Sender<(String, bool)>,
    #[serde(skip)]
    clip_presence_rx: Receiver<(String, bool)>,
    // Vision description generation tracking (bulk vision progress separate from indexing)
    #[serde(skip)]
    vision_started: usize,
    #[serde(skip)]
    vision_completed: usize,
    #[serde(skip)]
    vision_pending: usize, // scheduled but not yet final
    // Database paging state
    #[serde(skip)]
    db_offset: usize,
    #[serde(skip)]
    db_limit: usize,
    #[serde(skip)]
    db_last_batch_len: usize,
    #[serde(skip)]
    db_loading: bool,
    // Track current scan id for cancellation
    #[serde(skip)]
    current_scan_id: Option<u64>,
}

impl Default for FileExplorer {
    fn default() -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
        let (ai_update_tx, ai_update_rx) = crossbeam::channel::unbounded();
        let (meta_tx, meta_rx) = crossbeam::channel::unbounded();
        let (clip_presence_tx, clip_presence_rx) = crossbeam::channel::unbounded();
        let current_path = directories::UserDirs::new()
            .unwrap()
            .picture_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let mut this = Self {
            table: Default::default(),
            viewer: FileTableViewer::new(thumbnail_tx.clone(), ai_update_tx),
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
            show_similar_modal: false,
            similar_origin: None,
            similar_results: Vec::new(),
            follow_active_vision: true,
            clip_presence: std::collections::HashMap::new(),
            meta_tx,
            meta_rx,
            clip_presence_tx,
            clip_presence_rx,
            vision_started: 0,
            vision_completed: 0,
            vision_pending: 0,
            db_offset: 0,
            db_limit: 500,
            db_last_batch_len: 0,
            db_loading: false,
            current_scan_id: None,
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

        MenuBar::new()
            .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
            .ui(ui, |ui| {
                ui.set_height(25.);
                ui.horizontal_top(|ui| {
                    let fs_mode = self.viewer.mode == viewer::ExplorerMode::FileSystem;
                    if ui
                        .add_enabled(fs_mode, Button::new("⬅").min_size(vec2(20., 20.)))
                        .clicked()
                    {
                        self.nav_back();
                    }
                    if ui
                        .add_enabled(fs_mode, Button::new("⬆").min_size(vec2(20., 20.)))
                        .clicked()
                    {
                        self.nav_up();
                    }
                    if ui
                        .add_enabled(fs_mode, Button::new("➡").min_size(vec2(20., 20.)))
                        .clicked()
                    {
                        self.nav_forward();
                    }
                    if ui
                        .add_enabled(fs_mode, Button::new("⟲").min_size(vec2(20., 20.)))
                        .clicked()
                    {
                        self.refresh();
                    }
                    if ui
                        .add_enabled(fs_mode, Button::new("🏠").min_size(vec2(20., 20.)))
                        .clicked()
                    {
                        self.nav_home();
                    }
                    ui.separator();
                    let path_edit = TextEdit::singleline(&mut self.current_path)
                        .frame(true)
                        .hint_text(if self.viewer.mode == viewer::ExplorerMode::Database {
                            "Path Prefix Filter"
                        } else {
                            "Current Directory"
                        })
                        .desired_width(300.)
                        .ui(ui);
                    if self.viewer.mode == viewer::ExplorerMode::Database {
                        // Apply filter on Enter
                        if (path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                            || ui.button("Apply Filter").clicked()
                        {
                            // Reset paging and reload
                            self.db_offset = 0;
                            self.db_last_batch_len = 0;
                            self.load_database_rows();
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("⚙ Options").clicked() {
                            self.open_quick_access = true;
                        }
                        TextEdit::singleline(&mut self.viewer.filter)
                            .desired_width(200.0)
                            .hint_text("Search for files")
                            .ui(ui);
                        ui.separator();
                        egui::ComboBox::new("Table Mode", "")
                            .selected_text(match self.viewer.mode {
                                viewer::ExplorerMode::FileSystem => "File Explorer",
                                viewer::ExplorerMode::Database => "Database Explorer",
                            })
                            .show_ui(ui, |ui| {
                                let prev = self.viewer.mode;
                                if ui
                                    .selectable_value(
                                        &mut self.viewer.mode,
                                        viewer::ExplorerMode::FileSystem,
                                        "File Explorer",
                                    )
                                    .clicked()
                                    && prev != viewer::ExplorerMode::FileSystem
                                {
                                    self.populate_current_directory();
                                }
                                if ui
                                    .selectable_value(
                                        &mut self.viewer.mode,
                                        viewer::ExplorerMode::Database,
                                        "Database Explorer",
                                    )
                                    .clicked()
                                    && prev != viewer::ExplorerMode::Database
                                {
                                    self.load_database_rows();
                                }
                            });
                        if self.viewer.mode == viewer::ExplorerMode::Database {
                            if ui.button("🔄 Reload DB").clicked() {
                                self.load_database_rows();
                            }
                        }
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
                            bar = bar
                                .text(RichText::new("Scan Complete").color(Color32::LIGHT_GREEN));
                        }
                        bar.ui(ui);
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
                    } else if !ai_ready {
                        ui.label("AI: Loading");
                    } else {
                        ui.label("AI: Idle");
                    }

                    ui.add_space(ui.available_width() / 2.5);
                    // Breadcrumbs (avoid borrow conflicts by cloning path first)
                    let current_path_clone = self.current_path.clone();
                    let parts: Vec<String> = current_path_clone
                        .split(['\\', '/'])
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect();
                    let root_has_slash = current_path_clone.starts_with('/');
                    let mut accum = if root_has_slash {
                        String::from("/")
                    } else {
                        String::new()
                    };
                    ui.horizontal(|ui| {
                        for (i, part) in parts.iter().enumerate() {
                            if !accum.ends_with(std::path::MAIN_SEPARATOR) && !accum.is_empty() {
                                accum.push(std::path::MAIN_SEPARATOR);
                            }
                            accum.push_str(part);
                            let display = if part.is_empty() {
                                std::path::MAIN_SEPARATOR.to_string()
                            } else {
                                part.clone()
                            };
                            if ui
                                .selectable_label(false, RichText::new(display).underline())
                                .clicked()
                            {
                                self.push_history(accum.clone());
                                self.populate_current_directory();
                            }
                            if i < parts.len() - 1 {
                                ui.label(RichText::new("›").weak());
                            }
                            // Remove trailing segment for next iteration accumulation clone safety
                        }
                    });

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
            Renderer::new(&mut self.table, &mut self.viewer)
                .with_style_modify(|s| {
                    s.single_click_edit_mode = true;
                    s.table_row_height = Some(60.0);
                    s.auto_shrink = [false, false].into();
                })
                .ui(ui);
            // Summary inline (counts) if selection active
            if !self.selected.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.selected.len()));
            }
            if self.viewer.mode == viewer::ExplorerMode::Database {
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

        // Similar images modal (rendered each frame if active)
        if self.show_similar_modal {
            let mut open_flag = true; // represent window open state locally
            let title = if let Some(orig) = &self.similar_origin {
                format!(
                    "Similar To: {}",
                    std::path::Path::new(orig)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(orig)
                )
            } else {
                "Similar Images".to_string()
            };
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
                        ui.horizontal(|ui| {
                            ui.label(format!("{} results", self.similar_results.len()));
                        });
                        ui.separator();
                        for meta in self.similar_results.iter() {
                            ui.horizontal(|ui| {
                                if ui.button("Open").clicked() {
                                    let _ = open::that(meta.path.clone());
                                }
                                ui.label(RichText::new(&meta.filename).strong());
                                if let Some(score) =
                                    meta.clip_similarity_score.or(meta.similarity_score)
                                {
                                    ui.label(format!("score: {:.4}", score));
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("Preview").clicked() {
                                        // Attempt to load thumbnail row for this path into current_thumb (FileSystem mode only currently)
                                        if let Some(row) =
                                            self.table.iter().find(|r| r.path == meta.path)
                                        {
                                            self.current_thumb = row.clone();
                                            self.open_preview_pane = true;
                                        }
                                    }
                                });
                            });
                        }
                    }
                });
            if !open_flag {
                self.show_similar_modal = false;
            }
        }
    }

    fn load_database_rows(&mut self) {
        if self.db_loading {
            return;
        }
        self.db_loading = true;
        // When offset is zero we are (re)loading fresh page; clear existing rows
        if self.db_offset == 0 {
            self.table.clear();
            self.thumb_scheduled.clear();
            self.pending_thumb_rows.clear();
        }
        let tx = self.thumbnail_tx.clone();
        let offset = self.db_offset;
        let limit = self.db_limit;
        let path_filter = if self.current_path.trim().is_empty() {
            None
        } else {
            Some(self.current_path.clone())
        };
        tokio::spawn(async move {
            match crate::Thumbnail::load_thumbnails_page(
                offset,
                limit,
                None,
                None,
                None,
                None,
                None,
                path_filter.as_deref(),
            )
            .await
            {
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
