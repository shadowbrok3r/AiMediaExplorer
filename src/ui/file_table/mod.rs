use std::{borrow::Cow, collections::HashMap, sync::Arc};
use crossbeam::channel::{Receiver, Sender};
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};
use crate::{ScanEnvelope, Thumbnail, utilities::windows::{gpu_mem_mb, system_mem_mb, smoothed_cpu01, smoothed_ram01, smoothed_vram01}};
use egui_data_table::Renderer;
use viewer::FileTableViewer;
use humansize::DECIMAL;
use serde::Serialize;
use eframe::egui::*;

// Size filter inputs are now persisted via UiSettings; temporary globals removed.

pub mod quick_access_pane;
pub mod preview_pane;
pub mod explorer;
pub mod receive;
pub mod viewer;
pub mod codec;

// Request to open a new filtered tab in the dock
#[derive(Clone, Debug)]
pub enum FilterRequest {
    NewTab { title: String, rows: Vec<crate::database::Thumbnail> },
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

fn _load_png_as_white_lineart(path: &std::path::Path) -> ColorImage {
    use image::GenericImageView;
    let dyn_img = image::ImageReader::open(path).unwrap().decode().unwrap();
    let (w, h) = dyn_img.dimensions();
    let pixels = vec![Color32::WHITE; w as usize * h as usize]; // fill every pixel
    let mut out = ColorImage::new([w as usize, h as usize], pixels);

    for (x, y, pixel) in dyn_img.pixels() {
        let image::Rgba([r, g, b, a]) = pixel;
        // If it's “near black” but visible, make it white, keep alpha
        if a > 0 && r < 32 && g < 32 && b < 32 {
            out[(x as usize, y as usize)] = Color32::from_rgba_unmultiplied(255, 255, 255, a);
        } else {
            out[(x as usize, y as usize)] = Color32::from_rgba_unmultiplied(r, g, b, a);
        }
    }
    out
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
    // Similar image search UI state
    #[serde(skip)]
    similar_origin: Option<String>,
    #[serde(skip)]
    similar_results: Vec<SimilarResult>,
    // removed legacy similar modal/table flags (now using viewer.showing_similarity)
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
}

impl Default for FileExplorer {
    fn default() -> Self {
        let (thumbnail_tx, thumbnail_rx) = crossbeam::channel::unbounded();
        let (scan_tx, scan_rx) = crossbeam::channel::unbounded();
        let (ai_update_tx, ai_update_rx) = crossbeam::channel::unbounded();
        let (clip_embedding_tx, clip_embedding_rx) = crossbeam::channel::unbounded();
    let (db_preload_tx, db_preload_rx) = crossbeam::channel::unbounded();
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
            similar_origin: None,
            similar_results: Vec::new(),
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
        };
        // Initial shallow directory population (non-recursive)
        this.populate_current_directory();
        this
    }
}

impl FileExplorer {
    pub fn set_rows(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = viewer::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
    }
    // Set the table rows from DB results and switch to Database viewing mode
    pub fn set_rows_from_db(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = viewer::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
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
                    ui.add_space(5.);
                    if ui.button("⚙").on_hover_text("Open Options side panel").clicked() {
                        self.open_quick_access = !self.open_quick_access;
                    }
                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                    let err_color = ui.style().visuals.error_fg_color;
                    let font = FontId::proportional(15.);
                    let size = vec2(25., 25.);
                    if Button::new(RichText::new("⬆").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_up(); }
                    if Button::new(RichText::new("⬅").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_back(); }
                    if Button::new(RichText::new("➡").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_forward(); }
                    if Button::new(RichText::new("⟲").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.refresh(); }
                    if Button::new(RichText::new("🏠").font(font).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_home(); }
                    ui.separator();
                    let path_edit = TextEdit::singleline(&mut self.current_path)
                        .hint_text(if self.viewer.mode == viewer::ExplorerMode::Database {
                            if self.ai_search_enabled { "AI Search (text prompt)" } else { "Path Prefix Filter" }
                        } else {
                            "Current Directory"
                        })
                        .desired_width(300.)
                        .ui(ui);
                    if self.viewer.mode == viewer::ExplorerMode::Database {
                        let mut ai_box = self.ai_search_enabled;
                        if ui.checkbox(&mut ai_box, "AI").on_hover_text("Use AI semantic search across the database (text prompt)").clicked() {
                            self.ai_search_enabled = ai_box;
                        }
                    }
                    if self.viewer.mode == viewer::ExplorerMode::Database {
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
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.menu_button("🔻", |ui| {
                            ui.label(RichText::new("Excluded Terms (substring, case-insensitive)").italics());
                            ui.horizontal(|ui| {
                                let resp = TextEdit::singleline(&mut self.excluded_term_input)
                                    .hint_text("term")
                                    .desired_width(140.)
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

                        TextEdit::singleline(&mut self.viewer.filter)
                        .desired_width(200.0)
                        .hint_text("Search for files")
                        .ui(ui);

                        ui.separator();

                        let style = StyleModifier::default();
                        style.apply(ui.style_mut());
                        
                        MenuButton::new("Scan")
                        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style))
                        .ui(ui, |ui| {
                            ui.label(RichText::new("Size Filters (KB)").italics());
                            ui.horizontal(|ui| {
                                ui.label("Min:");
                                let mut min_kb: i64 = self.viewer.ui_settings.db_min_size_bytes
                                    .map(|b| (b / 1024) as i64)
                                    .unwrap_or(0);
                                let mut max_kb: i64 = self.viewer.ui_settings.db_max_size_bytes
                                    .map(|b| (b / 1024) as i64)
                                    .unwrap_or(0);
                                let min_resp = ui.add(egui::DragValue::new(&mut min_kb).speed(1).range(0..=i64::MAX).suffix(" KB")).on_hover_text("Minimum file size in KiB");
                                ui.label("Max:");
                                let max_resp = ui.add(egui::DragValue::new(&mut max_kb).speed(10).range(0..=i64::MAX).suffix(" KB")).on_hover_text("Maximum file size in KiB (0 = no max)");
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
                                }
                            });
                            ui.add_space(4.0);
                            ui.label(RichText::new("Types").italics());
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.viewer.types_show_images, "Images");
                                ui.checkbox(&mut self.viewer.types_show_videos, "Videos");
                                ui.checkbox(&mut self.viewer.types_show_dirs, "Folders");
                            });
                            ui.horizontal(|ui| {
                                let mut skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                if ui.checkbox(&mut skip_icons, "Skip likely icons").on_hover_text("Filter out tiny images (.ico, <= 16–64px, very small files)").changed() {
                                    self.viewer.ui_settings.filter_skip_icons = skip_icons;
                                    crate::database::settings::save_settings(&self.viewer.ui_settings);
                                }
                            });
                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("💡 Recursive Scan").clicked() {
                                    self.recursive_scan = true;
                                    self.scan_done = false;
                                    self.table.clear();
                                    // Reset previous scan snapshot
                                    self.last_scan_rows.clear();
                                    self.last_scan_paths.clear();
                                    self.last_scan_root = Some(self.current_path.clone());
                                    self.file_scan_progress = 0.0;
                                    let scan_id = crate::next_scan_id();
                                    let tx = self.scan_tx.clone();
                                    let recurse = self.recursive_scan.clone();
                                    let mut filters = crate::Filters::default();
                                    filters.root = std::path::PathBuf::from(self.current_path.clone());
                                    filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                                    filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                                    filters.include_images = self.viewer.types_show_images;
                                    filters.include_videos = self.viewer.types_show_videos;
                                    filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                    filters.excluded_terms = self.excluded_terms.clone();
                                    // Remember current scan id so cancel can target it
                                    self.current_scan_id = Some(scan_id);
                                    tokio::spawn(async move {
                                        crate::spawn_scan(filters, tx, recurse, scan_id).await;
                                    });
                                }
                                if ui.button("🖩 Bulk Generate Descriptions").on_hover_text("Generate AI descriptions for images missing caption/description").clicked() {
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
                                if ui.button("Generate Missing CLIP Embeddings").clicked() {
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
                                if ui.button("🗙 Cancel Scan").on_hover_text("Cancel active recursive scan").clicked() {
                                    if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
                                }
                                if ui.button("↩ Return to Active Scan").on_hover_text("Restore the last recursive scan results without rescanning").clicked() {
                                    if !self.last_scan_rows.is_empty() {
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
                                if ui.button("🛑 Cancel Bulk Descriptions").on_hover_text("Stop scheduling/streaming new vision descriptions").clicked() {
                                    crate::ai::GLOBAL_AI_ENGINE.cancel_bulk_descriptions();
                                    crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
                                    self.viewer.bulk_cancel_requested = true;
                                }
                                if ui.button("🔄 Re-scan Current Folder").on_hover_text("Shallow scan with current filters (applies 'Skip likely icons')").clicked() {
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
                            });
                        });

                        ui.menu_button("👁", |ui| {
                            ui.checkbox(&mut self.open_preview_pane, "Show Preview Pane");
                            ui.checkbox(&mut self.open_quick_access, "Show Quick Access (this panel)");
                            ui.separator();
                            ui.checkbox(&mut self.follow_active_vision, "Follow active vision (auto-select)")
                                .on_hover_text("When enabled, the preview auto-selects the image currently being described.");
                            ui.separator();
                            ui.label(RichText::new("File Types").italics());
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.viewer.types_show_images, "Images");
                                ui.checkbox(&mut self.viewer.types_show_videos, "Videos");
                                ui.checkbox(&mut self.viewer.types_show_dirs, "Folders");
                            });
                            if ui.button("Group by Category").clicked() {
                                // TODO implement grouping pipeline
                            }
                        });
                        
                        ui.menu_button("Table", |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("Reload Page").clicked() { self.load_database_rows(); }
                                if ui.button("Clear Table").clicked() { self.table.clear(); }
                            });

                            ui.add_space(4.0);
                            if matches!(self.viewer.mode, viewer::ExplorerMode::Database) {
                                ui.label(format!("Loaded Rows: {} (offset {})", self.table.len(), self.db_offset));
                                if self.db_loading { ui.colored_label(Color32::YELLOW, "Loading..."); }
                            }
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
            Renderer::new(&mut self.table, &mut self.viewer)
                .with_style_modify(|s| {
                    s.single_click_edit_mode = true;
                    s.table_row_height = Some(60.0);
                    s.auto_shrink = [false, false].into();
                })
                .ui(ui);
            // Check for tag/category open-tab actions requested by the viewer
            if !self.viewer.requested_tabs.is_empty() {
                let actions = std::mem::take(&mut self.viewer.requested_tabs);
                for act in actions.into_iter() {
                    match act {
                        crate::ui::file_table::viewer::TabAction::OpenCategory(cat) => {
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
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows });
                        }
                        crate::ui::file_table::viewer::TabAction::OpenTag(tag) => {
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
                                .push(crate::ui::file_table::FilterRequest::NewTab { title, rows });
                        }
                    }
                }
            }
            // Summary inline (counts) if selection active
            if !self.selected.is_empty() {
                ui.separator();
                ui.label(format!("Selected: {}", self.selected.len()));
            }
            if self.viewer.mode == viewer::ExplorerMode::Database && !self.viewer.showing_similarity {
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