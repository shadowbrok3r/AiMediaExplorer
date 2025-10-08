use egui::{Button, Color32, CornerRadius, FontId, Image, ImageSource, KeyboardShortcut, RichText, ScrollArea, TextureOptions, Vec2, Widget};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use std::{borrow::Cow, collections::HashMap, path::Path, sync::Arc};
use egui_data_table::{
    viewer::{default_hotkeys, CustomActionContext, CustomActionEditor, RowCodec, UiActionContext}, CustomMenuItem, RowViewer, SelectionSnapshot, UiAction
};
use egui_extras::Column as TableColumnConfig;
use crossbeam::channel::Sender;
use humansize::DECIMAL;
use serde::Serialize;

use crate::{generate_image_thumb_data, generate_video_thumb_data, ui::file_table::AIUpdate, Thumbnail, UiSettings};
use codec::ThumbCodec;

pub mod codec;

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExplorerMode {
    #[default]
    FileSystem,
    Database,
}

impl ExplorerMode {
    pub fn to_str(&self) -> &str {
        match self {
            ExplorerMode::FileSystem => "FileSystem",
            ExplorerMode::Database => "Database",
        }
    }
}

#[derive(Serialize)]
pub struct FileTableViewer {
    pub filter: String,
    pub row_protection: bool,
    #[serde(skip)]
    pub hotkeys: Vec<(KeyboardShortcut, UiAction)>,
    pub show_hotkeys: bool,
    #[serde(skip)]
    thumbnail_tx: Sender<Thumbnail>,
    #[serde(skip)]
    // Cache raw decoded bytes so we only base64 decode once per unique file/hash.
    pub thumb_cache: HashMap<String, Arc<[u8]>>, // key: filename or hash
    pub mode: ExplorerMode,
    // Flag to stop scheduling bulk description tasks
    #[serde(skip)]
    pub bulk_cancel_requested: bool,
    pub ui_settings: crate::UiSettings,
    #[serde(skip)]
    pub ai_update_tx: Sender<AIUpdate>,
    #[serde(skip)]
    pub clip_embedding_tx: Sender<crate::ClipEmbeddingRow>,
    // When showing similarity results inline, toggle this to add a scores column
    pub showing_similarity: bool,
    // Map path -> similarity score to display/sort when showing_similarity is true
    #[serde(skip)]
    pub similar_scores: HashMap<String, f32>,
    // Track which paths have a CLIP embedding
    #[serde(skip)]
    pub clip_presence: std::collections::HashSet<String>,
    // Track which hashes have CLIP embeddings (for duplicate content across paths)
    #[serde(skip)]
    pub clip_presence_hashes: std::collections::HashSet<String>,
    // Type visibility toggles (UI): affect table view filtering and scans
    pub types_show_images: bool,
    pub types_show_videos: bool,
    // Archive passwords for accessing encrypted archives
    #[serde(skip)]
    pub archive_passwords: std::collections::HashMap<String, String>,
    pub types_show_dirs: bool,
    // Requests to open tabs based on user clicks in cells
    #[serde(skip)]
    pub requested_tabs: Vec<TabAction>,
    #[serde(skip)]
    pub selected: std::collections::HashSet<String>,
}

// Actions requested from table cells
#[derive(Clone, Debug)]
pub enum TabAction { OpenCategory(String), OpenTag(String), OpenSimilar(String), OpenArchive(String) }

impl FileTableViewer {
    pub fn new(thumbnail_tx: Sender<Thumbnail>, ai_update_tx: Sender<AIUpdate>, clip_embedding_tx: Sender<crate::ClipEmbeddingRow>) -> Self {
        Self {
            filter: Default::default(),
            row_protection: false,
            hotkeys: vec![],
            show_hotkeys: false,
            thumb_cache: HashMap::new(),
            thumbnail_tx,
            mode: ExplorerMode::default(),
            ui_settings: UiSettings::default(),
            bulk_cancel_requested: false,
            clip_embedding_tx,
            ai_update_tx,
            showing_similarity: false,
            similar_scores: HashMap::new(),
            clip_presence: std::collections::HashSet::new(),
            clip_presence_hashes: std::collections::HashSet::new(),
            types_show_images: true,
            types_show_videos: true,
            types_show_dirs: true,
            requested_tabs: Vec::new(),
            archive_passwords: std::collections::HashMap::new(),
            selected: std::collections::HashSet::new(),
        }
    }

    // Immutable helper to check if a row passes current viewer filters
    pub fn row_passes_filter(&self, row: &Thumbnail) -> bool {
        match self.mode {
            ExplorerMode::FileSystem | ExplorerMode::Database => {
                // Apply simple type visibility (view-only) before text search
                if row.file_type == "<DIR>" && !self.types_show_dirs { return false; }
                if row.file_type != "<DIR>" {
                    let ext_opt = if row.path.starts_with("zip://") {
                        std::path::Path::new(&row.filename)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                    } else {
                        std::path::Path::new(&row.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                    };
                    if let Some(ext) = ext_opt {
                        if crate::is_image(ext.as_str()) && !self.types_show_images { return false; }
                        if crate::is_video(ext.as_str()) && !self.types_show_videos { return false; }
                    }
                }
                if self.filter.trim().is_empty() { return true; }
                let f = self.filter.to_lowercase();
                let tags_join = row.tags.join(",");
                row.filename.to_lowercase().contains(&f)
                    || row.file_type.to_lowercase().contains(&f)
                    || row.path.to_lowercase().contains(&f)
                    || tags_join.to_lowercase().contains(&f)
                    || row
                        .category
                        .as_ref()
                        .map(|s| s.to_lowercase().contains(&f))
                        .unwrap_or(false)
            }
        }
    }
}

/* ----------------------------------------------------------------------------------------------
 *  Thumbnail Table Implementation
 *  Base Columns (both modes):
 *   0 Thumbnail (placeholder / future image)
 *   1 Name       (filename)
 *   2 Path       (path)
 *   3 Category   (category)
 *   4 Tags       (comma separated)
 *   5 Modified   (filesystem modified; in DB mode shows modified if present else '-')
 *   6 Size       (bytes -> human)
 *   7 Type       (file_type)
 *
 *  Extra Columns (Database mode only):
 *   8 Hash       (content hash if present)
 *   9 DB Created (row creation timestamp)
 * ------------------------------------------------------------------------------------------- */

impl RowViewer<Thumbnail> for FileTableViewer {
    fn try_create_codec(&mut self, _copy_full_row: bool) -> Option<impl RowCodec<Thumbnail>> {
        Some(ThumbCodec)
    }

    fn num_columns(&mut self) -> usize {
        // New base ordering (FS):
        // 0 Thumb, 1 Name, 2 Path, 3 Size, 4 Type, 5 CLIP, 6 Modified, 7 DB Created, 8 Category, 9 Tags
        // New base ordering (DB): same + 10 Hash
        let base = match self.mode { ExplorerMode::FileSystem => 10, ExplorerMode::Database => 11 };
        if self.showing_similarity { base + 1 } else { base }
    }

    fn column_name(&mut self, column: usize) -> std::borrow::Cow<'static, str> {
        let names_fs = [
            "", "Name", "Path", "Size", "Type", "CLIP", "Modified", "DB Created", "Category", "Tags",
        ]; // 10
        let names_db = [
            "", "Name", "Path", "Size", "Type", "CLIP", "Modified", "DB Created", "Category", "Tags", "Hash",
        ]; // 11
        let mut vec: Vec<Cow<'static, str>> = match self.mode {
            ExplorerMode::FileSystem => names_fs.iter().map(|s| (*s).into()).collect(),
            ExplorerMode::Database => names_db.iter().map(|s| (*s).into()).collect(),
        };
        if self.showing_similarity { vec.push("Similarity".into()); }
        vec[column].clone()
    }

    fn is_sortable_column(&mut self, column: usize) -> bool {
        column < self.num_columns()
    }

    fn row_filter_hash(&mut self) -> &impl std::hash::Hash {
        &self.filter
    }

    fn filter_row(&mut self, row: &Thumbnail) -> bool {
        self.row_passes_filter(row)
    }

    fn hotkeys(&mut self, context: &UiActionContext) -> Vec<(KeyboardShortcut, UiAction)> {
        let hot = default_hotkeys(context);
        self.hotkeys.clone_from(&hot);
        hot
    }

    fn is_editable_cell(&mut self, _column: usize, _row: usize, _row_value: &Thumbnail) -> bool {
        // Only allow editing for Category (3) and Tags (4) regardless of mode.
        // matches!(column, 3 | 4)
        false
    }


    fn show_cell_view(&mut self, ui: &mut egui::Ui, row: &Thumbnail, column: usize) {
        // CLIP is now column 5. Similarity (if shown) is appended at the end.
        let clip_idx = 5usize;
        if column == clip_idx {
            let has_by_path = self.clip_presence.contains(&row.path);
            let has_by_hash = row
                .hash
                .as_ref()
                .map(|h| self.clip_presence_hashes.contains(h))
                .unwrap_or(false);
            let has = has_by_path || has_by_hash;
            ui.label(if has { "Yes" } else { "No" });
        }
        let similarity_idx = if self.showing_similarity {
            match self.mode { ExplorerMode::FileSystem => 10, ExplorerMode::Database => 11 }
        } else { usize::MAX };
        if column == similarity_idx {
            if let Some(score) = self.similar_scores.get(&row.path) {
                ui.label(format!("{score:.4}"));
            } else {
                ui.label("-");
            }
        }
        match column {
            0 => {
                ui.vertical_centered_justified(|ui| {
                    // Thumbnail (mode-agnostic currently)
                    if row.file_type == "<DIR>" {
                        egui::Image::new(eframe::egui::include_image!(
                            "../../../../assets/Icons/folder.png"
                        ))
                        .fit_to_exact_size(ui.available_size())
                        .texture_options(
                            TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear))
                        ).ui(ui);
                    } else if row.file_type == "<ARCHIVE>" {
                        egui::Image::new(eframe::egui::include_image!(
                            "../../../../assets/Icons/zip.png"
                        ))
                        .fit_to_exact_size(ui.available_size())
                        .texture_options(
                            TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear))
                        ).ui(ui);
                    } else {
                        let cache_key = row.path.clone();
                        if row.thumbnail_b64.is_none() {
                            egui::Image::new(eframe::egui::include_image!(
                                "../../../../assets/Icons/broken_link.png"
                            ))
                            .fit_to_exact_size(ui.available_size())
                            .texture_options(
                                TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear))
                            ).ui(ui);
                        } else {
                            if !self.thumb_cache.contains_key(&cache_key) {
                                if let Some(mut b64) = row.thumbnail_b64.clone() {
                                    if b64.starts_with("data:image/png;base64,") {
                                        let (_, end) =
                                            b64.split_once("data:image/png;base64,").unwrap_or_default();
                                        b64 = end.to_string();
                                    }
                                    if let Ok(decoded) = B64.decode(b64.as_bytes()) {
                                        self.thumb_cache.insert(
                                            cache_key.clone(),
                                            Arc::from(decoded.into_boxed_slice()),
                                        );
                                    }
                                }
                            }
                            if let Some(bytes_arc) = self.thumb_cache.get(&cache_key) {
                                let img_src = ImageSource::Bytes {
                                    uri: Cow::from(format!("bytes://{}", cache_key)),
                                    bytes: egui::load::Bytes::Shared(bytes_arc.clone()),
                                };
                                Image::new(img_src)
                                .show_loading_spinner(true)
                                .fit_to_exact_size(ui.available_size())
                                .texture_options(TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear)))
                                .ui(ui); 
                            }
                        }
                    }
                });
            }
            1 => {
                // Name (make the label clickable so row click registers)
                let _ = ui.add(
                    egui::Label::new(format!(" {}", &row.filename))
                        .sense(egui::Sense::click()),
                );
            }
            2 => {
                // Path or DB key depending on mode (make clickable to trigger selection)
                match self.mode {
                    ExplorerMode::FileSystem => {
                        let _ = ui.add(egui::Label::new(&row.path).sense(egui::Sense::click()));
                    }
                    ExplorerMode::Database => {
                        let _ = ui.add(egui::Label::new(&row.path).sense(egui::Sense::click()));
                    }
                };
            }
            3 => {
                // Size
                if row.size == 0 {
                    ui.label("-");
                } else {
                    ui.label(humansize::format_size(row.size, DECIMAL));
                }
            }
            4 => {
                // Type
                ui.label(RichText::new(&row.file_type).font(FontId::monospace(11.)));
            }
            6 => {
                // Modified mm/dd/yyyy (or '-')
                use chrono::Datelike;
                if let Some(m) = &row.modified {
                    ui.label(format!("{}/{}/{}", m.month(), m.day(), m.year()));
                } else {
                    ui.label("-");
                }
            }
            7 => {
                // DB Created date
                if let Some(c) = &row.db_created {
                    use chrono::Datelike;
                    ui.label(format!("{}/{}/{}", c.month(), c.day(), c.year()));
                } else {
                    ui.label("-");
                }
            }
            8 => {
                ui.vertical_centered_justified(|ui| {
                    // Category (click to open in new tab)
                    if let Some(cat) = row.category.as_ref() {
                        let resp = ui.button(egui::RichText::new(cat).underline());
                        if resp.clicked() {
                            self.requested_tabs.push(TabAction::OpenCategory(cat.clone()));
                        }
                    } else {
                        ui.label("-");
                    }
                });
            }
            9 => {
                // Tags as chips
                if row.tags.is_empty() {
                    ui.label("-");
                } else {
                    ScrollArea::vertical().show(ui, |ui| {
                        for tag in &row.tags {
                            let color = color_for_tag(tag);
                            let txt = format!("{}", tag);
                            let _ = Button::new(txt).fill(color).corner_radius(CornerRadius::same(4)).ui(ui);
                        }
                    });
                }
            }
            10 => {
                // Hash (DB mode only)
                ui.label(row.hash.as_deref().unwrap_or("-"));
            }
            _ => {},
        };
    }

    fn show_cell_editor(
        &mut self,
        ui: &mut egui::Ui,
        row: &mut Thumbnail,
        column: usize,
    ) -> Option<egui::Response> {
        // No data to set for CLIP/Similarity columns
        let clip_idx = 5usize;
        let similarity_idx = if self.showing_similarity {
            match self.mode { 
                ExplorerMode::FileSystem => 10, 
                ExplorerMode::Database => 11 
            }
        } else {
            usize::MAX 
        };

        if column == clip_idx || column == similarity_idx { 
            return None; 
        }

        match column {
            0 => {
                let cache_key = row.path.clone();
                if row.thumbnail_b64.is_none() {
                    Some(ui.add_sized(
                        ui.available_size(),
                        egui::Image::new(eframe::egui::include_image!(
                            "../../../../assets/Icons/broken_link.png"
                        )).texture_options(TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear))),
                    ))
                } else {
                    if !self.thumb_cache.contains_key(&cache_key) {
                        if let Some(mut b64) = row.thumbnail_b64.clone() {
                            if b64.starts_with("data:image/png;base64,") {
                                let (_, end) =
                                    b64.split_once("data:image/png;base64,").unwrap_or_default();
                                b64 = end.to_string();
                            }
                            if let Ok(decoded) = B64.decode(b64.as_bytes()) {
                                self.thumb_cache.insert(
                                    cache_key.clone(),
                                    Arc::from(decoded.into_boxed_slice()),
                                );
                            }
                        }
                    }
                    if let Some(bytes_arc) = self.thumb_cache.get(&cache_key) {
                        let img_src = ImageSource::Bytes {
                            uri: Cow::from(format!("bytes://{}", cache_key)),
                            bytes: egui::load::Bytes::Shared(bytes_arc.clone()),
                        };
                        Some(
                            Image::new(img_src)
                                .bg_fill(Color32::WHITE)
                                .fit_to_exact_size(Vec2::splat(50.))
                                .ui(ui),
                        ) // .paint_at(ui, rect);
                    } else {
                        Some(ui.label("❌"))
                    }
                }
            }
            9 => {
                let mut response = None;
                ScrollArea::vertical().show(ui, |ui| {
                    for tag in &row.tags {
                        let color = color_for_tag(tag);
                        let txt = format!("{}", tag);
                        let resp = Button::new(txt).fill(color).corner_radius(CornerRadius::same(4)).ui(ui);
                        if resp.hovered() {
                            response = Some(resp.clone());
                            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                        }
                        if resp.clicked() {
                            self.requested_tabs.push(TabAction::OpenTag(tag.clone()));
                            response = Some(resp);
                        }
                    }
                });

                response
            }
            8 => {
                if let Some(cat) = row.category.as_ref() {
                    let resp = ui.button(egui::RichText::new(cat).underline());
                    if resp.clicked() {
                        self.requested_tabs.push(TabAction::OpenCategory(cat.clone()));
                    }
                    Some(resp)
                } else {
                    ui.label("-");
                    None
                }
            }
            // Non-editable columns
            _ => None,
        }
    }

    fn on_cell_view_response(
        &mut self,
        row: &Thumbnail,
        column: usize,
        resp: &egui::Response,
    ) -> Option<Box<Thumbnail>> {
        match column {
            2 => {
                if resp.hovered() { resp.clone().on_hover_text(&row.path); }
                if resp.clicked() {
                    // Mirror main click behavior: preview + thumb generation
                    if self.mode != ExplorerMode::Database {
                        if row.thumbnail_b64.is_none() {
                            if row.path.starts_with("zip://") || row.path.starts_with("7z://") || row.path.starts_with("tar://") {
                                let path = row.path.clone();
                                let thumbnail_tx = self.thumbnail_tx.clone();
                                let row_clone = row.clone();
                                let archive_passwords = self.archive_passwords.clone();
                                tokio::spawn(async move {
                                    if let Some(scheme_end) = path.find("://") {
                                        if let Some(internal_start) = path.find("!/") {
                                            let scheme = &path[..scheme_end];
                                            let archive_path = &path[scheme_end + 3..internal_start];
                                            let internal_path = &path[internal_start + 2..];
                                            let filename = std::path::Path::new(internal_path).file_name().and_then(|n| n.to_str()).unwrap_or(internal_path);
                                            let internal_dir = std::path::Path::new(internal_path).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                                            let password = archive_passwords.get(archive_path).map(|s| s.as_str());
                                            match crate::utilities::archive::extract_and_generate_thumbnail(scheme, archive_path, &internal_dir, filename, password) {
                                                Ok(Some(thumb_b64)) => {
                                                    let thumb = Thumbnail { thumbnail_b64: Some(thumb_b64), ..row_clone };
                                                    let _ = thumbnail_tx.try_send(thumb);
                                                }
                                                Ok(None) => {}
                                                Err(err) => {
                                                    let es = err.to_string();
                                                    if es.contains("PasswordRequired") {
                                                        let mut t = Thumbnail::default();
                                                        t.path = archive_path.to_string();
                                                        t.file_type = "<PASSWORD_REQUIRED>".to_string();
                                                        let _ = thumbnail_tx.try_send(t);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }
                    let _ = self.thumbnail_tx.try_send(row.clone());
                }
            }
            8 => {
                if let Some(cat) = row.category.as_ref() {
                    if resp.clicked() {
                        log::info!("Clicked on: {cat}");
                        self.requested_tabs.push(TabAction::OpenCategory(cat.clone()));
                    }
                }
            }
            9 => {
                for tag in &row.tags {
                    if resp.clicked() {
                        log::info!("Clicked on: {tag}");
                        self.requested_tabs.push(TabAction::OpenTag(tag.clone()));
                    }
                }
            }
            _ => {
                if resp.clicked() {
                    // Handle virtual folder rows first (category/tag views)
                    if row.file_type == "<DIR>" {
                        if let Some(rest) = row.path.strip_prefix("cat://") {
                            let cat = rest.trim_end_matches('/').to_string();
                            self.requested_tabs.push(TabAction::OpenCategory(cat));
                            return None;
                        }
                        if let Some(rest) = row.path.strip_prefix("tag://") {
                            let tag = rest.trim_end_matches('/').to_string();
                            self.requested_tabs.push(TabAction::OpenTag(tag));
                            return None;
                        }
                    }
                    if self.mode == ExplorerMode::Database {
                        // In DB mode, disable on-demand generation attempts (no filesystem walk triggers)
                    } else {
                        if row.thumbnail_b64.is_none() {
                            // Check if this is a virtual archive path
                            if row.path.starts_with("zip://") 
                                || row.path.starts_with("7z://") 
                                || row.path.starts_with("tar://") 
                            {
                                log::info!("Getting thumbnail for {}", row.path);
                                // For virtual archive paths, use async extraction-based thumbnail generation
                                let path = row.path.clone();
                                let thumbnail_tx = self.thumbnail_tx.clone();
                                let row_clone = row.clone();
                                let archive_passwords = self.archive_passwords.clone();
                                tokio::spawn(async move {
                                    // Parse the virtual path: scheme://archive_path!/internal_path
                                    if let Some(scheme_end) = path.find("://") {
                                        log::info!("scheme_end: {scheme_end}");
                                        if let Some(internal_start) = path.find("!/") {
                                            log::info!("internal_start: {internal_start}");
                                            let scheme = &path[..scheme_end];
                                            let archive_path = &path[scheme_end + 3..internal_start];
                                            let internal_path = &path[internal_start + 2..];
                                            log::info!("internal_path: {internal_path}");
                                            // Extract filename from internal path
                                            let filename = std::path::Path::new(internal_path)
                                                .file_name()
                                                .and_then(|n| n.to_str())
                                                .unwrap_or(internal_path);
                                            
                                            // Get directory part of internal path
                                            let internal_dir = std::path::Path::new(internal_path)
                                                .parent()
                                                .map(|p| p.to_string_lossy().to_string())
                                                .unwrap_or_default();
                                            
                                            // Look up the stored password for this archive
                                            let password = archive_passwords.get(archive_path).map(|s| s.as_str());
                                            
                                            match crate::utilities::archive::extract_and_generate_thumbnail(
                                                scheme, archive_path, &internal_dir, filename, password
                                            ) {
                                                Ok(Some(thumb_b64)) => {
                                                    let thumb = Thumbnail {
                                                        thumbnail_b64: Some(thumb_b64),
                                                        ..row_clone
                                                    };
                                                    let _ = thumbnail_tx.try_send(thumb);
                                                }
                                                Ok(None) => {
                                                    log::warn!("No thumbnail generated for archive file: {}", path);
                                                }
                                                Err(err) => {
                                                    let es = err.to_string();
                                                    if es.contains("PasswordRequired") {
                                                        // queue modal via control thumbnail
                                                        let mut t = Thumbnail::default();
                                                        t.path = archive_path.to_string();
                                                        t.file_type = "<PASSWORD_REQUIRED>".to_string();
                                                        let _ = thumbnail_tx.try_send(t);
                                                    } else {
                                                        log::warn!("Failed to generate thumbnail for archive file: {}: {}", path, es);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                });
                            } else {
                                log::info!("Not a archive: {:?}", row.path);
                                // Regular filesystem path - use direct thumbnail generation
                                // Determine media kind from file extension
                                let ext_opt = std::path::Path::new(&row.path)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase());
                            
                                if let Some(ext) = ext_opt {
                                    if crate::is_video(ext.as_str()) || crate::is_image(ext.as_str()) {
                                        let path_for_task = row.path.clone();
                                        let file_name_for_task = std::path::Path::new(&path_for_task).file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                                        let tx_thumb = self.thumbnail_tx.clone();
                                        let is_image_kind = crate::is_image(ext.as_str());
                                        let clip_tx_opt = if is_image_kind { Some(self.clip_embedding_tx.clone()) } else { None };
                                        // Heavy IO/decoding moved off UI thread
                                        tokio::spawn(async move {
                                            let path_clone_for_block = path_for_task.clone();
                                            let thumb_res = tokio::task::spawn_blocking(move || {
                                                if is_image_kind {
                                                    generate_image_thumb_data(Path::new(&path_clone_for_block))
                                                } else {
                                                    generate_video_thumb_data(Path::new(&path_clone_for_block))
                                                }
                                            }).await.ok().and_then(|r| r.ok());
                                            if let Some(b64) = thumb_res {
                                                let mut t = Thumbnail { thumbnail_b64: Some(b64), ..Thumbnail::new(&file_name_for_task) };
                                                t.path = path_for_task.clone();
                                                let _ = tx_thumb.try_send(t.clone());
                                                if let Some(clip_tx) = clip_tx_opt {
                                                    tokio::spawn(async move {
                                                        let _ = clip_tx.try_send(t.get_embedding().await.unwrap_or_default());
                                                    });
                                                }
                                            }
                                        });
                                    }
                                }
                            }
                        }
                    }
                    // If clicking an archive (zip), request to open virtual view
                    if let Some(ext) = std::path::Path::new(&row.path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase()) 
                    {
                        if ext == "zip" || row.file_type == "<ARCHIVE>" { 
                            self.requested_tabs.push(TabAction::OpenArchive(row.path.clone()));
                        }
                    } else if row.file_type == "<ARCHIVE>" {
                        self.requested_tabs.push(TabAction::OpenArchive(row.path.clone()));
                    }
                    log::info!("Showing new thumbnail: {}", row.path);
                    // Always notify selection so preview can reflect current row
                    let _ = self.thumbnail_tx.try_send(row.clone());
                    // Skip embedding generation for virtual paths inside zip
                    if !row.path.starts_with("zip://") 
                        && !row.path.starts_with("tar://") 
                        && !row.path.starts_with("7z://") 
                    {
                        let tx = self.clip_embedding_tx.clone();
                        let thumb = row.clone();
                        tokio::spawn(async move {
                            let row = thumb.get_embedding().await.unwrap_or_default();
                            let _ = tx.try_send(row);
                        });
                    }
                }
            }
        }
        None
    }

    fn set_cell_value(&mut self, src: &Thumbnail, dst: &mut Thumbnail, column: usize) {
        match column {
            0 => dst.thumbnail_b64 = src.thumbnail_b64.clone(),
            1 => dst.filename = src.filename.clone(),
            2 => dst.path = src.path.clone(),
            3 => dst.size = src.size,
            4 => dst.file_type = src.file_type.clone(),
            6 => dst.modified = src.modified.clone(),
            7 => dst.db_created = src.db_created.clone(),
            8 => dst.category = src.category.clone(),
            9 => dst.tags = src.tags.clone(),
            10 => if self.mode == ExplorerMode::Database { dst.hash = src.hash.clone(); },
            _ => {},
        }
    }

    fn compare_cell(&self, l: &Thumbnail, r: &Thumbnail, column: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        // Special columns: CLIP (5), Similarity (last if present)
        if column == 5 {
            // CLIP presence (Yes before No), considering both path and hash presence
            let l_has = self.clip_presence.contains(&l.path)
                || l.hash.as_ref().map(|h| self.clip_presence_hashes.contains(h)).unwrap_or(false);
            let r_has = self.clip_presence.contains(&r.path)
                || r.hash.as_ref().map(|h| self.clip_presence_hashes.contains(h)).unwrap_or(false);
            let sl = l_has as u8;
            let sr = r_has as u8;
            return sr.cmp(&sl);
        }
        let similarity_idx = if self.showing_similarity {
            match self.mode { ExplorerMode::FileSystem => 10, ExplorerMode::Database => 11 }
        } else { usize::MAX };
        if column == similarity_idx {
            let sl = self.similar_scores.get(&l.path).copied().unwrap_or(f32::MIN);
            let sr = self.similar_scores.get(&r.path).copied().unwrap_or(f32::MIN);
            return sr.partial_cmp(&sl).unwrap_or(Equal);
        }
        match column {
            0 => Equal, // thumbnail not sortable
            1 => l.filename.cmp(&r.filename),
            2 => l.path.cmp(&r.path),
            3 => l.size.cmp(&r.size),
            4 => l.file_type.cmp(&r.file_type),
            6 => l.modified.cmp(&r.modified),
            7 => l.db_created.cmp(&r.db_created),
            8 => l.category.cmp(&r.category),
            9 => l.tags.join(",").cmp(&r.tags.join(",")),
            10 => l.hash.cmp(&r.hash),
            _ => Equal,
        }
    }

    fn new_empty_row(&mut self) -> Thumbnail {
        Thumbnail::default()
    }

    fn on_highlight_change(&mut self, highlighted: &[&Thumbnail], unhighlighted: &[&Thumbnail]) {
        for row in unhighlighted.iter().filter(|r| r.file_type != "<DIR>") {
            self.selected.remove(&row.path);
        }
        for row in highlighted.iter().filter(|r| r.file_type != "<DIR>") {
            self.selected.insert(row.path.to_string());
            // Send a selection event to open preview pane via receiver
            let _ = self.thumbnail_tx.try_send((*row).clone());
        }
    }

    fn column_render_config(&mut self, column: usize, _is_editing: bool) -> TableColumnConfig {
        let base = TableColumnConfig::auto();
        match column {
            0 => base.at_least(75.).at_most(100.).resizable(true),  // thumbnail
            1 => base.at_least(160.).clip(true).resizable(true),            // name
            2 => base.at_least(220.).clip(true).resizable(true),            // path
            3 => base.at_least(70.).at_most(90.),                  // size
            4 => base.at_least(50.).at_most(60.),                  // type
            5 => base.at_least(50.).at_most(60.),                  // CLIP
            6 => base.at_least(100.).at_most(120.),                // modified
            7 => base.at_least(100.).at_most(120.),                // db created
            8 => base.at_least(110.).at_most(140.).resizable(true),// category
            9 => base.at_least(140.).clip(true).resizable(true),           // tags
            10 => base.at_least(140.).at_most(200.).clip(true),    // hash (DB)
            _ => base,
        }
    }

    // --- Custom context menu & actions ---
    fn custom_context_menu_items(
        &mut self,
        _context: &UiActionContext,
        selection: &SelectionSnapshot<'_, Thumbnail>,
    ) -> Vec<CustomMenuItem> {
        let has_dirs = selection
            .selected_rows
            .iter()
            .any(|(_, r)| r.file_type == "<DIR>" && !r.path.is_empty());
        let mut items = vec![
            CustomMenuItem::new(
                "generate_description", 
                "Generate Description"
            )
            .icon("⚡")
            .enabled(true),
            CustomMenuItem::new(
                "regenerate_description",
                "Regenerate Description (overwrite)"
            ).icon("♻").enabled(true),
            CustomMenuItem::new(
                "generate_clip_embeddings", 
                "Generate CLIP Embedding"
            )
            .icon("⚡")
            .enabled(true),
            CustomMenuItem::new(
                "regenerate_clip_embedding",
                "Regenerate CLIP Embedding"
            ).icon("♻").enabled(true),
            CustomMenuItem::new(
                "add_selection_to_group",
                "Add selection to active group"
            )
            .icon("➕")
            .enabled(true),
            CustomMenuItem::new(
                "remove_selection_from_group",
                "Remove selection from active group"
            )
            .icon("➖")
            .enabled(true),
            CustomMenuItem::new(
                "open_selection_in_new_tab",
                "Open selection in new tab"
            )
            .icon("🗂")
            .enabled(true),
            CustomMenuItem::new(
                "exclude_dirs_from_scan",
                "Exclude selected directories from recursive scans"
            )
            .icon("🚫")
            .enabled(has_dirs),
            CustomMenuItem::new(
                "find_similar",
                "Find Similar (CLIP)"
            ).icon("🔍").enabled(!selection.selected_rows.is_empty()),
            CustomMenuItem::new(
                "refine_selection",
                "Refine Selection"
            ).icon("🛠").enabled(!selection.selected_rows.is_empty()),
            // Future: rerank actions (category / tags) can be added once sorting hook in FileExplorer is exposed.
        ];
        // In Database mode, allow attaching to chat window
        if self.mode == ExplorerMode::Database {
            let can_attach = !selection.selected_rows.is_empty();
            items.push(
                CustomMenuItem::new(
                    "attach_to_chat_window",
                    "Attach to chat window"
                ).icon("📎").enabled(can_attach)
            );
        }
        items
    }

    fn on_custom_action_ex(
        &mut self,
        action_id: &'static str,
        ctx: &CustomActionContext<'_, Thumbnail>,
        _editor: &mut CustomActionEditor<Thumbnail>,
    ) {
        match action_id {
            "attach_to_chat_window" => {
                // Collect file paths from selected rows (skip directories)
                let paths: Vec<String> = ctx
                    .selection
                    .selected_rows
                    .iter()
                    .filter(|(_, r)| r.file_type != "<DIR>")
                    .map(|(_, r)| r.path.clone())
                    .collect();
                if !paths.is_empty() {
                    crate::ui::assistant::request_attach_to_chat(paths);
                }
            }
            "open_selection_in_new_tab" => {
                // Build a title from first selected name and count
                let rows: Vec<crate::database::Thumbnail> = ctx
                    .selection
                    .selected_rows
                    .iter()
                    .map(|(_, r)| (*r).clone())
                    .collect();
                let first = rows[0].filename.clone();
                let count = rows.len();
                let title = if count == 1 { format!("Selected: {}", first) } else { format!("Selected ({}) - {}", count, first) };
                crate::app::OPEN_TAB_REQUESTS
                    .lock()
                    .unwrap()
                    .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
            }
            "generate_description" => {
                let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                let prompt = self.ui_settings.ai_prompt_template.clone();
                // editor..set_cell(row_id, NAME, r)
                for (_, row) in ctx.selection.selected_rows.iter() {
                    if row.path.starts_with("zip://") || row.path.starts_with("tar://") || row.path.starts_with("7z://") { continue; }
                    if self.bulk_cancel_requested { break; }
                    if row.file_type == "<DIR>" { continue; }
                    if let Some(ext) = Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) { if !crate::is_image(ext.as_str()) { continue; } } else { continue; }
                    // Respect size bounds from UiSettings
                    if let Some(minb) = self.ui_settings.db_min_size_bytes { if row.size < minb { continue; } }
                    if let Some(maxb) = self.ui_settings.db_max_size_bytes { if row.size > maxb { continue; } }
                    // Respect overwrite setting: skip if existing data and overwrite disabled
                    let overwrite = self.ui_settings.overwrite_descriptions;
                    if !overwrite && (row.caption.is_some() || row.description.is_some()) { continue; }
                    let path_str = row.path.clone();
                    let path_str_clone = path_str.clone();
                    let tx_updates = self.ai_update_tx.clone();
                    let prompt_clone = prompt.clone();
                    let eng = engine.clone();
                    tokio::spawn(async move {
                        eng.stream_vision_description(Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
                            if let Some(vd) = final_opt {
                                let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Final {
                                    path: path_str.clone(),
                                    description: vd.description.clone(),
                                    caption: Some(vd.caption.clone()),
                                    category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                    tags: vd.tags.clone(),
                                });
                            } else {
                                let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Interim { path: path_str.clone(), text: interim.to_string() });
                            }
                        }).await;
                    });
                }
            }
            "regenerate_description" => {
                let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                let prompt = self.ui_settings.ai_prompt_template.clone();
                for (_, row) in ctx.selection.selected_rows.iter() {
                    if row.file_type == "<DIR>" { continue; }
                    if let Some(ext) = Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) { if !crate::is_image(ext.as_str()) { continue; } } else { continue; }
                    let path_str = row.path.clone();
                    let path_str_clone = path_str.clone();
                    let tx_updates = self.ai_update_tx.clone();
                    let prompt_clone = prompt.clone();
                    let eng = engine.clone();
                    tokio::spawn(async move {
                        eng.stream_vision_description(Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
                            if let Some(vd) = final_opt {
                                let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Final {
                                    path: path_str.clone(),
                                    description: vd.description.clone(),
                                    caption: Some(vd.caption.clone()),
                                    category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                    tags: vd.tags.clone(),
                                });
                            } else {
                                let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Interim { path: path_str.clone(), text: interim.to_string() });
                            }
                        }).await;
                    });
                }
            }
            "generate_clip_embeddings" => {
                for (_, row) in ctx.selection.selected_rows.clone() {
                    if row.path.starts_with("zip://") || row.path.starts_with("tar://") || row.path.starts_with("7z://") { continue; }
                    let path = row.path.clone();
                    // Skip if outside size bounds
                    if let Some(minb) = self.ui_settings.db_min_size_bytes { if row.size < minb { continue; } }
                    if let Some(maxb) = self.ui_settings.db_max_size_bytes { if row.size > maxb { continue; } }
                    let thumb = row.clone();
                    let tx = self.clip_embedding_tx.clone();
                    tokio::spawn(async move {
                        // Ensure engine and model are ready
                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                        let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await?;
                        let _ = tx.try_send(thumb.get_embedding().await?);
                        log::info!("[CLIP] Manual per-item generation: added {added} for {path}");
                        Ok::<(), anyhow::Error>(())
                    });
                }
            }
            "regenerate_clip_embedding" => {
                for (_, row) in ctx.selection.selected_rows.clone() {
                    if row.file_type == "<DIR>" { continue; }
                    let path = row.path.clone();
                    let thumb = row.clone();
                    let tx = self.clip_embedding_tx.clone();
                    tokio::spawn(async move {
                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                        // Force re-generation by not checking existing presence
                        let _ = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await?;
                        let _ = tx.try_send(thumb.get_embedding().await?);
                        Ok::<(), anyhow::Error>(())
                    });
                }
            }
            "add_selection_to_group" => {
                if let Some(group_name) = crate::ui::file_table::active_group_name() {
                    // Collect paths -> upsert and associate
                    let rows: Vec<crate::database::Thumbnail> = ctx
                        .selection
                        .selected_rows
                        .iter()
                        .map(|(_, r)| (*r).clone())
                        .filter(|r| r.file_type != "<DIR>")
                        .collect();
                    tokio::spawn(async move {
                        if let Ok(ids) = crate::database::upsert_rows_and_get_ids(rows).await {
                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                            }
                        }
                    });
                } else {
                    log::warn!("No active logical group; cannot add selection");
                }
            }
            "remove_selection_from_group" => {
                if let Some(group_name) = crate::ui::file_table::active_group_name() {
                    let paths: Vec<String> = ctx
                        .selection
                        .selected_rows
                        .iter()
                        .map(|(_, r)| r.path.clone())
                        .filter(|p| !p.is_empty())
                        .collect();
                    tokio::spawn(async move {
                        if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                            // Load ids for given paths
                            let mut ids: Vec<surrealdb::RecordId> = Vec::new();
                            for p in paths.into_iter() {
                                if let Ok(Some(id)) = crate::Thumbnail::get_thumbnail_id_by_path(&p).await {
                                    ids.push(id);
                                }
                            }
                            // Remove each id
                            for id in ids.iter() {
                                let _ = crate::database::LogicalGroup::remove_thumbnail(&g.id, id).await;
                            }
                        }
                    });
                } else {
                    log::warn!("No active logical group; cannot remove selection");
                }
            }
            "exclude_dirs_from_scan" => {
                // Collect selected directory paths
                let mut add: Vec<String> = ctx
                    .selection
                    .selected_rows
                    .iter()
                    .filter_map(|(_, r)| if r.file_type == "<DIR>" { Some(r.path.clone()) } else { None })
                    .filter(|p| !p.trim().is_empty())
                    .collect();
                if add.is_empty() { return; }
                // Ensure settings vector exists and extend with unique values
                if self.ui_settings.excluded_dirs.is_none() {
                    self.ui_settings.excluded_dirs = Some(Vec::new());
                }
                if let Some(ref mut dirs) = self.ui_settings.excluded_dirs {
                    for p in add.drain(..) {
                        if !dirs.contains(&p) {
                            dirs.push(p);
                        }
                    }
                }
                crate::database::settings::save_settings(&self.ui_settings);
                log::info!("Added selected directories to excluded_dirs for recursive scans");
            }
            "find_similar" => {
                if let Some((_, first)) = ctx.selection.selected_rows.first() {
                    let origin = first.path.clone();
                    // Request embedding and open a similarity tab via AIUpdate path used elsewhere
                    let tx_updates = self.ai_update_tx.clone();
                    tokio::spawn(async move {
                        if let Ok(Some(thumb)) = crate::Thumbnail::get_thumbnail_by_path(&origin).await {
                            if let Ok(embed_row) = thumb.clone().get_embedding().await {
                                let embed_vec = embed_row.embedding.clone();
                                if embed_vec.is_empty() { return; }
                                if let Ok(hits) = crate::database::ClipEmbeddingRow::find_similar_by_embedding(&embed_vec, 256, 256, 0).await {
                                    let mut rows_out = Vec::new();
                                    let mut scores_out = std::collections::HashMap::new();
                                    for hit in hits.into_iter() {
                                        if hit.path == origin { continue; }
                                        let t = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                        // Convert distance (lower better) to similarity (higher better)
                                        let cosine_sim = 1.0 - hit.dist; // if dist = 1 - cos_sim
                                        let norm_sim = ((cosine_sim + 1.0) / 2.0).clamp(0.0, 1.0); // map [-1,1] -> [0,1]
                                        rows_out.push(t.clone());
                                        scores_out.insert(hit.path.clone(), norm_sim);
                                        if rows_out.len() >= 96 { break; }
                                    }
                                    let mut results = Vec::with_capacity(rows_out.len());
                                    for t in rows_out.into_iter() { 
                                        let score = scores_out.get(&t.path).copied();
                                        results.push(crate::ui::file_table::SimilarResult { thumb: t, created: None, updated: None, similarity_score: score, clip_similarity_score: score });
                                    }
                                    let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: origin.clone(), results });
                                }
                            }
                        }
                    });
                }
            }
            "refine_selection" => {
                // Collect paths (non-dirs)
                let paths: Vec<String> = ctx.selection.selected_rows.iter().filter(|(_, r)| r.file_type != "<DIR>").map(|(_, r)| r.path.clone()).collect();
                if !paths.is_empty() { crate::ui::refine::request_refine_for_paths(paths); }
            }
            _ => {},
        }
    }
}

/* ------------------------------------------- Helpers ------------------------------------------ */

fn color_for_tag(tag: &str) -> Color32 {
    // Simple stable hash -> color mapping
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    tag.hash(&mut h);
    let v = h.finish();
    let r = (v & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let b = ((v >> 16) & 0xFF) as u8;
    Color32::from_rgb(r / 2 + 64, g / 2 + 64, b / 2 + 64)
}
