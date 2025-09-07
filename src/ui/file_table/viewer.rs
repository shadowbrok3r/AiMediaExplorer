use egui::{Color32, FontId, Image, ImageSource, KeyboardShortcut, Stroke, Vec2, Widget};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use std::{borrow::Cow, collections::HashMap, path::Path, sync::Arc};
use egui_data_table::{
    viewer::{default_hotkeys, CustomActionContext, CustomActionEditor, RowCodec, UiActionContext}, CustomMenuItem, RowViewer, SelectionSnapshot, UiAction
};
use egui_extras::Column as TableColumnConfig;
use crossbeam::channel::Sender;
use humansize::DECIMAL;
use serde::Serialize;

use super::codec::ThumbCodec;
use crate::{generate_image_thumb_data, generate_video_thumb_data, ui::file_table::AIUpdate, Thumbnail, UiSettings};

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExplorerMode {
    #[default]
    FileSystem,
    Database,
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
}

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
            ai_update_tx
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
        match self.mode {
            ExplorerMode::FileSystem => 8,
            ExplorerMode::Database => 10,
        }
    }

    fn column_name(&mut self, column: usize) -> std::borrow::Cow<'static, str> {
        match self.mode {
            ExplorerMode::FileSystem => [
                "", "Name", "Path", "Category", "Tags", "Modified", "Size", "Type",
            ][column]
                .into(),
            ExplorerMode::Database => [
                "",
                "Name",
                "Path",
                "Category",
                "Tags",
                "Modified",
                "Size",
                "Type",
                "Hash",
                "DB Created",
            ][column]
                .into(),
        }
    }

    fn is_sortable_column(&mut self, column: usize) -> bool {
        column < self.num_columns()
    }

    fn row_filter_hash(&mut self) -> &impl std::hash::Hash {
        &self.filter
    }

    fn filter_row(&mut self, row: &Thumbnail) -> bool {
        match self.mode {
            ExplorerMode::FileSystem | ExplorerMode::Database => {
                if self.filter.trim().is_empty() {
                    return true;
                }
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

    fn hotkeys(&mut self, context: &UiActionContext) -> Vec<(KeyboardShortcut, UiAction)> {
        let hot = default_hotkeys(context);
        self.hotkeys.clone_from(&hot);
        hot
    }

    fn is_editable_cell(&mut self, column: usize, _row: usize, _row_value: &Thumbnail) -> bool {
        // Only allow editing for Category (3) and Tags (4) regardless of mode.
        matches!(column, 3 | 4)
    }

    fn show_cell_view(&mut self, ui: &mut egui::Ui, row: &Thumbnail, column: usize) {
        match column {
            0 => {
                // Thumbnail (mode-agnostic currently)
                if row.file_type == "<DIR>" {
                    ui.add_sized(
                        ui.available_size(),
                        egui::Image::new(eframe::egui::include_image!(
                            "../../../assets/Icons/folder.png"
                        )),
                    );
                    return;
                }
                let cache_key = row.path.clone();
                if row.thumbnail_b64.is_none() {
                    ui.label("📄");
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
                            .max_size(ui.available_size())
                            .ui(ui); // .bg_fill(Color32::WHITE)
                    } else {
                        ui.label("❌");
                    }
                }
            }
            1 => {
                // Name
                ui.label(&row.filename);
            }
            2 => {
                // Path or DB key depending on mode
                match self.mode {
                    ExplorerMode::FileSystem => ui.label(&row.path),
                    ExplorerMode::Database => ui.label(&row.path), // placeholder; could show DB id
                };
            }
            3 => {
                // Category
                ui.label(row.category.as_deref().unwrap_or("-"));
            }
            4 => {
                // Tags as chips
                if row.tags.is_empty() {
                    ui.label("-");
                } else {
                    ui.horizontal(|ui| {
                        for tag in &row.tags {
                            let color = color_for_tag(tag);
                            let txt = format!("{}", tag);
                            let pad = egui::vec2(6.0, 2.0);
                            let galley = ui.painter().layout_no_wrap(
                                txt.clone(),
                                FontId::monospace(12.),
                                Color32::WHITE,
                            );
                            let (rect, resp) = ui.allocate_exact_size(
                                galley.rect.size() + pad * 2.0,
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 6.0, color);
                            ui.painter().galley(rect.min + pad, galley, Color32::WHITE);
                            let stroke = Stroke {
                                width: 1.0,
                                color: color.gamma_multiply(0.6),
                            };
                            ui.painter().rect_filled(rect, 6.0, color);
                            ui.painter()
                                .rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Outside);
                            if resp.hovered() {
                                ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                            }
                        }
                    });
                }
            }
            5 => {
                // Modified mm/dd/yyyy (or '-')
                use chrono::Datelike;
                if let Some(m) = &row.modified {
                    ui.label(format!("{}/{}/{}", m.month(), m.day(), m.year()));
                } else {
                    ui.label("-");
                }
            }
            6 => {
                // Size
                if row.size == 0 {
                    ui.label("-");
                } else {
                    ui.label(humansize::format_size(row.size, DECIMAL));
                }
            }
            7 => {
                // Type
                ui.label(&row.file_type);
            }
            8 => {
                // Hash (DB mode only)
                if self.mode == ExplorerMode::Database {
                    ui.label(row.hash.as_deref().unwrap_or("-"));
                }
            }
            9 => {
                // DB Created date (DB mode only)
                if self.mode == ExplorerMode::Database {
                    if let Some(c) = &row.db_created {
                        use chrono::Datelike;
                        ui.label(format!("{}/{}/{}", c.month(), c.day(), c.year()));
                    }
                }
            }
            _ => unreachable!(),
        };
    }

    fn show_cell_editor(
        &mut self,
        ui: &mut egui::Ui,
        row: &mut Thumbnail,
        column: usize,
    ) -> Option<egui::Response> {
        match column {
            0 => {
                let cache_key = row.path.clone();
                if row.thumbnail_b64.is_none() {
                    Some(ui.label("📄"))
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
            4 => {
                // Tags editable
                let mut tag_str = row.tags.join(", ");
                let resp = ui.text_edit_singleline(&mut tag_str);
                if resp.changed() {
                    row.tags = tag_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Some(resp)
            }
            3 => {
                // Category
                let mut v = row.category.clone().unwrap_or_default();
                let resp = ui.text_edit_singleline(&mut v);
                if resp.changed() {
                    row.category = if v.is_empty() { None } else { Some(v) };
                }
                Some(resp)
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
                if resp.hovered() {
                    resp.clone().on_hover_text(&row.path);
                }
            }
            _ => {
                if resp.clicked() {
                    if self.mode == ExplorerMode::Database {
                        // In DB mode, disable on-demand generation attempts (no filesystem walk triggers)
                    } else {
                        if row.thumbnail_b64.is_none() {
                            match row.file_type.as_str() {
                                "video" => {
                                    if let Ok(b64) = generate_video_thumb_data(Path::new(&row.path))
                                    {
                                        let thumb = Thumbnail {
                                            thumbnail_b64: Some(b64),
                                            ..row.clone()
                                        };
                                        let _ = self.thumbnail_tx.try_send(thumb.clone());
                                        let tx = self.clip_embedding_tx.clone();
                                        tokio::spawn(async move {
                                            let _ = tx.try_send(thumb.get_embedding().await.unwrap_or_default());
                                        });
                                    }
                                }
                                "image" => {
                                    if let Ok(b64) = generate_image_thumb_data(Path::new(&row.path))
                                    {
                                        // embedding presence is tracked via clip embeddings table now
                                        let thumb = Thumbnail {
                                            thumbnail_b64: Some(b64),
                                            ..row.clone()
                                        };
                                        let _ = self.thumbnail_tx.try_send(thumb.clone());
                                        let tx = self.clip_embedding_tx.clone();
                                        tokio::spawn(async move {
                                            let _ = tx.try_send(thumb.get_embedding().await.unwrap_or_default());
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    let _ = self.thumbnail_tx.try_send(row.clone());
                }
            }
        }
        None
    }

    fn set_cell_value(&mut self, src: &Thumbnail, dst: &mut Thumbnail, column: usize) {
        match column {
            0 => { /* thumbnail placeholder - no data */ }
            1 => dst.filename = src.filename.clone(),
            2 => dst.path = src.path.clone(),
            3 => dst.category = src.category.clone(),
            4 => dst.tags = src.tags.clone(),
            5 => dst.modified = src.modified.clone(),
            6 => dst.size = src.size,
            7 => dst.file_type = src.file_type.clone(),
            8 => dst.hash = src.hash.clone(),
            9 => dst.db_created = src.db_created.clone(),
            _ => unreachable!(),
        }
    }

    fn compare_cell(&self, l: &Thumbnail, r: &Thumbnail, column: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match column {
            0 => std::cmp::Ordering::Equal, // thumbnail not sortable
            1 => l.filename.cmp(&r.filename),
            2 => l.path.cmp(&r.path),
            3 => l.category.cmp(&r.category),
            4 => l.tags.join(",").cmp(&r.tags.join(",")),
            5 => l.modified.cmp(&r.modified),
            6 => l.size.cmp(&r.size),
            7 => l.file_type.cmp(&r.file_type),
            8 => l.hash.cmp(&r.hash),
            9 => l.db_created.cmp(&r.db_created),
            _ => Equal,
        }
    }

    fn new_empty_row(&mut self) -> Thumbnail {
        Thumbnail::default()
    }

    fn column_render_config(&mut self, column: usize, _is_editing: bool) -> TableColumnConfig {
        let base = TableColumnConfig::auto();
        match column {
            0 => base.at_least(50.).at_most(75.).resizable(true), // thumbnail
            1 => base.at_least(160.).clip(true).resizable(true),  // name
            2 => base.at_least(220.).clip(true).resizable(true),  // path
            3 => base.at_least(110.).at_most(140.).resizable(true), // category
            4 => base.at_least(140.).clip(true).resizable(true),  // tags
            5 => base.at_least(100.).at_most(120.),               // modified
            6 => base.at_least(70.).at_most(70.),                 // size
            7 => base.at_least(50.).at_most(60.),                 // type
            8 => base.at_least(140.).at_most(200.).clip(true),    // hash
            9 => base.at_least(110.).at_most(130.),               // db created
            _ => base,
        }
    }

    // --- Custom context menu & actions ---
    fn custom_context_menu_items(
        &mut self,
        _context: &UiActionContext,
        _selection: &SelectionSnapshot<'_, Thumbnail>,
    ) -> Vec<CustomMenuItem> {
        vec![
            CustomMenuItem::new(
                "generate_description", 
                "Generate Description"
            )
            .icon("⚡")
            .enabled(true),
            CustomMenuItem::new(
                "generate_clip_embeddings", 
                "Generate CLIP Embedding"
            )
            .icon("⚡")
            .enabled(true)
        ]
    }

    fn on_custom_action_ex(
        &mut self,
        action_id: &'static str,
        ctx: &CustomActionContext<'_, Thumbnail>,
        _editor: &mut CustomActionEditor<Thumbnail>,
    ) {
        match action_id {
            "generate_description" => {
                let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                let prompt = self.ui_settings.ai_prompt_template.clone();
                // editor..set_cell(row_id, NAME, r)
                for (_, row) in ctx.selection.selected_rows.iter() {
                    if self.bulk_cancel_requested { break; }
                    if row.file_type == "<DIR>" { continue; }
                    if let Some(ext) = Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) { if !crate::is_image(ext.as_str()) { continue; } } else { continue; }
                    if row.caption.is_some() || row.description.is_some() { continue; }
                    let path_str = row.path.clone();
                    let path_str_clone = path_str.clone();
                    let tx_updates = self.ai_update_tx.clone();
                    let prompt_clone = prompt.clone();
                    let eng = engine.clone();
                    tokio::spawn(async move {
                        eng.stream_vision_description(Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
                            if let Some(vd) = final_opt {
                                let _ = tx_updates.try_send(super::AIUpdate::Final {
                                    path: path_str.clone(),
                                    description: vd.description.clone(),
                                    caption: Some(vd.caption.clone()),
                                    category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                    tags: vd.tags.clone(),
                                });
                            } else {
                                let _ = tx_updates.try_send(super::AIUpdate::Interim { path: path_str.clone(), text: interim.to_string() });
                            }
                        }).await;
                    });
                }
            }
            "generate_clip_embeddings" => {
                for (_, row) in ctx.selection.selected_rows.clone() {
                    let path = row.path.clone();
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
