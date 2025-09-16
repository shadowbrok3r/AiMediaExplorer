use egui::{containers::menu::MenuConfig, style::StyleModifier};
use eframe::egui::*;

pub mod logical_groups;
pub mod selection_menu;
pub mod filter_menu;
pub mod table_menu;
pub mod scan_menu;

impl super::FileExplorer {
    pub fn navbar(&mut self, ui: &mut Ui) {
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
                    if self.open_quick_access { RichText::new("⚙").font(font.clone()).color(err_color) } else { RichText::new("⚙").font(font.clone()) }
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
                if Button::new(RichText::new("🏠").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_home(); }
                if self.viewer.mode == super::table::ExplorerMode::Database {
                    if Button::new(RichText::new("🗙").font(font.clone()).color(err_color)).min_size(size).ui(ui).on_hover_text("Double click to Clear Database Table").double_clicked() { self.table.clear(); self.table_index.clear(); }
                }
                ui.separator();
                let path_edit = TextEdit::singleline(&mut self.current_path)
                .hint_text(if self.viewer.mode == super::table::ExplorerMode::Database {
                    if self.ai_search_enabled { "AI Search (text prompt)" } else { "Path Prefix Filter" }
                } else {
                    "Current Directory"
                })
                .desired_width(400.)
                .ui(ui);

                match self.viewer.mode {
                    super::table::ExplorerMode::FileSystem => {
                        if path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                            self.refresh();
                        }
                    },
                    super::table::ExplorerMode::Database => {
                        ui.checkbox(&mut self.ai_search_enabled, "AI").on_hover_text("Use AI semantic search across the database (text prompt)");

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
                                                        let (mut created, mut updated, mut clip_sim) = (None, None, None);
                                                        if let Ok(rows) = crate::database::ClipEmbeddingRow::load_clip_embeddings_for_path(&thumb.path).await {
                                                            for row in rows.iter() {
                                                                created = row.created.clone();
                                                                updated = row.updated.clone();
                                                                if clip_sim.is_none() && !row.embedding.is_empty() { clip_sim = Some(crate::ai::clip::dot(&q, &row.embedding)); }
                                                            }
                                                        }
                                                        // If we have distance from the hit, convert it (lower distance => higher similarity)
                                                        let cosine_sim = 1.0 - hit.dist;
                                                        let norm_sim = ((cosine_sim + 1.0)/2.0).clamp(0.0, 1.0);
                                                        let final_sim = clip_sim.unwrap_or(norm_sim);
                                                        results.push(crate::ui::file_table::SimilarResult { thumb, created, updated, similarity_score: Some(final_sim), clip_similarity_score: Some(final_sim) });
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
                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    
                    self.filter_menu(ui);

                    if ui.button(if self.open_preview_pane { RichText::new("👁").color(err_color) } else { RichText::new("👁").font(font) }).clicked() {
                        self.open_preview_pane = !self.open_preview_pane;
                    };

                    ui.separator();

                    TextEdit::singleline(&mut self.viewer.filter)
                    .desired_width(200.0)
                    .hint_text("Search for files")
                    .ui(ui);

                    ui.separator();

                    self.scan_menu(ui);
                    if self.viewer.showing_similarity {
                        if self.similarity_origin_path.is_none() && !self.current_path.is_empty() {
                            self.similarity_origin_path = Some(self.current_path.clone());
                        }
                        ui.separator();
                        if ui.button(format!("Load {} more", self.similarity_batch_size)).on_hover_text("Append more similar results").clicked() {
                            self.append_more_similar();
                        }
                    }
                    self.table_menu(ui);
                });
            });
        });
    }
}