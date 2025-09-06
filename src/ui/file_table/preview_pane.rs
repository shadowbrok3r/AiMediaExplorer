use eframe::egui::*;
use humansize::DECIMAL;
// use surrealdb::RecordId; // no longer needed: presence check relies on in-memory metadata only

impl super::FileExplorer {
    pub fn preview_pane(&mut self, ui: &mut Ui) {
        SidePanel::right("MainPageRightPanel")
            .default_width(400.)
            .show_animated_inside(ui, self.open_preview_pane, |ui| {
                // Auto-follow active vision path asynchronously (avoid blocking UI thread)
                if self.open_preview_pane { // only poll when pane visible
                    let cur_path = self.current_thumb.path.clone();
                    let tx_thumb = self.thumbnail_tx.clone();
                    let preview_follow = self.follow_active_vision; // assume a bool field (add if not existing)
                    if preview_follow {
                        tokio::spawn(async move {
                            if let Some(active) = crate::ai::GLOBAL_AI_ENGINE.get_active_vision_path().await {
                                if !active.is_empty() && active != cur_path {
                                    if let Ok(row_opt) = crate::Thumbnail::get_thumbnail_by_path(&active).await {
                                        if let Some(row) = row_opt { let _ = tx_thumb.send(row); }
                                    }
                                }
                            }
                        });
                    }
                    // Consume any updated thumb from channel (non-blocking)
                    while let Ok(new_thumb) = self.thumbnail_rx.try_recv() {
                        if new_thumb.path != self.current_thumb.path { self.current_thumb = new_thumb; }
                    }
                }
                ui.vertical_centered(|ui| {
                    let name = &self.current_thumb.filename;
                    let cache_key = &self.current_thumb.path;
                    let thumb_cache = &self.viewer.thumb_cache;
                    ui.heading(name);

                    super::get_img_ui(thumb_cache, cache_key, ui);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Size:").underline().strong());
                        ui.label(humansize::format_size(self.current_thumb.size, DECIMAL));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label(&self.current_thumb.file_type);
                            // CLIP embedding presence indicator (use in-memory metadata only)
                            // Refresh clip presence cache asynchronously (fire & forget throttled by absence of entry)
                            let path_for_clip = self.current_thumb.path.clone();
                            if !self.clip_presence.contains_key(&path_for_clip) && !path_for_clip.is_empty() {
                                let tx = self.clip_presence_tx.clone();
                                let p = path_for_clip.clone();
                                tokio::spawn(async move {
                                    let has = crate::ai::GLOBAL_AI_ENGINE
                                        .get_file_metadata(&p)
                                        .await
                                        .map(|m| m.clip_embedding.is_some())
                                        .unwrap_or(false);
                                    let _ = tx.try_send((p, has));
                                });
                            }
                            let have_clip = *self.clip_presence.get(&path_for_clip).unwrap_or(&false);
                            let badge = if have_clip || self.current_thumb.embedding.is_some() { 
                                RichText::new("CLIP ✓").color(Color32::LIGHT_GREEN) 
                            } else { 
                                RichText::new("CLIP X").color(ui.style().visuals.error_fg_color) 
                            };

                            ui.label(badge);
                            ui.label(RichText::new("Type:").underline().strong());
                        });
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Open").clicked() {
                            let _ = open::that(self.current_thumb.path.clone());
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.button("Show in Folder").clicked() {
                                let pb = std::path::PathBuf::from(self.current_thumb.path.clone());
                                if let Some(parent) = pb.parent() { let _ = open::that(parent); }
                            }
                        });        
                    });

                    ScrollArea::vertical()
                        .auto_shrink(false)
                        .show(ui, |ui| {
                            // -------- AI Description & Editable Metadata --------
                            ui.heading("Ai Description");
                            ui.group(|ui| {
                                let path_key = self.current_thumb.path.clone();

                                // Streaming interim text (if any)
                                if let Some(interim) = self.streaming_interim.get(&path_key) {
                                    ui.colored_label(Color32::LIGHT_BLUE, RichText::new("(Streaming…)"));
                                    ScrollArea::vertical().max_height(160.).show(ui, |ui| {
                                        ui.label(interim);
                                    });
                                    ui.separator();
                                }

                                // Description editor
                                if let Some(desc) = &mut self.current_thumb.description {
                                    let resp = TextEdit::multiline(desc)
                                        .desired_width(ui.available_width())
                                        .desired_rows(6)
                                        .ui(ui);
                                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let path_clone = path_key.clone();
                                        let desc_clone = desc.clone();
                                        tokio::spawn(async move {
                                            if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                                let mut new_meta = meta.clone();
                                                new_meta.description = Some(desc_clone);
                                                if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited description failed: {e}"); }
                                            }
                                        });
                                    }
                                } else if self.streaming_interim.get(&path_key).is_none() {
                                    ui.label("No Description yet");
                                }

                                ui.add_space(8.);
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("Category:").underline().strong());
                                    let mut cat = self.current_thumb.category.clone().unwrap_or_default();
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        let cat_resp = TextEdit::singleline(&mut cat).desired_width(ui.available_width()).ui(ui);
                                        if cat_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                            self.current_thumb.category = if cat.trim().is_empty() { None } else { Some(cat.clone()) };
                                            let path_clone = path_key.clone();
                                            let cat_opt = self.current_thumb.category.clone();
                                            tokio::spawn(async move {
                                                if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                                    let mut new_meta = meta.clone();
                                                    new_meta.category = cat_opt.clone();
                                                    if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited category failed: {e}"); }
                                                }
                                            });
                                        }
                                    });
                                });

                                ui.add_space(8.);
                                ui.label(RichText::new("Caption:").underline().strong());
                                let mut cap = self.current_thumb.caption.clone().unwrap_or_default();
                                let cap_resp = TextEdit::multiline(&mut cap).desired_width(ui.available_width()).desired_rows(2).ui(ui);
                                if cap_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    self.current_thumb.caption = if cap.trim().is_empty() { None } else { Some(cap.clone()) };
                                    let path_clone = path_key.clone();
                                    let cap_opt = self.current_thumb.caption.clone();
                                    tokio::spawn(async move {
                                        if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                            let mut new_meta = meta.clone();
                                            new_meta.caption = cap_opt.clone();
                                            if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited caption failed: {e}"); }
                                        }
                                    });
                                }

                                ui.add_space(8.);
                                ui.label(RichText::new("Tags (comma-separated):").underline().strong());
                                let mut tags_line = self.current_thumb.tags.join(", ");
                                let tags_resp = TextEdit::singleline(&mut tags_line).desired_width(ui.available_width()).ui(ui);
                                if tags_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    let new_tags: Vec<String> = tags_line.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                                    self.current_thumb.tags = new_tags.clone();
                                    let path_clone = path_key.clone();
                                    tokio::spawn(async move {
                                        if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                            let mut new_meta = meta.clone();
                                            new_meta.tags = new_tags.clone();
                                            if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { 
                                                log::warn!("Persist edited tags failed: {e}"); 
                                            }
                                        }
                                    });
                                }
                                ui.horizontal_wrapped(|ui| {
                                    for tag in &self.current_thumb.tags { ui.label(RichText::new(tag)); }
                                });

                                // Completeness warning
                                let have_all = self.current_thumb.caption.is_some()
                                    && self.current_thumb.category.is_some()
                                    && !self.current_thumb.tags.is_empty()
                                    && self.current_thumb.description.as_ref().map(|d| !d.trim().is_empty()).unwrap_or(false);
                                if !have_all && self.current_thumb.description.is_some() && self.streaming_interim.get(&path_key).is_none() {
                                    ui.colored_label(Color32::YELLOW, "Parsed AI metadata incomplete (expected caption, category, tags, description)");
                                }

                                ui.add_space(10.);
                                ui.horizontal(|ui| {
                                    let overwrite_allowed = self.viewer.ui_settings.overwrite_descriptions;
                                    let already_has = self.current_thumb.description.is_some();
                                    let can_generate = !self.streaming_interim.contains_key(&path_key) && (!already_has || overwrite_allowed);
                                    let btn_text = if already_has { if overwrite_allowed { "Regenerate (Overwrite)" } else { "Generated" } } else { "Generate Description" };
                                
                                    let btn = ui.add_enabled(can_generate, Button::new(btn_text));
                                    if self.streaming_interim.contains_key(&path_key) {
                                        ui.label(RichText::new("Generating…").weak());
                                        Spinner::new().ui(ui);
                                    }
                                    if btn.clicked() && can_generate {
                                        // Start streaming generation for this single file
                                        let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                        let path_for_stream = path_key.clone();
                                        self.streaming_interim.insert(path_for_stream.clone(), String::new());
                                        let tx_updates = self.viewer.ai_update_tx.clone();
                                        let prompt = self.viewer.ui_settings.ai_prompt_template.clone();
                                        tokio::spawn(async move {
                                            let path_string = path_for_stream;
                                            let arc_path = std::sync::Arc::new(path_string);
                                            let engine_call_path = (*arc_path).clone();
                                            let cb_path = arc_path.clone();
                                            engine.stream_vision_description(std::path::Path::new(&engine_call_path), &prompt, move |interim, final_opt| {
                                                let p_clone = (*cb_path).clone();
                                                if let Some(vd) = final_opt {
                                                    let _ = tx_updates.try_send(super::AIUpdate::Final {
                                                        path: p_clone,
                                                        description: vd.description.clone(),
                                                        caption: Some(vd.caption.clone()),
                                                        category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                                        tags: vd.tags.clone(),
                                                    });
                                                } else {
                                                    let _ = tx_updates.try_send(super::AIUpdate::Interim { path: p_clone, text: interim.to_string() });
                                                }
                                            }).await;
                                        });
                                    }
                                    
                                    // Similarity search button (requires clip embedding availability; will fall back to generating on-demand)
                                    if ui.button("Find Similar").clicked() {
                                        let sel_path = path_key.clone();
                                        let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                        let tx_updates = self.viewer.ai_update_tx.clone();
                                        self.similar_results.clear();
                                        self.show_similar_modal = true;
                                        tokio::spawn(async move {
                                            // Ensure embedding exists; generate if missing
                                            if let Some(meta) = engine.get_file_metadata(&sel_path).await {
                                                if meta.clip_embedding.is_none() { let _ = engine.clip_generate_for_paths(&[sel_path.clone()]).await; }
                                            }
                                            let results = engine.clip_search_image(&sel_path, 50).await;
                                            let _ = tx_updates.try_send(super::AIUpdate::SimilarResults { origin_path: sel_path.clone(), results });
                                        });
                                    }
                                    
                                    if ui.button("Generate CLIP").clicked() {
                                        let path = self.current_thumb.path.clone();
                                        tokio::spawn(async move {
                                            // Ensure engine and model are ready
                                            let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                            let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await?;
                                            log::info!("[CLIP] Manual per-item generation: added {added} for {path}");
                                            Ok::<(), anyhow::Error>(())
                                        });
                                    }
                                });
                            });
                        });
                });
            });
    }
}