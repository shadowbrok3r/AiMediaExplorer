use eframe::egui::*;
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};

impl crate::ui::file_table::FileExplorer {
    pub fn scan_menu(&mut self, ui: &mut Ui) {
        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        
        MenuButton::new("Scan")
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
        .ui(ui, |ui| {
            ui.vertical_centered_justified(|ui| { 
                ui.set_width(250.);
                ui.vertical_centered(|ui| 
                    ui.heading(RichText::new("Recursive Scanning").underline().color(ui.style().visuals.error_fg_color))
                );
                if Button::new("Recursive Scan").right_text("💡").ui(ui).clicked() {
                    let title = format!("Scan: {}", self.current_path);
                    let path = self.current_path.clone();
                    crate::app::OPEN_TAB_REQUESTS
                        .lock()
                        .unwrap()
                        .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive: true, background: false });
                }
                if Button::new("Scan Images").right_text("🖼").ui(ui).on_hover_text("Recursive scan including only images").clicked() {
                    self.recursive_scan = true;
                    self.scan_done = false;
                    self.table.clear();
                    self.table_index.clear();
                    self.last_scan_rows.clear();
                    self.last_scan_paths.clear();
                    self.last_scan_root = Some(self.current_path.clone());
                    self.file_scan_progress = 0.0;
                    let scan_id = crate::next_scan_id();
                    let tx = self.scan_tx.clone();
                    let recurse = true;
                    let mut filters = crate::Filters::default();
                    filters.root = std::path::PathBuf::from(self.current_path.clone());
                    filters.include_images = true;
                    filters.include_videos = false;
                    filters.include_archives = false;
                    filters.excluded_terms = self.excluded_terms.clone();
                    self.owning_scan_id = Some(scan_id);
                    self.current_scan_id = Some(scan_id);
                    tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                }
                if Button::new("Scan Videos").right_text("📹").ui(ui).on_hover_text("Recursive scan including only videos").clicked() {
                    self.recursive_scan = true;
                    self.scan_done = false;
                    self.table.clear();
                    self.table_index.clear();
                    self.last_scan_rows.clear();
                    self.last_scan_paths.clear();
                    self.last_scan_root = Some(self.current_path.clone());
                    self.file_scan_progress = 0.0;
                    let scan_id = crate::next_scan_id();
                    let tx = self.scan_tx.clone();
                    let recurse = true;
                    let mut filters = crate::Filters::default();
                    filters.root = std::path::PathBuf::from(self.current_path.clone());
                    filters.include_images = false;
                    filters.include_videos = true;
                    filters.include_archives = false;
                    filters.excluded_terms = self.excluded_terms.clone();
                    self.owning_scan_id = Some(scan_id);
                    self.current_scan_id = Some(scan_id);
                    tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                }
                if Button::new("Scan Archives (.zip/.7z/.rar/.tar)").right_text("🗄").ui(ui).on_hover_text("Recursive scan including only archive containers").clicked() {
                    self.recursive_scan = true;
                    self.scan_done = false;
                    self.table.clear();
                    self.table_index.clear();
                    self.last_scan_rows.clear();
                    self.last_scan_paths.clear();
                    self.last_scan_root = Some(self.current_path.clone());
                    self.file_scan_progress = 0.0;
                    let scan_id = crate::next_scan_id();
                    let tx = self.scan_tx.clone();
                    let recurse = true;
                    let mut filters = crate::Filters::default();
                    filters.root = std::path::PathBuf::from(self.current_path.clone());
                    filters.include_images = false;
                    filters.include_videos = false;
                    filters.include_archives = true;
                    filters.excluded_terms = self.excluded_terms.clone();
                    self.owning_scan_id = Some(scan_id);
                    self.current_scan_id = Some(scan_id);
                    tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                }
                
                if Button::new("Re-scan Current Folder").right_text("🔄").ui(ui).on_hover_text("Shallow scan with current filters (applies 'Skip likely icons')").clicked() {
                    self.recursive_scan = false;
                    self.scan_done = false;
                    self.table.clear();
                    self.table_index.clear();
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
                    // Lock incoming scan updates to this tab by setting owning + current scan ids
                    self.owning_scan_id = Some(scan_id);
                    self.current_scan_id = Some(scan_id);
                    tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                }
    
                if Button::new("Cancel Scan").right_text(RichText::new("■").color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text("Cancel active recursive scan").clicked() {
                    if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
                }
                // (Performance Tweaks moved to Quick Access pane)
                ui.separator();
                ui.vertical_centered(|ui| 
                    ui.heading(RichText::new("Database Save").underline().color(ui.style().visuals.error_fg_color))
                );
                if Button::new("Save Current View to DB").right_text(RichText::new("💾")).ui(ui).on_hover_text("Upsert all currently visible rows into the database and add them to the active logical group").clicked() {
                    let group_opt = self.active_logical_group_name.clone();
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
                                    if let Some(group_name) = group_opt {
                                        match crate::database::LogicalGroup::get_by_name(&group_name).await {
                                            Ok(Some(g)) => {
                                                if let Err(e) = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await {
                                                    log::error!("Add thumbs to group failed: {e:?}");
                                                } else {
                                                    log::info!("Saved {} rows to DB and associated with group '{}'", ids.len(), group_name);
                                                }
                                            }
                                            Ok(None) => log::warn!("Active group '{}' not found during save", group_name),
                                            Err(e) => log::error!("get_by_name failed: {e:?}"),
                                        }
                                    } else {
                                        log::info!("Saved {} rows to DB (no group association)", ids.len());
                                    }
                                }
                                Err(e) => log::error!("Upsert rows failed: {e:?}"),
                            }
                        });
                    }
                }
                
                if !self.last_scan_rows.is_empty() {
                    if Button::new("Return to Active Scan").right_text("↩").ui(ui).on_hover_text("Restore the last recursive scan results without rescanning").clicked() {
                    
                        self.table.clear();
                        self.table_index.clear();
                        for r in self.last_scan_rows.clone().into_iter() {
                            self.table.push(r.clone());
                            let idx = self.table.len()-1;
                            self.table_index.insert(r.path.clone(), idx);
                        }
                        self.viewer.showing_similarity = false;
                        self.viewer.similar_scores.clear();
                        self.scan_done = true;
                        self.file_scan_progress = 1.0;
                    }
                }
                // Auto-save to DB toggle (persisted)
                if ui.checkbox(&mut self.viewer.ui_settings.auto_save_to_database, "Auto save to database").on_hover_text("When enabled, newly discovered files will be saved to the database automatically (requires an active logical group)").changed() {
                    crate::database::settings::save_settings(&self.viewer.ui_settings);
                }

                ui.separator();
                ui.vertical_centered(|ui| 
                    ui.heading(RichText::new("Bulk Operations").underline().color(ui.style().visuals.error_fg_color))
                );
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
                        // Respect overwrite setting: skip existing descriptions unless overwrite_descriptions is enabled
                        let overwrite = self.viewer.ui_settings.overwrite_descriptions;
                        if !overwrite && (row.caption.is_some() || row.description.is_some()) { continue; }
                        let path_str = row.path.clone();
                        let path_str_clone = path_str.clone();
                        let tx_updates = self.viewer.ai_update_tx.clone();
                        let prompt_clone = prompt.clone();
                        let eng = engine.clone();
                        tokio::spawn(async move {
                            eng.stream_vision_description(std::path::Path::new(&path_str_clone), &prompt_clone, move |interim, final_opt| {
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
                        self.vision_started += 1;
                        self.vision_pending += 1;
                        scheduled += 1;
                    }
                    if scheduled == 0 { log::info!("[AI] Bulk Generate: nothing to schedule"); } else { log::info!("[AI] Bulk Generate scheduled {scheduled} items"); }
                    // reset flags after start to allow next run triggers later
                    self.viewer.bulk_cancel_requested = false;
                    crate::ai::GLOBAL_AI_ENGINE.reset_bulk_cancel();
                }
                
                if Button::new("Generate CLIP Embeddings").right_text("⚡").ui(ui).clicked() {
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
                
                ui.separator();
                ui.checkbox(&mut self.follow_active_vision, "Follow active vision (auto-select)")
                    .on_hover_text("When enabled, the preview auto-selects the image currently being described.");

            });
        });
    }
}
