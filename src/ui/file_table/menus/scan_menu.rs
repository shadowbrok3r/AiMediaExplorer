use eframe::egui::*;
use egui::containers::menu::{MenuButton, MenuConfig};

use crate::icons::{self, svg};

impl crate::ui::file_table::FileExplorer {
    /// Starts a recursive scan of the current path that replaces this tab's rows.
    fn start_kind_scan(&mut self, images: bool, videos: bool, archives: bool) {
        self.stop_work();
        self.recursive_scan = true;
        self.scan_done = false;
        self.table.clear();
        self.last_scan_rows.clear();
        self.last_scan_paths.clear();
        self.last_scan_root = Some(self.current_path.clone());
        self.file_scan_progress = 0.0;
        let mut filters = crate::Filters::default();
        filters.root = std::path::PathBuf::from(self.current_path.clone());
        filters.include_images = images;
        filters.include_videos = videos;
        filters.include_archives = archives;
        filters.excluded_terms = self.excluded_terms.clone();
        self.launch_recorded_scan(filters, true);
    }

    pub fn scan_menu(&mut self, ui: &mut Ui) {
        MenuButton::new("Scan")
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .ui(ui, |ui| {
            ui.vertical_centered_justified(|ui| {
                ui.set_width(250.);
                let in_folder = matches!(self.nav.scope, crate::command::Scope::Folder(_));
                ui.vertical_centered(|ui|
                    ui.heading(RichText::new("Recursive Scanning").underline().color(crate::theme::primary(ui)))
                );
                if ui.add_enabled_ui(in_folder, |ui| icons::button(ui, svg::AME_SCAN, "Recursive Scan")).inner.on_hover_text("Scan this folder and its subfolders in a new tab").clicked() {
                    crate::command::send(crate::command::AppCommand::OpenScope {
                        scope: crate::command::Scope::Folder(self.current_path.clone()),
                        target: crate::command::Target::NewTab,
                        title: None,
                        recursive: true,
                    });
                }
                if ui.add_enabled_ui(in_folder, |ui| icons::button(ui, svg::IMAGE, "Scan Images")).inner.on_hover_text("Recursive scan including only images").clicked() {
                    self.start_kind_scan(true, false, false);
                }
                if ui.add_enabled_ui(in_folder, |ui| icons::button(ui, svg::VIDEO_CAMERA, "Scan Videos")).inner.on_hover_text("Recursive scan including only videos").clicked() {
                    self.start_kind_scan(false, true, false);
                }
                if ui.add_enabled_ui(in_folder, |ui| icons::button(ui, svg::ARCHIVE, "Scan Archives (.zip/.7z/.rar/.tar)")).inner.on_hover_text("Recursive scan including only archive containers").clicked() {
                    self.start_kind_scan(false, false, true);
                }

                if ui.add_enabled_ui(in_folder, |ui| icons::button(ui, svg::ARROWS_CLOCKWISE, "Re-scan Current Folder")).inner.on_hover_text("Shallow scan with current filters (applies 'Skip likely icons')").clicked() {
                    self.stop_work();
                    self.recursive_scan = false;
                    self.scan_done = false;
                    self.table.clear();
                    self.last_scan_rows.clear();
                    self.last_scan_paths.clear();
                    self.last_scan_root = Some(self.current_path.clone());
                    self.file_scan_progress = 0.0;
                    let mut filters = crate::Filters::default();
                    filters.root = std::path::PathBuf::from(self.current_path.clone());
                    filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                    filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                    filters.include_images = self.viewer.types_show_images;
                    filters.include_videos = self.viewer.types_show_videos;
                    filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                    filters.excluded_terms = self.excluded_terms.clone();
                    self.launch_recorded_scan(filters, false);
                }

                if crate::ui::danger_icon_button(ui, svg::STOP, "Cancel Scan").on_hover_text("Cancel active recursive scan").clicked() {
                    if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
                }
                ui.separator();
                ui.vertical_centered(|ui|
                    ui.heading(RichText::new("Database Save").underline().color(crate::theme::primary(ui)))
                );
                if icons::button(ui, svg::FLOPPY_DISK, "Save Current View to DB").on_hover_text("Save all currently visible files to the library and add them to the active collection").clicked() {
                    self.sync_view();
                    let rows: Vec<crate::database::Thumbnail> = self.view.rows(&self.table).filter(|r| r.file_type != "<DIR>").cloned().collect();
                    self.save_rows_to_library(rows);
                }

                if !self.last_scan_rows.is_empty() {
                    if icons::button(ui, svg::ARROW_COUNTER_CLOCKWISE, "Return to Active Scan").on_hover_text("Restore the last recursive scan results without rescanning").clicked() {
                        self.table.replace_all(self.last_scan_rows.iter().cloned());
                        self.viewer.showing_similarity = false;
                        self.viewer.similar_scores.clear();
                        self.scan_done = true;
                        self.file_scan_progress = 1.0;
                    }
                }
                let mut auto_save = self.viewer.ui_settings.auto_save_to_database;
                if ui.checkbox(&mut auto_save, "Auto save to database").on_hover_text("When enabled, newly discovered files will be saved to the database automatically (requires an active logical group)").changed() {
                    crate::settings::update("scan:auto-save", |s| s.auto_save_to_database = auto_save);
                    self.viewer.sync_settings();
                }

                ui.separator();
                ui.vertical_centered(|ui|
                    ui.heading(RichText::new("Bulk Operations").underline().color(crate::theme::primary(ui)))
                );
                if icons::button(ui, svg::AME_AI_DESCRIBE, "Bulk Generate Descriptions").on_hover_text("Generate AI descriptions for images missing caption/description").clicked() {
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

                if icons::button(ui, svg::AME_CLIP_EMBED, "Generate CLIP Embeddings").clicked() {
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

                if crate::ui::danger_icon_button(ui, svg::STOP, "Cancel Bulk Descriptions").on_hover_text("Stop scheduling/streaming new vision descriptions").clicked() {
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
