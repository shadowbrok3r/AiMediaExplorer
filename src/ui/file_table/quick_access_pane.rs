use crate::{next_scan_id, spawn_scan, Filters};
use humansize::{format_size, DECIMAL};
use std::path::{PathBuf, Path};
use crate::list_drive_infos;
use eframe::egui::*;

impl super::FileExplorer {
    /// Rich "Quick Access" / Options panel consolidating actions previously in the gear menu.
    pub fn quick_access_pane(&mut self, ui: &mut Ui) {
        SidePanel::left("MainPageLeftPanel")
            .max_width(420.)
            .min_width(240.)
            .show_animated_inside(ui, self.open_quick_access, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Options");
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.button("X").on_hover_text("Close panel").clicked() { self.open_quick_access = false; }
                        });
                    });
                    ui.separator();

                    ScrollArea::vertical().auto_shrink([false;2]).show(ui, |ui| {
                        // Scan Section
                        CollapsingHeader::new(RichText::new("Scan & Generation").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label(RichText::new("Size Filters (MB)").italics());
                                ui.horizontal(|ui| {
                                    ui.label("Min:");
                                    // Use parent module statics
                                    let mut min_txt = unsafe { super::MIN_SIZE_MB.map(|v| (v / 1_000_000).to_string()).unwrap_or_default() };
                                    if ui.add(TextEdit::singleline(&mut min_txt).desired_width(60.)).lost_focus() { /* no-op */ }
                                    ui.label("Max:");
                                    let mut max_txt = unsafe { super::MAX_SIZE_MB.map(|v| (v / 1_000_000).to_string()).unwrap_or_default() };
                                    if ui.add(TextEdit::singleline(&mut max_txt).desired_width(60.)).lost_focus() { /* no-op */ }
                                    if ui.button("Apply").clicked() {
                                        unsafe {
                                            super::MIN_SIZE_MB = min_txt.trim().parse::<u64>().ok().map(|m| m * 1_000_000);
                                            super::MAX_SIZE_MB = max_txt.trim().parse::<u64>().ok().map(|m| m * 1_000_000);
                                        }
                                    }
                                    if ui.button("Clear").on_hover_text("Clear size constraints").clicked() {
                                        unsafe { super::MIN_SIZE_MB = None; super::MAX_SIZE_MB = None; }
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
                                        let scan_id = next_scan_id();
                                        let tx = self.scan_tx.clone();
                                        let recurse = self.recursive_scan.clone();
                                        let mut filters = Filters::default();
                                        filters.root = PathBuf::from(self.current_path.clone());
                                        unsafe {
                                            filters.min_size_bytes = super::MIN_SIZE_MB;
                                            filters.max_size_bytes = super::MAX_SIZE_MB;
                                        }
                                        filters.excluded_terms = self.excluded_terms.clone();
                                        // Remember current scan id so cancel can target it
                                        self.current_scan_id = Some(scan_id);
                                        tokio::spawn(async move {
                                            spawn_scan(filters, tx, recurse, scan_id).await;
                                        });
                                    }
                                    if ui.button("🖩 Bulk Generate Descriptions").on_hover_text("Generate AI descriptions for images missing caption/description").clicked() {
                                        let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                        let prompt = self.viewer.ui_settings.ai_prompt_template.clone();
                                        let mut scheduled = 0usize;
                                        for row in self.table.iter() {
                                            if self.viewer.bulk_cancel_requested { break; }
                                            if row.file_type == "<DIR>" { continue; }
                                            if let Some(ext) = Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) { if !crate::is_image(ext.as_str()) { continue; } } else { continue; }
                                            if row.caption.is_some() || row.description.is_some() { continue; }
                                            let path_str = row.path.clone();
                                            let path_str_clone = path_str.clone();
                                            let tx_updates = self.viewer.ai_update_tx.clone();
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
                                        let paths: Vec<String> = self
                                            .table
                                            .iter()
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
                                });
                                ui.add_space(6.0);
                                ui.label(format!("Scan Progress: {:.1}%", self.file_scan_progress * 100.0));
                                ui.label(format!("Vision Gen: started {} | pending {} | completed {}", self.vision_started, self.vision_pending, self.vision_completed));
                            
                            });


                        ui.collapsing("Quick Access", |ui| {
                            // Recents Section
                            CollapsingHeader::new(RichText::new("Recent Directories").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                // Load settings snapshot (cached) and render recent paths
                                let settings = crate::database::settings::load_settings();
                                if settings.recent_paths.is_empty() {
                                    ui.label(RichText::new("No recent directories yet").weak());
                                } else {
                                    for p in settings.recent_paths.iter() {
                                        ui.horizontal(|ui| {
                                            if Button::image_and_text(eframe::egui::include_image!("../../../assets/Icons/folder.png"), p).ui(ui).clicked() {
                                                self.push_history(p.clone());
                                                if self.viewer.mode == super::viewer::ExplorerMode::Database {
                                                    self.db_offset = 0;
                                                    self.db_last_batch_len = 0;
                                                    self.load_database_rows();
                                                } else {
                                                    self.populate_current_directory();
                                                }
                                            }
                                        });
                                    }
                                }
                            });
                            ui.vertical_centered_justified(|ui| {
                                ui.add_space(5.);
                                ui.heading("User Directories");
                                ui.add_space(5.);
                                for access in crate::quick_access().iter() {
                                    if Button::new(&access.label).min_size(vec2(20., 20.)).right_text(&access.icon).ui(ui).on_hover_text(&access.label).clicked() {
                                        self.set_path(access.path.to_string_lossy());
                                    }
                                }
                                ui.add_space(5.);
                                ui.heading("Drives");
                                ui.add_space(5.);
                                for drive in list_drive_infos() {
                                    ui.separator();
                                    let root = drive.root.clone();
                                    let display = format!("{} - {}", drive.drive_type, drive.root);
                                    let path = PathBuf::from(root.clone());
                                    let free = format_size(drive.free, DECIMAL);
                                    let total = format_size(drive.total, DECIMAL);

                                    let response = Button::new(display)
                                    .right_text(&drive.label)
                                    .ui(ui).on_hover_text(format!("{free} free of {total}"));


                                    if response.clicked() {
                                        self.set_path(path.to_string_lossy());
                                    }
                                }
                            });
                        });
                    });
                });
            });
    }
}