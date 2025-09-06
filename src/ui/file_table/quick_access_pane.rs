use eframe::egui::*;
use std::path::{PathBuf, Path};
use crate::{next_scan_id, spawn_scan, Filters};

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
                            if ui.button("✕").on_hover_text("Close panel").clicked() { self.open_quick_access = false; }
                        });
                    });
                    ui.separator();

                    ScrollArea::vertical().auto_shrink([false;2]).show(ui, |ui| {
                        // Scan Section
                        CollapsingHeader::new(RichText::new("Scan & Generation").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Directory & database scanning plus AI generation utilities.");
                                ui.separator();
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
                                        tokio::spawn(async move {
                                            let count = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                            log::info!("[CLIP] Manual generation completed for {count} images");
                                            Ok::<(), anyhow::Error>(())
                                        });
                                    }
                                    if ui.button("🗙 Cancel Scan").on_hover_text("Cancel active recursive scan").clicked() {
                                        if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
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

                        // View Section
                        CollapsingHeader::new(RichText::new("View & Layout").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.open_preview_pane, "Show Preview Pane");
                                ui.checkbox(&mut self.open_quick_access, "Show Quick Access (this panel)");
                                if ui.button("Group by Category").clicked() {
                                    // TODO implement grouping pipeline
                                }
                                ui.add_space(4.0);
                                ui.label("Mode:");
                                let prev = self.viewer.mode;
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(matches!(self.viewer.mode, super::viewer::ExplorerMode::FileSystem), "File System").clicked() && !matches!(prev, super::viewer::ExplorerMode::FileSystem) {
                                        self.populate_current_directory();
                                    }
                                    if ui.selectable_label(matches!(self.viewer.mode, super::viewer::ExplorerMode::Database), "Database").clicked() && !matches!(prev, super::viewer::ExplorerMode::Database) {
                                        self.load_database_rows();
                                    }
                                });
                            });

                        // Filters Section
                        CollapsingHeader::new(RichText::new("Filters").strong())
                            .default_open(true)
                            .show(ui, |ui| {
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

                        // Database Section
                        CollapsingHeader::new(RichText::new("Database").strong())
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.button("Reload Page").clicked() { self.load_database_rows(); }
                                    if ui.button("Clear Table").clicked() { self.table.clear(); }
                                });
                                ui.horizontal(|ui| {
                                    if ui.button("Switch -> DB Mode").clicked() { if !matches!(self.viewer.mode, super::viewer::ExplorerMode::Database) { self.viewer.mode = super::viewer::ExplorerMode::Database; self.load_database_rows(); } }
                                    if ui.button("Switch -> FS Mode").clicked() { if !matches!(self.viewer.mode, super::viewer::ExplorerMode::FileSystem) { self.viewer.mode = super::viewer::ExplorerMode::FileSystem; self.populate_current_directory(); } }
                                });
                                ui.add_space(4.0);
                                if matches!(self.viewer.mode, super::viewer::ExplorerMode::Database) {
                                    ui.label(format!("Loaded Rows: {} (offset {})", self.table.len(), self.db_offset));
                                    if self.db_loading { ui.colored_label(Color32::YELLOW, "Loading..."); }
                                }
                            });
                    });
                });
            });
    }
}