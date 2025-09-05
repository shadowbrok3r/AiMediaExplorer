use std::path::PathBuf;

use eframe::egui::*;
use humansize::{format_size, DECIMAL};

use crate::list_drive_infos;

// We assume SmartMediaApp has (or will get) a boolean `ai_initializing` and `ai_ready` flags plus `open_settings_modal`.
// If they don't exist yet, they need to be added to `SmartMediaApp` (app.rs). For now we optimistically reference via super::MainPage's parent.

impl super::MainPage {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(25.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {

                ui.menu_button("File", |ui| {
                    ui.menu_button("AI", |ui| {
                        // Enable AI Search: spawn async global init if not already in progress
                        if ui.button("Enable AI Search").clicked() {
                            // Use a notification window or log; spawn async task for init
                            tokio::spawn(async move {
                                crate::ai::init_global_ai_engine_async().await;
                            });
                        }
                        if ui.button("Enable Auto Indexing").clicked() {
                            // Toggle auto indexing flag inside global engine (atomic set true)
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        if ui.button("Enable Auto CLIP").clicked() {
                            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            tokio::spawn(async move {
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await;
                                log::info!("[CLIP] Auto CLIP enabled via navbar; backfill added {added}");
                            });
                        }
                        if ui.button("Generate CLIP (Missing)").clicked() {
                            tokio::spawn(async move {
                                let count_before = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await;
                                let remaining = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Manual missing generation added {added}; remaining {remaining} (was {count_before})");
                            });
                        }
                        if ui.button("Generate CLIP (Selected)").clicked() {
                            let path = self.file_explorer.current_thumb.path.clone();
                            if !path.is_empty() {
                                tokio::spawn(async move {
                                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await;
                                    log::info!("[CLIP] Selected generation for {path} -> added {added}");
                                });
                            } else {
                                log::warn!("[CLIP] No current preview path set for generation");
                            }
                        }
                        if ui.button("CLIP Missing Count").clicked() {
                            tokio::spawn(async move {
                                let missing = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Missing embeddings: {missing}");
                            });
                        }
                    });
                    if ui.button("Preferences").clicked() {
                        crate::app::OPEN_SETTINGS_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                ui.menu_button("Quick Access", |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.add_space(5.);
                        ui.heading("User Directories");
                        ui.add_space(5.);
                        for access in crate::quick_access().iter() {
                            ui.separator();
                            if Button::new(&access.icon).min_size(vec2(20., 20.)).right_text(&access.label).ui(ui).on_hover_text(&access.label).clicked() {
                                self.file_explorer.set_path(access.path.to_string_lossy());
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
                                self.file_explorer.set_path(path.to_string_lossy());
                            }
                        }
                    });
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.checkbox(&mut self.open_log_window, "View Logs");
                });
            });
            
            Window::new("Logs")
            .open(&mut self.open_log_window)
            .show(ctx, |ui| {
                egui_logger::logger_ui()
                .warn_color(Color32::from_rgb(94, 215, 221)) 
                .error_color(Color32::from_rgb(255, 55, 102)) 
                .log_levels([true, true, true, false, false])
                .enable_category("eframe".to_string(), false)
                .enable_category("eframe::native::glow_integration".to_string(), false)
                .enable_category("egui_glow::shader_version".to_string(), false)
                .enable_category("egui_glow::painter".to_string(), false)
                .show(ui);
            })
        });
    }
}