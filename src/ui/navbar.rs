use crate::{ui::status::{self, GlobalStatusIndicator}, utilities::windows::{gpu_mem_mb, system_mem_mb, smoothed_cpu01, smoothed_ram01, smoothed_vram01}};
use humansize::{format_size, DECIMAL};
use std::path::PathBuf;
use eframe::egui::*;
use crate::list_drive_infos;

// We assume SmartMediaApp has (or will get) a boolean `ai_initializing` and `ai_ready` flags plus `open_settings_modal`.
// If they don't exist yet, they need to be added to `SmartMediaApp` (app.rs). For now we optimistically reference via super::MainPage's parent.

impl crate::app::SmartMediaApp {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(25.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button("File", |ui| {
                    ui.menu_button("AI", |ui| {
                        // Enable AI Search: spawn async global init if not already in progress
                        if ui.button("Enable AI Search").clicked() {
                            status::JOY_STATUS.set_state(status::StatusState::Initializing, "Loading vision model");
                            tokio::spawn(async move {
                                crate::ai::init_global_ai_engine_async().await;
                                status::JOY_STATUS.set_state(status::StatusState::Idle, "Ready");
                            });
                        }
                        if ui.button("Enable Auto Indexing").clicked() {
                            // Toggle auto indexing flag inside global engine (atomic set true)
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::JOY_STATUS.set_state(status::StatusState::Running, "Auto descriptions enabled");
                        }
                        if ui.button("Enable Auto CLIP").clicked() {
                            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::CLIP_STATUS.set_state(status::StatusState::Running, "Auto CLIP backfill");
                            tokio::spawn(async move {
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                log::info!("[CLIP] Auto CLIP enabled via navbar; backfill added {added}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, "Idle");
                                Ok::<(), anyhow::Error>(())
                            });
                        }
                        if ui.button("Generate CLIP (Missing)").clicked() {
                            status::CLIP_STATUS.set_state(status::StatusState::Running, "Generating missing");
                            tokio::spawn(async move {
                                let count_before = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                let remaining = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Manual missing generation added {added}; remaining {remaining} (was {count_before})");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, format!("Remaining {remaining}"));
                                Ok::<(), anyhow::Error>(())
                            });
                        }
                        if ui.button("Generate CLIP (Selected)").clicked() {
                            let path = self.file_explorer.current_thumb.path.clone();
                            if !path.is_empty() {
                                status::CLIP_STATUS.set_state(status::StatusState::Running, "Selected path");
                                tokio::spawn(async move {
                                    match crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await {
                                        Ok(added) => log::info!("[CLIP] Manual per-item generation: added {added} for {path}"),
                                        Err(e) => log::error!("engine.clip_generate_for_paths: {e:?}")
                                    }
                                    status::CLIP_STATUS.set_state(status::StatusState::Idle, "Idle");
                                    Ok::<(), anyhow::Error>(())
                                });
                            } else {
                                log::warn!("[CLIP] No current preview path set for generation");
                            }
                        }
                        if ui.button("CLIP Missing Count").clicked() {
                            tokio::spawn(async move {
                                let missing = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Missing embeddings: {missing}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, format!("Missing {missing}"));
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
                    ctx.request_repaint_after(std::time::Duration::from_millis(300));
                    ui.checkbox(&mut self.open_log_window, "View Logs");
                    ui.separator();
                    // System metrics (CPU, RAM, VRAM)
                    let cpu01 = smoothed_cpu01();
                    ProgressBar::new(cpu01)
                    .desired_width(100.)
                    .desired_height(3.)
                    .fill(ui.style().visuals.error_fg_color)
                    .ui(ui);

                    ui.label(format!("CPU: {:.2}%", cpu01 * 100.0));
                    ui.separator(); 

                    let ram01 = smoothed_ram01();
                    ProgressBar::new(ram01)
                    .desired_width(100.)
                    .desired_height(3.)
                    .fill(ui.style().visuals.error_fg_color)
                    .ui(ui);

                    if let Some((used_mb, total_mb)) = system_mem_mb() {
                        ui.label(format!("RAM: {:.0}/{:.0} MiB", used_mb, total_mb));
                    } else {
                        ui.label("RAM: n/a");
                    }
                    ui.separator(); 

                    let vram01 = smoothed_vram01();
                    ProgressBar::new(vram01)
                    .desired_width(100.)
                    .desired_height(3.)
                    .fill(ui.style().visuals.error_fg_color)
                    .ui(ui);

                    let (v_used, v_total) = gpu_mem_mb().unwrap_or_default();
                    ui.label(format!("VRAM: {:.0}/{:.0} MiB", v_used, v_total)); 
                    ui.separator(); 

                    status::status_bar_inline(ui);
                });
            });
        });
    }
}

