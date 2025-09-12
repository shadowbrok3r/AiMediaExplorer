use crate::{utilities::windows::{gpu_mem_mb, system_mem_mb, smoothed_cpu01, smoothed_ram01, smoothed_vram01}, ui::{status::{self, GlobalStatusIndicator}, tabs::TABS}};
use eframe::egui::*;
use egui_dock::SurfaceIndex;
use humansize::DECIMAL;

// We assume SmartMediaApp has (or will get) a boolean `ai_initializing` and `ai_ready` flags plus `open_settings_modal`.
// If they don't exist yet, they need to be added to `SmartMediaApp` (app.rs). For now we optimistically reference via super::MainPage's parent.

impl crate::app::SmartMediaApp {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(25.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button(" File ", |ui| {
                    ui.menu_button("Database", |ui| {
                        // When in Database mode, allow viewing the entire DB at once
                        if self.context.file_explorer.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
                            if ui.button("View Entire Database").on_hover_text("Show all thumbnails from all directories").clicked() {
                                self.context.file_explorer.load_all_database_rows();
                                ui.close();
                            }
                            ui.separator();
                        }
                        if ui.button("Choose DB Folder…").clicked() {
                            if let Some(dir) = rfd::FileDialog::new().set_title("Choose database folder").pick_folder() {
                                let path_str = dir.display().to_string();
                                // Persist path and reconnect DB
                                if let Err(e) = crate::database::set_db_path(&path_str) { log::error!("set_db_path failed: {e}"); }
                                // Reconnect asynchronously and refresh state
                                let db_ready_tx = self.context.db_ready_tx.clone();
                                tokio::spawn(async move {
                                    // Close and re-init is not exposed directly; re-call new() which connects and defines schema
                                    crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Initializing, format!("Opening DB at {}", path_str));
                                    let _ = crate::database::new(db_ready_tx.clone()).await;
                                    crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
                                });
                                ui.close();
                            }
                        }
                        if ui.button("Export Backup…").clicked() {
                            if let Some(file) = rfd::FileDialog::new()
                                .set_title("Export SurrealDB backup to .surql")
                                .add_filter("SurrealQL", &["surql"])
                                .set_file_name("backup.surql")
                                .save_file()
                            {
                                // Confirm export
                                let do_it = rfd::MessageDialog::new()
                                    .set_title("Export Database")
                                    .set_description(&format!("Export database to {}?", file.display()))
                                    .set_level(rfd::MessageLevel::Info)
                                    .set_buttons(rfd::MessageButtons::YesNo)
                                    .show();
                                if do_it == rfd::MessageDialogResult::Yes {
                                    let tx = self.context.toast_tx.clone();
                                    tokio::spawn(async move {
                                        match crate::database::export_to(&file) .await {
                                            Ok(_) => { let _ = tx.try_send((egui_toast::ToastKind::Success, format!("Exported DB to {}", file.display()))); },
                                            Err(e) => { let _ = tx.try_send((egui_toast::ToastKind::Error, format!("DB export failed: {e}"))); },
                                        }
                                    });
                                }
                                ui.close();
                            }
                        }
                        if ui.button("Import…").clicked() {
                            if let Some(file) = rfd::FileDialog::new()
                                .set_title("Import SurrealDB .surql file")
                                .add_filter("SurrealQL", &["surql"]).pick_file()
                            {
                                // Confirm import (can overwrite)
                                let do_it = rfd::MessageDialog::new()
                                    .set_title("Import Database")
                                    .set_description(&format!("Import from {}? This may overwrite existing data.", file.display()))
                                    .set_level(rfd::MessageLevel::Warning)
                                    .set_buttons(rfd::MessageButtons::YesNo)
                                    .show();
                                if do_it == rfd::MessageDialogResult::Yes {
                                    let tx = self.context.toast_tx.clone();
                                    tokio::spawn(async move {
                                        match crate::database::import_from(&file).await {
                                            Ok(_) => { let _ = tx.try_send((egui_toast::ToastKind::Success, format!("Imported DB from {}", file.display()))); },
                                            Err(e) => { let _ = tx.try_send((egui_toast::ToastKind::Error, format!("DB import failed: {e}"))); },
                                        }
                                    });
                                }
                                ui.close();
                            }
                        }
                    });
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
                            let path = self.context.file_explorer.current_thumb.path.clone();
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

                ui.menu_button(" View ", |ui| {
                    if ui.button("Egui Settings").clicked() {
                        self.context.open_ui_settings = true;
                    }

                    // allow certain tabs to be toggled
                    for tab in TABS {
                        if ui
                            .selectable_label(self.context.open_tabs.contains(tab), tab)
                            .clicked()
                        {
                            if let Some(index) = self.tree.find_tab(&tab.to_string()) {
                                self.tree.remove_tab(index);
                                self.context.open_tabs.remove(tab);
                            } else {
                                self.tree[SurfaceIndex::main()]
                                    .push_to_focused_leaf(tab.to_string());
                            }

                            ui.close();
                        }
                    }
                });

                ui.add_space(5.);
                ui.separator();
                ui.add_space(5.);

                let current_path_clone = self.context.file_explorer.current_path.clone();
                let parts: Vec<String> = current_path_clone
                    .split(['\\', '/'])
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                let root_has_slash = current_path_clone.starts_with('/');
                let mut accum = if root_has_slash {
                    String::from("/")
                } else {
                    String::new()
                };
                ui.horizontal(|ui| {
                    for (i, part) in parts.iter().enumerate() {
                        if !accum.ends_with(std::path::MAIN_SEPARATOR) && !accum.is_empty() {
                            accum.push(std::path::MAIN_SEPARATOR);
                        }
                        accum.push_str(part);
                        let display = if part.is_empty() {
                            std::path::MAIN_SEPARATOR.to_string()
                        } else {
                            part.clone()
                        };
                        if ui
                            .selectable_label(false, RichText::new(display).underline())
                            .clicked()
                        {
                            self.context.file_explorer.push_history(accum.clone());
                            if self.context.file_explorer.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
                                // In DB mode, treat breadcrumbs as path prefix changes
                                self.context.file_explorer.db_offset = 0;
                                self.context.file_explorer.db_last_batch_len = 0;
                                self.context.file_explorer.load_database_rows();
                            } else {
                                self.context.file_explorer.populate_current_directory();
                            }
                        }
                        if i < parts.len() - 1 {
                            ui.label(RichText::new("›").weak());
                        }
                        // Remove trailing segment for next iteration accumulation clone safety
                    }
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    status::status_bar_inline(ui);
                });
            });
        });

        TopBottomPanel::bottom("FileExplorer Bottom Panel")
        .exact_height(22.)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.context.file_explorer.file_scan_progress > 0.0 {
                    let mut bar = ProgressBar::new(self.context.file_explorer.file_scan_progress)
                        .animate(true)
                        .desired_width(100.)
                        .show_percentage();

                    if self.context.file_explorer.scan_done {
                        self.context.file_explorer.file_scan_progress = 0.;
                        bar = bar.text(RichText::new("Scan Complete").color(Color32::LIGHT_GREEN));
                    }
                    bar.ui(ui);
                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                }

                // AI status/progress moved to global indicators (JOY/CLIP); no local bars here
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
                let vram01 = smoothed_vram01();
                let (v_used, v_total) = gpu_mem_mb().unwrap_or_default();
                ui.label(format!("VRAM: {:.0}/{:.0} MiB", v_used, v_total)); 
                ProgressBar::new(vram01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);

                ui.separator();

                // System metrics (CPU, RAM, VRAM)
                let cpu01 = smoothed_cpu01();
                ui.label(format!("CPU: {:.2}%", cpu01 * 100.0));
                ProgressBar::new(cpu01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);

                ui.separator(); 

                let ram01 = smoothed_ram01();
                if let Some((used_mb, total_mb)) = system_mem_mb() {
                    ui.label(format!("RAM: {:.0}/{:.0} MiB", used_mb, total_mb));
                } else {
                    ui.label("RAM: n/a");
                }
                ProgressBar::new(ram01)
                .desired_width(100.)
                .desired_height(3.)
                .fill(ui.style().visuals.error_fg_color)
                .ui(ui);
                    
                ui.separator(); 

                ui.add_space(10.);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Selection count from current table
                    let selected_cnt = self.context.file_explorer.selection_count();
                    let mut img_cnt = 0usize;
                    let mut vid_cnt = 0usize;
                    let mut dir_cnt = 0usize;
                    let mut total_size = 0u64;
                    for r in self.context.file_explorer.table.iter() {
                        if r.file_type == "<DIR>" {
                            dir_cnt += 1;
                            continue;
                        }
                        if let Some(ext) = std::path::Path::new(&r.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                        {
                            if crate::is_image(ext.as_str()) {
                                img_cnt += 1;
                            }
                            if crate::is_video(ext.as_str()) {
                                vid_cnt += 1;
                            }
                        }
                        total_size += r.size;
                    }
                    ui.label(format!("Selected: {}", selected_cnt));
                    ui.separator();
                    ui.label(format!("Dirs: {dir_cnt}"));
                    ui.separator();
                    ui.label(format!("Images: {img_cnt}"));
                    ui.separator();
                    ui.label(format!("Videos: {vid_cnt}"));
                    ui.separator();
                    ui.label(format!(
                        "Total Size: {}",
                        humansize::format_size(total_size, DECIMAL)
                    ));
                });
            });
        });
    }
}

