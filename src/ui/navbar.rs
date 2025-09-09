use crate::ui::{status::{self, GlobalStatusIndicator}, tabs::TABS};
use eframe::egui::*;
use egui_dock::SurfaceIndex;

// We assume SmartMediaApp has (or will get) a boolean `ai_initializing` and `ai_ready` flags plus `open_settings_modal`.
// If they don't exist yet, they need to be added to `SmartMediaApp` (app.rs). For now we optimistically reference via super::MainPage's parent.

impl crate::app::SmartMediaApp {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(25.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button(" File ", |ui| {
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
                            if self.context.file_explorer.viewer.mode == crate::ui::file_table::viewer::ExplorerMode::Database {
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
    }
}

