use crate::{icons::{self, svg}, ui::{dialogs, status::{self, GlobalStatusIndicator}, tabs::TABS}};
use eframe::egui::*;
use egui_dock::SurfaceIndex;
use egui_toast::ToastKind;

impl crate::app::SmartMediaApp {
    pub fn navbar(&mut self, ui: &mut Ui) {
        Panel::top("MainPageTopPanel")
        .exact_size(24.)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button(" File ", |ui| {
                    icons::menu_button(ui, svg::AME_DATABASE, "Database", |ui| {
                        if icons::button(ui, svg::AME_THUMBNAIL_GRID, "View Entire Database").on_hover_text("Show all thumbnails from all directories").clicked() {
                            crate::app::OPEN_TAB_REQUESTS
                                .lock()
                                .unwrap()
                                .push(crate::ui::file_table::FilterRequest::OpenDatabaseAll { title: "Entire Database".to_string(), background: false });
                        }
                        if icons::button(ui, svg::AME_REFINE, "Refine (AI, DB-only)").on_hover_text("Open AI Refinements panel to generate proposals from the database").clicked() {
                            self.focus_or_open_tab("AI Refinements");
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_CATEGORY, "Group by Categories").on_hover_text("Show virtual folders for each category").clicked() {
                            let ex = self.context.active_explorer();
                            ex.viewer.mode = super::file_table::table::ExplorerMode::Database;
                            ex.load_virtual_categories_view();
                            ui.close();
                        }
                        if icons::button(ui, svg::TAG, "Group by Tags").on_hover_text("Show virtual folders for each tag").clicked() {
                            let ex = self.context.active_explorer();
                            ex.viewer.mode = super::file_table::table::ExplorerMode::Database;
                            ex.load_virtual_tags_view();
                            ui.close();
                        }
                        ui.separator();
                        if icons::button(ui, svg::FOLDER_OPEN, "Choose DB Folder…").clicked() {
                            dialogs::pick_folder(ui.ctx(), "Choose database folder", move |dir| {
                                let path_str = dir.display().to_string();
                                match crate::database::set_db_path(&path_str) {
                                    Ok(()) => dialogs::toast(egui_toast::ToastKind::Info, format!("Database folder set to {path_str}. Restart the app to open it.")),
                                    Err(e) => {
                                        log::error!("set_db_path failed: {e}");
                                        dialogs::toast(egui_toast::ToastKind::Error, format!("Could not set the database folder: {e}"));
                                    }
                                }
                            });
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_DATABASE, "Database maintenance…").clicked() {
                            crate::ui::devtools::toggle_db_maintenance(ui.ctx());
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_DB_EXPORT, "Export Backup…").clicked() {
                            let ctx = ui.ctx().clone();
                            let confirm_ctx = ctx.clone();
                            dialogs::save_file(&ctx, "Export SurrealDB backup to .surql", "backup.surql", Some(("SurrealQL", &["surql"])), move |file| {
                                let description = format!("Export database to {}?", file.display());
                                dialogs::confirm(&confirm_ctx, "Export Database", &description, rfd::MessageLevel::Info, move || {
                                    tokio::spawn(async move {
                                        match crate::database::export_to(&file).await {
                                            Ok(_) => dialogs::toast(ToastKind::Success, format!("Exported DB to {}", file.display())),
                                            Err(e) => dialogs::toast(ToastKind::Error, format!("DB export failed: {e}")),
                                        }
                                    });
                                });
                            });
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_DB_IMPORT, "Import…").clicked() {
                            let ctx = ui.ctx().clone();
                            let confirm_ctx = ctx.clone();
                            dialogs::pick_file(&ctx, "Import SurrealDB .surql file", move |file| {
                                let description = format!("Import from {}? This may overwrite existing data.", file.display());
                                dialogs::confirm(&confirm_ctx, "Import Database", &description, rfd::MessageLevel::Warning, move || {
                                    tokio::spawn(async move {
                                        match crate::database::import_from(&file).await {
                                            Ok(_) => dialogs::toast(ToastKind::Success, format!("Imported DB from {}", file.display())),
                                            Err(e) => dialogs::toast(ToastKind::Error, format!("DB import failed: {e}")),
                                        }
                                    });
                                });
                            });
                            ui.close();
                        }
                    });

                    icons::menu_button(ui, svg::AME_MODEL, "AI", |ui| {
                        // Enable AI Search: spawn async global init if not already in progress
                        if icons::button(ui, svg::AME_SEMANTIC_SEARCH, "Enable AI Search").clicked() {
                            status::VISION_STATUS.set_state(status::StatusState::Initializing, "Loading vision model");
                            tokio::spawn(async move {
                                crate::ai::init_global_ai_engine_async().await;
                                status::VISION_STATUS.set_state(status::StatusState::Idle, "Ready");
                            });
                        }
                        if icons::button(ui, svg::AME_AI_DESCRIBE, "Enable Auto Indexing").clicked() {
                            // Toggle auto indexing flag inside global engine (atomic set true)
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::VISION_STATUS.set_state(status::StatusState::Running, "Auto descriptions enabled");
                        }
                        if icons::button(ui, svg::AME_CLIP_EMBED, "Enable Auto CLIP").clicked() {
                            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::CLIP_STATUS.set_state(status::StatusState::Running, "Auto CLIP backfill");
                            tokio::spawn(async move {
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                log::info!("[CLIP] Auto CLIP enabled via navbar; backfill added {added}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, "Idle");
                                Ok::<(), anyhow::Error>(())
                            });
                        }
                        if icons::button(ui, svg::AME_EMBEDDING, "Generate CLIP (Missing)").clicked() {
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
                        if icons::button(ui, svg::CURSOR_CLICK, "Generate CLIP (Selected)").clicked() {
                            let path = self.context.active_explorer().current_thumb.path.clone();
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
                        if icons::button(ui, svg::GAUGE, "CLIP Missing Count").clicked() {
                            tokio::spawn(async move {
                                let missing = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Missing embeddings: {missing}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, format!("Missing {missing}"));
                            });
                        }
                    });
                    if icons::button(ui, svg::GEAR, "Preferences").clicked() {
                        crate::app::OPEN_SETTINGS_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });

                ui.menu_button(" View ", |ui| {
                    if icons::button(ui, svg::PALETTE, "Egui Settings").clicked() {
                        self.context.open_ui_settings = true;
                    }

                    // allow certain tabs to be toggled
                    for tab in TABS {
                        if icons::selectable_button(ui, tab_icon(tab), self.context.open_tabs.contains(tab), tab)
                            .clicked()
                        {
                            if let Some(index) = self.tree.find_tab(&tab.to_string()) {
                                self.tree.remove_tab(index);
                                self.context.open_tabs.remove(tab);
                            } else {
                                self.tree[SurfaceIndex::main()]
                                    .push_to_focused_leaf(tab.to_string());
                                self.context.open_tabs.insert(tab.to_string());
                            }

                            ui.close();
                        }
                    }

                    ui.separator();
                    // Toggle for Assistant Viewport window
                    let mut was_open = self.context.assistant_window_open;
                    if icons::checkbox(ui, &mut was_open, svg::AME_ASSISTANT, "AI Assistant Window").clicked() {
                        self.context.assistant_window_open = was_open;
                    }
                    ui.separator();
                    if icons::button(ui, svg::SLIDERS, "Theme panel").clicked() {
                        crate::theme::debug_panel::toggle(ui.ctx());
                        ui.close();
                    }
                    if icons::button(ui, svg::AME_LOGO, "Icon gallery").clicked() {
                        icons::gallery::toggle(ui.ctx());
                        ui.close();
                    }
                    icons::menu_button(ui, svg::WRENCH, "Devtools", |ui| {
                        if icons::button(ui, svg::DATABASE, "Settings store").clicked() {
                            crate::ui::devtools::toggle_settings_store(ui.ctx());
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_DATABASE, "Database maintenance").clicked() {
                            crate::ui::devtools::toggle_db_maintenance(ui.ctx());
                            ui.close();
                        }
                        if icons::button(ui, svg::AME_THUMBNAIL_GRID, "Thumbnail storage").clicked() {
                            crate::ui::devtools::toggle_thumb_storage(ui.ctx());
                            ui.close();
                        }
                        if icons::button(ui, svg::IMAGE, "Thumbnail cache").clicked() {
                            crate::ui::devtools::thumb_cache::toggle(ui.ctx());
                            ui.close();
                        }
                    });
                });

                ui.add_space(5.);
                ui.separator();
                ui.add_space(5.);

                let current_path_clone = self.context.active_explorer().current_path.clone();

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
                            let ex = self.context.active_explorer();
                            ex.push_history(accum.clone());
                            if ex.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
                                // In DB mode, treat breadcrumbs as path prefix changes
                                ex.db_offset = 0;
                                ex.db_last_batch_len = 0;
                                ex.load_database_rows();
                            } else {
                                ex.populate_current_directory();
                            }
                        }
                        if i < parts.len() - 1 {
                            let size = icons::IconSize::Sm.px(ui);
                            let weak = ui.visuals().weak_text_color();
                            svg::CARET_RIGHT.show_tinted(ui, size, weak);
                        }
                    }
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    status::status_bar_inline(ui);
                });
            });
        });

        icons::gallery::show_window(ui.ctx());

        let owner = self
            .context
            .active_tab_title
            .clone()
            .unwrap_or_else(|| crate::app::MAIN_EXPLORER_TAB.to_string());
        crate::ui::statusbar::show(ui, self.context.active_explorer(), &owner);
    }
}

/// Icon of a dockable tab by title.
fn tab_icon(tab: &str) -> icons::Icon {
    match tab {
        "File Explorer" => svg::FOLDER_OPEN,
        "Logs" => svg::TERMINAL_WINDOW,
        "AI Refinements" => svg::AME_REFINE,
        _ => svg::TABS,
    }
}
