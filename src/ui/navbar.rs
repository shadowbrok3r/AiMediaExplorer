use crate::{icons::{self, svg}, theme::{chrome, controls::MenuButton}, ui::{dialogs, status::{self, GlobalStatusIndicator}, tabs::TabKind}};
use eframe::egui::*;
use egui_dock::SurfaceIndex;

impl crate::app::SmartMediaApp {
    /// File menu button for the title strip.
    pub fn file_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("File").show(ui, |ui| self.file_menu_items(ui));
    }

    /// View menu button for the title strip.
    pub fn view_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("View").show(ui, |ui| self.view_menu_items(ui));
    }

    fn file_menu_items(&mut self, ui: &mut Ui) {
        icons::menu_button(ui, svg::AME_DATABASE, "Database", |ui| {
            if icons::button(ui, svg::AME_THUMBNAIL_GRID, "View Entire Database").on_hover_text("Show all thumbnails from all directories").clicked() {
                crate::command::send(crate::command::AppCommand::open(crate::command::Scope::Library));
            }
            if icons::button(ui, svg::AME_REFINE, "Refine (AI, DB-only)").on_hover_text("Open AI Refinements panel to generate proposals from the database").clicked() {
                self.focus_or_open_tab(TabKind::Refine);
                ui.close();
            }
            if icons::button(ui, svg::AME_CATEGORY, "Group by Categories").on_hover_text("Show virtual folders for each category").clicked() {
                let ex = self.context.active_explorer();
                ex.load_virtual_categories_view();
                ui.close();
            }
            if icons::button(ui, svg::TAG, "Group by Tags").on_hover_text("Show virtual folders for each tag").clicked() {
                let ex = self.context.active_explorer();
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
                            let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Database, format!("Export database to {}", file.display())).user_initiated();
                            let _ = crate::tasks::track(spec, |_| crate::database::export_to(&file)).await;
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
                            let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Database, format!("Import database from {}", file.display())).user_initiated();
                            let _ = crate::tasks::track(spec, |_| crate::database::import_from(&file)).await;
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
        if icons::button(ui, svg::PALETTE, "Appearance…").on_hover_text("Theme presets, colors, density and text size").clicked() {
            self.context.appearance.open();
            ui.close();
        }
        ui.separator();
        if icons::button(ui, svg::POWER, "Exit").clicked() {
            ui.send_viewport_cmd(ViewportCommand::Close);
        }
    }

    fn view_menu_items(&mut self, ui: &mut Ui) {
        if icons::button(ui, svg::PALETTE, "Appearance…").on_hover_text("Theme presets, colors, density and text size").clicked() {
            self.context.appearance.open();
            ui.close();
        }
        if icons::button(ui, svg::SLIDERS, "Egui Settings").clicked() {
            self.context.open_ui_settings = true;
        }

        // allow certain tabs to be toggled
        for tab in TabKind::TOGGLED {
            if icons::selectable_button(ui, tab_icon(&tab), self.context.open_tabs.contains(&tab), tab.menu_label())
                .clicked()
            {
                if let Some(index) = self.tree.find_tab(&tab) {
                    self.tree.remove_tab(index);
                    self.context.open_tabs.remove(&tab);
                } else {
                    self.tree[SurfaceIndex::main()].push_to_focused_leaf(tab);
                    self.context.open_tabs.insert(tab);
                }

                ui.close();
            }
        }

        ui.separator();
        let mut inspector_open = self.context.inspector.is_open();
        if icons::checkbox(ui, &mut inspector_open, svg::SIDEBAR_SIMPLE, "Inspector").on_hover_text("Preview and details of the selection (Ctrl+I)").changed() {
            self.context.inspector.set_open(inspector_open);
        }
        // Toggle for Assistant Viewport window
        let mut was_open = self.context.assistant_window_open;
        if icons::checkbox(ui, &mut was_open, svg::AME_ASSISTANT, "AI Assistant Window").clicked() {
            self.context.assistant_window_open = was_open;
        }
        ui.separator();
        if icons::button(ui, svg::AME_LOGO, "Icon gallery").clicked() {
            icons::gallery::toggle(ui.ctx());
            ui.close();
        }
        icons::menu_button(ui, svg::WRENCH, "Devtools", |ui| {
            if icons::button(ui, svg::SLIDERS, "Theme panel").clicked() {
                crate::theme::debug_panel::toggle(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::DATABASE, "Settings store").clicked() {
                crate::ui::devtools::toggle_settings_store(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_DATABASE, "Database maintenance").clicked() {
                crate::ui::devtools::toggle_db_maintenance(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::FILE_ARROW_DOWN, "Legacy import").clicked() {
                crate::ui::devtools::toggle_legacy_import(ui.ctx());
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
            if icons::button(ui, svg::LIST_CHECKS, "Tasks").clicked() {
                crate::ui::devtools::toggle_tasks(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::TERMINAL_WINDOW, "Command log").clicked() {
                crate::ui::devtools::toggle_command_log(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::GAUGE, "Perf").clicked() {
                crate::ui::devtools::toggle_perf(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::STACK, "Row store").clicked() {
                crate::ui::devtools::toggle_row_store(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_LOGICAL_GROUP, "Collections").clicked() {
                crate::ui::devtools::toggle_collections(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::VIDEO_CAMERA, "Video").clicked() {
                crate::ui::devtools::toggle_video(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_FOLDER_TREE, "Scope").clicked() {
                crate::ui::devtools::toggle_scope(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::FUNNEL, "Filters").clicked() {
                crate::ui::devtools::toggle_filters(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::SQUARES_FOUR, "Grid").clicked() {
                crate::ui::devtools::toggle_grid(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::SIDEBAR_SIMPLE, "Inspector").clicked() {
                crate::ui::devtools::toggle_inspector(ui.ctx());
                ui.close();
            }
        });

        ui.separator();
        let _ = chrome::toggle_row(ui, &mut self.chrome, "Custom title bar");
    }

    /// Icon gallery window and the bottom status bar of the focused explorer.
    pub fn status_bar(&mut self, ui: &mut Ui) {
        icons::gallery::show_window(ui.ctx());

        let owner = self.context.active_owner();
        crate::ui::taskbar::show(ui, self.context.active_explorer(), &owner);
    }
}

/// Icon of a dockable tab.
fn tab_icon(tab: &TabKind) -> icons::Icon {
    match tab {
        TabKind::Browser { .. } => svg::FOLDER_OPEN,
        TabKind::Logs => svg::TERMINAL_WINDOW,
        TabKind::Refine => svg::AME_REFINE,
        _ => svg::TABS,
    }
}
