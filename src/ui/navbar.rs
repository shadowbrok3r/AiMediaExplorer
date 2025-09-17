use crate::{utilities::windows::{gpu_mem_mb, system_mem_mb, smoothed_cpu01, smoothed_ram01, smoothed_vram01}, ui::{status::{self, GlobalStatusIndicator}, tabs::TABS}};
use eframe::egui::*;
use egui_dock::SurfaceIndex;
use humansize::DECIMAL;

impl crate::app::SmartMediaApp {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(24.)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button(" File ", |ui| {
                    ui.menu_button("Database", |ui| {
                        if ui.button("View Entire Database").on_hover_text("Show all thumbnails from all directories").clicked() {
                            crate::app::OPEN_TAB_REQUESTS
                                .lock()
                                .unwrap()
                                .push(crate::ui::file_table::FilterRequest::OpenDatabaseAll { title: "Entire Database".to_string(), background: false });
                        }
                        if ui.button("Refine (AI, DB-only)").on_hover_text("Open AI Refinements panel to generate proposals from the database").clicked() {
                            // Ensure tab is open/focused
                            let label = "AI Refinements".to_string();
                            if let Some((surface, node, _tab)) = self.tree.find_tab(&label) {
                                self.tree.set_focused_node_and_surface((surface, node));
                            } else {
                                self.tree[SurfaceIndex::main()].push_to_focused_leaf(label.clone());
                                self.context.open_tabs.insert(label);
                            }
                            ui.close();
                        }
                        if ui.button("Group by Categories").on_hover_text("Show virtual folders for each category").clicked() {
                            self.context.file_explorer.viewer.mode = super::file_table::table::ExplorerMode::Database;
                            self.context.file_explorer.load_virtual_categories_view();
                            ui.close();
                        }
                        if ui.button("Group by Tags").on_hover_text("Show virtual folders for each tag").clicked() {
                            self.context.file_explorer.viewer.mode = super::file_table::table::ExplorerMode::Database;
                            self.context.file_explorer.load_virtual_tags_view();
                            ui.close();
                        }
                        ui.separator();
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
                            status::VISION_STATUS.set_state(status::StatusState::Initializing, "Loading vision model");
                            tokio::spawn(async move {
                                crate::ai::init_global_ai_engine_async().await;
                                status::VISION_STATUS.set_state(status::StatusState::Idle, "Ready");
                            });
                        }
                        if ui.button("Enable Auto Indexing").clicked() {
                            // Toggle auto indexing flag inside global engine (atomic set true)
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::VISION_STATUS.set_state(status::StatusState::Running, "Auto descriptions enabled");
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
        .exact_height(24.)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Use active explorer (current tab) for progress and stats
                let ex = self.context.active_explorer();

                if ex.file_scan_progress > 0.0 {
                    let mut bar = ProgressBar::new(ex.file_scan_progress)
                        .animate(true)
                        .desired_width(100.)
                        .show_percentage();

                    if ex.scan_done {
                        ex.file_scan_progress = 0.;
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

                // Thumbnail generation progress (visible, non-dir, image/video entries)
                let mut thumbs_total = 0usize;
                let mut thumbs_done = 0usize;
                {
                    let viewer = &ex.viewer;
                    for r in ex.table.iter() {
                        if !viewer.row_passes_filter(&r) { continue; }
                        if r.file_type == "<DIR>" { continue; }
                        if let Some(ext) = std::path::Path::new(&r.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                        {
                            if crate::is_image(ext.as_str()) || crate::is_video(ext.as_str()) {
                                thumbs_total += 1;
                                if viewer.thumb_cache.contains_key(&r.path) { thumbs_done += 1; }
                            }
                        }
                    }
                }
                let prog = if thumbs_total == 0 { 0.0 } else { thumbs_done as f32 / thumbs_total as f32 };
                ui.label(format!("Thumbs: {thumbs_done}/{thumbs_total}"));
                ProgressBar::new(prog)
                    .desired_width(100.)
                    .desired_height(3.)
                    .fill(ui.style().visuals.selection.bg_fill)
                    .ui(ui);
                ui.separator();

                ui.add_space(10.);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Selected count (from table selection)
                    let selected_cnt = ex.selection_count();

                    // Visible vs filtered counts using current viewer row filter
                    let mut visible_cnt = 0usize;
                    let mut all_img_cnt = 0usize;
                    let mut all_vid_cnt = 0usize;
                    let mut all_dir_cnt = 0usize;
                    let mut all_total_size = 0u64; // sum across visible non-dirs

                    // Selected-only tallies
                    let mut sel_img_cnt = 0usize;
                    let mut sel_vid_cnt = 0usize;
                    let mut sel_dir_cnt = 0usize; // will generally be 0 (dirs not selected)
                    let mut sel_total_size = 0u64;

                    {
                        let viewer = &ex.viewer;
                        for r in ex.table.iter() {
                            let is_visible = viewer.row_passes_filter(&r);
                            if is_visible {
                                visible_cnt += 1;
                                if r.file_type == "<DIR>" {
                                    all_dir_cnt += 1;
                                } else {
                                    all_total_size = all_total_size.saturating_add(r.size);
                                    if let Some(ext) = std::path::Path::new(&r.path)
                                        .extension()
                                        .and_then(|e| e.to_str())
                                        .map(|s| s.to_ascii_lowercase())
                                    {
                                        if crate::is_image(ext.as_str()) { all_img_cnt += 1; }
                                        if crate::is_video(ext.as_str()) { all_vid_cnt += 1; }
                                    }
                                }
                            }

                            if viewer.selected.contains(&r.path) {
                                if r.file_type == "<DIR>" {
                                    sel_dir_cnt += 1; // typically 0 due to selection rules
                                } else {
                                    sel_total_size = sel_total_size.saturating_add(r.size);
                                    if let Some(ext) = std::path::Path::new(&r.path)
                                        .extension()
                                        .and_then(|e| e.to_str())
                                        .map(|s| s.to_ascii_lowercase())
                                    {
                                        if crate::is_image(ext.as_str()) { sel_img_cnt += 1; }
                                        if crate::is_video(ext.as_str()) { sel_vid_cnt += 1; }
                                    }
                                }
                            }
                        }
                    }
                    // Global filtered vs total (for recursive scans) falls back to page if not recursive
                    let (global_filtered, global_total) = if ex.is_recursive_scan() {
                        (ex.recursive_total_filtered(), ex.recursive_total_unfiltered())
                    } else { (visible_cnt, ex.table.len()) };
                    let filtered_out = global_total.saturating_sub(global_filtered);

                    // Decide which stats to display: selected vs full table
                    let show_selected = selected_cnt > 0;
                    let (img_cnt, vid_cnt, dir_cnt, total_size) = if show_selected {
                        (sel_img_cnt, sel_vid_cnt, sel_dir_cnt, sel_total_size)
                    } else {
                        (all_img_cnt, all_vid_cnt, all_dir_cnt, all_total_size)
                    };

                    ui.label(format!(
                        "Total Size: {}",
                        humansize::format_size(total_size, DECIMAL)
                    ));
                    ui.separator();
                    ui.label(format!("Videos: {vid_cnt}"));
                    ui.separator();
                    ui.label(format!("Images: {img_cnt}"));
                    ui.separator();
                    ui.label(format!("Dirs: {dir_cnt}"));
                    ui.separator();
                    ui.label(format!("Filtered {global_filtered} of {global_total} (out: {filtered_out})"));
                    ui.separator();
                    ex.selection_menu(ui);
                    ui.separator();
                    ex.logical_group_menu(ui);
                });
            });
        });
    }
}

