use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};
use chrono::{Local, NaiveDate};
use eframe::egui::*;
use surrealdb::RecordId;

impl super::FileExplorer {
    pub fn navbar(&mut self, ui: &mut Ui) {
        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        MenuBar::new()
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .style(style)
        .ui(ui, |ui| {
            ui.set_height(25.);
            ui.horizontal_top(|ui| {
                let err_color = ui.style().visuals.error_fg_color;
                let font = FontId::proportional(15.);
                let size = vec2(25., 25.);
                ui.add_space(5.);
                if Button::new(
                    RichText::new("⚙").font(font.clone()).color(err_color)
                )
                .min_size(size)
                .ui(ui)
                .on_hover_text("Open Options side panel")
                .clicked() {
                    self.open_quick_access = !self.open_quick_access;
                }
                
                ui.add_space(5.);
                ui.separator();
                ui.add_space(5.);
                if Button::new(RichText::new("⬆").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_up(); }
                if Button::new(RichText::new("⬅").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_back(); }
                if Button::new(RichText::new("➡").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_forward(); }
                if Button::new(RichText::new("⟲").font(font.clone()).color(err_color)).min_size(size).ui(ui).clicked() { self.refresh(); }
                if Button::new(RichText::new("🏠").font(font).color(err_color)).min_size(size).ui(ui).clicked() { self.nav_home(); }
                ui.separator();
                let path_edit = TextEdit::singleline(&mut self.current_path)
                .hint_text(if self.viewer.mode == super::table::ExplorerMode::Database {
                    if self.ai_search_enabled { "AI Search (text prompt)" } else { "Path Prefix Filter" }
                } else {
                    "Current Directory"
                })
                .desired_width(300.)
                .ui(ui);

                // If browsing inside a zip archive, show a password button
                // Zip password button removed; we show a modal automatically when needed

                match self.viewer.mode {
                    super::table::ExplorerMode::FileSystem => {
                        if path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                            self.refresh();
                        }
                    },
                    super::table::ExplorerMode::Database => {
                        let mut ai_box = self.ai_search_enabled;
                        if ui.checkbox(&mut ai_box, "AI").on_hover_text("Use AI semantic search across the database (text prompt)").clicked() {
                            self.ai_search_enabled = ai_box;
                        }
                        // Apply filter on Enter
                        if (path_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                            || ui.button("Apply Filter").clicked()
                        {
                            if self.ai_search_enabled {
                                // Run AI semantic search over the whole DB using text prompt in current_path
                                let query = self.current_path.trim().to_string();
                                if !query.is_empty() {
                                    let tx_updates = self.viewer.ai_update_tx.clone();
                                    let types_show_images = self.viewer.types_show_images;
                                    let types_show_videos = self.viewer.types_show_videos;
                                    let minb = self.viewer.ui_settings.db_min_size_bytes;
                                    let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                    tokio::spawn(async move {
                                        // Ensure engine ready
                                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                        // Embed query text
                                        let q_vec_opt = {
                                            let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
                                            if let Some(engine) = guard.as_mut() {
                                                engine.embed_text(&query).ok()
                                            } else { None }
                                        };
                                        if let Some(q) = q_vec_opt {
                                            let mut results: Vec<crate::ui::file_table::SimilarResult> = Vec::new();
                                            match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, 48, 96).await {
                                                Ok(hits) => {
                                                    for hit in hits.into_iter() {
                                                        // Get thumbnail record (prefer embedded thumb_ref on hit)
                                                        let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                                        // Optional: filter by type and size
                                                        let ext_ok = std::path::Path::new(&thumb.path)
                                                            .extension()
                                                            .and_then(|e| e.to_str())
                                                            .map(|s| s.to_ascii_lowercase());
                                                        let mut type_ok = true;
                                                        if let Some(ext) = ext_ok.as_ref() {
                                                            if crate::is_image(ext.as_str()) { type_ok = types_show_images; }
                                                            else if crate::is_video(ext.as_str()) { type_ok = types_show_videos; }
                                                            else { type_ok = false; }
                                                        }
                                                        if !type_ok { continue; }
                                                        if let Some(mn) = minb { if thumb.size < mn { continue; } }
                                                        if let Some(mx) = maxb { if thumb.size > mx { continue; } }

                                                        // Compute similarity vs stored embedding (prefer clip_similarity_score)
                                                        let (mut created, mut updated, mut stored_sim, mut clip_sim) = (None, None, None, None);
                                                        if let Ok(rows) = crate::database::ClipEmbeddingRow::load_clip_embeddings_for_path(&thumb.path).await {
                                                            for row in rows.iter() {
                                                                created = row.created.clone();
                                                                updated = row.updated.clone();
                                                                stored_sim = row.similarity_score.or(row.clip_similarity_score);
                                                                if clip_sim.is_none() && !row.embedding.is_empty() { clip_sim = Some(crate::ai::clip::dot(&q, &row.embedding)); }
                                                            }
                                                        }
                                                        results.push(crate::ui::file_table::SimilarResult { thumb, created, updated, similarity_score: stored_sim, clip_similarity_score: clip_sim });
                                                    }
                                                }
                                                Err(e) => log::error!("[AI] text knn failed: {e:?}"),
                                            }
                                            let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: format!("query:{query}"), results });
                                        }
                                    });
                                }
                            } else {
                                // Reset paging and reload
                                self.db_offset = 0;
                                self.db_last_batch_len = 0;
                                self.load_database_rows();
                            }
                        }
                    },
                }
                
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    MenuButton::new("🔻")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style))
                    .ui(ui, |ui| {
                        ui.set_width(250.);
                        ui.heading("Excluded Terms");
                        ui.horizontal(|ui| {
                            let resp = TextEdit::singleline(&mut self.excluded_term_input)
                                .hint_text("term")
                                .desired_width(140.)
                                .ui(ui);
                            let add_clicked = ui.button("Add").clicked();
                            if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || add_clicked {
                                resp.request_focus();
                                let term = self.excluded_term_input.trim().to_ascii_lowercase();
                                if !term.is_empty() && !self.excluded_terms.iter().any(|t| t == &term) { self.excluded_terms.push(term); }
                                // Manual change invalidates active preset label
                                self.active_filter_group = None;
                                self.excluded_term_input.clear();
                                // Apply excluded terms immediately to current table (keep directories)
                                if !self.excluded_terms.is_empty() {
                                    let terms = self.excluded_terms.clone();
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        let lp = r.path.to_ascii_lowercase();
                                        !terms.iter().any(|t| lp.contains(t))
                                    });
                                }
                            }
                            if ui.button("Clear All").clicked() { 
                                self.excluded_terms.clear();
                                self.active_filter_group = None;
                                // Note: clearing does not restore previously filtered rows to keep UX consistent with size filters
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            let mut remove_idx: Option<usize> = None;
                            for (i, term) in self.excluded_terms.iter().enumerate() {
                                if ui.add(Button::new(format!("{} ✖", term)).small()).clicked() { remove_idx = Some(i); }
                            }
                            if let Some(i) = remove_idx { 
                                self.excluded_terms.remove(i);
                                // Re-apply exclusion after removal (cannot restore already removed rows)
                                if !self.excluded_terms.is_empty() {
                                    let terms = self.excluded_terms.clone();
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        let lp = r.path.to_ascii_lowercase();
                                        !terms.iter().any(|t| lp.contains(t))
                                    });
                                }
                            }
                        });

                        ui.separator();
                        ui.add_space(4.0);
                        ui.heading("Size Filters (KB)");
                        ui.horizontal(|ui| {
                            ui.label("Min:");
                            let mut min_kb: i64 = self.viewer.ui_settings.db_min_size_bytes
                                .map(|b| (b / 1024) as i64)
                                .unwrap_or(0);
                            let mut max_kb: i64 = self.viewer.ui_settings.db_max_size_bytes
                                .map(|b| (b / 1024) as i64)
                                .unwrap_or(0);
                            let min_resp = egui::DragValue::new(&mut min_kb).speed(1).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Minimum file size in KiB");
                            ui.label("Max:");
                            let max_resp = egui::DragValue::new(&mut max_kb).speed(10).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Maximum file size in KiB (0 = no max)");
                            let changed = min_resp.changed() || max_resp.changed();
                            if changed || ui.button("Apply").clicked() {
                                let min_b = if min_kb <= 0 { None } else { Some(min_kb as u64 * 1024) };
                                let max_b = if max_kb <= 0 { None } else { Some(max_kb as u64 * 1024) };
                                // Enforce min <= max if both set
                                let (min_b, max_b) = match (min_b, max_b) {
                                    (Some(a), Some(b)) if a > b => (Some(b), Some(a)),
                                    other => other,
                                };
                                self.viewer.ui_settings.db_min_size_bytes = min_b;
                                self.viewer.ui_settings.db_max_size_bytes = max_b;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                                // Apply to current table view
                                let minb = self.viewer.ui_settings.db_min_size_bytes;
                                let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                self.table.retain(|r| {
                                    if r.file_type == "<DIR>" { return true; }
                                    let ok_min = minb.map(|m| r.size >= m).unwrap_or(true);
                                    let ok_max = maxb.map(|m| r.size <= m).unwrap_or(true);
                                    ok_min && ok_max
                                });
                            }
                            if ui.button("Clear").on_hover_text("Clear size constraints").clicked() {
                                self.min_size_mb_input.clear();
                                self.max_size_mb_input.clear();
                                self.viewer.ui_settings.db_min_size_bytes = None;
                                self.viewer.ui_settings.db_max_size_bytes = None;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);
                        ui.heading("Extension filters");
                        ui.horizontal(|ui| {
                            let changed_i = ui.checkbox(&mut self.viewer.types_show_images, "Images").changed();
                            let changed_v = ui.checkbox(&mut self.viewer.types_show_videos, "Videos").changed();
                            let changed_d = ui.checkbox(&mut self.viewer.types_show_dirs, "Folders").changed();
                            if changed_i || changed_v || changed_d { self.active_filter_group = None; }
                        });
                        ui.horizontal(|ui| {
                            let mut skip_icons = self.viewer.ui_settings.filter_skip_icons;
                            if ui.checkbox(&mut skip_icons, "Skip likely icons").on_hover_text("Filter out tiny images (.ico, <= 16–64px, and small files)").changed() {
                                self.viewer.ui_settings.filter_skip_icons = skip_icons;
                                crate::database::settings::save_settings(&self.viewer.ui_settings);
                                self.active_filter_group = None;
                                if skip_icons {
                                    // Apply to current table using same heuristic as scanner
                                    let tiny_thresh = self.viewer.ui_settings.db_min_size_bytes.unwrap_or(10 * 1024);
                                    self.table.retain(|r| {
                                        if r.file_type == "<DIR>" { return true; }
                                        // ico extension
                                        let is_ico = std::path::Path::new(&r.path)
                                            .extension()
                                            .and_then(|e| e.to_str())
                                            .map(|s| s.eq_ignore_ascii_case("ico"))
                                            .unwrap_or(false);
                                        if is_ico { return false; }
                                        // Evaluate size (use row.size, fall back to fs)
                                        let mut size_val = r.size;
                                        if size_val == 0 { if let Ok(md) = std::fs::metadata(&r.path) { size_val = md.len(); } }
                                        if size_val > tiny_thresh { return true; }
                                        // Only images: check tiny dims
                                        let is_img = std::path::Path::new(&r.path)
                                            .extension()
                                            .and_then(|e| e.to_str())
                                            .map(|s| s.to_ascii_lowercase())
                                            .map(|ext| crate::is_image(ext.as_str()))
                                            .unwrap_or(false);
                                        if !is_img { return true; }
                                        let tiny_dims = image::ImageReader::open(&r.path)
                                            .ok()
                                            .and_then(|r| r.with_guessed_format().ok())
                                            .and_then(|r| r.into_dimensions().ok())
                                            .map(|(w,h)| w <= 64 && h <= 64)
                                            .unwrap_or(false);
                                        !(size_val <= tiny_thresh && tiny_dims)
                                    });
                                }
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);
                        ui.heading("Save filters as group");
                        ui.horizontal(|ui| {
                            ui.add_sized([180., 22.], TextEdit::singleline(&mut self.filter_group_name_input).hint_text("Group name"));
                            let save_clicked = ui.button("Save").clicked();
                            if save_clicked && !self.filter_group_name_input.trim().is_empty() {
                                let name = self.filter_group_name_input.trim().to_string();
                                let include_images = self.viewer.types_show_images;
                                let include_videos = self.viewer.types_show_videos;
                                let include_dirs = self.viewer.types_show_dirs;
                                let skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                let minb = self.viewer.ui_settings.db_min_size_bytes;
                                let maxb = self.viewer.ui_settings.db_max_size_bytes;
                                let excluded_terms = self.excluded_terms.clone();
                                let tx_groups = self.filter_groups_tx.clone();
                                tokio::spawn(async move {
                                    let g = crate::database::FilterGroup {
                                        id: None,
                                        name,
                                        include_images,
                                        include_videos,
                                        include_dirs,
                                        skip_icons,
                                        min_size_bytes: minb,
                                        max_size_bytes: maxb,
                                        excluded_terms,
                                        created: None,
                                        updated: None,
                                    };
                                    match crate::database::save_filter_group(g).await {
                                        Ok(_) => {
                                            // After saving, refresh list
                                            match crate::database::list_filter_groups().await {
                                                Ok(groups) => { let _ = tx_groups.try_send(groups); },
                                                Err(e) => log::error!("list_filter_groups after save failed: {e:?}"),
                                            }
                                        }
                                        Err(e) => log::error!("save filter group failed: {e:?}"),
                                    }
                                });
                                self.filter_group_name_input.clear();
                            }
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Saved filter groups").italics());
                            if ui.button("⟲ Refresh").clicked() {
                                let tx = self.filter_groups_tx.clone();
                                tokio::spawn(async move {
                                    match crate::database::list_filter_groups().await {
                                        Ok(groups) => { let _ = tx.try_send(groups); },
                                        Err(e) => log::error!("list_filter_groups failed: {e:?}"),
                                    }
                                });
                            }
                        });
                        if self.filter_groups.is_empty() {
                            ui.label(RichText::new("No saved groups yet.").weak());
                        } else {
                            egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                                let groups = self.filter_groups.clone();
                                for g in groups.iter() {
                                    ui.horizontal(|ui| {
                                        ui.label(&g.name);
                                        if ui.button("Apply").on_hover_text("Apply this filter group to settings and current table").clicked() {
                                            // Apply toggles and settings
                                            self.viewer.types_show_images = g.include_images;
                                            self.viewer.types_show_videos = g.include_videos;
                                            self.viewer.types_show_dirs = g.include_dirs;
                                            self.viewer.ui_settings.filter_skip_icons = g.skip_icons;
                                            self.viewer.ui_settings.db_min_size_bytes = g.min_size_bytes;
                                            self.viewer.ui_settings.db_max_size_bytes = g.max_size_bytes;
                                            crate::database::settings::save_settings(&self.viewer.ui_settings);
                                            self.excluded_terms = g.excluded_terms.clone();
                                            self.excluded_term_input.clear();
                                            self.active_filter_group = Some(g.name.clone());
                                            // Prune current table with the newly applied filters
                                            self.apply_filters_to_current_table();
                                        }
                                        if ui.button("Delete").on_hover_text("Delete this saved group").clicked() {
                                            let name = g.name.clone();
                                            let name_for_spawn = name.clone();
                                            let tx = self.filter_groups_tx.clone();
                                            let active = self.active_filter_group.clone();
                                            tokio::spawn(async move {
                                                match crate::database::delete_filter_group_by_name(&name_for_spawn).await {
                                                    Ok(_) => {
                                                        match crate::database::list_filter_groups().await {
                                                            Ok(groups) => { let _ = tx.try_send(groups); },
                                                            Err(e) => log::error!("refresh groups after delete failed: {e:?}"),
                                                        }
                                                    }
                                                    Err(e) => log::error!("delete filter group failed: {e:?}"),
                                                }
                                            });
                                            if active.as_deref() == Some(&name) { self.active_filter_group = None; }
                                        }
                                    });
                                }
                            });
                        }
                        ui.separator();
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            if ui.button("Reload Rows").on_hover_text("Reload current view rows").clicked() {
                                if self.viewer.mode == super::table::ExplorerMode::Database {
                                    self.db_offset = 0;
                                    self.db_last_batch_len = 0;
                                    self.load_database_rows();
                                } else {
                                    self.table.clear();
                                    self.populate_current_directory();
                                }
                            }
                            if !self.last_scan_rows.is_empty() {
                                if ui.button("Restore Last Scan").on_hover_text("Restore last recursive scan results").clicked() {
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
                        });
                    });

                    ui.menu_button("👁", |ui| {
                        ui.set_width(250.);
                        ui.checkbox(&mut self.open_preview_pane, "Show Preview Pane");
                        ui.checkbox(&mut self.open_quick_access, "Show Quick Access");
                        ui.separator();
                        ui.checkbox(&mut self.follow_active_vision, "Follow active vision (auto-select)")
                            .on_hover_text("When enabled, the preview auto-selects the image currently being described.");
                        ui.separator();
                        // Auto-save to DB toggle (persisted)
                        let mut auto_save = self.viewer.ui_settings.auto_save_to_database;
                        if ui.checkbox(&mut auto_save, "Auto save to database").on_hover_text("When enabled, newly discovered files will be saved to the database automatically (requires an active logical group)").changed() {
                            self.viewer.ui_settings.auto_save_to_database = auto_save;
                            crate::database::settings::save_settings(&self.viewer.ui_settings);
                        }
                        if self.active_logical_group_name.is_none() {
                            ui.colored_label(Color32::YELLOW, "No active logical group. Auto save and indexing are gated.");
                        } else {
                            ui.colored_label(Color32::LIGHT_GREEN, format!("Active Group: {}", self.active_logical_group_name.clone().unwrap_or_default()));
                        }

                    });

                    ui.separator();

                    TextEdit::singleline(&mut self.viewer.filter)
                    .desired_width(200.0)
                    .hint_text("Search for files")
                    .ui(ui);

                    ui.separator();

                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    
                    MenuButton::new("Scan")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
                    .ui(ui, |ui| {
                        ui.vertical_centered_justified(|ui| { 
                            ui.set_width(250.);
                            ui.heading("Recursive Scanning");
                            if Button::new("Recursive Scan").right_text("💡").ui(ui).clicked() {
                                let title = format!("Scan: {}", self.current_path);
                                let path = self.current_path.clone();
                                crate::app::OPEN_TAB_REQUESTS
                                    .lock()
                                    .unwrap()
                                    .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive: true, background: false });
                            }

                            ui.separator();
                            ui.heading("Scan by Type (Recursive)");
                            if Button::new("Scan Images").ui(ui).on_hover_text("Recursive scan including only images").clicked() {
                                self.recursive_scan = true;
                                self.scan_done = false;
                                self.table.clear();
                                self.last_scan_rows.clear();
                                self.last_scan_paths.clear();
                                self.last_scan_root = Some(self.current_path.clone());
                                self.file_scan_progress = 0.0;
                                let scan_id = crate::next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = true;
                                let mut filters = crate::Filters::default();
                                filters.root = std::path::PathBuf::from(self.current_path.clone());
                                filters.include_images = true;
                                filters.include_videos = false;
                                filters.include_archives = false;
                                filters.excluded_terms = self.excluded_terms.clone();
                                self.owning_scan_id = Some(scan_id);
                                self.current_scan_id = Some(scan_id);
                                tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                            }
                            if Button::new("Scan Videos").ui(ui).on_hover_text("Recursive scan including only videos").clicked() {
                                self.recursive_scan = true;
                                self.scan_done = false;
                                self.table.clear();
                                self.last_scan_rows.clear();
                                self.last_scan_paths.clear();
                                self.last_scan_root = Some(self.current_path.clone());
                                self.file_scan_progress = 0.0;
                                let scan_id = crate::next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = true;
                                let mut filters = crate::Filters::default();
                                filters.root = std::path::PathBuf::from(self.current_path.clone());
                                filters.include_images = false;
                                filters.include_videos = true;
                                filters.include_archives = false;
                                filters.excluded_terms = self.excluded_terms.clone();
                                self.owning_scan_id = Some(scan_id);
                                self.current_scan_id = Some(scan_id);
                                tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                            }
                            if Button::new("Scan Archives (.zip/.7z/.rar/.tar)").ui(ui).on_hover_text("Recursive scan including only archive containers").clicked() {
                                self.recursive_scan = true;
                                self.scan_done = false;
                                self.table.clear();
                                self.last_scan_rows.clear();
                                self.last_scan_paths.clear();
                                self.last_scan_root = Some(self.current_path.clone());
                                self.file_scan_progress = 0.0;
                                let scan_id = crate::next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = true;
                                let mut filters = crate::Filters::default();
                                filters.root = std::path::PathBuf::from(self.current_path.clone());
                                filters.include_images = false;
                                filters.include_videos = false;
                                filters.include_archives = true;
                                filters.excluded_terms = self.excluded_terms.clone();
                                self.owning_scan_id = Some(scan_id);
                                self.current_scan_id = Some(scan_id);
                                tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                            }
                            
                            if Button::new("Re-scan Current Folder").right_text("🔄").ui(ui).on_hover_text("Shallow scan with current filters (applies 'Skip likely icons')").clicked() {
                                self.recursive_scan = false;
                                self.scan_done = false;
                                self.table.clear();
                                self.last_scan_rows.clear();
                                self.last_scan_paths.clear();
                                self.last_scan_root = Some(self.current_path.clone());
                                self.file_scan_progress = 0.0;
                                let scan_id = crate::next_scan_id();
                                let tx = self.scan_tx.clone();
                                let recurse = false;
                                let mut filters = crate::Filters::default();
                                filters.root = std::path::PathBuf::from(self.current_path.clone());
                                filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                                filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                                filters.include_images = self.viewer.types_show_images;
                                filters.include_videos = self.viewer.types_show_videos;
                                filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                                filters.excluded_terms = self.excluded_terms.clone();
                                // Lock incoming scan updates to this tab by setting owning + current scan ids
                                self.owning_scan_id = Some(scan_id);
                                self.current_scan_id = Some(scan_id);
                                tokio::spawn(async move { crate::spawn_scan(filters, tx, recurse, scan_id).await; });
                            }
                
                            if Button::new("Cancel Scan").right_text(RichText::new("■").color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text("Cancel active recursive scan").clicked() {
                                if let Some(id) = self.current_scan_id.take() { crate::utilities::scan::cancel_scan(id); }
                            }
                            // (Performance Tweaks moved to Quick Access pane)
                            ui.separator();
                            ui.heading("Database Save");
                            
                            if Button::new("Save Current View to DB").right_text(RichText::new("💾")).ui(ui).on_hover_text("Upsert all currently visible rows into the database and add them to the active logical group").clicked() {
                                let group_opt = self.active_logical_group_name.clone();
                                let rows: Vec<crate::database::Thumbnail> = self
                                    .table
                                    .iter()
                                    .filter(|r| r.file_type != "<DIR>")
                                    .cloned()
                                    .collect();
                                if !rows.is_empty() {
                                    tokio::spawn(async move {
                                        match crate::database::upsert_rows_and_get_ids(rows).await {
                                            Ok(ids) => {
                                                if let Some(group_name) = group_opt {
                                                    match crate::database::LogicalGroup::get_by_name(&group_name).await {
                                                        Ok(Some(g)) => {
                                                            if let Err(e) = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await {
                                                                log::error!("Add thumbs to group failed: {e:?}");
                                                            } else {
                                                                log::info!("Saved {} rows to DB and associated with group '{}'", ids.len(), group_name);
                                                            }
                                                        }
                                                        Ok(None) => log::warn!("Active group '{}' not found during save", group_name),
                                                        Err(e) => log::error!("get_by_name failed: {e:?}"),
                                                    }
                                                } else {
                                                    log::info!("Saved {} rows to DB (no group association)", ids.len());
                                                }
                                            }
                                            Err(e) => log::error!("Upsert rows failed: {e:?}"),
                                        }
                                    });
                                }
                            }
                            
                            if !self.last_scan_rows.is_empty() {
                                if Button::new("Return to Active Scan").right_text("↩").ui(ui).on_hover_text("Restore the last recursive scan results without rescanning").clicked() {
                                
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
                        
                            ui.separator();

                            ui.heading("Bulk Operations");
                            if Button::new("Bulk Generate Descriptions").right_text("🖩").ui(ui).on_hover_text("Generate AI descriptions for images missing caption/description").clicked() {
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
                            
                            if Button::new("Generate Missing CLIP Embeddings").right_text("⚡").ui(ui).clicked() {
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

                            if Button::new("Cancel Bulk Descriptions").right_text(RichText::new("■").color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text("Stop scheduling/streaming new vision descriptions").clicked() {
                                crate::ai::GLOBAL_AI_ENGINE.cancel_bulk_descriptions();
                                crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
                                self.viewer.bulk_cancel_requested = true;
                            }
                            
                        });
                    });

                    // Basic backup actions: copy/move all visible files to a selected directory
                    MenuButton::new("Files")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
                    .ui(ui, |ui| {
                        ui.set_width(250.);
                        ui.vertical_centered_justified(|ui| {
                            ui.heading("Backup Files");
                            if ui.button("Copy Visible Files to Folder…").clicked() {
                                if let Some(dir) = rfd::FileDialog::new().set_title("Choose backup destination").pick_folder() {
                                    self.backup_copy_visible_to_dir(dir);
                                }
                            }
                            if ui.button("Move Visible Files to Folder…").clicked() {
                                if let Some(dir) = rfd::FileDialog::new().set_title("Choose destination for move").pick_folder() {
                                    self.backup_move_visible_to_dir(dir);
                                }
                            }
                            ui.label("Operates on all non-directory rows currently visible in the table.");
                        });
                    });
                    
                    MenuButton::new("Table")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style))
                    .ui(ui, |ui| {
                        ui.vertical_centered_justified(|ui| {
                            let db_res = ui.selectable_value(&mut self.viewer.mode, super::table::ExplorerMode::Database, "Database");
                            let fs_res = ui.selectable_value(&mut self.viewer.mode, super::table::ExplorerMode::FileSystem, "FileSystem");

                            if db_res.changed() || fs_res.changed() {
                                match self.viewer.mode {
                                    super::table::ExplorerMode::Database => self.load_database_rows(),
                                    super::table::ExplorerMode::FileSystem => self.populate_current_directory()
                                }
                            }
                            if ui.button("Reload Page").clicked() { self.load_database_rows(); }
                            if ui.button("Clear Table").clicked() { self.table.clear(); }
                            ui.separator();
                            if ui.button(RichText::new("Delete Visible From DB").color(ui.style().visuals.error_fg_color))
                                .on_hover_text("Delete all currently visible (filtered) rows from the database, including their CLIP embeddings. Files on disk are NOT affected.")
                                .clicked()
                            {
                                // Confirm destructive action
                                let count = self.table.iter().filter(|r| r.file_type != "<DIR>").count();
                                if count > 0 {
                                    if rfd::MessageDialog::new()
                                        .set_title("Delete from Database")
                                        .set_description(&format!("Permanently delete {} records from the database (and their embeddings)? This will not delete files on disk.", count))
                                        .set_level(rfd::MessageLevel::Warning)
                                        .set_buttons(rfd::MessageButtons::YesNo)
                                        .show() == rfd::MessageDialogResult::Yes
                                    {
                                        let paths: Vec<RecordId> = self
                                            .table
                                            .iter()
                                            .filter(|r| r.file_type != "<DIR>")
                                            .map(|r| r.id.clone())
                                            .collect();
                                        tokio::spawn(async move {
                                            match crate::database::delete_thumbnails_and_embeddings_by_paths(paths).await {
                                                Ok((emb, thumbs)) => log::info!("Deleted {} embeddings and {} thumbnails.", emb, thumbs),
                                                Err(e) => log::error!("Delete visible failed: {e:?}"),
                                            }
                                        });
                                        // Remove from the current table immediately for UX
                                        self.table.retain(|r| r.file_type == "<DIR>");
                                    }
                                }
                            }
                        });

                        ui.add_space(4.0);
                        if matches!(self.viewer.mode, super::table::ExplorerMode::Database) {
                            ui.label(format!("Loaded Rows: {} (offset {})", self.table.len(), self.db_offset));
                            if self.db_loading { ui.colored_label(Color32::YELLOW, "Loading..."); }
                        }
                    });

                    let style = StyleModifier::default();
                    style.apply(ui.style_mut());
                    MenuButton::new("Logical Groups")
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
                    .ui(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.set_width(470.);
                            ui.heading("Logical Groups");
                            ui.horizontal(|ui| {
                                if let Some(name) = &self.active_logical_group_name {
                                    ui.label(format!("Active: {}", name));
                                } else {
                                    ui.label("Active: (none)");
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("⟲ Refresh").clicked() {
                                        let tx = self.logical_groups_tx.clone();
                                        tokio::spawn(async move {
                                            match crate::database::LogicalGroup::list_all().await {
                                                Ok(groups) => { let _ = tx.try_send(groups); },
                                                Err(e) => log::error!("refresh logical groups failed: {e:?}"),
                                            }
                                        });
                                    }
                                });
                            });
                            ui.separator();
                            ui.label(RichText::new("Operations").italics());
                            ui.horizontal(|ui| {
                                ui.label("Add selection to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_target.trim().to_string();
                                        if !target.is_empty() && !self.viewer.selected.is_empty() {
                                            // Gather selected rows and upsert to get ids
                                            let rows: Vec<crate::database::Thumbnail> = self
                                                .table
                                                .iter()
                                                .filter(|r| r.file_type != "<DIR>" && self.viewer.selected.contains(&r.path))
                                                .cloned()
                                                .collect();
                                            if !rows.is_empty() {
                                                tokio::spawn(async move {
                                                    // Ensure rows exist in DB and get their ids
                                                    match crate::database::upsert_rows_and_get_ids(rows).await {
                                                        Ok(ids) => {
                                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                                            } else {
                                                                if let Ok(_) = crate::database::LogicalGroup::create(&target).await {
                                                                    if let Ok(Some(g2)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                        let _ = crate::database::LogicalGroup::add_thumbnails(&g2.id, &ids).await;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Err(e) => log::error!("Add selection: upsert failed: {e:?}"),
                                                    }
                                                });
                                            }
                                        }
                                    }
                                });
                            });

                            ui.add_space(5.);

                            ui.horizontal(|ui| {
                                ui.label("Add current view to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_view_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_view_target.trim().to_string();
                                        if !target.is_empty() {
                                            let rows: Vec<crate::database::Thumbnail> = self
                                            .table
                                            .iter()
                                            .filter(|r| r.file_type != "<DIR>")
                                            .cloned()
                                            .collect();

                                            if !rows.is_empty() {
                                                tokio::spawn(async move {
                                                    match crate::database::upsert_rows_and_get_ids(rows).await {
                                                        Ok(ids) => {
                                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                                            } else {
                                                                if let Ok(_) = crate::database::LogicalGroup::create(&target).await {
                                                                    if let Ok(Some(g2)) = crate::database::LogicalGroup::get_by_name(&target).await {
                                                                        let _ = crate::database::LogicalGroup::add_thumbnails(&g2.id, &ids).await;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Err(e) => log::error!("Add view: upsert failed: {e:?}"),
                                                    }
                                                });
                                            }
                                        }
                                    }
                                });
                            });

                            ui.add_space(5.);

                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Copy From: "));
                                TextEdit::singleline(&mut self.group_copy_src).hint_text("Group to Copy From").desired_width(150.).ui(ui);
                                ui.label(RichText::new(" -> "));
                                TextEdit::singleline(&mut self.group_copy_dst).hint_text("Destination group").desired_width(150.).ui(ui);

                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("Copy").clicked() {
                                        let src = self.group_copy_src.trim().to_string();
                                        let dst = self.group_copy_dst.trim().to_string();
                                        if !src.is_empty() && !dst.is_empty() && src != dst {
                                            tokio::spawn(async move {
                                                if let Ok(Some(src_g)) = crate::database::LogicalGroup::get_by_name(&src).await {
                                                    let ids = crate::Thumbnail::fetch_ids_by_logical_group_id(&src_g.id).await.unwrap_or_default();
                                                    if !ids.is_empty() {
                                                        // Ensure destination exists and get its id
                                                        let dst_gid: Option<surrealdb::RecordId> = match crate::database::LogicalGroup::get_by_name(&dst).await {
                                                            Ok(Some(g)) => Some(g.id),
                                                            _ => {
                                                                let _ = crate::database::LogicalGroup::create(&dst).await;
                                                                match crate::database::LogicalGroup::get_by_name(&dst).await { Ok(Some(g)) => Some(g.id), _ => None }
                                                            }
                                                        };
                                                        if let Some(gid) = dst_gid {
                                                            let _ = crate::database::LogicalGroup::add_thumbnails(&gid, &ids).await;
                                                        }
                                                    }
                                                }
                                            });
                                        }
                                    }
                                });
                            });
                            
                            ui.add_space(5.);
                            
                            ui.horizontal(|ui| {
                                ui.label("Add unassigned to:");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let response = TextEdit::singleline(&mut self.group_add_unassigned_target).hint_text("Group name").desired_width(150.).ui(ui);
                                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                        let target = self.group_add_unassigned_target.trim().to_string();
                                        if !target.is_empty() {
                                            tokio::spawn(async move {
                                                match crate::database::LogicalGroup::list_unassigned_thumbnail_ids().await {
                                                    Ok(ids) => {
                                                        log::info!("Thumbnails without a group: {}", ids.len());
                                                        if !ids.is_empty() {
                                                            // ensure destination group exists
                                                            let gid_opt: Option<surrealdb::RecordId> = match crate::database::LogicalGroup::get_by_name(&target).await {
                                                                Ok(Some(g)) => Some(g.id),
                                                                _ => {
                                                                    let _ = crate::database::LogicalGroup::create(&target).await;
                                                                    match crate::database::LogicalGroup::get_by_name(&target).await { Ok(Some(g)) => Some(g.id), _ => None }
                                                                }
                                                            };
                                                            if let Some(gid) = gid_opt {
                                                                let _ = crate::database::LogicalGroup::add_thumbnails(&gid, &ids).await;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => log::error!("List unassigned failed: {e:?}"),
                                                }
                                            });
                                        }
                                    }
                                    if let Some(gname) = &self.active_logical_group_name {
                                        if ui.button("To Current Group").on_hover_text("Add all unassigned thumbnails to the active group").clicked() {
                                            let target = gname.clone();
                                            tokio::spawn(async move {
                                                match crate::database::LogicalGroup::list_unassigned_thumbnail_ids().await {
                                                    Ok(ids) => {
                                                        if ids.is_empty() { return; }
                                                        // Ensure group exists then add
                                                        let gid_opt: Option<surrealdb::RecordId> = match crate::database::LogicalGroup::get_by_name(&target).await {
                                                            Ok(Some(g)) => Some(g.id),
                                                            _ => {
                                                                let _ = crate::database::LogicalGroup::create(&target).await;
                                                                match crate::database::LogicalGroup::get_by_name(&target).await { Ok(Some(g)) => Some(g.id), _ => None }
                                                            }
                                                        };
                                                        if let Some(gid) = gid_opt {
                                                            if let Err(e) = crate::database::LogicalGroup::add_thumbnails(&gid, &ids).await {
                                                                log::error!("Add unassigned to current group failed: {e:?}");
                                                            }
                                                        }
                                                    }
                                                    Err(e) => log::error!("List unassigned failed: {e:?}"),
                                                }
                                            });
                                        }
                                    }
                                });
                            });

                            ui.separator();

                            ui.label(RichText::new("Create new group").italics());
                            
                            ui.horizontal(|ui| {
                                let response = TextEdit::singleline(&mut self.group_create_name_input).hint_text("Group name").desired_width(150.).ui(ui);
                                let btn = ui.with_layout(Layout::right_to_left(Align::Center), |ui| ui.button("Create")).inner;
                                if ( response.lost_focus() && response.changed() ) || btn.clicked() {
                                    let name = self.group_create_name_input.trim().to_string();
                                    if !name.is_empty() {
                                        let tx = self.logical_groups_tx.clone();
                                        tokio::spawn(async move {
                                            match crate::database::LogicalGroup::create(&name).await {
                                                Ok(_) => {
                                                    match crate::database::LogicalGroup::list_all().await {
                                                        Ok(groups) => { let _ = tx.try_send(groups); },
                                                        Err(e) => log::error!("list groups after create failed: {e:?}"),
                                                    }
                                                }
                                                Err(e) => log::error!("create group failed: {e:?}"),
                                            }
                                        });
                                        self.group_create_name_input.clear();
                                    }
                                }
                            });
                            ui.separator();
                            if self.logical_groups.is_empty() {
                                ui.label(RichText::new("No groups defined yet.").weak());
                            } else {
                                egui::ScrollArea::vertical().max_height(180.).show(ui, |ui| {
                                    for g in self.logical_groups.clone().into_iter() {
                                        ui.horizontal(|ui| {
                                            // Name or rename editor
                                            if self.group_rename_target.as_deref() == Some(g.name.as_str()) {
                                                ui.add_sized([180., 22.], TextEdit::singleline(&mut self.group_rename_input));
                                                if ui.button("Save").on_hover_text("Rename group").clicked() {
                                                    let new_name = self.group_rename_input.trim().to_string();
                                                    if !new_name.is_empty() {
                                                        let gid = g.id.clone();
                                                        let tx = self.logical_groups_tx.clone();
                                                        tokio::spawn(async move {
                                                            match crate::database::LogicalGroup::rename(&gid, &new_name).await {
                                                                Ok(_) => {
                                                                    match crate::database::LogicalGroup::list_all().await {
                                                                        Ok(groups) => { let _ = tx.try_send(groups); },
                                                                        Err(e) => log::error!("list groups after rename failed: {e:?}"),
                                                                    }
                                                                }
                                                                Err(e) => log::error!("rename group failed: {e:?}"),
                                                            }
                                                        });
                                                    }
                                                    self.group_rename_target = None;
                                                    self.group_rename_input.clear();
                                                }
                                                if ui.button("Cancel").clicked() {
                                                    self.group_rename_target = None;
                                                    self.group_rename_input.clear();
                                                }
                                            } else {
                                                ui.label(&g.name);
                                            }

                                            if ui.button("Select").on_hover_text("Use this as the active group (no mode switch)").clicked() {
                                                self.active_logical_group_name = Some(g.name.clone());
                                            }
                                            if ui.button("Open").on_hover_text("Load this group's thumbnails").clicked() {
                                                self.active_logical_group_name = Some(g.name.clone());
                                                self.viewer.mode = super::table::ExplorerMode::Database;
                                                self.table.clear();
                                                self.db_offset = 0;
                                                self.db_last_batch_len = 0;
                                                self.db_loading = true;
                                                self.load_logical_group_by_name(g.name.clone());
                                            }
                                            if self.group_rename_target.as_deref() != Some(g.name.as_str()) {
                                                if ui.button("Rename").clicked() {
                                                    self.group_rename_target = Some(g.name.clone());
                                                    self.group_rename_input = g.name.clone();
                                                }
                                            }
                                            if ui.button("Delete").on_hover_text("Delete this group").clicked() {
                                                let gid = g.id.clone();
                                                let tx = self.logical_groups_tx.clone();
                                                let active = self.active_logical_group_name.clone();
                                                let name = g.name.clone();
                                                tokio::spawn(async move {
                                                    match crate::database::LogicalGroup::delete(&gid).await {
                                                        Ok(_) => {
                                                            match crate::database::LogicalGroup::list_all().await {
                                                                Ok(groups) => { let _ = tx.try_send(groups); },
                                                                Err(e) => log::error!("list groups after delete failed: {e:?}"),
                                                            }
                                                        }
                                                        Err(e) => log::error!("delete group failed: {e:?}"),
                                                    }
                                                });
                                                if active.as_deref() == Some(name.as_str()) {
                                                    self.active_logical_group_name = None;
                                                }
                                            }
                                        });
                                    }
                                });
                            }
                        });
                    });

                    // Selection operations menu
                    MenuButton::new(format!("Selection {}", self.selection_count()))
                    .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
                    .ui(ui, |ui| {
                        ui.set_width(350.);
                        ui.heading("Selection");
                        let selected_dirs: Vec<String> = self
                            .table
                            .iter()
                            .filter(|r| r.file_type == "<DIR>" && self.viewer.selected.contains(&r.path))
                            .map(|r| r.path.clone())
                            .collect();
                        let count = selected_dirs.len();
                        if count == 0 {
                            ui.label(RichText::new("No directories selected").weak());
                        } else {
                            ui.label(format!("Selected directories: {}", count));
                        }
                        ui.separator();
                        let disabled = count == 0;
                        let btn = ui.add_enabled(!disabled, Button::new("Exclude selected directories from recursive scans").small());
                        if btn.clicked() {
                            if self.viewer.ui_settings.excluded_dirs.is_none() {
                                self.viewer.ui_settings.excluded_dirs = Some(Vec::new());
                            }
                            if let Some(ref mut dirs) = self.viewer.ui_settings.excluded_dirs {
                                for p in selected_dirs.into_iter() {
                                    if !dirs.contains(&p) { dirs.push(p); }
                                }
                            }
                            crate::database::settings::save_settings(&self.viewer.ui_settings);
                            self.apply_filters_to_current_table();
                            ui.close();
                        }
                    });

                    ui.separator();

                    if ui.button("✖").on_hover_text("Clear end date").clicked() {
                        self.viewer.ui_settings.filter_modified_before = None;
                        crate::database::settings::save_settings(&self.viewer.ui_settings);
                        self.active_filter_group = None;
                        self.apply_filters_to_current_table();
                    }

                    // Parse current settings or default to today (without forcing persistence until user changes)
                    let today: NaiveDate = Local::now().date_naive();
                    let mut after_date: NaiveDate = self.viewer.ui_settings
                        .filter_modified_after
                        .as_deref()
                        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
                        .unwrap_or(today);

                    let mut before_date: NaiveDate = self.viewer.ui_settings
                        .filter_modified_before
                        .as_deref()
                        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
                        .unwrap_or(today);

                    let before_id = format!("{before_date} End date");
                    let resp_before = egui_extras::DatePickerButton::new(&mut before_date)
                    .id_salt(&before_id)
                    .ui(ui);

                    ui.label("  ->  ");

                    if ui.button("✖").on_hover_text("Clear start date").clicked() {
                        self.viewer.ui_settings.filter_modified_after = None;
                        crate::database::settings::save_settings(&self.viewer.ui_settings);
                        self.active_filter_group = None;
                        self.apply_filters_to_current_table();
                    }

                    let after_id =  format!("{after_date} Start date");
                    let resp_after = egui_extras::DatePickerButton::new(&mut after_date)
                    .id_salt(&after_id)
                    .ui(ui);

                    if resp_after.changed() {
                        self.viewer.ui_settings.filter_modified_after = Some(after_date.format("%Y-%m-%d").to_string());
                        crate::database::settings::save_settings(&self.viewer.ui_settings);
                        self.active_filter_group = None;
                        self.apply_filters_to_current_table();
                    }
                    if resp_before.changed() {
                        self.viewer.ui_settings.filter_modified_before = Some(before_date.format("%Y-%m-%d").to_string());
                        crate::database::settings::save_settings(&self.viewer.ui_settings);
                        self.active_filter_group = None;
                        self.apply_filters_to_current_table();
                    }

                });
            });
        });
    }
}