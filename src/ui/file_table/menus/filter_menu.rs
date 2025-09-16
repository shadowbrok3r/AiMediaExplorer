use chrono::{DateTime, Local, NaiveDate};
use eframe::egui::*;
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};

impl crate::ui::file_table::FileExplorer {
    pub fn filter_menu(&mut self, ui: &mut Ui) {
        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        
        MenuButton::new("🔻")
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::IgnoreClicks).style(style))
        .ui(ui, |ui| {
            ui.set_width(300.);
            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Excluded Terms").underline().color(ui.style().visuals.error_fg_color))
            );
            ui.add_space(5.);
            ui.horizontal(|ui| {
                let resp = TextEdit::singleline(&mut self.excluded_term_input)
                .hint_text("term")
                .desired_width(140.)
                .ui(ui);

                if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
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
                        // Rebuild index after retain
                        self.table_index.clear();
                        for (i, r) in self.table.iter().enumerate() { self.table_index.insert(r.path.clone(), i); }
                    }
                }
                
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Clear All").clicked() { 
                        self.excluded_terms.clear();
                        self.active_filter_group = None;
                        // Note: clearing does not restore previously filtered rows to keep UX consistent with size filters
                    }
                });
            });
            ui.separator();
            ui.add_space(5.);
            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Filter by Date Range").underline().color(ui.style().visuals.error_fg_color))
            );
            ui.add_space(5.);
            ui.horizontal(|ui| {
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

                let after_id =  format!("{after_date} Start date");
                let resp_after = egui_extras::DatePickerButton::new(&mut after_date)
                .id_salt(&after_id)
                .ui(ui);

                if ui.button("✖").on_hover_text("Clear start date").clicked() {
                    self.viewer.ui_settings.filter_modified_after = None;
                    crate::database::settings::save_settings(&self.viewer.ui_settings);
                    self.active_filter_group = None;
                    self.apply_filters_to_current_table();
                }

                ui.label("  ->  ");

                let before_id = format!("{before_date} End date");
                let resp_before = egui_extras::DatePickerButton::new(&mut before_date)
                .id_salt(&before_id)
                .ui(ui);

                if ui.button("✖").on_hover_text("Clear end date").clicked() {
                    self.viewer.ui_settings.filter_modified_before = None;
                    crate::database::settings::save_settings(&self.viewer.ui_settings);
                    self.active_filter_group = None;
                    self.apply_filters_to_current_table();
                }

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
                        // Rebuild index after retain
                        self.table_index.clear();
                        for (i, r) in self.table.iter().enumerate() { self.table_index.insert(r.path.clone(), i); }
                    }
                }
            });

            ui.add_space(5.);
            ui.separator();
            ui.add_space(5.);

            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Size Filters (KB)").underline().color(ui.style().visuals.error_fg_color))
            );

            ui.add_space(5.);
            
            ui.horizontal(|ui| {
                ui.label("Min:");
                let mut min_kb: i64 = self.viewer.ui_settings.db_min_size_bytes
                    .map(|b| (b / 1024) as i64)
                    .unwrap_or(0);
                let mut max_kb: i64 = self.viewer.ui_settings.db_max_size_bytes
                    .map(|b| (b / 1024) as i64)
                    .unwrap_or(0);
                let min_resp = egui::DragValue::new(&mut min_kb).speed(1).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Minimum file size in KiB");
                ui.add_space(20.);
                ui.label("Max:");
                let max_resp = egui::DragValue::new(&mut max_kb).speed(10).range(0..=i64::MAX).suffix(" KB").ui(ui).on_hover_text("Maximum file size in KiB (0 = no max)");
                ui.add_space(5.);
                let changed = min_resp.changed() || max_resp.changed();
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Clear").on_hover_text("Clear size constraints").clicked() {
                        self.min_size_mb_input.clear();
                        self.max_size_mb_input.clear();
                        self.viewer.ui_settings.db_min_size_bytes = None;
                        self.viewer.ui_settings.db_max_size_bytes = None;
                        crate::database::settings::save_settings(&self.viewer.ui_settings);
                        self.active_filter_group = None;
                    }
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
                });
            });

            ui.add_space(5.);
            ui.separator();
            ui.add_space(5.);

            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Extension filters").underline().color(ui.style().visuals.error_fg_color))
            );
            ui.add_space(5.);
            ui.vertical(|ui| {
                let changed_i = ui.checkbox(&mut self.viewer.types_show_images, "Images").changed();
                let changed_v = ui.checkbox(&mut self.viewer.types_show_videos, "Videos").changed();
                let changed_d = ui.checkbox(&mut self.viewer.types_show_dirs, "Folders").changed();
                if changed_i || changed_v || changed_d {
                    // Immediate effect via RowViewer::filter_row; avoid destructive pruning
                    self.active_filter_group = None;
                }

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
            ui.add_space(5.);
            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Save filters as group").underline().color(ui.style().visuals.error_fg_color))
            );
            ui.add_space(5.);
            ui.horizontal(|ui| {
                ui.add_sized([180., 22.], TextEdit::singleline(&mut self.filter_group_name_input).hint_text("Group name"));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
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
            });
            ui.add_space(5.);
            ui.separator();
            ui.add_space(5.);
            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Saved filter groups").underline().color(ui.style().visuals.error_fg_color))
            );
            ui.add_space(5.);
            ui.horizontal(|ui| {
                ui.label("");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("⟲").clicked() {
                        let tx = self.filter_groups_tx.clone();
                        tokio::spawn(async move {
                            match crate::database::list_filter_groups().await {
                                Ok(groups) => { let _ = tx.try_send(groups); },
                                Err(e) => log::error!("list_filter_groups failed: {e:?}"),
                            }
                        });
                    }
                });
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
            
            if !self.last_scan_rows.is_empty() {
                ui.separator();
                ui.add_space(4.0);
                ui.horizontal(|ui| {
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
                });
            }
        });
    }

    pub fn apply_filters_to_current_table(&mut self) {
        use crate::utilities::filtering::FiltersExt;
        // Build a Filters value from UI settings for consistent evaluation
        let mut filters = crate::Filters::default();
        filters.include_images = self.viewer.types_show_images;
        filters.include_videos = self.viewer.types_show_videos;
        // Archives are not shown in DB table rows normally; keep false
        filters.include_archives = false;
        filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
        filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
        filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
        // Date filters from UI
        filters.modified_after = self.viewer.ui_settings.filter_modified_after.clone();
        filters.modified_before = self.viewer.ui_settings.filter_modified_before.clone();
        let include_dirs = self.viewer.types_show_dirs;
        let excluded_terms = self.excluded_terms.clone();
        let have_date_filters = filters.modified_after.is_some() || filters.modified_before.is_some();

        self.table.retain(|r| {
            // Keep directories independent of FiltersExt logic
            if r.file_type == "<DIR>" { return include_dirs; }

            // Excluded terms (path/filename contains any term)
            if !excluded_terms.is_empty() {
                let lp = r.path.to_ascii_lowercase();
                if excluded_terms.iter().any(|t| lp.contains(t)) { return false; }
            }

            let path = std::path::Path::new(&r.path);
            // Quick kind check from extension to mirror scanner behavior
            let is_archive_file = path.extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .map(|e| crate::is_archive(e.as_str()))
                .unwrap_or(false);
            let kind = if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                if crate::is_image(ext.as_str()) { crate::utilities::types::MediaKind::Image }
                else if crate::is_video(ext.as_str()) { crate::utilities::types::MediaKind::Video }
                else if crate::is_archive(ext.as_str()) { crate::utilities::types::MediaKind::Archive }
                else { crate::utilities::types::MediaKind::Other }
            } else { crate::utilities::types::MediaKind::Other };
            if !filters.kind_allowed(&kind, is_archive_file) { return false; }

            // Use FiltersExt to evaluate size and skip_icons heuristics (dates are not applied to DB view here)
            let size_val = if r.size == 0 {
                std::fs::metadata(&r.path).ok().map(|m| m.len()).unwrap_or(0)
            } else { r.size };
            // Use strict UI variant: still allows small non-image files, but filters tiny icon-like images
            if !filters.skip_icons_strict_allows(path, size_val) { return false; }
            if !filters.size_ok(size_val) { return false; }
            // Apply date filter if active (use filesystem metadata; DB may not have timezone-local timestamps)
            if have_date_filters {
                let modified: Option<DateTime<Local>> = std::fs::metadata(&r.path)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|st| DateTime::<Local>::from(st));
                if !filters.date_ok(modified, None, /*recursive*/ false) { return false; }
            }
            true
        });
        // Rebuild the fast path index after retain changes indices
        self.table_index.clear();
        for (i, r) in self.table.iter().enumerate() {
            self.table_index.insert(r.path.clone(), i);
        }
    }

}

