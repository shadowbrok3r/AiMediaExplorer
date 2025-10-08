use eframe::egui::*;
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};

use crate::ui::file_table::table::ExplorerMode;

impl crate::ui::file_table::FileExplorer {
    pub fn selection_menu(&mut self, ui: &mut Ui) {
        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        // Selection operations menu
        MenuButton::new(format!("Select ({})", self.selection_count()))
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
        .ui(ui, |ui| {
            ui.set_width(350.);
            ui.vertical_centered(|ui| 
                ui.heading(RichText::new("Backup Files").underline().color(ui.style().visuals.error_fg_color))
            );

            let selected_dirs: Vec<String> = self
                .table
                .iter()
                .filter(|r| r.file_type == "<DIR>" && self.viewer.selected.contains(&r.path))
                .map(|r| r.path.clone())
                .collect();

            let all_paths: Vec<String> = self
                .table
                .iter()
                .map(|r| r.path.clone())
                .collect();
            
            let selected_files: Vec<String> = self
                .table
                .iter()
                .filter(|r| r.file_type != "<DIR>" && self.viewer.selected.contains(&r.path))
                .map(|r| r.path.clone())
                .collect();

            let selected_to_copy = if selected_files.is_empty() {
                all_paths
            } else {
                selected_files
            };

            let len = selected_to_copy.len();
            let count = selected_dirs.len();

            ui.horizontal(|ui| {
                let hover_txt = "Operates on all non-directory rows currently visible in the table.";
                if ui.button(format!("Copy {len} items...")).on_hover_text(hover_txt).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().set_title("Choose backup destination").pick_folder() {
                        self.backup_copy_visible_to_dir(dir);
                    }
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button(format!("Move {len} items...")).on_hover_text(hover_txt).clicked() {
                        if let Some(dir) = rfd::FileDialog::new().set_title("Choose destination for move").pick_folder() {
                            self.backup_move_visible_to_dir(dir);
                        }
                    }
                });
            });

            ui.separator();

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
        
            if self.viewer.mode == ExplorerMode::Database {
                ui.horizontal(|ui| {
                    if ui.button(RichText::new("Attach to chat window").color(ui.style().visuals.strong_text_color())).clicked() {
                        // Send selected file paths to chat window as attachments
                        let selected_files: Vec<String> = self
                            .table
                            .iter()
                            .filter(|r| r.file_type != "<DIR>" && self.viewer.selected.contains(&r.path))
                            .map(|r| r.path.clone())
                            .collect();
                        crate::ui::assistant::request_attach_to_chat(selected_files);
                        ui.close();
                    }
                    if ui.button(RichText::new(format!("Delete {} From DB", self.selection_count())).color(ui.style().visuals.error_fg_color))
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
                                let paths: Vec<surrealdb::RecordId> = self
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
                    if ui.button(RichText::new("Refine selection").color(ui.style().visuals.warn_fg_color)).on_hover_text("Send current selection to AI Refinements tab").clicked() {
                        let paths: Vec<String> = self
                            .table
                            .iter()
                            .filter(|r| r.file_type != "<DIR>" && self.viewer.selected.contains(&r.path))
                            .map(|r| r.path.clone())
                            .collect();
                        crate::ui::refine::request_refine_for_paths(paths);
                        ui.close();
                    }
                });
            }
        });
    }
}