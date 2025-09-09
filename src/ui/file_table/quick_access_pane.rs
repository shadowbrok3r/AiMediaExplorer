use humansize::{format_size, DECIMAL};
use crate::list_drive_infos;
use std::path::PathBuf;
use eframe::egui::*;

impl super::FileExplorer {
    /// Rich "Quick Access" / Options panel consolidating actions previously in the gear menu.
    pub fn quick_access_pane(&mut self, ui: &mut Ui) {
        SidePanel::left("MainPageLeftPanel")
            .max_width(420.)
            .min_width(240.)
            .show_animated_inside(ui, self.open_quick_access, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Quick Access");
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.button(RichText::new("✖").color(ui.style().visuals.error_fg_color)).on_hover_text("Close panel").clicked() { 
                                self.open_quick_access = false; 
                            }
                        });
                    });

                    ui.separator();

                    ScrollArea::vertical().auto_shrink([false;2]).show(ui, |ui| {
                        CollapsingHeader::new(RichText::new("User Directories").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.vertical_centered_justified(|ui| {
                                for access in crate::quick_access().iter() {
                                    if Button::new(&access.label).right_text(RichText::new(&access.icon).color(ui.style().visuals.error_fg_color)).ui(ui).on_hover_text(&access.label).clicked() {
                                        self.set_path(access.path.to_string_lossy());
                                    }
                                }
                            });
                        });
                        CollapsingHeader::new(RichText::new("Drives").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.vertical_centered_justified(|ui| {
                                for drive in list_drive_infos() {
                                    let root = drive.root.clone();
                                    let display = format!("{} - {}", drive.drive_type, drive.root);
                                    let path = PathBuf::from(root.clone());
                                    let free = format_size(drive.free, DECIMAL);
                                    let total = format_size(drive.total, DECIMAL);

                                    let response = Button::new(display)
                                    .right_text(RichText::new(&drive.label).color(ui.style().visuals.error_fg_color))
                                    .ui(ui).on_hover_text(format!("{free} free of {total}"));

                                    if response.clicked() {
                                        self.set_path(path.to_string_lossy());
                                    }
                                }
                            });
                        });       
                        CollapsingHeader::new(RichText::new("Recent Directories").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.vertical_centered_justified(|ui| {
                                // Load settings snapshot (cached) and render recent paths
                                let settings = crate::database::settings::load_settings();
                                if settings.recent_paths.is_empty() {
                                    ui.label(RichText::new("No recent directories yet").weak());
                                } else {
                                    for p in settings.recent_paths.iter() {
                                        if Button::new(p).right_text(RichText::new("🗀").color(ui.style().visuals.error_fg_color)).ui(ui).clicked() {
                                            self.push_history(p.clone());
                                            if self.viewer.mode == super::viewer::ExplorerMode::Database {
                                                self.db_offset = 0;
                                                self.db_last_batch_len = 0;
                                                self.load_database_rows();
                                            } else {
                                                self.populate_current_directory();
                                            }
                                        }
                                    }
                                }
                            });
                        }); 
                    });
                });
            });
    }
}