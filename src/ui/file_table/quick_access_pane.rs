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
                                    let resp = Button::new(&access.label)
                                        .right_text(RichText::new(&access.icon).color(ui.style().visuals.error_fg_color))
                                        .ui(ui)
                                        .on_hover_text("Click: open recursive in new tab (hold Shift for shallow)");
                                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                    let middle = resp.middle_clicked();
                                    if resp.clicked() || middle {
                                        let title = format!("{}", access.label);
                                        let path = access.path.to_string_lossy().to_string();
                                        let shift = ui.input(|i| i.modifiers.shift);
                                        let recursive = !shift; // default recursive, Shift = shallow
                                        let background = ctrl || middle; // ctrl/middle -> background
                                        crate::app::OPEN_TAB_REQUESTS
                                            .lock()
                                            .unwrap()
                                            .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
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
                                    .ui(ui).on_hover_text(format!("{free} free of {total} — Click: open recursive (Shift: shallow)"));

                                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                    let middle = response.middle_clicked();
                                    if response.clicked() || middle {
                                        let title = format!("Drive: {}", drive.label);
                                        let path = path.to_string_lossy().to_string();
                                        let shift = ui.input(|i| i.modifiers.shift);
                                        let recursive = !shift; // default recursive
                                        let background = ctrl || middle;
                                        crate::app::OPEN_TAB_REQUESTS
                                            .lock()
                                            .unwrap()
                                            .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
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
                                if settings.as_ref().map(|s| s.recent_paths.is_empty()).unwrap_or(true) {
                                    ui.label(RichText::new("No recent directories yet").weak());
                                } else {
                                    for p in settings.unwrap().recent_paths.iter() {
                                        let resp = Button::new(p)
                                            .right_text(RichText::new("🗀").color(ui.style().visuals.error_fg_color))
                                            .ui(ui)
                                            .on_hover_text("Click: open recursive in new tab (Shift: shallow)");
                                        let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                        let middle = resp.middle_clicked();
                                        if resp.clicked() || middle {
                                            let title = format!("Recent: {}", p);
                                            let path = p.clone();
                                            let shift = ui.input(|i| i.modifiers.shift);
                                            let recursive = !shift; // default recursive
                                            let background = ctrl || middle;
                                            crate::app::OPEN_TAB_REQUESTS
                                                .lock()
                                                .unwrap()
                                                .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
                                        }
                                    }
                                }
                            });
                        }); 
                        // Always expose Logical Groups to allow selecting an active group even in FileSystem mode
                        if true {
                            CollapsingHeader::new(RichText::new("Logical Groups").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if let Some(name) = &self.active_logical_group_name {
                                        ui.label(format!("Active: {}", name));
                                    } else {
                                        ui.label("Active: (none)");
                                    }
                                    if ui.button("Refresh").clicked() {
                                        let tx = self.logical_groups_tx.clone();
                                        tokio::spawn(async move {
                                            match crate::database::LogicalGroup::list_all().await {
                                                Ok(groups) => { let _ = tx.try_send(groups); },
                                                Err(e) => log::error!("refresh logical groups failed: {e:?}"),
                                            }
                                        });
                                    }
                                });
                                ui.separator();
                                if self.logical_groups.is_empty() {
                                    ui.label(RichText::new("No groups defined yet.").weak());
                                } else {
                                    ScrollArea::vertical().max_height(160.).show(ui, |ui| {
                                        for g in self.logical_groups.clone().into_iter() {
                                            ui.horizontal(|ui| {
                                                let active = self.active_logical_group_name.as_deref() == Some(g.name.as_str());
                                                let lbl = if active { RichText::new(format!("• {}", g.name)).strong() } else { RichText::new(g.name.clone()) };
                                                ui.label(lbl);
                                                if ui.button("Select").on_hover_text("Use this as the active group (no mode switch)").clicked() {
                                                    self.active_logical_group_name = Some(g.name.clone());
                                                }
                                                if ui.button("Open").on_hover_text("Load this group's thumbnails").clicked() {
                                                    self.active_logical_group_name = Some(g.name.clone());
                                                    self.viewer.mode = super::viewer::ExplorerMode::Database;
                                                    self.table.clear();
                                                    self.db_offset = 0;
                                                    self.db_last_batch_len = 0;
                                                    self.db_loading = true;
                                                    self.load_logical_group_by_name(g.name.clone());
                                                }
                                            });
                                        }
                                    });
                                }
                            });
                        }
                    });
                });
            });
    }
}