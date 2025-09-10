use humansize::{format_size, DECIMAL};
use crate::{list_drive_infos};
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
                                                self.viewer.mode = super::table::ExplorerMode::Database;
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
                                    .ui(ui).on_hover_text(format!("{free} free of {total} — Click: open recursive (Shift: recursive)"));

                                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                    let middle = response.middle_clicked();
                                    if response.clicked() || middle {
                                        let title = format!("Drive: {}", drive.label);
                                        let path = path.to_string_lossy().to_string();
                                        let recursive = ui.input(|i| i.modifiers.shift);
                                        let background = ctrl || middle;
                                        crate::app::OPEN_TAB_REQUESTS
                                            .lock()
                                            .unwrap()
                                            .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
                                    }
                                }
                            });
                        });       
                        // WSL (Windows Subsystem for Linux) distros and mounts
                        CollapsingHeader::new(RichText::new("WSL (Linux)"))
                        .default_open(false)
                        .show(ui, |ui| {
                            #[cfg(windows)]
                            {
                                ui.horizontal(|ui| {
                                    ui.label("WSL Distros:");
                                    if ui.button("🔄").on_hover_text("Refresh WSL data").clicked() {
                                        self.refresh_wsl_cache();
                                    }
                                });
                                
                                let distros = self.get_cached_wsl_distros();
                                if distros.is_empty() {
                                    ui.label(RichText::new("No WSL distros detected (check `wsl -l -q`) ").weak());
                                } else {
                                    for d in distros.iter().filter(|d| !d.is_empty()) {
                                        ui.separator();
                                        ui.label(RichText::new(format!("Distro: {}", d)).strong());
                                        for access in crate::utilities::explorer::wsl_dynamic_mounts(d) {
                                            let resp = Button::new(&access.label)
                                            .right_text(RichText::new(&access.icon).color(ui.style().visuals.error_fg_color))
                                            .ui(ui)
                                            .on_hover_text("Click: open recursive in new tab (Shift: recursive)");

                                            let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                            let middle = resp.middle_clicked();
                                            if resp.clicked() || middle {
                                                let title = access.label.clone();
                                                let path = access.path.to_string_lossy().to_string();
                                                let recursive = ui.input(|i| i.modifiers.shift);
                                                let background = ctrl || middle;
                                                crate::app::OPEN_TAB_REQUESTS
                                                    .lock()
                                                    .unwrap()
                                                    .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
                                            }
                                        }
                                    }
                                }
                                
                                // Physical Drives for WSL mounting
                                ui.separator();
                                ui.label(RichText::new("Physical Drives (WSL Mount)").strong());
                                let drives = self.get_cached_physical_drives();
                                if drives.is_empty() {
                                    ui.label(RichText::new("No physical drives detected").weak());
                                } else {
                                    ScrollArea::vertical().max_height(200.).show(ui, |ui| {
                                        for drive in drives {
                                            ui.separator();
                                            ui.horizontal(|ui| {
                                                ui.label(RichText::new(&drive.model).strong());
                                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                                    let size_gb = drive.size as f64 / (1024.0 * 1024.0 * 1024.0);
                                                    ui.label(format!("{:.1} GB", size_gb));
                                                });
                                            });
                                            ui.label(format!("Device: {} ({} partitions)", drive.device_id, drive.partitions));
                                            
                                            // Show partition mount buttons
                                            ui.horizontal(|ui| {
                                                for partition in 0..drive.partitions {
                                                    let mount_btn = Button::new(format!("Mount P{}", partition))
                                                        .small()
                                                        .ui(ui)
                                                        .on_hover_text(format!("Mount partition {} of {}", partition, drive.device_id));
                                                    
                                                    if mount_btn.clicked() {
                                                        let device_id = drive.device_id.clone();
                                                        tokio::spawn(async move {
                                                            match crate::utilities::explorer::mount_wsl_drive(&device_id, partition) {
                                                                Ok(msg) => {
                                                                    log::info!("WSL mount successful: {}", msg);
                                                                }
                                                                Err(e) => {
                                                                    log::error!("WSL mount failed: {}", e);
                                                                }
                                                            }
                                                        });
                                                    }
                                                }
                                                
                                                // Unmount button
                                                let unmount_btn = Button::new("Unmount")
                                                    .small()
                                                    .ui(ui)
                                                    .on_hover_text(format!("Unmount all partitions of {}", drive.device_id));
                                                
                                                if unmount_btn.clicked() {
                                                    let device_id = drive.device_id.clone();
                                                    tokio::spawn(async move {
                                                        match crate::utilities::explorer::unmount_wsl_drive(&device_id) {
                                                            Ok(msg) => {
                                                                log::info!("WSL unmount successful: {}", msg);
                                                            }
                                                            Err(e) => {
                                                                log::error!("WSL unmount failed: {}", e);
                                                            }
                                                        }
                                                    });
                                                }
                                            });
                                        }
                                    });
                                }
                            }
                            #[cfg(not(windows))]
                            {
                                ui.label(RichText::new("WSL is only available on Windows").weak());
                            }
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
                                    for p in settings.unwrap_or_default().recent_paths.iter() {
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
                    });
                });
            });
    }
}