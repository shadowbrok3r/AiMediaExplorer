use humansize::{format_size, DECIMAL};
use crate::{list_drive_infos};
use std::path::PathBuf;
use eframe::egui::*;

impl super::FileExplorer {
    /// Rich "Quick Access" / Options panel consolidating actions previously in the gear menu.
    pub fn quick_access_pane(&mut self, ui: &mut Ui) {
        SidePanel::left("MainPageLeftPanel")
            .max_width(300.)
            .min_width(260.)
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
                        .default_open(false)
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
                                                self.table_index.clear();
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
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.vertical_centered_justified(|ui| {
                                for access in crate::quick_access().iter() {
                                    let resp = Button::new(&access.label).wrap_mode(TextWrapMode::Truncate)
                                        .right_text(RichText::new(&access.icon).color(ui.style().visuals.error_fg_color))
                                        .ui(ui)
                                        .on_hover_text("Click: open recursive in new tab (hold Shift for recursive)");
                                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                    let middle = resp.middle_clicked();
                                    if resp.clicked() || middle {
                                        let title = format!("{}", access.label);
                                        let path = access.path.to_string_lossy().to_string();
                                        let recursive = ui.input(|i| i.modifiers.shift);
                                        let background = ctrl || middle; // ctrl/middle -> background
                                        crate::app::OPEN_TAB_REQUESTS
                                            .lock()
                                            .unwrap()
                                            .push(crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background });
                                    }
                                }
                                ui.separator();
                                ui.heading("Recent Paths");
                                ui.separator();
                                // Load settings snapshot (cached) and render recent paths
                                for p in self.viewer.ui_settings.recent_paths.iter() {
                                    let resp = Button::new(p)
                                    .wrap_mode(TextWrapMode::Truncate)
                                    .right_text(RichText::new("🗀").color(ui.style().visuals.error_fg_color))
                                    .ui(ui)
                                    .on_hover_text("Click: open recursive in new tab (Shift: recursive)");

                                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                                    let middle = resp.middle_clicked();
                                    if resp.clicked() || middle {
                                        let title = format!("Recent: {}", p);
                                        let path = p.clone();
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
                        CollapsingHeader::new(RichText::new("Drives").strong())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.vertical_centered_justified(|ui| {
                                for drive in list_drive_infos() {
                                    let root = drive.root.clone();
                                    let display = format!("{} - {}", drive.drive_type, drive.root);
                                    let path = PathBuf::from(root.clone());
                                    let free = format_size(drive.free, DECIMAL);
                                    let total = format_size(drive.total, DECIMAL);

                                    let response = Button::new(display)
                                    .wrap_mode(TextWrapMode::Truncate)
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
                        #[cfg(windows)]
                        CollapsingHeader::new(RichText::new("WSL (Linux)"))
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("WSL Distros");
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button("🔄").on_hover_text("Refresh WSL data").clicked() {
                                        self.refresh_wsl_cache();
                                    }
                                });
                            });
                            
                            if self.cached_wsl_distros.is_none() {
                                ui.label(RichText::new("No WSL distros detected (check `wsl -l -q`) ").weak());
                            } else if let Some(distros) = self.cached_wsl_distros.clone() {
                                for d in distros.into_iter().filter(|d| !d.is_empty()) {
                                    for access in crate::utilities::explorer::wsl_dynamic_mounts(&d) {
                                        ui.horizontal(|ui| {
                                            let resp = Button::new(&access.label)
                                                .wrap_mode(TextWrapMode::Truncate)
                                                .right_text(RichText::new(&access.icon).color(ui.style().visuals.error_fg_color))
                                                .ui(ui)
                                                .on_hover_text("Click: open in new tab (Shift: recursive; Ctrl/Middle: background)");

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

                                            // Jump current table to this path
                                            if Button::new("Go").small().ui(ui).on_hover_text("Jump current tab to this path").clicked() {
                                                let path = access.path.to_string_lossy().to_string();
                                                self.viewer.mode = super::table::ExplorerMode::FileSystem;
                                                self.set_path(path);
                                            }
                                        });
                                    }
                                }
                            }
                            
                            // Physical Drives for WSL mounting
                            ui.separator();
                            ui.label(RichText::new("Physical Drives (WSL Mount)").strong());
                            if self.cached_physical_drives.is_none() {
                                ui.label(RichText::new("No physical drives detected").weak());
                            } else if let Some(drives) = &self.cached_physical_drives {
                                ScrollArea::both().max_height(400.).max_width(300.).show(ui, |ui| {
                                    for drive in drives {
                                        ui.separator();
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new(&drive.model).strong());
                                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                                let size_gb = drive.size as f64 / (1024.0 * 1024.0 * 1024.0);
                                                ui.label(format!("{:.1} GB", size_gb));
                                            });
                                        });
                                        ui.label(format!("{} ({} parts)", drive.device_id, drive.partitions));
                                        
                                        // Show partition mount buttons
                                        ui.horizontal(|ui| {
                                            for partition in 0..drive.partitions {
                                                let mount_btn = Button::new(format!("Mount P{}", partition))
                                                .wrap_mode(TextWrapMode::Truncate)
                                                .small()
                                                .ui(ui)
                                                .on_hover_text(format!("Mount partition {} of {}", partition, drive.device_id));
                                                
                                                if mount_btn.clicked() {
                                                    let device_id = drive.device_id.clone();
                                                    // Capture a sender to request UI refresh after mount
                                                    let refresh = self.scan_tx.clone();
                                                    tokio::spawn(async move {
                                                        match crate::utilities::explorer::mount_wsl_drive(&device_id, partition) {
                                                            Ok(msg) => {
                                                                log::info!("WSL mount successful: {}", msg);
                                                                // Hint: after mounting, drives should appear under \\wsl.localhost\\<Distro>\\mnt\\wsl
                                                                let _ = refresh; // placeholder to indicate potential refresh hook
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
                                                // Capture a sender to request UI refresh after unmount
                                                let refresh = self.scan_tx.clone();
                                                tokio::spawn(async move {
                                                    match crate::utilities::explorer::unmount_wsl_drive(&device_id) {
                                                        Ok(msg) => {
                                                            log::info!("WSL unmount successful: {}", msg);
                                                            let _ = refresh; // placeholder
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
                        });

                        // Performance tweaks (moved from Scan menu)
                        CollapsingHeader::new(RichText::new("Performance Tweaks").strong())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("FoundBatch size:");
                                egui::DragValue::new(&mut self.batch_size).range(16..=4096).ui(ui);
                                if ui.button("Apply").on_hover_text("Save and use new batch size for next scans").clicked() {
                                    self.viewer.ui_settings.scan_found_batch_max = Some(self.batch_size);
                                    crate::database::settings::save_settings(&self.viewer.ui_settings);
                                }
                            });
                        });

                        // Recent Scans
                        CollapsingHeader::new(RichText::new("Recent Scans").strong())
                        .default_open(false)
                        .show(ui, |ui| {
                            if ui.button("Refresh").clicked() {
                                let tx = self.scan_tx.clone();
                                tokio::spawn(async move {
                                    let _ = tx; // placeholder; refresh drives UI indirectly
                                });
                            }
                            ui.add_space(4.0);
                            // Fetch recent on first open (lazy)
                            // For simplicity, fetch each frame but small limit; a better approach caches locally
                            let mut recent: Vec<crate::database::CachedScan> = Vec::new();
                            let handle = std::thread::spawn(|| {});
                            drop(handle);
                            // Blocking inside UI is not ideal, but as a first pass we use a channel
                            let (txs, rxs) = crossbeam::channel::bounded(1);
                            tokio::spawn(async move {
                                let rows = crate::database::CachedScan::list_recent(10).await.unwrap_or_default();
                                let _ = txs.send(rows);
                            });
                            if let Ok(rows) = rxs.recv_timeout(std::time::Duration::from_millis(50)) { recent = rows; }
                            if recent.is_empty() {
                                ui.label(RichText::new("No recent scans").weak());
                            } else {
                                for sc in recent.into_iter() {
                                    let title = sc.title.clone().unwrap_or_else(|| sc.root.clone());
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{}", title));
                                        if ui.button("Open").clicked() {
                                            // Load paths and open in a new tab
                                            tokio::spawn(async move {
                                                let mut offset = 0usize;
                                                let limit = 5000usize;
                                                let mut rows: Vec<crate::database::Thumbnail> = Vec::new();
                                                loop {
                                                    let paths = crate::database::CachedScanItem::list_paths(&sc.id, offset, limit).await.unwrap_or_default();
                                                    if paths.is_empty() { break; }
                                                    offset += paths.len();
                                                    // Hydrate rows via DB if present, else minimal rows from disk
                                                    if let Ok(ths) = crate::database::Thumbnail::find_thumbs_from_paths(paths.clone()).await {
                                                        rows.extend(ths);
                                                    } else {
                                                        for p in paths.into_iter() {
                                                            let mut r = crate::Thumbnail::default();
                                                            r.path = p.clone();
                                                            r.filename = std::path::Path::new(&p).file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
                                                            rows.push(r);
                                                        }
                                                    }
                                                }
                                                crate::app::OPEN_TAB_REQUESTS
                                                    .lock()
                                                    .unwrap()
                                                    .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
                                            });
                                        }
                                    });
                                }
                            }
                        });

                        // Compact scan performance panel (visible when we have data)
                        if self.perf_last_total.is_some() || !self.perf_batches.is_empty() {
                            CollapsingHeader::new("Scan performance")
                            .default_open(false)
                            .show(ui, |ui| {
                                if let Some(total) = self.perf_last_total {
                                    ui.label(format!("Last scan total: {:.3}s", total.as_secs_f32()));
                                }
                                // Toggle per-1k display
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut self.perf_show_per_1k, "Show per-1k rates");
                                });

                                if !self.perf_batches.is_empty() {
                                    let show = self.perf_batches.len().min(10);
                                    ui.label("Recent batch timings (len, recv_gap, ui_time) and optional per-1k:");
                                    // Compute averages over visible window
                                    let mut sum_recv = 0.0f32;
                                    let mut sum_ui = 0.0f32;
                                    let mut sum_len = 0usize;
                                    for (len, recv, ui_t) in self.perf_batches.iter().rev().take(show) {
                                        sum_recv += recv.as_secs_f32();
                                        sum_ui += ui_t.as_secs_f32();
                                        sum_len += *len;
                                        if self.perf_show_per_1k {
                                            let per_1k_recv = if *len > 0 { recv.as_secs_f32() * (1000.0 / *len as f32) } else { 0.0 };
                                            let per_1k_ui = if *len > 0 { ui_t.as_secs_f32() * (1000.0 / *len as f32) } else { 0.0 };
                                            ui.label(format!("  len={} recv={:.3}s ui={:.3}s | recv/1k={:.3}s ui/1k={:.3}s", len, recv.as_secs_f32(), ui_t.as_secs_f32(), per_1k_recv, per_1k_ui));
                                        } else {
                                            ui.label(format!("  len={} recv={:.3}s ui={:.3}s", len, recv.as_secs_f32(), ui_t.as_secs_f32()));
                                        }
                                    }
                                    // Averages over the shown window
                                    if sum_len > 0 {
                                        let avg_recv_per_1k = sum_recv * (1000.0 / sum_len as f32);
                                        let avg_ui_per_1k = sum_ui * (1000.0 / sum_len as f32);
                                        ui.separator();
                                        ui.label(format!("Avg over last {}: recv/1k={:.3}s ui/1k={:.3}s", show, avg_recv_per_1k, avg_ui_per_1k));
                                    }
                                }
                                if ui.button("Clear timings").clicked() {
                                    self.perf_batches.clear();
                                    self.perf_last_total = None;
                                }
                            });
                        }
                    });
                });
            });
    }
}