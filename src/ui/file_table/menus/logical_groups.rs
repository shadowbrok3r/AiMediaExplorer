use eframe::egui::*;
use egui::{containers::menu::{MenuButton, MenuConfig}, style::StyleModifier};

impl crate::ui::file_table::FileExplorer {
    pub fn logical_group_menu(&mut self, ui: &mut Ui) {
        let style = StyleModifier::default();
        style.apply(ui.style_mut());
        let txt = if let Some(name) = &self.active_logical_group_name {
            RichText::new(format!("Logical Groups ({name})")).color(Color32::LIGHT_GREEN)
        } else {
            RichText::new(format!("Logical Groups")).color(Color32::LIGHT_YELLOW)
        };
        
        MenuButton::new(txt)
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone()))
        .ui(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.set_width(470.);
                ui.horizontal(|ui| {
                    if self.active_logical_group_name.is_none() {
                        ui.colored_label(Color32::YELLOW, "No active logical group. Auto save and indexing are gated.");
                    } else {
                        ui.colored_label(Color32::LIGHT_GREEN, format!("Active Group: {}", self.active_logical_group_name.clone().unwrap_or_default()));
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
                ui.add_space(5.);
                ui.vertical_centered(|ui| 
                    ui.heading(RichText::new("Operations").underline().color(ui.style().visuals.error_fg_color))
                );
                ui.add_space(5.);
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
                                            let dst_gid: Option<surrealdb::types::RecordId> = match crate::database::LogicalGroup::get_by_name(&dst).await {
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
                                                let gid_opt: Option<surrealdb::types::RecordId> = match crate::database::LogicalGroup::get_by_name(&target).await {
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
                                            let gid_opt: Option<surrealdb::types::RecordId> = match crate::database::LogicalGroup::get_by_name(&target).await {
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

                ui.add_space(5.);
                ui.vertical_centered(|ui| 
                    ui.heading(RichText::new("Create New Group").underline().color(ui.style().visuals.error_fg_color))
                );
                ui.add_space(5.);
                
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
                                    self.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
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
    }
}