use std::collections::HashSet;
use std::path::PathBuf;

use eframe::egui::*;
use egui::containers::menu::{MenuButton, MenuConfig};
use egui_toast::ToastKind;
use surrealdb::types::RecordId;

use crate::database::Thumbnail;
use crate::ui::dialogs;

/// Names listed in a confirmation before the rest are summarised as a count.
const NAME_PREVIEW: usize = 5;

/// Splits selected rows into the file rows and the paths of the directory rows; no rows split into nothing.
pub fn split_selection<'a>(selected: &[&'a Thumbnail]) -> (Vec<&'a Thumbnail>, Vec<String>) {
    let files = selected.iter().copied().filter(|r| r.file_type != "<DIR>").collect();
    let dirs = selected.iter().filter(|r| r.file_type == "<DIR>").map(|r| r.path.clone()).collect();
    (files, dirs)
}

/// First `max` names joined, with a trailing "and N more".
pub fn name_preview(names: &[String], max: usize) -> String {
    let shown = names.iter().take(max).cloned().collect::<Vec<_>>().join(", ");
    match names.len().saturating_sub(max) {
        0 => shown,
        more => format!("{shown} and {more} more"),
    }
}

/// Rows a delete would remove; `None` when nothing is selected.
#[derive(Clone, Debug, PartialEq)]
pub struct DeletePlan {
    pub ids: Vec<RecordId>,
    pub paths: Vec<String>,
    pub names: Vec<String>,
}

pub fn plan_delete(rows: &[&Thumbnail]) -> Option<DeletePlan> {
    if rows.is_empty() {
        return None;
    }
    Some(DeletePlan {
        ids: rows.iter().map(|r| r.id.clone()).collect(),
        paths: rows.iter().map(|r| r.path.clone()).collect(),
        names: rows.iter().map(|r| r.filename.clone()).collect(),
    })
}

#[derive(Clone)]
enum Pending {
    Delete(DeletePlan),
    Move { paths: Vec<PathBuf>, names: Vec<String> },
}

impl crate::ui::file_table::FileExplorer {
    /// Text of the Select menu button.
    pub fn selection_label(&self) -> String {
        format!("Select ({})", self.selection_count())
    }

    /// Select menu button plus the confirmation dialogs it raises; `owner` keys per-tab state.
    pub fn selection_menu(&mut self, ui: &mut Ui, owner: &str) {
        let pending_id = Id::new(("ame_selection_pending", owner));
        self.show_pending(ui.ctx(), pending_id);

        MenuButton::new(self.selection_label())
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .ui(ui, |ui| {
            ui.set_width(350.);
            ui.vertical_centered(|ui|
                ui.heading(RichText::new("Backup Files").underline().color(crate::theme::primary(ui)))
            );

            self.sync_view();
            let selected = self.selected_rows();
            let (files, selected_dirs) = split_selection(&selected);
            let file_paths: Vec<String> = files.iter().map(|r| r.path.clone()).collect();
            let file_names: Vec<String> = files.iter().map(|r| r.filename.clone()).collect();
            let delete_plan = plan_delete(&files);
            let n = file_paths.len();
            let has_files = n > 0;
            let hint = if has_files { "Acts on the selected files only" } else { "Select files in the table first" };

            ui.horizontal(|ui| {
                if ui.add_enabled(has_files, Button::new(format!("Copy {n} items..."))).on_hover_text(hint).clicked() {
                    let paths: Vec<PathBuf> = file_paths.iter().map(PathBuf::from).collect();
                    dialogs::pick_folder(ui.ctx(), "Choose backup destination", move |dir| {
                        crate::ui::file_table::FileExplorer::backup_copy_files(paths, dir);
                    });
                    ui.close();
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.add_enabled(has_files, Button::new(format!("Move {n} items..."))).on_hover_text(hint).clicked() {
                        let paths: Vec<PathBuf> = file_paths.iter().map(PathBuf::from).collect();
                        ui.ctx().data_mut(|d| d.insert_temp(pending_id, Pending::Move { paths, names: file_names.clone() }));
                        ui.close();
                    }
                });
            });

            ui.separator();

            if selected_dirs.is_empty() {
                ui.label(RichText::new("No directories selected").weak());
            } else {
                ui.label(format!("Selected directories: {}", selected_dirs.len()));
            }
            ui.separator();
            let exclude = ui.add_enabled(!selected_dirs.is_empty(), Button::new("Exclude selected directories from recursive scans").small());
            if exclude.clicked() {
                crate::settings::update("selection:exclude-dirs", |s| {
                    let dirs = s.excluded_dirs.get_or_insert_with(Vec::new);
                    for p in &selected_dirs {
                        if !dirs.contains(p) { dirs.push(p.clone()); }
                    }
                });
                self.edit_filters(|spec| spec.excluded_dirs.extend(selected_dirs.iter().cloned()));
                ui.close();
            }

            ui.horizontal(|ui| {
                if ui.add_enabled(has_files, Button::new("Attach to chat window")).on_hover_text(hint).clicked() {
                    crate::ui::assistant::request_attach_to_chat(file_paths.clone());
                    ui.close();
                }
                let delete = ui.add_enabled(delete_plan.is_some(), crate::theme::danger_button(format!("Delete {n} From DB")))
                    .on_hover_text("Remove the selected rows from the database, including their CLIP embeddings. Files on disk are NOT affected.");
                if delete.clicked() {
                    if let Some(plan) = delete_plan.clone() {
                        ui.ctx().data_mut(|d| d.insert_temp(pending_id, Pending::Delete(plan)));
                        ui.close();
                    }
                }
                if ui.add_enabled(has_files, Button::new("Refine selection")).on_hover_text("Send the selected files to the AI Refinements tab").clicked() {
                    crate::ui::refine::request_refine_for_paths(file_paths.clone());
                    ui.close();
                }
            });
        });
    }

    /// Draws the pending confirmation, if any, and runs the action once confirmed.
    fn show_pending(&mut self, ctx: &Context, id: Id) {
        let Some(pending) = ctx.data(|d| d.get_temp::<Pending>(id)) else { return };
        let (title, body, names, confirm_label) = match &pending {
            Pending::Delete(plan) => (
                "Delete from database",
                format!("Permanently delete {} records from the database, including their embeddings? Files on disk are not touched.", plan.ids.len()),
                &plan.names,
                format!("Delete {}", plan.ids.len()),
            ),
            Pending::Move { paths, names } => (
                "Move files",
                format!("Move {} files out of their current folders into a folder you choose next?", paths.len()),
                names,
                format!("Choose folder and move {}", paths.len()),
            ),
        };
        let mut decision: Option<bool> = None;
        let gap = crate::theme::metrics(ctx).gap.y;
        Window::new(title)
            .id(id.with("window"))
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_max_width(420.0);
                ui.label(body);
                ui.add_space(gap);
                ui.label(RichText::new(name_preview(names, NAME_PREVIEW)).weak());
                ui.add_space(gap * 2.0);
                ui.horizontal(|ui| {
                    if ui.add(crate::theme::danger_button(confirm_label)).clicked() { decision = Some(true); }
                    if ui.button("Cancel").clicked() { decision = Some(false); }
                });
            });
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            decision = Some(false);
        }
        let Some(confirmed) = decision else { return };
        ctx.data_mut(|d| d.remove::<Pending>(id));
        if !confirmed {
            return;
        }
        match pending {
            Pending::Delete(plan) => self.delete_rows_from_db(plan),
            Pending::Move { paths, .. } => {
                dialogs::pick_folder(ctx, "Choose destination for move", move |dir| {
                    crate::ui::file_table::FileExplorer::backup_move_files(paths, dir);
                });
            }
        }
    }

    fn delete_rows_from_db(&mut self, plan: DeletePlan) {
        let removed: HashSet<String> = plan.paths.iter().cloned().collect();
        self.table.retain(|r| !removed.contains(&r.path));
        self.sync_view();
        let (ids, paths) = (plan.ids, plan.paths);
        tokio::spawn(async move {
            match super::db_delete::delete_rows(ids, paths).await {
                Ok((emb, thumbs)) => {
                    log::info!("Deleted {emb} embeddings and {thumbs} thumbnails.");
                    dialogs::toast(ToastKind::Success, format!("Deleted {thumbs} records and {emb} embeddings"));
                }
                Err(e) => {
                    log::error!("Delete selected failed: {e:?}");
                    dialogs::toast(ToastKind::Error, format!("Delete failed: {e}"));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, kind: &str) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit('/').next().unwrap_or(path).into();
        t.file_type = kind.into();
        t
    }

    #[test]
    fn no_selected_rows_split_into_nothing_and_plan_no_delete() {
        let (files, dirs) = split_selection(&[]);
        assert!(files.is_empty() && dirs.is_empty());
        assert!(plan_delete(&files).is_none());
    }

    #[test]
    fn files_and_directories_are_split() {
        let rows = vec![row("/a/1.jpg", "jpg"), row("/a/sub", "<DIR>"), row("/a/2.jpg", "jpg")];
        let refs: Vec<&Thumbnail> = rows.iter().collect();
        let (files, dirs) = split_selection(&refs);
        let paths: Vec<_> = files.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(paths, vec!["/a/1.jpg", "/a/2.jpg"]);
        assert_eq!(dirs, vec!["/a/sub".to_string()]);
    }

    #[test]
    fn the_delete_plan_carries_the_exact_selection() {
        let rows = vec![row("/a/1.jpg", "jpg"), row("/a/3.jpg", "jpg")];
        let picked: Vec<&Thumbnail> = rows.iter().collect();
        let plan = plan_delete(&picked).expect("two rows selected");
        assert_eq!(plan.paths, vec!["/a/1.jpg".to_string(), "/a/3.jpg".to_string()]);
        assert_eq!(plan.names, vec!["1.jpg".to_string(), "3.jpg".to_string()]);
        assert_eq!(plan.ids.len(), 2);
    }

    #[test]
    fn name_preview_lists_the_first_names_then_a_count() {
        let names: Vec<String> = (1..=8).map(|i| format!("{i}.jpg")).collect();
        assert_eq!(name_preview(&names[..3], 5), "1.jpg, 2.jpg, 3.jpg");
        assert_eq!(name_preview(&names, 5), "1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg and 3 more");
        assert_eq!(name_preview(&[], 5), "");
    }

    #[test]
    fn exactly_the_preview_count_has_no_remainder() {
        let names: Vec<String> = (1..=5).map(|i| format!("{i}.jpg")).collect();
        assert_eq!(name_preview(&names, 5), "1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg");
    }
}
