//! Context menu actions of the details view and what happens when a row takes the focus.

use std::path::Path;

use super::{ExplorerMode, FileTableViewer};
use crate::Thumbnail;
use crate::command::ActionId;
use crate::command::actions::{Facts, SPECS, availability, spec_of};
use crate::icons::Icon;

/// An entry of the row context menu; [`MenuAction::id`] keeps the ids of the previous table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MenuAction {
    GenerateDescription,
    RegenerateDescription,
    GenerateClipEmbeddings,
    RegenerateClipEmbedding,
    AddSelectionToGroup,
    RemoveSelectionFromGroup,
    OpenSelectionInNewTab,
    ExcludeDirsFromScan,
    FindSimilar,
    RefineSelection,
    AttachToChatWindow,
    Open,
    Reveal,
    CopyPath,
    AddToLibrary,
    RemoveFromLibrary,
    TagSelection,
    CategorySelection,
    CopyTo,
    MoveTo,
}

impl MenuAction {
    pub const ALL: [MenuAction; 20] = [
        MenuAction::GenerateDescription,
        MenuAction::RegenerateDescription,
        MenuAction::GenerateClipEmbeddings,
        MenuAction::RegenerateClipEmbedding,
        MenuAction::AddSelectionToGroup,
        MenuAction::RemoveSelectionFromGroup,
        MenuAction::OpenSelectionInNewTab,
        MenuAction::ExcludeDirsFromScan,
        MenuAction::FindSimilar,
        MenuAction::RefineSelection,
        MenuAction::AttachToChatWindow,
        MenuAction::Open,
        MenuAction::Reveal,
        MenuAction::CopyPath,
        MenuAction::AddToLibrary,
        MenuAction::RemoveFromLibrary,
        MenuAction::TagSelection,
        MenuAction::CategorySelection,
        MenuAction::CopyTo,
        MenuAction::MoveTo,
    ];

    /// The registry action this item runs.
    pub fn action_id(self) -> ActionId {
        match self {
            MenuAction::GenerateDescription => ActionId::Describe,
            MenuAction::RegenerateDescription => ActionId::RegenerateDescription,
            MenuAction::GenerateClipEmbeddings => ActionId::GenerateClip,
            MenuAction::RegenerateClipEmbedding => ActionId::RegenerateClip,
            MenuAction::AddSelectionToGroup => ActionId::AddToCollection,
            MenuAction::RemoveSelectionFromGroup => ActionId::RemoveFromCollection,
            MenuAction::OpenSelectionInNewTab => ActionId::OpenInNewTab,
            MenuAction::ExcludeDirsFromScan => ActionId::ExcludeFolder,
            MenuAction::FindSimilar => ActionId::FindSimilar,
            MenuAction::RefineSelection => ActionId::Refine,
            MenuAction::AttachToChatWindow => ActionId::AttachToAssistant,
            MenuAction::Open => ActionId::Open,
            MenuAction::Reveal => ActionId::Reveal,
            MenuAction::CopyPath => ActionId::CopyPath,
            MenuAction::AddToLibrary => ActionId::AddToLibrary,
            MenuAction::RemoveFromLibrary => ActionId::RemoveFromLibrary,
            MenuAction::TagSelection => ActionId::Tag,
            MenuAction::CategorySelection => ActionId::Category,
            MenuAction::CopyTo => ActionId::CopyTo,
            MenuAction::MoveTo => ActionId::MoveTo,
        }
    }

    /// The item that runs a registry action.
    pub fn from_action(id: ActionId) -> MenuAction {
        MenuAction::ALL.into_iter().find(|m| m.action_id() == id).expect("every action has a menu item")
    }

    pub fn id(self) -> &'static str {
        spec_of(self.action_id()).menu_id
    }

    pub fn from_id(id: &str) -> Option<MenuAction> {
        MenuAction::ALL.into_iter().find(|a| a.id() == id)
    }

    pub fn label(self) -> &'static str {
        spec_of(self.action_id()).label
    }

    pub fn icon(self) -> Icon {
        spec_of(self.action_id()).icon
    }
}

/// What the selection under a context menu contains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MenuTargets {
    pub rows: usize,
    pub dirs: usize,
}

/// Entries of the context menu in registry order, each with whether it can run on `targets`.
pub fn menu_entries(targets: MenuTargets) -> Vec<(MenuAction, bool)> {
    menu_entries_with(targets, super::super::action_bar::db_ready(), super::super::action_bar::ai_ready())
}

/// [`menu_entries`] with the library and the vision model given.
pub fn menu_entries_with(targets: MenuTargets, db_ready: bool, ai_ready: bool) -> Vec<(MenuAction, bool)> {
    let facts = Facts::from_counts(targets.rows, targets.dirs, db_ready, ai_ready);
    SPECS.iter().map(|s| (MenuAction::from_action(s.id), availability(s.id, &facts).is_ok())).collect()
}

/// Asks for a folder and then copies or moves `paths` into it as a tracked task.
pub fn pick_destination(ctx: &egui::Context, moving: bool, paths: Vec<std::path::PathBuf>) {
    let title = if moving { "Choose destination for move" } else { "Choose backup destination" };
    crate::ui::dialogs::pick_folder(ctx, title, move |dir| {
        if moving {
            crate::ui::file_table::FileExplorer::backup_move_files(paths, dir);
        } else {
            crate::ui::file_table::FileExplorer::backup_copy_files(paths, dir);
        }
    });
}

fn is_virtual(path: &str) -> bool {
    path.starts_with("zip://") || path.starts_with("tar://") || path.starts_with("7z://")
}

fn is_image_path(path: &str) -> bool {
    Path::new(path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).is_some_and(|ext| crate::is_image(ext.as_str()))
}

impl FileTableViewer {
    /// Runs `action` on the selected `rows`, which arrive in view order.
    pub fn run_action(&mut self, action: MenuAction, rows: &[&Thumbnail]) {
        match action {
            MenuAction::AttachToChatWindow => {
                let paths: Vec<String> = rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect();
                if !paths.is_empty() {
                    crate::ui::assistant::request_attach_to_chat(paths);
                }
            }
            MenuAction::OpenSelectionInNewTab => {
                let Some(first) = rows.first() else { return };
                let count = rows.len();
                let title = if count == 1 { format!("Selected: {}", first.filename) } else { format!("Selected ({}) - {}", count, first.filename) };
                let rows: Vec<Thumbnail> = rows.iter().map(|r| (*r).clone()).collect();
                crate::command::send(crate::command::AppCommand::OpenRows(crate::command::RowsTab { title, rows, ..Default::default() }));
            }
            MenuAction::GenerateDescription => self.describe(rows, false),
            MenuAction::RegenerateDescription => self.describe(rows, true),
            MenuAction::GenerateClipEmbeddings => {
                for row in rows {
                    if is_virtual(&row.path) {
                        continue;
                    }
                    self.spawn_clip_generation(row);
                }
            }
            MenuAction::RegenerateClipEmbedding => {
                for row in rows.iter().filter(|r| r.file_type != "<DIR>") {
                    self.spawn_clip_generation(row);
                }
            }
            MenuAction::AddSelectionToGroup => crate::ui::file_table::menus::logical_groups::run_group_action(crate::ui::file_table::menus::logical_groups::GroupAction::Add, rows),
            MenuAction::RemoveSelectionFromGroup => crate::ui::file_table::menus::logical_groups::run_group_action(crate::ui::file_table::menus::logical_groups::GroupAction::Remove, rows),
            MenuAction::ExcludeDirsFromScan => {
                let mut add: Vec<String> = rows.iter().filter(|r| r.file_type == "<DIR>").map(|r| r.path.clone()).filter(|p| !p.trim().is_empty()).collect();
                if add.is_empty() {
                    return;
                }
                crate::settings::update("table:exclude-dirs", |s| {
                    let dirs = s.excluded_dirs.get_or_insert_with(Vec::new);
                    for p in add.drain(..) {
                        if !dirs.contains(&p) {
                            dirs.push(p);
                        }
                    }
                });
                self.sync_settings();
                log::info!("Added selected directories to excluded_dirs for recursive scans");
            }
            MenuAction::FindSimilar => {
                let Some(first) = rows.first() else { return };
                self.find_similar(first.path.clone());
            }
            MenuAction::RefineSelection => {
                let paths: Vec<String> = rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect();
                if !paths.is_empty() {
                    crate::ui::refine::request_refine_for_paths(paths);
                }
            }
            MenuAction::Open | MenuAction::Reveal | MenuAction::CopyPath | MenuAction::AddToLibrary | MenuAction::RemoveFromLibrary | MenuAction::TagSelection | MenuAction::CategorySelection | MenuAction::CopyTo | MenuAction::MoveTo => {
                log::warn!("[menu] {} runs through the action executor, not the viewer", action.id());
            }
        }
    }

    fn describe(&mut self, rows: &[&Thumbnail], regenerate: bool) {
        let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
        let prompt = self.ui_settings.ai_prompt_template.clone();
        for row in rows {
            if !regenerate && is_virtual(&row.path) {
                continue;
            }
            if !regenerate && self.bulk_cancel_requested {
                break;
            }
            if row.file_type == "<DIR>" || !is_image_path(&row.path) {
                continue;
            }
            if !regenerate && !self.ui_settings.overwrite_descriptions && (row.caption.is_some() || row.description.is_some()) {
                continue;
            }
            let path_str = row.path.clone();
            let path_for_run = path_str.clone();
            let tx_updates = self.ai_update_tx.clone();
            let prompt_clone = prompt.clone();
            let eng = engine.clone();
            tokio::spawn(async move {
                eng.stream_vision_description(Path::new(&path_for_run), &prompt_clone, move |interim, final_opt| {
                    if let Some(vd) = final_opt {
                        let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Final {
                            path: path_str.clone(),
                            description: vd.description.clone(),
                            caption: Some(vd.caption.clone()),
                            category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                            tags: vd.tags.clone(),
                        });
                    } else {
                        let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Interim { path: path_str.clone(), text: interim.to_string() });
                    }
                })
                .await;
            });
        }
    }

    fn spawn_clip_generation(&self, row: &Thumbnail) {
        let path = row.path.clone();
        let thumb = row.clone();
        let tx = self.clip_embedding_tx.clone();
        tokio::spawn(async move {
            let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
            let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await?;
            let _ = tx.try_send(thumb.get_embedding().await?);
            log::info!("[CLIP] Manual per-item generation: added {added} for {path}");
            Ok::<(), anyhow::Error>(())
        });
    }

    /// Shows the files most like `origin` in the tab, ordered by CLIP similarity.
    pub fn find_similar(&self, origin: String) {
        crate::command::send(crate::command::AppCommand::OpenScope {
            scope: crate::command::Scope::Similar(origin),
            target: crate::command::Target::Current,
            title: None,
            recursive: false,
        });
    }

    /// Side effects of a row taking the focus: archive thumbnails on demand and a CLIP presence check.
    pub fn on_row_focused(&self, row: &Thumbnail) {
        if row.path.is_empty() || row.file_type == "<DIR>" {
            return;
        }
        if self.mode == ExplorerMode::FileSystem && is_virtual(&row.path) && !self.thumbs.has_thumb(&row.path) {
            self.spawn_archive_thumbnail(row);
        }
        if !is_virtual(&row.path) {
            let tx = self.clip_embedding_tx.clone();
            let thumb = row.clone();
            tokio::spawn(async move {
                let embedding = thumb.get_embedding().await.unwrap_or_default();
                let _ = tx.try_send(embedding);
            });
        }
    }

    fn spawn_archive_thumbnail(&self, row: &Thumbnail) {
        let path = row.path.clone();
        let thumbnail_tx = self.thumbnail_tx.clone();
        let row_clone = row.clone();
        let archive_passwords = self.archive_passwords.clone();
        tokio::spawn(async move {
            let Some(scheme_end) = path.find("://") else { return };
            let Some(internal_start) = path.find("!/") else { return };
            let scheme = &path[..scheme_end];
            let archive_path = &path[scheme_end + 3..internal_start];
            let internal_path = &path[internal_start + 2..];
            let filename = Path::new(internal_path).file_name().and_then(|n| n.to_str()).unwrap_or(internal_path);
            let internal_dir = Path::new(internal_path).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
            let password = archive_passwords.get(archive_path).cloned();
            match crate::utilities::archive::extract_and_generate_thumbnail_blocking(scheme.to_string(), archive_path.to_string(), internal_dir, filename.to_string(), password).await {
                Ok(Some(thumb_b64)) => {
                    let _ = thumbnail_tx.try_send(Thumbnail { thumbnail_b64: Some(thumb_b64), ..row_clone });
                }
                Ok(None) => log::warn!("No thumbnail generated for archive file: {}", path),
                Err(err) => {
                    let es = err.to_string();
                    if es.contains("PasswordRequired") {
                        let mut t = Thumbnail::default();
                        t.path = archive_path.to_string();
                        t.file_type = "<PASSWORD_REQUIRED>".to_string();
                        let _ = thumbnail_tx.try_send(t);
                    } else {
                        log::warn!("Failed to generate thumbnail for archive file: {}: {}", path, es);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_ids_keep_the_ids_of_the_previous_table_first() {
        let ids: Vec<&str> = MenuAction::ALL.iter().take(11).map(|a| a.id()).collect();
        assert_eq!(
            ids,
            [
                "generate_description",
                "regenerate_description",
                "generate_clip_embeddings",
                "regenerate_clip_embedding",
                "add_selection_to_group",
                "remove_selection_from_group",
                "open_selection_in_new_tab",
                "exclude_dirs_from_scan",
                "find_similar",
                "refine_selection",
                "attach_to_chat_window",
            ]
        );
        for a in MenuAction::ALL {
            assert_eq!(MenuAction::from_id(a.id()), Some(a));
            assert_eq!(MenuAction::from_action(a.action_id()), a);
            assert!(!a.label().is_empty());
        }
        assert_eq!(MenuAction::from_id("nope"), None);
    }

    #[test]
    fn every_registry_action_has_one_menu_item() {
        assert_eq!(MenuAction::ALL.len(), SPECS.len());
        for spec in &SPECS {
            assert_eq!(MenuAction::from_action(spec.id).label(), spec.label);
            assert_eq!(MenuAction::from_action(spec.id).id(), spec.menu_id);
        }
    }

    #[test]
    fn labels_contain_no_special_glyphs() {
        for a in MenuAction::ALL {
            assert!(a.label().chars().all(|c| c.is_ascii()), "{}", a.label());
        }
    }

    #[test]
    fn every_entry_is_offered_in_every_scope_in_registry_order() {
        let entries = menu_entries_with(MenuTargets { rows: 2, dirs: 0 }, true, true);
        assert_eq!(entries.len(), SPECS.len());
        let order: Vec<ActionId> = entries.iter().map(|(a, _)| a.action_id()).collect();
        assert_eq!(order, SPECS.iter().map(|s| s.id).collect::<Vec<_>>());
        assert!(entries.iter().any(|(a, _)| *a == MenuAction::AttachToChatWindow));
    }

    #[test]
    fn exclude_directories_needs_a_selected_folder() {
        let enabled = |t: MenuTargets| menu_entries_with(t, true, true).into_iter().find(|(a, _)| *a == MenuAction::ExcludeDirsFromScan).unwrap().1;
        assert!(!enabled(MenuTargets { rows: 3, dirs: 0 }));
        assert!(enabled(MenuTargets { rows: 3, dirs: 1 }));
    }

    #[test]
    fn row_actions_need_at_least_one_row() {
        let none = menu_entries_with(MenuTargets::default(), true, true);
        for (a, enabled) in none {
            assert!(!enabled, "{a:?} needs rows");
        }
        assert!(menu_entries_with(MenuTargets { rows: 1, dirs: 0 }, true, true).iter().filter(|(a, _)| *a != MenuAction::ExcludeDirsFromScan).all(|(_, e)| *e));
    }

    #[test]
    fn the_menu_and_the_registry_agree_on_what_can_run() {
        for (rows, dirs) in [(0, 0), (1, 0), (3, 1), (2, 2)] {
            for (db, ai) in [(true, true), (false, true), (true, false)] {
                let facts = Facts::from_counts(rows, dirs, db, ai);
                for (item, enabled) in menu_entries_with(MenuTargets { rows, dirs }, db, ai) {
                    assert_eq!(enabled, availability(item.action_id(), &facts).is_ok(), "{item:?} on {rows} rows, {dirs} folders, db {db}, ai {ai}");
                }
            }
        }
    }
}
