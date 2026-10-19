//! Context menu actions of the details view and what happens when a row takes the focus.

use std::path::Path;

use super::{ExplorerMode, FileTableViewer};
use crate::Thumbnail;
use crate::icons::{Icon, svg};

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
}

impl MenuAction {
    pub const ALL: [MenuAction; 11] = [
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
    ];

    pub fn id(self) -> &'static str {
        match self {
            MenuAction::GenerateDescription => "generate_description",
            MenuAction::RegenerateDescription => "regenerate_description",
            MenuAction::GenerateClipEmbeddings => "generate_clip_embeddings",
            MenuAction::RegenerateClipEmbedding => "regenerate_clip_embedding",
            MenuAction::AddSelectionToGroup => "add_selection_to_group",
            MenuAction::RemoveSelectionFromGroup => "remove_selection_from_group",
            MenuAction::OpenSelectionInNewTab => "open_selection_in_new_tab",
            MenuAction::ExcludeDirsFromScan => "exclude_dirs_from_scan",
            MenuAction::FindSimilar => "find_similar",
            MenuAction::RefineSelection => "refine_selection",
            MenuAction::AttachToChatWindow => "attach_to_chat_window",
        }
    }

    pub fn from_id(id: &str) -> Option<MenuAction> {
        MenuAction::ALL.into_iter().find(|a| a.id() == id)
    }

    pub fn label(self) -> &'static str {
        match self {
            MenuAction::GenerateDescription => "Generate Description",
            MenuAction::RegenerateDescription => "Regenerate Description (overwrite)",
            MenuAction::GenerateClipEmbeddings => "Generate CLIP Embedding",
            MenuAction::RegenerateClipEmbedding => "Regenerate CLIP Embedding",
            MenuAction::AddSelectionToGroup => "Add selection to active group",
            MenuAction::RemoveSelectionFromGroup => "Remove selection from active group",
            MenuAction::OpenSelectionInNewTab => "Open selection in new tab",
            MenuAction::ExcludeDirsFromScan => "Exclude selected directories from recursive scans",
            MenuAction::FindSimilar => "Find Similar (CLIP)",
            MenuAction::RefineSelection => "Refine Selection",
            MenuAction::AttachToChatWindow => "Attach to chat window",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            MenuAction::GenerateDescription => svg::AME_AI_DESCRIBE,
            MenuAction::RegenerateDescription | MenuAction::RegenerateClipEmbedding => svg::AME_REGENERATE,
            MenuAction::GenerateClipEmbeddings => svg::AME_CLIP_EMBED,
            MenuAction::AddSelectionToGroup => svg::PLUS_CIRCLE,
            MenuAction::RemoveSelectionFromGroup => svg::MINUS_CIRCLE,
            MenuAction::OpenSelectionInNewTab => svg::AME_NEW_TAB,
            MenuAction::ExcludeDirsFromScan => svg::AME_EXCLUDE_FOLDER,
            MenuAction::FindSimilar => svg::AME_FIND_SIMILAR,
            MenuAction::RefineSelection => svg::AME_REFINE,
            MenuAction::AttachToChatWindow => svg::PAPERCLIP,
        }
    }
}

/// What the selection under a context menu contains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MenuTargets {
    pub rows: usize,
    pub dirs: usize,
}

/// Entries shown for the current mode, each with whether it can run on `targets`.
pub fn menu_entries(mode: ExplorerMode, targets: MenuTargets) -> Vec<(MenuAction, bool)> {
    let any = targets.rows > 0;
    MenuAction::ALL
        .into_iter()
        .filter(|a| *a != MenuAction::AttachToChatWindow || mode == ExplorerMode::Database)
        .map(|a| {
            let enabled = match a {
                MenuAction::AddSelectionToGroup | MenuAction::RemoveSelectionFromGroup => true,
                MenuAction::ExcludeDirsFromScan => targets.dirs > 0,
                _ => any,
            };
            (a, enabled)
        })
        .collect()
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
                crate::app::OPEN_TAB_REQUESTS
                    .lock()
                    .unwrap()
                    .push(crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity: false, similar_scores: None, origin_path: None, background: false });
            }
            MenuAction::GenerateDescription => self.describe(rows, false),
            MenuAction::RegenerateDescription => self.describe(rows, true),
            MenuAction::GenerateClipEmbeddings => {
                for row in rows {
                    if is_virtual(&row.path) {
                        continue;
                    }
                    if let Some(minb) = self.ui_settings.db_min_size_bytes {
                        if row.size < minb {
                            continue;
                        }
                    }
                    if let Some(maxb) = self.ui_settings.db_max_size_bytes {
                        if row.size > maxb {
                            continue;
                        }
                    }
                    self.spawn_clip_generation(row);
                }
            }
            MenuAction::RegenerateClipEmbedding => {
                for row in rows.iter().filter(|r| r.file_type != "<DIR>") {
                    self.spawn_clip_generation(row);
                }
            }
            MenuAction::AddSelectionToGroup => {
                let Some(group_name) = crate::ui::file_table::active_group_name() else {
                    log::warn!("No active logical group; cannot add selection");
                    return;
                };
                let rows: Vec<Thumbnail> = rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| (*r).clone()).collect();
                tokio::spawn(async move {
                    if let Ok(ids) = crate::database::upsert_rows_and_get_ids(rows).await {
                        if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                            let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                        }
                    }
                });
            }
            MenuAction::RemoveSelectionFromGroup => {
                let Some(group_name) = crate::ui::file_table::active_group_name() else {
                    log::warn!("No active logical group; cannot remove selection");
                    return;
                };
                let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).filter(|p| !p.is_empty()).collect();
                tokio::spawn(async move {
                    if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                        let mut ids: Vec<surrealdb::types::RecordId> = Vec::new();
                        for p in paths {
                            if let Ok(Some(id)) = crate::Thumbnail::get_thumbnail_id_by_path(&p).await {
                                ids.push(id);
                            }
                        }
                        for id in ids.iter() {
                            let _ = crate::database::LogicalGroup::remove_thumbnail(&g.id, id).await;
                        }
                    }
                });
            }
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
            if !regenerate {
                if let Some(minb) = self.ui_settings.db_min_size_bytes {
                    if row.size < minb {
                        continue;
                    }
                }
                if let Some(maxb) = self.ui_settings.db_max_size_bytes {
                    if row.size > maxb {
                        continue;
                    }
                }
                if !self.ui_settings.overwrite_descriptions && (row.caption.is_some() || row.description.is_some()) {
                    continue;
                }
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

    /// Searches the CLIP index for rows like `origin` and sends them back as a similarity update.
    pub fn find_similar(&self, origin: String) {
        let tx_updates = self.ai_update_tx.clone();
        tokio::spawn(async move {
            let Ok(Some(thumb)) = crate::Thumbnail::get_thumbnail_by_path(&origin).await else { return };
            let Ok(embed_row) = thumb.clone().get_embedding().await else { return };
            let embed_vec = embed_row.embedding.clone();
            if embed_vec.is_empty() {
                return;
            }
            let Ok(hits) = crate::database::ClipEmbeddingRow::find_similar_by_embedding(&embed_vec, 256, 256, 0).await else { return };
            let mut results = Vec::new();
            for hit in hits.into_iter() {
                if hit.path == origin {
                    continue;
                }
                let t = match hit.thumb_ref {
                    Some(t) => t,
                    None => crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default(),
                };
                let cosine_sim = 1.0 - hit.dist;
                let norm_sim = ((cosine_sim + 1.0) / 2.0).clamp(0.0, 1.0);
                results.push(crate::ui::file_table::SimilarResult { thumb: t, created: None, updated: None, similarity_score: Some(norm_sim), clip_similarity_score: Some(norm_sim) });
                if results.len() >= 96 {
                    break;
                }
            }
            let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: origin.clone(), results });
        });
    }

    /// Side effects of a row taking the focus: archive thumbnails on demand and a CLIP presence check.
    pub fn on_row_focused(&self, row: &Thumbnail) {
        if row.path.is_empty() || row.file_type == "<DIR>" {
            return;
        }
        if self.mode != ExplorerMode::Database && is_virtual(&row.path) && !self.thumbs.has_thumb(&row.path) {
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
    fn menu_ids_are_the_ids_of_the_previous_table() {
        let ids: Vec<&str> = MenuAction::ALL.iter().map(|a| a.id()).collect();
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
            assert!(!a.label().is_empty());
        }
        assert_eq!(MenuAction::from_id("nope"), None);
    }

    #[test]
    fn labels_contain_no_special_glyphs() {
        for a in MenuAction::ALL {
            assert!(a.label().chars().all(|c| c.is_ascii()), "{}", a.label());
        }
    }

    #[test]
    fn attach_to_chat_only_appears_in_database_mode() {
        let t = MenuTargets { rows: 2, dirs: 0 };
        assert!(!menu_entries(ExplorerMode::FileSystem, t).iter().any(|(a, _)| *a == MenuAction::AttachToChatWindow));
        assert!(menu_entries(ExplorerMode::Database, t).iter().any(|(a, _)| *a == MenuAction::AttachToChatWindow));
        assert_eq!(menu_entries(ExplorerMode::FileSystem, t).len(), 10);
        assert_eq!(menu_entries(ExplorerMode::Database, t).len(), 11);
    }

    #[test]
    fn exclude_directories_needs_a_selected_folder() {
        let enabled = |t: MenuTargets| menu_entries(ExplorerMode::FileSystem, t).into_iter().find(|(a, _)| *a == MenuAction::ExcludeDirsFromScan).unwrap().1;
        assert!(!enabled(MenuTargets { rows: 3, dirs: 0 }));
        assert!(enabled(MenuTargets { rows: 3, dirs: 1 }));
    }

    #[test]
    fn row_actions_need_at_least_one_row() {
        let none = menu_entries(ExplorerMode::Database, MenuTargets::default());
        for (a, enabled) in none {
            let expected = matches!(a, MenuAction::AddSelectionToGroup | MenuAction::RemoveSelectionFromGroup);
            assert_eq!(enabled, expected, "{a:?}");
        }
        assert!(menu_entries(ExplorerMode::Database, MenuTargets { rows: 1, dirs: 0 }).iter().filter(|(a, _)| *a != MenuAction::ExcludeDirsFromScan).all(|(_, e)| *e));
    }
}
