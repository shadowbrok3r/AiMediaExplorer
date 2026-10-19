//! The application as a command sink: what each command does to tabs, panels and engines.

use std::time::Instant;

use super::dispatch::{CommandError, CommandResult, CommandSink};
use super::model::{ActionId, DevtoolsWindow, RowsTab, Section, Selector, Target, ToastLevel, Tool};
use super::scope::{Scope, SmartView};
use crate::app::SmartMediaApp;
use crate::status::{CLIP_STATUS, GlobalStatusIndicator, StatusState};
use crate::tasks::{TaskEvent, TaskId};
use crate::ui::file_table::FileExplorer;

/// Most files `Open` and `Reveal` act on at once.
const OPEN_LIMIT: usize = 20;

impl SmartMediaApp {
    fn open_explorer_tab(&mut self, title: &str, background: bool, build: impl FnOnce(&mut FileExplorer)) -> String {
        let title = crate::ui::tabs::uniquify_title(&self.context.open_tabs, title);
        self.open_tab(&title, background);
        let mut explorer = FileExplorer::new(true);
        build(&mut explorer);
        self.context.filtered_tabs.insert(title.clone(), explorer);
        title
    }

    fn open_tool_tab(&mut self, tab: &str, background: bool) {
        if self.context.open_tabs.contains(tab) {
            if !background {
                self.focus_or_open_tab(tab);
            }
        } else {
            self.open_tab(tab, background);
        }
    }

    /// Rows of a tag or category fetched in the background, then shown in a new tab through the bus.
    fn open_fetched(&self, title: String, background: bool, origin: String, fetch: impl std::future::Future<Output = anyhow::Result<Vec<crate::database::Thumbnail>>> + Send + 'static) {
        tokio::spawn(async move {
            match fetch.await {
                Ok(rows) => super::send(super::AppCommand::OpenRows(RowsTab { title, rows, origin_path: Some(origin), background, ..Default::default() })),
                Err(e) => super::send(super::AppCommand::Toast { kind: ToastLevel::Error, text: format!("{title}: {e}") }),
            }
        });
    }
}

impl CommandSink for SmartMediaApp {
    fn open_scope(&mut self, scope: &Scope, target: Target, title: Option<&str>, recursive: bool) -> CommandResult {
        let title = title.map(str::to_string).unwrap_or_else(|| scope.default_title());
        let background = target == Target::Background;
        match scope {
            Scope::Folder(path) | Scope::Archive(path) => {
                if target == Target::Current {
                    self.context.active_explorer().init_open_path(path, recursive);
                    return Ok(format!("showing {path} in the focused tab"));
                }
                let opened = self.open_explorer_tab(&title, background, |ex| ex.init_open_path(path, recursive));
                Ok(format!("opened tab '{opened}'"))
            }
            Scope::Library => {
                if target == Target::Current {
                    let ex = self.context.active_explorer();
                    ex.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
                    ex.load_all_database_rows();
                    return Ok("showing the library in the focused tab".to_string());
                }
                let opened = self.open_explorer_tab(&title, background, |ex| {
                    ex.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
                    ex.load_all_database_rows();
                });
                Ok(format!("opened tab '{opened}'"))
            }
            Scope::Tag(tag) => {
                let tag = tag.clone();
                self.open_fetched(title.clone(), background, format!("tag:{tag}"), async move { crate::database::Thumbnail::fetch_by_tag(&tag).await });
                Ok(format!("loading '{title}'"))
            }
            Scope::Category(category) => {
                let category = category.clone();
                self.open_fetched(title.clone(), background, format!("cat:{category}"), async move { crate::database::Thumbnail::fetch_by_category(&category).await });
                Ok(format!("loading '{title}'"))
            }
            Scope::Smart(SmartView::FailedThumbs) => {
                let failed: Vec<(String, String)> = std::iter::once(&self.context.file_explorer)
                    .chain(self.context.filtered_tabs.values())
                    .flat_map(|ex| ex.failed_thumbs.iter().cloned())
                    .collect();
                if failed.is_empty() {
                    return Err(CommandError::not_found("no thumbnails have failed"));
                }
                let title = format!("Failed thumbnails ({})", failed.len());
                self.open_rows(RowsTab { title, rows: failed_thumb_rows(&failed), background, ..Default::default() })
            }
            other => Err(CommandError::unsupported(format!("no loader for {} scopes yet", scope_kind(other)))),
        }
    }

    fn open_rows(&mut self, tab: RowsTab) -> CommandResult {
        let RowsTab { title, rows, showing_similarity, similar_scores, origin_path, background } = tab;
        let count = rows.len();
        let opened = self.open_explorer_tab(&title, background, |ex| {
            ex.set_rows(rows);
            if showing_similarity {
                ex.viewer.showing_similarity = true;
                if let Some(scores) = similar_scores {
                    ex.viewer.similar_scores = scores;
                }
                if let Some(origin) = origin_path {
                    ex.current_path = origin;
                }
            }
        });
        Ok(format!("opened tab '{opened}' with {count} rows"))
    }

    fn open_tool(&mut self, tool: Tool, path: Option<&str>, target: Target) -> CommandResult {
        let background = target == Target::Background;
        if let (Tool::ImageEdit, Some(path)) = (tool, path) {
            self.context.image_edit.open_with_path(path);
        }
        match tool {
            Tool::Refinements | Tool::Logs => self.focus_or_open_tab(tool.tab_title()),
            Tool::ImageEdit => self.open_tool_tab(tool.tab_title(), background),
        }
        Ok(format!("opened {}", tool.tab_title()))
    }

    fn resolve(&mut self, on: &Selector) -> Vec<String> {
        let ex = self.context.active_explorer();
        match on {
            Selector::Paths(paths) => paths.clone(),
            Selector::Selection => {
                let mut paths: Vec<String> = ex.viewer.selected.iter().cloned().collect();
                paths.sort();
                paths
            }
            Selector::Focus => Some(ex.current_thumb.path.clone()).filter(|p| !p.is_empty()).into_iter().collect(),
            Selector::View => ex.table.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect(),
        }
    }

    fn run_action(&mut self, action: ActionId, paths: Vec<String>) -> CommandResult {
        let count = paths.len();
        match action {
            ActionId::Open | ActionId::Reveal => {
                if count > OPEN_LIMIT {
                    return Err(CommandError::invalid(format!("{} acts on at most {OPEN_LIMIT} files at once", action.label())));
                }
                for path in &paths {
                    let target = if action == ActionId::Reveal { std::path::Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default() } else { path.into() };
                    open::that(&target).map_err(|e| CommandError::failed(format!("{}: {e}", target.display())))?;
                }
                Ok(format!("{} {count} files", action.label()))
            }
            ActionId::CopyPath => {
                self.egui_ctx.copy_text(paths.join("\n"));
                Ok(format!("copied {count} paths"))
            }
            ActionId::GenerateClip => {
                CLIP_STATUS.set_state(StatusState::Running, "Selected paths");
                tokio::spawn(async move {
                    let outcome = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&paths).await;
                    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
                    let (kind, text) = match outcome {
                        Ok(added) => (ToastLevel::Success, format!("Generated CLIP embeddings for {added} of {} files", paths.len())),
                        Err(e) => (ToastLevel::Error, format!("CLIP generation failed: {e}")),
                    };
                    super::send(super::AppCommand::Toast { kind, text });
                });
                Ok(format!("generating CLIP for {count} files"))
            }
            ActionId::Describe => Err(CommandError::unsupported("describing chosen files is not wired to the command bus yet")),
            ActionId::AttachToAssistant => self.attach_to_assistant(paths),
            ActionId::Refine => self.refine(paths),
        }
    }

    fn show_settings(&mut self, section: Option<Section>) -> CommandResult {
        match section {
            Some(Section::Appearance) => self.context.appearance.open(),
            Some(Section::Advanced) => self.context.open_ui_settings = true,
            _ => self.context.open_settings_modal = true,
        }
        Ok(format!("settings opened{}", section.map(|s| format!(" at {s:?}")).unwrap_or_default()))
    }

    fn attach_to_assistant(&mut self, paths: Vec<String>) -> CommandResult {
        let count = paths.len();
        crate::ui::assistant::ATTACH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(paths);
        self.context.assistant_window_open = true;
        Ok(format!("attached {count} files to the assistant"))
    }

    fn refine(&mut self, paths: Vec<String>) -> CommandResult {
        let count = paths.len();
        crate::ui::refine::REFINE_SELECTION_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(paths);
        self.focus_or_open_tab(Tool::Refinements.tab_title());
        Ok(format!("refining {count} files"))
    }

    fn toast(&mut self, level: ToastLevel, text: String) -> CommandResult {
        self.context.toast_tx.try_send((level.kind(), text)).map_err(|e| CommandError::failed(e.to_string()))?;
        Ok("toast queued".to_string())
    }

    fn task_update(&mut self, id: TaskId, event: TaskEvent) -> CommandResult {
        crate::tasks::with(|r| r.apply_event(id, event, Instant::now())).map_err(CommandError::not_found)
    }

    fn clear_selection(&mut self) -> CommandResult {
        let ex = self.context.active_explorer();
        let count = ex.viewer.selected.len();
        ex.viewer.selected.clear();
        Ok(format!("cleared {count} selected"))
    }

    fn focus_path(&mut self, path: &str) -> CommandResult {
        let ex = self.context.active_explorer();
        let Some(row) = ex.table.iter().find(|r| r.path == path).cloned() else {
            return Err(CommandError::not_found(format!("{path} is not in the focused tab")));
        };
        ex.current_thumb = row;
        ex.viewer.selected.clear();
        ex.viewer.selected.insert(path.to_string());
        Ok(format!("focused {path}"))
    }

    fn open_devtools(&mut self, window: DevtoolsWindow) -> CommandResult {
        crate::ui::devtools::open(&self.egui_ctx, window);
        Ok(format!("opened {window:?}"))
    }

    fn show_task_center(&mut self) -> CommandResult {
        crate::ui::taskbar::set_center_open(&self.egui_ctx, true);
        Ok("task center opened".to_string())
    }
}

/// Rows describing each failed thumbnail, marked with the failed-thumbnail category.
pub fn failed_thumb_rows(failed: &[(String, String)]) -> Vec<crate::database::Thumbnail> {
    failed
        .iter()
        .map(|(path, error)| {
            let p = std::path::Path::new(path);
            let filename = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let mut row = crate::database::Thumbnail::new(&filename);
            row.file_type = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).unwrap_or_default();
            row.path = path.clone();
            row.filename = filename;
            row.description = Some(format!("Thumbnail failed: {error}"));
            row.category = Some("<FAILED_THUMB>".to_string());
            row.parent_dir = p.parent().map(|pp| pp.to_string_lossy().to_string()).unwrap_or_default();
            row
        })
        .collect()
}

fn scope_kind(scope: &Scope) -> &'static str {
    match scope {
        Scope::Folder(_) => "folder",
        Scope::Archive(_) => "archive",
        Scope::Library => "library",
        Scope::Recent => "recent",
        Scope::Collection(_) => "collection",
        Scope::Tag(_) => "tag",
        Scope::Category(_) => "category",
        Scope::Semantic(_) => "semantic",
        Scope::Similar(_) => "similar",
        Scope::Scan(_) => "scan",
        Scope::Smart(_) => "smart",
    }
}
