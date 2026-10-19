//! The application as a command sink: what each command does to tabs, panels and engines.

use std::time::Instant;

use super::browser::{DialogKind, QueryMode};
use super::dispatch::{CommandError, CommandResult, CommandSink};
use super::model::{ActionId, DevtoolsWindow, NavOp, RowsTab, Section, Selector, Target, ToastLevel, Tool};
use super::scope::Scope;
use crate::app::SmartMediaApp;
use crate::status::{CLIP_STATUS, GlobalStatusIndicator, StatusState};
use crate::tasks::{TaskEvent, TaskId};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::scope::nav::check_openable;
use crate::ui::tabs::{BrowserId, TabKind};
use crate::ui::file_table::table::view_mode::{TileFit, ViewMode};

/// Most files `Open` and `Reveal` act on at once.
const OPEN_LIMIT: usize = 20;

impl SmartMediaApp {
    /// Adds a browser tab whose explorer `build` sets up; returns its id and title.
    fn open_browser_tab(&mut self, background: bool, build: impl FnOnce(&mut FileExplorer) -> Result<(), String>) -> Result<(BrowserId, String), CommandError> {
        let mut explorer = FileExplorer::new(true);
        build(&mut explorer).map_err(CommandError::not_found)?;
        let id = BrowserId::next();
        let title = explorer.tab_title();
        self.context.filtered_tabs.insert(id, explorer);
        self.open_tab(TabKind::browser(id), background);
        Ok((id, title))
    }

    fn open_tool_tab(&mut self, tab: TabKind, background: bool) {
        if self.context.open_tabs.contains(&tab) {
            if !background {
                self.focus_or_open_tab(tab);
            }
        } else {
            self.open_tab(tab, background);
        }
    }

    /// Brings the tab of the active explorer to the front, reopening the main one when it was closed.
    fn focus_active_browser(&mut self) {
        let kind = self.context.active_kind();
        self.focus_or_open_tab(kind);
    }
}

impl CommandSink for SmartMediaApp {
    fn open_scope(&mut self, scope: &Scope, target: Target, title: Option<&str>, recursive: bool) -> CommandResult {
        check_openable(scope).map_err(CommandError::not_found)?;
        match target {
            Target::Current => {
                self.focus_active_browser();
                let explorer = self.context.active_explorer();
                explorer.open_scope_recursive(scope.clone(), recursive).map_err(CommandError::not_found)?;
                if let Some(title) = title {
                    explorer.nav.title = Some(title.to_string());
                }
                Ok(format!("showing {scope} in the focused tab"))
            }
            Target::NewTab | Target::Background => {
                let (_, opened) = self.open_browser_tab(target == Target::Background, |ex| {
                    ex.start_at(scope.clone(), recursive)?;
                    if let Some(title) = title {
                        ex.nav.title = Some(title.to_string());
                    }
                    Ok(())
                })?;
                Ok(format!("opened tab '{opened}'"))
            }
        }
    }

    fn open_rows(&mut self, tab: RowsTab) -> CommandResult {
        let RowsTab { title, rows, showing_similarity, similar_scores, origin_path: _, background } = tab;
        let count = rows.len();
        let (_, opened) = self.open_browser_tab(background, |ex| {
            ex.show_rows(Some(title), rows);
            if showing_similarity {
                ex.viewer.showing_similarity = true;
                if let Some(scores) = similar_scores {
                    ex.viewer.similar_scores = scores;
                }
            }
            Ok(())
        })?;
        Ok(format!("opened tab '{opened}' with {count} rows"))
    }

    fn open_tool(&mut self, tool: Tool, path: Option<&str>, target: Target) -> CommandResult {
        let background = target == Target::Background;
        if let (Tool::ImageEdit, Some(path)) = (tool, path) {
            self.context.image_edit.open_with_path(path);
        }
        let kind = TabKind::from(tool);
        match tool {
            Tool::Refinements | Tool::Logs => self.focus_or_open_tab(kind),
            Tool::ImageEdit => self.open_tool_tab(kind, background),
        }
        Ok(format!("opened {}", tool.tab_title()))
    }

    fn resolve(&mut self, on: &Selector) -> Vec<String> {
        let ex = self.context.active_explorer();
        match on {
            Selector::Paths(paths) => paths.clone(),
            Selector::Selection => {
                ex.sync_view();
                let mut paths: Vec<String> = ex.selected_rows().iter().map(|r| r.path.clone()).collect();
                paths.sort();
                paths
            }
            Selector::Focus => Some(ex.current_thumb.path.clone()).filter(|p| !p.is_empty()).into_iter().collect(),
            Selector::View => {
                ex.sync_view();
                ex.view.rows(&ex.table).filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect()
            }
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
            ActionId::Describe
            | ActionId::RegenerateDescription
            | ActionId::RegenerateClip
            | ActionId::AddToLibrary
            | ActionId::RemoveFromLibrary
            | ActionId::FindSimilar
            | ActionId::Tag
            | ActionId::Category
            | ActionId::AddToCollection
            | ActionId::RemoveFromCollection
            | ActionId::OpenInNewTab
            | ActionId::CopyTo
            | ActionId::MoveTo
            | ActionId::ExcludeFolder => self.context.active_explorer().run_action_paths(&self.egui_ctx, action, paths),
            ActionId::AttachToAssistant => self.attach_to_assistant(paths),
            ActionId::Refine => self.refine(paths),
        }
    }

    fn show_settings(&mut self, section: Option<Section>) -> CommandResult {
        self.context.settings_page.select(section);
        self.focus_or_open_tab(TabKind::Settings);
        Ok(format!("settings opened{}", section.map(|s| format!(" at {s:?}")).unwrap_or_default()))
    }

    fn backup(&mut self, op: crate::ui::settings::backup::Op, path: Option<String>, dir: Option<String>, redact_keys: bool, include_thumbs: bool) -> CommandResult {
        let choice = crate::ui::settings::backup::ExportChoice { redact_keys, include_thumb_blobs: include_thumbs };
        crate::ui::settings::backup::run_command(op, path.as_deref(), dir.as_deref(), choice)
    }

    fn wizard(&mut self, step: Option<crate::ui::wizard::Step>) -> CommandResult {
        match step {
            Some(step) => self.context.wizard.open_at(step, crate::ui::wizard::Reason::Asked),
            None => self.context.wizard.open_next_pending(),
        }
        Ok(format!("wizard opened at {}", self.context.wizard.current().label()))
    }

    fn attach_to_assistant(&mut self, paths: Vec<String>) -> CommandResult {
        let count = paths.len();
        crate::ui::assistant::ATTACH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(paths);
        crate::ui::assistant::show_assistant(self);
        Ok(format!("attached {count} files to the assistant"))
    }

    fn refine(&mut self, paths: Vec<String>) -> CommandResult {
        let count = paths.len();
        crate::ui::refine::REFINE_SELECTION_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(paths);
        self.focus_or_open_tab(TabKind::Refine);
        Ok(format!("refining {count} files"))
    }

    fn undo(&mut self) -> CommandResult {
        crate::ui::journal_ui::start(crate::database::journal::Direction::Undo)
    }

    fn redo(&mut self) -> CommandResult {
        crate::ui::journal_ui::start(crate::database::journal::Direction::Redo)
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
        ex.sync_view();
        let count = ex.selection_count();
        ex.clear_selection();
        Ok(format!("cleared {count} selected"))
    }

    fn query(&mut self, mode: QueryMode, text: String, run: bool) -> CommandResult {
        self.focus_active_browser();
        self.context.active_explorer().apply_query_command(mode, &text, run)
    }

    fn open_dialog(&mut self, dialog: DialogKind, folder: Option<String>) -> CommandResult {
        self.focus_active_browser();
        let ctx = self.egui_ctx.clone();
        self.context.active_explorer().open_dialog(&ctx, dialog, folder)
    }

    fn focus_path(&mut self, path: &str) -> CommandResult {
        let ex = self.context.active_explorer();
        if !ex.table.contains_path(path) {
            return Err(CommandError::not_found(format!("{path} is not in the focused tab")));
        }
        ex.sync_view();
        ex.select_paths([path]);
        if ex.selection_count() == 0 {
            return Err(CommandError::not_found(format!("{path} is hidden by the filters of the focused tab")));
        }
        self.context.inspector.set_open(true);
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

    fn navigate(&mut self, op: NavOp) -> CommandResult {
        self.focus_active_browser();
        let ex = self.context.active_explorer();
        let moved = match op {
            NavOp::Back => ex.go_back(),
            NavOp::Forward => ex.go_forward(),
            NavOp::Up => ex.go_up(),
            NavOp::Home => ex.go_home(),
            NavOp::LoadMore => ex.load_more(),
            NavOp::Refresh => {
                ex.reload_scope();
                true
            }
            NavOp::Cancel => {
                ex.cancel_load();
                true
            }
        };
        if moved { Ok(format!("{}: now at {}", op.label(), ex.scope())) } else { Err(CommandError::invalid(format!("{} has nowhere to go from {}", op.label(), ex.scope()))) }
    }

    fn connect_database(&mut self) -> CommandResult {
        if crate::database::is_connected() {
            return Ok("the database is already connected".to_string());
        }
        let path = self.context.connect_database();
        Ok(format!("opening the database at {path}"))
    }

    fn set_filter(&mut self, filter: &serde_json::Value, replace: bool) -> CommandResult {
        self.context.active_explorer().apply_filter_patch(filter, replace).map_err(CommandError::invalid)
    }

    fn clear_filters(&mut self) -> CommandResult {
        let ex = self.context.active_explorer();
        let count = ex.filter_spec().chips().len();
        ex.clear_filters();
        Ok(format!("cleared {count} filters"))
    }

    fn apply_saved_view(&mut self, name: &str) -> CommandResult {
        self.context.active_explorer().apply_saved_view(name).map_err(CommandError::not_found)
    }

    fn set_view_mode(&mut self, mode: ViewMode, size: Option<f32>, fit: Option<TileFit>, captions: Option<bool>, stress: Option<f32>) -> CommandResult {
        if size.is_some_and(|s| !s.is_finite()) || stress.is_some_and(|s| !s.is_finite()) {
            return Err(CommandError::invalid("size and stress must be finite numbers"));
        }
        let ex = self.context.active_explorer();
        ex.set_view_mode(mode);
        if let Some(size) = size {
            ex.set_tile_size(size);
        }
        if let Some(fit) = fit {
            ex.set_tile_fit(fit);
        }
        if let Some(on) = captions {
            ex.set_captions(on);
        }
        if let Some(rate) = stress {
            ex.browse.grid.set_stress(Some(rate));
        }
        Ok(format!("{} view with {:.0} px tiles", mode.label(), ex.browse.tile))
    }

    fn inspector_toggle(&mut self, open: Option<bool>) -> CommandResult {
        let inspector = &mut self.context.inspector;
        let now_open = match open {
            Some(open) => {
                inspector.set_open(open);
                open
            }
            None => inspector.toggle(),
        };
        self.egui_ctx.request_repaint();
        Ok(format!("inspector {}", if now_open { "shown" } else { "hidden" }))
    }

    fn open_palette(&mut self, query: Option<String>) -> CommandResult {
        self.shell.palette.open_with(query);
        self.egui_ctx.request_repaint();
        Ok("palette opened".to_string())
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
