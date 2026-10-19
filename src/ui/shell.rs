//! Window shell: the sidebar panel host, the shared menu and palette actions, and the overlays they open.

use std::time::Instant;

use eframe::egui::{Context, CursorIcon, Frame, Id, Key, Margin, Modifiers, Panel, Rect, Sense, TextEdit, Ui, ViewportCommand, Window};

use crate::app::SmartMediaApp;
use crate::command::{self, AppCommand, Scope, Target};
use crate::status::{CLIP_STATUS, GlobalStatusIndicator, StatusState, VISION_STATUS};
use crate::ui::dialogs;
use crate::ui::file_table::FileExplorer;
use crate::ui::palette::{self, ActionContext, Effect, PaletteState, Sources};
use crate::ui::sidebar::layout::{DEFAULT_WIDTH, LayoutStore, UiLayout, clamp_width};
use crate::ui::sidebar::links::{self, Link};
use crate::ui::sidebar::{Mode, SectionId, SidebarAction, SidebarInput, SidebarState, Source};

/// Tags and categories offered to the palette.
const PALETTE_LINKS: usize = 300;

/// An operation shared by the menus and the palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShellAction {
    ToggleSidebar,
    ToggleBottomBar,
    ToggleRightRail,
    OpenFolder,
    ScanFolder,
    ExportBackup,
    ImportBackup,
    ChooseDbFolder,
    DatabaseMaintenance,
    NewCollection,
    ShowCollections,
    ConnectLibrary,
    EnableAiSearch,
    EnableAutoIndexing,
    EnableAutoClip,
    ClipMissing,
    ClipSelected,
    ClipMissingCount,
    ShowAbout,
    Quit,
}

impl ShellAction {
    pub const ALL: [ShellAction; 20] = [
        Self::ToggleSidebar,
        Self::ToggleBottomBar,
        Self::ToggleRightRail,
        Self::OpenFolder,
        Self::ScanFolder,
        Self::ExportBackup,
        Self::ImportBackup,
        Self::ChooseDbFolder,
        Self::DatabaseMaintenance,
        Self::NewCollection,
        Self::ShowCollections,
        Self::ConnectLibrary,
        Self::EnableAiSearch,
        Self::EnableAutoIndexing,
        Self::EnableAutoClip,
        Self::ClipMissing,
        Self::ClipSelected,
        Self::ClipMissingCount,
        Self::ShowAbout,
        Self::Quit,
    ];

    /// Stable name used in the registry ids.
    pub fn id(self) -> &'static str {
        match self {
            Self::ToggleSidebar => "toggle_sidebar",
            Self::ToggleBottomBar => "toggle_bottom_bar",
            Self::ToggleRightRail => "toggle_right_rail",
            Self::OpenFolder => "open_folder",
            Self::ScanFolder => "scan_folder",
            Self::ExportBackup => "export_backup",
            Self::ImportBackup => "import_backup",
            Self::ChooseDbFolder => "choose_db_folder",
            Self::DatabaseMaintenance => "database_maintenance",
            Self::NewCollection => "new_collection",
            Self::ShowCollections => "show_collections",
            Self::ConnectLibrary => "connect_library",
            Self::EnableAiSearch => "enable_ai_search",
            Self::EnableAutoIndexing => "enable_auto_indexing",
            Self::EnableAutoClip => "enable_auto_clip",
            Self::ClipMissing => "clip_missing",
            Self::ClipSelected => "clip_selected",
            Self::ClipMissingCount => "clip_missing_count",
            Self::ShowAbout => "show_about",
            Self::Quit => "quit",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ToggleSidebar => "Toggle sidebar",
            Self::ToggleBottomBar => "Toggle bottom bar",
            Self::ToggleRightRail => "Toggle right rail",
            Self::OpenFolder => "Open folder...",
            Self::ScanFolder => "Scan folder...",
            Self::ExportBackup => "Export backup...",
            Self::ImportBackup => "Import backup...",
            Self::ChooseDbFolder => "Choose database folder...",
            Self::DatabaseMaintenance => "Database backup and restore",
            Self::NewCollection => "New collection...",
            Self::ShowCollections => "Show collections",
            Self::ConnectLibrary => "Connect library",
            Self::EnableAiSearch => "Enable AI search",
            Self::EnableAutoIndexing => "Enable auto indexing",
            Self::EnableAutoClip => "Enable auto CLIP",
            Self::ClipMissing => "Generate CLIP for missing files",
            Self::ClipSelected => "Generate CLIP for the previewed file",
            Self::ClipMissingCount => "Count files missing CLIP",
            Self::ShowAbout => "About",
            Self::Quit => "Quit",
        }
    }
}

/// The window chrome state the app keeps besides the dock.
pub struct Shell {
    pub layout: LayoutStore,
    pub sidebar: SidebarState,
    pub palette: PaletteState,
    pub about_open: bool,
    /// Text of the new-collection dialog while it is open.
    pub new_collection: Option<String>,
    /// Finished tasks counted last frame; a change reloads the sidebar lists.
    finished_tasks: usize,
}

impl Default for Shell {
    fn default() -> Self {
        Self { layout: LayoutStore::memory(), sidebar: SidebarState::default(), palette: PaletteState::default(), about_open: false, new_collection: None, finished_tasks: 0 }
    }
}

impl Shell {
    /// The shell of a running app: saved layout and live sidebar data.
    pub fn live() -> Self {
        Self { layout: LayoutStore::load(), sidebar: SidebarState::live(), ..Default::default() }
    }
}

/// Where the sidebar panel was drawn in the last frame.
#[derive(Clone, Copy, Debug)]
pub struct SidebarLayout {
    pub rect: Rect,
    pub mode: Mode,
}

fn sidebar_layout_id() -> Id {
    Id::new("ame_sidebar_layout")
}

/// The layout recorded by the last [`SmartMediaApp::side_panels`] on `ctx`.
pub fn last_sidebar(ctx: &Context) -> Option<SidebarLayout> {
    ctx.data(|d| d.get_temp::<SidebarLayout>(sidebar_layout_id()))
}

fn focused_scope(ex: &FileExplorer) -> Option<Scope> {
    Some(ex.scope().clone())
}

/// Thumbnails that failed in the main explorer and every tab.
fn failed_thumbs_total(app: &crate::app::SmartMediaContext) -> usize {
    std::iter::once(&app.file_explorer).chain(app.filtered_tabs.values()).map(|ex| ex.failed_thumbs.len()).sum()
}

impl SmartMediaApp {
    /// Draws the sidebar panel (or its icon rail) left of the dock.
    pub fn side_panels(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        self.shell.layout.layout.right_rail_open = self.context.inspector.is_open();
        let mode = crate::ui::sidebar::mode_for(ctx.content_rect().width(), self.shell.layout.layout.sidebar_open);
        if mode == Mode::Hidden {
            ctx.data_mut(|d| d.insert_temp(sidebar_layout_id(), SidebarLayout { rect: Rect::NOTHING, mode }));
            return;
        }
        let recent = crate::settings::store().read(|s| s.recent_paths.clone());
        let failed = failed_thumbs_total(&self.context);
        let active = focused_scope(self.context.active_explorer());
        let input = SidebarInput { db_ready: self.context.db_ready, active: active.as_ref(), failed_thumbs: failed, recent_paths: &recent, now: Instant::now() };
        let Shell { layout, sidebar, .. } = &mut self.shell;
        let tokens = crate::theme::tokens(&ctx);
        let (width, margin) = match mode {
            Mode::Rail => (crate::ui::sidebar::RAIL_WIDTH, Margin::symmetric(4, 6)),
            _ => (clamp_width(layout.layout.sidebar_width), Margin::symmetric(6, 6)),
        };
        let frame = Frame::new().fill(tokens.base).inner_margin(margin);
        let panel = Panel::left("ame_sidebar").resizable(false).exact_size(width).frame(frame).show(ui, |ui| match mode {
            Mode::Rail => crate::ui::sidebar::show_rail(ui, sidebar, &input),
            _ => crate::ui::sidebar::show_full(ui, sidebar, &mut layout.layout, &input),
        });
        if mode == Mode::Full {
            resize_handle(ui, panel.response.rect, &mut layout.layout);
        }
        ctx.data_mut(|d| d.insert_temp(sidebar_layout_id(), SidebarLayout { rect: panel.response.rect, mode }));
        self.apply_sidebar_actions(panel.inner);
    }

    fn apply_sidebar_actions(&mut self, actions: Vec<SidebarAction>) {
        for action in actions {
            match action {
                SidebarAction::Command(cmd) => command::send(cmd),
                SidebarAction::Connect => self.context.show_db_select_modal = true,
            }
        }
    }

    /// The facts that decide which palette actions are enabled.
    fn action_context(&mut self) -> ActionContext {
        let ex = self.context.active_explorer();
        let (selected, focus_file) = (ex.selection_count(), !ex.current_thumb.path.is_empty());
        let layout = &self.shell.layout.layout;
        ActionContext {
            selected,
            focus_file,
            db_ready: self.context.db_ready,
            sidebar_open: layout.sidebar_open,
            bottom_bar_open: layout.bottom_bar_open,
            right_rail_open: self.context.inspector.is_open(),
        }
    }

    /// Palette entries from the registry and the cached sidebar data.
    pub fn palette_entries(&mut self) -> Vec<palette::Entry> {
        let cx = self.action_context();
        let data = &self.shell.sidebar.data;
        let raw = data.places();
        let recent = crate::settings::store().read(|s| s.recent_paths.clone());
        let places = links::places_links(&raw.quick, &raw.drives, &raw.wsl, &recent);
        let collections = links::collection_links(&data.collections());
        let tags: Vec<Link> = data.tags().iter().take(PALETTE_LINKS).map(links::tag_link).collect();
        let categories: Vec<Link> = data.categories().iter().take(PALETTE_LINKS).map(links::category_link).collect();
        let scans = links::scan_links(&data.scans());
        palette::entries::build(&cx, &Sources { places: &places, collections: &collections, tags: &tags, categories: &categories, scans: &scans })
    }

    /// Keyboard shortcuts, the palette and the dialogs of the shell; call once per frame after the panels.
    pub fn shell_overlays(&mut self, ctx: &Context) {
        let (palette_key, sidebar_key) = ctx.input_mut(|i| (i.consume_key(Modifiers::COMMAND, Key::K), i.consume_key(Modifiers::COMMAND, Key::B)));
        if palette_key {
            self.shell.palette.toggle();
        }
        if sidebar_key {
            self.run_shell_action(ctx, ShellAction::ToggleSidebar);
        }
        let finished = crate::tasks::with(|r| r.counts().finished);
        if std::mem::replace(&mut self.shell.finished_tasks, finished) != finished {
            self.shell.sidebar.data.invalidate_all();
        }
        if self.shell.palette.open {
            let (now, ready) = (Instant::now(), self.context.db_ready);
            Source::ALL.iter().for_each(|s| self.shell.sidebar.data.poll(ctx, *s, ready || *s == Source::Places, now));
            let entries = self.palette_entries();
            let db_ready = self.context.db_ready;
            if let Some(effect) = palette::show(ctx, &mut self.shell.palette, &entries, db_ready) {
                self.run_effect(ctx, effect);
            }
        }
        self.about_window(ctx);
        self.new_collection_window(ctx);
        if !ctx.input(|i| i.pointer.any_down()) {
            self.shell.layout.save_if_changed();
        }
        self.publish_devtools(ctx);
        self.run_devtools_requests(ctx);
    }

    /// Records what the open devtools windows of the shell show.
    fn publish_devtools(&mut self, ctx: &Context) {
        use crate::ui::devtools;
        devtools::sidebar::publish(ctx, &self.shell);
        if devtools::actions::is_open(ctx) {
            let cx = self.action_context();
            devtools::actions::publish(ctx, palette::registry(&cx));
        }
        if devtools::perf::is_open(ctx) {
            devtools::scan_perf::publish(ctx, self.context.active_explorer().scan_perf());
        }
    }

    /// Runs the effect of a palette entry.
    pub fn run_effect(&mut self, ctx: &Context, effect: Effect) {
        match effect {
            Effect::Command(cmd) => command::send(cmd),
            Effect::Shell(action) => self.run_shell_action(ctx, action),
        }
    }

    /// Does what the menu item or palette entry `action` names.
    pub fn run_shell_action(&mut self, ctx: &Context, action: ShellAction) {
        match action {
            ShellAction::ToggleSidebar => self.shell.layout.layout.sidebar_open ^= true,
            ShellAction::ToggleBottomBar => self.shell.layout.layout.bottom_bar_open ^= true,
            ShellAction::ToggleRightRail => {
                let open = self.context.inspector.toggle();
                self.shell.layout.layout.right_rail_open = open;
            }
            ShellAction::OpenFolder => dialogs::pick_folder(ctx, "Open folder", |dir| command::send(open_folder_command(&dir, false))),
            ShellAction::ScanFolder => dialogs::pick_folder(ctx, "Scan folder", |dir| command::send(open_folder_command(&dir, true))),
            ShellAction::ExportBackup => export_backup(ctx),
            ShellAction::ImportBackup => import_backup(ctx),
            ShellAction::ChooseDbFolder => choose_db_folder(ctx),
            ShellAction::DatabaseMaintenance => crate::ui::devtools::open_db_maintenance(ctx),
            ShellAction::NewCollection => self.shell.new_collection = Some(String::new()),
            ShellAction::ShowCollections => {
                let layout = &mut self.shell.layout.layout;
                layout.sidebar_open = true;
                layout.set_section_open(SectionId::Collections.key(), true);
            }
            ShellAction::ConnectLibrary => self.context.show_db_select_modal = true,
            ShellAction::EnableAiSearch => {
                VISION_STATUS.set_state(StatusState::Initializing, "Loading vision model");
                tokio::spawn(async move {
                    crate::ai::init_global_ai_engine_async().await;
                    VISION_STATUS.set_state(StatusState::Idle, "Ready");
                });
            }
            ShellAction::EnableAutoIndexing => {
                crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                VISION_STATUS.set_state(StatusState::Running, "Auto descriptions enabled");
            }
            ShellAction::EnableAutoClip => {
                crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                CLIP_STATUS.set_state(StatusState::Running, "Auto CLIP backfill");
                tokio::spawn(async move {
                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                    log::info!("[CLIP] Auto CLIP enabled via menu; backfill added {added}");
                    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
                    Ok::<(), anyhow::Error>(())
                });
            }
            ShellAction::ClipMissing => {
                CLIP_STATUS.set_state(StatusState::Running, "Generating missing");
                tokio::spawn(async move {
                    let before = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                    let remaining = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                    log::info!("[CLIP] Manual missing generation added {added}; remaining {remaining} (was {before})");
                    CLIP_STATUS.set_state(StatusState::Idle, format!("Remaining {remaining}"));
                    Ok::<(), anyhow::Error>(())
                });
            }
            ShellAction::ClipSelected => {
                let path = self.context.active_explorer().current_thumb.path.clone();
                if path.is_empty() {
                    log::warn!("[CLIP] No current preview path set for generation");
                    return;
                }
                CLIP_STATUS.set_state(StatusState::Running, "Selected path");
                tokio::spawn(async move {
                    match crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await {
                        Ok(added) => log::info!("[CLIP] Manual per-item generation: added {added} for {path}"),
                        Err(e) => log::error!("engine.clip_generate_for_paths: {e:?}"),
                    }
                    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
                });
            }
            ShellAction::ClipMissingCount => {
                tokio::spawn(async move {
                    let missing = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                    log::info!("[CLIP] Missing embeddings: {missing}");
                    CLIP_STATUS.set_state(StatusState::Idle, format!("Missing {missing}"));
                });
            }
            ShellAction::ShowAbout => self.shell.about_open = true,
            ShellAction::Quit => ctx.send_viewport_cmd(ViewportCommand::Close),
        }
        ctx.request_repaint();
    }

    fn about_window(&mut self, ctx: &Context) {
        if !self.shell.about_open {
            return;
        }
        let mut open = true;
        Window::new("About Smart Media").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.heading("Smart Media Explorer");
            ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
            ui.label("Media explorer with local AI search, tagging and collections.");
            ui.separator();
            ui.label(format!("Database folder: {}", crate::database::get_db_path()));
            let layout = self.shell.layout.path().map(|p| p.display().to_string()).unwrap_or_else(|| "not saved".to_string());
            ui.label(format!("Layout file: {layout}"));
            ui.label("Logs: Help > Logs");
        });
        self.shell.about_open = open;
    }

    fn new_collection_window(&mut self, ctx: &Context) {
        let Some(mut name) = self.shell.new_collection.take() else { return };
        let mut open = true;
        let (mut create, mut cancel) = (false, false);
        Window::new("New collection").open(&mut open).collapsible(false).resizable(false).anchor(eframe::egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            let edit = ui.add(TextEdit::singleline(&mut name).hint_text("Collection name").desired_width(260.0));
            edit.request_focus();
            let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            ui.horizontal(|ui| {
                let ready = !name.trim().is_empty();
                create = (ui.add_enabled(ready, eframe::egui::Button::new("Create")).clicked() || enter) && ready;
                cancel = ui.button("Cancel").clicked();
            });
        });
        if create {
            self.shell.sidebar.data.create_collection(ctx, name.trim().to_string());
        } else if open && !cancel {
            self.shell.new_collection = Some(name);
        }
    }

    /// Runs the buttons pressed in the Sidebar and Actions devtools windows.
    fn run_devtools_requests(&mut self, ctx: &Context) {
        for request in crate::ui::devtools::sidebar::take_requests(ctx) {
            crate::ui::devtools::sidebar::apply(request, &mut self.shell, ctx);
        }
        if crate::ui::devtools::scan_perf::take_clear(ctx) {
            self.context.active_explorer().clear_scan_perf();
        }
        for id in crate::ui::devtools::actions::take_requests(ctx) {
            let cx = self.action_context();
            let entries = palette::registry(&cx);
            if let Some(entry) = entries.into_iter().find(|e| e.id == id).filter(|e| e.enabled.is_ok()) {
                self.run_effect(ctx, entry.effect);
            }
        }
    }
}

fn resize_handle(ui: &mut Ui, panel: Rect, layout: &mut UiLayout) {
    let handle = Rect::from_x_y_ranges((panel.right() - 4.0)..=(panel.right() + 2.0), panel.y_range());
    let response = ui.interact(handle, Id::new("ame_sidebar_resize"), Sense::drag());
    let active = response.hovered() || response.dragged();
    if active {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
        let tokens = crate::theme::tokens(ui.ctx());
        ui.painter().vline(panel.right(), panel.y_range(), eframe::egui::Stroke::new(2.0, tokens.primary));
    }
    if response.dragged() {
        layout.sidebar_width = clamp_width(layout.sidebar_width + response.drag_delta().x);
    }
    if response.double_clicked() {
        layout.sidebar_width = DEFAULT_WIDTH;
    }
}

/// The command that opens a folder picked in a dialog.
pub fn open_folder_command(dir: &std::path::Path, scan: bool) -> AppCommand {
    let scope = Scope::Folder(dir.display().to_string());
    if scan { AppCommand::OpenScope { scope, target: Target::NewTab, title: None, recursive: true } } else { AppCommand::OpenScope { scope, target: Target::Current, title: None, recursive: false } }
}

fn export_backup(ctx: &Context) {
    let confirm_ctx = ctx.clone();
    dialogs::save_file(ctx, "Export SurrealDB backup to .surql", "backup.surql", Some(("SurrealQL", &["surql"])), move |file| {
        let description = format!("Export database to {}?", file.display());
        dialogs::confirm(&confirm_ctx, "Export Database", &description, rfd::MessageLevel::Info, move || {
            tokio::spawn(async move {
                let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Database, format!("Export database to {}", file.display())).user_initiated();
                let _ = crate::tasks::track(spec, |_| crate::database::export_to(&file)).await;
            });
        });
    });
}

fn import_backup(ctx: &Context) {
    let confirm_ctx = ctx.clone();
    dialogs::pick_file(ctx, "Import SurrealDB .surql file", move |file| {
        let description = format!("Import from {}? This may overwrite existing data.", file.display());
        dialogs::confirm(&confirm_ctx, "Import Database", &description, rfd::MessageLevel::Warning, move || {
            tokio::spawn(async move {
                let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Database, format!("Import database from {}", file.display())).user_initiated();
                let _ = crate::tasks::track(spec, |_| crate::database::import_from(&file)).await;
            });
        });
    });
}

fn choose_db_folder(ctx: &Context) {
    dialogs::pick_folder(ctx, "Choose database folder", move |dir| {
        let path_str = dir.display().to_string();
        match crate::database::set_db_path(&path_str) {
            Ok(()) => dialogs::toast(egui_toast::ToastKind::Info, format!("Database folder set to {path_str}. Restart the app to open it.")),
            Err(e) => {
                log::error!("set_db_path failed: {e}");
                dialogs::toast(egui_toast::ToastKind::Error, format!("Could not set the database folder: {e}"));
            }
        }
    });
}

#[cfg(test)]
mod tests;
