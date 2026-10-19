//! The palette's entries: the static action registry and the entries made from sidebar data.

use super::{Effect, Entry, Kind};
use crate::command::{ActionId, AppCommand, DevtoolsWindow, Scope, Section, SmartView, Target, Tool};
use crate::icons::{Icon, svg};
use crate::ui::shell::ShellAction;
use crate::ui::sidebar::links::{Link, PlacesModel, smart_label};

/// Most tags or categories offered to the palette.
const MAX_LINK_ENTRIES: usize = 300;
const NOT_CONNECTED: &str = "Library not connected";
const NO_SELECTION: &str = "No files selected";

/// The state that decides which actions can run.
#[derive(Clone, Copy, Debug, Default)]
pub struct ActionContext {
    pub selected: usize,
    pub focus_file: bool,
    pub db_ready: bool,
    pub sidebar_open: bool,
    pub bottom_bar_open: bool,
    pub right_rail_open: bool,
}

/// Sidebar data the palette searches.
pub struct Sources<'a> {
    pub places: &'a PlacesModel,
    pub collections: &'a [Link],
    pub tags: &'a [Link],
    pub categories: &'a [Link],
    pub scans: &'a [Link],
}

/// Windows of the Devtools menu the palette lists.
pub const DEVTOOLS: [DevtoolsWindow; 28] = [
    DevtoolsWindow::ThemePanel,
    DevtoolsWindow::SettingsStore,
    DevtoolsWindow::Wizard,
    DevtoolsWindow::DbMaintenance,
    DevtoolsWindow::LegacyImport,
    DevtoolsWindow::ThumbStorage,
    DevtoolsWindow::ThumbCache,
    DevtoolsWindow::Tasks,
    DevtoolsWindow::CommandLog,
    DevtoolsWindow::Perf,
    DevtoolsWindow::ZeroClaw,
    DevtoolsWindow::ZeroClawJobs,
    DevtoolsWindow::RowStore,
    DevtoolsWindow::Collections,
    DevtoolsWindow::Query,
    DevtoolsWindow::BrowserActions,
    DevtoolsWindow::Video,
    DevtoolsWindow::Scope,
    DevtoolsWindow::Filters,
    DevtoolsWindow::Grid,
    DevtoolsWindow::Inspector,
    DevtoolsWindow::Journal,
    DevtoolsWindow::Workspace,
    DevtoolsWindow::Refine,
    DevtoolsWindow::Sidebar,
    DevtoolsWindow::Actions,
    DevtoolsWindow::McpServer,
    DevtoolsWindow::ZeroClawUsage,
];

/// Words shown for a devtools window.
pub fn devtools_label(window: DevtoolsWindow) -> String {
    let debug = format!("{window:?}");
    let mut out = String::new();
    for (i, c) in debug.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push(' ');
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Words shown for a settings section.
pub fn section_label(section: Section) -> &'static str {
    match section {
        Section::General => "General",
        Section::Library => "Library",
        Section::AiModels => "AI models",
        Section::Automation => "Automation",
        Section::Assistant => "Assistant and providers",
        Section::MetadataTools => "Metadata tools",
        Section::Shortcuts => "Shortcuts",
        Section::Appearance => "Appearance",
        Section::Advanced => "Advanced",
    }
}

const SECTIONS: [Section; 9] = [
    Section::General,
    Section::Library,
    Section::AiModels,
    Section::Automation,
    Section::Assistant,
    Section::MetadataTools,
    Section::Shortcuts,
    Section::Appearance,
    Section::Advanced,
];

fn entry(id: impl Into<String>, title: impl Into<String>, icon: Icon, effect: Effect) -> Entry {
    Entry { id: id.into(), kind: Kind::Action, title: title.into(), detail: String::new(), keywords: String::new(), shortcut: None, icon, effect, enabled: Ok(()) }
}

impl Entry {
    fn keywords(mut self, words: &str) -> Self {
        self.keywords = words.to_string();
        self
    }

    fn shortcut(mut self, shortcut: &'static str) -> Self {
        self.shortcut = Some(shortcut);
        self
    }

    fn needs(mut self, ok: bool, reason: &str) -> Self {
        if !ok && self.enabled.is_ok() {
            self.enabled = Err(reason.to_string());
        }
        self
    }

    fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }
}

fn command(id: impl Into<String>, title: impl Into<String>, icon: Icon, cmd: AppCommand) -> Entry {
    entry(id, title, icon, Effect::Command(cmd))
}

fn shell(action: ShellAction, icon: Icon) -> Entry {
    entry(format!("shell.{}", action.id()), action.label(), icon, Effect::Shell(action))
}

fn action_icon(action: ActionId) -> Icon {
    match action {
        ActionId::Open => svg::ARROW_SQUARE_OUT,
        ActionId::Reveal => svg::AME_REVEAL_IN_EXPLORER,
        ActionId::CopyPath => svg::COPY,
        ActionId::GenerateClip => svg::AME_CLIP_EMBED,
        ActionId::Describe => svg::AME_AI_DESCRIBE,
        ActionId::AttachToAssistant => svg::PAPERCLIP,
        ActionId::Refine => svg::AME_REFINE,
        other => crate::command::actions::spec_of(other).icon,
    }
}

fn action_slug(action: ActionId) -> &'static str {
    match action {
        ActionId::Open => "open",
        ActionId::Reveal => "reveal",
        ActionId::CopyPath => "copy_path",
        ActionId::GenerateClip => "generate_clip",
        ActionId::Describe => "describe",
        ActionId::AttachToAssistant => "attach_to_assistant",
        ActionId::Refine => "refine",
        ActionId::AddToLibrary => "add_to_library",
        ActionId::RemoveFromLibrary => "remove_from_library",
        ActionId::RegenerateDescription => "regenerate_description",
        ActionId::RegenerateClip => "regenerate_clip",
        ActionId::FindSimilar => "find_similar",
        ActionId::Tag => "tag",
        ActionId::Category => "category",
        ActionId::AddToCollection => "add_to_collection",
        ActionId::RemoveFromCollection => "remove_from_collection",
        ActionId::OpenInNewTab => "open_in_new_tab",
        ActionId::CopyTo => "copy_to",
        ActionId::MoveTo => "move_to",
        ActionId::ExcludeFolder => "exclude_folder",
    }
}

/// Every static action, with whether it can run in `cx`.
pub fn registry(cx: &ActionContext) -> Vec<Entry> {
    let mut all = Vec::new();
    let library = |e: Entry| e.needs(cx.db_ready, NOT_CONNECTED);

    all.push(library(command("open_scope.library", "Library: All media", svg::AME_THUMBNAIL_GRID, AppCommand::OpenScope { scope: Scope::Library, target: Target::Current, title: None, recursive: false }).keywords("database everything")));
    all.push(library(command("open_scope.recent", "Library: Recent", svg::CLOCK_COUNTER_CLOCKWISE, AppCommand::OpenScope { scope: Scope::Recent, target: Target::Current, title: None, recursive: false })));
    for view in SmartView::ALL {
        let scope = Scope::Smart(view);
        let (short, icon) = smart_label(view);
        all.push(library(
            command(format!("open_scope.smart.{}", view.slug()), format!("Library: {short}"), icon, AppCommand::OpenScope { scope: scope.clone(), target: Target::Current, title: None, recursive: false })
                .detail(scope.format())
                .keywords(&format!("smart view {}", view.label())),
        ));
    }

    for action in ActionId::ALL {
        let title = format!("{}: selected files", action.label());
        let run = AppCommand::Run { action, on: crate::command::Selector::Selection };
        all.push(command(format!("run.{}", action_slug(action)), title, action_icon(action), run).needs(cx.selected > 0, NO_SELECTION));
    }
    all.push(command("clear_selection", "Clear selection", svg::X, AppCommand::ClearSelection).needs(cx.selected > 0, NO_SELECTION));
    all.push(library(command("undo", "Undo metadata change", svg::ARROW_COUNTER_CLOCKWISE, AppCommand::Undo).shortcut("Ctrl+Z").keywords("revert journal")));
    all.push(library(command("redo", "Redo metadata change", svg::ARROW_CLOCKWISE, AppCommand::Redo).shortcut("Ctrl+Shift+Z").keywords("reapply")));

    for (tool, icon, words) in [(Tool::ImageEdit, svg::AME_IMAGE_EDIT, "qwen edit"), (Tool::Refinements, svg::AME_REFINE, "refine proposals"), (Tool::Logs, svg::TERMINAL_WINDOW, "log console")] {
        let name = format!("{:?}", tool).to_lowercase();
        all.push(command(format!("open_tool.{name}"), format!("Open tool: {}", tool.tab_title()), icon, AppCommand::OpenTool { tool, path: None, target: Target::NewTab }).keywords(words));
    }

    all.push(command("show_settings", "Settings", svg::GEAR, AppCommand::ShowSettings { section: None }).shortcut("Ctrl+,").keywords("preferences options"));
    for section in SECTIONS {
        let slug = serde_json::to_value(section).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        all.push(command(format!("show_settings.{slug}"), format!("Settings: {}", section_label(section)), svg::GEAR, AppCommand::ShowSettings { section: Some(section) }).keywords("preferences options"));
    }

    all.push(command("show_task_center", "Show task center", svg::LIST_CHECKS, AppCommand::ShowTaskCenter).keywords("tasks progress background"));
    all.push(command("open_palette", "Command palette", svg::MAGNIFYING_GLASS, AppCommand::OpenPalette { query: None }).shortcut("Ctrl+K").keywords("search run"));

    for window in DEVTOOLS {
        let slug = serde_json::to_value(window).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        all.push(command(format!("open_devtools.{slug}"), format!("Devtools: {}", devtools_label(window)), svg::WRENCH, AppCommand::OpenDevtools { window }).keywords("debug developer"));
    }

    all.push(shell(ShellAction::ToggleSidebar, svg::SIDEBAR_SIMPLE).shortcut("Ctrl+B").detail(on_off(cx.sidebar_open)));
    all.push(shell(ShellAction::ToggleBottomBar, svg::LIST_CHECKS).detail(on_off(cx.bottom_bar_open)).keywords("status bar tasks"));
    all.push(shell(ShellAction::ToggleRightRail, svg::COLUMNS).detail(on_off(cx.right_rail_open)).keywords("inspector assistant"));
    all.push(shell(ShellAction::OpenFolder, svg::FOLDER_OPEN).keywords("browse directory"));
    all.push(shell(ShellAction::ScanFolder, svg::AME_SCAN).keywords("recursive index"));
    all.push(command("dialog.process", "Process files...", svg::AME_MODEL, AppCommand::Dialog { dialog: crate::command::DialogKind::Process, folder: None }).keywords("describe clip embeddings bulk batch"));
    all.push(shell(ShellAction::ExportBackup, svg::AME_DB_EXPORT).keywords("database surql save"));
    all.push(shell(ShellAction::ImportBackup, svg::AME_DB_IMPORT).keywords("database surql restore load"));
    all.push(shell(ShellAction::ChooseDbFolder, svg::FOLDER_OPEN).keywords("database location"));
    all.push(shell(ShellAction::DatabaseMaintenance, svg::AME_DATABASE).keywords("backup restore snapshot verify"));
    all.push(shell(ShellAction::NewCollection, svg::FOLDER_PLUS).needs(cx.db_ready, NOT_CONNECTED));
    all.push(shell(ShellAction::ShowCollections, svg::AME_LOGICAL_GROUP));
    all.push(shell(ShellAction::ConnectLibrary, svg::PLUGS_CONNECTED).needs(!cx.db_ready, "Library is already connected").keywords("database setup"));
    all.push(shell(ShellAction::EnableAiSearch, svg::AME_SEMANTIC_SEARCH).keywords("load vision model"));
    all.push(shell(ShellAction::EnableAutoIndexing, svg::AME_AI_DESCRIBE).keywords("descriptions"));
    all.push(shell(ShellAction::EnableAutoClip, svg::AME_CLIP_EMBED).keywords("embeddings backfill"));
    all.push(shell(ShellAction::ClipMissing, svg::AME_EMBEDDING).keywords("embeddings backfill"));
    all.push(shell(ShellAction::ClipSelected, svg::CURSOR_CLICK).needs(cx.focus_file, "No file is previewed").keywords("embedding"));
    all.push(shell(ShellAction::ClipMissingCount, svg::GAUGE).keywords("embeddings"));
    all.push(shell(ShellAction::ShowAbout, svg::INFO));
    all.push(shell(ShellAction::Quit, svg::POWER).keywords("exit close"));
    all
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

fn from_link(kind: Kind, link: &Link, enabled: Result<(), String>) -> Entry {
    Entry {
        id: format!("link.{}", link.key),
        kind,
        title: link.label.clone(),
        detail: link.scope.format(),
        keywords: format!("{} {}", kind.label(), link.tooltip.replace('\n', " ")),
        shortcut: None,
        icon: link.icon,
        effect: Effect::Command(link.command(Target::Current, false)),
        enabled,
    }
}

/// The registry plus an entry for every place, collection, tag, category and scan.
pub fn build(cx: &ActionContext, sources: &Sources) -> Vec<Entry> {
    let mut all = registry(cx);
    let library = |ok: bool| if ok { Ok(()) } else { Err(NOT_CONNECTED.to_string()) };
    let places = sources.places;
    all.extend(places.quick.iter().chain(&places.drives).chain(&places.wsl).map(|l| from_link(Kind::Place, l, Ok(()))));
    all.extend(places.recent.iter().map(|l| from_link(Kind::Recent, l, Ok(()))));
    all.extend(sources.collections.iter().map(|l| from_link(Kind::Collection, l, library(cx.db_ready))));
    all.extend(sources.tags.iter().take(MAX_LINK_ENTRIES).map(|l| from_link(Kind::Tag, l, library(cx.db_ready))));
    all.extend(sources.categories.iter().take(MAX_LINK_ENTRIES).map(|l| from_link(Kind::Category, l, library(cx.db_ready))));
    all.extend(sources.scans.iter().map(|l| from_link(Kind::Scan, l, library(cx.db_ready))));
    all
}

/// The "search the library" entry for a typed `text`.
pub fn search_entry(text: &str, db_ready: bool) -> Option<Entry> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let scope = Scope::Semantic(text.to_string());
    Some(Entry {
        id: "search.library".to_string(),
        kind: Kind::Search,
        title: format!("Search library for \"{text}\""),
        detail: scope.format(),
        keywords: String::new(),
        shortcut: None,
        icon: svg::AME_SEMANTIC_SEARCH,
        effect: Effect::Command(AppCommand::OpenScope { scope, target: Target::Current, title: None, recursive: false }),
        enabled: if db_ready { Ok(()) } else { Err(NOT_CONNECTED.to_string()) },
    })
}
