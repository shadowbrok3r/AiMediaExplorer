use eframe::egui::Context;
use crossbeam::channel::{Receiver, Sender};
use egui_dock::{DockState, NodePath, SurfaceIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::{ui::{file_table::FileExplorer, refine::RefinementsPanel, tabs::{BrowserId, TabKind}}, UiSettings};
use egui_toast::Toasts;
use crate::theme::chrome::Chrome;

/// Requests the settings window from anywhere by queueing a command.
pub static OPEN_SETTINGS_REQUEST: crate::command::legacy::SettingsRequest = crate::command::legacy::SettingsRequest;
/// Requests tabs from anywhere by queueing commands.
pub static OPEN_TAB_REQUESTS: crate::command::legacy::TabRequests = crate::command::legacy::TabRequests;
pub const DEFAULT_JOYCAPTION_PATH: &str = r#"G:\Users\Owner\Documents\1-MyStuff\TechStuff\GitHub\smart-media\llama-joycaption-beta-one-hf-llava"#;
pub const MAX_NEW_TOKENS: usize = 200;
pub const TEMPERATURE: f32 = 0.5;

pub struct SmartMediaApp {
    pub tree: DockState<TabKind>,
    pub context: SmartMediaContext,
    pub egui_ctx: Context,
    pub chrome: Chrome,
}

/// Preferences window state: the settings when it opened and the copy being edited.
pub struct PrefsDraft {
    pub baseline: UiSettings,
    pub draft: UiSettings,
}

pub struct SmartMediaContext {
    pub first_run: bool,
    pub page: Page,
    pub db_ready_tx: Sender<()>,
    pub db_ready_rx: Receiver<()>,
    pub db_ready: bool,
    // Database selection modal state (shown when no database is found)
    pub show_db_select_modal: bool,
    pub db_select_mode: DbSelectMode,
    pub db_websocket_url: String,
    pub db_local_path: String,
    pub db_connection_error: Option<String>,
    // UI State
    pub open_settings_modal: bool,
    // AI init status flags
    pub ai_initializing: bool,
    pub ai_ready: bool,
    // Theme picker window, its database store and the theme sync
    pub appearance: crate::theme::Appearance,
    // Preferences edits, present only while the window is open
    pub prefs: Option<PrefsDraft>,
    // Set by the Egui Settings Save button; handled once the frame storage is reachable
    pub egui_prefs_save_requested: bool,
    // Dock add-button presses waiting to become new explorer tabs
    pub pending_tab_adds: Vec<NodePath>,
    pub file_explorer: crate::ui::file_table::FileExplorer,
    pub open_tabs: HashSet<TabKind>,
    // Browsers other than the main one, by tab id
    pub filtered_tabs: std::collections::HashMap<BrowserId, crate::ui::file_table::FileExplorer>,
    pub assistant: crate::ui::assistant::AssistantPanel,
    pub refinements: crate::ui::refine::RefinementsPanel,
    pub image_edit: crate::ui::image_edit::ImageEditPanel,
    // Watchers that feed the task registry
    pub task_poller: crate::tasks::Poller,
    pub open_ui_settings: bool,
    // Toggle for showing the AI Assistant as a separate viewport window
    pub assistant_window_open: bool,
    // UI state for adding excluded directories
    pub new_excluded_dir: String,
    // Toasts manager and channel for async notifications
    pub toasts: Toasts,
    pub toast_tx: Sender<(egui_toast::ToastKind, String)>,
    pub toast_rx: Receiver<(egui_toast::ToastKind, String)>,
    // Channel for AI Refinements proposals
    pub refine_tx: Sender<Vec<crate::ui::refine::RefinementProposal>>,
    pub refine_rx: Receiver<Vec<crate::ui::refine::RefinementProposal>>,
    // The focused browser tab (updated each frame from the dock state)
    pub active_tab: Option<TabKind>,
    pub ref_old_category: String,
    pub ref_new_category: String,
    pub ref_old_tag: String,
    pub ref_new_tag: String,
    pub ref_delete_tag: String,
    pub ref_limit_tags: i32,
}

#[derive(PartialEq, Debug, Serialize, Deserialize, Default)]
pub enum Page {
    #[default]
    Main
}

#[derive(PartialEq, Debug, Clone, Default)]
pub enum DbSelectMode {
    #[default]
    Local,
    WebSocket,
}

impl SmartMediaApp {
    pub fn new(cc: &eframe::CreationContext<'_>, chrome: Chrome) -> Self {
        let appearance = crate::theme::setup(cc);
        restore_egui_preferences(&cc.egui_ctx, cc.storage);
        crate::command::install_repaint(&cc.egui_ctx);
        let mut app = Self::assemble(cc.egui_ctx.clone(), appearance, chrome, FileExplorer::new(false));
        app.context.task_poller = crate::tasks::Poller::standard();
        app
    }

    /// The app over an installed theme: dock layout, panels and the main explorer.
    pub fn assemble(egui_ctx: Context, appearance: crate::theme::Appearance, chrome: Chrome, file_explorer: FileExplorer) -> Self {
        let (db_ready_tx, db_ready_rx) = crossbeam::channel::bounded(1);
        let (toast_tx, toast_rx) = crossbeam::channel::unbounded();
        let (refine_tx, refine_rx) = crossbeam::channel::unbounded();
        
        let mut tree = DockState::new(vec![TabKind::MAIN, TabKind::Logs]);

        "Undock".clone_into(&mut tree.translations.tab_context_menu.eject_button);

        // let [a, b] = tree.main_surface_mut().split_left(NodeIndex::root(), 0.3, vec!["Inspector".to_owned()]);

        let mut open_tabs = HashSet::new();

        for node in tree[SurfaceIndex::main()].iter() {
            if let Some(tabs) = node.tabs() {
                for tab in tabs {
                    open_tabs.insert(*tab);
                }
            }
        }

        let context = SmartMediaContext {
            first_run: true,
            page: Default::default(),
            db_ready_tx, db_ready_rx,
            db_ready: false,
            show_db_select_modal: false,
            db_select_mode: DbSelectMode::Local,
            db_websocket_url: "ws://localhost:8000".to_string(),
            db_local_path: crate::database::DB_DEFAULT_PATH.to_string(),
            db_connection_error: None,
            open_settings_modal: false,
            ai_initializing: false,
            ai_ready: false,
            appearance,
            prefs: None,
            egui_prefs_save_requested: false,
            pending_tab_adds: Vec::new(),
            file_explorer,
            open_tabs,
            filtered_tabs: std::collections::HashMap::new(),
            assistant: crate::ui::assistant::AssistantPanel::new(),
            refinements: RefinementsPanel::new(refine_tx.clone(), toast_tx.clone()),
            image_edit: Default::default(),
            task_poller: crate::tasks::Poller::new(),
            open_ui_settings: false,
            assistant_window_open: false,
            new_excluded_dir: String::new(),
            toasts: crate::theme::toast::style(Toasts::new().anchor(eframe::egui::Align2::RIGHT_TOP, (-10.0, 10.0))),
            toast_tx,
            toast_rx,
            refine_tx,
            refine_rx,
            active_tab: None,
            ref_old_category: String::new(),
            ref_new_category: String::new(),
            ref_old_tag: String::new(),
            ref_new_tag: String::new(),
            ref_delete_tag: String::new(),
            ref_limit_tags: 0,
        };

        Self {
            context,
            tree,
            egui_ctx,
            chrome,
        }
    }
}


impl SmartMediaContext {
    /// The explorer of the focused browser tab, or the main explorer when that tab is unknown.
    pub fn active_explorer(&mut self) -> &mut crate::ui::file_table::FileExplorer {
        explorer_for(&mut self.file_explorer, &mut self.filtered_tabs, self.active_tab.as_ref())
    }

    /// The tab the active explorer is shown in.
    pub fn active_kind(&self) -> TabKind {
        match self.active_tab {
            Some(kind @ TabKind::Browser { id }) if id == BrowserId::MAIN || self.filtered_tabs.contains_key(&id) => kind,
            _ => TabKind::MAIN,
        }
    }

    /// Key that tells the active explorer's per-tab state from the others'.
    pub fn active_owner(&self) -> String {
        match self.active_kind() {
            TabKind::Browser { id } if id != BrowserId::MAIN => format!("browser-{}", id.0),
            _ => MAIN_EXPLORER_TAB.to_string(),
        }
    }

    /// The explorer of browser tab `id`.
    pub fn explorer_mut(&mut self, id: BrowserId) -> Option<&mut crate::ui::file_table::FileExplorer> {
        if id == BrowserId::MAIN { Some(&mut self.file_explorer) } else { self.filtered_tabs.get_mut(&id) }
    }

    /// Every browser tab with the title its scope gives, the main one first.
    pub fn browser_titles(&self) -> Vec<(BrowserId, String)> {
        let mut others: Vec<(BrowserId, String)> = self.filtered_tabs.iter().map(|(id, ex)| (*id, ex.tab_title())).collect();
        others.sort_by_key(|(id, _)| *id);
        std::iter::once((BrowserId::MAIN, self.file_explorer.tab_title())).chain(others).collect()
    }

    /// The browser tab titled `title`, if there is one.
    pub fn browser_titled(&self, title: &str) -> Option<(BrowserId, &crate::ui::file_table::FileExplorer)> {
        std::iter::once((BrowserId::MAIN, &self.file_explorer)).chain(self.filtered_tabs.iter().map(|(id, ex)| (*id, ex))).find(|(_, ex)| ex.tab_title() == title)
    }
}

/// Title of the always-present main explorer tab.
pub const MAIN_EXPLORER_TAB: &str = "File Explorer";

/// The explorer shown in tab `kind`, or `main` for the main tab and unknown tabs.
pub fn explorer_for<'a>(
    main: &'a mut FileExplorer,
    tabs: &'a mut std::collections::HashMap<BrowserId, FileExplorer>,
    kind: Option<&TabKind>,
) -> &'a mut FileExplorer {
    if let Some(id) = kind.and_then(TabKind::browser_id).filter(|id| *id != BrowserId::MAIN) {
        if let Some(ex) = tabs.get_mut(&id) {
            return ex;
        }
    }
    main
}

/// True for tabs that render a file explorer.
pub fn is_explorer_tab(kind: &TabKind, tabs: &std::collections::HashMap<BrowserId, FileExplorer>) -> bool {
    kind.browser_id().is_some_and(|id| id == BrowserId::MAIN || tabs.contains_key(&id))
}

const EGUI_PREFS_KEY: &str = "ame_egui_options_v1";

/// Writes the live egui options to eframe storage.
pub fn save_egui_preferences(ctx: &Context, storage: &mut dyn eframe::Storage) {
    let options = ctx.options(|o| o.clone());
    eframe::set_value(storage, EGUI_PREFS_KEY, &options);
}

/// Applies the options written by [`save_egui_preferences`], keeping the installed theme styles.
pub fn restore_egui_preferences(ctx: &Context, storage: Option<&dyn eframe::Storage>) {
    if let Some(options) = storage.and_then(|s| eframe::get_value::<eframe::egui::Options>(s, EGUI_PREFS_KEY)) {
        crate::theme::apply_preferences(ctx, options);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marked(path: &str) -> FileExplorer {
        let mut ex = FileExplorer::new(true);
        ex.current_path = path.to_string();
        ex
    }

    #[tokio::test]
    async fn the_focused_tab_selects_its_own_explorer() {
        let mut main = marked("main");
        let mut tabs = std::collections::HashMap::new();
        let other = BrowserId::next();
        tabs.insert(other, marked("scan"));
        assert_eq!(explorer_for(&mut main, &mut tabs, Some(&TabKind::browser(other))).current_path, "scan");
        assert_eq!(explorer_for(&mut main, &mut tabs, Some(&TabKind::MAIN)).current_path, "main");
        assert_eq!(explorer_for(&mut main, &mut tabs, Some(&TabKind::browser(BrowserId::next()))).current_path, "main");
        assert_eq!(explorer_for(&mut main, &mut tabs, Some(&TabKind::Logs)).current_path, "main");
        assert_eq!(explorer_for(&mut main, &mut tabs, None).current_path, "main");
    }

    #[tokio::test]
    async fn only_browser_tabs_count_as_explorers() {
        let mut tabs = std::collections::HashMap::new();
        let other = BrowserId::next();
        tabs.insert(other, marked("scan"));
        assert!(is_explorer_tab(&TabKind::MAIN, &tabs));
        assert!(is_explorer_tab(&TabKind::browser(other), &tabs));
        assert!(!is_explorer_tab(&TabKind::browser(BrowserId::next()), &tabs));
        assert!(!is_explorer_tab(&TabKind::Logs, &tabs));
        assert!(!is_explorer_tab(&TabKind::Refine, &tabs));
    }
}
