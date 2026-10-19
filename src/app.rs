use eframe::egui::Context;
use crossbeam::channel::{Receiver, Sender};
use egui_dock::{DockState, NodePath, SurfaceIndex};
use std::collections::HashSet;
use crate::ui::{file_table::FileExplorer, refine::RefinementsPanel, tabs::{BrowserId, TabKind}};
use egui_toast::Toasts;
use crate::theme::chrome::Chrome;

pub const DEFAULT_JOYCAPTION_PATH: &str = r#"G:\Users\Owner\Documents\1-MyStuff\TechStuff\GitHub\smart-media\llama-joycaption-beta-one-hf-llava"#;
pub const MAX_NEW_TOKENS: usize = 200;
pub const TEMPERATURE: f32 = 0.5;

pub struct SmartMediaApp {
    pub tree: DockState<TabKind>,
    pub context: SmartMediaContext,
    pub egui_ctx: Context,
    pub chrome: Chrome,
    pub shell: crate::ui::shell::Shell,
}

pub struct SmartMediaContext {
    pub first_run: bool,
    pub db_ready_tx: Sender<()>,
    pub db_ready_rx: Receiver<()>,
    pub db_ready: bool,
    // First-run wizard, shown while no library is configured
    pub wizard: crate::ui::wizard::Wizard,
    // Workspace file, launch restore and the state dump
    pub workspace: crate::ui::workspace::Session,
    // AI init status flags
    pub ai_initializing: bool,
    pub ai_ready: bool,
    // Theme picker window, its database store and the theme sync
    pub appearance: crate::theme::Appearance,
    // Settings tab state, kept while the tab is closed
    pub settings_page: crate::ui::settings::SettingsPage,
    // Set by the Appearance section's Save button; handled once the frame storage is reachable
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
    // Right rail bound to the focused browser
    pub inspector: crate::ui::inspector::Inspector,
    // Watchers that feed the task registry
    pub task_poller: crate::tasks::Poller,
    // Toggle for showing the AI Assistant as a separate viewport window
    pub assistant_window_open: bool,
    // Toasts manager and channel for async notifications
    pub toasts: Toasts,
    pub toast_tx: Sender<(egui_toast::ToastKind, String)>,
    pub toast_rx: Receiver<(egui_toast::ToastKind, String)>,
    // The focused browser tab (updated each frame from the dock state)
    pub active_tab: Option<TabKind>,
}

impl SmartMediaApp {
    pub fn new(cc: &eframe::CreationContext<'_>, chrome: Chrome) -> Self {
        let appearance = crate::theme::setup(cc);
        restore_egui_preferences(&cc.egui_ctx, cc.storage);
        crate::command::install_repaint(&cc.egui_ctx);
        let launch = crate::ui::workspace::launch();
        let mut app = Self::assemble(cc.egui_ctx.clone(), appearance, chrome, FileExplorer::new(launch.restoring()));
        app.context.task_poller = crate::tasks::Poller::standard();
        app.context.inspector.load_prefs();
        app.shell = crate::ui::shell::Shell::live();
        app.context.inspector.set_open(app.shell.layout.layout.right_rail_open);
        let applied = crate::ui::workspace::restore_launch(&mut app, launch);
        app.context.workspace = crate::ui::workspace::Session::live(launch, applied.as_ref());
        app
    }

    /// The app over an installed theme: dock layout, panels and the main explorer.
    pub fn assemble(egui_ctx: Context, appearance: crate::theme::Appearance, chrome: Chrome, file_explorer: FileExplorer) -> Self {
        let (db_ready_tx, db_ready_rx) = crossbeam::channel::bounded(1);
        let (toast_tx, toast_rx) = crossbeam::channel::unbounded();
        
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
            db_ready_tx, db_ready_rx,
            db_ready: false,
            wizard: Default::default(),
            workspace: crate::ui::workspace::Session::memory(),
            ai_initializing: false,
            ai_ready: false,
            appearance,
            settings_page: Default::default(),
            egui_prefs_save_requested: false,
            pending_tab_adds: Vec::new(),
            file_explorer,
            open_tabs,
            filtered_tabs: std::collections::HashMap::new(),
            assistant: crate::ui::assistant::AssistantPanel::new(),
            refinements: RefinementsPanel::new(toast_tx.clone()),
            image_edit: Default::default(),
            inspector: Default::default(),
            task_poller: crate::tasks::Poller::new(),
            assistant_window_open: false,
            toasts: crate::theme::toast::style(Toasts::new().anchor(eframe::egui::Align2::RIGHT_TOP, (-10.0, 10.0))),
            toast_tx,
            toast_rx,
            active_tab: None,
        };

        Self {
            context,
            tree,
            egui_ctx,
            chrome,
            shell: Default::default(),
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
