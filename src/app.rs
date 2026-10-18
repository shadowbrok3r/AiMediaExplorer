use eframe::egui::Context;
use crossbeam::channel::{Receiver, Sender};
use egui_dock::{DockState, NodePath, SurfaceIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use crate::{ui::{file_table::FileExplorer, refine::RefinementsPanel}, UiSettings};
use egui_toast::Toasts;

// Global atomic flag to request opening the settings modal from anywhere (e.g., navbar without direct &mut SmartMediaApp access)
pub static OPEN_SETTINGS_REQUEST: Lazy<std::sync::atomic::AtomicBool> = Lazy::new(|| std::sync::atomic::AtomicBool::new(false));
// Global queue for dynamic tab open requests
pub static OPEN_TAB_REQUESTS: Lazy<Mutex<Vec<crate::ui::file_table::FilterRequest>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub const DEFAULT_JOYCAPTION_PATH: &str = r#"G:\Users\Owner\Documents\1-MyStuff\TechStuff\GitHub\smart-media\llama-joycaption-beta-one-hf-llava"#;
pub const MAX_NEW_TOKENS: usize = 200;
pub const TEMPERATURE: f32 = 0.5;

pub struct SmartMediaApp {
    pub tree: DockState<String>,
    pub context: SmartMediaContext,
    pub egui_ctx: Context,
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
    pub open_tabs: HashSet<String>,
    // Map of dynamic tab title -> a dedicated FileExplorer instance with filters applied
    pub filtered_tabs: std::collections::HashMap<String, crate::ui::file_table::FileExplorer>,
    pub assistant: crate::ui::assistant::AssistantPanel,
    pub refinements: crate::ui::refine::RefinementsPanel,
    pub image_edit: crate::ui::image_edit::ImageEditPanel,
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
    // Title of the focused explorer tab (updated each frame from the dock state)
    pub active_tab_title: Option<String>,
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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let appearance = crate::theme::setup(cc);
        restore_egui_preferences(&cc.egui_ctx, cc.storage);
        let (db_ready_tx, db_ready_rx) = crossbeam::channel::bounded(1);
        let (toast_tx, toast_rx) = crossbeam::channel::unbounded();
        let (refine_tx, refine_rx) = crossbeam::channel::unbounded();
        
        let mut tree = DockState::new(vec![
            "File Explorer".to_owned(),
            "Logs".to_owned(),
        ]);

        "Undock".clone_into(&mut tree.translations.tab_context_menu.eject_button);

        // let [a, b] = tree.main_surface_mut().split_left(NodeIndex::root(), 0.3, vec!["Inspector".to_owned()]);

        let mut open_tabs = HashSet::new();

        for node in tree[SurfaceIndex::main()].iter() {
            if let Some(tabs) = node.tabs() {
                for tab in tabs {
                    open_tabs.insert(tab.clone());
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
            file_explorer: FileExplorer::new(false),
            open_tabs,
            filtered_tabs: std::collections::HashMap::new(),
            assistant: crate::ui::assistant::AssistantPanel::new(),
            refinements: RefinementsPanel::new(refine_tx.clone(), toast_tx.clone()),
            image_edit: Default::default(),
            open_ui_settings: false,
            assistant_window_open: false,
            new_excluded_dir: String::new(),
            toasts: crate::theme::toast::style(Toasts::new().anchor(eframe::egui::Align2::RIGHT_TOP, (-10.0, 10.0))),
            toast_tx,
            toast_rx,
            refine_tx,
            refine_rx,
            active_tab_title: None,
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
            egui_ctx: cc.egui_ctx.clone(),
        }
    }
}


impl SmartMediaContext {
    /// The explorer of the focused explorer tab, or the main explorer when that tab is unknown.
    pub fn active_explorer(&mut self) -> &mut crate::ui::file_table::FileExplorer {
        explorer_for(&mut self.file_explorer, &mut self.filtered_tabs, self.active_tab_title.as_deref())
    }

    pub fn open_image_edit_with_path(&mut self, path: &str, tree: &mut egui_dock::DockState<String>) {
        // Ensure the Image Edit tab exists and is focused
        let title = "Image Edit".to_string();
        if !self.open_tabs.contains(&title) {
            tree[egui_dock::SurfaceIndex::main()].push_to_focused_leaf(title.clone());
            self.open_tabs.insert(title.clone());
        }
        self.image_edit.open_with_path(path);
    }
}

/// Title of the always-present main explorer tab.
pub const MAIN_EXPLORER_TAB: &str = "File Explorer";

/// The explorer shown in tab `title`, or `main` for the main tab and unknown titles.
pub fn explorer_for<'a>(
    main: &'a mut FileExplorer,
    tabs: &'a mut std::collections::HashMap<String, FileExplorer>,
    title: Option<&str>,
) -> &'a mut FileExplorer {
    if let Some(title) = title.filter(|t| *t != MAIN_EXPLORER_TAB) {
        if let Some(ex) = tabs.get_mut(title) {
            return ex;
        }
    }
    main
}

/// True for tab titles that render a file explorer.
pub fn is_explorer_tab(title: &str, tabs: &std::collections::HashMap<String, FileExplorer>) -> bool {
    title == MAIN_EXPLORER_TAB || tabs.contains_key(title)
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
        tabs.insert("Scan: x".to_string(), marked("scan"));
        assert_eq!(explorer_for(&mut main, &mut tabs, Some("Scan: x")).current_path, "scan");
        assert_eq!(explorer_for(&mut main, &mut tabs, Some(MAIN_EXPLORER_TAB)).current_path, "main");
        assert_eq!(explorer_for(&mut main, &mut tabs, Some("closed tab")).current_path, "main");
        assert_eq!(explorer_for(&mut main, &mut tabs, None).current_path, "main");
    }

    #[tokio::test]
    async fn only_explorer_tabs_count_as_explorers() {
        let mut tabs = std::collections::HashMap::new();
        tabs.insert("Scan: x".to_string(), marked("scan"));
        assert!(is_explorer_tab(MAIN_EXPLORER_TAB, &tabs));
        assert!(is_explorer_tab("Scan: x", &tabs));
        assert!(!is_explorer_tab("Logs", &tabs));
        assert!(!is_explorer_tab("AI Refinements", &tabs));
    }
}
