use eframe::{egui::{Context, FontData, FontDefinitions, FontFamily}};
use crossbeam::channel::{Receiver, Sender};
use egui_dock::{DockState, SurfaceIndex};
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
pub const DEFAULT_JOYCAPTION_PATH: &str = r#"G:\Users\Owner\Desktop\llama-joycaption-beta-one-hf-llava"#;
pub const MAX_NEW_TOKENS: usize = 200;
pub const TEMPERATURE: f32 = 0.5;

pub struct SmartMediaApp {
    pub tree: DockState<String>,
    pub context: SmartMediaContext,
}

pub struct SmartMediaContext {
    pub first_run: bool,
    pub page: Page,
    pub ui_settings: UiSettings,
    pub ui_settings_tx: Sender<UiSettings>,
    pub ui_settings_rx: Receiver<UiSettings>,
    pub db_ready_tx: Sender<()>,
    pub db_ready_rx: Receiver<()>,
    pub db_ready: bool,
    // UI State
    pub open_settings_modal: bool,
    // AI init status flags
    pub ai_initializing: bool,
    pub ai_ready: bool,
    // Draft copy of settings while editing in modal
    pub settings_draft: Option<UiSettings>,
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
    // Currently focused tab title (updated by TabViewer::ui)
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

impl SmartMediaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        setup_custom_fonts(&cc.egui_ctx);
        let (ui_settings_tx, ui_settings_rx) = crossbeam::channel::bounded(1);
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
            ui_settings: UiSettings::default(),
            ui_settings_tx, ui_settings_rx,
            db_ready_tx, db_ready_rx,
            db_ready: false,
            open_settings_modal: false,
            ai_initializing: false,
            ai_ready: false,
            settings_draft: None,
            file_explorer: FileExplorer::new(false),
            open_tabs,
            filtered_tabs: std::collections::HashMap::new(),
            assistant: crate::ui::assistant::AssistantPanel::new(),
            refinements: RefinementsPanel::new(refine_tx.clone(), toast_tx.clone()),
            image_edit: Default::default(),
            open_ui_settings: false,
            assistant_window_open: false,
            new_excluded_dir: String::new(),
            toasts: Toasts::new().anchor(eframe::egui::Align2::RIGHT_TOP, (-10.0, 10.0)),
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
        }
    }
}


fn setup_custom_fonts(ctx: &Context) {
    // Start with the default fonts (we will be adding to them rather than replacing them).
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "Monaspace".to_owned(),
        std::sync::Arc::new(
            FontData::from_static(include_bytes!("../assets/fonts/MonaspaceNeon-Regular.otf"))
        ),
    );

    fonts.font_data.insert(
        "UbuntuSansMono".to_owned(),
        std::sync::Arc::new(
            FontData::from_static(include_bytes!("../assets/fonts/UbuntuSansMono-Regular.otf"))
        ),
    );

    fonts.font_data.insert(
        "UbuntuMonoNerdFont".to_owned(),
        std::sync::Arc::new(
            FontData::from_static(include_bytes!("../assets/fonts/UbuntuMonoNerdFont-Regular.ttf"))
        ),
    ); 

    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "UbuntuSansMono".to_owned());

    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(1, "UbuntuMonoNerdFont".to_owned());

    fonts.font_data.insert(
        "Regular".to_owned(),
        std::sync::Arc::new(
            FontData::from_static(include_bytes!("../assets/fonts/MonaspaceNeon-Regular.otf"))
        ),
    );
    fonts.families.insert(
        FontFamily::Name("Regular".into()),
        vec!["Regular".to_owned()],
    );
    fonts.font_data.insert(
        "Bold".to_owned(),
        std::sync::Arc::new(
            FontData::from_static(include_bytes!("../assets/fonts/MonaspaceNeon-Bold.otf"))
        ),
    );
    fonts
        .families
        .insert(FontFamily::Name("Bold".into()), vec!["Bold".to_owned()]);
    // Tell egui to use these fonts:
    ctx.set_fonts(fonts);
}

impl SmartMediaContext {
    /// Returns a mutable reference to the FileExplorer corresponding to the currently active tab.
    /// Falls back to the main file_explorer if the active tab is unknown.
    pub fn active_explorer(&mut self) -> &mut crate::ui::file_table::FileExplorer {
        if let Some(title) = self.active_tab_title.clone() {
            if title == "File Explorer" {
                return &mut self.file_explorer;
            }
            if let Some(ex) = self.filtered_tabs.get_mut(&title) {
                return ex;
            }
        }
        &mut self.file_explorer
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