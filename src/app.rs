use eframe::{egui::{Context, FontData, FontDefinitions, FontFamily}};
use crossbeam::channel::{Receiver, Sender};
use egui_dock::{DockState, SurfaceIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use crate::{ui::file_table::FileExplorer, UiSettings};
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
    pub open_ui_settings: bool,
    // UI state for adding excluded directories
    pub new_excluded_dir: String,
    // Toasts manager and channel for async notifications
    pub toasts: Toasts,
    pub toast_tx: Sender<(egui_toast::ToastKind, String)>,
    pub toast_rx: Receiver<(egui_toast::ToastKind, String)>,
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
        
        let mut tree = DockState::new(vec![
            "File Explorer".to_owned(),
            "Logs".to_owned()
        ]);

        "Undock".clone_into(&mut tree.translations.tab_context_menu.eject_button);

        // let [a, b] = tree.main_surface_mut().split_left(
        //     NodeIndex::root(),
        //     0.3,
        //     vec![
        //         "Inspector".to_owned()
        //     ],
        // );

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
            open_ui_settings: false,
            new_excluded_dir: String::new(),
            toasts: Toasts::new().anchor(eframe::egui::Align2::RIGHT_TOP, (-10.0, 10.0)),
            toast_tx,
            toast_rx,
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