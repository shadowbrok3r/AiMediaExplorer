use eframe::{egui::{Context, FontData, FontDefinitions, FontFamily}};
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use crate::UiSettings;

// Global atomic flag to request opening the settings modal from anywhere (e.g., navbar without direct &mut SmartMediaApp access)
pub static OPEN_SETTINGS_REQUEST: Lazy<std::sync::atomic::AtomicBool> = Lazy::new(|| std::sync::atomic::AtomicBool::new(false));
pub const DEFAULT_JOYCAPTION_PATH: &str = r#"G:\Users\Owner\Desktop\llama-joycaption-beta-one-hf-llava"#;
pub const MAX_NEW_TOKENS: usize = 200;
pub const TEMPERATURE: f32 = 0.5;

pub struct SmartMediaApp {
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
    pub open_log_window: bool
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
        Self {
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
            file_explorer: Default::default(),
            open_log_window: Default::default(),
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
        .insert(0, "UbuntuMonoNerdFont".to_owned());

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