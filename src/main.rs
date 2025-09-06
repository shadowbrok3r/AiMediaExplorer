pub mod utilities;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui;
pub mod receive;

pub use utilities::{explorer::*, scan::*, thumbs::*, types::*, files::*};
pub use database::*;


impl eframe::App for app::SmartMediaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive(ctx);
        self.navbar(ctx);
        egui::CentralPanel::default()
        .show(ctx, |ui| {
            self.file_explorer.ui(ui);
        });
    }

    fn persist_egui_memory(&self) -> bool { true }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) { }
}

#[tokio::main]
async fn main() -> eframe::Result<()> {
    egui_logger::builder()
        .max_level(log::LevelFilter::Info) // defaults to Debug
        .init()
        .unwrap();

    let _ = eframe::run_native(
        format!("Smart Media {}", env!("CARGO_PKG_VERSION")).as_str(),
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1000.0, 750.0])
                .with_drag_and_drop(true),
                // .with_icon(load_icon()),
                // .with_always_on_top(),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(app::SmartMediaApp::new(cc)))),
    );
    
    Ok(())
}
