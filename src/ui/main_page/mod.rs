use eframe::egui::*;

pub mod navbar;
pub mod file_table;

#[derive(Default)]
pub struct MainPage {
    pub file_explorer: file_table::FileExplorer,
    pub open_log_window: bool
}

impl MainPage {
    pub fn ui(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.navbar(ctx);
        CentralPanel::default()
        .show(ctx, |ui| {
            self.file_explorer.ui(ui);
        });
    }
}


