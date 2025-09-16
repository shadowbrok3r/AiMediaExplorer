use eframe::egui::*;

impl crate::ui::file_table::FileExplorer {
    pub fn table_menu(&mut self, ui: &mut Ui) {
        let btn_txt = match self.viewer.mode {
            crate::ui::file_table::table::ExplorerMode::FileSystem => ("Database Mode", "Current: FileSystem Mode"),
            crate::ui::file_table::table::ExplorerMode::Database => ("FileSystem Mode", "Current: Database Mode"),
        };
        if Button::new(btn_txt.0).ui(ui).on_hover_text(btn_txt.1).clicked() {
            match self.viewer.mode {
                crate::ui::file_table::table::ExplorerMode::FileSystem => {
                    self.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
                    // Open Entire Database in a new tab (do not hijack current FS view)
                    crate::app::OPEN_TAB_REQUESTS
                    .lock()
                    .unwrap()
                    .push(crate::ui::file_table::FilterRequest::OpenDatabaseAll { 
                        title: "Entire Database".to_string(), 
                        background: false 
                    });
                },
                crate::ui::file_table::table::ExplorerMode::Database => {
                    self.viewer.mode = crate::ui::file_table::table::ExplorerMode::FileSystem;
                    self.populate_current_directory();
                },
            }
        }
        ui.add_space(4.0);
        if matches!(self.viewer.mode, crate::ui::file_table::table::ExplorerMode::Database) {
            ui.label(format!("Loaded Rows: {} (offset {})", self.table.len(), self.db_offset));
            if self.db_loading { ui.colored_label(Color32::YELLOW, "Loading..."); }
        }
    }
}