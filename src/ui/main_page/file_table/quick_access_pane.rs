use std::path::PathBuf;

use eframe::egui::*;
use humansize::{format_size, DECIMAL};

use crate::{list_drive_infos, quick_access};

impl super::FileExplorer {
    pub fn quick_access_pane(&mut self, ui: &mut Ui) {
        SidePanel::left("MainPageLeftPanel")
        .max_width(400.)
        .min_width(200.)
        .show_animated_inside(ui, self.open_quick_access, |ui| {
            ui.collapsing("Quick Access", |ui| {
                for access in quick_access().iter() {
                    if Button::new(format!("{} {}", access.icon, access.label)).min_size(vec2(ui.available_width(), 20.)).ui(ui).clicked() {
                        self.current_path = access.path.to_string_lossy().to_string();
                        self.populate_current_directory();
                    }
                }
            });

            ui.add_space(10.);
            
            ui.collapsing("Drives", |ui| {
                for drive in list_drive_infos() {
                    let root = drive.root.clone();
                    let display = format!("{} - {}", drive.drive_type, drive.root);
                    let path = PathBuf::from(root.clone());
                    let free = format_size(drive.free, DECIMAL);
                    let total = format_size(drive.total, DECIMAL);

                    let response = Button::new(display)
                    .min_size(vec2(ui.available_width(), 45.))
                    .right_text(drive.label)
                    .ui(ui).on_hover_text(format!("{free} free of {total}"));


                    if response.clicked() {
                        self.current_path = path.to_string_lossy().to_string();
                        self.populate_current_directory();
                    }
                }
            });
        });
    }
}