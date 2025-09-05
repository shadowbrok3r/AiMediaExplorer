use eframe::egui::*;
use egui_extras::*;

impl super::FileExplorer {
    pub fn file_list(&mut self, ui: &mut Ui) {

        let total_rows = self.files.len();
        let available_height = ui.available_height();
        let mut table = TableBuilder::new(ui)
            .striped(self.striped)
            .resizable(self.resizable)
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::auto())
            .column(
                Column::remainder()
                    .at_least(40.0)
                    .clip(true)
                    .resizable(true),
            )
            .column(Column::auto())
            .column(Column::remainder())
            .column(Column::remainder())
            .column(Column::remainder())
            .min_scrolled_height(0.0)
            .max_scroll_height(available_height);
        
        table
            .header(20.0, |mut header| {
                header.col(|ui| ui.label(""));
                header.col(|ui| {
                    Sides::new().show(
                        ui,
                        |ui| {
                            ui.strong("Name");
                        },
                        |ui| {
                            self.reversed ^= ui.button(if self.reversed { "⬆" } else { "⬇" }).clicked();
                        },
                    );
                });
                header.col(|ui| {
                    ui.strong("Path");
                });
                header.col(|ui| {
                    ui.strong("Category");
                });
                header.col(|ui| {
                    ui.strong("Modified");
                });
                header.col(|ui| {
                    ui.strong("Size");
                });
                header.col(|ui| {
                    ui.strong("Type");
                });
            })
            .body(|mut body| {
                let row_height = 48;
                body.rows((0..total_rows).map(row_height), |mut row| {
                    let row_index = if self.reversed {
                        self.num_rows - 1 - row.index()
                    } else {
                        row.index()
                    };

                    row.set_selected(self.selection.contains(&row_index));
                    row.set_overline(self.overline && row_index % 7 == 3);

                    row.col(|ui| {
                        ui.label(row_index.to_string());
                    });
                    row.col(|ui| {
                        ui.label(long_text(row_index));
                    });
                    row.col(|ui| {
                        expanding_content(ui);
                    });
                    row.col(|ui| {
                        ui.checkbox(&mut self.checked, "Click me");
                    });
                    row.col(|ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        if thick_row(row_index) {
                            ui.heading("Extra thick row");
                        } else {
                            ui.label("Normal row");
                        }
                    });

                    self.toggle_row_selection(row_index, &row.response());
                });
            });
    }
}