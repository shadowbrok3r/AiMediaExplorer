pub mod utilities;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui;
pub mod receive;

pub use utilities::{explorer::*, scan::*, thumbs::*, types::*};
pub use database::*;
use eframe::egui::*;

impl eframe::App for app::SmartMediaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.context.receive(ctx);
        self.navbar(ctx);

        // Create any requested dynamic tabs (tags/categories)
        {
            let mut q = app::OPEN_TAB_REQUESTS.lock().unwrap();
            for req in q.drain(..) {
                match req {
                    crate::ui::file_table::FilterRequest::NewTab { title, rows } => {
                        if !self.context.open_tabs.contains(&title) {
                            self.tree[egui_dock::SurfaceIndex::main()].push_to_focused_leaf(title.clone());
                            self.context.open_tabs.insert(title.clone());
                        }
                        let mut ex = crate::ui::file_table::FileExplorer::default();
                        ex.set_rows(rows);
                        self.context.filtered_tabs.insert(title, ex);
                    }
                }
            }
        }

        CentralPanel::default()
        .frame(Frame::central_panel(&ctx.style()).inner_margin(0.))
        .show(ctx, |ui| {
            egui_dock::DockArea::new(&mut self.tree)
                .style(egui_dock::Style::from_egui(ui.style()))
                .show_close_buttons(true)
                .show_add_buttons(true)
                .draggable_tabs(true)
                .show_tab_name_on_hover(true)
                .allowed_splits(egui_dock::AllowedSplits::All)
                .show_leaf_close_all_buttons(true)
                .show_leaf_collapse_buttons(true)
                .show_secondary_button_hint(true)
                .secondary_button_on_modifier(true)
                .secondary_button_context_menu(true)
                .show_inside(ui, &mut self.context);
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
