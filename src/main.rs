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
                    crate::ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity, similar_scores, origin_path, background: _ } => {
                        let title = uniquify_title(&self.context.open_tabs, &title);
                        if !self.context.open_tabs.contains(&title) {
                            self.tree[egui_dock::SurfaceIndex::main()].push_to_focused_leaf(title.clone());
                            self.context.open_tabs.insert(title.clone());
                        }
                        if title.starts_with("Image Edit") {
                            if let Some(origin) = origin_path.as_ref() {
                                self.context.image_edit.open_with_path(origin);
                            }
                            // No filtered tab to manage; the TabViewer renders self.context.image_edit
                        } else {
                            let mut ex = crate::ui::file_table::FileExplorer::new(true);
                            ex.set_rows(rows);
                            if showing_similarity {
                                ex.viewer.showing_similarity = true;
                                if let Some(scores) = similar_scores { ex.viewer.similar_scores = scores; }
                                if let Some(origin) = origin_path { ex.current_path = origin; }
                            }
                            self.context.filtered_tabs.insert(title.clone(), ex);
                        }
                    }
                    crate::ui::file_table::FilterRequest::OpenDatabaseAll { title, background } => {
                        let title = uniquify_title(&self.context.open_tabs, &title);
                        if !self.context.open_tabs.contains(&title) {
                            self.tree[egui_dock::SurfaceIndex::main()].push_to_focused_leaf(title.clone());
                            self.context.open_tabs.insert(title.clone());
                        }
                        let mut ex = crate::ui::file_table::FileExplorer::new(true);
                        ex.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
                        ex.load_all_database_rows();
                        self.context.filtered_tabs.insert(title.clone(), ex);
                        if background {
                            // Leave focus as-is
                        }
                    }
                    crate::ui::file_table::FilterRequest::OpenPath { title, path, recursive, background } => {
                        let title = uniquify_title(&self.context.open_tabs, &title);
                        if !self.context.open_tabs.contains(&title) {
                            self.tree[egui_dock::SurfaceIndex::main()].push_to_focused_leaf(title.clone());
                            self.context.open_tabs.insert(title.clone());
                        }
                        let mut ex = crate::ui::file_table::FileExplorer::new(true);
                        ex.init_open_path(&path, recursive);
                        self.context.filtered_tabs.insert(title.clone(), ex);
                        if background {
                            // As above, egui_dock focuses on push; no direct defocus API. Future enhancement: track previous and re-focus.
                        }
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

    fn save(&mut self, _storage: &mut dyn eframe::Storage) { 
        // storage.set_string("UiSettings", self.);
    }
}

fn uniquify_title(open_tabs: &std::collections::HashSet<String>, base: &str) -> String {
    if !open_tabs.contains(base) {
        return base.to_string();
    }
    let mut idx = 2;
    loop {
        let cand = format!("{} ({})", base, idx);
        if !open_tabs.contains(&cand) {
            return cand;
        }
        idx += 1;
        if idx > 9999 { return format!("{} ({})", base, idx); }
    }
}

#[tokio::main]
async fn main() -> eframe::Result<()> {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let _ = windows::Win32::System::Threading::SetPriorityClass(
                windows::Win32::System::Threading::GetCurrentProcess(), 
                windows::Win32::System::Threading::ABOVE_NORMAL_PRIORITY_CLASS
            );
        }
    }

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
