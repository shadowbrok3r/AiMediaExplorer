#![recursion_limit = "512"]
pub mod utilities;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui;
pub mod receive;
pub mod icons;
pub mod theme;

pub use utilities::{explorer::*, scan::*, thumbs::*, types::*};
pub use database::*;
use eframe::egui::*;

impl eframe::App for app::SmartMediaApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        theme::glass::poll_outcome(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let started = std::time::Instant::now();
        let ctx = ui.ctx().clone();
        theme::glass::frost_open_windows(&ctx);
        self.context.receive(&ctx);
        self.navbar(ui);

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
        .frame(Frame::central_panel(&ctx.global_style()).inner_margin(0.))
        .show(ui, |ui| {
            egui_dock::DockArea::new(&mut self.tree)
                .style(theme::dock_style(&ctx))
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

        theme::debug_panel::show(&ctx);
        theme::debug_panel::record_ui_time(&ctx, started.elapsed());
        theme::debug_panel::record_frame_cpu(&ctx, frame.info().cpu_usage);
    }

    fn persist_egui_memory(&self) -> bool { true }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        theme::save(&self.egui_ctx, storage);
    }

    fn on_exit(&mut self, gl: Option<&eframe::glow::Context>) {
        if let Some(gl) = gl {
            theme::glass::shutdown(gl);
        }
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

    let db_cli_args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(code) = database::cli::run(&db_cli_args) {
        std::process::exit(code);
    }

    // Theme flags: `--theme-dump` prints the resolved tokens and metrics as JSON and exits without a window, `--theme <path.json>` (or SMART_MEDIA_THEME) forces a theme file for the run, `--no-frost` disables the glass blur, `--theme-panel` opens the Theme window.
    let theme_options = match theme::cli::Options::from_env() {
        Ok(options) => options,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if theme_options.theme_dump {
        match theme::cli::dump(&theme_options) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("theme dump failed: {e:#}");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    egui_logger::builder()
        .max_level(log::LevelFilter::Info) // defaults to Debug
        .init()
        .unwrap();
    theme::cli::install(theme_options);

    let _ = eframe::run_native(
        format!("Smart Media {}", env!("CARGO_PKG_VERSION")).as_str(),
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1000.0, 750.0])
                .with_drag_and_drop(true),
            ..Default::default()
        },
        Box::new(|cc| {
            // Fire up MCP stdio server in the background for testing
            tokio::spawn(async move {
                if let Err(e) = crate::ai::mcp::serve_stdio_background().await {
                    log::warn!("[MCP] stdio server failed to start: {e}");
                }
            });
            Ok(Box::new(app::SmartMediaApp::new(cc)))
        }),
    );
    
    Ok(())
}
