#![recursion_limit = "512"]
pub mod utilities;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui;
pub mod receive;
pub mod icons;
pub mod theme;
pub mod settings;

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
        self.sync_focused_explorer();
        self.title_strip(ui);
        settings::banner::show(ui);
        self.status_bar(ui);
        self.drain_open_tab_requests();
        self.open_refine_tab_if_requested();

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
        self.process_tab_adds();

        if std::mem::take(&mut self.context.egui_prefs_save_requested) {
            let saved = frame.storage_mut().map(|storage| {
                app::save_egui_preferences(&ctx, storage);
                storage.flush();
            });
            match saved {
                Some(()) => ui::dialogs::toast(egui_toast::ToastKind::Success, "Egui settings saved"),
                None => ui::dialogs::toast(egui_toast::ToastKind::Error, "Egui settings not saved: no persistent storage"),
            }
        }

        ui::devtools::show(&ctx);
        theme::debug_panel::show(&ctx);
        self.context.appearance.update(&ctx);
        self.context.appearance.show(&ctx);
        theme::debug_panel::record_ui_time(&ctx, started.elapsed());
        theme::debug_panel::record_frame_cpu(&ctx, frame.info().cpu_usage);
        self.window_edges(ui);
    }

    fn persist_egui_memory(&self) -> bool { true }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.context.appearance.save(&self.egui_ctx, storage);
        app::save_egui_preferences(&self.egui_ctx, storage);
    }

    fn on_exit(&mut self, gl: Option<&eframe::glow::Context>) {
        settings::flush_blocking(std::time::Duration::from_millis(1500));
        self.context.appearance.flush(&self.egui_ctx);
        if let Some(gl) = gl {
            theme::glass::shutdown(gl);
        }
    }
}

impl app::SmartMediaApp {
    /// Records the focused dock tab as the active explorer when it shows one.
    fn sync_focused_explorer(&mut self) {
        let focused = self.tree.find_active_focused().map(|(_, tab)| tab.clone());
        if let Some(title) = focused {
            if app::is_explorer_tab(&title, &self.context.filtered_tabs) {
                self.context.active_tab_title = Some(title);
            }
        }
    }

    /// Focuses the tab `title` if it is open, otherwise opens it in the focused leaf.
    pub fn focus_or_open_tab(&mut self, title: &str) {
        ui::tabs::focus_or_open_tab(&mut self.tree, &mut self.context.open_tabs, title);
    }

    /// Opens `title` in the focused leaf; with `background` the previously focused tab stays in front.
    fn open_tab(&mut self, title: &str, background: bool) {
        ui::tabs::open_tab(&mut self.tree, &mut self.context.open_tabs, title, background);
    }

    /// Shows the AI Refinements tab when a refine selection is waiting for it.
    fn open_refine_tab_if_requested(&mut self) {
        let waiting = !ui::refine::REFINE_SELECTION_REQUESTS.lock().unwrap().is_empty();
        if waiting {
            self.focus_or_open_tab("AI Refinements");
        }
    }

    /// Turns dock add-button presses into new explorer tabs.
    fn process_tab_adds(&mut self) {
        for path in std::mem::take(&mut self.context.pending_tab_adds) {
            let title = ui::tabs::uniquify_title(&self.context.open_tabs, "Explorer");
            self.tree.set_focused_node_and_surface(path);
            self.tree.push_to_focused_leaf(title.clone());
            self.context.open_tabs.insert(title.clone());
            self.context.filtered_tabs.insert(title, ui::file_table::FileExplorer::new(false));
        }
    }

    /// Creates the tabs requested through `OPEN_TAB_REQUESTS`.
    fn drain_open_tab_requests(&mut self) {
        let requests: Vec<_> = app::OPEN_TAB_REQUESTS.lock().unwrap().drain(..).collect();
        for req in requests {
            match req {
                ui::file_table::FilterRequest::NewTab { title, rows, showing_similarity, similar_scores, origin_path, background } => {
                    if title.starts_with("Image Edit") {
                        if let Some(origin) = origin_path.as_ref() {
                            self.context.image_edit.open_with_path(origin);
                        }
                        if self.context.open_tabs.contains(IMAGE_EDIT_TAB) {
                            if !background { self.focus_or_open_tab(IMAGE_EDIT_TAB); }
                        } else {
                            self.open_tab(IMAGE_EDIT_TAB, background);
                        }
                        continue;
                    }
                    let title = ui::tabs::uniquify_title(&self.context.open_tabs, &title);
                    self.open_tab(&title, background);
                    let mut ex = ui::file_table::FileExplorer::new(true);
                    ex.set_rows(rows);
                    if showing_similarity {
                        ex.viewer.showing_similarity = true;
                        if let Some(scores) = similar_scores { ex.viewer.similar_scores = scores; }
                        if let Some(origin) = origin_path { ex.current_path = origin; }
                    }
                    self.context.filtered_tabs.insert(title, ex);
                }
                ui::file_table::FilterRequest::OpenDatabaseAll { title, background } => {
                    let title = ui::tabs::uniquify_title(&self.context.open_tabs, &title);
                    self.open_tab(&title, background);
                    let mut ex = ui::file_table::FileExplorer::new(true);
                    ex.viewer.mode = ui::file_table::table::ExplorerMode::Database;
                    ex.load_all_database_rows();
                    self.context.filtered_tabs.insert(title, ex);
                }
                ui::file_table::FilterRequest::OpenPath { title, path, recursive, background } => {
                    let title = ui::tabs::uniquify_title(&self.context.open_tabs, &title);
                    self.open_tab(&title, background);
                    let mut ex = ui::file_table::FileExplorer::new(true);
                    ex.init_open_path(&path, recursive);
                    self.context.filtered_tabs.insert(title, ex);
                }
            }
        }
    }
}

/// Title of the single Image Edit tool tab.
const IMAGE_EDIT_TAB: &str = "Image Edit";

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

    // Theme flags: `--theme-dump` prints the resolved tokens and metrics as JSON and exits without a window, `--theme <path.json>` (or EGUI_THEME_KIT_THEME, or the old SMART_MEDIA_THEME) forces a theme file for the run, `--no-frost` (or EGUI_THEME_KIT_NO_FROST, or the old SMART_MEDIA_NO_FROST) disables the glass blur, `--theme-panel` opens the Theme window.
    let theme_options = match theme::cli_options() {
        Ok(options) => options,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    match theme::cli::handle_dump(&theme_options, &theme::font_config()) {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(e) => {
            eprintln!("theme dump failed: {e:#}");
            std::process::exit(1);
        }
    }

    // `--mcp-stdio` (or SMART_MEDIA_MCP_STDIO=1) starts the MCP stdio server; it is off otherwise.
    egui_logger::builder()
        .max_level(log::LevelFilter::Info) // defaults to Debug
        .init()
        .unwrap();
    theme::cli::install(theme_options);

    let chrome = ui::titlebar::resolve_chrome();
    let _ = eframe::run_native(
        format!("Smart Media {}", env!("CARGO_PKG_VERSION")).as_str(),
        eframe::NativeOptions {
            viewport: ui::titlebar::viewport(&chrome),
            ..Default::default()
        },
        Box::new(move |cc| {
            theme::title_bar::windows::client_area::install(cc);
            Ok(Box::new(app::SmartMediaApp::new(cc, chrome)))
        }),
    );
    
    Ok(())
}
