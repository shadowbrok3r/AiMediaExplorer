#![recursion_limit = "512"]
pub mod utilities;
pub mod status;
pub mod command;
pub mod tasks;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui;
pub mod receive;
pub mod icons;
pub mod theme;
pub mod settings;
pub mod video;

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
        if let Some(op) = ui::tabs::read_nav_keys(&ctx) {
            command::send(command::AppCommand::Navigate { op });
        }
        self.pump_commands(&ctx);
        self.context.receive(&ctx);
        self.sync_focused_explorer();
        ui::devtools::scope::publish(&ctx, self.context.active_explorer());
        ai::mcp::snapshot::publish_frame(self);
        self.poll_tasks();
        self.title_strip(ui);
        settings::banner::show(ui);
        self.status_bar(ui);
        self.side_panels(ui);
        self.inspector_rail(ui);
        self.pump_commands(&ctx);

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
        self.shell_overlays(&ctx);
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
        let focused = self.tree.find_active_focused().map(|(_, tab)| *tab);
        if let Some(kind) = focused {
            if app::is_explorer_tab(&kind, &self.context.filtered_tabs) {
                self.context.active_tab = Some(kind);
            }
        }
    }

    /// Focuses the tab `kind` if it is open, otherwise opens it in the focused leaf.
    pub fn focus_or_open_tab(&mut self, kind: ui::tabs::TabKind) {
        ui::tabs::focus_or_open_tab(&mut self.tree, &mut self.context.open_tabs, kind);
    }

    /// Opens `kind` in the focused leaf; with `background` the previously focused tab stays in front.
    fn open_tab(&mut self, kind: ui::tabs::TabKind, background: bool) {
        ui::tabs::open_tab(&mut self.tree, &mut self.context.open_tabs, kind, background);
    }

    /// Turns dock add-button presses into new explorer tabs.
    fn process_tab_adds(&mut self) {
        for path in std::mem::take(&mut self.context.pending_tab_adds) {
            let id = ui::tabs::BrowserId::next();
            self.tree.set_focused_node_and_surface(path);
            self.tree.push_to_focused_leaf(ui::tabs::TabKind::browser(id));
            self.context.open_tabs.insert(ui::tabs::TabKind::browser(id));
            self.context.filtered_tabs.insert(id, ui::file_table::FileExplorer::new(false));
        }
    }

    /// Runs the commands queued on the bus since the last frame.
    fn pump_commands(&mut self, ctx: &egui::Context) {
        command::pump_global(self);
        if !command::global().is_empty() {
            ctx.request_repaint();
        }
    }

    /// Feeds the task registry from its sources and turns finished user tasks into toasts.
    fn poll_tasks(&mut self) {
        self.egui_ctx.request_repaint_after(tasks::IDLE_POLL);
        let probe = ui::taskbar::adapter::app_probe(&mut self.context);
        tasks::poll(&mut self.context.task_poller, &probe);
        for notice in tasks::with(|r| r.take_notices()) {
            let _ = self.context.toast_tx.try_send((notice.level.kind(), notice.text));
        }
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

    egui_logger::builder()
        .max_level(log::LevelFilter::Info) // defaults to Debug
        .init()
        .unwrap();
    ai::openai_compat::install_crypto_provider();
    if let Err(e) = ai::mcp::init(&db_cli_args) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    theme::cli::install(theme_options);

    match command::cli::startup_commands(&db_cli_args) {
        Ok(commands) => commands.into_iter().for_each(command::send),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }

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
