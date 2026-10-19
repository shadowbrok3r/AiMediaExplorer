use crate::command::{self, AppCommand, Scope, Target, Tool};
use crate::icons::{self, Icon, svg};
use crate::theme::{chrome, controls::MenuButton};
use crate::ui::shell::ShellAction;
use crate::ui::tabs::TabKind;
use eframe::egui::*;
use egui_dock::SurfaceIndex;

/// Menu entry with an icon and an optional shortcut hint.
fn item(ui: &mut Ui, icon: Icon, text: &str, shortcut: Option<&str>) -> Response {
    let mut button = icons::button_widget(ui, icon, text);
    if let Some(shortcut) = shortcut {
        button = button.shortcut_text(shortcut);
    }
    ui.add(button)
}

/// Menu entry that shows whether its option is on.
fn toggle_item(ui: &mut Ui, icon: Icon, on: bool, text: &str, shortcut: Option<&str>) -> Response {
    let size = icons::IconSize::Md.px(ui);
    let mut button = Button::selectable(on, (icon.image(size), WidgetText::from(text))).image_tint_follows_text_color(true);
    if let Some(shortcut) = shortcut {
        button = button.shortcut_text(shortcut);
    }
    ui.add(button)
}

impl crate::app::SmartMediaApp {
    /// File menu button for the title strip.
    pub fn file_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("File").show(ui, |ui| self.file_menu_items(ui));
    }

    /// View menu button for the title strip.
    pub fn view_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("View").show(ui, |ui| self.view_menu_items(ui));
    }

    /// Tools menu button for the title strip.
    pub fn tools_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("Tools").show(ui, |ui| self.tools_menu_items(ui));
    }

    /// Help menu button for the title strip.
    pub fn help_menu(&mut self, ui: &mut Ui) {
        let _ = MenuButton::new("Help").show(ui, |ui| self.help_menu_items(ui));
    }

    /// Menu entry that runs a shell action and closes the menu.
    fn shell_item(&mut self, ui: &mut Ui, action: ShellAction, icon: Icon, shortcut: Option<&str>) {
        if item(ui, icon, action.label(), shortcut).clicked() {
            self.run_shell_action(ui.ctx(), action);
            ui.close();
        }
    }

    fn file_menu_items(&mut self, ui: &mut Ui) {
        self.shell_item(ui, ShellAction::OpenFolder, svg::FOLDER_OPEN, None);
        self.shell_item(ui, ShellAction::ScanFolder, svg::AME_SCAN, None);
        ui.separator();
        icons::menu_button(ui, svg::AME_DATABASE, "Database", |ui| {
            if item(ui, svg::AME_THUMBNAIL_GRID, "View Entire Database", None).on_hover_text("Show all thumbnails from all directories").clicked() {
                command::send(AppCommand::OpenScope { scope: Scope::Library, target: Target::NewTab, title: Some("Entire Database".to_string()), recursive: false });
                ui.close();
            }
            ui.separator();
            self.shell_item(ui, ShellAction::DatabaseMaintenance, svg::AME_DATABASE, None);
            self.shell_item(ui, ShellAction::ExportBackup, svg::AME_DB_EXPORT, None);
            self.shell_item(ui, ShellAction::ImportBackup, svg::AME_DB_IMPORT, None);
            self.shell_item(ui, ShellAction::ChooseDbFolder, svg::FOLDER_OPEN, None);
        });
        ui.separator();
        if item(ui, svg::GEAR, "Preferences", Some("Ctrl+,")).clicked() {
            command::send(AppCommand::ShowSettings { section: None });
            ui.close();
        }
        ui.separator();
        self.shell_item(ui, ShellAction::Quit, svg::POWER, None);
    }

    fn view_menu_items(&mut self, ui: &mut Ui) {
        let layout = &self.shell.layout.layout;
        for (action, icon, on, label, shortcut) in [
            (ShellAction::ToggleSidebar, svg::SIDEBAR_SIMPLE, layout.sidebar_open, "Sidebar", Some("Ctrl+B")),
            (ShellAction::ToggleBottomBar, svg::LIST_CHECKS, layout.bottom_bar_open, "Bottom bar", None),
            (ShellAction::ToggleRightRail, svg::COLUMNS, self.context.inspector.is_open(), "Right rail", Some("Ctrl+I")),
        ] {
            if toggle_item(ui, icon, on, label, shortcut).clicked() {
                self.run_shell_action(ui.ctx(), action);
            }
        }
        ui.separator();
        if item(ui, svg::PALETTE, "Appearance…", None).on_hover_text("Theme presets, colors, density and text size").clicked() {
            command::send(AppCommand::ShowSettings { section: Some(crate::command::Section::Appearance) });
            ui.close();
        }
        let _ = chrome::toggle_row(ui, &mut self.chrome, "Custom title bar");
        ui.separator();

        icons::menu_button(ui, svg::TABS, "Tabs", |ui| {
            for tab in TabKind::TOGGLED {
                if icons::selectable_button(ui, tab_icon(&tab), self.context.open_tabs.contains(&tab), tab.menu_label()).clicked() {
                    if let Some(index) = self.tree.find_tab(&tab) {
                        self.tree.remove_tab(index);
                        self.context.open_tabs.remove(&tab);
                    } else {
                        self.tree[SurfaceIndex::main()].push_to_focused_leaf(tab);
                        self.context.open_tabs.insert(tab);
                    }
                    ui.close();
                }
            }
        });
        let mut was_open = self.context.assistant_window_open;
        if icons::checkbox(ui, &mut was_open, svg::AME_ASSISTANT, "AI Assistant Window").clicked() {
            self.context.assistant_window_open = was_open;
        }
        ui.separator();
        self.devtools_menu(ui);
    }

    fn devtools_menu(&mut self, ui: &mut Ui) {
        icons::menu_button(ui, svg::WRENCH, "Devtools", |ui| {
            if icons::button(ui, svg::AME_LOGO, "Icon gallery").clicked() {
                icons::gallery::toggle(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::SLIDERS, "Theme panel").clicked() {
                crate::theme::debug_panel::toggle(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::DATABASE, "Settings store").clicked() {
                crate::ui::devtools::toggle_settings_store(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::SPARKLE, "Wizard").clicked() {
                crate::ui::devtools::toggle_wizard(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_DATABASE, "Database maintenance").clicked() {
                crate::ui::devtools::toggle_db_maintenance(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::FILE_ARROW_DOWN, "Legacy import").clicked() {
                crate::ui::devtools::toggle_legacy_import(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_THUMBNAIL_GRID, "Thumbnail storage").clicked() {
                crate::ui::devtools::toggle_thumb_storage(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::IMAGE, "Thumbnail cache").clicked() {
                crate::ui::devtools::thumb_cache::toggle(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::LIST_CHECKS, "Tasks").clicked() {
                crate::ui::devtools::toggle_tasks(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::TERMINAL_WINDOW, "Command log").clicked() {
                crate::ui::devtools::toggle_command_log(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::GAUGE, "Perf").clicked() {
                crate::ui::devtools::toggle_perf(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_ASSISTANT, "ZeroClaw").clicked() {
                crate::ui::devtools::toggle_zeroclaw(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::LIST_CHECKS, "ZeroClaw jobs").clicked() {
                crate::ui::devtools::toggle_zc_jobs(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::STACK, "Row store").clicked() {
                crate::ui::devtools::toggle_row_store(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_LOGICAL_GROUP, "Collections").clicked() {
                crate::ui::devtools::toggle_collections(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::MAGNIFYING_GLASS, "Query").clicked() {
                crate::ui::devtools::toggle_query(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::LIST_CHECKS, "Browser actions").clicked() {
                crate::ui::devtools::toggle_browser_actions(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::VIDEO_CAMERA, "Video").clicked() {
                crate::ui::devtools::toggle_video(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::SIDEBAR_SIMPLE, "Sidebar").clicked() {
                crate::ui::devtools::toggle_sidebar(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::CLOCK_COUNTER_CLOCKWISE, "Journal").clicked() {
                crate::ui::devtools::toggle_journal(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::TABS, "Workspace").clicked() {
                crate::ui::devtools::toggle_workspace(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_REFINE, "Refine and Image Edit").clicked() {
                crate::ui::devtools::toggle_refine(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::LIST_BULLETS, "Actions").clicked() {
                crate::ui::devtools::toggle_actions(ui.ctx());
                ui.close();
            }
            if icons::button(ui, svg::AME_MCP, "MCP server").clicked() {
                crate::ui::devtools::toggle_mcp(ui.ctx());
                ui.close();
            }
        });
    }

    fn tools_menu_items(&mut self, ui: &mut Ui) {
        if item(ui, svg::ARROW_COUNTER_CLOCKWISE, "Undo metadata change", Some("Ctrl+Z")).clicked() {
            command::send(AppCommand::Undo);
            ui.close();
        }
        if item(ui, svg::ARROW_CLOCKWISE, "Redo metadata change", Some("Ctrl+Shift+Z")).clicked() {
            command::send(AppCommand::Redo);
            ui.close();
        }
        ui.separator();
        self.shell_item(ui, ShellAction::ScanFolder, svg::AME_SCAN, None);
        if item(ui, svg::AME_MODEL, "Process files...", None).on_hover_text("Describe or embed the selection, this view or a folder").clicked() {
            command::send(AppCommand::Dialog { dialog: command::DialogKind::Process, folder: None });
            ui.close();
        }
        icons::menu_button(ui, svg::AME_MODEL, "AI processing", |ui| {
            self.shell_item(ui, ShellAction::EnableAiSearch, svg::AME_SEMANTIC_SEARCH, None);
            self.shell_item(ui, ShellAction::EnableAutoIndexing, svg::AME_AI_DESCRIBE, None);
            self.shell_item(ui, ShellAction::EnableAutoClip, svg::AME_CLIP_EMBED, None);
            self.shell_item(ui, ShellAction::ClipMissing, svg::AME_EMBEDDING, None);
            self.shell_item(ui, ShellAction::ClipSelected, svg::CURSOR_CLICK, None);
            self.shell_item(ui, ShellAction::ClipMissingCount, svg::GAUGE, None);
        });
        if item(ui, svg::AME_REFINE, "Refine (AI, DB-only)", None).on_hover_text("Open AI Refinements panel to generate proposals from the database").clicked() {
            command::send(AppCommand::OpenTool { tool: Tool::Refinements, path: None, target: Target::NewTab });
            ui.close();
        }
        if item(ui, svg::AME_IMAGE_EDIT, "Image edit", None).clicked() {
            command::send(AppCommand::OpenTool { tool: Tool::ImageEdit, path: None, target: Target::NewTab });
            ui.close();
        }
        icons::menu_button(ui, svg::AME_LOGICAL_GROUP, "Collections", |ui| {
            self.shell_item(ui, ShellAction::ShowCollections, svg::AME_LOGICAL_GROUP, None);
            self.shell_item(ui, ShellAction::NewCollection, svg::FOLDER_PLUS, None);
        });
        ui.separator();
        icons::menu_button(ui, svg::SLIDERS, "Scan settings", |ui| scan_settings(ui));
    }

    fn help_menu_items(&mut self, ui: &mut Ui) {
        if item(ui, svg::TERMINAL_WINDOW, "Logs", None).clicked() {
            command::send(AppCommand::OpenTool { tool: Tool::Logs, path: None, target: Target::NewTab });
            ui.close();
        }
        if item(ui, svg::MAGNIFYING_GLASS, "Command palette", Some("Ctrl+K")).clicked() {
            command::send(AppCommand::OpenPalette { query: None });
            ui.close();
        }
        ui.separator();
        self.shell_item(ui, ShellAction::ShowAbout, svg::INFO, None);
    }

    /// Icon gallery window and the bottom status bar of the focused explorer.
    pub fn status_bar(&mut self, ui: &mut Ui) {
        icons::gallery::show_window(ui.ctx());
        if !self.shell.layout.layout.bottom_bar_open {
            return;
        }

        let owner = self.context.active_owner();
        crate::ui::taskbar::show(ui, self.context.active_explorer(), &owner);
    }
}

/// Batch size of the next scan, saved with the settings.
fn scan_settings(ui: &mut Ui) {
    let mut batch = crate::settings::store().read(|s| s.scan_found_batch_max).unwrap_or(128);
    ui.horizontal(|ui| {
        ui.label("FoundBatch size");
        if ui.add(DragValue::new(&mut batch).range(16..=4096)).on_hover_text("Rows per scan batch, used by the next scan").changed() {
            crate::settings::update("tools:scan_batch", |s| s.scan_found_batch_max = Some(batch));
        }
    });
}

/// Icon of a dockable tab.
fn tab_icon(tab: &TabKind) -> icons::Icon {
    match tab {
        TabKind::Browser { .. } => svg::FOLDER_OPEN,
        TabKind::Logs => svg::TERMINAL_WINDOW,
        TabKind::Refine => svg::AME_REFINE,
        _ => svg::TABS,
    }
}
