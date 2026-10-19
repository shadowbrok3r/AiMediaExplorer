use egui::containers::menu::MenuConfig;
use eframe::egui::*;

use crate::icons::{self, svg};

pub mod breadcrumbs;
pub mod db_delete;
pub mod logical_groups;
pub mod selection_menu;
pub mod filter_menu;
pub mod scan_menu;

/// Width reserved for the right-hand navbar controls, search field aside, before their measured width is known.
const RIGHT_FIXED_ESTIMATE: f32 = 270.;
const PATH_MIN_WIDTH: f32 = 120.;
const PATH_MAX_WIDTH: f32 = 400.;
const SEARCH_MIN_WIDTH: f32 = 80.;
const SEARCH_PREFERRED_WIDTH: f32 = 200.;

impl super::FileExplorer {
    pub fn navbar(&mut self, ui: &mut Ui) {
        Panel::top("file_explorer_navbar")
        .exact_size(28.)
        .show_separator_line(false)
        .show(ui, |ui| {
        MenuBar::new()
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .ui(ui, |ui| {
            ui.set_height(25.);
            ui.horizontal_top(|ui| {
                ui.add_space(5.);
                let (can_up, can_back, can_forward) = (self.can_go_up(), self.can_go_back(), self.can_go_forward());
                if ui.add_enabled_ui(can_up, |ui| icons::icon_button(ui, svg::ARROW_UP, "Up (Alt+Up)")).inner.clicked() { self.nav_up(); }
                if ui.add_enabled_ui(can_back, |ui| icons::icon_button(ui, svg::ARROW_LEFT, "Back (Alt+Left)")).inner.clicked() { self.nav_back(); }
                if ui.add_enabled_ui(can_forward, |ui| icons::icon_button(ui, svg::ARROW_RIGHT, "Forward (Alt+Right)")).inner.clicked() { self.nav_forward(); }
                if icons::icon_button(ui, svg::ARROW_CLOCKWISE, "Refresh").clicked() { self.refresh(); }
                if icons::icon_button(ui, svg::HOUSE_SIMPLE, "Home").clicked() { self.nav_home(); }
                ui.separator();
                let right_id = ui.id().with("navbar_right_width");
                let right_fixed = ui.data(|d| d.get_temp::<f32>(right_id)).unwrap_or(RIGHT_FIXED_ESTIMATE);
                let flex = ui.available_width() - right_fixed - 3. * ui.spacing().item_spacing.x;
                let path_w = (flex - SEARCH_PREFERRED_WIDTH).clamp(PATH_MIN_WIDTH, PATH_MAX_WIDTH);
                let search_w = (flex - path_w).clamp(SEARCH_MIN_WIDTH, SEARCH_PREFERRED_WIDTH);
                let address = TextEdit::singleline(&mut self.nav.input)
                    .hint_text("Path or scope (lib:all, tag:x, col:Name)")
                    .desired_width(path_w)
                    .ui(ui);
                if address.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                    if let Err(reason) = self.submit_address() {
                        crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
                    }
                }
                if self.is_loading() {
                    ui.add(Spinner::new().size(14.));
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    self.filter_menu(ui);

                    if icons::toggle_button(ui, svg::EYE, self.open_preview_pane, "Toggle preview pane").clicked() {
                        self.open_preview_pane = !self.open_preview_pane;
                    };

                    ui.separator();

                    let by_meaning = ui.add_enabled_ui(!self.viewer.filter.trim().is_empty(), |ui| icons::icon_button(ui, svg::AME_SEMANTIC_SEARCH, "Search the library by meaning (CLIP)")).inner;
                    if by_meaning.clicked() {
                        let text = self.viewer.filter.trim().to_string();
                        if let Err(reason) = self.open_scope(crate::command::Scope::Semantic(text)) {
                            crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
                        }
                    }

                    TextEdit::singleline(&mut self.viewer.filter)
                    .desired_width(search_w)
                    .hint_text("Search for files")
                    .ui(ui);

                    ui.separator();

                    self.scan_menu(ui);
                    let fixed = ui.min_rect().width() - search_w;
                    ui.data_mut(|d| d.insert_temp(right_id, fixed));
                });
            });
        });
        });
    }
}
