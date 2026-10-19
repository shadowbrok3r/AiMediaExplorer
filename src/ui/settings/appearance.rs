//! Appearance section: the theme picker and the egui options that used to live in their own windows.

use eframe::egui::{ScrollArea, Ui};

use super::sections::name;
use super::widgets::Focus;
use crate::theme::Appearance as Theme;

/// Part of the section on show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tab {
    #[default]
    Theme,
    EguiOptions,
}

#[derive(Default)]
pub struct Appearance {
    pub tab: Tab,
}

impl Appearance {
    /// Draws the section; `save_egui` is set when the egui options should be written to storage.
    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, theme: &mut Theme, save_egui: &mut bool) {
        ui.horizontal(|ui| {
            for (tab, label) in [(Tab::Theme, name::THEME), (Tab::EguiOptions, name::EGUI_OPTIONS)] {
                let response = ui.selectable_label(self.tab == tab, label);
                focus.mark(ui, label, &response);
                if response.clicked() {
                    self.tab = tab;
                }
            }
        });
        if focus.is(name::EGUI_OPTIONS) {
            self.tab = Tab::EguiOptions;
        } else if focus.is(name::THEME) {
            self.tab = Tab::Theme;
        }
        ui.separator();
        match self.tab {
            Tab::Theme => theme.show_embedded(ui),
            Tab::EguiOptions => {
                ui.horizontal(|ui| {
                    ui.weak("Changes apply at once. Save keeps them across restarts.");
                    if ui.button("Save options").on_hover_text("Keep these options across restarts").clicked() {
                        *save_egui = true;
                    }
                });
                ScrollArea::vertical().id_salt("settings-egui-options").auto_shrink([false, false]).show(ui, |ui| ui.ctx().clone().settings_ui(ui));
            }
        }
    }
}
