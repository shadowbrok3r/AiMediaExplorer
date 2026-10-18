//! The app theme: egui-theme-kit re-exported, plus the app font stack, command-line aliases and the Appearance window.

mod appearance;
mod config;
#[cfg(test)]
mod tests;

pub use appearance::{Appearance, setup};
pub use config::{cli_options, font_config};
pub use egui_theme_kit::*;

/// The kit font module with `definitions` building the app font stack.
pub mod fonts {
    pub use egui_theme_kit::fonts::*;

    /// The app font stack at `scale`: the kit's Inter faces, the Ubuntu monospace stack, Monaspace and Phosphor.
    pub fn definitions(scale: TextScale) -> eframe::egui::FontDefinitions {
        definitions_with(scale, &super::font_config())
    }
}
