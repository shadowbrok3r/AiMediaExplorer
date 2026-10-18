//! SVG icon system: Phosphor Regular subset, custom `AME_*` icons, font glyphs, widgets and a debug gallery.

pub mod gallery;
pub mod glyphs;
pub mod pack;
pub mod widgets;

#[cfg(test)]
mod tests;

pub use glyphs::{
    drive_icon, file_icon, file_icon_svg, folder_shortcut_icon, folder_shortcut_svg, icon,
    icon_colored, icon_sized, install_fonts, p,
};
pub use pack as svg;
pub use pack::*;
pub use widgets::*;

/// Installs egui_extras' image loaders (SVG included); idempotent.
pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
}
