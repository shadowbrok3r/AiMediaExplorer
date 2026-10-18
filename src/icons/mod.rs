//! SVG icon system: the app pack generated over egui-icon-pack, font glyphs, file and place mappings.

pub mod gallery;
pub mod glyphs;
pub mod pack;

#[cfg(test)]
mod tests;

pub use egui_icon_pack::{install, widgets};
pub use glyphs::{
    drive_icon, file_icon, file_icon_svg, folder_shortcut_icon, folder_shortcut_svg, icon,
    icon_colored, icon_sized, install_fonts, p,
};
pub use pack as svg;
pub use pack::*;
pub use widgets::*;
