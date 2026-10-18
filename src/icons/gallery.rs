//! Debug gallery of the app icon pack.

use egui::{Context, Ui};

pub use egui_icon_pack::gallery::{is_open, set_open, toggle};

/// Draws the gallery window when open; call once per frame.
pub fn show_window(ctx: &Context) {
    egui_icon_pack::gallery::show_window(ctx, &super::pack::PACK);
}

/// Gallery contents, usable inside any `Ui`.
pub fn show(ui: &mut Ui) {
    egui_icon_pack::gallery::show(ui, &super::pack::PACK);
}
