//! The Scan and Process dialogs of the browser.

use eframe::egui::Context;

use super::FileExplorer;

pub mod process;
pub mod scan;

impl FileExplorer {
    /// Draws the dialogs of the tab that are open.
    pub fn show_dialogs(&mut self, ctx: &Context) {
        scan::show(ctx, self);
        process::show(ctx, self);
    }
}
