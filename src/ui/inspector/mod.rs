//! The right rail: preview, details, metadata editing and AI actions for the focused browser's selection.

mod ai_tab;
pub mod backend;
mod batch_tab;
pub mod facts;
pub mod host;
mod info_tab;
pub mod lightbox;
pub mod meta;
mod meta_tab;
mod panel;
pub mod prefs;
pub mod preview;
pub mod state;
pub mod tags;
pub mod target;
pub mod video;
pub mod view;
mod widgets;

#[cfg(test)]
mod tests;

pub use host::{HostArgs, HostOutput};
pub use state::{Inspector, RowEdit};

use crate::app::SmartMediaApp;

impl SmartMediaApp {
    /// Draws the inspector rail for the focused browser tab and applies its row edits to every browser.
    pub fn inspector_rail(&mut self, ui: &mut eframe::egui::Ui) {
        let tab = self.context.active_owner();
        let ctx = &mut self.context;
        let ex = crate::app::explorer_for(&mut ctx.file_explorer, &mut ctx.filtered_tabs, ctx.active_tab.as_ref());
        let out = host::show(ui, &mut ctx.inspector, HostArgs { ex, tab: &tab, assistant_open: &mut ctx.assistant_window_open });
        for edit in &out.edits {
            state::apply_edit(&mut ctx.file_explorer, edit);
            ctx.filtered_tabs.values_mut().for_each(|ex| state::apply_edit(ex, edit));
        }
        for (level, text) in out.notices {
            let _ = ctx.toast_tx.try_send((level.kind(), text));
        }
    }
}
