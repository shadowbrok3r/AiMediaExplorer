//! Keys of the browser: the plain keys act on the selection while the browser has the keyboard; search, path and refresh work from anywhere in the window.

use eframe::egui::*;

use super::FileExplorer;
use super::action_bar::{KEYS, Origin};
use crate::app::SmartMediaApp;
use crate::command::{self, AppCommand, NavOp};
use crate::ui::tabs::TabKind;

impl FileExplorer {
    /// True when the tab has the keyboard: the pointer is over it or the last press was in its rows.
    fn has_keyboard(&self, ui: &Ui) -> bool {
        ui.rect_contains_pointer(ui.available_rect_before_wrap()) || self.browse.grid.keyboard_active() || self.details.keyboard_active()
    }

    /// Reads the keys of the browser: Esc clears the selection and then the query; T, C, D and Delete act on the selection. Never fires while a field has the focus or a menu or dialog is open.
    pub(crate) fn run_shortcuts(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        if self.toolbar.bar.popover.is_some() || self.toolbar.scan.open || self.toolbar.process.open {
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                self.close_popover();
                self.toolbar.scan.open = false;
                self.toolbar.process.open = false;
            }
            return;
        }
        let busy = self.toolbar.path.editing;
        let in_query = ctx.memory(|m| m.had_focus_last_frame(super::query::text_id(self)));
        if busy || in_query || !self.has_keyboard(ui) || ctx.egui_wants_keyboard_input() || Popup::is_any_open(&ctx) {
            return;
        }
        self.sync_view();
        if ui.input(|i| i.key_pressed(Key::Escape)) {
            if !self.selection.is_empty() {
                ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));
                self.clear_selection();
            } else if !self.toolbar.query.text.is_empty() {
                ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));
                self.toolbar.query.text.clear();
                self.apply_query_live();
            }
        }
        if self.selection.is_empty() {
            return;
        }
        for (key, action) in KEYS {
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, key)) {
                let _ = self.run_action_ui(&ctx, action, Origin::Key);
                break;
            }
        }
    }
}

impl SmartMediaApp {
    /// Reads the keys that work from anywhere in the window while a browser tab is in front: Ctrl+F searches, Ctrl+L types a path, F5 and Ctrl+R read the scope again.
    pub fn browser_keys(&mut self, ctx: &Context) {
        if self.shell.palette.open {
            return;
        }
        let browser = self.tree.find_active_focused().map(|(_, tab)| *tab).is_none_or(|tab| matches!(tab, TabKind::Browser { .. }));
        if !browser {
            return;
        }
        let (find, path, refresh) = ctx.input_mut(|i| (i.consume_key(Modifiers::COMMAND, Key::F), i.consume_key(Modifiers::COMMAND, Key::L), i.consume_key(Modifiers::NONE, Key::F5) | i.consume_key(Modifiers::COMMAND, Key::R)));
        if find {
            self.context.active_explorer().toolbar.query.focus = true;
        }
        if path {
            self.context.active_explorer().edit_address();
        }
        if refresh {
            command::send(AppCommand::Navigate { op: NavOp::Refresh });
        }
        if find || path || refresh {
            ctx.request_repaint();
        }
    }
}
#[cfg(test)]
mod tests;
