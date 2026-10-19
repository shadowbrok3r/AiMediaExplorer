//! Ctrl+J and requests to show the assistant: open the right rail on the Ask tab and focus the message box.

use egui::{Context, Key, Modifiers};

use super::api;
use super::composer;
use crate::app::SmartMediaApp;
use crate::ui::inspector::Inspector;
use crate::ui::inspector::prefs::RailTab;

/// Shows the Ask tab of the rail (unless the assistant is in its own window) and focuses the composer.
fn show_in_rail(ctx: &Context, insp: &mut Inspector, popped_out: bool) {
    if !popped_out {
        insp.set_open(true);
        insp.edit_prefs(|p| p.rail_tab = RailTab::Ask);
    }
    composer::request_focus(ctx);
}

/// Reads Ctrl+J and the queued requests to show the assistant; true when it was asked for.
pub fn handle_shortcuts(ctx: &Context, insp: &mut Inspector, popped_out: bool) -> bool {
    let pressed = ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::J));
    let requested = api::take_open();
    if pressed || requested {
        show_in_rail(ctx, insp, popped_out);
    }
    pressed || requested
}

/// Shows the assistant: the rail's Ask tab, or the pop-out window when it is open.
pub fn show_assistant(app: &mut SmartMediaApp) {
    let popped_out = app.context.assistant_window_open;
    show_in_rail(&app.egui_ctx, &mut app.context.inspector, popped_out);
}
