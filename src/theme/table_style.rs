//! Selection and focus colors for `egui-data-table`, derived from the published tokens.

use eframe::egui::Context;

use super::tokens::{Tokens, mix};

/// Share of the primary role mixed into the focused cell's wash.
const FOCUS_MIX: f32 = 0.3;

/// Sets the table's selection, drag and focus colors from the tokens published on `ctx`.
pub fn apply(ctx: &Context, style: &mut egui_data_table::Style) {
    let t = Tokens::get(ctx);
    style.bg_selected_cell = Some(t.primary_soft);
    style.bg_selected_highlight_cell = Some(mix(t.primary_soft, t.primary, FOCUS_MIX));
    style.fg_selected_highlight_cell = Some(t.text);
    style.fg_drag_selection = Some(t.primary_border);
    style.focused_row_stroke = Some(t.primary_border);
}

#[cfg(test)]
mod tests {
    use super::super::presets::bootstrap_startup_theme;
    use super::*;

    #[test]
    fn selection_colors_follow_the_tokens() {
        let ctx = Context::default();
        bootstrap_startup_theme(&ctx);
        let t = Tokens::get(&ctx);
        let mut style = egui_data_table::Style::default();
        apply(&ctx, &mut style);
        assert_eq!(style.bg_selected_cell, Some(t.primary_soft));
        assert_eq!(style.fg_drag_selection, Some(t.primary_border));
        assert_eq!(style.focused_row_stroke, Some(t.primary_border));
        assert_eq!(style.fg_selected_highlight_cell, Some(t.text));
        assert_ne!(style.bg_selected_highlight_cell, style.bg_selected_cell);
    }
}
