//! The strip above the rows: Grid or Details switch, tile size, view options, sort and the item count.

use egui::{Align, ComboBox, Frame, Key, Layout, Margin, Modifiers, Ui};

use super::columns::Col;
use super::view_mode::{SizePreset, TILE_MAX, TILE_MIN, TileFit, ViewMode};
use crate::icons::{self, svg};
use crate::theme::controls::{Segment, Slider, segmented};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::SortSpec;
use crate::ui::titlebar::kit_icon;

/// Width below which the strip keeps the mode switch and folds the rest into a menu.
pub const COMPACT_BELOW: f32 = 760.0;

/// Draws the strip for `ex` and applies its changes.
pub fn show(ui: &mut Ui, ex: &mut FileExplorer) {
    shortcuts(ui, ex);
    let compact = ui.available_width() < COMPACT_BELOW;
    Frame::new().inner_margin(Margin::symmetric(4, 3)).show(ui, |ui| {
        ui.horizontal(|ui| {
            mode_switch(ui, ex, compact);
            let grid = ex.browse.mode == ViewMode::Grid;
            if compact {
                icons::icon_menu_button(ui, svg::SLIDERS_HORIZONTAL, "View options", |ui| {
                    if grid {
                        size_controls(ui, ex);
                        ui.separator();
                        option_items(ui, ex);
                        ui.separator();
                    }
                    sort_controls(ui, ex);
                });
            } else {
                if grid {
                    size_controls(ui, ex);
                    icons::icon_menu_button(ui, svg::SLIDERS_HORIZONTAL, "View options", |ui| option_items(ui, ex));
                    ui.separator();
                }
                sort_controls(ui, ex);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.weak(count_text(ex.view.len(), compact));
            });
        });
    });
}

// F9 switches the view; Ctrl+1 and Ctrl+2 pick Grid and Details, for the explorer under the pointer or in keyboard use.
fn shortcuts(ui: &mut Ui, ex: &mut FileExplorer) {
    let hovered = ui.rect_contains_pointer(ui.available_rect_before_wrap());
    let active = hovered || ex.browse.grid.keyboard_active() || ex.details.keyboard_active();
    if !active || ui.ctx().egui_wants_keyboard_input() || egui::Popup::is_any_open(ui.ctx()) {
        return;
    }
    let (toggle, grid, details) = ui.input_mut(|i| (i.consume_key(Modifiers::NONE, Key::F9), i.consume_key(Modifiers::COMMAND, Key::Num1), i.consume_key(Modifiers::COMMAND, Key::Num2)));
    if toggle {
        ex.set_view_mode(ex.browse.mode.other());
    }
    if grid {
        ex.set_view_mode(ViewMode::Grid);
    }
    if details {
        ex.set_view_mode(ViewMode::Details);
    }
}

fn mode_switch(ui: &mut Ui, ex: &mut FileExplorer, compact: bool) {
    let mut mode = ex.browse.mode;
    let label = |text: &'static str| if compact { "" } else { text };
    let resp = segmented(
        ui,
        &mut mode,
        &[
            Segment::new(ViewMode::Grid, label("Grid")).icon(kit_icon(svg::SQUARES_FOUR)),
            Segment::new(ViewMode::Details, label("Details")).icon(kit_icon(svg::LIST)),
        ],
    );
    resp.on_hover_text("Grid or details view (F9)");
    ex.set_view_mode(mode);
}

fn size_controls(ui: &mut Ui, ex: &mut FileExplorer) {
    let mut preset = SizePreset::of(ex.browse.tile);
    let segments: Vec<Segment<'_, Option<SizePreset>>> = SizePreset::ALL.into_iter().map(|p| Segment::new(Some(p), p.label())).collect();
    if segmented(ui, &mut preset, &segments).changed() {
        if let Some(p) = preset {
            ex.set_tile_size(p.px());
        }
    }
    let mut tile = ex.browse.tile;
    if Slider::new(&mut tile, TILE_MIN..=TILE_MAX).width(120.0).step(4.0).show(ui).on_hover_text("Thumbnail size").changed() {
        ex.set_tile_size(tile);
    }
}

fn option_items(ui: &mut Ui, ex: &mut FileExplorer) {
    let mut fit = ex.browse.fit;
    ui.radio_value(&mut fit, TileFit::Crop, TileFit::Crop.label());
    ui.radio_value(&mut fit, TileFit::Fit, TileFit::Fit.label());
    ex.set_tile_fit(fit);
    ui.separator();
    let mut captions = ex.browse.captions;
    ui.checkbox(&mut captions, "Show file names");
    ex.set_captions(captions);
    let mut badges = ex.browse.badges;
    ui.checkbox(&mut badges, "Show badges");
    ex.set_badges(badges);
}

fn sort_controls(ui: &mut Ui, ex: &mut FileExplorer) {
    let ctx = ui.ctx().clone();
    let columns: Vec<Col> = Col::available(ex.viewer.mode, ex.viewer.showing_similarity).into_iter().filter(|c| c.sortable()).collect();
    let current = ex.view.sort().keys().first().copied();
    let name = current.and_then(|k| Col::from_id(k.column)).map_or("Unsorted", |c| c.menu_name());
    let mut picked = None;
    ui.label("Sort");
    ComboBox::from_id_salt("ame_strip_sort").selected_text(name).width(96.0).show_ui(ui, |ui| {
        for col in columns {
            if ui.selectable_label(current.is_some_and(|k| k.column == col.id()), col.menu_name()).clicked() {
                picked = Some(SortSpec::single(col.id(), current.is_some_and(|k| k.descending)));
            }
        }
    });
    let descending = current.is_some_and(|k| k.descending);
    let (icon, tip) = if descending { (svg::SORT_DESCENDING, "Descending") } else { (svg::SORT_ASCENDING, "Ascending") };
    if ui.add_enabled(current.is_some(), icons::icon_button_widget(ui, icon, icons::IconSize::Md)).on_hover_text(tip).clicked() {
        if let Some(k) = current {
            picked = Some(SortSpec::single(k.column, !k.descending));
        }
    }
    if let Some(spec) = picked {
        ex.view.set_sort(spec);
        ex.details.save_prefs(&ctx, ex.view.sort());
    }
}

/// `12,345 items`, or just the number when there is little room.
pub fn count_text(n: usize, compact: bool) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if compact {
        out
    } else if n == 1 {
        "1 item".to_string()
    } else {
        format!("{out} items")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_get_thousands_separators() {
        assert_eq!(count_text(0, false), "0 items");
        assert_eq!(count_text(1, false), "1 item");
        assert_eq!(count_text(999, false), "999 items");
        assert_eq!(count_text(1_000, false), "1,000 items");
        assert_eq!(count_text(1_234_567, false), "1,234,567 items");
        assert_eq!(count_text(100_000, true), "100,000");
        assert_eq!(count_text(1, true), "1");
    }
}
