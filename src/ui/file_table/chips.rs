//! The filter strip: a removable chip per active condition, the "+ Filter" editor, saved views and the hidden-rows note.

use eframe::egui::*;
use egui::containers::menu::{MenuButton, MenuConfig};

use super::FileExplorer;
use super::rows::{Chip, ChipKind};
use crate::icons::{self, svg};

/// What the user asked for in the strip this frame.
#[derive(Default)]
struct Actions {
    remove: Option<ChipKind>,
    clear: bool,
}

#[derive(Clone, Default)]
struct ViewsUi {
    renaming: Option<String>,
    rename_text: String,
    new_name: String,
}

/// Plural form of `count` followed by `noun`.
pub fn plural(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

/// Text of the note beside the chips, when the filters hide rows or leave nothing.
pub fn hidden_note(shown: usize, total: usize, filters: usize) -> Option<String> {
    if filters == 0 || total == 0 {
        return None;
    }
    if shown == 0 {
        return Some(format!("No items match the {}", plural(filters, "active filter")));
    }
    let hidden = total.saturating_sub(shown);
    (hidden > 0).then(|| format!("{} hidden by {}", plural(hidden, "item"), plural(filters, "filter")))
}

fn chip_button(ui: &mut Ui, chip: &Chip) -> Response {
    let button = icons::button_widget(ui, svg::X, &chip.label).small().corner_radius(CornerRadius::same(9));
    ui.add(button).on_hover_text(format!("Remove: {}", chip.label))
}

fn add_button(ui: &mut Ui, ex: &mut FileExplorer) {
    let button = icons::button_widget(ui, svg::PLUS, "Filter").small().corner_radius(CornerRadius::same(9));
    let (response, _) = MenuButton::from_button(button)
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .ui(ui, |ui| ex.filter_editor(ui, "strip"));
    response.on_hover_text("Add or edit filters");
}

fn views_button(ui: &mut Ui, ex: &mut FileExplorer) {
    let label = match &ex.active_filter_group {
        Some(name) => format!("View: {name}"),
        None => "Views".to_string(),
    };
    let button = icons::button_widget(ui, svg::AME_FILTER_GROUP, label).small().corner_radius(CornerRadius::same(9));
    let (response, _) = MenuButton::from_button(button)
        .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
        .ui(ui, |ui| views_menu(ui, ex));
    if response.clicked() {
        ex.refresh_saved_views();
    }
    response.on_hover_text("Saved filter views");
}

fn views_menu(ui: &mut Ui, ex: &mut FileExplorer) {
    ui.set_width(300.0);
    let id = Id::unique(("ame_filter_views", ex.viewer.scope));
    let mut state: ViewsUi = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Saved views").strong().color(crate::theme::primary(ui)));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if icons::icon_button(ui, svg::ARROWS_CLOCKWISE, "Reload the saved views").clicked() {
                ex.refresh_saved_views();
            }
        });
    });
    let views = ex.saved_views().to_vec();
    if views.is_empty() {
        ui.weak("No saved views yet.");
    }
    let mut error: Option<String> = None;
    for group in &views {
        if state.renaming.as_deref() == Some(group.name.as_str()) {
            ui.horizontal(|ui| {
                let resp = ui.add(TextEdit::singleline(&mut state.rename_text).desired_width(150.0));
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                if enter || icons::icon_button(ui, svg::CHECK, "Rename").clicked() {
                    match ex.rename_view(&group.name, &state.rename_text) {
                        Ok(()) => state.renaming = None,
                        Err(e) => error = Some(e),
                    }
                }
                if icons::icon_button(ui, svg::X, "Cancel").clicked() {
                    state.renaming = None;
                }
            });
            continue;
        }
        ui.horizontal(|ui| {
            let active = ex.active_filter_group.as_deref() == Some(group.name.as_str());
            if ui.selectable_label(active, &group.name).on_hover_text("Apply this view").clicked() {
                ex.apply_view(group);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if icons::icon_button(ui, svg::TRASH, "Delete this view").clicked() {
                    ex.delete_view(&group.name);
                }
                if icons::icon_button(ui, svg::PENCIL_SIMPLE, "Rename this view").clicked() {
                    state.renaming = Some(group.name.clone());
                    state.rename_text = group.name.clone();
                }
                if icons::icon_button(ui, svg::FLOPPY_DISK, "Replace this view with the current filters").clicked() {
                    if let Err(e) = ex.save_view(&group.name) {
                        error = Some(e);
                    }
                }
            });
        });
    }
    ui.separator();
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 70.0).max(80.0);
        let resp = ui.add(TextEdit::singleline(&mut state.new_name).hint_text("Save current filters as").desired_width(width));
        let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        if (ui.button("Save").clicked() || enter) && !state.new_name.trim().is_empty() {
            match ex.save_view(&state.new_name) {
                Ok(()) => state.new_name.clear(),
                Err(e) => error = Some(e),
            }
        }
    });
    if let Some(e) = error {
        ui.colored_label(crate::theme::error(ui), e);
    }
    ui.data_mut(|d| d.insert_temp(id, state));
}

/// Draws the strip for `ex` and applies what the user did in it.
pub fn show(ui: &mut Ui, ex: &mut FileExplorer) {
    ex.bind_filter_waker(ui.ctx());
    ex.run_filter_commands(ui.ctx());
    ex.sync_view();
    ex.forget_changed_view();
    let chips = ex.filter_spec().chips();
    let (shown, total) = ex.filter_counts();
    let note = hidden_note(shown, total, chips.len());
    let mut act = Actions::default();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 3.0);
        for chip in &chips {
            if chip_button(ui, chip).clicked() {
                act.remove = Some(chip.kind.clone());
            }
        }
        add_button(ui, ex);
        views_button(ui, ex);
        if let Some(note) = &note {
            ui.style_mut().wrap_mode = Some(TextWrapMode::Extend);
            ui.add_space(6.0);
            let text = RichText::new(note);
            ui.label(if shown == 0 { text.color(crate::theme::warn(ui)) } else { text.weak() });
            if ui.link("Clear").on_hover_text("Remove every filter").clicked() {
                act.clear = true;
            }
        }
    });
    if let Some(kind) = act.remove {
        ex.edit_filters(|s| s.remove(&kind));
    }
    if act.clear {
        ex.clear_filters();
    }
    ex.publish_filter_debug(ui.ctx());
}

#[cfg(test)]
mod tests;
