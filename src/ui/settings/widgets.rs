//! Row widgets shared by the sections: captions that the search can point at, optional text, secrets, sizes and pickers.

use std::cell::Cell;

use eframe::egui::{Align, DragValue, Response, RichText, Stroke, StrokeKind, TextEdit, Ui};

use crate::icons::{self, svg};

/// The setting the search result asked to show.
#[derive(Default)]
pub struct Focus {
    pub label: Option<&'static str>,
    scrolled: Cell<bool>,
}

impl Focus {
    pub fn set(&mut self, label: Option<&'static str>) {
        self.label = label;
        self.scrolled.set(false);
    }

    /// Scrolls to and outlines `response` when it belongs to the focused setting.
    pub fn mark(&self, ui: &Ui, label: &str, response: &Response) {
        if self.label != Some(label) {
            return;
        }
        if !self.scrolled.replace(true) {
            response.scroll_to_me(Some(Align::Center));
        }
        let stroke = Stroke::new(1.5, crate::theme::tokens(ui.ctx()).primary);
        ui.painter().rect_stroke(response.rect.expand(3.0), 4.0, stroke, StrokeKind::Outside);
    }

    pub fn is(&self, label: &str) -> bool {
        self.label == Some(label)
    }
}

/// A caption that the search can focus.
pub fn caption(ui: &mut Ui, focus: &Focus, label: &'static str) -> Response {
    let response = ui.label(label);
    focus.mark(ui, label, &response);
    response
}

/// A section heading drawn as strong text with a rule under it.
pub fn group(ui: &mut Ui, title: &str) -> Response {
    ui.add_space(8.0);
    let response = ui.label(RichText::new(title).strong().size(15.0));
    ui.separator();
    response
}

/// A heading that the search can focus.
pub fn titled(ui: &mut Ui, focus: &Focus, title: &'static str) {
    let response = group(ui, title);
    focus.mark(ui, title, &response);
}

/// A checkbox with an optional hover text; true when it changed.
pub fn toggle(ui: &mut Ui, focus: &Focus, label: &'static str, hover: &str, value: &mut bool) -> bool {
    let mut response = ui.checkbox(value, label);
    if !hover.is_empty() {
        response = response.on_hover_text(hover);
    }
    focus.mark(ui, label, &response);
    response.changed()
}

/// A single-line text field over an optional string; an empty text stores `None`.
pub fn opt_text(ui: &mut Ui, slot: &mut Option<String>, hint: &str, width: f32) -> Response {
    let mut text = slot.clone().unwrap_or_default();
    let response = ui.add(TextEdit::singleline(&mut text).hint_text(hint).desired_width(width));
    if response.changed() {
        *slot = if text.trim().is_empty() { None } else { Some(text) };
    }
    response
}

/// A masked field for an API key; `env` names the variable used while the field is empty.
pub fn secret(ui: &mut Ui, slot: &mut Option<String>, env: Option<&str>, width: f32) -> Response {
    let mut text = slot.clone().unwrap_or_default();
    let hint = match env.filter(|name| std::env::var(name).is_ok()) {
        Some(name) => format!("using ${name}"),
        None => "not set".to_string(),
    };
    let response = ui.add(TextEdit::singleline(&mut text).password(true).hint_text(hint).desired_width(width));
    if response.changed() {
        *slot = if text.trim().is_empty() { None } else { Some(text.trim().to_string()) };
    }
    response
}

/// Kibibytes in a drag field with a "no limit" checkbox over an optional byte count; true when it changed.
pub fn size_limit(ui: &mut Ui, hover: &str, slot: &mut Option<u64>) -> bool {
    let mut changed = false;
    let mut limited = slot.is_some();
    if ui.checkbox(&mut limited, "Limit").changed() {
        *slot = if limited { Some(slot.unwrap_or(10 * 1024)) } else { None };
        changed = true;
    }
    if let Some(bytes) = slot.as_mut() {
        let mut kib = (*bytes / 1024) as f64;
        let response = ui.add(DragValue::new(&mut kib).range(0.0..=f64::from(u32::MAX)).speed(8.0).suffix(" KiB")).on_hover_text(hover);
        if response.changed() {
            *bytes = (kib.max(0.0) as u64) * 1024;
            changed = true;
        }
    } else {
        ui.weak("no limit");
    }
    changed
}

/// A list of strings with a remove button on each and a field that adds one; true when it changed.
pub fn string_list(ui: &mut Ui, id_salt: &str, list: &mut Vec<String>, input: &mut String, hint: &str, folder_icon: bool) -> bool {
    let mut changed = false;
    let mut remove = None;
    for (index, item) in list.iter().enumerate() {
        ui.horizontal(|ui| {
            if folder_icon {
                icons::label(ui, svg::FOLDER, item.as_str());
            } else {
                ui.label(item.as_str());
            }
            if icons::icon_button(ui, svg::X, "Remove").clicked() {
                remove = Some(index);
            }
        });
    }
    if let Some(index) = remove {
        list.remove(index);
        changed = true;
    }
    ui.horizontal(|ui| {
        let response = ui.add(TextEdit::singleline(input).hint_text(hint).desired_width(260.0).id_salt(id_salt));
        let enter = response.lost_focus() && ui.input(|i| i.key_pressed(eframe::egui::Key::Enter));
        let ready = !input.trim().is_empty();
        if (ui.add_enabled(ready, eframe::egui::Button::new("Add")).clicked() || (enter && ready)) && !list.iter().any(|p| p == input.trim()) {
            list.push(input.trim().to_string());
            input.clear();
            changed = true;
        }
    });
    changed
}

/// A text field with a Browse button that fills it through the picker slot `slot`.
pub fn path_field(ui: &mut Ui, text: &mut String, hint: &str, dialog_title: &str, slot: &'static str, folder: bool) -> Response {
    let response = ui.add(TextEdit::singleline(text).hint_text(hint).desired_width(340.0));
    if icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked() {
        if folder {
            crate::ui::dialogs::pick_folder_into(ui.ctx(), dialog_title, slot);
        } else {
            crate::ui::dialogs::pick_file_into(ui.ctx(), dialog_title, slot);
        }
    }
    response
}

/// Text for a byte count.
pub fn human(bytes: u64) -> String {
    crate::database::snapshot::human(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_focus_matches_only_its_own_label() {
        let mut focus = Focus::default();
        assert!(!focus.is("x"));
        focus.set(Some("x"));
        assert!(focus.is("x") && !focus.is("y"));
        focus.set(None);
        assert!(!focus.is("x"));
    }
}
