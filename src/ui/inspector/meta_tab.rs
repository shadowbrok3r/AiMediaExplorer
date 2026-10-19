//! The Metadata tab: caption, category, tags as chips and the description, saved automatically.

use std::time::Instant;

use egui::{Button, Key, RichText, TextEdit, Ui};

use super::meta::{Field, SaveState};
use super::state::Inspector;
use super::tags::{suggest, take_complete, with_tag, without_tag};
use super::target::{stream_view, is_dir};
use super::widgets::{ChipEvent, chip};
use crate::Thumbnail;
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;

/// Suggestions shown under a field.
const SUGGESTIONS: usize = 8;

/// The save indicator line: state, error text and the retry button.
fn indicator(ui: &mut Ui, insp: &mut Inspector) {
    let state = insp.meta.state();
    ui.horizontal(|ui| {
        match state {
            SaveState::Clean => {
                ui.label(RichText::new(state.label()).weak());
            }
            SaveState::Dirty => {
                ui.colored_label(crate::theme::warn(ui), state.label());
            }
            SaveState::Saving => {
                ui.add(egui::Spinner::new().size(14.0));
                ui.label(state.label());
            }
            SaveState::Saved => {
                icons::label(ui, svg::CHECK_CIRCLE, RichText::new(state.label()).color(crate::theme::success(ui)));
            }
            SaveState::Error => {
                icons::label(ui, svg::WARNING, RichText::new(state.label()).color(crate::theme::error(ui)));
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let can_save = matches!(state, SaveState::Dirty | SaveState::Error);
            let label = if state == SaveState::Error { "Retry" } else { "Save now" };
            if ui.add_enabled(can_save, Button::new(label)).clicked() {
                insp.flush_now(Instant::now());
            }
        });
    });
    if let Some(error) = insp.meta.error().map(str::to_string) {
        ui.colored_label(crate::theme::error(ui), error);
    }
}

fn suggestion_row(ui: &mut Ui, label: &str, items: &[String]) -> Option<String> {
    let mut picked = None;
    if !items.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(label).weak().small());
            for item in items {
                if ui.add(Button::new(RichText::new(item).small()).small()).clicked() {
                    picked = Some(item.clone());
                }
            }
        });
    }
    picked
}

fn text_field(ui: &mut Ui, insp: &mut Inspector, path: &str, field: Field, label: &str, hint: &str, now: Instant) {
    ui.label(RichText::new(label).strong());
    let mut text = match field {
        Field::Caption => insp.meta.draft.caption.clone(),
        Field::Category => insp.meta.draft.category.clone(),
        _ => String::new(),
    };
    let response = TextEdit::singleline(&mut text).hint_text(hint).desired_width(f32::INFINITY).id_salt(("inspector_meta", field.label(), path)).show(ui).response;
    if response.changed() {
        insp.meta.set_text(field, text.clone(), now);
    }
    if response.has_focus() {
        insp.suggest_field = Some(field);
    }
    if response.lost_focus() {
        insp.flush_now(now);
    }
    if field == Field::Category && insp.suggest_field == Some(Field::Category) {
        let picks = suggest(&insp.known_categories, &text, &[], SUGGESTIONS);
        if let Some(pick) = suggestion_row(ui, "Categories:", &picks) {
            insp.meta.set_text(Field::Category, pick, now);
            insp.suggest_field = None;
            insp.flush_now(now);
        }
    }
}

fn add_tags(insp: &mut Inspector, new: Vec<String>, now: Instant) {
    if new.is_empty() {
        return;
    }
    let mut tags = insp.meta.draft.tags.clone();
    for tag in &new {
        tags = with_tag(&tags, tag);
    }
    insp.meta.set_tags(tags, now);
    insp.flush_now(now);
}

fn tags_field(ui: &mut Ui, insp: &mut Inspector, path: &str, now: Instant) {
    ui.label(RichText::new("Tags").strong());
    let mut removed: Option<String> = None;
    ui.horizontal_wrapped(|ui| {
        if insp.meta.draft.tags.is_empty() {
            ui.label(RichText::new("No tags").weak());
        }
        for tag in insp.meta.draft.tags.clone() {
            if chip(ui, &tag, None, true, false) == ChipEvent::Removed {
                removed = Some(tag);
            }
        }
    });
    if let Some(tag) = removed {
        let tags = without_tag(&insp.meta.draft.tags, &tag);
        insp.meta.set_tags(tags, now);
        insp.flush_now(now);
    }
    let mut input = std::mem::take(&mut insp.tag_input);
    let response = TextEdit::singleline(&mut input).hint_text("Add a tag, then Enter or comma").desired_width(f32::INFINITY).id_salt(("inspector_tag_input", path)).show(ui).response;
    if response.has_focus() {
        insp.suggest_field = Some(Field::Tags);
    }
    if response.changed() && input.contains(',') {
        let (done, rest) = take_complete(&input);
        add_tags(insp, done, now);
        input = rest;
    }
    let entered = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    if entered {
        add_tags(insp, super::tags::split_input(&input), now);
        input.clear();
        response.request_focus();
    }
    if insp.suggest_field == Some(Field::Tags) {
        let picks = suggest(&insp.known_tags, &input, &insp.meta.draft.tags, SUGGESTIONS);
        if let Some(pick) = suggestion_row(ui, "Tags:", &picks) {
            add_tags(insp, vec![pick], now);
            input.clear();
        }
    }
    insp.tag_input = input;
}

pub fn show(ui: &mut Ui, insp: &mut Inspector, ex: &FileExplorer, row: &Thumbnail) {
    if is_dir(row) {
        ui.weak("Folders have no metadata.");
        return;
    }
    let now = Instant::now();
    insp.want_lists(now);
    if insp.meta.path() != Some(row.path.as_str()) {
        ui.weak("Loading...");
        return;
    }
    indicator(ui, insp);
    if let Some(lib) = insp.library.get(&row.path) {
        if !lib.facts.indexed && lib.error.is_none() {
            ui.label(RichText::new("Not in the library yet. The first edit adds it.").weak().small());
        }
    }
    ui.add_space(4.0);
    text_field(ui, insp, &row.path, Field::Caption, "Caption", "A short title", now);
    ui.add_space(4.0);
    text_field(ui, insp, &row.path, Field::Category, "Category", "For example landscape", now);
    ui.add_space(4.0);
    tags_field(ui, insp, &row.path, now);
    ui.add_space(4.0);
    ui.label(RichText::new("Description").strong());
    let streaming = stream_view(Some(row.path.as_str()), &ex.streaming_interim).is_some();
    let mut text = insp.meta.draft.description.clone();
    let response = ui
        .add_enabled_ui(!streaming, |ui| TextEdit::multiline(&mut text).hint_text("What the file shows").desired_rows(6).desired_width(f32::INFINITY).id_salt(("inspector_description", row.path.as_str())).show(ui).response)
        .inner;
    if streaming {
        ui.label(RichText::new("The description is being generated.").weak().small());
    }
    if response.changed() {
        insp.meta.set_text(Field::Description, text, now);
    }
    if response.has_focus() {
        insp.suggest_field = Some(Field::Description);
    }
    if response.lost_focus() {
        insp.flush_now(now);
    }
    let dirty = insp.meta.dirty_fields();
    if !dirty.is_empty() {
        ui.label(RichText::new(format!("Unsaved: {}", dirty.labels())).weak().small());
    }
}
