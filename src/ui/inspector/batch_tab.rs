//! The multi-selection editor: counts, common tags, batch tag and category changes, collections and AI work for every selected file.

use std::time::Instant;

use egui::{Button, ComboBox, Key, RichText, TextEdit, Ui};
use humansize::DECIMAL;

use super::meta::Field;
use super::state::{BatchAction, Inspector};
use super::tags::{split_input, suggest, take_complete};
use super::target::{AiState, current_ai_state};
use super::widgets::{ChipEvent, chip, count_noun};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;

const SUGGESTIONS: usize = 8;

enum Act {
    Batch(BatchAction),
    Collection { name: String, add: bool },
}

fn counts_line(s: &super::target::BatchSummary) -> String {
    let mut parts = vec![count_noun(s.files, "file", "files")];
    if s.dirs > 0 {
        parts.push(count_noun(s.dirs, "folder", "folders"));
    }
    if s.images > 0 {
        parts.push(count_noun(s.images, "image", "images"));
    }
    if s.videos > 0 {
        parts.push(count_noun(s.videos, "video", "videos"));
    }
    parts.join(", ")
}

fn tags_section(ui: &mut Ui, insp: &mut Inspector, s: &super::target::BatchSummary, acts: &mut Vec<Act>) {
    ui.label(RichText::new("Tags").strong());
    let common: Vec<_> = s.tally.common().into_iter().cloned().collect();
    let partial: Vec<_> = s.tally.partial().into_iter().cloned().collect();
    if common.is_empty() && partial.is_empty() {
        ui.label(RichText::new("No file has tags").weak());
    }
    if !common.is_empty() {
        ui.label(RichText::new("On every file").weak().small());
        ui.horizontal_wrapped(|ui| {
            for t in &common {
                if chip(ui, &t.tag, None, true, false) == ChipEvent::Removed {
                    acts.push(Act::Batch(BatchAction::RemoveTags(vec![t.tag.clone()])));
                }
            }
        });
    }
    if !partial.is_empty() {
        ui.label(RichText::new("On some files").weak().small());
        ui.horizontal_wrapped(|ui| {
            for t in &partial {
                let ratio = format!("{}/{}", t.count, s.tally.rows);
                if chip(ui, &t.tag, Some(&ratio), true, true) == ChipEvent::Removed {
                    acts.push(Act::Batch(BatchAction::RemoveTags(vec![t.tag.clone()])));
                }
            }
        });
    }
    let mut input = std::mem::take(&mut insp.batch.tag_input);
    let mut add: Vec<String> = Vec::new();
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 110.0).max(80.0);
        let response = TextEdit::singleline(&mut input).hint_text("Add tags, comma separated").desired_width(width).id_salt("inspector_batch_tags").show(ui).response;
        if response.has_focus() {
            insp.suggest_field = Some(Field::Tags);
        }
        if response.changed() && input.contains(',') {
            let (done, rest) = take_complete(&input);
            add.extend(done);
            input = rest;
        }
        let entered = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let clicked = ui.add_enabled(s.files > 0, Button::new(format!("Add to {}", s.files))).clicked();
        if entered || clicked {
            add.extend(split_input(&input));
            input.clear();
        }
    });
    if insp.suggest_field == Some(Field::Tags) {
        let present: Vec<String> = s.tally.tags.iter().map(|t| t.tag.clone()).collect();
        let picks = suggest(&insp.known_tags, &input, &present, SUGGESTIONS);
        if !picks.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for pick in &picks {
                    if ui.add(Button::new(RichText::new(pick).small()).small()).clicked() {
                        add.push(pick.clone());
                    }
                }
            });
        }
    }
    insp.batch.tag_input = input;
    if !add.is_empty() {
        acts.push(Act::Batch(BatchAction::AddTags(add)));
    }
}

fn category_section(ui: &mut Ui, insp: &mut Inspector, s: &super::target::BatchSummary, acts: &mut Vec<Act>) {
    ui.label(RichText::new("Category").strong());
    let mut parts: Vec<String> = s.categories.iter().map(|(c, n)| format!("{c} ({n})")).collect();
    if s.uncategorized > 0 {
        parts.push(format!("none ({})", s.uncategorized));
    }
    ui.label(RichText::new(parts.join(", ")).weak().small());
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 130.0).max(80.0);
        let response = TextEdit::singleline(&mut insp.batch.category_input).hint_text("Category for all").desired_width(width).id_salt("inspector_batch_category").show(ui).response;
        let entered = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let typed = !insp.batch.category_input.trim().is_empty();
        if (ui.add_enabled(typed && s.files > 0, Button::new("Set")).clicked() || (entered && typed)) && s.files > 0 {
            acts.push(Act::Batch(BatchAction::SetCategory(Some(insp.batch.category_input.trim().to_string()))));
            insp.batch.category_input.clear();
        }
        if ui.add_enabled(s.files > 0, Button::new("Clear")).on_hover_text("Remove the category from every selected file").clicked() {
            acts.push(Act::Batch(BatchAction::SetCategory(None)));
        }
    });
}

fn collections_section(ui: &mut Ui, insp: &mut Inspector, s: &super::target::BatchSummary, acts: &mut Vec<Act>) {
    ui.label(RichText::new("Collections").strong());
    let now = Instant::now();
    insp.want_collections(now);
    let names: Vec<String> = insp.collections.iter().map(|c| c.name.clone()).collect();
    let selected = insp.batch.collection_pick.clone().filter(|p| names.contains(p));
    ui.horizontal_wrapped(|ui| {
        ComboBox::from_id_salt("inspector_batch_collection").selected_text(selected.clone().unwrap_or_else(|| "Choose a collection".to_string())).width(170.0).show_ui(ui, |ui| {
            for c in &insp.collections {
                let label = format!("{} ({})", c.name, c.count);
                if ui.selectable_label(selected.as_deref() == Some(c.name.as_str()), label).clicked() {
                    insp.batch.collection_pick = Some(c.name.clone());
                }
            }
        });
        let pick = selected.clone();
        if ui.add_enabled(pick.is_some() && s.files > 0, Button::new(format!("Add {}", s.files))).clicked() {
            acts.push(Act::Collection { name: pick.clone().unwrap_or_default(), add: true });
        }
        if ui.add_enabled(pick.is_some() && s.files > 0, Button::new("Remove")).clicked() {
            acts.push(Act::Collection { name: pick.unwrap_or_default(), add: false });
        }
    });
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 130.0).max(80.0);
        let response = TextEdit::singleline(&mut insp.batch.new_collection).hint_text("New collection name").desired_width(width).id_salt("inspector_batch_new_collection").show(ui).response;
        let entered = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let typed = !insp.batch.new_collection.trim().is_empty();
        if (ui.add_enabled(typed && s.files > 0, Button::new("Create and add")).clicked() || (entered && typed)) && s.files > 0 {
            acts.push(Act::Collection { name: insp.batch.new_collection.trim().to_string(), add: true });
            insp.batch.new_collection.clear();
        }
    });
}

fn ai_section(ui: &mut Ui, s: &super::target::BatchSummary, acts: &mut Vec<Act>) {
    ui.label(RichText::new("AI").strong());
    let ready = current_ai_state() == AiState::Ready;
    let n = s.files;
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(ready && n > 0, icons::button_widget(ui, svg::AME_AI_DESCRIBE, format!("Describe {n}"))).on_disabled_hover_text("Choose a vision model folder in the settings").clicked() {
            acts.push(Act::Batch(BatchAction::Describe { regenerate: false }));
        }
        if ui.add_enabled(ready && n > 0, icons::button_widget(ui, svg::AME_REGENERATE, "Regenerate")).on_hover_text("Replace the existing descriptions").clicked() {
            acts.push(Act::Batch(BatchAction::Describe { regenerate: true }));
        }
        if ui.add_enabled(n > 0, icons::button_widget(ui, svg::AME_CLIP_EMBED, format!("CLIP {n}"))).clicked() {
            acts.push(Act::Batch(BatchAction::GenerateClip));
        }
        if ui.add_enabled(n > 0, icons::button_widget(ui, svg::AME_REFINE, format!("Refine {n}"))).clicked() {
            acts.push(Act::Batch(BatchAction::Refine));
        }
        if ui.add_enabled(n > 0, icons::button_widget(ui, svg::AME_ASSISTANT, format!("Ask AI {n}"))).clicked() {
            acts.push(Act::Batch(BatchAction::AttachToAssistant));
        }
    });
}

fn confirm_section(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer) {
    let Some(pending) = insp.batch.confirm.clone() else { return };
    ui.group(|ui| {
        ui.label(RichText::new(format!("{} for {} files?", pending.action.label(), pending.count)).strong());
        ui.label("This runs the model on every selected file and can take a long time.");
        ui.horizontal(|ui| {
            if ui.button(format!("{} {}", pending.action.label(), pending.count)).clicked() {
                insp.run_batch(ex, pending.action.clone(), true);
            }
            if ui.button("Cancel").clicked() {
                insp.batch.confirm = None;
            }
        });
    });
}

pub fn show(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, count: usize) {
    let s = insp.summary(ex).clone();
    ui.label(RichText::new(count_noun(count, "item selected", "items selected")).heading());
    ui.label(RichText::new(format!("{}, {}", counts_line(&s), humansize::format_size(s.size, DECIMAL))).weak());
    if s.files == 0 {
        ui.label("Only folders are selected; the actions below need files.");
    }
    ui.add_space(6.0);
    let mut acts: Vec<Act> = Vec::new();
    confirm_section(ui, insp, ex);
    tags_section(ui, insp, &s, &mut acts);
    ui.add_space(6.0);
    category_section(ui, insp, &s, &mut acts);
    ui.add_space(6.0);
    collections_section(ui, insp, &s, &mut acts);
    ui.add_space(6.0);
    ai_section(ui, &s, &mut acts);
    if let Some(last) = &insp.last_batch {
        ui.add_space(6.0);
        ui.label(RichText::new(last.as_str()).weak().small());
    }
    insp.want_lists(Instant::now());
    for act in acts {
        match act {
            Act::Batch(action) => insp.run_batch(ex, action, false),
            Act::Collection { name, add } => insp.run_collection(ex, &name, add),
        }
    }
}
