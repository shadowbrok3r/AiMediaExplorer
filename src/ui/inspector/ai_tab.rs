//! The AI and Edit tabs: describing with streamed text for the focused file, CLIP, similar files, refining and the image editor.

use egui::{RichText, ScrollArea, Ui};

use super::target::{AiState, can_describe, current_ai_state, has_description, is_dir, is_image_path, stream_view};
use crate::Thumbnail;
use crate::command::{ActionId, AppCommand, Scope, Section, Selector, Target as OpenTarget, Tool};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::table::actions::MenuAction;
use crate::ui::file_table::table::cells::has_clip;

/// Starts describing `row` and shows the streaming text at once.
pub fn describe(ex: &mut FileExplorer, row: &Thumbnail) {
    ex.streaming_interim.insert(row.path.clone(), String::new());
    ex.viewer.run_action(MenuAction::RegenerateDescription, &[row]);
}

/// Queues CLIP generation for `row` through the command bus.
pub fn generate_clip(row: &Thumbnail) {
    crate::command::send(AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Paths(vec![row.path.clone()]) });
}

pub fn find_similar(row: &Thumbnail) {
    crate::command::send(AppCommand::open(Scope::Similar(row.path.clone())));
}

pub fn edit_image(row: &Thumbnail) {
    crate::command::send(AppCommand::OpenTool { tool: Tool::ImageEdit, path: Some(row.path.clone()), target: OpenTarget::NewTab });
}

fn not_configured(ui: &mut Ui) {
    ui.group(|ui| {
        icons::label(ui, svg::WARNING, RichText::new("AI is not configured").strong());
        ui.label("Choose the vision model folder to describe files.");
        if icons::button(ui, svg::GEAR, "Choose the model folder").clicked() {
            crate::command::send(AppCommand::ShowSettings { section: Some(Section::AiModels) });
        }
    });
}

pub fn show(ui: &mut Ui, ex: &mut FileExplorer, row: &Thumbnail) {
    if is_dir(row) {
        ui.weak("Folders cannot be described.");
        return;
    }
    let ai = current_ai_state();
    if ai == AiState::NotConfigured {
        not_configured(ui);
        ui.add_space(6.0);
    }
    let streaming = stream_view(Some(row.path.as_str()), &ex.streaming_interim).map(str::to_string);
    ui.label(RichText::new("Description").strong());
    match &streaming {
        Some(text) => {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0));
                ui.label("Generating...");
                if ui.small_button("Dismiss").on_hover_text("Stop showing this stream").clicked() {
                    ex.streaming_interim.remove(&row.path);
                }
            });
            ScrollArea::vertical().id_salt("inspector_stream").max_height(170.0).show(ui, |ui| {
                ui.label(if text.is_empty() { "Waiting for the model..." } else { text.as_str() });
            });
        }
        None => {
            ui.label(RichText::new(if has_description(row) { "This file has a description. See the Metadata tab." } else { "This file has no description yet." }).weak());
        }
    }
    let describable = can_describe(row);
    let ready = ai == AiState::Ready && describable && streaming.is_none();
    ui.horizontal(|ui| {
        let (label, icon) = if has_description(row) { ("Regenerate", svg::AME_REGENERATE) } else { ("Describe", svg::AME_AI_DESCRIBE) };
        let hover = if !describable { "Only local pictures can be described" } else if ai == AiState::NotConfigured { "Choose a vision model folder first" } else { "Describe this file with the vision model" };
        if ui.add_enabled(ready, icons::button_widget(ui, icon, label)).on_hover_text(hover).clicked() {
            describe(ex, row);
        }
    });
    let mut follow = ex.follow_active_vision;
    if ui.checkbox(&mut follow, "Follow descriptions of other files").on_hover_text("Off by default: the rail keeps showing the file you selected").changed() {
        ex.follow_active_vision = follow;
    }
    ui.separator();
    ui.label(RichText::new("CLIP").strong());
    let has = has_clip(&ex.viewer, row);
    ui.label(RichText::new(if has { "An embedding exists for this file." } else { "No embedding yet." }).weak());
    ui.horizontal(|ui| {
        if icons::button(ui, svg::AME_CLIP_EMBED, if has { "Regenerate CLIP" } else { "Generate CLIP" }).clicked() {
            generate_clip(row);
        }
        let similar = ui.add_enabled(is_image_path(&row.path), icons::button_widget(ui, svg::AME_FIND_SIMILAR, "Find similar")).on_hover_text("Open the files most like this one");
        if similar.clicked() {
            find_similar(row);
        }
    });
    ui.separator();
    super::suggest::show(ui, row);
    ui.separator();
    ui.label(RichText::new("Refine").strong());
    ui.label(RichText::new("Review AI proposals for the tags and category of this file.").weak());
    if icons::button(ui, svg::AME_REFINE, "Refine this item").clicked() {
        crate::ui::refine::request_refine_for_paths(vec![row.path.clone()]);
    }
}

pub fn edit_tab(ui: &mut Ui, row: &Thumbnail) {
    if is_dir(row) {
        ui.weak("Folders cannot be edited.");
        return;
    }
    ui.label("Open this picture in the Image Edit tool to change it with a prompt.");
    let editable = is_image_path(&row.path);
    if ui.add_enabled(editable, icons::button_widget(ui, svg::AME_IMAGE_EDIT, "Open in Image Edit")).on_disabled_hover_text("Only pictures can be edited").clicked() {
        edit_image(row);
    }
}
