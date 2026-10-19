//! Drawing of the wizard dialog and the checklist card.

use std::path::Path;

use eframe::egui::{Align, Context, Frame, Id, Layout, Margin, Modal, RichText, TextEdit, Ui};

use super::model::{Choice, Step, detect_model_folders, model_roots};
use super::Wizard;
use crate::command::AppCommand;
use crate::icons::{self, svg};
use crate::settings::location::{Check, Via};
use crate::settings::ui_prefs::{self, AiMode, StepState};
use crate::ui::dialogs;
use crate::ui::file_table::FileExplorer;

const SLOT_LOCATION: &str = "wizard-location-folder";
const SLOT_FOLDER: &str = "wizard-add-folder";
const SLOT_VISION: &str = "wizard-vision-folder";
const SLOT_CARD: &str = "wizard-card-scan-folder";
/// Dialog width in points.
const WIDTH: f32 = 560.0;

fn state_icon(state: StepState) -> icons::Icon {
    match state {
        StepState::Done => svg::CHECK_CIRCLE,
        StepState::Skipped => svg::MINUS_CIRCLE,
        StepState::Pending => svg::CIRCLE,
    }
}

/// Draws the wizard when it is open; `library_open` is whether the library is open.
pub fn show(ctx: &Context, wizard: &mut Wizard, library_open: bool) {
    if !wizard.open {
        return;
    }
    take_picked(wizard);
    let width = (ctx.content_rect().width() - 48.0).clamp(300.0, WIDTH);
    let frame = Frame::window(&ctx.global_style()).inner_margin(Margin::same(16));
    Modal::new(Id::unique("ame_first_run_wizard")).frame(frame).show(ctx, |ui| {
        ui.set_width(width - 32.0);
        header(ui, wizard);
        ui.separator();
        match wizard.current() {
            Step::Location => location_step(ui, wizard, library_open),
            Step::Folders => folders_step(ui, wizard),
            Step::Ai => ai_step(ui, wizard),
            Step::Scan => scan_step(ui, wizard),
        }
        if let Some(error) = &wizard.error {
            ui.add_space(6.0);
            ui.colored_label(ui.visuals().error_fg_color, error.as_str());
        }
        ui.add_space(8.0);
        ui.separator();
        footer(ui, wizard);
    });
}

fn take_picked(wizard: &mut Wizard) {
    if let Some(dir) = dialogs::take_picked(SLOT_LOCATION) {
        if let Some(form) = wizard.location.as_mut() {
            form.choice = Choice::Custom;
            form.custom = dir.display().to_string();
        }
    }
    if let Some(dir) = dialogs::take_picked(SLOT_FOLDER) {
        wizard.folders.add(&dir.display().to_string(), true);
    }
    if let Some(dir) = dialogs::take_picked(SLOT_VISION) {
        wizard.ai.vision_dir = dir.display().to_string();
    }
}

fn header(ui: &mut Ui, wizard: &Wizard) {
    let step = wizard.current();
    ui.horizontal(|ui| {
        icons::heading(ui, svg::AME_LOGO, "Set up your library");
    });
    ui.label(format!("Step {} of {}: {}", step.index() + 1, Step::ALL.len(), step.label()));
    let checklist = ui_prefs::get().checklist;
    ui.horizontal_wrapped(|ui| {
        for s in Step::ALL {
            let text = RichText::new(s.label());
            let text = if s == step { text.strong() } else { text.weak() };
            icons::label(ui, state_icon(checklist.state(s)), text);
        }
    });
}

fn location_step(ui: &mut Ui, wizard: &mut Wizard, library_open: bool) {
    ui.add_space(4.0);
    ui.label("The library is a database folder that holds your tags, descriptions and thumbnails. Your photos and videos stay where they are.");
    ui.add_space(6.0);
    let Some(form) = wizard.location.as_mut() else { return };
    for (index, candidate) in form.detected.clone().iter().enumerate() {
        let via = match candidate.via {
            Via::PointerFile => "named by db/path.txt",
            Via::DefaultFolder => "found in db/ai_search",
        };
        ui.radio_value(&mut form.choice, Choice::Legacy(index), format!("Use the library found at {} ({via})", candidate.path.display()));
    }
    ui.radio_value(&mut form.choice, Choice::Default, format!("Create a new library at {}", form.default_path.display()));
    ui.horizontal_wrapped(|ui| {
        ui.radio_value(&mut form.choice, Choice::Custom, "Another folder:");
        let response = ui.add(TextEdit::singleline(&mut form.custom).hint_text("Folder of the library").desired_width(260.0));
        if response.changed() {
            form.choice = Choice::Custom;
        }
        if icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked() {
            dialogs::pick_folder_into(ui.ctx(), "Choose the library folder", SLOT_LOCATION);
        }
    });
    ui.add_space(4.0);
    match form.check() {
        Check::Existing => {
            ui.label("This folder holds a library. It is opened as it is, nothing is overwritten.");
        }
        Check::New => {
            ui.label("A new library is created here. Nothing that exists is overwritten.");
        }
        Check::Rejected(why) => {
            ui.colored_label(ui.visuals().error_fg_color, why);
        }
    }
    if library_open {
        ui.weak("A library is open already. A different folder opens the next time the application starts.");
    }
}

fn folders_step(ui: &mut Ui, wizard: &mut Wizard) {
    ui.add_space(4.0);
    ui.label("Choose the folders the library is built from. You can add more later in Settings > Library.");
    ui.add_space(4.0);
    if wizard.folders.entries.is_empty() {
        ui.weak("No Pictures or Videos folder was found. Add a folder below.");
    }
    for (path, on) in wizard.folders.entries.iter_mut() {
        ui.checkbox(on, path.as_str());
    }
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::FOLDER_PLUS, "Add folder...").clicked() {
            dialogs::pick_folder_into(ui.ctx(), "Add a folder to the library", SLOT_FOLDER);
        }
        ui.add(TextEdit::singleline(&mut wizard.folders.input).hint_text("or type a path").desired_width(220.0));
        if ui.add_enabled(!wizard.folders.input.trim().is_empty(), eframe::egui::Button::new("Add")).clicked() {
            let text = std::mem::take(&mut wizard.folders.input);
            wizard.folders.add(&text, true);
        }
    });
}

fn ai_step(ui: &mut Ui, wizard: &mut Wizard) {
    ui.add_space(4.0);
    ui.label("AI features describe files and search them by meaning. Pick how they run; this can change in Settings at any time.");
    ui.add_space(4.0);
    for mode in AiMode::ALL {
        ui.radio_value(&mut wizard.ai.mode, mode, mode.label());
    }
    ui.add_space(6.0);
    match wizard.ai.mode {
        AiMode::Off => {
            ui.weak("Nothing is loaded and no model is needed. Browsing, tags and collections work as usual.");
        }
        AiMode::Local => local_models(ui, wizard),
        AiMode::Zeroclaw => {
            wizard.zc_done = crate::zc::onboarding_ui(ui);
        }
    }
}

fn local_models(ui: &mut Ui, wizard: &mut Wizard) {
    ui.label("Vision model folder");
    ui.horizontal_wrapped(|ui| {
        ui.add(TextEdit::singleline(&mut wizard.ai.vision_dir).hint_text("Folder with the model").desired_width(300.0));
        if icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked() {
            dialogs::pick_folder_into(ui.ctx(), "Choose the vision model folder", SLOT_VISION);
        }
        if ui.button("Detect").on_hover_text("Look for model folders in the usual places").clicked() {
            wizard.ai.found = detect_model_folders(&model_roots(), 6);
        }
    });
    match wizard.ai.vision_check() {
        Some(check) if check.is_ok() => {
            ui.label(check.text());
        }
        Some(check) => {
            ui.colored_label(ui.visuals().warn_fg_color, check.text());
        }
        None => {
            ui.weak("Without a folder, describing files is not available yet. CLIP search downloads its model when first used.");
        }
    }
    for dir in wizard.ai.found.clone() {
        if ui.link(dir.display().to_string()).clicked() {
            wizard.ai.vision_dir = dir.display().to_string();
        }
    }
}

fn scan_step(ui: &mut Ui, wizard: &mut Wizard) {
    ui.add_space(4.0);
    let watched = ui_prefs::get().library.watched;
    if watched.is_empty() {
        ui.label("There are no folders to scan yet. Go back and add a folder, or skip this step.");
    } else {
        ui.label("These folders are scanned, one tab each:");
        for path in &watched {
            icons::label(ui, svg::FOLDER, path.as_str());
        }
    }
    ui.add_space(4.0);
    ui.checkbox(&mut wizard.save_scanned, "Save scanned files to the library").on_hover_text("Without this the scan only lists files; nothing is stored");
}

fn footer(ui: &mut Ui, wizard: &mut Wizard) {
    let step = wizard.current();
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        let (primary, can_go) = match step {
            Step::Location => ("Continue", wizard.location.as_ref().is_some_and(|f| f.check().is_usable())),
            Step::Folders => ("Continue", true),
            Step::Ai => ("Continue", wizard.ai.mode != AiMode::Zeroclaw || wizard.zc_done),
            Step::Scan => ("Start scan", !ui_prefs::get().library.watched.is_empty()),
        };
        if ui.add_enabled(can_go, crate::theme::primary_button(primary)).clicked() {
            advance(wizard, step);
        }
        let skip = if step == Step::Location { "Use the default" } else { "Skip this step" };
        if ui.button(skip).clicked() {
            skip_step(wizard, step);
        }
        if step.prev().is_some() && ui.button("Back").clicked() {
            wizard.back();
        }
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            if ui.add(eframe::egui::Button::new(RichText::new("Skip setup").weak()).frame(false)).on_hover_text("Skips every step; a missing library gets the default folder").clicked() {
                if let Err(e) = wizard.skip_all() {
                    wizard.error = Some(e);
                }
            }
        });
    });
}

fn advance(wizard: &mut Wizard, step: Step) {
    let result: Result<(), String> = match step {
        Step::Location => wizard.continue_location().map(|_| ()),
        Step::Folders => {
            wizard.continue_folders();
            Ok(())
        }
        Step::Ai => {
            wizard.continue_ai();
            Ok(())
        }
        Step::Scan => {
            wizard.start_scan();
            Ok(())
        }
    };
    if let Err(e) = result {
        wizard.error = Some(e);
    }
}

fn skip_step(wizard: &mut Wizard, step: Step) {
    if step == Step::Location {
        if let Err(e) = wizard.use_default_location() {
            wizard.error = Some(e);
        }
    } else {
        wizard.finish_step(StepState::Skipped);
    }
}

/// True when `ex` shows an empty library and the checklist still has work.
pub fn card_applies(ex: &FileExplorer) -> bool {
    super::is_empty_library(ex.scope(), ex.table.len(), ex.is_loading()) && ui_prefs::get().checklist.card_visible(true)
}

/// Draws the checklist card of the setup steps.
pub fn checklist_card(ui: &mut Ui) {
    let checklist = ui_prefs::get().checklist;
    if let Some(dir) = dialogs::take_picked(SLOT_CARD) {
        crate::command::send(crate::ui::shell::open_folder_command(Path::new(&dir), true));
    }
    let (done, total) = checklist.progress();
    Frame::group(ui.style()).inner_margin(Margin::same(12)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Set up your library").strong().size(15.0));
            ui.weak(format!("{done} of {total} steps"));
        });
        for step in Step::ALL {
            let state = checklist.state(step);
            ui.horizontal(|ui| {
                icons::label(ui, state_icon(state), step.label());
                ui.weak(state.label());
                if state != StepState::Done && ui.small_button("Open").clicked() {
                    crate::command::send(AppCommand::Wizard { step: Some(step) });
                }
            });
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if ui.add(crate::theme::primary_button("Scan a folder...")).clicked() {
                dialogs::pick_folder_into(ui.ctx(), "Scan a folder", SLOT_CARD);
            }
            if ui.button("Open setup").clicked() {
                crate::command::send(AppCommand::Wizard { step: None });
            }
            if ui.button("Hide").on_hover_text("Settings > Advanced can bring the checklist back").clicked() {
                ui_prefs::update(|p| p.checklist.dismissed = true);
            }
        });
    });
    ui.add_space(6.0);
}
