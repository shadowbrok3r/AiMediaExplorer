//! Devtools window for the first-run wizard: its state, a force button per step, the checklist and ui_prefs.json.

use eframe::egui::{CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::command::{AppCommand, Section};
use crate::settings::ui_prefs::{self, StepState};
use crate::ui::wizard::model::Step;
use crate::ui::wizard::{Reason, Wizard};

fn open_id() -> Id {
    Id::unique("ame_devtools_wizard_open")
}

fn info_id() -> Id {
    Id::unique("ame_devtools_wizard_info")
}

fn requests_id() -> Id {
    Id::unique("ame_devtools_wizard_requests")
}

/// What the window asks the application to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Close,
    /// Reset the checklist and open the wizard at its first step.
    SimulateFirstRun,
}

/// What the wizard was doing at the last publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub open: bool,
    pub step: Step,
    pub reason: Reason,
    pub zc_done: bool,
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Records the wizard state for the window.
pub fn publish(ctx: &Context, wizard: &Wizard) {
    if !is_open(ctx) {
        return;
    }
    let info = Info { open: wizard.open, step: wizard.current(), reason: wizard.reason, zc_done: wizard.zc_done };
    ctx.data_mut(|d| d.insert_temp(info_id(), info));
}

/// Takes the buttons pressed since the last call.
pub fn take_requests(ctx: &Context) -> Vec<Request> {
    ctx.data_mut(|d| d.remove_temp::<Vec<Request>>(requests_id())).unwrap_or_default()
}

/// Queues `what` as the window's buttons do.
pub fn request(ctx: &Context, what: Request) {
    ctx.data_mut(|d| {
        let mut list = d.get_temp::<Vec<Request>>(requests_id()).unwrap_or_default();
        list.push(what);
        d.insert_temp(requests_id(), list);
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Wizard").open(&mut open).default_width(520.0).default_height(560.0).show(ctx, |ui| body(ui));
    set_open(ctx, open);
}

/// Window contents.
pub fn body(ui: &mut Ui) {
    let info = ui.ctx().data(|d| d.get_temp::<Info>(info_id()));
    Grid::new("wizard_dev_info").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label("Wizard open");
        ui.monospace(info.map(|i| i.open.to_string()).unwrap_or_else(|| "unknown until the next frame".to_string()));
        ui.end_row();
        ui.label("Current step");
        ui.monospace(info.map(|i| i.step.slug().to_string()).unwrap_or_default());
        ui.end_row();
        ui.label("Opened because");
        ui.monospace(info.map(|i| format!("{:?}", i.reason)).unwrap_or_default());
        ui.end_row();
        ui.label("ZeroClaw step done");
        ui.monospace(info.map(|i| i.zc_done.to_string()).unwrap_or_default());
        ui.end_row();
    });

    ui.add_space(6.0);
    ui.label(RichText::new("Force a step").strong());
    ui.horizontal_wrapped(|ui| {
        for step in Step::ALL {
            if ui.button(step.label()).on_hover_text(format!("{{\"cmd\":\"wizard\",\"step\":\"{}\"}}", step.slug())).clicked() {
                crate::command::send(AppCommand::Wizard { step: Some(step) });
            }
        }
        if ui.button("First pending").clicked() {
            crate::command::send(AppCommand::Wizard { step: None });
        }
        if ui.button("Close wizard").clicked() {
            request(ui.ctx(), Request::Close);
        }
    });
    if ui.button("Simulate first run").on_hover_text("Resets the checklist and opens the wizard at the first step").clicked() {
        request(ui.ctx(), Request::SimulateFirstRun);
    }

    ui.separator();
    ui.label(RichText::new("Checklist").strong());
    let checklist = ui_prefs::get().checklist;
    for step in Step::ALL {
        ui.horizontal(|ui| {
            ui.label(format!("{:<18}", step.label()));
            for state in [StepState::Pending, StepState::Done, StepState::Skipped] {
                if ui.selectable_label(checklist.state(step) == state, state.label()).clicked() {
                    ui_prefs::update(|p| p.checklist.set(step, state));
                }
            }
        });
    }
    ui.horizontal(|ui| {
        let mut dismissed = checklist.dismissed;
        if ui.checkbox(&mut dismissed, "Card dismissed").changed() {
            ui_prefs::update(|p| p.checklist.dismissed = dismissed);
        }
        ui.label(format!("AI mode: {}", checklist.ai_mode.label()));
        if ui.button("Reset checklist").clicked() {
            ui_prefs::update(|p| p.checklist = Default::default());
        }
    });
    ui.horizontal_wrapped(|ui| {
        if ui.button("Open Settings > Library").clicked() {
            crate::command::send(AppCommand::ShowSettings { section: Some(Section::Library) });
        }
    });

    ui.separator();
    CollapsingHeader::new("ui_prefs.json").default_open(true).show(ui, |ui| {
        let (path, note, saves, error) = ui_prefs::with(|s| (s.path().map(|p| p.display().to_string()), s.note.text(), s.saves, s.last_error.clone()));
        ui.weak(path.unwrap_or_else(|| "memory only".to_string()));
        ui.label(format!("Load: {note}; written {saves} times"));
        if let Some(error) = error {
            ui.colored_label(ui.visuals().error_fg_color, format!("Last write failed: {error}"));
        }
        let text = crate::ui::settings::advanced::ui_prefs_text();
        let mut view = text.as_str();
        ScrollArea::vertical().id_salt("wizard_dev_json").max_height(260.0).show(ui, |ui| {
            ui.add(TextEdit::multiline(&mut view).code_editor().desired_width(f32::INFINITY));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{CentralPanel, Pos2, RawInput, Rect, vec2};

    fn frame(ctx: &Context) -> String {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 800.0))), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| body(ui));
        });
        out.textures_delta.clear();
        crate::ui::test_support::texts(&out.shapes).into_iter().map(|(t, _)| t).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn the_window_lists_the_steps_the_checklist_and_the_prefs_file() {
        let ctx = crate::ui::test_support::test_ctx();
        frame(&ctx);
        let text = frame(&ctx);
        for step in Step::ALL {
            assert!(text.contains(step.label()), "{}:\n{text}", step.label());
        }
        assert!(text.contains("Simulate first run") && text.contains("Reset checklist"));
        assert!(text.contains("ui_prefs.json"));
    }

    #[test]
    fn requests_are_queued_and_taken_once() {
        let ctx = Context::default();
        assert!(take_requests(&ctx).is_empty());
        request(&ctx, Request::Close);
        request(&ctx, Request::SimulateFirstRun);
        assert_eq!(take_requests(&ctx), vec![Request::Close, Request::SimulateFirstRun]);
        assert!(take_requests(&ctx).is_empty());
    }

    #[test]
    fn publishing_records_the_state_only_while_the_window_is_open() {
        let ctx = Context::default();
        let mut wizard = Wizard::default();
        wizard.open = true;
        publish(&ctx, &wizard);
        assert!(ctx.data(|d| d.get_temp::<Info>(info_id())).is_none());
        set_open(&ctx, true);
        publish(&ctx, &wizard);
        let info = ctx.data(|d| d.get_temp::<Info>(info_id())).unwrap();
        assert!(info.open);
        assert_eq!(info.step, Step::Location);
    }
}
