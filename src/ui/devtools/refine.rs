//! Devtools window for the Refine and Image Edit tabs: state of both, sample proposals, bulk buttons and the test backend of Image Edit.

use eframe::egui::{Context, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::icons::{self, svg};
use crate::ui::image_edit::ImageEditPanel;
use crate::ui::refine::{Confirm, RefinementsPanel};

fn open_id() -> Id {
    Id::unique("ame_devtools_refine_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_refine_snapshot")
}

fn requests_id() -> Id {
    Id::unique("ame_devtools_refine_requests")
}

fn form_id() -> Id {
    Id::unique("ame_devtools_refine_form")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// What the tabs look like right now.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub proposals: usize,
    pub shown: usize,
    pub by_source: Vec<(String, usize)>,
    pub run: String,
    pub scope: usize,
    pub stored_pending: usize,
    pub stored_error: Option<String>,
    pub confirm: Option<String>,
    pub applying: usize,
    pub applied: String,
    pub notice: Option<String>,
    pub edit_phase: String,
    pub edit_source: Option<String>,
    pub edit_history: usize,
    pub edit_debug_backend: bool,
    pub edit_prompt: String,
}

/// An action the window asks of the tabs.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Add proposals with made-up values for these files of the library.
    Sample(Vec<String>),
    ClearGenerated,
    RefreshStore,
    CancelRun,
    CheckAllShown(bool),
    AskAcceptShown,
    ConfirmPending,
    DeclinePending,
    UseDebugBackend(bool),
    SetPrompt(String),
    RunEdit,
    CancelEdit,
}

pub fn publish(ctx: &Context, snapshot: Snapshot) {
    ctx.data_mut(|d| d.insert_temp(snapshot_id(), snapshot));
}

pub fn latest(ctx: &Context) -> Option<Snapshot> {
    ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id()))
}

pub fn push_request(ctx: &Context, request: Request) {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Vec<Request>>(requests_id()).push(request));
    ctx.request_repaint();
}

pub fn take_requests(ctx: &Context) -> Vec<Request> {
    ctx.data_mut(|d| d.remove_temp::<Vec<Request>>(requests_id())).unwrap_or_default()
}

/// Paths from text, one per line, blanks and repeats skipped.
pub fn parse_paths(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    text.lines().map(str::trim).map(|l| l.trim_matches('"')).filter(|l| !l.is_empty()).filter(|l| seen.insert(l.to_string())).map(str::to_string).collect()
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Refine and Image Edit").open(&mut open).default_width(640.0).default_height(660.0).show(ctx, |ui| match latest(ctx) {
        Some(snapshot) => {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| body(ctx, ui, &snapshot));
        }
        None => {
            ui.weak("Waiting for the first frame.");
        }
    });
    set_open(ctx, open);
}

fn kv(ui: &mut Ui, key: &str, value: impl Into<String>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{key}:")).strong());
        ui.label(value.into());
    });
}

fn body(ctx: &Context, ui: &mut Ui, s: &Snapshot) {
    ui.heading("Refine");
    kv(ui, "Proposals", format!("{} ({} shown)", s.proposals, s.shown));
    for (source, n) in &s.by_source {
        kv(ui, "  Source", format!("{source}: {n}"));
    }
    kv(ui, "Run", s.run.clone());
    kv(ui, "Selection scope", format!("{} files", s.scope));
    kv(ui, "Stored proposals", format!("{} pending{}", s.stored_pending, s.stored_error.as_ref().map(|e| format!(", error: {e}")).unwrap_or_default()));
    kv(ui, "Decided", s.applied.clone());
    if s.applying > 0 {
        kv(ui, "Writing", format!("{} proposals", s.applying));
    }
    if let Some(c) = &s.confirm {
        ui.label(RichText::new(format!("Waiting for an answer: {c}")).color(crate::theme::warn(ui)));
    }
    if let Some(n) = &s.notice {
        ui.weak(n.clone());
    }
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh stored").clicked() {
            push_request(ctx, Request::RefreshStore);
        }
        if icons::button(ui, svg::STOP, "Cancel run").clicked() {
            push_request(ctx, Request::CancelRun);
        }
        if icons::button(ui, svg::BROOM, "Clear generated").clicked() {
            push_request(ctx, Request::ClearGenerated);
        }
        if icons::button(ui, svg::LIST_CHECKS, "Check all shown").clicked() {
            push_request(ctx, Request::CheckAllShown(true));
        }
        if icons::button(ui, svg::MINUS_CIRCLE, "Uncheck all shown").clicked() {
            push_request(ctx, Request::CheckAllShown(false));
        }
        if icons::button(ui, svg::CHECKS, "Ask to accept shown").clicked() {
            push_request(ctx, Request::AskAcceptShown);
        }
        if s.confirm.is_some() {
            if icons::button(ui, svg::CHECK, "Confirm").clicked() {
                push_request(ctx, Request::ConfirmPending);
            }
            if icons::button(ui, svg::X, "Decline").clicked() {
                push_request(ctx, Request::DeclinePending);
            }
        }
    });
    let mut paths = ctx.data(|d| d.get_temp::<String>(form_id())).unwrap_or_default();
    ui.label("Library files to make sample proposals for, one path per line:");
    ui.add(TextEdit::multiline(&mut paths).desired_rows(3).desired_width(f32::INFINITY).hint_text("/path/a.jpg"));
    ctx.data_mut(|d| d.insert_temp(form_id(), paths.clone()));
    if ui.add_enabled(!parse_paths(&paths).is_empty(), icons::button_widget(ui, svg::AME_PROPOSAL, "Add sample proposals")).on_hover_text("Made-up categories, tags and captions with confidences, so the tab can be tried without a model").clicked() {
        push_request(ctx, Request::Sample(parse_paths(&paths)));
    }
    ui.separator();
    ui.heading("Image Edit");
    kv(ui, "State", s.edit_phase.clone());
    kv(ui, "Source", s.edit_source.clone().unwrap_or_else(|| "none".to_string()));
    kv(ui, "History", format!("{} results", s.edit_history));
    let mut debug = s.edit_debug_backend;
    if ui.checkbox(&mut debug, "Test backend (no model: tints or grays the picture in a moment)").changed() {
        push_request(ctx, Request::UseDebugBackend(debug));
    }
    ui.label(RichText::new("Put \"simulate oom\" in the prompt to see the out-of-memory state.").small().weak());
    let mut prompt = s.edit_prompt.clone();
    if ui.add(TextEdit::singleline(&mut prompt).desired_width(f32::INFINITY).hint_text("Prompt")).changed() {
        push_request(ctx, Request::SetPrompt(prompt));
    }
    ui.horizontal(|ui| {
        if icons::button(ui, svg::AME_IMAGE_EDIT, "Run edit").clicked() {
            push_request(ctx, Request::RunEdit);
        }
        if icons::button(ui, svg::STOP, "Cancel edit").clicked() {
            push_request(ctx, Request::CancelEdit);
        }
    });
}

/// The state of both tabs for the window.
pub fn snapshot(refine: &RefinementsPanel, edit: &ImageEditPanel) -> Snapshot {
    let mut by_source: Vec<(String, usize)> = Vec::new();
    for p in refine.proposals() {
        let label = p.source.label();
        match by_source.iter_mut().find(|(l, _)| *l == label) {
            Some(slot) => slot.1 += 1,
            None => by_source.push((label, 1)),
        }
    }
    let tally = refine.tally;
    Snapshot {
        proposals: refine.proposals().len(),
        shown: refine.visible().len(),
        by_source,
        run: format!("{:?}", refine.run_state()),
        scope: refine.scope_len(),
        stored_pending: refine.store_status.pending,
        stored_error: refine.store_status.error.clone(),
        confirm: refine.confirm.as_ref().map(|c| match c {
            Confirm::Accept(plan) => format!("apply {}", plan.summary()),
            Confirm::Reject(keys) => format!("reject {} proposals", keys.len()),
        }),
        applying: refine.applying_len(),
        applied: format!("{} fields in {} files applied, {} rejected, {} failed", tally.applied_fields, tally.applied_files, tally.rejected, tally.failed),
        notice: refine.notice.as_ref().map(|(_, t)| t.clone()),
        edit_phase: edit.phase().label(),
        edit_source: edit.current_path.clone(),
        edit_history: edit.history().len(),
        edit_debug_backend: edit.uses_debug_backend(),
        edit_prompt: edit.params.prompt.clone(),
    }
}

/// Runs what the window asked for and publishes the state while it is open; called once per frame.
pub fn frame(refine: &mut RefinementsPanel, edit: &mut ImageEditPanel, ctx: &Context) {
    for request in take_requests(ctx) {
        match request {
            Request::Sample(paths) => refine.inject_samples(paths),
            Request::ClearGenerated => refine.clear_local(),
            Request::RefreshStore => {
                refine.store_status.at = None;
                refine.refresh_store();
            }
            Request::CancelRun => refine.cancel_run(),
            Request::CheckAllShown(on) => {
                let shown = refine.visible();
                crate::ui::refine::model::set_all(&mut refine.proposals, &shown, on);
            }
            Request::AskAcceptShown => refine.ask_accept_view(),
            Request::ConfirmPending => refine.confirm_pending(),
            Request::DeclinePending => refine.confirm = None,
            Request::UseDebugBackend(on) => edit.use_debug_backend(on),
            Request::SetPrompt(text) => edit.params.prompt = text,
            Request::RunEdit => {
                if let Err(e) = edit.start() {
                    log::warn!("[devtools] image edit not started: {e}");
                }
            }
            Request::CancelEdit => edit.cancel(),
        }
    }
    if is_open(ctx) {
        publish(ctx, snapshot(refine, edit));
    }
}

#[cfg(test)]
mod tests;
