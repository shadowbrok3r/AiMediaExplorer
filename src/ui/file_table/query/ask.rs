//! The Ask mode of the query bar: a sentence goes to the ZeroClaw gateway, comes back as a filter, and is shown as chips with an Apply button before anything changes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crossbeam::channel::{Receiver, bounded};
use eframe::egui::*;

use crate::ai::mcp::ctx::DbSource;
use crate::icons::{self, svg};
use crate::ui::file_table::rows::Chip;
use crate::zc::jobs::ZcJobs;
use crate::zc::jobs::ask::{self, Preview};

/// Where the Ask mode is.
#[derive(Clone, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    Working { sentence: String, started: Instant },
    /// The filter the sentence stands for, waiting for Apply.
    Ready(Preview),
    /// The reply was not a filter or the gateway failed; nothing was changed.
    Failed { sentence: String, error: String },
}

type Answer = Result<Preview, String>;

/// State of the Ask mode of one query bar.
#[derive(Default)]
pub struct AskState {
    pub phase: Phase,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Answer>>,
    jobs: Option<Arc<ZcJobs>>,
    db: Option<DbSource>,
    /// Questions asked so far.
    pub asked: u64,
    /// Filters applied so far.
    pub applied: u64,
}

impl std::fmt::Debug for AskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AskState").field("phase", &self.phase).field("asked", &self.asked).field("applied", &self.applied).finish_non_exhaustive()
    }
}

impl AskState {
    /// An Ask state that uses `jobs` and `db`, for tests and tools with their own gateway.
    pub fn with(jobs: Arc<ZcJobs>, db: DbSource) -> Self {
        Self { jobs: Some(jobs), db: Some(db), ..Default::default() }
    }

    pub fn use_backend(&mut self, jobs: Arc<ZcJobs>, db: DbSource) {
        self.jobs = Some(jobs);
        self.db = Some(db);
    }

    pub fn jobs(&self) -> Arc<ZcJobs> {
        self.jobs.clone().unwrap_or_else(|| crate::zc::jobs::jobs().clone())
    }

    fn db(&self) -> DbSource {
        self.db.clone().unwrap_or(DbSource::Global)
    }

    /// Why the Ask mode cannot be used now; none while the gateway is Online or still being checked.
    pub fn unavailable(&self) -> Option<String> {
        let jobs = self.jobs();
        match jobs.availability() {
            availability if availability.is_ready() || jobs.is_settling() => None,
            availability => availability.reason().map(str::to_string),
        }
    }

    pub fn is_working(&self) -> bool {
        matches!(self.phase, Phase::Working { .. })
    }

    /// The preview that waits for Apply.
    pub fn preview(&self) -> Option<&Preview> {
        match &self.phase {
            Phase::Ready(p) => Some(p),
            _ => None,
        }
    }

    /// Sends `sentence` to the gateway. Fails at once, changing nothing, when Ask cannot be used or there is nothing to ask.
    pub fn start(&mut self, sentence: &str) -> Result<(), String> {
        let sentence = sentence.trim().to_string();
        if sentence.is_empty() {
            return Err("Describe the files to look for first".to_string());
        }
        if let Some(reason) = self.unavailable() {
            return Err(reason);
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else { return Err("no async runtime is available".to_string()) };
        self.cancel.store(true, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let (tx, rx) = bounded(1);
        self.rx = Some(rx);
        self.asked += 1;
        self.phase = Phase::Working { sentence: sentence.clone(), started: Instant::now() };
        let (jobs, db) = (self.jobs(), self.db());
        runtime.spawn(async move {
            let answer = ask::ask(&jobs, &db, &sentence, None, &cancel).await;
            let _ = tx.send(answer);
            crate::command::request_repaint();
        });
        Ok(())
    }

    /// Takes in the answer when it has arrived; true when the phase changed.
    pub fn poll(&mut self) -> bool {
        let Some(rx) = &self.rx else { return false };
        let Ok(answer) = rx.try_recv() else { return false };
        self.rx = None;
        let sentence = match &self.phase {
            Phase::Working { sentence, .. } => sentence.clone(),
            _ => String::new(),
        };
        self.phase = match answer {
            Ok(preview) => Phase::Ready(preview),
            Err(e) if e == "Cancelled" => Phase::Idle,
            Err(error) => Phase::Failed { sentence, error },
        };
        true
    }

    /// Stops the question in flight.
    pub fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if self.is_working() {
            self.phase = Phase::Idle;
            self.rx = None;
        }
    }

    /// Drops the preview or the error.
    pub fn dismiss(&mut self) {
        if !self.is_working() {
            self.phase = Phase::Idle;
        }
    }

    /// Hands out the preview for applying and goes back to idle.
    pub fn take_preview(&mut self) -> Option<Preview> {
        match std::mem::take(&mut self.phase) {
            Phase::Ready(p) => Some(p),
            other => {
                self.phase = other;
                None
            }
        }
    }
}

/// What the user pressed in the preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Apply,
    Dismiss,
    Cancel,
}

fn chip_pill(ui: &mut Ui, chip: &Chip) {
    let fill = ui.visuals().widgets.inactive.weak_bg_fill;
    ui.label(RichText::new(format!(" {} ", chip.label)).small().background_color(fill));
}

/// Draws the preview under the query field and returns what the user pressed; draws nothing while idle.
pub fn preview_popup(ctx: &Context, id: Id, field: Rect, state: &AskState, replaces_filters: bool, unavailable: Option<&str>) -> Option<Action> {
    if matches!(state.phase, Phase::Idle) && unavailable.is_none() {
        return None;
    }
    let mut action = None;
    let width = field.width().clamp(280.0, 520.0);
    Area::new(id).order(Order::Foreground).fixed_pos(field.left_bottom() + vec2(0.0, 3.0)).show(ctx, |ui| {
        Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(width);
            match (&state.phase, unavailable) {
                (Phase::Idle, Some(reason)) => {
                    ui.label(RichText::new(format!("Ask is unavailable: {reason}")).color(crate::theme::warn(ui)));
                }
                (Phase::Working { sentence, .. }, _) => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Asking ZeroClaw...");
                        if icons::button(ui, svg::STOP, "Cancel").clicked() {
                            action = Some(Action::Cancel);
                        }
                    });
                    ui.label(RichText::new(sentence).small().weak());
                }
                (Phase::Ready(preview), _) => {
                    ui.label(RichText::new(format!("Filters for \"{}\"", crate::ui::refine::model::clipped(&preview.sentence, 80))).strong());
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = vec2(6.0, 4.0);
                        for chip in &preview.chips {
                            chip_pill(ui, chip);
                        }
                    });
                    let note = if replaces_filters { "Replaces the filters of this tab. Rows are only hidden, never removed." } else { "Rows are only hidden, never removed." };
                    ui.label(RichText::new(format!("{note} Answered by {}.", if preview.model.is_empty() { "the gateway" } else { preview.model.as_str() })).small().weak());
                    ui.horizontal(|ui| {
                        if icons::button(ui, svg::CHECK, "Apply").on_hover_text("Apply these filters to the tab (Enter again does the same)").clicked() {
                            action = Some(Action::Apply);
                        }
                        if icons::button(ui, svg::X, "Dismiss").clicked() {
                            action = Some(Action::Dismiss);
                        }
                    });
                }
                (Phase::Failed { error, .. }, _) => {
                    ui.label(RichText::new(crate::ui::refine::model::clipped(error, 300)).color(crate::theme::error(ui)));
                    ui.label(RichText::new("Nothing was changed.").small().weak());
                    if icons::button(ui, svg::X, "Dismiss").clicked() {
                        action = Some(Action::Dismiss);
                    }
                }
                (Phase::Idle, None) => {}
            }
        });
    });
    action
}

#[cfg(test)]
mod tests;
