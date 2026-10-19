//! First-run wizard: a blocking dialog that sets up the library location, folders, AI mode and the first scan, and the checklist card that remembers what is left.

pub mod model;
mod view;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use crate::command::{AppCommand, Scope};
use crate::settings::ui_prefs::{self, AiMode, StepState};

pub use model::Step;
use model::{AiForm, Choice, FoldersForm, LocationForm};
pub use view::{card_applies, checklist_card, show};

/// Why the wizard opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reason {
    /// No library was found at the configured folder.
    #[default]
    NoLibrary,
    /// A command, the checklist card or Devtools asked for it.
    Asked,
}

/// What finishing a step did, for the status line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The library folder was set and opens now.
    Opening(PathBuf),
    /// The library folder was set and opens on the next start.
    NextStart(PathBuf),
}

/// The wizard state.
#[derive(Default)]
pub struct Wizard {
    pub open: bool,
    pub step: Step,
    pub reason: Reason,
    pub location: Option<LocationForm>,
    pub folders: FoldersForm,
    pub ai: AiForm,
    pub save_scanned: bool,
    pub error: Option<String>,
    pub zc_done: bool,
    forms_ready: bool,
}

impl Wizard {
    /// Opens at the first step because no library exists.
    pub fn open_first_run(&mut self) {
        self.open_at(Step::Location, Reason::NoLibrary);
    }

    /// Opens at `step`.
    pub fn open_at(&mut self, step: Step, reason: Reason) {
        self.build_forms();
        self.open = true;
        self.step = step;
        self.reason = reason;
        self.error = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.error = None;
    }

    pub fn current(&self) -> Step {
        self.step
    }

    /// Builds the forms from the machine and the saved preferences.
    pub fn build_forms(&mut self) {
        let prefs = ui_prefs::get();
        self.location = Some(LocationForm::here());
        self.folders = FoldersForm::prefilled(&model::default_folders(), &prefs.library.watched);
        self.ai = AiForm { mode: prefs.checklist.ai_mode, vision_dir: crate::settings::snapshot().joycaption_model_dir.unwrap_or_default(), found: Vec::new() };
        self.save_scanned = true;
        self.zc_done = false;
        self.forms_ready = true;
    }

    /// Moves to the step before, when there is one.
    pub fn back(&mut self) {
        if let Some(prev) = self.current().prev() {
            self.step = prev;
            self.error = None;
        }
    }

    /// Records `state` for the current step and moves on; the last step closes the wizard.
    pub fn finish_step(&mut self, state: StepState) {
        let step = self.current();
        ui_prefs::update(|p| p.checklist.set(step, state));
        match step.next() {
            Some(next) => self.step = next,
            None => self.close(),
        }
        self.error = None;
    }

    /// Takes the chosen library folder; false and an error text when it cannot be used.
    pub fn continue_location(&mut self) -> Result<Outcome, String> {
        let form = self.location.as_ref().ok_or_else(|| "The wizard is not ready".to_string())?;
        let (path, _) = form.resolve()?;
        let outcome = self.set_library_folder(path)?;
        self.finish_step(StepState::Done);
        Ok(outcome)
    }

    /// Takes the default folder for the library, the way skipping the first step does.
    pub fn use_default_location(&mut self) -> Result<Outcome, String> {
        if let Some(form) = self.location.as_mut() {
            form.choice = Choice::Default;
        }
        self.continue_location()
    }

    fn set_library_folder(&self, path: PathBuf) -> Result<Outcome, String> {
        crate::database::set_db_path(&path).map_err(|e| format!("Could not use {}: {e}", path.display()))?;
        if crate::database::is_connected() {
            return Ok(Outcome::NextStart(path));
        }
        crate::command::send(AppCommand::ConnectDatabase);
        Ok(Outcome::Opening(path))
    }

    /// Saves the ticked folders as the watched folders.
    pub fn continue_folders(&mut self) -> usize {
        let chosen = self.folders.chosen();
        ui_prefs::update(|p| {
            for path in &chosen {
                p.watch(path);
            }
        });
        self.finish_step(if chosen.is_empty() { StepState::Skipped } else { StepState::Done });
        chosen.len()
    }

    /// Records the AI mode and, for local models, the vision folder.
    pub fn continue_ai(&mut self) {
        self.continue_ai_with(crate::settings::store());
    }

    /// [`Self::continue_ai`] writing the vision folder into `store`.
    pub fn continue_ai_with(&mut self, store: &std::sync::Arc<crate::settings::SettingsStore>) {
        let mode = self.ai.mode;
        ui_prefs::update(|p| p.checklist.ai_mode = mode);
        if mode == AiMode::Local {
            let dir = self.ai.vision_dir.trim().to_string();
            if !dir.is_empty() {
                store.update("wizard:ai", |s| s.joycaption_model_dir = Some(dir));
            }
        }
        self.finish_step(StepState::Done);
    }

    /// Starts a recursive scan of each watched folder and finishes the wizard.
    pub fn start_scan(&mut self) -> usize {
        self.start_scan_with(crate::settings::store())
    }

    /// [`Self::start_scan`] writing the save switch into `store`.
    pub fn start_scan_with(&mut self, store: &std::sync::Arc<crate::settings::SettingsStore>) -> usize {
        let watched = ui_prefs::get().library.watched;
        if self.save_scanned {
            store.update("wizard:scan", |s| s.auto_save_to_database = true);
        }
        for path in &watched {
            crate::command::send(crate::ui::shell::open_folder_command(std::path::Path::new(path), true));
        }
        self.finish_step(if watched.is_empty() { StepState::Skipped } else { StepState::Done });
        watched.len()
    }

    /// Skips every step that is still to do; a missing library gets the default folder.
    pub fn skip_all(&mut self) -> Result<(), String> {
        if !ui_prefs::get().checklist.location.is_finished() {
            self.step = Step::Location;
            self.use_default_location()?;
        }
        ui_prefs::update(|p| {
            for step in Step::ALL {
                if !p.checklist.state(step).is_finished() {
                    p.checklist.set(step, StepState::Skipped);
                }
            }
        });
        self.close();
        Ok(())
    }

    /// Opens the wizard at the first unfinished step, or the first step when all are done.
    pub fn open_next_pending(&mut self) {
        let step = ui_prefs::get().checklist.first_pending().unwrap_or(Step::Location);
        self.open_at(step, Reason::Asked);
    }
}

/// Marks the location step done when a library is already configured and found.
pub fn mark_location_done_if_present() {
    ui_prefs::update(|p| {
        if p.checklist.location == StepState::Pending {
            p.checklist.location = StepState::Done;
        }
    });
}

/// True when the explorer is an empty, loaded library listing.
pub fn is_empty_library(scope: &Scope, rows: usize, loading: bool) -> bool {
    *scope == Scope::Library && rows == 0 && !loading
}
