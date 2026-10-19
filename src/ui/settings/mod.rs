//! The Settings page: a dock tab with a section list, a search box and Apply and Revert for the edited settings.

pub mod advanced;
pub mod appearance;
pub mod assistant;
pub mod automation;
pub mod backup;
pub mod draft;
pub mod engines;
pub mod events;
pub mod general;
pub mod library;
pub mod metadata;
pub mod metadata_ops;
pub mod models;
pub mod sections;
pub mod shortcuts;
mod widgets;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use eframe::egui::{Frame, Margin, Panel, RichText, ScrollArea, TextEdit, Ui};

use crate::command::Section;
use crate::icons::{self, svg};
use crate::settings::SettingsStore;

use draft::{Applied, Draft, effects_for};
use engines::{AiFlags, EngineAction, run_effects};
use widgets::Focus;

/// Width below which the section list becomes a row of buttons above the content.
pub const NARROW_BELOW: f32 = 640.0;
const LIST_WIDTH: f32 = 190.0;

/// What the page needs from the application each frame.
pub struct Env<'a> {
    pub db_ready: bool,
    pub appearance: &'a mut crate::theme::Appearance,
    /// Set when the egui options should be written to storage.
    pub save_egui: &'a mut bool,
    pub ai_ready: &'a mut bool,
    pub ai_initializing: &'a mut bool,
    /// Set to show the assistant window.
    pub assistant_window: &'a mut bool,
}

/// The page state kept for the whole run, so edits survive closing the tab.
pub struct SettingsPage {
    pub section: Section,
    pub query: String,
    focus: Focus,
    store: Arc<SettingsStore>,
    pub draft: Draft,
    pub general: general::General,
    pub library: library::Library,
    pub models: models::Models,
    pub assistant: assistant::Assistant,
    pub metadata: metadata::Metadata,
    pub shortcuts: shortcuts::Shortcuts,
    pub appearance: appearance::Appearance,
    pub advanced: advanced::Advanced,
    tx: events::Tx,
    rx: events::Rx,
    seen_generation: u64,
    /// Words of the last Apply or Revert.
    pub status: Option<String>,
}

impl Default for SettingsPage {
    fn default() -> Self {
        Self::with_store(Arc::clone(crate::settings::store()))
    }
}

impl SettingsPage {
    /// A page over `store`.
    pub fn with_store(store: Arc<SettingsStore>) -> Self {
        let (tx, rx) = events::channel();
        Self {
            section: Section::General,
            query: String::new(),
            focus: Focus::default(),
            draft: Draft::of(&store),
            general: general::General::new(),
            library: library::Library::new(),
            models: models::Models::default(),
            assistant: assistant::Assistant::default(),
            metadata: metadata::Metadata::new(),
            shortcuts: shortcuts::Shortcuts::default(),
            appearance: appearance::Appearance::default(),
            advanced: advanced::Advanced::default(),
            seen_generation: store.generation(),
            store,
            tx,
            rx,
            status: None,
        }
    }

    /// Shows `section`, or keeps the current one when `None`.
    pub fn select(&mut self, section: Option<Section>) {
        if let Some(section) = section {
            self.section = section;
        }
        self.query.clear();
        self.focus.set(None);
    }

    /// Shows the setting the search found.
    pub fn jump_to(&mut self, entry: &sections::Entry) {
        self.section = entry.section;
        self.query.clear();
        self.focus.set(Some(entry.label));
    }

    pub fn is_dirty(&self) -> bool {
        self.draft.is_dirty() || self.general.is_dirty()
    }

    /// Number of settings with unapplied edits.
    pub fn dirty_count(&self) -> usize {
        self.draft.dirty_fields().len() + self.general.dirty_parts().len()
    }

    /// Writes the edits: one patch for the library settings and the saved files for the rest.
    pub fn apply(&mut self, flags: &mut AiFlags) -> Applied {
        let applied = self.draft.apply(&self.store);
        let others = self.general.apply();
        let effects = effects_for(&applied.fields);
        run_effects(&effects, &self.draft.value, flags);
        let total = applied.fields.len() + others;
        self.status = Some(match (total, self.store.is_loaded()) {
            (0, _) => "No settings changed".to_string(),
            (n, true) => format!("Applied {n} {}", if n == 1 { "change" } else { "changes" }),
            (n, false) => format!("Applied {n} {} in memory; they are saved once the library opens", if n == 1 { "change" } else { "changes" }),
        });
        applied
    }

    /// Drops every unapplied edit.
    pub fn revert(&mut self) {
        self.draft.revert(self.store.snapshot());
        self.general.revert();
        self.status = Some("Edits discarded".to_string());
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            let _ = self.library.on_event(&event) || self.assistant.on_event(&event) || self.metadata.on_event(&event);
        }
    }

    fn sync(&mut self) {
        let generation = self.store.generation();
        if generation != self.seen_generation {
            self.seen_generation = generation;
            self.draft.sync(&self.store.snapshot());
        }
        self.general.sync();
    }

    /// Draws the page.
    pub fn ui(&mut self, ui: &mut Ui, env: &mut Env) {
        self.drain_events();
        self.sync();
        self.toolbar(ui, env);
        ui.add_space(4.0);
        if !self.store.is_loaded() {
            let text = match self.store.load_failure() {
                Some(why) => format!("The stored settings could not be read ({why}). Changes stay in memory."),
                None => "The library is not open yet, so the stored settings are not loaded. Changes stay in memory until it is.".to_string(),
            };
            ui.colored_label(ui.visuals().warn_fg_color, text);
            ui.add_space(4.0);
        }
        if !self.query.trim().is_empty() {
            self.results(ui);
            return;
        }
        if ui.available_width() < NARROW_BELOW {
            ui.horizontal_wrapped(|ui| self.section_buttons(ui));
            ui.separator();
            self.body(ui, env);
        } else {
            let frame = Frame::new().inner_margin(Margin::symmetric(0, 4));
            Panel::left("settings_sections").resizable(false).exact_size(LIST_WIDTH).frame(frame).show(ui, |ui| {
                ui.vertical(|ui| self.section_buttons(ui));
            });
            self.body(ui, env);
        }
    }

    fn toolbar(&mut self, ui: &mut Ui, env: &mut Env) {
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.query).hint_text("Search settings").desired_width(220.0).id_salt("settings-search"));
            if !self.query.is_empty() && icons::icon_button(ui, svg::X, "Clear the search").clicked() {
                self.query.clear();
            }
            ui.separator();
            let dirty = self.is_dirty();
            if dirty {
                icons::label(ui, svg::PENCIL_SIMPLE, RichText::new(format!("Unsaved changes ({})", self.dirty_count())).color(ui.visuals().warn_fg_color));
            } else if let Some(text) = &self.status {
                ui.weak(text.as_str());
            }
            if ui.add_enabled(dirty, eframe::egui::Button::new("Revert")).on_hover_text("Discard the edits and show the saved settings").clicked() {
                self.revert();
            }
            if ui.add_enabled(dirty, crate::theme::primary_button("Apply")).on_hover_text("Save the edits").clicked() {
                let mut flags = AiFlags { ready: env.ai_ready, initializing: env.ai_initializing };
                self.apply(&mut flags);
            }
        });
    }

    fn section_buttons(&mut self, ui: &mut Ui) {
        for section in sections::ALL {
            let selected = self.section == section;
            if icons::selectable_button(ui, sections::icon(section), selected, sections::label(section)).clicked() {
                self.select(Some(section));
            }
        }
    }

    fn results(&mut self, ui: &mut Ui) {
        let hits = sections::search(&self.query);
        if hits.is_empty() {
            ui.weak(format!("No setting matches \"{}\".", self.query.trim()));
            return;
        }
        ui.label(RichText::new(format!("{} {}", hits.len(), if hits.len() == 1 { "setting" } else { "settings" })).strong());
        let mut jump = None;
        for entry in hits {
            ui.horizontal_wrapped(|ui| {
                icons::label(ui, sections::icon(entry.section), entry.label);
                ui.weak(sections::label(entry.section));
                if ui.small_button("Show").clicked() {
                    jump = Some(*entry);
                }
            });
        }
        if let Some(entry) = jump {
            self.jump_to(&entry);
        }
    }

    fn body(&mut self, ui: &mut Ui, env: &mut Env) {
        let section = self.section;
        ui.heading(sections::label(section));
        if sections::scrolls_itself(section) {
            self.section_ui(ui, env);
        } else {
            ScrollArea::vertical().id_salt(("settings-body", section as u8)).auto_shrink([false, false]).show(ui, |ui| {
                self.section_ui(ui, env);
                ui.add_space(24.0);
            });
        }
    }

    fn section_ui(&mut self, ui: &mut Ui, env: &mut Env) {
        let focus = &self.focus;
        match self.section {
            Section::General => self.general.ui(ui, focus),
            Section::Library => {
                let lib_env = library::LibraryEnv { db_ready: env.db_ready, tx: &self.tx };
                self.library.ui(ui, focus, &mut self.draft, &lib_env);
            }
            Section::AiModels => {
                let requests = self.models.ui(ui, focus, &mut self.draft);
                if let Some(action) = requests.engine {
                    self.run_engine_action(action, env);
                }
            }
            Section::Automation => automation::ui(ui, focus, &mut self.draft),
            Section::Assistant => self.assistant.ui(ui, focus, &mut self.draft, &self.tx, env.assistant_window),
            Section::MetadataTools => self.metadata.ui(ui, focus, env.db_ready, &self.tx),
            Section::Shortcuts => self.shortcuts.ui(ui, focus),
            Section::Appearance => self.appearance.ui(ui, focus, env.appearance, env.save_egui),
            Section::Advanced => self.advanced.ui(ui, focus, &self.store, &mut self.draft),
        }
    }

    /// Applies pending edits, then runs the model button that was pressed.
    fn run_engine_action(&mut self, action: EngineAction, env: &mut Env) {
        if needs_applied_edits(action) && self.is_dirty() {
            let mut flags = AiFlags { ready: env.ai_ready, initializing: env.ai_initializing };
            self.apply(&mut flags);
        }
        engines::run(action);
    }
}

/// True when Ctrl+, was pressed this frame; the key is consumed.
pub fn open_key_pressed(ctx: &eframe::egui::Context) -> bool {
    ctx.input_mut(|i| i.consume_key(eframe::egui::Modifiers::COMMAND, eframe::egui::Key::Comma))
}

/// True for the buttons that load a model from the saved settings.
pub fn needs_applied_edits(action: EngineAction) -> bool {
    matches!(action, EngineAction::LoadVision | EngineAction::ReloadVision | EngineAction::ReloadClip | EngineAction::ReloadReranker)
}

/// Draws the Settings tab of the application.
pub fn show(context: &mut crate::app::SmartMediaContext, ui: &mut Ui) {
    let crate::app::SmartMediaContext { settings_page, appearance, egui_prefs_save_requested, ai_ready, ai_initializing, assistant_window_open, db_ready, .. } = context;
    let mut env = Env { db_ready: *db_ready, appearance, save_egui: egui_prefs_save_requested, ai_ready, ai_initializing, assistant_window: assistant_window_open };
    settings_page.ui(ui, &mut env);
}
