//! Advanced section: log level, performance HUD, the raw settings, and resets.

use std::sync::Arc;

use eframe::egui::{CollapsingHeader, ComboBox, ScrollArea, TextEdit, Ui};
use log::LevelFilter;

use super::draft::Draft;
use super::sections::name;
use super::widgets::{Focus, caption, group};
use crate::command::{AppCommand, DevtoolsWindow};
use crate::database::settings::{UiSettings, changed_fields, copy_fields};
use crate::settings::{SettingsStore, is_secret_field, ui_prefs};

/// Name recorded in the store's patch log for a reset.
pub const RESET_WHO: &str = "settings:reset";

/// Levels the picker offers.
pub const LEVELS: [LevelFilter; 5] = [LevelFilter::Error, LevelFilter::Warn, LevelFilter::Info, LevelFilter::Debug, LevelFilter::Trace];

/// The settings fields a reset puts back to their defaults.
pub fn reset_fields(current: &UiSettings, clear_keys: bool) -> Vec<&'static str> {
    let defaults = reset_defaults(clear_keys);
    changed_fields(current, &defaults).into_iter().filter(|f| clear_keys || !is_secret_field(f)).collect()
}

/// The defaults a reset copies from; credential fields stay out unless `clear_keys`.
fn reset_defaults(clear_keys: bool) -> UiSettings {
    let mut defaults = UiSettings::default();
    if clear_keys {
        defaults.openai_api_key = None;
        defaults.grok_api_key = None;
        defaults.gemini_api_key = None;
        defaults.groq_api_key = None;
        defaults.openrouter_api_key = None;
    }
    defaults
}

/// Puts the settings back to their defaults as one patch; returns the fields changed.
pub fn reset_settings(store: &Arc<SettingsStore>, clear_keys: bool) -> Vec<&'static str> {
    let fields = reset_fields(&store.snapshot(), clear_keys);
    if fields.is_empty() {
        return fields;
    }
    let defaults = reset_defaults(clear_keys);
    store.update(RESET_WHO, |cur| copy_fields(cur, &defaults, &fields));
    fields
}

/// Pretty JSON of the interface preferences: the file as saved, else what is live.
pub fn ui_prefs_text() -> String {
    ui_prefs::with(|s| s.file_text().unwrap_or_else(|_| s.prefs.to_json()))
}

/// State of the advanced section.
#[derive(Default)]
pub struct Advanced {
    confirm_reset: bool,
    clear_keys: bool,
    confirm_ui_reset: bool,
    pub notice: Option<String>,
    json_cache: Option<(u64, String)>,
}

impl Advanced {
    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, store: &Arc<SettingsStore>, draft: &mut Draft) {
        group(ui, "Diagnostics");
        ui.horizontal(|ui| {
            caption(ui, focus, name::LOG_LEVEL);
            let current = log::max_level();
            let mut chosen = current;
            ComboBox::from_id_salt("settings-log-level").selected_text(current.to_string()).show_ui(ui, |ui| {
                for level in LEVELS {
                    ui.selectable_value(&mut chosen, level, level.to_string());
                }
            });
            if chosen != current {
                log::set_max_level(chosen);
            }
            ui.weak("Applies at once, until the application closes.");
        });
        ui.horizontal(|ui| {
            let mut on = crate::ui::devtools::perf::is_open(ui.ctx());
            let response = ui.checkbox(&mut on, name::PERF_HUD).on_hover_text("Frame time and scan timings in a Devtools window");
            focus.mark(ui, name::PERF_HUD, &response);
            if response.changed() {
                crate::ui::devtools::perf::set_open(ui.ctx(), on);
            }
        });
        ui.horizontal(|ui| {
            let response = ui.label(name::MCP);
            focus.mark(ui, name::MCP, &response);
            if ui.button("Open the MCP tools server window").clicked() {
                crate::command::send(AppCommand::OpenDevtools { window: DevtoolsWindow::McpServer });
            }
            if ui.button("Settings store").on_hover_text("Patch history and save status").clicked() {
                crate::command::send(AppCommand::OpenDevtools { window: DevtoolsWindow::SettingsStore });
            }
        });

        group(ui, name::RAW_JSON);
        let response = ui.label("Read-only. API keys are hidden.");
        focus.mark(ui, name::RAW_JSON, &response);
        let generation = store.generation();
        if self.json_cache.as_ref().map(|(g, _)| *g != generation).unwrap_or(true) {
            self.json_cache = Some((generation, serde_json::to_string_pretty(&store.json_redacted()).unwrap_or_default()));
        }
        CollapsingHeader::new("Settings record").default_open(false).show(ui, |ui| {
            let mut text = self.json_cache.as_ref().map(|(_, t)| t.as_str()).unwrap_or("");
            ScrollArea::vertical().id_salt("settings-raw-json").max_height(260.0).show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
            });
        });
        CollapsingHeader::new("Interface preferences (ui_prefs.json)").default_open(false).show(ui, |ui| {
            if let Some(path) = ui_prefs::with(|s| s.path().map(|p| p.display().to_string())) {
                ui.weak(path);
            }
            let text = ui_prefs_text();
            let mut view = text.as_str();
            ScrollArea::vertical().id_salt("settings-ui-prefs-json").max_height(220.0).show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut view).code_editor().desired_width(f32::INFINITY));
            });
        });
        ui.weak("ext_enabled and group_by_category are stored in the record but this version does not use them.");

        group(ui, name::RESET);
        let response = ui.label("Puts the library settings back to their defaults. Interface preferences and the library itself are not touched.");
        focus.mark(ui, name::RESET, &response);
        ui.checkbox(&mut self.clear_keys, "Also clear the API keys");
        ui.checkbox(&mut self.confirm_reset, "I understand that the settings are replaced");
        if ui.add_enabled(self.confirm_reset, crate::theme::danger_button("Reset settings")).clicked() {
            let fields = reset_settings(store, self.clear_keys);
            draft.revert(store.snapshot());
            self.notice = Some(if fields.is_empty() { "The settings already have their defaults".to_string() } else { format!("Reset {} settings", fields.len()) });
            self.confirm_reset = false;
        }
        ui.add_space(6.0);
        if ui.button("Show the setup checklist again").on_hover_text("The card on an empty library comes back").clicked() {
            ui_prefs::update(|p| p.checklist.dismissed = false);
            self.notice = Some("The setup checklist shows on an empty library again".to_string());
        }
        ui.checkbox(&mut self.confirm_ui_reset, "Also reset the interface preferences and the setup checklist");
        if ui.add_enabled(self.confirm_ui_reset, crate::theme::danger_button("Reset interface preferences")).clicked() {
            ui_prefs::with(|s| s.reset());
            crate::ui::file_table::table::view_mode::update_app_prefs(|p| *p = Default::default());
            crate::ui::file_table::table::view_mode::flush_app_prefs();
            self.notice = Some("The interface preferences are back to their defaults".to_string());
            self.confirm_ui_reset = false;
        }
        if let Some(text) = &self.notice {
            ui.label(text.as_str());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Backend, Loaded};
    use async_trait::async_trait;

    struct Nop;

    #[async_trait]
    impl Backend for Nop {
        async fn load(&self) -> anyhow::Result<Loaded> {
            Ok(Loaded { settings: UiSettings::default(), existed: true })
        }
        async fn merge(&self, _: &[&'static str], _: &UiSettings) -> anyhow::Result<()> {
            Ok(())
        }
    }

    async fn store_with_edits() -> Arc<SettingsStore> {
        let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        store.update("edit", |s| {
            s.filter_skip_icons = true;
            s.clip_model = Some("clip-vit-b32".into());
            s.groq_api_key = Some("gsk-secret".into());
            s.recent_paths = vec!["/a".into()];
        });
        store
    }

    #[tokio::test]
    async fn a_reset_changes_the_settings_but_keeps_the_keys_unless_asked() {
        let store = store_with_edits().await;
        let fields = reset_settings(&store, false);
        assert!(fields.contains(&"filter_skip_icons") && fields.contains(&"clip_model"));
        assert!(!fields.contains(&"groq_api_key"));
        let live = store.snapshot();
        assert!(!live.filter_skip_icons);
        assert_eq!(live.clip_model, UiSettings::default().clip_model);
        assert_eq!(live.groq_api_key.as_deref(), Some("gsk-secret"));
        assert!(live.recent_paths.is_empty());
    }

    #[tokio::test]
    async fn a_reset_with_keys_clears_them_in_the_same_patch() {
        let store = store_with_edits().await;
        let before = store.patches().len();
        let fields = reset_settings(&store, true);
        assert!(fields.contains(&"groq_api_key"));
        assert_eq!(store.patches().len(), before + 1, "one patch");
        assert_eq!(store.snapshot().groq_api_key, None);
    }

    #[tokio::test]
    async fn resetting_defaults_changes_nothing() {
        let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
        store.apply_loaded("test", Loaded { settings: reset_defaults(false), existed: true });
        let generation = store.generation();
        assert!(reset_settings(&store, false).is_empty());
        assert_eq!(store.generation(), generation);
    }

    #[test]
    fn the_prefs_text_is_json_even_without_a_file() {
        let text = ui_prefs_text();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["version"], ui_prefs::VERSION);
    }

    #[test]
    fn the_log_levels_run_from_error_to_trace() {
        assert!(LEVELS.windows(2).all(|w| w[0] < w[1]));
    }
}
