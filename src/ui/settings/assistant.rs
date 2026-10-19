//! Assistant and providers section: the ZeroClaw card, the direct providers, their keys and a connection test.

use std::collections::HashMap;

use eframe::egui::{ComboBox, RichText, Ui};

use super::draft::Draft;
use super::events::{Event, Tx};
use super::sections::name;
use super::widgets::{Focus, caption, group, opt_text, secret};
use crate::UiSettings;

/// A provider the assistant can talk to directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provider {
    pub id: &'static str,
    pub label: &'static str,
    /// Environment variable that supplies the key when none is saved.
    pub env: Option<&'static str>,
    /// Model used when none is chosen.
    pub model: &'static str,
}

/// The providers of the assistant, in the order the list shows them.
pub const PROVIDERS: [Provider; 7] = [
    Provider { id: "local-joycaption", label: "Local (JoyCaption/LLaVA)", env: None, model: "" },
    Provider { id: "openai", label: "OpenAI", env: Some("OPENAI_API_KEY"), model: "gpt-5-mini" },
    Provider { id: "grok", label: "Grok (xAI)", env: Some("GROK_API_KEY"), model: "grok-4-fast-non-reasoning" },
    Provider { id: "gemini", label: "Google Gemini", env: Some("GEMINI_API_KEY"), model: "gemma-3-27b-it" },
    Provider { id: "groq", label: "Groq", env: Some("GROQ_API_KEY"), model: "llama-3.1-70b-versatile" },
    Provider { id: "openrouter", label: "OpenRouter", env: Some("OPENROUTER_API_KEY"), model: "meta-llama/llama-3.1-70b-instruct" },
    Provider { id: "custom", label: "Custom (OpenAI-compatible)", env: None, model: "llama3.1" },
];

/// The provider with `id`.
pub fn provider(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// The provider the settings select.
pub fn selected(s: &UiSettings) -> &'static Provider {
    s.ai_chat_provider.as_deref().and_then(provider).unwrap_or(&PROVIDERS[0])
}

/// The settings field that holds the key of `id`.
pub fn key_slot<'a>(s: &'a mut UiSettings, id: &str) -> Option<&'a mut Option<String>> {
    match id {
        "openai" | "custom" => Some(&mut s.openai_api_key),
        "grok" => Some(&mut s.grok_api_key),
        "gemini" => Some(&mut s.gemini_api_key),
        "groq" => Some(&mut s.groq_api_key),
        "openrouter" => Some(&mut s.openrouter_api_key),
        _ => None,
    }
}

/// The key to use for `id`: the saved one, else the environment variable.
pub fn effective_key(s: &mut UiSettings, id: &str) -> Option<String> {
    let env = provider(id).and_then(|p| p.env);
    let saved = key_slot(s, id).and_then(|k| k.clone()).filter(|k| !k.trim().is_empty());
    saved.or_else(|| env.and_then(|name| std::env::var(name).ok()))
}

/// The endpoint override the connection test uses for `id`; OpenRouter always uses its own.
pub fn base_url_for(s: &UiSettings, id: &str) -> Option<String> {
    match id {
        "openrouter" => Some("https://openrouter.ai/api/v1".to_string()),
        _ => s.openai_base_url.clone().filter(|u| !u.trim().is_empty()),
    }
}

/// How a connection test stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestState {
    Running,
    Ok(usize),
    Failed(String),
}

impl TestState {
    pub fn text(&self) -> String {
        match self {
            Self::Running => "Testing...".to_string(),
            Self::Ok(1) => "Connected, 1 model".to_string(),
            Self::Ok(n) => format!("Connected, {n} models"),
            Self::Failed(e) => format!("Failed: {e}"),
        }
    }
}

/// State of the assistant section.
#[derive(Default)]
pub struct Assistant {
    pub tests: HashMap<String, TestState>,
}

impl Assistant {
    pub fn on_event(&mut self, event: &Event) -> bool {
        let Event::Tested(id, result) = event else { return false };
        let state = match result {
            Ok(n) => TestState::Ok(*n),
            Err(e) => TestState::Failed(e.clone()),
        };
        self.tests.insert(id.clone(), state);
        true
    }

    fn test(&mut self, id: &str, s: &mut UiSettings, tx: &Tx) {
        let (key, base, id_owned) = (effective_key(s, id), base_url_for(s, id), id.to_string());
        self.tests.insert(id.to_string(), TestState::Running);
        let tx = tx.clone();
        tokio::spawn(async move {
            let result = tokio::time::timeout(std::time::Duration::from_secs(20), crate::ai::openai_compat::list_models_via_async_openai(&id_owned, key, base))
                .await
                .map_err(|_| "no answer within 20 seconds".to_string())
                .and_then(|r| r.map(|models| models.len()).map_err(|e| format!("{e:#}")));
            let _ = tx.send(Event::Tested(id_owned, result));
            crate::command::request_repaint();
        });
    }

    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, draft: &mut Draft, tx: &Tx, open_assistant: &mut bool) {
        group(ui, name::ZEROCLAW);
        let heading = ui.label(RichText::new("Agent gateway").weak());
        focus.mark(ui, name::ZEROCLAW, &heading);
        crate::zc::connection_ui(ui);

        group(ui, "Direct providers");
        let s = &mut draft.value;
        ui.horizontal(|ui| {
            caption(ui, focus, name::PROVIDER);
            let current = selected(s);
            let mut chosen = current.id;
            ComboBox::from_id_salt("settings-provider").selected_text(current.label).show_ui(ui, |ui| {
                for p in PROVIDERS {
                    if p.id == "openai" || p.id == "custom" {
                        ui.separator();
                    }
                    ui.selectable_value(&mut chosen, p.id, p.label);
                }
            });
            if chosen != current.id {
                s.ai_chat_provider = Some(chosen.to_string());
            }
        });
        let current = selected(s);
        if current.id == "local-joycaption" {
            ui.weak("The assistant runs on the local vision model. No key is needed.");
        } else {
            ui.horizontal(|ui| {
                caption(ui, focus, name::MODEL);
                let mut model = s.openai_default_model.clone();
                let response = opt_text(ui, &mut model, current.model, 260.0);
                if response.changed() {
                    s.openai_default_model = model;
                }
            });
            if current.id != "openrouter" {
                ui.horizontal(|ui| {
                    caption(ui, focus, name::BASE_URL).on_hover_text("OpenAI-compatible endpoint, for example http://host:port/v1");
                    let mut url = s.openai_base_url.clone();
                    let response = opt_text(ui, &mut url, if current.id == "custom" { "http://localhost:11434/v1" } else { "the provider default" }, 300.0);
                    if response.changed() {
                        s.openai_base_url = url;
                    }
                });
            }
            if current.id == "openai" {
                ui.horizontal(|ui| {
                    caption(ui, focus, name::ORGANIZATION);
                    let mut org = s.openai_organization.clone();
                    if opt_text(ui, &mut org, "optional", 260.0).changed() {
                        s.openai_organization = org;
                    }
                });
            }
        }

        group(ui, "API keys");
        for p in PROVIDERS.iter().filter(|p| p.id != "local-joycaption") {
            ui.horizontal_wrapped(|ui| {
                let label = if p.id == current.id { format!("{} (in use)", p.label) } else { p.label.to_string() };
                ui.label(RichText::new(label).strong());
                let response = ui.label(name::API_KEY);
                focus.mark(ui, name::API_KEY, &response);
                if let Some(slot) = key_slot(s, p.id) {
                    secret(ui, slot, p.env, 260.0);
                }
                let state = self.tests.get(p.id).cloned();
                let can = !matches!(state, Some(TestState::Running));
                if ui.add_enabled(can, eframe::egui::Button::new(name::TEST)).clicked() {
                    self.test(p.id, s, tx);
                }
                if let Some(state) = state {
                    let color = match state {
                        TestState::Ok(_) => crate::theme::success_ctx(ui.ctx()),
                        TestState::Failed(_) => ui.visuals().error_fg_color,
                        TestState::Running => ui.visuals().text_color(),
                    };
                    ui.colored_label(color, state.text());
                }
            });
        }
        ui.weak("Keys are stored in the library settings. An empty field uses the environment variable of the provider.");

        group(ui, name::RECENT_MODELS);
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::RECENT_MODELS);
            if s.recent_models.is_empty() {
                ui.weak("none");
            }
            for model in &s.recent_models {
                ui.label(RichText::new(model.as_str()).monospace());
            }
            if ui.add_enabled(!s.recent_models.is_empty(), eframe::egui::Button::new("Clear history")).clicked() {
                s.recent_models.clear();
                s.last_used_model = None;
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Browse OpenRouter models").on_hover_text("Opens the assistant window, where the model browser lives").clicked() {
                *open_assistant = true;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_has_a_unique_id_and_the_selected_one_defaults_to_local() {
        let ids: std::collections::BTreeSet<_> = PROVIDERS.iter().map(|p| p.id).collect();
        assert_eq!(ids.len(), PROVIDERS.len());
        let mut s = UiSettings::default();
        assert_eq!(selected(&s).id, "local-joycaption");
        s.ai_chat_provider = Some("groq".into());
        assert_eq!(selected(&s).id, "groq");
        s.ai_chat_provider = Some("something-new".into());
        assert_eq!(selected(&s).id, "local-joycaption", "an unknown provider falls back to local");
    }

    #[test]
    fn each_provider_reads_and_writes_its_own_key_field() {
        let mut s = UiSettings::default();
        for (id, field) in [("openai", "a"), ("grok", "b"), ("gemini", "c"), ("groq", "d"), ("openrouter", "e")] {
            *key_slot(&mut s, id).unwrap() = Some(field.to_string());
        }
        assert_eq!((s.openai_api_key.as_deref(), s.grok_api_key.as_deref(), s.gemini_api_key.as_deref(), s.groq_api_key.as_deref(), s.openrouter_api_key.as_deref()), (Some("a"), Some("b"), Some("c"), Some("d"), Some("e")));
        assert!(key_slot(&mut s, "local-joycaption").is_none());
        *key_slot(&mut s, "custom").unwrap() = Some("shared".to_string());
        assert_eq!(s.openai_api_key.as_deref(), Some("shared"), "a custom endpoint shares the OpenAI key field, as before");
    }

    #[test]
    fn an_empty_saved_key_falls_back_to_the_environment() {
        let mut s = UiSettings::default();
        s.openai_api_key = Some("  ".into());
        s.groq_api_key = Some("saved".into());
        assert_eq!(effective_key(&mut s, "groq").as_deref(), Some("saved"));
        assert_eq!(effective_key(&mut s, "local-joycaption"), None);
        let env = std::env::var("OPENAI_API_KEY").ok();
        assert_eq!(effective_key(&mut s, "openai"), env);
    }

    #[test]
    fn openrouter_ignores_the_endpoint_override_and_others_use_it() {
        let mut s = UiSettings::default();
        s.openai_base_url = Some("http://host:1/v1".into());
        assert_eq!(base_url_for(&s, "openrouter").as_deref(), Some("https://openrouter.ai/api/v1"));
        assert_eq!(base_url_for(&s, "custom").as_deref(), Some("http://host:1/v1"));
        s.openai_base_url = Some(" ".into());
        assert_eq!(base_url_for(&s, "custom"), None);
    }

    #[test]
    fn test_results_are_kept_per_provider() {
        let mut a = Assistant::default();
        assert!(a.on_event(&Event::Tested("openai".into(), Ok(3))));
        assert!(a.on_event(&Event::Tested("groq".into(), Err("401".into()))));
        assert_eq!(a.tests["openai"].text(), "Connected, 3 models");
        assert_eq!(a.tests["groq"].text(), "Failed: 401");
        assert_eq!(TestState::Ok(1).text(), "Connected, 1 model");
        assert!(!a.on_event(&Event::Edited(Ok(String::new()))));
    }

    #[tokio::test]
    async fn a_connection_test_to_a_dead_endpoint_fails_with_a_message() {
        let (tx, rx) = super::super::events::channel();
        let mut a = Assistant::default();
        let mut s = UiSettings::default();
        s.openai_base_url = Some("http://127.0.0.1:9/v1".into());
        s.openai_api_key = Some("k".into());
        a.test("custom", &mut s, &tx);
        assert_eq!(a.tests["custom"], TestState::Running);
        let event = tokio::task::spawn_blocking(move || rx.recv_timeout(std::time::Duration::from_secs(30))).await.unwrap().expect("an answer");
        assert!(a.on_event(&event));
        assert!(matches!(a.tests["custom"], TestState::Failed(_)), "{:?}", a.tests["custom"]);
    }
}
