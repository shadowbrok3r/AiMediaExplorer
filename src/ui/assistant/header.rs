//! The assistant header: which backend answers and whether it is healthy, the session switcher and the banners that say what is wrong.

use egui::{ComboBox, RichText, TextEdit, Ui};

use super::backend::{BackendKind, DirectState, Fix, Resolved};
use super::sessions::Sessions;
use crate::UiSettings;
use crate::icons::{self, svg};
use crate::zc::events::LinkState;
use crate::zc::health::{Health, HealthKind};
use crate::zc::profile::{BackendPref, ZcProfile};
use crate::zc::widgets;

/// What a click in the header asks for.
#[derive(Debug, PartialEq, Eq)]
pub enum Act {
    NewChat,
    Switch(usize),
    AskDelete(usize),
    ConfirmDelete(usize),
    CancelDelete,
    SetBackend(BackendPref),
    OpenSettings,
    PopOut,
    Dock,
    Retry,
    Pair,
    StartTools,
    BrowseModels,
    SetProvider(String),
    SetModel(String),
    ToolsDebug,
}

pub struct Env<'a> {
    pub resolved: &'a Resolved,
    pub profile: &'a ZcProfile,
    pub health: &'a Health,
    pub link: LinkState,
    pub sessions: &'a Sessions,
    pub direct: &'a DirectState,
    pub settings: &'a UiSettings,
    /// Set when the tools server is expected but not running.
    pub tools_offline: bool,
    pub popped_out: bool,
    /// The session list is drawn elsewhere (the pop-out window's side panel).
    pub list_beside: bool,
}

/// `ZeroClaw - media`, `Direct - openrouter` or `Local`.
pub fn chip_text(resolved: &Resolved, profile: &ZcProfile, direct: &DirectState) -> String {
    match resolved.kind() {
        Some(BackendKind::ZeroClaw) => {
            let agent = profile.agent.trim();
            if agent.is_empty() { "ZeroClaw".to_string() } else { format!("ZeroClaw - {agent}") }
        }
        Some(BackendKind::Direct) => format!("Direct - {}", direct.provider),
        Some(BackendKind::Local) => "Local".to_string(),
        None => "Not set up".to_string(),
    }
}

/// The dot colour: the gateway's health for ZeroClaw, green for the other ready backends, grey when nothing is chosen.
fn dot_kind(resolved: &Resolved, health: &Health) -> HealthKind {
    match resolved.kind() {
        Some(BackendKind::ZeroClaw) => health.kind(),
        Some(_) if resolved.is_ready() => HealthKind::Online,
        Some(_) => HealthKind::Offline,
        None => HealthKind::Unconfigured,
    }
}

fn backend_menu(ui: &mut Ui, env: &Env, acts: &mut Vec<Act>) {
    ui.label(RichText::new("Backend").strong());
    let current = env.profile.backend_pref();
    for pref in BackendPref::ALL {
        if ui.selectable_label(current == pref, pref.label()).clicked() {
            acts.push(Act::SetBackend(pref));
        }
    }
    ui.separator();
    widgets::health_row(ui, env.health);
    ui.label(RichText::new(format!("{} - agent {}", env.profile.host_label(), env.profile.agent.trim())).weak());
    ui.label(RichText::new(format!("Socket: {}", env.link.label())).weak());
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::PULSE, "Check").clicked() {
            acts.push(Act::Retry);
        }
        if icons::button(ui, svg::GEAR, "Connection settings").clicked() {
            acts.push(Act::OpenSettings);
        }
    });
    ui.separator();
    ui.label(RichText::new("Direct provider").strong());
    let providers = ["local-joycaption", "openai", "gemini", "groq", "grok", "openrouter", "custom"];
    let mut provider = env.direct.provider.clone();
    ComboBox::from_id_salt("ame_assistant_provider").selected_text(&provider).show_ui(ui, |ui| {
        for p in providers {
            ui.selectable_value(&mut provider, p.to_string(), p);
        }
    });
    if provider != env.direct.provider {
        acts.push(Act::SetProvider(provider));
    }
    let mut model = env.settings.openai_default_model.clone().unwrap_or_default();
    let edit = ui.add(TextEdit::singleline(&mut model).hint_text("model").desired_width(ui.available_width().min(240.0)));
    if edit.lost_focus() && !model.trim().is_empty() && Some(&model) != env.settings.openai_default_model.as_ref() {
        acts.push(Act::SetModel(model.trim().to_string()));
    }
    if env.direct.provider == "openrouter" && icons::button(ui, svg::LIST_MAGNIFYING_GLASS, "Browse OpenRouter models").clicked() {
        acts.push(Act::BrowseModels);
    }
    if !env.direct.has_key && env.direct.provider != "local-joycaption" {
        ui.label(RichText::new("No API key yet; add one in Settings.").color(ui.visuals().warn_fg_color));
    }
}

fn sessions_menu(ui: &mut Ui, env: &Env, acts: &mut Vec<Act>) {
    let title = env.sessions.active_info().map(|s| s.title.clone()).unwrap_or_else(|| "No chat".to_string());
    let width = (ui.available_width() - 70.0).clamp(80.0, 320.0);
    ComboBox::from_id_salt("ame_assistant_sessions").selected_text(title).width(width).show_ui(ui, |ui| {
        for (i, s) in env.sessions.list.iter().enumerate() {
            ui.horizontal(|ui| {
                let selected = env.sessions.active == Some(i);
                let legacy = if s.is_legacy() { " (old)" } else { "" };
                if ui.selectable_label(selected, format!("{}{legacy}", s.title)).clicked() {
                    acts.push(Act::Switch(i));
                }
                if env.sessions.confirm_delete == Some(i) {
                    if ui.small_button("Delete").clicked() {
                        acts.push(Act::ConfirmDelete(i));
                    }
                    if ui.small_button("Keep").clicked() {
                        acts.push(Act::CancelDelete);
                    }
                } else if icons::icon_button(ui, svg::TRASH, "Delete this chat (the gateway keeps its own copy)").clicked() {
                    acts.push(Act::AskDelete(i));
                }
            });
        }
        if env.sessions.list.is_empty() {
            ui.label(RichText::new("No chats yet").weak());
        }
    });
}

fn banner(ui: &mut Ui, text: &str, button: Option<(&str, icons::Icon, Act)>, acts: &mut Vec<Act>) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(text).color(ui.visuals().warn_fg_color));
        if let Some((label, icon, act)) = button {
            if icons::small_button(ui, icon, label).clicked() {
                acts.push(act);
            }
        }
    });
}

/// Draws the header and its banners.
pub fn show(ui: &mut Ui, env: &Env) -> Vec<Act> {
    let mut acts = Vec::new();
    ui.horizontal_wrapped(|ui| {
        let text = chip_text(env.resolved, env.profile, env.direct);
        let kind = dot_kind(env.resolved, env.health);
        let color = widgets::health_color(ui, kind);
        icons::menu_button(ui, svg::AME_ASSISTANT, &text, |ui| backend_menu(ui, env, &mut acts));
        widgets::dot(ui, color).on_hover_text(match env.resolved.kind() {
            Some(BackendKind::ZeroClaw) => format!("{}: {}", env.health.label(), env.health.detail()),
            _ => env.resolved.kind().map(|k| k.label().to_string()).unwrap_or_else(|| "Not set up".to_string()),
        });
        if !env.list_beside {
            sessions_menu(ui, env, &mut acts);
        }
        if icons::icon_button(ui, svg::AME_NEW_CHAT, "New chat").clicked() {
            acts.push(Act::NewChat);
        }
        if env.popped_out {
            if icons::icon_button(ui, svg::SIDEBAR_SIMPLE, "Dock the assistant back into the right rail").clicked() {
                acts.push(Act::Dock);
            }
        } else if icons::icon_button(ui, svg::ARROW_SQUARE_OUT, "Pop out into its own window").clicked() {
            acts.push(Act::PopOut);
        }
        if icons::icon_button(ui, svg::AME_MCP, "Tools tester").clicked() {
            acts.push(Act::ToolsDebug);
        }
    });
    if let Resolved::Unavailable { reason, fix, .. } = env.resolved {
        let button = match fix {
            Fix::Pair => Some(("Re-pair", svg::LINK, Act::Pair)),
            Fix::Settings => Some(("Settings", svg::GEAR, Act::OpenSettings)),
            Fix::Retry => Some(("Retry", svg::ARROW_CLOCKWISE, Act::Retry)),
            Fix::Nothing => None,
        };
        banner(ui, reason, button, &mut acts);
    }
    if env.tools_offline && env.resolved.kind() == Some(BackendKind::ZeroClaw) {
        banner(ui, "The tools server is off, so the agent cannot read your library.", Some(("Start tools", svg::PLAY, Act::StartTools)), &mut acts);
    }
    acts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zc::health::Health;

    fn profile() -> ZcProfile {
        ZcProfile { host: "10.0.0.2".into(), agent: "media".into(), ..Default::default() }
    }

    #[test]
    fn the_chip_names_the_backend_and_its_agent_or_provider() {
        let direct = DirectState { provider: "openrouter".into(), has_key: true };
        assert_eq!(chip_text(&Resolved::Ready(BackendKind::ZeroClaw), &profile(), &direct), "ZeroClaw - media");
        assert_eq!(chip_text(&Resolved::Ready(BackendKind::Direct), &profile(), &direct), "Direct - openrouter");
        assert_eq!(chip_text(&Resolved::Ready(BackendKind::Local), &profile(), &direct), "Local");
        assert_eq!(chip_text(&Resolved::NeedsSetup, &profile(), &direct), "Not set up");
        let unavailable = Resolved::Unavailable { backend: BackendKind::ZeroClaw, reason: "x".into(), fix: Fix::Retry };
        assert_eq!(chip_text(&unavailable, &ZcProfile::default(), &direct), "ZeroClaw");
    }

    #[test]
    fn the_dot_follows_gateway_health_only_for_zeroclaw() {
        let online = Health::Online { agents: vec![], require_pairing: true, latency_ms: 1 };
        assert_eq!(dot_kind(&Resolved::Ready(BackendKind::ZeroClaw), &online), HealthKind::Online);
        assert_eq!(dot_kind(&Resolved::Ready(BackendKind::ZeroClaw), &Health::Unauthorized), HealthKind::Unauthorized);
        assert_eq!(dot_kind(&Resolved::Ready(BackendKind::Local), &Health::Unauthorized), HealthKind::Online);
        assert_eq!(dot_kind(&Resolved::NeedsSetup, &online), HealthKind::Unconfigured);
    }
}
