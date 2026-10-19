//! The assistant backends and the rule that picks one: ZeroClaw over the gateway socket, a direct provider, or local models only.

use std::sync::Arc;

use crossbeam::channel::Sender;
use zeroclaw_client::{ApprovalDecision, InlineImage};

use crate::UiSettings;
use crate::ui::file_table::AIUpdate;
use crate::zc::health::Health;
use crate::zc::profile::{BackendPref, ZcProfile};
use crate::zc::runtime::ZcRuntime;

/// The engines that can answer a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    ZeroClaw,
    Direct,
    Local,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ZeroClaw => "zeroclaw",
            Self::Direct => "direct",
            Self::Local => "local",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ZeroClaw => "ZeroClaw",
            Self::Direct => "Direct",
            Self::Local => "Local",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        [Self::ZeroClaw, Self::Direct, Self::Local].into_iter().find(|k| k.as_str() == text)
    }
}

/// What would make an unavailable backend work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fix {
    Pair,
    Settings,
    Retry,
    Nothing,
}

/// The outcome of the backend rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolved {
    Ready(BackendKind),
    /// Nothing is set up yet; the onboarding card is shown.
    NeedsSetup,
    /// The backend that was asked for (or that Auto settled on) cannot answer now.
    Unavailable { backend: BackendKind, reason: String, fix: Fix },
}

impl Resolved {
    pub fn kind(&self) -> Option<BackendKind> {
        match self {
            Self::Ready(k) => Some(*k),
            Self::Unavailable { backend, .. } => Some(*backend),
            Self::NeedsSetup => None,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

/// The state of the direct provider the rule looks at.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectState {
    pub provider: String,
    pub has_key: bool,
}

impl DirectState {
    pub fn of(settings: &UiSettings) -> Self {
        let provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
        let has_key = match provider.as_str() {
            "openai" => key_set(&settings.openai_api_key, "OPENAI_API_KEY"),
            "grok" => key_set(&settings.grok_api_key, "GROK_API_KEY"),
            "gemini" => key_set(&settings.gemini_api_key, "GEMINI_API_KEY"),
            "groq" => key_set(&settings.groq_api_key, "GROQ_API_KEY"),
            "openrouter" => key_set(&settings.openrouter_api_key, "OPENROUTER_API_KEY"),
            "custom" => settings.openai_base_url.as_deref().is_some_and(|u| !u.trim().is_empty()),
            _ => false,
        };
        Self { provider, has_key }
    }
}

fn key_set(slot: &Option<String>, env: &str) -> bool {
    slot.as_deref().is_some_and(|k| !k.trim().is_empty()) || std::env::var(env).is_ok_and(|k| !k.trim().is_empty())
}

/// What the rule needs to know.
pub struct Inputs<'a> {
    pub pref: BackendPref,
    pub profile: &'a ZcProfile,
    pub has_token: bool,
    pub health: &'a Health,
    pub direct: &'a DirectState,
}

fn gateway_state(profile: &ZcProfile, health: &Health) -> Resolved {
    let zc = |reason: String, fix: Fix| Resolved::Unavailable { backend: BackendKind::ZeroClaw, reason, fix };
    match health {
        Health::Online { .. } => Resolved::Ready(BackendKind::ZeroClaw),
        Health::Checking | Health::Unconfigured(_) => zc(format!("Checking {}", profile.host_label()), Fix::Nothing),
        Health::Offline(why) => zc(format!("The gateway is offline: {why}"), Fix::Retry),
        Health::Unauthorized => zc("The gateway refused the token".to_string(), Fix::Pair),
        Health::UnknownAgent { agent, .. } => zc(format!("The gateway has no agent named '{agent}'"), Fix::Settings),
    }
}

/// Picks the backend. Auto means ZeroClaw once it is set up and reachable, and the onboarding card before that; it never falls back to a cloud provider. An explicit pick that cannot answer is reported, not replaced.
pub fn resolve(inputs: &Inputs) -> Resolved {
    let set_up = inputs.profile.is_configured() && inputs.has_token;
    match inputs.pref {
        BackendPref::Auto if !set_up => Resolved::NeedsSetup,
        BackendPref::Auto => gateway_state(inputs.profile, inputs.health),
        BackendPref::ZeroClaw if !set_up => Resolved::Unavailable { backend: BackendKind::ZeroClaw, reason: "ZeroClaw is not set up: add a gateway address, an agent and a token".to_string(), fix: Fix::Settings },
        BackendPref::ZeroClaw => gateway_state(inputs.profile, inputs.health),
        BackendPref::Direct => {
            let d = inputs.direct;
            if d.provider == "local-joycaption" {
                Resolved::Unavailable { backend: BackendKind::Direct, reason: "No direct provider is chosen: pick one in Settings".to_string(), fix: Fix::Settings }
            } else if !d.has_key {
                Resolved::Unavailable { backend: BackendKind::Direct, reason: format!("No API key for {}", d.provider), fix: Fix::Settings }
            } else {
                Resolved::Ready(BackendKind::Direct)
            }
        }
        BackendPref::Local => Resolved::Ready(BackendKind::Local),
    }
}

/// Everything a backend needs to run one turn.
pub struct TurnRequest {
    pub session: String,
    /// Number events of this turn carry (the connection epoch, or the panel's own counter).
    pub epoch: u64,
    pub turn: u64,
    /// What the person typed.
    pub text: String,
    /// `text` with the context snapshot in front.
    pub prompt: String,
    pub attachments: Vec<String>,
    /// Pictures prepared for the gateway; empty unless the profile allows them.
    pub images: Vec<InlineImage>,
    /// Earlier text turns for providers that keep no conversation, as (is_user, text).
    pub history: Vec<(bool, String)>,
    pub settings: UiSettings,
    pub model: Option<String>,
    pub temperature: f32,
    pub ai_update_tx: Sender<AIUpdate>,
}

/// An engine that answers chat messages. Events come back on the panel's channels tagged with the request's session and epoch.
pub trait AssistantBackend: Send {
    fn kind(&self) -> BackendKind;
    /// Starts a turn.
    fn send(&mut self, req: TurnRequest) -> Result<(), String>;
    /// Stops the running turn and keeps late events out.
    fn stop(&mut self);
    /// Answers an approval card; false when this backend asks for none.
    fn approve(&mut self, _request_id: &str, _decision: ApprovalDecision) -> bool {
        false
    }
    /// Whether the agent can call tools.
    fn tools(&self) -> bool {
        false
    }
}

/// Chat over the gateway socket of the ZeroClaw runtime.
pub struct ZeroClawBackend {
    rt: Arc<ZcRuntime>,
}

impl ZeroClawBackend {
    pub fn new(rt: Arc<ZcRuntime>) -> Self {
        Self { rt }
    }
}

impl AssistantBackend for ZeroClawBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::ZeroClaw
    }

    fn send(&mut self, req: TurnRequest) -> Result<(), String> {
        if self.rt.live_session().as_deref() != Some(req.session.as_str()) {
            return Err("The chat socket is not open for this session".to_string());
        }
        self.rt.send(req.prompt, req.images);
        Ok(())
    }

    fn stop(&mut self) {
        self.rt.stop();
    }

    fn approve(&mut self, request_id: &str, decision: ApprovalDecision) -> bool {
        self.rt.approve(request_id, decision)
    }

    fn tools(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn online() -> Health {
        Health::Online { agents: vec!["media".into()], require_pairing: true, latency_ms: 3 }
    }

    fn configured() -> ZcProfile {
        ZcProfile { host: "10.0.0.2".into(), agent: "media".into(), ..Default::default() }
    }

    fn pick(pref: BackendPref, profile: &ZcProfile, has_token: bool, health: &Health, direct: &DirectState) -> Resolved {
        resolve(&Inputs { pref, profile, has_token, health, direct })
    }

    fn keyed() -> DirectState {
        DirectState { provider: "openrouter".into(), has_key: true }
    }

    fn nothing() -> DirectState {
        DirectState { provider: "local-joycaption".into(), has_key: false }
    }

    #[test]
    fn auto_uses_zeroclaw_when_it_is_set_up_and_online() {
        assert_eq!(pick(BackendPref::Auto, &configured(), true, &online(), &keyed()), Resolved::Ready(BackendKind::ZeroClaw));
    }

    #[test]
    fn auto_shows_onboarding_until_a_gateway_and_token_exist() {
        let unset = ZcProfile::default();
        assert_eq!(pick(BackendPref::Auto, &unset, false, &Health::Unconfigured("x".into()), &keyed()), Resolved::NeedsSetup);
        assert_eq!(pick(BackendPref::Auto, &configured(), false, &Health::Unconfigured("x".into()), &keyed()), Resolved::NeedsSetup, "a gateway without a token is not set up");
    }

    #[test]
    fn auto_never_falls_back_to_a_cloud_provider() {
        for health in [Health::Offline("refused".into()), Health::Unauthorized, Health::Checking, Health::UnknownAgent { agent: "media".into(), available: vec![] }] {
            let r = pick(BackendPref::Auto, &configured(), true, &health, &keyed());
            assert!(matches!(&r, Resolved::Unavailable { backend: BackendKind::ZeroClaw, .. }), "{health:?} gave {r:?}");
        }
        assert_eq!(pick(BackendPref::Auto, &ZcProfile::default(), false, &Health::Checking, &keyed()), Resolved::NeedsSetup);
    }

    #[test]
    fn an_offline_gateway_offers_retry_and_a_refused_token_offers_pairing() {
        let offline = pick(BackendPref::Auto, &configured(), true, &Health::Offline("down".into()), &nothing());
        assert!(matches!(&offline, Resolved::Unavailable { fix: Fix::Retry, reason, .. } if reason.contains("down")));
        let refused = pick(BackendPref::Auto, &configured(), true, &Health::Unauthorized, &nothing());
        assert!(matches!(refused, Resolved::Unavailable { fix: Fix::Pair, .. }));
    }

    #[test]
    fn an_explicit_zeroclaw_pick_reports_what_is_missing() {
        let r = pick(BackendPref::ZeroClaw, &ZcProfile::default(), false, &Health::Unconfigured("x".into()), &keyed());
        assert!(matches!(&r, Resolved::Unavailable { backend: BackendKind::ZeroClaw, fix: Fix::Settings, .. }));
        assert_eq!(pick(BackendPref::ZeroClaw, &configured(), true, &online(), &nothing()), Resolved::Ready(BackendKind::ZeroClaw));
    }

    #[test]
    fn a_direct_pick_needs_a_provider_and_a_key() {
        assert_eq!(pick(BackendPref::Direct, &configured(), true, &online(), &keyed()), Resolved::Ready(BackendKind::Direct));
        let none = pick(BackendPref::Direct, &configured(), true, &online(), &nothing());
        assert!(matches!(&none, Resolved::Unavailable { backend: BackendKind::Direct, reason, .. } if reason.contains("provider")));
        let nokey = pick(BackendPref::Direct, &configured(), true, &online(), &DirectState { provider: "openai".into(), has_key: false });
        assert!(matches!(&nokey, Resolved::Unavailable { reason, .. } if reason.contains("openai")));
    }

    #[test]
    fn local_only_is_always_available() {
        assert_eq!(pick(BackendPref::Local, &ZcProfile::default(), false, &Health::Offline("x".into()), &nothing()), Resolved::Ready(BackendKind::Local));
    }

    #[test]
    fn direct_state_reads_the_provider_and_its_key() {
        let mut s = UiSettings::default();
        s.ai_chat_provider = Some("openrouter".into());
        s.openrouter_api_key = Some("sk-x".into());
        assert_eq!(DirectState::of(&s), DirectState { provider: "openrouter".into(), has_key: true });
        s.openrouter_api_key = Some("  ".into());
        s.ai_chat_provider = Some("custom".into());
        s.openai_base_url = Some("http://localhost:11434/v1".into());
        assert!(DirectState::of(&s).has_key, "a custom endpoint needs no key");
        s.ai_chat_provider = None;
        assert_eq!(DirectState::of(&s).provider, "local-joycaption");
    }

    #[test]
    fn backend_names_round_trip() {
        for k in [BackendKind::ZeroClaw, BackendKind::Direct, BackendKind::Local] {
            assert_eq!(BackendKind::parse(k.as_str()), Some(k));
        }
        assert_eq!(BackendKind::parse("cloud"), None);
    }
}
