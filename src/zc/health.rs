//! Gateway health: the probe, its taxonomy and what it shows in the status bar and under Needs attention.

use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use zeroclaw_client::{AuthState, Config, Error, RestClient};

use super::profile::ZcProfile;
use crate::command::{AppCommand, Section};
use crate::status::{GlobalStatusIndicator, RegisteredStatus, StatusState};
use crate::tasks::{AttentionItem, AttentionKind};

/// The whole probe, health request and token check together.
pub const PROBE_BUDGET: Duration = Duration::from_secs(6);
/// Time between probes while nothing else asks for one.
pub const SWEEP: Duration = Duration::from_secs(45);
/// Key of the Needs attention entry for a refused token.
pub const ATTENTION_SOURCE: &str = "zeroclaw";

pub static ZEROCLAW_STATUS: Lazy<RegisteredStatus> = Lazy::new(|| RegisteredStatus::register("ZEROCLAW", None));

/// What the last probe found out.
#[derive(Clone, Debug, PartialEq)]
pub enum Health {
    /// No address, agent or token yet; the text says what is missing.
    Unconfigured(String),
    Checking,
    Online { agents: Vec<String>, require_pairing: bool, latency_ms: u64 },
    /// The gateway did not answer.
    Offline(String),
    /// The gateway refused the token (HTTP 401 or 403).
    Unauthorized,
    /// The token works but the gateway has no agent of that name.
    UnknownAgent { agent: String, available: Vec<String> },
}

/// [`Health`] without its data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthKind {
    Unconfigured,
    Checking,
    Online,
    Offline,
    Unauthorized,
    UnknownAgent,
}

impl HealthKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unconfigured => "unconfigured",
            Self::Checking => "checking",
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Unauthorized => "unauthorized",
            Self::UnknownAgent => "unknown_agent",
        }
    }
}

impl Health {
    pub fn kind(&self) -> HealthKind {
        match self {
            Self::Unconfigured(_) => HealthKind::Unconfigured,
            Self::Checking => HealthKind::Checking,
            Self::Online { .. } => HealthKind::Online,
            Self::Offline(_) => HealthKind::Offline,
            Self::Unauthorized => HealthKind::Unauthorized,
            Self::UnknownAgent { .. } => HealthKind::UnknownAgent,
        }
    }

    pub fn is_online(&self) -> bool {
        matches!(self, Self::Online { .. })
    }

    /// True when the sweep must leave the gateway alone until the person acts.
    pub fn is_refused(&self) -> bool {
        matches!(self, Self::Unauthorized)
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Unconfigured(_) => "Not set up",
            Self::Checking => "Checking",
            Self::Online { .. } => "Online",
            Self::Offline(_) => "Offline",
            Self::Unauthorized => "Token refused",
            Self::UnknownAgent { .. } => "Unknown agent",
        }
    }

    /// One sentence for hover cards, banners and the command result.
    pub fn detail(&self) -> String {
        match self {
            Self::Unconfigured(why) => why.clone(),
            Self::Checking => "Asking the gateway".to_string(),
            Self::Online { agents, latency_ms, .. } => format!("{} agents, {latency_ms} ms", agents.len()),
            Self::Offline(why) => why.clone(),
            Self::Unauthorized => "The gateway refused the token. Pair again in Settings > Assistant".to_string(),
            Self::UnknownAgent { agent, available } => {
                if available.is_empty() {
                    format!("The gateway has no agent named '{agent}'")
                } else {
                    format!("The gateway has no agent named '{agent}'; it offers {}", available.join(", "))
                }
            }
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({ "state": self.kind().as_str(), "detail": self.detail() });
        match self {
            Self::Online { agents, require_pairing, latency_ms } => {
                v["agents"] = agents.clone().into();
                v["require_pairing"] = (*require_pairing).into();
                v["latency_ms"] = (*latency_ms).into();
            }
            Self::UnknownAgent { available, .. } => v["agents"] = available.clone().into(),
            _ => {}
        }
        v
    }
}

/// Maps a client error from a health request onto the taxonomy.
pub fn classify_error(error: &Error) -> Health {
    if error.is_auth() { Health::Unauthorized } else { Health::Offline(short_reason(error)) }
}

fn short_reason(error: &Error) -> String {
    match error {
        Error::Timeout(_) => "no answer within the time limit".to_string(),
        Error::Connect(text) => {
            let text = text.trim();
            if text.is_empty() { "cannot connect".to_string() } else { text.to_string() }
        }
        other => other.to_string(),
    }
}

/// The client configuration for `profile`; the token is passed in, never stored here.
pub fn client_config(profile: &ZcProfile, token: &str) -> Config {
    let mut cfg = Config::new(profile.host.trim(), token).with_agent(profile.agent.trim()).with_device_name("aimediaexplorer");
    cfg.connect_timeout = PROBE_BUDGET;
    cfg.request_timeout = PROBE_BUDGET;
    cfg
}

/// Checks the gateway: reachability, the token and the agent alias, within [`PROBE_BUDGET`].
pub async fn probe(profile: &ZcProfile, token: Option<&str>) -> Health {
    if profile.host.trim().is_empty() {
        return Health::Unconfigured("No gateway address".to_string());
    }
    if let Err(e) = profile.endpoint() {
        return Health::Unconfigured(format!("Bad gateway address: {e}"));
    }
    let Some(token) = token.filter(|t| !t.trim().is_empty()) else {
        return Health::Unconfigured("No token: pair with the gateway or import one".to_string());
    };
    if profile.agent.trim().is_empty() {
        return Health::Unconfigured("No agent alias".to_string());
    }
    let started = Instant::now();
    let rest = match RestClient::new(&client_config(profile, token)) {
        Ok(r) => r,
        Err(e) => return Health::Unconfigured(e.to_string()),
    };
    match tokio::time::timeout(PROBE_BUDGET, rest.probe()).await {
        Err(_) => Health::Offline("no answer within the time limit".to_string()),
        Ok(Err(e)) => classify_error(&e),
        Ok(Ok(probe)) => match probe.auth {
            AuthState::Rejected(_) => Health::Unauthorized,
            AuthState::NotChecked => Health::Unconfigured("No token: pair with the gateway or import one".to_string()),
            AuthState::Accepted => {
                if probe.agents.iter().any(|a| a == profile.agent.trim()) {
                    Health::Online { agents: probe.agents, require_pairing: probe.health.require_pairing, latency_ms: started.elapsed().as_millis() as u64 }
                } else {
                    Health::UnknownAgent { agent: profile.agent.trim().to_string(), available: probe.agents }
                }
            }
        },
    }
}

/// The Needs attention entry for a refused token.
pub fn attention_item(profile: &ZcProfile) -> AttentionItem {
    AttentionItem {
        key: "zeroclaw-token".to_string(),
        kind: AttentionKind::Gateway,
        title: "ZeroClaw token refused".to_string(),
        detail: format!("{} did not accept the token. It is not retried; pair again.", profile.host_label()),
        count: None,
        action: Some(AppCommand::ShowSettings { section: Some(Section::Assistant) }),
    }
}

static LAST_MODEL: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Shows `health` on the ZEROCLAW engine dot and keeps the Needs attention list in step.
pub fn publish(health: &Health, profile: &ZcProfile) {
    let status = &*ZEROCLAW_STATUS;
    status.clear_error();
    match health {
        Health::Unconfigured(why) => status.set_state(StatusState::Idle, why.clone()),
        Health::Checking => status.set_state(StatusState::Initializing, format!("Checking {}", profile.host_label())),
        Health::Online { agents, latency_ms, .. } => {
            status.set_state(StatusState::Idle, format!("Online, {} agents, {latency_ms} ms", agents.len()));
            set_model_once(&format!("{} @ {}", profile.agent.trim(), profile.host_label()));
        }
        Health::Offline(why) => status.set_state(StatusState::Waiting, format!("Offline: {why}")),
        Health::UnknownAgent { .. } => status.set_state(StatusState::Waiting, health.detail()),
        Health::Unauthorized => {
            status.set_state(StatusState::Error, health.detail());
            status.set_error(health.detail());
        }
    }
    let items = if health.is_refused() { vec![attention_item(profile)] } else { Vec::new() };
    crate::tasks::with(|r| r.set_attention(ATTENTION_SOURCE, items));
    crate::command::request_repaint();
}

/// The model string leaks on every change, so it is only set when it differs.
fn set_model_once(model: &str) {
    let mut last = LAST_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if *last != model {
        *last = model.to_string();
        ZEROCLAW_STATUS.set_model(model);
    }
}

/// Holds the global status and attention list still while a test reads them.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_client::mock::{MockGateway, MockOptions};

    fn profile_for(mock: &MockGateway, agent: &str) -> ZcProfile {
        ZcProfile { host: mock.url(), agent: agent.into(), ..Default::default() }
    }

    #[tokio::test]
    async fn an_incomplete_profile_is_unconfigured_and_never_touches_the_network() {
        let p = ZcProfile::default();
        assert_eq!(probe(&p, Some("t")).await.kind(), HealthKind::Unconfigured);
        let p = ZcProfile { host: "127.0.0.1:9".into(), ..Default::default() };
        assert!(matches!(probe(&p, None).await, Health::Unconfigured(w) if w.contains("token")));
        assert!(matches!(probe(&p, Some("t")).await, Health::Unconfigured(w) if w.contains("agent")));
        let bad = ZcProfile { host: "ftp://x".into(), agent: "a".into(), ..Default::default() };
        assert!(matches!(probe(&bad, Some("t")).await, Health::Unconfigured(w) if w.contains("address")));
    }

    #[tokio::test]
    async fn a_reachable_gateway_with_the_agent_is_online() {
        let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into(), "coder".into()], ..Default::default() }).await;
        let health = probe(&profile_for(&mock, "media"), mock.token()).await;
        match health {
            Health::Online { agents, .. } => assert_eq!(agents, vec!["media".to_string(), "coder".to_string()]),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_refused_token_is_unauthorized() {
        let mock = MockGateway::start().await;
        assert_eq!(probe(&profile_for(&mock, "default"), Some("wrong")).await, Health::Unauthorized);
    }

    #[tokio::test]
    async fn an_agent_the_gateway_lacks_is_reported_with_the_alternatives() {
        let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into()], ..Default::default() }).await;
        match probe(&profile_for(&mock, "nope"), mock.token()).await {
            Health::UnknownAgent { agent, available } => assert_eq!((agent.as_str(), available), ("nope", vec!["media".to_string()])),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_closed_port_is_offline() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let p = ZcProfile { host: addr.to_string(), agent: "a".into(), ..Default::default() };
        assert_eq!(probe(&p, Some("t")).await.kind(), HealthKind::Offline);
    }

    #[test]
    fn errors_map_onto_the_taxonomy() {
        assert_eq!(classify_error(&Error::Unauthorized), Health::Unauthorized);
        assert_eq!(classify_error(&Error::Forbidden), Health::Unauthorized);
        assert_eq!(classify_error(&Error::Timeout("request")).kind(), HealthKind::Offline);
        assert!(matches!(classify_error(&Error::Connect("refused".into())), Health::Offline(t) if t == "refused"));
        assert_eq!(classify_error(&Error::Http { status: 503, path: "/health".into(), message: String::new() }).kind(), HealthKind::Offline);
    }

    #[test]
    fn only_a_refused_token_stops_the_sweep() {
        assert!(Health::Unauthorized.is_refused());
        assert!(!Health::Offline("x".into()).is_refused());
        assert!(!Health::Unconfigured("x".into()).is_refused());
    }

    #[test]
    fn the_details_name_what_to_do() {
        assert!(Health::Unauthorized.detail().contains("Pair again"));
        let unknown = Health::UnknownAgent { agent: "x".into(), available: vec!["a".into(), "b".into()] };
        assert!(unknown.detail().contains("x") && unknown.detail().contains("a, b"));
        let json = Health::Online { agents: vec!["m".into()], require_pairing: true, latency_ms: 4 }.to_json();
        assert_eq!((json["state"].as_str(), json["latency_ms"].as_u64()), (Some("online"), Some(4)));
    }

    #[test]
    fn a_refused_token_lists_a_re_pair_item_and_other_states_clear_it() {
        let _serial = serial();
        let profile = ZcProfile { host: "10.0.0.5".into(), agent: "a".into(), ..Default::default() };
        publish(&Health::Unauthorized, &profile);
        let items: Vec<_> = crate::tasks::with(|r| r.attention()).into_iter().filter(|i| i.key == "zeroclaw-token").collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, AttentionKind::Gateway);
        assert!(items[0].detail.contains("10.0.0.5") && items[0].detail.contains("not retried"));
        assert!(matches!(items[0].action, Some(AppCommand::ShowSettings { section: Some(Section::Assistant) })));
        assert_eq!(ZEROCLAW_STATUS.snapshot().state, StatusState::Error);
        publish(&Health::Offline("down".into()), &profile);
        assert!(crate::tasks::with(|r| r.attention()).iter().all(|i| i.key != "zeroclaw-token"));
        assert_eq!(ZEROCLAW_STATUS.snapshot().state, StatusState::Waiting);
        assert!(ZEROCLAW_STATUS.snapshot().error.is_none());
    }
}
