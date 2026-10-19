use std::fmt;
use std::time::Duration;

use url::Url;

use crate::error::{Error, Result};
use crate::text::component;

/// The gateway's own port, used when the configured host carries none.
pub const DEFAULT_PORT: u16 = 42617;

/// Bearer token; `Debug` prints no value and there is no `Display`.
#[derive(Clone, PartialEq, Eq)]
pub struct Token(String);

impl Token {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        let value = value.trim();
        (!value.is_empty()).then(|| Self(value.to_string()))
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

/// Where the WebSocket upgrade carries the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WsAuth {
    /// `Authorization` header first, `?token=` when the header is refused with HTTP 401.
    #[default]
    Auto,
    Header,
    Query,
}

/// Backoff schedule for automatic reconnects.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconnectPolicy {
    pub enabled: bool,
    pub initial: Duration,
    pub max: Duration,
    pub multiplier: f64,
    /// Fraction of each delay randomised downwards, 0.0 to 1.0.
    pub jitter: f64,
    /// Consecutive failed attempts before giving up; `None` retries forever.
    pub max_attempts: Option<u32>,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            initial: Duration::from_millis(500),
            max: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: 0.2,
            max_attempts: None,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    base_url: String,
    token: Option<Token>,
    pub agent: String,
    pub device_name: String,
    pub connect_timeout: Duration,
    pub handshake_timeout: Duration,
    pub write_timeout: Duration,
    pub request_timeout: Duration,
    pub complete_timeout: Duration,
    pub ping_interval: Duration,
    pub pong_timeout: Duration,
    /// Wait for `turn_resume` after a mid-turn reconnect before asking the gateway about the turn.
    pub resume_grace: Duration,
    pub max_frame_bytes: usize,
    pub max_outbound_bytes: usize,
    pub max_rest_body_bytes: usize,
    pub event_buffer: usize,
    pub reconnect: ReconnectPolicy,
    pub ws_auth: WsAuth,
    /// Extra trusted root certificate (PEM) for self-hosted gateways.
    pub extra_root_pem: Option<String>,
}

impl Config {
    /// `base_url` accepts `host`, `host:port`, or an `http`/`https`/`ws`/`wss` URL with an optional path prefix.
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            token: Token::new(token),
            agent: String::new(),
            device_name: "zeroclaw-client".into(),
            connect_timeout: Duration::from_secs(15),
            handshake_timeout: Duration::from_secs(20),
            write_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(20),
            complete_timeout: Duration::from_secs(90),
            ping_interval: Duration::from_secs(20),
            pong_timeout: Duration::from_secs(60),
            resume_grace: Duration::from_secs(10),
            max_frame_bytes: 16 * 1024 * 1024,
            max_outbound_bytes: 24 * 1024 * 1024,
            max_rest_body_bytes: 24 * 1024 * 1024,
            event_buffer: 1024,
            reconnect: ReconnectPolicy::default(),
            ws_auth: WsAuth::Auto,
            extra_root_pem: None,
        }
    }

    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = agent.into();
        self
    }

    pub fn with_device_name(mut self, name: impl Into<String>) -> Self {
        self.device_name = name.into();
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    pub fn token(&self) -> Option<&Token> {
        self.token.as_ref()
    }

    pub fn set_token(&mut self, token: impl Into<String>) {
        self.token = Token::new(token);
    }

    pub fn agent_name(&self) -> &str {
        self.agent.trim()
    }

    pub fn endpoint(&self) -> Result<Endpoint> {
        Endpoint::parse(&self.base_url)
    }

    /// Replace every occurrence of the token (raw and percent-encoded) in `text`.
    pub fn redact(&self, text: &str) -> String {
        redact_token(text, self.token.as_ref())
    }

    /// Take the first ZeroClaw machine out of a ZC Codex `config.json`: the selected one when it is a ZeroClaw entry.
    pub fn from_zc_codex_json(text: &str) -> Option<Config> {
        use serde_json::Value;
        let v: Value = serde_json::from_str(text).ok()?;
        let machines = v.get("machines")?.as_array()?;
        let usable = |m: &Value| {
            let kind = m.get("kind").and_then(Value::as_str).unwrap_or("");
            let host = m.get("host").and_then(Value::as_str).unwrap_or("").trim();
            let token = m.get("token").and_then(Value::as_str).unwrap_or("").trim();
            (kind == "ZeroClaw" && !host.is_empty() && !token.is_empty()).then(|| {
                let agent = m.get("agent").and_then(Value::as_str).unwrap_or("");
                Config::new(host, token).with_agent(agent)
            })
        };
        let selected = match v.get("selected") {
            Some(Value::Number(n)) => n.as_u64().and_then(|i| machines.get(i as usize)),
            Some(Value::String(name)) => machines
                .iter()
                .find(|m| m.get("name").and_then(Value::as_str) == Some(name.as_str())),
            _ => None,
        };
        selected.and_then(usable).or_else(|| machines.iter().find_map(usable))
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("base_url", &self.base_url)
            .field("token", &if self.token.is_some() { "<set>" } else { "<unset>" })
            .field("agent", &self.agent)
            .field("ws_auth", &self.ws_auth)
            .field("reconnect", &self.reconnect)
            .finish_non_exhaustive()
    }
}

pub(crate) fn redact_token(text: &str, token: Option<&Token>) -> String {
    let Some(token) = token else { return text.to_string() };
    let raw = token.expose();
    let encoded = component(raw);
    let mut out = text.replace(raw, "***");
    if encoded != raw {
        out = out.replace(&encoded, "***");
    }
    out
}

/// A parsed gateway origin with an optional path prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    tls: bool,
    /// `http(s)://host[:port][/prefix]` with no trailing slash.
    http_base: Url,
}

impl Endpoint {
    pub fn parse(raw: &str) -> Result<Self> {
        let raw = raw.trim().trim_end_matches('/');
        if raw.is_empty() {
            return Err(Error::Config("gateway address is empty".into()));
        }
        let (scheme, rest) = match raw.split_once("://") {
            Some((s, r)) => (s.to_ascii_lowercase(), r),
            None => ("http".to_string(), raw),
        };
        let tls = match scheme.as_str() {
            "http" | "ws" => false,
            "https" | "wss" => true,
            other => return Err(Error::Config(format!("unsupported scheme `{other}`"))),
        };
        let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let (authority, tail) = rest.split_at(authority_end);
        if authority.is_empty() || authority.contains('@') {
            return Err(Error::Config("gateway address needs a host and no credentials".into()));
        }
        let authority = if tls || has_port(authority) {
            authority.to_string()
        } else {
            format!("{authority}:{DEFAULT_PORT}")
        };
        let tail = tail.split(['?', '#']).next().unwrap_or("").trim_end_matches('/');
        let http_scheme = if tls { "https" } else { "http" };
        let url = Url::parse(&format!("{http_scheme}://{authority}{tail}"))
            .map_err(|e| Error::Config(format!("invalid gateway address: {e}")))?;
        if url.host_str().is_none() {
            return Err(Error::Config("gateway address has no host".into()));
        }
        Ok(Self { tls, http_base: url })
    }

    pub fn tls(&self) -> bool {
        self.tls
    }

    /// `host[:port]` as shown to people, no scheme or path.
    pub fn host_port(&self) -> String {
        let host = self.http_base.host_str().unwrap_or("");
        match self.http_base.port() {
            Some(p) => format!("{host}:{p}"),
            None => host.to_string(),
        }
    }

    /// An `http(s)` URL for `path` (leading slash) with query pairs appended.
    pub fn http_url(&self, path: &str, query: &[(&str, &str)]) -> Url {
        self.build(self.http_base.scheme(), path, query)
    }

    /// A `ws(s)` URL for `path` with query pairs appended.
    pub fn ws_url(&self, path: &str, query: &[(&str, &str)]) -> Url {
        self.build(if self.tls { "wss" } else { "ws" }, path, query)
    }

    fn build(&self, scheme: &str, path: &str, query: &[(&str, &str)]) -> Url {
        let mut url = self.http_base.clone();
        let prefix = url.path().trim_end_matches('/').to_string();
        url.set_path(&format!("{prefix}{path}"));
        let _ = url.set_scheme(scheme);
        if !query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (k, v) in query {
                pairs.append_pair(k, v);
            }
        }
        url
    }
}

/// True when the authority ends in `:port`; IPv6 colons do not count.
fn has_port(authority: &str) -> bool {
    match authority.rsplit_once(':') {
        Some((before, after)) => {
            !after.is_empty()
                && after.chars().all(|c| c.is_ascii_digit())
                && (!before.contains(':') || before.ends_with(']'))
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_host_gets_the_gateway_port() {
        let e = Endpoint::parse("100.96.121.37").unwrap();
        assert!(!e.tls());
        assert_eq!(e.host_port(), "100.96.121.37:42617");
        assert_eq!(e.ws_url("/ws/chat", &[("agent", "a b")]).as_str(), "ws://100.96.121.37:42617/ws/chat?agent=a+b");
        assert_eq!(e.http_url("/health", &[]).as_str(), "http://100.96.121.37:42617/health");
    }

    #[test]
    fn explicit_port_is_kept_even_when_it_is_a_scheme_default() {
        assert_eq!(Endpoint::parse("host:9000").unwrap().host_port(), "host:9000");
        assert_eq!(Endpoint::parse("http://host:80").unwrap().host_port(), "host");
        assert_eq!(Endpoint::parse("host:80").unwrap().http_url("/x", &[]).as_str(), "http://host/x");
    }

    #[test]
    fn tls_schemes_keep_the_implied_port() {
        for raw in ["wss://claw.example.app", "https://claw.example.app/"] {
            let e = Endpoint::parse(raw).unwrap();
            assert!(e.tls());
            assert_eq!(e.host_port(), "claw.example.app");
            assert_eq!(e.ws_url("/ws/chat", &[]).as_str(), "wss://claw.example.app/ws/chat");
            assert_eq!(e.http_url("/api/complete", &[]).as_str(), "https://claw.example.app/api/complete");
        }
    }

    #[test]
    fn path_prefix_is_preserved() {
        let e = Endpoint::parse("https://example.app/zc/").unwrap();
        assert_eq!(e.ws_url("/ws/chat", &[]).as_str(), "wss://example.app/zc/ws/chat");
        assert_eq!(e.http_url("/health", &[]).as_str(), "https://example.app/zc/health");
    }

    #[test]
    fn ipv6_literals_are_handled() {
        assert_eq!(Endpoint::parse("[::1]").unwrap().host_port(), "[::1]:42617");
        assert_eq!(Endpoint::parse("[::1]:9000").unwrap().host_port(), "[::1]:9000");
    }

    #[test]
    fn bad_addresses_are_rejected() {
        assert!(Endpoint::parse("").is_err());
        assert!(Endpoint::parse("  ").is_err());
        assert!(Endpoint::parse("ftp://host").is_err());
        assert!(Endpoint::parse("http://user:pw@host").is_err());
        assert!(Endpoint::parse("http://").is_err());
    }

    #[test]
    fn token_is_redacted_everywhere() {
        let cfg = Config::new("host", "s3cret/tok+en").with_agent("x");
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("s3cret"), "{dbg}");
        assert!(dbg.contains("<set>"));
        assert!(!format!("{:?}", cfg.token().unwrap()).contains("s3cret"));
        let leaked = "GET /ws?token=s3cret%2Ftok%2Ben and s3cret/tok+en";
        let clean = cfg.redact(leaked);
        assert!(!clean.contains("s3cret"), "{clean}");
        assert_eq!(clean, "GET /ws?token=*** and ***");
    }

    #[test]
    fn blank_token_means_unset() {
        assert!(!Config::new("h", "   ").has_token());
        assert!(Config::new("h", " t ").has_token());
        assert_eq!(Config::new("h", " t ").token().unwrap().expose(), "t");
    }

    #[test]
    fn zc_codex_config_prefers_the_selected_zeroclaw_machine() {
        let cfg = r#"{"machines":[
            {"name":"desktop","host":"100.0.0.1:7420","token":"a","kind":"Codex","agent":""},
            {"name":"zeroclaw","host":"100.0.0.2:42617","token":"b","kind":"ZeroClaw","agent":"companion"},
            {"name":"public","host":"wss://claw.example.app","token":"c","kind":"ZeroClaw","agent":"companion"}
        ],"selected":2}"#;
        let c = Config::from_zc_codex_json(cfg).unwrap();
        assert_eq!(c.base_url(), "wss://claw.example.app");
        assert_eq!(c.token().unwrap().expose(), "c");
        assert_eq!(c.agent_name(), "companion");
        let by_name = cfg.replace("\"selected\":2", "\"selected\":\"zeroclaw\"");
        assert_eq!(Config::from_zc_codex_json(&by_name).unwrap().base_url(), "100.0.0.2:42617");
    }

    #[test]
    fn zc_codex_config_falls_through_and_rejects_junk() {
        let cfg = r#"{"machines":[
            {"name":"desktop","host":"h","token":"a","kind":"Codex"},
            {"name":"zeroclaw","host":"100.0.0.2:42617","token":"b","kind":"ZeroClaw"}
        ],"selected":0}"#;
        assert_eq!(Config::from_zc_codex_json(cfg).unwrap().base_url(), "100.0.0.2:42617");
        assert!(Config::from_zc_codex_json(r#"{"machines":[{"kind":"Codex","host":"h","token":"t"}]}"#).is_none());
        assert!(Config::from_zc_codex_json(r#"{"machines":[{"kind":"ZeroClaw","host":"h","token":""}]}"#).is_none());
        assert!(Config::from_zc_codex_json("not json").is_none());
    }
}
