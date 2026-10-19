//! The gateway profile: the assistant's connection choices, stored as one nested settings field. The token never lives here.

use std::net::IpAddr;

use serde::{Deserialize, Serialize};
use surrealdb::types::SurrealValue;
use zeroclaw_client::Endpoint;

/// `/api/complete` hint used when the profile names none.
pub const DEFAULT_HINT: &str = "quick";
/// Port the tools server listens on unless the profile says otherwise.
pub const DEFAULT_TOOLS_PORT: u16 = 9871;
/// Interface the tools server binds unless the tailnet toggle names another.
pub const LOOPBACK: &str = "127.0.0.1";

/// Which engine answers in the assistant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPref {
    /// ZeroClaw once it is set up and reachable, onboarding before that.
    #[default]
    Auto,
    #[serde(rename = "zeroclaw")]
    ZeroClaw,
    Direct,
    Local,
}

impl BackendPref {
    pub const ALL: [BackendPref; 4] = [Self::Auto, Self::ZeroClaw, Self::Direct, Self::Local];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ZeroClaw => "zeroclaw",
            Self::Direct => "direct",
            Self::Local => "local",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::ZeroClaw => "ZeroClaw",
            Self::Direct => "Direct provider",
            Self::Local => "Local only",
        }
    }

    /// Unknown names read as `Auto`.
    pub fn parse(text: &str) -> Self {
        Self::ALL.into_iter().find(|b| b.as_str().eq_ignore_ascii_case(text.trim())).unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, SurrealValue)]
pub struct ZcProfile {
    /// Label shown in the assistant header.
    pub name: String,
    /// `host`, `host:port` or an `http(s)`/`ws(s)` URL; a bare host gets the gateway port.
    pub host: String,
    /// Agent alias every chat session is bound to.
    pub agent: String,
    /// `/api/complete` route hint for titles and one-shot jobs.
    pub complete_hint: String,
    /// Images may leave the machine inside chat messages.
    pub allow_images: bool,
    /// Serve the MCP tools to the gateway with the application.
    pub tools_enabled: bool,
    /// Interface the tools server binds; loopback unless it is exposed on the tailnet.
    pub tools_bind: String,
    pub tools_port: u16,
    /// [`BackendPref`] as text.
    pub backend: String,
}

impl Default for ZcProfile {
    fn default() -> Self {
        Self {
            name: "ZeroClaw".into(),
            host: String::new(),
            agent: String::new(),
            complete_hint: DEFAULT_HINT.into(),
            allow_images: false,
            tools_enabled: false,
            tools_bind: LOOPBACK.into(),
            tools_port: DEFAULT_TOOLS_PORT,
            backend: BackendPref::Auto.as_str().into(),
        }
    }
}

impl ZcProfile {
    /// A gateway address and an agent alias are set.
    pub fn is_configured(&self) -> bool {
        !self.host.trim().is_empty() && !self.agent.trim().is_empty()
    }

    pub fn backend_pref(&self) -> BackendPref {
        BackendPref::parse(&self.backend)
    }

    pub fn hint(&self) -> &str {
        let h = self.complete_hint.trim();
        if h.is_empty() { DEFAULT_HINT } else { h }
    }

    /// The parsed address, or why it is not one.
    pub fn endpoint(&self) -> Result<Endpoint, String> {
        Endpoint::parse(&self.host).map_err(|e| e.to_string())
    }

    /// `host[:port]` for display, or the text as typed when it does not parse.
    pub fn host_label(&self) -> String {
        self.endpoint().map(|e| e.host_port()).unwrap_or_else(|_| self.host.trim().to_string())
    }

    /// Name for the header chip.
    pub fn display_name(&self) -> String {
        let name = self.name.trim();
        if name.is_empty() { self.host_label() } else { name.to_string() }
    }

    /// True when the gateway sits on this machine or a private network, so images stay inside it.
    pub fn is_local_origin(&self) -> bool {
        self.endpoint().map(|e| is_local_host(&e.host_port(), e.tls())).unwrap_or(false)
    }

    /// Whether the tools server is exposed beyond loopback.
    pub fn tools_on_tailnet(&self) -> bool {
        let bind = self.tools_bind.trim();
        !bind.is_empty() && bind != LOOPBACK && bind != "localhost" && bind != "::1"
    }
}

/// True for loopback, private, link-local and tailnet addresses and for `.local`, `.lan` and `.ts.net` names; a TLS origin on a public name is not local.
pub fn is_local_host(host_port: &str, tls: bool) -> bool {
    let host = match host_port.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !h.ends_with(':') => h,
        _ => host_port,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => {
                let o = v4.octets();
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || (o[0] == 100 && (64..=127).contains(&o[1]))
            }
            IpAddr::V6(v6) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00,
        };
    }
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".local") || lower.ends_with(".lan") || lower.ends_with(".ts.net") {
        return true;
    }
    !tls && !lower.contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_profile_is_unconfigured_text_only_and_auto() {
        let p = ZcProfile::default();
        assert!(!p.is_configured());
        assert!(!p.allow_images && !p.tools_enabled);
        assert_eq!(p.backend_pref(), BackendPref::Auto);
        assert_eq!((p.hint(), p.tools_port, p.tools_bind.as_str()), ("quick", 9871, "127.0.0.1"));
    }

    #[test]
    fn a_profile_needs_a_host_and_an_agent() {
        let mut p = ZcProfile { host: "100.96.121.37".into(), ..Default::default() };
        assert!(!p.is_configured());
        p.agent = "media".into();
        assert!(p.is_configured());
        p.host = "  ".into();
        assert!(!p.is_configured());
    }

    #[test]
    fn host_parsing_follows_the_client_crate() {
        let p = ZcProfile { host: "100.96.121.37".into(), agent: "a".into(), ..Default::default() };
        assert_eq!(p.host_label(), "100.96.121.37:42617");
        let tls = ZcProfile { host: "wss://claw.example.app".into(), ..Default::default() };
        assert_eq!(tls.host_label(), "claw.example.app");
        assert!(tls.endpoint().unwrap().tls());
        let bad = ZcProfile { host: "ftp://x".into(), ..Default::default() };
        assert!(bad.endpoint().is_err());
        assert_eq!(bad.host_label(), "ftp://x");
    }

    #[test]
    fn the_backend_pick_round_trips_and_unknown_text_reads_as_auto() {
        for b in BackendPref::ALL {
            assert_eq!(BackendPref::parse(b.as_str()), b);
        }
        assert_eq!(BackendPref::parse(" Direct "), BackendPref::Direct);
        assert_eq!(BackendPref::parse("cloud"), BackendPref::Auto);
        assert_eq!(serde_json::to_string(&BackendPref::ZeroClaw).unwrap(), "\"zeroclaw\"");
    }

    #[test]
    fn local_origins_are_private_networks_and_not_public_tls_names() {
        for local in ["127.0.0.1:42617", "192.168.1.5:42617", "10.0.0.2:1", "100.96.121.37:42617", "localhost:1", "box.local:1", "host.tail1234.ts.net", "[::1]:42617", "myserver:42617"] {
            assert!(is_local_host(local, false), "{local}");
        }
        assert!(!is_local_host("claw.example.app", true));
        assert!(!is_local_host("8.8.8.8:42617", false));
        assert!(!is_local_host("claw.shadowbroker.app", true));
        assert!(ZcProfile { host: "100.96.121.37".into(), ..Default::default() }.is_local_origin());
        assert!(!ZcProfile { host: "wss://claw.example.app".into(), ..Default::default() }.is_local_origin());
        assert!(!ZcProfile { host: "garbage://x".into(), ..Default::default() }.is_local_origin());
    }

    #[test]
    fn tailnet_exposure_is_any_bind_other_than_loopback() {
        let mut p = ZcProfile::default();
        assert!(!p.tools_on_tailnet());
        p.tools_bind = "100.116.134.128".into();
        assert!(p.tools_on_tailnet());
        p.tools_bind = "localhost".into();
        assert!(!p.tools_on_tailnet());
    }

    #[test]
    fn the_profile_survives_json() {
        let p = ZcProfile { name: "n".into(), host: "h".into(), agent: "a".into(), allow_images: true, ..Default::default() };
        let back: ZcProfile = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }
}
