//! The tools server as the connection card sees it: status, masked token and start, stop, rotate and tailnet exposure, all through the MCP server's public API.

use crate::ai::mcp::command::{McpOp, run as run_mcp};
use crate::ai::mcp::config::McpConfig;
use crate::ai::mcp::{ServerState, auth, hub};
use crate::command::CommandResult;

use super::profile::ZcProfile;

/// What the card shows about the tools server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolsStatus {
    pub state: &'static str,
    pub detail: String,
    pub url: String,
    pub clients: usize,
    pub running: bool,
}

pub fn status() -> ToolsStatus {
    let hub = hub();
    let (state, detail, running) = match hub.state() {
        ServerState::Stopped => ("stopped", "Not listening".to_string(), false),
        ServerState::Starting => ("starting", "Starting".to_string(), false),
        ServerState::Running { addr, .. } => ("running", format!("Listening on {addr}"), true),
        ServerState::Failed(e) => ("failed", e, false),
    };
    ToolsStatus { state, detail, url: hub.url(), clients: hub.clients().len(), running }
}

/// The token with all but its first characters hidden; `None` until a token can be created.
pub fn masked_token() -> Option<String> {
    hub().token().ok().map(|t| auth::mask(&t))
}

/// The gateway configuration snippet with the token left as a placeholder.
pub fn gateway_snippet() -> String {
    hub().gateway_snippet()
}

pub fn start() -> CommandResult {
    run_mcp(McpOp::Start, None, None, None)
}

pub fn stop() -> CommandResult {
    run_mcp(McpOp::Stop, None, None, None)
}

pub fn rotate_token() -> CommandResult {
    run_mcp(McpOp::RotateToken, None, None, None)
}

/// The MCP settings that follow from `profile`: its port, and the interface and allowed host of the tailnet toggle.
pub fn config_for(profile: &ZcProfile, current: &McpConfig) -> McpConfig {
    let mut cfg = current.clone();
    cfg.port = profile.tools_port;
    cfg.enabled = profile.tools_enabled;
    if profile.tools_on_tailnet() {
        let bind = profile.tools_bind.trim().to_string();
        cfg.allow_tailnet = true;
        cfg.bind_addr = bind.clone();
        if !cfg.allowed_hosts.iter().any(|h| h == &bind) {
            cfg.allowed_hosts.push(bind);
        }
    } else {
        cfg.allow_tailnet = false;
    }
    cfg
}

/// Writes the profile's tools choices into the MCP settings; a running server keeps its old address until restarted.
pub fn apply_profile(profile: &ZcProfile) -> Result<(), String> {
    let hub = hub();
    let next = config_for(profile, &hub.config());
    if next != hub.config() {
        hub.set_config(next, true)?;
    }
    Ok(())
}

/// Applies the profile and starts the server when it is switched on and not yet running.
pub fn ensure_running(profile: &ZcProfile) {
    if !profile.tools_enabled {
        return;
    }
    if let Err(e) = apply_profile(profile) {
        log::warn!("[zc] tools settings not saved: {e}");
    }
    if matches!(hub().state(), ServerState::Stopped | ServerState::Failed(_)) {
        let _ = start();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_loopback_profile_keeps_the_tailnet_off() {
        let p = ZcProfile { tools_enabled: true, tools_port: 9999, ..Default::default() };
        let cfg = config_for(&p, &McpConfig { allow_tailnet: true, ..Default::default() });
        assert!(cfg.enabled && !cfg.allow_tailnet);
        assert_eq!(cfg.port, 9999);
    }

    #[test]
    fn the_tailnet_toggle_binds_that_interface_and_allows_its_host_once() {
        let p = ZcProfile { tools_enabled: true, tools_bind: "100.116.134.128".into(), ..Default::default() };
        let cfg = config_for(&p, &McpConfig::default());
        assert!(cfg.allow_tailnet);
        assert_eq!((cfg.bind_addr.as_str(), cfg.allowed_hosts.as_slice()), ("100.116.134.128", ["100.116.134.128".to_string()].as_slice()));
        let again = config_for(&p, &cfg);
        assert_eq!(again.allowed_hosts.len(), 1);
        assert!(cfg.binding().is_ok(), "the resulting settings are bindable");
    }

    #[test]
    fn a_profile_that_switches_tools_off_turns_the_start_flag_off() {
        let cfg = config_for(&ZcProfile::default(), &McpConfig { enabled: true, ..Default::default() });
        assert!(!cfg.enabled);
    }

    #[test]
    fn the_status_names_a_state_and_the_masked_token_hides_the_rest() {
        let s = status();
        assert!(["stopped", "starting", "running", "failed"].contains(&s.state));
        assert!(s.url.starts_with("http://") && s.url.ends_with("/mcp"));
        if let Some(masked) = masked_token() {
            assert!(masked.contains('*') || masked.len() <= 4);
        }
        assert!(gateway_snippet().contains("[[mcp.servers]]"));
        assert!(gateway_snippet().contains("<paste the token"), "the snippet never holds the real token");
    }
}
