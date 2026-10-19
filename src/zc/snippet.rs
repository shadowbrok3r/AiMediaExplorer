//! The gateway configuration snippet for the tools server, and a ping of the server through its own HTTP address.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::events::Pending;
use super::profile::{LOOPBACK, ZcProfile};
use super::runtime::ZcRuntime;
use crate::ai::mcp::config::McpConfig;
use crate::ai::mcp::server::{MCP_PATH, gateway_snippet_for};
use crate::ai::mcp::{Hub, hub};

/// Agent alias used in the snippet when the profile names none.
pub const DEFAULT_AGENT: &str = "media";
/// Longest wait for one request of the ping.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(8);

/// The address the gateway would use to reach the tools server on `port`: the interface of the profile or of the server settings when one is exposed, else loopback.
pub fn url_for(profile: &ZcProfile, cfg: &McpConfig, port: u16) -> String {
    let exposed = cfg.allow_tailnet && !cfg.bind_addr.trim().is_empty();
    let host = if profile.tools_on_tailnet() {
        profile.tools_bind.trim().to_string()
    } else if exposed {
        cfg.bind_addr.trim().to_string()
    } else {
        LOOPBACK.to_string()
    };
    format!("http://{host}:{port}{MCP_PATH}")
}

/// True when the address in the snippet can be reached from another machine.
fn reachable_from_outside(profile: &ZcProfile, cfg: &McpConfig) -> bool {
    profile.tools_on_tailnet() || (cfg.allow_tailnet && !cfg.bind_addr.trim().is_empty())
}

/// The text to paste into the gateway configuration, for the server of `hub`; the token is a placeholder.
pub fn build_with(hub: &Hub, profile: &ZcProfile) -> String {
    let cfg = hub.config();
    let agent = Some(profile.agent.trim()).filter(|a| !a.is_empty()).unwrap_or(DEFAULT_AGENT);
    let mut text = String::from("# Made by AiMediaExplorer. Add it to the gateway configuration (zeroclaw config) and fill in the token.\n");
    if !reachable_from_outside(profile, &cfg) {
        text.push_str("# The address below is this computer only. A gateway on another machine needs the tools server exposed on the tailnet.\n");
    }
    text.push('\n');
    let port = match hub.state() {
        crate::ai::mcp::ServerState::Running { addr, .. } => addr.port(),
        _ => profile.tools_port,
    };
    text.push_str(&gateway_snippet_for(&url_for(profile, &cfg, port), &hub.registry, agent));
    text
}

/// The snippet for the application's server and `profile`.
pub fn build(profile: &ZcProfile) -> String {
    build_with(hub(), profile)
}

/// How the ping went.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ping {
    pub ok: bool,
    pub detail: String,
    pub millis: u128,
}

fn failed(started: Instant, detail: impl Into<String>) -> Ping {
    Ping { ok: false, detail: detail.into(), millis: started.elapsed().as_millis() }
}

/// The JSON-RPC answer in `body`, which is plain JSON or an event stream whose `data:` lines carry it.
fn rpc_answer(body: &str, id: u64) -> Option<Value> {
    let lines = body.lines().map(str::trim).filter_map(|l| l.strip_prefix("data:")).map(str::trim);
    let candidates: Vec<Value> = if body.trim_start().starts_with('{') { serde_json::from_str(body).ok().into_iter().collect() } else { lines.filter_map(|l| serde_json::from_str(l).ok()).collect() };
    candidates.into_iter().find(|v| v["id"] == json!(id))
}

async fn post(client: &reqwest::Client, url: &str, token: &str, session: Option<&str>, body: Value) -> Result<(Option<String>, String), String> {
    let mut req = client.post(url).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").bearer_auth(token).json(&body);
    if let Some(s) = session {
        req = req.header("mcp-session-id", s);
    }
    let response = req.send().await.map_err(|e| format!("The tools server did not answer: {}", e.to_string().replace(token, "***")))?;
    let status = response.status();
    let session = response.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()).map(str::to_string);
    let text = response.text().await.map_err(|e| e.to_string())?;
    match status.as_u16() {
        401 => Err("The tools server refused the token (401)".to_string()),
        403 => Err("The tools server refused the request (403): the host is not on its allow list".to_string()),
        429 => Err("The tools server is rate limiting requests (429)".to_string()),
        code if !(200..300).contains(&code) => Err(format!("The tools server answered HTTP {code}")),
        _ => Ok((session, text)),
    }
}

/// Calls `util_ping` on the MCP server at `url` with `token`, the way a gateway would.
pub async fn ping_server(url: &str, token: &str) -> Ping {
    let started = Instant::now();
    let client = match reqwest::Client::builder().timeout(REQUEST_TIMEOUT).no_proxy().redirect(reqwest::redirect::Policy::none()).build() {
        Ok(c) => c,
        Err(e) => return failed(started, e.to_string()),
    };
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "aimediaexplorer-self-test", "version": "1"}}});
    let (session, body) = match post(&client, url, token, None, init).await {
        Ok(v) => v,
        Err(e) => return failed(started, e),
    };
    let Some(session) = session else { return failed(started, "The tools server did not open a session") };
    if rpc_answer(&body, 1).is_none_or(|v| v.get("result").is_none()) {
        return failed(started, "The tools server did not accept the handshake");
    }
    let outcome = async {
        post(&client, url, token, Some(&session), json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await?;
        let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "util_ping", "arguments": {"message": "snippet test"}}});
        let (_, body) = post(&client, url, token, Some(&session), call).await?;
        let answer = rpc_answer(&body, 2).ok_or("util_ping gave no answer")?;
        if let Some(error) = answer.get("error") {
            return Err(format!("util_ping failed: {}", error["message"].as_str().unwrap_or("unknown error")));
        }
        let text = answer["result"]["content"][0]["text"].as_str().unwrap_or("").to_string();
        if text.starts_with("pong") { Ok(text) } else { Err(format!("util_ping answered something else: {text}")) }
    }
    .await;
    let _ = client.delete(url).bearer_auth(token).header("mcp-session-id", &session).send().await;
    match outcome {
        Ok(text) => Ping { ok: true, detail: format!("util_ping answered \"{text}\""), millis: started.elapsed().as_millis() },
        Err(e) => failed(started, e),
    }
}

/// Pings the application's tools server at its own address.
pub fn test(rt: &ZcRuntime) -> Pending<Ping> {
    rt.job(|_| async move {
        let started = Instant::now();
        let hub = hub();
        if !matches!(hub.state(), crate::ai::mcp::ServerState::Running { .. }) {
            return failed(started, "The tools server is not running. Start it first.");
        }
        match hub.token() {
            Ok(token) => ping_server(&hub.url(), &token).await,
            Err(e) => failed(started, e),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_address_follows_the_profile_and_the_server_settings() {
        let cfg = McpConfig::default();
        let mut p = ZcProfile { tools_port: 9999, ..Default::default() };
        assert_eq!(url_for(&p, &cfg, 9999), "http://127.0.0.1:9999/mcp");
        p.tools_bind = "100.116.134.128".into();
        assert_eq!(url_for(&p, &cfg, 9999), "http://100.116.134.128:9999/mcp");
        p.tools_bind = "localhost".into();
        assert_eq!(url_for(&p, &cfg, 9999), "http://127.0.0.1:9999/mcp");
        let exposed = McpConfig { allow_tailnet: true, bind_addr: "100.64.0.9".into(), ..Default::default() };
        assert_eq!(url_for(&p, &exposed, 9999), "http://100.64.0.9:9999/mcp", "an interface set in the server settings is used when the profile has none");
        let stale = McpConfig { allow_tailnet: false, bind_addr: "100.64.0.9".into(), ..Default::default() };
        assert_eq!(url_for(&p, &stale, 9999), "http://127.0.0.1:9999/mcp", "a bind address is ignored while the tailnet switch is off");
        assert_eq!(url_for(&p, &cfg, 9892), "http://127.0.0.1:9892/mcp", "the port the server really listens on is the one named");
        assert!(reachable_from_outside(&ZcProfile { tools_bind: "100.1.1.1".into(), ..Default::default() }, &cfg));
        assert!(reachable_from_outside(&p, &exposed) && !reachable_from_outside(&p, &cfg));
    }

    #[test]
    fn the_snippet_names_the_agent_the_address_and_a_token_placeholder() {
        let p = ZcProfile { agent: "photos".into(), tools_bind: "100.116.134.128".into(), tools_port: 9014, ..Default::default() };
        let text = build(&p);
        assert!(text.contains("url = \"http://100.116.134.128:9014/mcp\""), "{text}");
        assert!(text.contains("[agents.photos]") && text.contains("[risk_profiles.photos]") && text.contains("risk_profile = \"photos\""));
        assert!(text.contains("pinned_resources = [\"library://guide\"]"));
        assert!(text.contains("Bearer <paste the token"));
        assert!(!text.contains("this computer only"), "an exposed server needs no warning");
        let local = build(&ZcProfile::default());
        assert!(local.contains("this computer only") && local.contains("[agents.media]"), "{local}");
        let odd = build(&ZcProfile { agent: "my agent".into(), ..Default::default() });
        assert!(odd.contains("[agents.\"my agent\"]") && odd.contains("risk_profile = \"my agent\""), "{odd}");
        for tool in crate::ai::mcp::server::ASK_FIRST {
            assert!(text.contains(&format!("always_ask = [")) && text.contains(&format!("\"aimedia__{tool}\"")));
        }
        if let Ok(token) = hub().token() {
            assert!(!text.contains(&token) && !local.contains(&token), "the real token is never part of the snippet");
        }
    }

    #[test]
    fn answers_are_read_from_plain_json_and_from_event_streams() {
        assert_eq!(rpc_answer(r#"{"jsonrpc":"2.0","id":2,"result":{"a":1}}"#, 2).unwrap()["result"]["a"], 1);
        let stream = "id: 0\ndata:\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"b\":2}}\n\n";
        assert_eq!(rpc_answer(stream, 2).unwrap()["result"]["b"], 2);
        assert!(rpc_answer(stream, 3).is_none());
        assert!(rpc_answer("", 1).is_none());
        assert!(rpc_answer("data: not json", 1).is_none());
    }
}
