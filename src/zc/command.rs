//! The `zc` application command: gateway setup, health, the chat socket and one-shot calls, for scripts and the Devtools window.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zeroclaw_client::CompleteRequest;

use super::profile::{BackendPref, ZcProfile};
use super::{import, runtime, save_profile, secrets, tools};
use crate::command::{AppCommand, CommandError, CommandResult, ToastLevel};

/// What a `zc` command does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZcOp {
    /// Report the profile, health, the chat socket, usage and the tools server as JSON.
    Status,
    /// Probe the gateway now.
    Check,
    /// Open the chat socket for `session` (a new id when none is given).
    Connect,
    Disconnect,
    /// Drop the socket and connect again at once.
    Reconnect,
    /// One model call through `/api/complete` with `prompt`, `hint`, `system` and `temperature`.
    Complete,
    /// Change the profile: `host`, `agent`, `name`, `hint`, `backend`, `allow_images`, `tools`, and a token from `token_file`.
    Configure,
    /// Exchange the pairing `code` for a token at `host`.
    Pair,
    /// Take the gateway from a ZC Codex config at `path` (default `~/.config/zc-codex/config.json`).
    Import,
    /// Send `text` as the person would from the assistant.
    Send,
    /// Stop the running turn.
    Stop,
    /// Show the assistant in the right rail.
    Open,
    /// Ask the agent to call the tools server's ping tool.
    PingTools,
    /// Delete the stored token.
    ForgetToken,
}

impl ZcOp {
    pub const ALL: [ZcOp; 14] = [
        Self::Status,
        Self::Check,
        Self::Connect,
        Self::Disconnect,
        Self::Reconnect,
        Self::Complete,
        Self::Configure,
        Self::Pair,
        Self::Import,
        Self::Send,
        Self::Stop,
        Self::Open,
        Self::PingTools,
        Self::ForgetToken,
    ];
}

/// The fields a `zc` command can carry; each op reads the ones it needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ZcArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// File holding the token; the token itself is never part of a command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_images: Option<bool>,
    /// Serve the MCP tools to the gateway.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
}

/// The prompt that asks the agent to prove the tools server is reachable.
pub const PING_PROMPT: &str = "Call the tool aimedia__util_ping with the message \"ping from the Devtools window\" and tell me exactly what it returned. If the tool is not available, say so.";

fn need<'a>(value: &'a Option<String>, what: &str) -> Result<&'a str, CommandError> {
    value.as_deref().map(str::trim).filter(|v| !v.is_empty()).ok_or_else(|| CommandError::invalid(format!("this command needs {what}")))
}

fn toast(kind: ToastLevel, text: String) {
    crate::command::send(AppCommand::Toast { kind, text });
}

/// The JSON `status` reports.
pub fn status_json() -> Value {
    let rt = runtime();
    let profile = rt.profile();
    let link = rt.link();
    let usage = rt.usage();
    let tools = tools::status();
    json!({
        "profile": {
            "name": profile.name, "host": profile.host_label(), "agent": profile.agent, "backend": profile.backend,
            "hint": profile.hint(), "allow_images": profile.allow_images, "tools_enabled": profile.tools_enabled,
            "tools_bind": profile.tools_bind, "tools_port": profile.tools_port, "configured": profile.is_configured(),
        },
        "token": { "stored": rt.has_token(), "file": rt.token_path().map(|p| p.display().to_string()) },
        "health": rt.health().to_json(),
        "link": { "state": link.state.label(), "session": link.session, "epoch": link.epoch, "detail": link.detail },
        "usage": { "turns": usage.turns, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd },
        "frames": rt.with_frames(|f| json!({ "kept": f.len(), "in": f.total_in, "out": f.total_out, "recording": f.recording().map(|r| r.path().display().to_string()) })),
        "tools": { "state": tools.state, "url": tools.url, "clients": tools.clients },
    })
}

/// Runs one `zc` command; work that needs the network finishes in the background and reports through a toast and the Devtools notes.
pub fn run(op: ZcOp, args: ZcArgs) -> CommandResult {
    let rt = runtime();
    match op {
        ZcOp::Status => Ok(status_json().to_string()),
        ZcOp::Check => {
            if rt.check() { Ok("checking the gateway".to_string()) } else { Ok("a check is already running".to_string()) }
        }
        ZcOp::Connect => {
            if !rt.profile().is_configured() {
                return Err(CommandError::invalid("the gateway is not set up: configure a host and an agent first"));
            }
            let session = args.session.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| format!("zc-{}", surrealdb_types::Uuid::new_v4()));
            rt.connect(&session, args.name.as_deref());
            Ok(format!("connecting session {session}"))
        }
        ZcOp::Disconnect => {
            rt.disconnect();
            Ok("socket closed".to_string())
        }
        ZcOp::Reconnect => if rt.reconnect() { Ok("reconnecting".to_string()) } else { Err(CommandError::invalid("no chat socket is open")) },
        ZcOp::Complete => {
            let prompt = need(&args.prompt, "a prompt")?.to_string();
            let mut request = CompleteRequest::new(prompt).with_hint(args.hint.clone().unwrap_or_else(|| rt.profile().hint().to_string()));
            if let Some(system) = args.system.clone() {
                request = request.with_system(system);
            }
            if let Some(t) = args.temperature {
                request = request.with_temperature(t);
            }
            let mut job = rt.complete(request);
            let sh = rt.shared().clone();
            tokio::spawn(async move {
                let result = loop {
                    if let Some(r) = job.take() {
                        break r;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                };
                match result {
                    Ok(reply) => {
                        sh.note(format!("complete [{}]: {}", reply.model, reply.response));
                        toast(ToastLevel::Success, format!("ZeroClaw: {}", crate::ai::mcp::activity::clip(&reply.response, 100)));
                    }
                    Err(e) => {
                        sh.note(format!("complete failed: {e}"));
                        toast(ToastLevel::Error, format!("ZeroClaw complete failed: {}", crate::ai::mcp::activity::clip(&e, 100)));
                    }
                }
            });
            Ok("calling /api/complete".to_string())
        }
        ZcOp::Configure => configure(&args),
        ZcOp::Pair => {
            let host = need(&args.host, "a host")?.to_string();
            let code = need(&args.code, "a pairing code")?.to_string();
            let agent = args.agent.clone();
            let mut job = rt.pair(&host, &code);
            let sh = rt.shared().clone();
            tokio::spawn(async move {
                let result = loop {
                    if let Some(r) = job.take() {
                        break r;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                };
                match result {
                    Ok(()) => {
                        let mut profile = runtime().profile();
                        profile.host = host;
                        if let Some(agent) = agent {
                            profile.agent = agent;
                        }
                        save_profile(profile);
                        sh.note("paired; the token is stored");
                        toast(ToastLevel::Success, "ZeroClaw: paired".to_string());
                    }
                    Err(e) => {
                        sh.note(format!("pairing failed: {e}"));
                        toast(ToastLevel::Error, format!("ZeroClaw pairing failed: {}", crate::ai::mcp::activity::clip(&e, 100)));
                    }
                }
            });
            Ok("pairing".to_string())
        }
        ZcOp::Import => {
            let path = match args.path.as_deref().filter(|p| !p.trim().is_empty()) {
                Some(p) => std::path::PathBuf::from(p),
                None => import::default_path().ok_or_else(|| CommandError::not_found("this platform has no config folder"))?,
            };
            let machine = import::import_from(&path).map_err(CommandError::not_found)?;
            let mut profile = rt.profile();
            profile.name = machine.name.clone();
            profile.host = machine.host.clone();
            profile.agent = machine.agent.clone();
            let summary = format!("imported {} ({}, agent {})", machine.name, machine.host, machine.agent);
            rt.save_token(machine.token()).map_err(CommandError::failed)?;
            save_profile(profile);
            Ok(summary)
        }
        ZcOp::Send => {
            let text = need(&args.text, "text")?.to_string();
            crate::ui::assistant::api::submit(text);
            Ok("queued for the assistant".to_string())
        }
        ZcOp::Stop => if rt.stop() { Ok("stopping the turn".to_string()) } else { Err(CommandError::invalid("no chat socket is open")) },
        ZcOp::Open => {
            crate::ui::assistant::api::request_open();
            Ok("opening the assistant".to_string())
        }
        ZcOp::PingTools => {
            crate::ui::assistant::api::submit(PING_PROMPT.to_string());
            Ok("asked the agent to ping the tools".to_string())
        }
        ZcOp::ForgetToken => {
            rt.clear_token().map_err(CommandError::failed)?;
            Ok("the token was deleted".to_string())
        }
    }
}

fn configure(args: &ZcArgs) -> CommandResult {
    let rt = runtime();
    let mut profile: ZcProfile = rt.profile();
    let mut changed = Vec::new();
    let mut set = |field: &mut String, value: &Option<String>, name: &'static str| {
        if let Some(v) = value {
            *field = v.trim().to_string();
            changed.push(name);
        }
    };
    set(&mut profile.host, &args.host, "host");
    set(&mut profile.agent, &args.agent, "agent");
    set(&mut profile.name, &args.name, "name");
    set(&mut profile.complete_hint, &args.hint, "hint");
    if let Some(b) = &args.backend {
        profile.backend = BackendPref::parse(b).as_str().to_string();
        changed.push("backend");
    }
    if let Some(v) = args.allow_images {
        profile.allow_images = v;
        changed.push("allow_images");
    }
    if let Some(v) = args.tools {
        profile.tools_enabled = v;
        changed.push("tools");
    }
    if let Some(file) = &args.token_file {
        let token = secrets::read_token_file(std::path::Path::new(file)).map_err(CommandError::invalid)?;
        rt.save_token(&token).map_err(CommandError::failed)?;
        changed.push("token");
    }
    if changed.is_empty() {
        return Err(CommandError::invalid("nothing to configure"));
    }
    if let Err(e) = profile.endpoint() {
        if !profile.host.trim().is_empty() {
            return Err(CommandError::invalid(e));
        }
    }
    let tools_changed = args.tools.is_some();
    save_profile(profile.clone());
    if tools_changed {
        if profile.tools_enabled {
            tools::ensure_running(&profile);
        } else {
            let _ = tools::stop();
        }
    }
    Ok(format!("configured: {}", changed.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> AppCommand {
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{json}: {e}"))
    }

    #[test]
    fn the_json_form_is_snake_case_with_flat_arguments() {
        let cmd = parse(r#"{"cmd":"zc","op":"configure","host":"127.0.0.1:42617","agent":"media","allow_images":true}"#);
        match cmd {
            AppCommand::Zc { op: ZcOp::Configure, args } => {
                assert_eq!((args.host.as_deref(), args.agent.as_deref(), args.allow_images), (Some("127.0.0.1:42617"), Some("media"), Some(true)));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(parse(r#"{"cmd":"zc","op":"ping_tools"}"#), AppCommand::Zc { op: ZcOp::PingTools, .. }));
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"zc","op":"explode"}"#).is_err());
        assert_eq!(parse(r#"{"cmd":"zc","op":"status"}"#).name(), "zc");
    }

    #[test]
    fn a_command_never_carries_a_token_value() {
        let args = ZcArgs { token_file: Some("/tmp/t".into()), ..Default::default() };
        let json = serde_json::to_string(&AppCommand::Zc { op: ZcOp::Configure, args }).unwrap();
        assert!(!json.contains("\"token\""), "{json}");
        let parsed = parse(r#"{"cmd":"zc","op":"configure","token":"abc","host":"h"}"#);
        assert!(!format!("{parsed:?}").contains("abc"), "an unknown token field is ignored, never kept");
    }

    #[test]
    fn the_command_docs_describe_every_op() {
        let docs = include_str!("../../docs/commands.md");
        for op in ZcOp::ALL {
            let name = serde_json::to_value(op).unwrap();
            let name = name.as_str().unwrap();
            assert!(docs.contains(&format!("`{name}`")), "docs/commands.md does not describe zc op {name}");
            let back: ZcOp = serde_json::from_value(Value::String(name.to_string())).unwrap();
            assert_eq!(back, op);
        }
        assert!(docs.contains("\"cmd\": \"zc\""));
    }

    #[test]
    fn commands_without_their_arguments_are_invalid() {
        assert_eq!(run(ZcOp::Complete, ZcArgs::default()).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(ZcOp::Pair, ZcArgs { host: Some("h".into()), ..Default::default() }).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(ZcOp::Send, ZcArgs::default()).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(ZcOp::Configure, ZcArgs::default()).unwrap_err().kind, crate::command::ErrorKind::Invalid);
    }

    #[test]
    fn status_is_json_with_no_token_in_it() {
        let text = run(ZcOp::Status, ZcArgs::default()).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert!(v["health"]["state"].is_string() && v["profile"]["tools_port"].is_number());
        assert!(v["token"]["stored"].is_boolean());
    }

    #[test]
    fn the_ping_prompt_names_the_tool_the_catalog_exposes() {
        assert!(PING_PROMPT.contains("aimedia__util_ping"));
    }
}
