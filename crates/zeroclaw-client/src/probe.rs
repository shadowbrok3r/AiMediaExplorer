//! `zcprobe` command line tool; exit codes 0 ok, 1 failure, 2 usage, 3 credentials refused, 4 unreachable.

use std::io::Write;
use std::time::Duration;

use serde_json::json;

use crate::attachments::InlineImage;
use crate::chat::{ChatSession, ConnectOptions};
use crate::config::{Config, Token, redact_token};
use crate::error::{Error, ExitClass};
use crate::event::{DisconnectReason, Event};
use crate::history::HistoryKind;
use crate::replay::line as replay_line;
use crate::rest::{AuthState, RestClient};
use crate::types::{CompleteRequest, group_sessions};

pub const USAGE: &str = "\
zcprobe - probe a ZeroClaw gateway

USAGE
  zcprobe [GLOBAL FLAGS] <COMMAND> [ARGS]

COMMANDS
  health                    reachability, token check and agent list
  sessions                  list sessions (--agent filters, --all keeps empty ones)
  history <session-id>      print the newest messages (--limit N, --tool-context)
  complete <prompt>         one-shot completion (--hint, --system, --temperature)
  chat <prompt>             run one chat turn and print the streamed events
                            (--session ID, --auto-approve, --image PATH, --thinking)
  frames [<prompt>]         dump raw frames as JSON lines (--session ID, --seconds N, --auto-approve)

GLOBAL FLAGS
  --url URL                 gateway address (env ZC_URL); host, host:port or http(s)/ws(s) URL
  --token TOKEN             bearer token (env ZC_TOKEN); prefer --token-file or the env
  --token-file PATH         read the token from a file
  --agent ALIAS             agent alias (env ZC_AGENT); required by chat and frames
  --timeout SECONDS         overall limit per command (default 30; complete 90; chat and frames 120)
  --json                    machine-readable output
  -h, --help                show this help

EXIT CODES
  0 ok   1 failure   2 usage   3 credentials refused   4 gateway unreachable
";

#[derive(Debug, Clone, PartialEq)]
pub struct Globals {
    pub url: Option<String>,
    pub token: Option<String>,
    pub agent: Option<String>,
    pub timeout: Option<Duration>,
    pub json: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help,
    Health,
    Sessions { all: bool },
    History { id: String, limit: usize, tool_context: bool },
    Complete { prompt: String, hint: String, system: Option<String>, temperature: Option<f64> },
    Chat { prompt: String, session: Option<String>, auto_approve: bool, images: Vec<String>, thinking: bool },
    Frames { prompt: Option<String>, session: Option<String>, seconds: u64, auto_approve: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub globals: Globals,
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn usage(msg: impl Into<String>) -> UsageError {
    UsageError(msg.into())
}

/// Parses `args` (without the program name). `env` supplies `ZC_URL`, `ZC_TOKEN` and `ZC_AGENT`.
pub fn parse_args(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<Args, UsageError> {
    let mut globals = Globals { url: None, token: None, agent: None, timeout: None, json: false };
    let mut positionals: Vec<String> = Vec::new();
    let mut opts: Vec<(String, Option<String>)> = Vec::new();
    let mut help = false;

    const WITH_VALUE: &[&str] = &[
        "url", "token", "token-file", "agent", "timeout", "limit", "hint", "system", "temperature", "session", "image", "seconds",
    ];
    const SWITCHES: &[&str] = &["json", "all", "tool-context", "auto-approve", "thinking", "help"];

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if arg == "-h" {
            help = true;
        } else if let Some(flag) = arg.strip_prefix("--") {
            if flag.is_empty() {
                positionals.extend(args[i..].iter().cloned());
                break;
            }
            let (name, inline) = match flag.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (flag, None),
            };
            if SWITCHES.contains(&name) {
                if inline.is_some() {
                    return Err(usage(format!("--{name} takes no value")));
                }
                if name == "help" {
                    help = true;
                } else {
                    opts.push((name.to_string(), None));
                }
            } else if WITH_VALUE.contains(&name) {
                let value = match inline {
                    Some(v) => v,
                    None => {
                        let v = args.get(i).ok_or_else(|| usage(format!("--{name} needs a value")))?;
                        i += 1;
                        v.clone()
                    }
                };
                opts.push((name.to_string(), Some(value)));
            } else {
                return Err(usage(format!("unknown flag --{name}")));
            }
        } else {
            positionals.push(arg.clone());
        }
    }

    let value = |name: &str| opts.iter().rev().find(|(n, _)| n == name).and_then(|(_, v)| v.clone());
    let has = |name: &str| opts.iter().any(|(n, _)| n == name);
    let number = |name: &str| -> Result<Option<u64>, UsageError> {
        value(name)
            .map(|v| v.parse::<u64>().map_err(|_| usage(format!("--{name} needs a whole number, got `{v}`"))))
            .transpose()
    };

    globals.json = has("json");
    globals.url = value("url").or_else(|| env("ZC_URL")).filter(|s| !s.trim().is_empty());
    globals.agent = value("agent").or_else(|| env("ZC_AGENT")).filter(|s| !s.trim().is_empty());
    globals.token = match (value("token"), value("token-file")) {
        (Some(t), _) => Some(t),
        (None, Some(path)) => Some(
            std::fs::read_to_string(&path)
                .map_err(|e| usage(format!("cannot read token file {path}: {e}")))?
                .trim()
                .to_string(),
        ),
        (None, None) => env("ZC_TOKEN"),
    }
    .filter(|s| !s.trim().is_empty());
    globals.timeout = number("timeout")?.map(Duration::from_secs);

    if help || positionals.first().is_some_and(|c| c == "help") {
        return Ok(Args { globals, command: Command::Help });
    }
    let Some(name) = positionals.first().cloned() else {
        return Err(usage("missing command; try --help"));
    };
    let rest = &positionals[1..];
    let joined = rest.join(" ");
    let command = match name.as_str() {
        "health" => Command::Health,
        "sessions" => Command::Sessions { all: has("all") },
        "history" => {
            let id = rest.first().cloned().ok_or_else(|| usage("history needs a session id"))?;
            if rest.len() > 1 {
                return Err(usage("history takes one session id"));
            }
            Command::History { id, limit: number("limit")?.unwrap_or(20).max(1) as usize, tool_context: has("tool-context") }
        }
        "complete" => {
            if joined.trim().is_empty() {
                return Err(usage("complete needs a prompt"));
            }
            let temperature = value("temperature")
                .map(|v| v.parse::<f64>().map_err(|_| usage(format!("--temperature needs a number, got `{v}`"))))
                .transpose()?;
            Command::Complete {
                prompt: joined,
                hint: value("hint").unwrap_or_else(|| crate::types::DEFAULT_COMPLETE_HINT.into()),
                system: value("system"),
                temperature,
            }
        }
        "chat" => {
            if joined.trim().is_empty() {
                return Err(usage("chat needs a prompt"));
            }
            Command::Chat {
                prompt: joined,
                session: value("session"),
                auto_approve: has("auto-approve"),
                images: opts.iter().filter(|(n, _)| n == "image").filter_map(|(_, v)| v.clone()).collect(),
                thinking: has("thinking"),
            }
        }
        "frames" => Command::Frames {
            prompt: (!joined.trim().is_empty()).then_some(joined),
            session: value("session"),
            seconds: number("seconds")?.unwrap_or(5),
            auto_approve: has("auto-approve"),
        },
        other => return Err(usage(format!("unknown command `{other}`; try --help"))),
    };
    Ok(Args { globals, command })
}

/// Output sinks that scrub the token from everything written.
struct Out<'a> {
    stdout: &'a mut dyn Write,
    stderr: &'a mut dyn Write,
    token: Option<Token>,
}

impl Out<'_> {
    fn scrub(&self, text: &str) -> String {
        redact_token(text, self.token.as_ref())
    }

    fn out(&mut self, text: &str) {
        let text = self.scrub(text);
        let _ = self.stdout.write_all(text.as_bytes());
        let _ = self.stdout.flush();
    }

    fn outln(&mut self, text: &str) {
        self.out(&format!("{text}\n"));
    }

    fn err(&mut self, text: &str) {
        let text = self.scrub(text);
        let _ = self.stderr.write_all(text.as_bytes());
        let _ = self.stderr.flush();
    }

    fn errln(&mut self, text: &str) {
        self.err(&format!("{text}\n"));
    }
}

/// Runs the tool and returns the process exit code.
pub async fn run(args: &[String], env: &dyn Fn(&str) -> Option<String>, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let parsed = match parse_args(args, env) {
        Ok(a) => a,
        Err(e) => {
            let _ = writeln!(stderr, "zcprobe: {e}");
            return ExitClass::Usage.code();
        }
    };
    let mut out = Out { stdout, stderr, token: parsed.globals.token.as_deref().and_then(Token::new) };
    if parsed.command == Command::Help {
        out.out(USAGE);
        return 0;
    }
    match execute(&parsed, &mut out).await {
        Ok(code) => code,
        Err(e) => {
            out.errln(&format!("zcprobe: {e}"));
            e.exit_class().code()
        }
    }
}

fn config_for(g: &Globals, needs_agent: bool) -> Result<Config, Error> {
    let url = g.url.clone().ok_or_else(|| Error::Config("no gateway address: pass --url or set ZC_URL".into()))?;
    let mut cfg = Config::new(url, g.token.clone().unwrap_or_default()).with_device_name("zcprobe");
    if needs_agent {
        let agent = g.agent.clone().ok_or_else(|| Error::Config("no agent: pass --agent or set ZC_AGENT".into()))?;
        cfg.agent = agent;
    } else if let Some(agent) = &g.agent {
        cfg.agent = agent.clone();
    }
    cfg.endpoint()?;
    Ok(cfg)
}

async fn execute(args: &Args, out: &mut Out<'_>) -> Result<u8, Error> {
    let g = &args.globals;
    match &args.command {
        Command::Help => Ok(0),
        Command::Health => {
            let cfg = config_for(g, false)?;
            let rest = RestClient::new(&cfg)?;
            let probe = with_timeout(g.timeout.unwrap_or(Duration::from_secs(30)), rest.probe()).await?;
            let auth = match probe.auth {
                AuthState::NotChecked => "not checked (no token)".to_string(),
                AuthState::Accepted => "accepted".to_string(),
                AuthState::Rejected(code) => format!("rejected (HTTP {code})"),
            };
            let agent_known = cfg.agent_name();
            let known = (!agent_known.is_empty() && probe.auth == AuthState::Accepted)
                .then(|| probe.agents.iter().any(|a| a == agent_known));
            if g.json {
                out.outln(
                    &json!({
                        "gateway": cfg.endpoint()?.host_port(), "status": probe.health.status,
                        "paired": probe.health.paired, "require_pairing": probe.health.require_pairing,
                        "auth": auth, "agents": probe.agents, "agent_known": known,
                    })
                    .to_string(),
                );
            } else {
                out.outln(&format!("gateway: {}", cfg.endpoint()?.host_port()));
                out.outln(&format!("status: {}", probe.health.status));
                out.outln(&format!("paired: {}", probe.health.paired));
                out.outln(&format!("require_pairing: {}", probe.health.require_pairing));
                out.outln(&format!("auth: {auth}"));
                if probe.auth == AuthState::Accepted {
                    out.outln(&format!("agents: {}", probe.agents.join(", ")));
                }
                if let Some(known) = known {
                    out.outln(&format!("agent {agent_known}: {}", if known { "known" } else { "unknown" }));
                }
            }
            Ok(match (probe.auth, known) {
                (AuthState::Rejected(_), _) => ExitClass::Auth.code(),
                (_, Some(false)) => ExitClass::Failure.code(),
                _ => 0,
            })
        }
        Command::Sessions { all } => {
            let cfg = config_for(g, false)?;
            let rest = RestClient::new(&cfg)?;
            let sessions = with_timeout(g.timeout.unwrap_or(Duration::from_secs(30)), rest.sessions()).await?;
            let agent = (!cfg.agent_name().is_empty()).then(|| cfg.agent_name().to_string());
            let rows = if *all {
                sessions
                    .iter()
                    .filter(|s| agent.as_deref().is_none_or(|a| s.agent_alias.as_deref() == Some(a)))
                    .map(|s| (s.session_id.clone(), s.name.clone(), s.message_count, s.last_activity.clone()))
                    .collect::<Vec<_>>()
            } else {
                group_sessions(agent.as_deref(), &sessions)
                    .into_iter()
                    .map(|r| (r.id, r.name, r.message_count, r.last_activity))
                    .collect()
            };
            for (id, name, count, last) in rows {
                if g.json {
                    out.outln(&json!({"session_id": id, "name": name, "message_count": count, "last_activity": last}).to_string());
                } else {
                    out.outln(&format!(
                        "{id}\t{count}\t{}\t{}",
                        last.as_deref().unwrap_or("-"),
                        name.as_deref().unwrap_or("-")
                    ));
                }
            }
            Ok(0)
        }
        Command::History { id, limit, tool_context } => {
            let cfg = config_for(g, false)?;
            let rest = RestClient::new(&cfg)?;
            let mut pager =
                with_timeout(g.timeout.unwrap_or(Duration::from_secs(30)), rest.history(id, *limit, *tool_context)).await?;
            for item in pager.next_page() {
                let kind = match item.kind {
                    HistoryKind::User => "user",
                    HistoryKind::Assistant => "assistant",
                    HistoryKind::Reasoning => "reasoning",
                    HistoryKind::ToolCall => "tool_call",
                    HistoryKind::ToolResult => "tool_result",
                    HistoryKind::Other => "other",
                };
                if g.json {
                    out.outln(
                        &json!({
                            "id": item.id, "kind": kind, "text": item.text, "name": item.name,
                            "args": item.args, "created_at": item.created_at, "images": item.images.len(),
                        })
                        .to_string(),
                    );
                } else {
                    let mut text = item.text.replace('\n', " ");
                    if text.chars().count() > 300 {
                        text = text.chars().take(300).collect::<String>() + "...";
                    }
                    let name = item.name.map(|n| format!(" {n}")).unwrap_or_default();
                    out.outln(&format!("{} {kind}{name}: {text}", item.id));
                }
            }
            if pager.has_more() {
                out.errln(&format!("({} older messages not shown; raise --limit)", pager.remaining()));
            }
            Ok(0)
        }
        Command::Complete { prompt, hint, system, temperature } => {
            let cfg = config_for(g, false)?;
            let rest = RestClient::new(&cfg)?;
            let mut request = CompleteRequest::new(prompt.clone()).with_hint(hint.clone());
            request.system = system.clone();
            request.temperature = *temperature;
            let reply = with_timeout(g.timeout.unwrap_or(Duration::from_secs(90)), rest.complete(&request)).await?;
            if g.json {
                out.outln(&json!({"response": reply.response, "model": reply.model, "hint": reply.hint}).to_string());
            } else {
                out.outln(&reply.response);
            }
            Ok(0)
        }
        Command::Chat { prompt, session, auto_approve, images, thinking } => {
            let cfg = config_for(g, true)?;
            let images = images.iter().map(InlineImage::read).collect::<Result<Vec<_>, _>>()?;
            let opts = ConnectOptions { session_id: session.clone(), ..ConnectOptions::default() };
            let limit = g.timeout.unwrap_or(Duration::from_secs(120));
            let chat = ChatSession::connect(cfg, opts);
            let (session, mut events) = with_timeout(Duration::from_secs(30), chat).await?;
            session.send_user_message(prompt, &images).await?;
            let code = tokio::time::timeout(limit, drain_chat(&session, &mut events, out, g.json, *auto_approve, *thinking))
                .await
                .unwrap_or_else(|_| {
                    out.errln("zcprobe: timed out waiting for the turn to finish");
                    1
                });
            session.close().await;
            Ok(code)
        }
        Command::Frames { prompt, session, seconds, auto_approve } => {
            let cfg = config_for(g, true)?;
            let opts = ConnectOptions { session_id: session.clone(), raw_tap: true, ..ConnectOptions::default() };
            let (chat, mut events) = with_timeout(Duration::from_secs(30), ChatSession::connect(cfg, opts)).await?;
            if let Some(p) = prompt {
                chat.send_text(p).await?;
            }
            let limit = match prompt {
                Some(_) => g.timeout.unwrap_or(Duration::from_secs(120)),
                None => Duration::from_secs(*seconds),
            };
            let finished = tokio::time::timeout(limit, async {
                while let Some(event) = events.next().await {
                    match &event {
                        Event::Raw(raw) => {
                            out.outln(&replay_line(raw.t_ms, raw.dir, &raw.frame));
                        }
                        Event::ApprovalRequest(a) => {
                            let decision = if *auto_approve { chat.approve(&a.request_id).await } else { chat.deny(&a.request_id).await };
                            if let Err(e) = decision {
                                out.errln(&format!("zcprobe: could not answer approval: {e}"));
                            }
                        }
                        Event::Done { .. } | Event::Aborted { .. } if prompt.is_some() => return 0,
                        Event::Error { ends_turn: true, .. } if prompt.is_some() => return 1,
                        Event::Disconnected { will_retry: false, reason } => {
                            out.errln(&format!("zcprobe: disconnected: {reason}"));
                            return ExitClass::Unreachable.code();
                        }
                        _ => {}
                    }
                }
                0
            })
            .await;
            chat.close().await;
            Ok(match finished {
                Ok(code) => code,
                Err(_) if prompt.is_some() => {
                    out.errln("zcprobe: timed out waiting for the turn to finish");
                    1
                }
                Err(_) => 0,
            })
        }
    }
}

async fn with_timeout<T>(limit: Duration, fut: impl std::future::Future<Output = Result<T, Error>>) -> Result<T, Error> {
    tokio::time::timeout(limit, fut).await.map_err(|_| Error::Timeout("command"))?
}

/// Prints one turn's events until it ends; returns the exit code.
async fn drain_chat(
    session: &ChatSession,
    events: &mut crate::chat::EventStream,
    out: &mut Out<'_>,
    json_mode: bool,
    auto_approve: bool,
    thinking: bool,
) -> u8 {
    let mut in_thinking = false;
    let mut wrote_text = false;
    while let Some(event) = events.next().await {
        if json_mode {
            out.outln(&event.to_json().to_string());
        }
        match &event {
            Event::Token { text, .. } => {
                if !json_mode {
                    if in_thinking {
                        out.errln("");
                        in_thinking = false;
                    }
                    out.out(text);
                    wrote_text = true;
                }
            }
            Event::Thinking { text, .. } => {
                if !json_mode && thinking {
                    if !in_thinking {
                        out.err("[thinking] ");
                        in_thinking = true;
                    }
                    out.err(text);
                }
            }
            Event::ToolCall { name, args, .. } if !json_mode => {
                out.errln(&format!("[tool_call] {name} {}", short(&args.to_string(), 200)));
            }
            Event::ToolResult { name, output, .. } if !json_mode => {
                out.errln(&format!("[tool_result] {name} {}", short(output, 200)));
            }
            Event::ApprovalRequest(a) => {
                let answer = if auto_approve { session.approve(&a.request_id).await } else { session.deny(&a.request_id).await };
                if !json_mode {
                    out.errln(&format!(
                        "[approval] {} {} -> {}",
                        a.tool,
                        short(&a.summary, 120),
                        if auto_approve { "approved" } else { "denied (pass --auto-approve to allow)" }
                    ));
                }
                if let Err(e) = answer {
                    out.errln(&format!("zcprobe: could not answer approval: {e}"));
                }
            }
            Event::Error { code, message, ends_turn, .. } => {
                if !json_mode {
                    out.errln(&format!("[error] {} {message}", code.as_deref().unwrap_or("")));
                }
                if *ends_turn {
                    return 1;
                }
            }
            Event::Done { usage, .. } => {
                if !json_mode {
                    if wrote_text {
                        out.outln("");
                    }
                    let ctx = usage.context_fraction().map(|f| format!(" ctx={:.0}%", f * 100.0)).unwrap_or_default();
                    out.errln(&format!(
                        "[done] in={} out={}{ctx} model={}",
                        usage.input_tokens.map_or("-".into(), |n| n.to_string()),
                        usage.output_tokens.map_or("-".into(), |n| n.to_string()),
                        usage.model.as_deref().unwrap_or("-")
                    ));
                }
                return 0;
            }
            Event::Aborted { .. } => {
                if !json_mode {
                    out.errln("[aborted]");
                }
                return 1;
            }
            Event::TurnLost { .. } => return 1,
            Event::Disconnected { will_retry: false, reason } => {
                if !json_mode {
                    out.errln(&format!("[disconnected] {reason}"));
                }
                return match reason {
                    DisconnectReason::Fatal(text) if text.contains("401") || text.contains("unauthorized") => ExitClass::Auth.code(),
                    _ => ExitClass::Unreachable.code(),
                };
            }
            _ => {}
        }
    }
    ExitClass::Unreachable.code()
}

fn short(text: &str, max: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= max { flat } else { flat.chars().take(max).collect::<String>() + "..." }
}

/// Entry point of the `zcprobe` binary.
pub fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("zcprobe: cannot start the async runtime: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let env = |name: &str| std::env::var(name).ok();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let code = runtime.block_on(run(&args, &env, &mut stdout, &mut stderr));
    std::process::ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn parse(s: &str) -> Result<Args, UsageError> {
        parse_args(&args(s), &no_env)
    }

    #[test]
    fn health_with_flags_before_and_after_the_command() {
        let a = parse("--url host:1 health --token t").unwrap();
        assert_eq!(a.command, Command::Health);
        assert_eq!(a.globals.url.as_deref(), Some("host:1"));
        assert_eq!(a.globals.token.as_deref(), Some("t"));
        let b = parse("health --url=h --agent=x").unwrap();
        assert_eq!(b.globals.url.as_deref(), Some("h"));
        assert_eq!(b.globals.agent.as_deref(), Some("x"));
    }

    #[test]
    fn environment_fills_in_missing_flags_and_flags_win() {
        let env = |k: &str| match k {
            "ZC_URL" => Some("env-host".to_string()),
            "ZC_TOKEN" => Some("env-token".to_string()),
            "ZC_AGENT" => Some("env-agent".to_string()),
            _ => None,
        };
        let a = parse_args(&args("health"), &env).unwrap();
        assert_eq!(a.globals.url.as_deref(), Some("env-host"));
        assert_eq!(a.globals.token.as_deref(), Some("env-token"));
        assert_eq!(a.globals.agent.as_deref(), Some("env-agent"));
        let b = parse_args(&args("health --url flag-host --token flag-token"), &env).unwrap();
        assert_eq!(b.globals.url.as_deref(), Some("flag-host"));
        assert_eq!(b.globals.token.as_deref(), Some("flag-token"));
    }

    #[test]
    fn blank_env_values_count_as_unset() {
        let env = |k: &str| (k == "ZC_URL").then(|| "  ".to_string());
        assert_eq!(parse_args(&args("health"), &env).unwrap().globals.url, None);
    }

    #[test]
    fn token_file_is_read_and_trimmed() {
        let dir = std::env::temp_dir().join(format!("zcprobe-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("token");
        std::fs::write(&path, "  file-token \n").unwrap();
        let a = parse_args(&["health".into(), "--token-file".into(), path.to_string_lossy().into_owned()], &no_env).unwrap();
        assert_eq!(a.globals.token.as_deref(), Some("file-token"));
        let missing = parse_args(&["health".into(), "--token-file".into(), "/nonexistent/zc/token".into()], &no_env);
        assert!(missing.unwrap_err().0.contains("cannot read token file"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn complete_joins_positionals_and_reads_options() {
        let a = parse("complete say hi --hint slow --temperature 0.5 --system be-brief").unwrap();
        assert_eq!(
            a.command,
            Command::Complete {
                prompt: "say hi".into(),
                hint: "slow".into(),
                system: Some("be-brief".into()),
                temperature: Some(0.5)
            }
        );
        let default = parse("complete hello").unwrap();
        assert!(matches!(default.command, Command::Complete { ref hint, .. } if hint == "quick"));
    }

    #[test]
    fn chat_collects_repeated_images_and_switches() {
        let a = parse("chat describe --session s1 --image a.png --image b.jpg --auto-approve --thinking --agent x").unwrap();
        assert_eq!(
            a.command,
            Command::Chat {
                prompt: "describe".into(),
                session: Some("s1".into()),
                auto_approve: true,
                images: vec!["a.png".into(), "b.jpg".into()],
                thinking: true
            }
        );
    }

    #[test]
    fn frames_prompt_is_optional() {
        assert_eq!(
            parse("frames").unwrap().command,
            Command::Frames { prompt: None, session: None, seconds: 5, auto_approve: false }
        );
        assert_eq!(
            parse("frames hello world --seconds 9").unwrap().command,
            Command::Frames { prompt: Some("hello world".into()), session: None, seconds: 9, auto_approve: false }
        );
    }

    #[test]
    fn history_and_sessions_options() {
        assert_eq!(
            parse("history abc --limit 5 --tool-context").unwrap().command,
            Command::History { id: "abc".into(), limit: 5, tool_context: true }
        );
        assert!(matches!(parse("history abc").unwrap().command, Command::History { limit: 20, .. }));
        assert_eq!(parse("sessions --all").unwrap().command, Command::Sessions { all: true });
        assert_eq!(parse("sessions").unwrap().command, Command::Sessions { all: false });
    }

    #[test]
    fn double_dash_ends_flag_parsing() {
        let a = parse("chat -- --not-a-flag here").unwrap();
        assert!(matches!(a.command, Command::Chat { ref prompt, .. } if prompt == "--not-a-flag here"));
    }

    #[test]
    fn usage_errors_are_reported() {
        for bad in [
            "",
            "bogus",
            "health --nope",
            "health --url",
            "chat",
            "complete",
            "history",
            "history a b",
            "health --json=1",
            "health --timeout abc",
            "complete x --temperature hot",
            "frames --seconds -1",
        ] {
            assert!(parse(bad).is_err(), "`{bad}` should be a usage error");
        }
    }

    #[test]
    fn help_forms() {
        assert_eq!(parse("--help").unwrap().command, Command::Help);
        assert_eq!(parse("-h").unwrap().command, Command::Help);
        assert_eq!(parse("help").unwrap().command, Command::Help);
        assert_eq!(parse("health --help").unwrap().command, Command::Help);
    }

    #[test]
    fn timeout_and_json_are_global() {
        let a = parse("sessions --timeout 7 --json").unwrap();
        assert_eq!(a.globals.timeout, Some(Duration::from_secs(7)));
        assert!(a.globals.json);
    }

    #[test]
    fn config_requires_an_address_and_an_agent_when_asked() {
        let g = Globals { url: None, token: None, agent: None, timeout: None, json: false };
        assert!(matches!(config_for(&g, false), Err(Error::Config(_))));
        let g = Globals { url: Some("h".into()), ..g };
        assert!(config_for(&g, false).is_ok());
        assert!(matches!(config_for(&g, true), Err(Error::Config(_))));
        let g = Globals { agent: Some("a".into()), ..g };
        assert_eq!(config_for(&g, true).unwrap().agent_name(), "a");
        let bad = Globals { url: Some("ftp://x".into()), ..g };
        assert!(config_for(&bad, false).is_err());
    }

    #[test]
    fn output_redaction_scrubs_the_token() {
        let mut so = Vec::new();
        let mut se = Vec::new();
        let mut out = Out { stdout: &mut so, stderr: &mut se, token: Token::new("sekrit/tok") };
        out.outln("value sekrit/tok and sekrit%2Ftok");
        out.errln("also sekrit/tok");
        drop(out);
        let all = format!("{}{}", String::from_utf8_lossy(&so), String::from_utf8_lossy(&se));
        assert!(!all.contains("sekrit"), "{all}");
        assert!(all.contains("***"));
    }

    #[test]
    fn short_truncates_on_char_boundaries() {
        assert_eq!(short("a\nb", 10), "a b");
        assert_eq!(short(&"é".repeat(10), 3), "ééé...");
    }
}
