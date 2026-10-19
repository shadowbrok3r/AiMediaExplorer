//! The `zc_memory` application command: the library guide, the gateway chats, the usage totals and the gateway configuration snippet, for scripts and the Devtools windows.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::events::Pending;
use super::{memory, runtime, sessions, snippet, usage};
use crate::ai::mcp::activity::clip;
use crate::command::{AppCommand, CommandError, CommandResult, ToastLevel};

/// What a `zc_memory` command does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOp {
    /// Report the guide: whether it is the shipped text, its length, the last push and the text.
    Guide,
    /// Replace the guide text with `text`; blank text or the shipped text brings the default back.
    GuideSet,
    GuideReset,
    /// Store the guide in the gateway memory of the agent.
    GuidePush,
    /// Ask the gateway what it holds under the guide key.
    GuideCheck,
    /// List the gateway chats of the agent.
    Sessions,
    /// Show the gateway session browser in the assistant.
    SessionsOpen,
    /// Open the gateway chat `session` in the assistant.
    SessionOpen,
    /// Name the gateway chat `session` as `name`.
    SessionRename,
    /// Delete the gateway chat `session`.
    SessionDelete,
    /// Report the usage totals, the sessions and the days.
    Usage,
    UsageReset,
    /// Return the gateway configuration snippet.
    Snippet,
    /// Call util_ping through the tools server's own address.
    SnippetTest,
}

impl MemoryOp {
    pub const ALL: [MemoryOp; 14] = [
        Self::Guide,
        Self::GuideSet,
        Self::GuideReset,
        Self::GuidePush,
        Self::GuideCheck,
        Self::Sessions,
        Self::SessionsOpen,
        Self::SessionOpen,
        Self::SessionRename,
        Self::SessionDelete,
        Self::Usage,
        Self::UsageReset,
        Self::Snippet,
        Self::SnippetTest,
    ];
}

/// The fields a `zc_memory` command can carry; each op reads the ones it needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryArgs {
    /// Guide text for `guide_set`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Gateway session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// New name for `session_rename`, and the name shown for `session_open`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

fn need<'a>(value: &'a Option<String>, what: &str) -> Result<&'a str, CommandError> {
    value.as_deref().map(str::trim).filter(|v| !v.is_empty()).ok_or_else(|| CommandError::invalid(format!("this command needs {what}")))
}

fn toast(kind: ToastLevel, text: String) {
    crate::command::send(AppCommand::Toast { kind, text });
}

/// Waits for `job` in the background and reports what `describe` makes of its result as a toast and a note of the runtime.
fn watch<T: Send + 'static>(what: &'static str, mut job: Pending<T>, describe: impl FnOnce(T) -> Result<String, String> + Send + 'static) {
    let sh = runtime().shared().clone();
    let work = async move {
        let result = loop {
            if let Some(r) = job.take() {
                break r;
            }
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        };
        match describe(result) {
            Ok(text) => {
                sh.note(format!("{what}: {text}"));
                toast(ToastLevel::Success, format!("ZeroClaw {what}: {}", clip(&text, 100)));
            }
            Err(e) => {
                sh.note(format!("{what} failed: {e}"));
                toast(ToastLevel::Error, format!("ZeroClaw {what} failed: {}", clip(&e, 100)));
            }
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(rt) => {
            rt.spawn(work);
        }
        Err(_) => {
            std::thread::spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    rt.block_on(work);
                }
            });
        }
    }
}

/// The JSON `guide` reports.
pub fn guide_json() -> Value {
    let guide = memory::guide();
    json!({
        "default": guide.is_default(),
        "length": guide.effective().chars().count(),
        "pushed_ms": guide.pushed_ms,
        "changed_since_push": guide.changed_since_push(),
        "text": guide.effective(),
    })
}

/// The JSON `usage` reports.
pub fn usage_json() -> Value {
    let data = usage::store().snapshot();
    json!({
        "since_ms": data.since_ms,
        "total": data.total,
        "sessions": data.sessions,
        "days": data.days,
        "file": usage::store().path().map(|p| p.display().to_string()),
    })
}

fn configured() -> Result<(), CommandError> {
    let rt = runtime();
    if !rt.profile().is_configured() || !rt.has_token() {
        return Err(CommandError::invalid("the gateway is not set up: configure a host, an agent and a token first"));
    }
    Ok(())
}

/// Runs one `zc_memory` command; work that needs the network finishes in the background and reports through a toast and the Devtools notes.
pub fn run(op: MemoryOp, args: MemoryArgs) -> CommandResult {
    let rt = runtime();
    match op {
        MemoryOp::Guide => Ok(guide_json().to_string()),
        MemoryOp::GuideSet => {
            let text = args.text.as_deref().unwrap_or("");
            memory::save_text(text);
            Ok(format!("guide saved ({} characters)", memory::guide().effective().chars().count()))
        }
        MemoryOp::GuideReset => {
            memory::reset_text();
            Ok("the shipped guide is back".to_string())
        }
        MemoryOp::GuidePush => {
            configured()?;
            let job = memory::push_guide(rt, crate::settings::store().clone());
            watch("guide push", job, |r| r.map(|p| format!("{} characters stored as {}", p.bytes, p.key)));
            Ok("pushing the guide".to_string())
        }
        MemoryOp::GuideCheck => {
            configured()?;
            let job = memory::remote(rt, memory::guide().effective().to_string());
            watch("guide check", job, |r| match r {
                Ok(Some(found)) if found.same_as_local => Ok(format!("the gateway holds the same guide ({} characters)", found.bytes)),
                Ok(Some(found)) => Ok(format!("the gateway holds a different guide ({} characters)", found.bytes)),
                Ok(None) => Ok("the gateway holds no guide".to_string()),
                Err(e) => Err(e),
            });
            Ok("asking the gateway".to_string())
        }
        MemoryOp::Sessions => {
            configured()?;
            watch("sessions", sessions::list(rt), |r| match r {
                Ok(rows) => Ok(format!("{} chats\n{}", rows.len(), rows.iter().map(|s| format!("{} | {} | {} messages | {}", s.id, s.name.as_deref().unwrap_or("-"), s.message_count, s.last_activity.as_deref().unwrap_or("-"))).collect::<Vec<_>>().join("\n"))),
                Err(e) => Err(e.message),
            });
            Ok("listing the gateway chats".to_string())
        }
        MemoryOp::SessionsOpen => {
            crate::ui::assistant::gateway_sessions::request_browser();
            Ok("opening the gateway session browser".to_string())
        }
        MemoryOp::SessionOpen => {
            let id = need(&args.session, "a session id")?.to_string();
            crate::ui::assistant::gateway_sessions::request_open(crate::ui::assistant::gateway_sessions::Open { id: id.clone(), name: args.name.clone() });
            Ok(format!("opening gateway chat {id}"))
        }
        MemoryOp::SessionRename => {
            configured()?;
            let (id, name) = (need(&args.session, "a session id")?.to_string(), need(&args.name, "a name")?.to_string());
            watch("rename", sessions::rename(rt, vec![format!("gw_{id}"), id.clone()], name), |r| r.map(|_| "renamed".to_string()).map_err(|e| e.message));
            Ok(format!("renaming {id}"))
        }
        MemoryOp::SessionDelete => {
            configured()?;
            let id = need(&args.session, "a session id")?.to_string();
            watch("delete", sessions::delete(rt, vec![id.clone(), format!("gw_{id}")]), |r| r.map(|any| if any { "deleted".to_string() } else { "no such chat".to_string() }).map_err(|e| e.message));
            Ok(format!("deleting {id}"))
        }
        MemoryOp::Usage => Ok(usage_json().to_string()),
        MemoryOp::UsageReset => {
            usage::store().reset();
            Ok("usage totals cleared".to_string())
        }
        MemoryOp::Snippet => Ok(snippet::build(&rt.profile())),
        MemoryOp::SnippetTest => {
            watch("tools ping", snippet::test(rt), |p| if p.ok { Ok(format!("{} in {} ms", p.detail, p.millis)) } else { Err(p.detail) });
            Ok("calling util_ping through the tools server".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_json_form_is_snake_case_and_flat() {
        let cmd: AppCommand = serde_json::from_str(r#"{"cmd":"zc_memory","op":"session_rename","session":"s1","name":"New name"}"#).unwrap();
        assert!(matches!(&cmd, AppCommand::ZcMemory { op: MemoryOp::SessionRename, args } if args.session.as_deref() == Some("s1") && args.name.as_deref() == Some("New name")));
        assert_eq!(cmd.name(), "zc_memory");
        let back = serde_json::to_value(&cmd).unwrap();
        assert_eq!((back["cmd"].clone(), back["op"].clone(), back["session"].clone()), (json!("zc_memory"), json!("session_rename"), json!("s1")));
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"zc_memory","op":"explode"}"#).is_err());
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"zc_memory"}"#).is_err());
    }

    #[test]
    fn the_command_docs_describe_every_op() {
        let docs = include_str!("../../docs/commands.md");
        for op in MemoryOp::ALL {
            let name = serde_json::to_value(op).unwrap().as_str().unwrap().to_string();
            assert!(docs.contains(&format!("`{name}`")), "docs/commands.md does not describe {name}");
        }
        assert!(docs.contains("\"cmd\": \"zc_memory\""));
    }

    #[test]
    fn ops_that_need_arguments_refuse_without_them() {
        let blank = MemoryArgs::default();
        assert_eq!(run(MemoryOp::SessionOpen, blank.clone()).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(MemoryOp::SessionRename, MemoryArgs { session: Some("s".into()), ..Default::default() }).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(MemoryOp::SessionDelete, blank).unwrap_err().kind, crate::command::ErrorKind::Invalid);
    }

    #[test]
    fn the_reports_are_json_and_the_snippet_never_holds_a_token() {
        let guide: Value = serde_json::from_str(&run(MemoryOp::Guide, Default::default()).unwrap()).unwrap();
        assert!(guide["length"].as_u64().unwrap() > 100 && guide["text"].is_string());
        let usage: Value = serde_json::from_str(&run(MemoryOp::Usage, Default::default()).unwrap()).unwrap();
        assert!(usage["total"]["turns"].is_number() && usage["sessions"].is_array() && usage["days"].is_array());
        let snippet = run(MemoryOp::Snippet, Default::default()).unwrap();
        assert!(snippet.contains("[[mcp.servers]]") && snippet.contains("Bearer <paste the token"));
        if let Ok(token) = crate::ai::mcp::hub().token() {
            assert!(!snippet.contains(&token));
        }
    }

    #[test]
    fn network_ops_say_when_the_gateway_is_not_set_up() {
        let rt = runtime();
        if rt.profile().is_configured() && rt.has_token() {
            return;
        }
        for op in [MemoryOp::GuidePush, MemoryOp::GuideCheck, MemoryOp::Sessions] {
            let err = run(op, Default::default()).unwrap_err();
            assert!(err.message.contains("not set up"), "{op:?}: {err}");
        }
    }
}
