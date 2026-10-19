//! The `mcp` application command: start, stop and inspect the server, review proposals and run tools.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::activity::clip;
use super::proposals::Status;
use super::server::{ServerState, hub};
use crate::command::{AppCommand, CommandError, CommandResult, ToastLevel};

/// What an `mcp` command does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpOp {
    /// Report whether the server runs, where, and the counters.
    Status,
    Start,
    Stop,
    /// List the pending proposals.
    Proposals,
    /// Accept the proposal `id` and write its changes to the library.
    Accept,
    /// Reject the proposal `id`.
    Reject,
    /// Run the tool `tool` with `args` in this process.
    Call,
    /// Write a new token.
    RotateToken,
    /// List the pending tag and category rename proposals with what each would touch.
    TaxonomyProposals,
    /// Apply the rename proposal `id` through the journaled tag or category functions.
    TaxonomyAccept,
    /// Reject the rename proposal `id`.
    TaxonomyReject,
    /// List the image edit jobs of the agent.
    EditJobs,
    /// Save the finished edit `id` next to its original; `args` `{"library": true}` also adds it to the library.
    EditAccept,
    /// Discard the finished edit `id`.
    EditReject,
}

/// Longest tool result kept in the Devtools notes.
const NOTE_CHARS: usize = 6000;

fn toast(kind: ToastLevel, text: String) {
    crate::command::send(AppCommand::Toast { kind, text });
}

fn spawn_note<F>(what: &'static str, work: F)
where
    F: std::future::Future<Output = Result<String, String>> + Send + 'static,
{
    tokio::spawn(async move {
        let outcome = work.await;
        let hub = hub();
        match outcome {
            Ok(text) => {
                hub.note(format!("{what}: {text}"));
                toast(ToastLevel::Success, format!("MCP {what}: {}", clip(&text, 100)));
            }
            Err(e) => {
                hub.note(format!("{what} failed: {e}"));
                toast(ToastLevel::Error, format!("MCP {what} failed: {}", clip(&e, 100)));
            }
        }
    });
}

/// Summary line of the server for `status`.
pub fn status_line() -> String {
    let hub = hub();
    let state = match hub.state() {
        ServerState::Stopped => "stopped".to_string(),
        ServerState::Starting => "starting".to_string(),
        ServerState::Running { addr, .. } => format!("running on {addr}"),
        ServerState::Failed(e) => format!("failed: {e}"),
    };
    let stats = hub.stats.snapshot();
    format!("{state}; {} clients, {} requests, {} tool calls, {} GPU jobs queued", hub.clients().len(), stats.requests, stats.tool_calls, hub.ctx.gpu.depth())
}

/// Runs one `mcp` command; work that needs the database or the network finishes in the background and reports through a toast and the Devtools notes.
pub fn run(op: McpOp, id: Option<String>, tool: Option<String>, args: Option<Value>) -> CommandResult {
    let hub = hub().clone();
    match op {
        McpOp::Status => Ok(status_line()),
        McpOp::Start => {
            spawn_note("start", async move { hub.start().await.map(|addr| format!("listening on {addr}")) });
            Ok("starting the MCP server".to_string())
        }
        McpOp::Stop => {
            spawn_note("stop", async move {
                hub.stop().await;
                Ok("stopped".to_string())
            });
            Ok("stopping the MCP server".to_string())
        }
        McpOp::RotateToken => {
            hub.rotate_token().map_err(CommandError::failed)?;
            hub.note("token rotated; update the clients".to_string());
            Ok("a new token was written; clients need it".to_string())
        }
        McpOp::Proposals => {
            spawn_note("proposals", async move {
                let (rows, total) = hub.ctx.proposals.list(Some(Status::Pending), None, 0, 50).await.map_err(|e| e.to_string())?;
                let lines: Vec<String> = rows.iter().map(|p| format!("{} {} {}", p.id, p.path, serde_json::to_string(&p.changes).unwrap_or_default())).collect();
                Ok(format!("{total} pending\n{}", lines.join("\n")))
            });
            Ok("listing pending proposals".to_string())
        }
        McpOp::Accept | McpOp::Reject => {
            let id = id.filter(|i| !i.trim().is_empty()).ok_or_else(|| CommandError::invalid("this command needs a proposal id"))?;
            let accept = op == McpOp::Accept;
            let shown = id.clone();
            spawn_note(if accept { "accept" } else { "reject" }, async move {
                let proposal = if accept { hub.ctx.proposals.accept(&id).await } else { hub.ctx.proposals.reject(&id).await };
                proposal.map(|p| format!("proposal {} is {}", p.id, p.status.as_str())).map_err(|e| e.to_string())
            });
            Ok(format!("{} proposal {shown}", if accept { "accepting" } else { "rejecting" }))
        }
        McpOp::TaxonomyProposals => {
            spawn_note("taxonomy proposals", async move {
                let store = &hub.ctx.agent.taxonomy;
                let (rows, total) = store.list(Some(Status::Pending), 0, 50).await.map_err(|e| e.to_string())?;
                let mut lines = Vec::new();
                for p in &rows {
                    let counts = match store.preview(&p.rename()).await {
                        Ok(v) => v.summary(),
                        Err(e) => e.to_string(),
                    };
                    lines.push(format!("{} {} ({counts})", p.id, p.rename().title()));
                }
                Ok(format!("{total} pending\n{}", lines.join("\n")))
            });
            Ok("listing pending taxonomy proposals".to_string())
        }
        McpOp::TaxonomyAccept | McpOp::TaxonomyReject => {
            let id = id.filter(|i| !i.trim().is_empty()).ok_or_else(|| CommandError::invalid("this command needs a proposal id"))?;
            let accept = op == McpOp::TaxonomyAccept;
            let shown = id.clone();
            spawn_note(if accept { "taxonomy accept" } else { "taxonomy reject" }, async move {
                let store = &hub.ctx.agent.taxonomy;
                let proposal = if accept { store.accept(&id).await } else { store.reject(&id).await };
                if accept {
                    crate::ui::sidebar::request_refresh();
                }
                proposal.map(|p| format!("proposal {} is {}{}", p.id, p.status.as_str(), p.applied.map(|n| format!(", {n} changes")).unwrap_or_default())).map_err(|e| e.to_string())
            });
            Ok(format!("{} taxonomy proposal {shown}", if accept { "accepting" } else { "rejecting" }))
        }
        McpOp::EditJobs => {
            let jobs = hub.ctx.agent.edits.list();
            let lines: Vec<String> = jobs.iter().map(|j| format!("{} {} {} review {} {}", j.id, j.state.as_str(), j.file_name(), j.review.as_str(), j.saved_to.as_deref().unwrap_or(""))).collect();
            Ok(format!("{} edit jobs\n{}", jobs.len(), lines.join("\n")))
        }
        McpOp::EditAccept => {
            let id = id.filter(|i| !i.trim().is_empty()).ok_or_else(|| CommandError::invalid("this command needs an edit job id"))?;
            let library = args.as_ref().and_then(|a| a.get("library")).and_then(Value::as_bool).unwrap_or(false);
            let shown = id.clone();
            spawn_note("edit accept", async move {
                let edits = &hub.ctx.agent.edits;
                let saved = if library { edits.add_to_library(&id, &hub.ctx.db).await } else { edits.save_new(&id) };
                if library {
                    crate::ui::sidebar::request_refresh();
                }
                saved.map(|(_, path)| format!("saved {}{}", path.display(), if library { " and added to the library" } else { "" })).map_err(|e| e.to_string())
            });
            Ok(format!("saving edit {shown}"))
        }
        McpOp::EditReject => {
            let id = id.filter(|i| !i.trim().is_empty()).ok_or_else(|| CommandError::invalid("this command needs an edit job id"))?;
            let job = hub.ctx.agent.edits.reject(&id).map_err(|e| CommandError::invalid(e.to_string()))?;
            Ok(format!("discarded the edit of {}", job.file_name()))
        }
        McpOp::Call => {
            let tool = tool.filter(|t| !t.trim().is_empty()).ok_or_else(|| CommandError::invalid("this command needs a tool name"))?;
            if hub.registry.get(&tool).is_none() {
                return Err(CommandError::not_found(format!("no tool named '{tool}'")));
            }
            let shown = tool.clone();
            spawn_note("call", async move {
                let started = std::time::Instant::now();
                let result = hub.call_local(&tool, args.unwrap_or(Value::Null), CancellationToken::new()).await;
                let ms = started.elapsed().as_millis();
                match result {
                    Ok(out) => Ok(format!("{tool} ok in {ms} ms\n{}", clip(&format!("{}\n{}", out.text, out.data), NOTE_CHARS))),
                    Err(e) => Err(format!("{tool} in {ms} ms: {e}")),
                }
            });
            Ok(format!("calling {shown}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::bus::testing;

    #[test]
    fn the_json_form_is_snake_case() {
        let cmd: AppCommand = serde_json::from_str(r#"{"cmd":"mcp","op":"rotate_token"}"#).unwrap();
        assert!(matches!(cmd, AppCommand::Mcp { op: McpOp::RotateToken, id: None, tool: None, args: None }));
        let call: AppCommand = serde_json::from_str(r#"{"cmd":"mcp","op":"call","tool":"library_stats","args":{"a":1}}"#).unwrap();
        assert!(matches!(&call, AppCommand::Mcp { op: McpOp::Call, tool: Some(t), args: Some(_), .. } if t == "library_stats"));
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"mcp","op":"explode"}"#).is_err());
        assert_eq!(call.name(), "mcp");
    }

    #[test]
    fn the_command_docs_describe_every_op() {
        let docs = include_str!("../../../docs/commands.md");
        for op in ["status", "start", "stop", "rotate_token", "proposals", "accept", "reject", "call", "taxonomy_proposals", "taxonomy_accept", "taxonomy_reject", "edit_jobs", "edit_accept", "edit_reject"] {
            assert!(docs.contains(&format!("`{op}`")), "docs/commands.md does not describe {op}");
            let parsed: McpOp = serde_json::from_value(Value::String(op.to_string())).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), Value::String(op.to_string()));
        }
        assert!(docs.contains("\"cmd\": \"mcp\""));
    }

    #[tokio::test]
    async fn commands_without_their_arguments_are_invalid() {
        let _serial = testing::serial();
        assert_eq!(run(McpOp::Accept, None, None, None).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(McpOp::Reject, Some("  ".into()), None, None).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        assert_eq!(run(McpOp::Call, None, None, None).unwrap_err().kind, crate::command::ErrorKind::Invalid);
        for op in [McpOp::TaxonomyAccept, McpOp::TaxonomyReject, McpOp::EditAccept, McpOp::EditReject] {
            assert_eq!(run(op, None, None, None).unwrap_err().kind, crate::command::ErrorKind::Invalid, "{op:?}");
        }
        assert_eq!(run(McpOp::Call, None, Some("no_such_tool".into()), None).unwrap_err().kind, crate::command::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn status_reports_a_stopped_server() {
        let text = run(McpOp::Status, None, None, None).unwrap();
        assert!(text.contains("stopped") || text.contains("running") || text.contains("failed") || text.contains("starting"), "{text}");
        assert!(text.contains("tool calls"));
    }
}
