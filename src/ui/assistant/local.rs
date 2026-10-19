//! The local backend (CLIP text search, JoyCaption describe) and the slash commands, which run in every backend.

use crossbeam::channel::Sender;
use tokio::task::JoinHandle;

use super::backend::{AssistantBackend, BackendKind, TurnRequest};
use super::emit::Emitter;
use crate::status::{GlobalStatusIndicator, StatusState};
use crate::ui::file_table::{AIUpdate, SimilarResult};
use crate::zc::events::Tagged;

/// Quick actions the composer understands, as (name, what it does).
pub const SLASH_COMMANDS: [(&str, &str); 4] = [
    ("/help", "List these commands"),
    ("/ping <text>", "Check that the tools work and echo the text"),
    ("/search <words>", "Search the library with CLIP and open the results in a tab"),
    ("/describe [path]", "Describe a picture with the local vision model; defaults to the selected one"),
];

/// The text of `/help`.
pub fn help_text() -> String {
    let mut lines = vec!["Quick actions:".to_string()];
    lines.extend(SLASH_COMMANDS.iter().map(|(c, d)| format!("{c}  -  {d}")));
    lines.join("\n")
}

/// True when the message is a slash command.
pub fn is_slash(text: &str) -> bool {
    text.trim_start().starts_with('/')
}

/// Splits `/name rest` into the lowercase name and the trimmed rest.
pub fn split_slash(text: &str) -> (String, String) {
    let body = text.trim_start().trim_start_matches('/');
    let mut parts = body.splitn(2, char::is_whitespace);
    (parts.next().unwrap_or("").to_lowercase(), parts.next().unwrap_or("").trim().to_string())
}

/// Runs a slash command and returns its reply.
pub async fn slash_reply(text: &str, selected: &str, updates: Sender<AIUpdate>) -> String {
    let (name, arg) = split_slash(text);
    match name.as_str() {
        "help" => help_text(),
        "ping" => match crate::ai::mcp::ping_tool(arg).await {
            Ok(echo) => format!("pong: {echo}"),
            Err(e) => format!("Ping failed: {e}"),
        },
        "search" => match crate::ai::mcp::media_search_tool(arg.clone(), Some(32)).await {
            Ok(hits) => {
                let mut results = Vec::new();
                for h in hits.iter() {
                    let thumb = crate::Thumbnail::get_thumbnail_by_path(&h.path).await.unwrap_or(None).unwrap_or_default();
                    results.push(SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(h.score) });
                }
                let _ = updates.try_send(AIUpdate::SimilarResults { origin_path: format!("mcp:media.search:{arg}"), results });
                format!("Found {} similar items for '{}'. See the results tab.", hits.len(), arg)
            }
            Err(e) => format!("Search failed: {e}"),
        },
        "describe" => {
            let chosen = if arg.is_empty() { selected.to_string() } else { arg };
            if chosen.is_empty() {
                "No file selected or path specified.".to_string()
            } else {
                match crate::ai::mcp::media_describe_tool(chosen).await {
                    Ok(resp) => format!("Caption: {}\nCategory: {}\nTags: {}", resp.caption, resp.category, resp.tags.join(", ")),
                    Err(e) => format!("Describe failed: {e}"),
                }
            }
        }
        other => format!("Unknown command: /{other}. Try /help."),
    }
}

pub struct LocalBackend {
    tx: Sender<Tagged>,
    task: Option<JoinHandle<()>>,
    current: Option<Emitter>,
}

impl LocalBackend {
    pub fn new(tx: Sender<Tagged>) -> Self {
        Self { tx, task: None, current: None }
    }

    /// Runs a slash command as a turn.
    pub fn slash(&mut self, req: &TurnRequest, selected: String) {
        let emitter = Emitter::new(self.tx.clone(), &req.session, req.epoch, req.turn);
        self.current = Some(emitter.clone());
        crate::status::ASSIST_STATUS.set_state(StatusState::Running, "MCP command");
        crate::status::ASSIST_STATUS.set_model("mcp");
        let (text, updates) = (req.text.clone(), req.ai_update_tx.clone());
        self.task = Some(tokio::spawn(async move {
            let reply = slash_reply(&text, &selected, updates).await;
            emitter.text(&reply);
            emitter.done(&reply);
            crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
        }));
    }
}

impl AssistantBackend for LocalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Local
    }

    fn send(&mut self, req: TurnRequest) -> Result<(), String> {
        let emitter = Emitter::new(self.tx.clone(), &req.session, req.epoch, req.turn);
        self.current = Some(emitter.clone());
        crate::status::ASSIST_STATUS.set_state(StatusState::Running, "Local request");
        crate::status::ASSIST_STATUS.set_model("local");
        let updates = req.ai_update_tx.clone();
        let model_name = req.settings.openai_default_model.clone().unwrap_or_else(|| "local-joycaption".into());
        let file = req.attachments.first().cloned().filter(|p| std::path::Path::new(p).is_file());
        let has_attachment = !req.attachments.is_empty();
        let instruction = req.text.clone();
        self.task = Some(match (file, has_attachment) {
            (Some(path), _) => tokio::spawn(async move {
                match tokio::fs::read(&path).await {
                    Ok(bytes) => {
                        let _ = crate::ai::joycap::ensure_worker_started().await;
                        let streamed = emitter.clone();
                        let interim = updates.clone();
                        let interim_path = path.clone();
                        let mut acc = String::new();
                        let _ = crate::ai::joycap::stream_describe_bytes_with_callback(bytes, &instruction, |tok| {
                            let s = tok.to_string();
                            acc.push_str(&s);
                            let _ = interim.try_send(AIUpdate::Interim { path: interim_path.clone(), text: s.clone() });
                            streamed.text(&s);
                        })
                        .await;
                        crate::settings::update("assistant:model-used", |s| s.push_recent_model(&model_name));
                        emitter.done(&acc);
                        crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
                    }
                    Err(e) => {
                        log::error!("read image failed: {e}");
                        emitter.error(format!("Could not read {path}: {e}"));
                        crate::status::ASSIST_STATUS.set_error(e.to_string());
                    }
                }
            }),
            (None, true) => tokio::spawn(async move {
                emitter.error("The attached file is not on disk.");
                crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
            }),
            (None, false) => tokio::spawn(async move {
                let reply = clip_search(&instruction, updates).await;
                crate::settings::update("assistant:model-used", |s| s.push_recent_model(&model_name));
                emitter.text(&reply);
                emitter.done(&reply);
                crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
            }),
        });
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        if let Some(e) = self.current.take() {
            e.aborted();
        }
        crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
    }
}

/// Searches the library by text with CLIP and hands the hits to the explorer; returns the chat reply.
async fn clip_search(query: &str, updates: Sender<AIUpdate>) -> String {
    let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
    let vector = {
        let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
        guard.as_mut().and_then(|engine| engine.embed_text(query).ok())
    };
    let Some(vector) = vector else {
        return "The CLIP model is not available, so the library cannot be searched by text. Load it in Settings.".to_string();
    };
    let mut results: Vec<SimilarResult> = Vec::new();
    match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&vector, 64, 128, 0).await {
        Ok(hits) => {
            for hit in hits {
                let thumb = match hit.thumb_ref {
                    Some(t) => t,
                    None => crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default(),
                };
                results.push(SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(hit.dist) });
            }
        }
        Err(e) => {
            log::error!("text search knn failed: {e}");
            return format!("The search failed: {e}");
        }
    }
    let found = results.len();
    let _ = updates.try_send(AIUpdate::SimilarResults { origin_path: format!("query:{query}"), results });
    format!("Found {found} items for '{query}'. See the results tab.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_messages_are_recognised_and_split() {
        assert!(is_slash("  /ping hi"));
        assert!(!is_slash("ping"));
        assert_eq!(split_slash("/Search red cars"), ("search".to_string(), "red cars".to_string()));
        assert_eq!(split_slash("/describe"), ("describe".to_string(), String::new()));
        assert_eq!(split_slash("/ping   spaced  out "), ("ping".to_string(), "spaced  out".to_string()));
    }

    #[test]
    fn help_lists_every_command() {
        let text = help_text();
        for (cmd, _) in SLASH_COMMANDS {
            assert!(text.contains(cmd), "{cmd}");
        }
    }

    #[tokio::test]
    async fn unknown_and_help_commands_reply_without_touching_models() {
        let (tx, _rx) = crossbeam::channel::unbounded();
        assert!(slash_reply("/help", "", tx.clone()).await.contains("/ping"));
        assert!(slash_reply("/nope", "", tx.clone()).await.contains("Unknown command: /nope"));
        assert_eq!(slash_reply("/describe", "", tx).await, "No file selected or path specified.");
    }
}
