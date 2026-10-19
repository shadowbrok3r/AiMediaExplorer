//! The direct backend: an OpenAI-compatible provider or OpenRouter with the conversation so far, answering as a stream.

use base64::Engine as _;
use crossbeam::channel::Sender;
use tokio::task::JoinHandle;

use super::backend::{AssistantBackend, BackendKind, TurnRequest};
use super::emit::Emitter;
use super::providers::provider_config;
use crate::ai::openai_compat::{HistoryTurn, stream_chat_reply};
use crate::status::{GlobalStatusIndicator, StatusState};
use crate::ui::file_table::AIUpdate;
use crate::zc::events::Tagged;

pub struct DirectBackend {
    tx: Sender<Tagged>,
    task: Option<JoinHandle<()>>,
    current: Option<Emitter>,
}

impl DirectBackend {
    pub fn new(tx: Sender<Tagged>) -> Self {
        Self { tx, task: None, current: None }
    }
}

/// The bytes of each attachment: the file, else the stored thumbnail of the library row.
async fn attachment_bytes(paths: &[String]) -> Vec<Vec<u8>> {
    let mut images = Vec::new();
    for p in paths {
        match tokio::fs::read(p).await {
            Ok(b) => images.push(b),
            Err(_) => {
                if let Ok(Some(thumb)) = crate::Thumbnail::get_thumbnail_by_path(p).await {
                    if let Some(b64) = thumb.thumbnail_b64 {
                        let cleaned = b64.split_once(',').map(|(_, v)| v).unwrap_or(&b64);
                        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()) {
                            images.push(bytes);
                        }
                    }
                }
            }
        }
    }
    images
}

impl AssistantBackend for DirectBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Direct
    }

    fn send(&mut self, req: TurnRequest) -> Result<(), String> {
        let cfg = provider_config(&req.settings, req.model.as_deref(), req.temperature)?;
        let emitter = Emitter::new(self.tx.clone(), &req.session, req.epoch, req.turn);
        self.current = Some(emitter.clone());
        let model = cfg.model.clone();
        let provider = cfg.provider.clone();
        crate::status::ASSIST_STATUS.set_state(StatusState::Running, "Cloud request");
        crate::status::ASSIST_STATUS.set_model(&provider);
        let history: Vec<HistoryTurn> = req.history.iter().map(|(user, text)| HistoryTurn { user: *user, text: text.clone() }).collect();
        let (prompt, attachments) = (req.prompt, req.attachments);
        let updates: Sender<AIUpdate> = req.ai_update_tx;
        let first_path = attachments.first().cloned().unwrap_or_default();
        self.task = Some(tokio::spawn(async move {
            let images = attachment_bytes(&attachments).await;
            let streamed = emitter.clone();
            match stream_chat_reply(cfg, &history, &prompt, &images, move |tok| streamed.text(tok)).await {
                Ok(full) => {
                    crate::settings::update("assistant:model-used", |s| s.push_recent_model(&model));
                    if !first_path.is_empty() {
                        let _ = updates.try_send(AIUpdate::Interim { path: first_path, text: full.clone() });
                    }
                    emitter.done(&full);
                    crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
                }
                Err(e) => {
                    log::warn!("direct assistant error: {e}");
                    emitter.error(format!("{e}"));
                    crate::status::ASSIST_STATUS.set_error(e.to_string());
                }
            }
        }));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UiSettings;
    use crate::zc::events::ChatEv;
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};

    /// Serves one canned streaming completion and records the request body.
    fn serve(bodies: Arc<Mutex<Vec<serde_json::Value>>>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let body_start;
                loop {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        body_start = i + 4;
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buf[..body_start]).to_lowercase();
                let want: usize = head.lines().find_map(|l| l.strip_prefix("content-length:").and_then(|v| v.trim().parse().ok())).unwrap_or(0);
                while buf.len() < body_start + want {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                if let Ok(v) = serde_json::from_slice(&buf[body_start..]) {
                    bodies.lock().unwrap().push(v);
                }
                let sse = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello \"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"there\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n{sse}");
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/v1")
    }

    fn settings(base: &str) -> UiSettings {
        let mut s = UiSettings::default();
        s.ai_chat_provider = Some("custom".into());
        s.openai_base_url = Some(base.into());
        s.openai_default_model = Some("test-model".into());
        s
    }

    fn request(base: &str, history: Vec<(bool, String)>) -> TurnRequest {
        let (ai_update_tx, _rx) = crossbeam::channel::unbounded();
        TurnRequest {
            session: "s".into(),
            epoch: 3,
            turn: 1,
            text: "and now?".into(),
            prompt: "and now?".into(),
            attachments: vec![],
            images: vec![],
            history,
            settings: settings(base),
            model: None,
            temperature: 0.2,
            ai_update_tx,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_provider_receives_the_conversation_and_the_reply_streams_back() {
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let base = serve(bodies.clone());
        let (tx, rx) = crossbeam::channel::unbounded();
        let mut backend = DirectBackend::new(tx);
        backend.send(request(&base, vec![(true, "first question".into()), (false, "first answer".into())])).unwrap();
        let mut events = Vec::new();
        for _ in 0..400 {
            events.extend(rx.try_iter());
            if events.iter().any(|t| matches!(t.ev, ChatEv::Done { .. } | ChatEv::Error { .. })) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let done = events.iter().find_map(|t| if let ChatEv::Done { full_response, turn, .. } = &t.ev { Some((full_response.clone(), *turn)) } else { None });
        assert_eq!(done, Some(("Hello there".to_string(), 1)), "{events:?}");
        assert!(events.iter().all(|t| t.session == "s" && t.epoch == 3));
        let body = bodies.lock().unwrap()[0].clone();
        let roles: Vec<&str> = body["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"], "the provider gets real history, not just the last message");
        assert_eq!(body["messages"][1]["content"], "first question");
        assert_eq!(body["messages"][3]["content"], "and now?");
        assert_eq!(body["model"], "test-model");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_provider_that_is_not_set_up_is_an_error_not_a_silent_skip() {
        let (tx, _rx) = crossbeam::channel::unbounded();
        let mut backend = DirectBackend::new(tx);
        let mut req = request("http://127.0.0.1:9/v1", vec![]);
        req.settings.ai_chat_provider = Some("local-joycaption".into());
        assert!(backend.send(req).unwrap_err().contains("provider"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stop_reports_an_aborted_turn() {
        let (tx, rx) = crossbeam::channel::unbounded();
        let mut backend = DirectBackend::new(tx);
        backend.send(request("http://127.0.0.1:9/v1", vec![])).unwrap();
        backend.stop();
        let got: Vec<Tagged> = rx.try_iter().collect();
        assert!(got.iter().any(|t| matches!(t.ev, ChatEv::Aborted { turn: 1 })));
    }
}
