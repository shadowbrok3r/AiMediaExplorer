//! One chat connection as a task: connects, turns client events into tagged interface events and answers the commands of its mailbox.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crossbeam::channel::{Receiver, Sender};
use tokio::sync::Notify;
use zeroclaw_client::{AbortOutcome, ApprovalDecision, ChatSession, ClientFrame, ConnectOptions, DisconnectReason, Error, Event, HistoryPager, InlineImage};

use super::coalesce::{Coalescer, Run, RunKind};
use super::events::{ChatEv, LinkState, PlanRow};
use super::runtime::Shared;

/// What the interface asks one connection to do.
pub enum SessionCmd {
    Send { text: String, images: Vec<InlineImage> },
    Approval { request_id: String, decision: ApprovalDecision },
    /// Stop the running turn and ignore its frames until the gateway confirms.
    Stop,
    /// Drop the socket and connect again at once.
    Reconnect,
    /// A frame typed by hand; only the two frames the gateway accepts can be sent.
    Raw(String),
    /// The newest page of the stored transcript, or with `more` the next older one.
    History { page_size: usize, more: bool },
    Close,
}

/// Commands of one connection on a crossbeam channel, with a wake-up for the task that owns it.
#[derive(Clone)]
pub struct Mailbox {
    tx: Sender<SessionCmd>,
    rx: Receiver<SessionCmd>,
    wake: Arc<Notify>,
}

impl Mailbox {
    pub fn new() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self { tx, rx, wake: Arc::new(Notify::new()) }
    }

    pub fn send(&self, cmd: SessionCmd) {
        let _ = self.tx.send(cmd);
        self.wake.notify_one();
    }

    fn take(&self) -> Vec<SessionCmd> {
        self.rx.try_iter().collect()
    }
}

/// Session and connection number carried by every event of one task.
#[derive(Clone, Debug)]
pub struct Tag {
    pub session: String,
    pub epoch: u64,
}

/// A refused token surfaces as a fatal disconnect whose text names the status.
pub fn is_refusal_text(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("http 401") || t.contains("http 403") || t.contains("unauthorized") || t.contains("forbidden")
}

fn page_event(items: Vec<zeroclaw_client::HistoryItem>, pager: &HistoryPager, initial: bool) -> ChatEv {
    ChatEv::History { items, has_more: pager.has_more(), total: pager.total_messages(), initial }
}

/// Runs the connection until it is closed, superseded or fails for good.
pub(super) async fn run(sh: Arc<Shared>, tag: Tag, name: Option<String>, mailbox: Mailbox) {
    let cfg = match sh.chat_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            sh.fail_link(&tag, e).await;
            return;
        }
    };
    sh.set_link(&tag, LinkState::Connecting, "opening the chat socket");
    let opts = ConnectOptions { session_id: Some(tag.session.clone()), name, cwd: None, raw_tap: true };
    let connecting = ChatSession::connect(cfg, opts);
    tokio::pin!(connecting);
    let (session, mut events) = loop {
        tokio::select! {
            result = &mut connecting => match result {
                Ok(pair) => break pair,
                Err(e) => {
                    sh.connect_failed(&tag, &e).await;
                    return;
                }
            },
            _ = mailbox.wake.notified() => {
                for cmd in mailbox.take() {
                    match cmd {
                        SessionCmd::Close => {
                            sh.set_link(&tag, LinkState::Closed, "closed while connecting");
                            sh.release(&tag);
                            return;
                        }
                        SessionCmd::Send { text, .. } => sh.emit(&tag, ChatEv::SendFailed { text, error: "still connecting".into() }).await,
                        _ => {}
                    }
                }
            }
        }
    };
    if !sh.is_current(&tag) {
        session.close().await;
        return;
    }

    let suppress = Arc::new(AtomicBool::new(false));
    let pager: Arc<std::sync::Mutex<Option<HistoryPager>>> = Arc::default();
    let mut co = Coalescer::default();
    'run: loop {
        let wait = co.wait(Instant::now());
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else { break 'run };
                handle_event(&sh, &tag, event, &mut co, &suppress).await;
            }
            _ = mailbox.wake.notified() => {
                for cmd in mailbox.take() {
                    if !handle_cmd(&sh, &tag, &session, cmd, &suppress, &pager).await {
                        break 'run;
                    }
                }
            }
            _ = async { match wait { Some(d) => tokio::time::sleep(d).await, None => std::future::pending::<()>().await } } => {
                if let Some(run) = co.flush() {
                    sh.emit(&tag, run_event(run)).await;
                }
            }
        }
    }
    if let Some(run) = co.flush() {
        sh.emit(&tag, run_event(run)).await;
    }
    if sh.link_state() != LinkState::Failed {
        sh.set_link(&tag, LinkState::Closed, "closed");
    }
    sh.release(&tag);
}

fn run_event(run: Run) -> ChatEv {
    match run.kind {
        RunKind::Text => ChatEv::Text { turn: run.turn, segment: run.segment, text: run.text },
        RunKind::Thinking => ChatEv::Thinking { turn: run.turn, segment: run.segment, text: run.text },
    }
}

async fn flushed(sh: &Shared, tag: &Tag, co: &mut Coalescer) {
    if let Some(run) = co.flush() {
        sh.emit(tag, run_event(run)).await;
    }
}

async fn handle_event(sh: &Arc<Shared>, tag: &Tag, event: Event, co: &mut Coalescer, suppress: &AtomicBool) {
    let hushed = suppress.load(Ordering::Relaxed);
    match event {
        Event::Raw(frame) => sh.record_frame(frame),
        Event::Connected { resumed, message_count, name, reconnect, .. } => {
            flushed(sh, tag, co).await;
            sh.set_link(tag, LinkState::Open, if reconnect { "reconnected" } else { "connected" });
            sh.emit(tag, ChatEv::Connected { resumed, message_count, name, reconnect }).await;
        }
        Event::Disconnected { reason, will_retry } => {
            flushed(sh, tag, co).await;
            if let DisconnectReason::Fatal(text) = &reason {
                if is_refusal_text(text) {
                    sh.refused();
                    sh.emit(tag, ChatEv::AuthRefused).await;
                }
            }
            let state = if will_retry { LinkState::Reconnecting } else { LinkState::Closed };
            sh.set_link(tag, state, reason.to_string());
        }
        Event::Reconnecting { attempt, delay, last_error } => {
            let why = last_error.map(|e| format!(": {e}")).unwrap_or_default();
            sh.set_link(tag, LinkState::Reconnecting, format!("attempt {attempt} in {} ms{why}", delay.as_millis()));
        }
        Event::TurnStarted { turn } => {
            flushed(sh, tag, co).await;
            sh.emit(tag, ChatEv::TurnStarted { turn }).await;
        }
        Event::Token { turn, segment, text } => {
            if !hushed {
                if let Some(run) = co.push(RunKind::Text, turn, segment, &text, Instant::now()) {
                    sh.emit(tag, run_event(run)).await;
                }
            }
        }
        Event::Thinking { turn, segment, text } => {
            if !hushed {
                if let Some(run) = co.push(RunKind::Thinking, turn, segment, &text, Instant::now()) {
                    sh.emit(tag, run_event(run)).await;
                }
            }
        }
        Event::ToolCall { turn, index, id, name, args } => {
            flushed(sh, tag, co).await;
            if !hushed {
                sh.emit(tag, ChatEv::ToolCall { turn, index, id, name, args }).await;
            }
        }
        Event::ToolResult { turn, index, id, name, output } => {
            flushed(sh, tag, co).await;
            if !hushed {
                sh.emit(tag, ChatEv::ToolResult { turn, index, id, name, output }).await;
            }
        }
        Event::ApprovalRequest(req) => {
            flushed(sh, tag, co).await;
            if !hushed {
                sh.emit(tag, ChatEv::Approval(req)).await;
            }
        }
        Event::Plan { turn, entries } => {
            flushed(sh, tag, co).await;
            if !hushed {
                let rows = entries.into_iter().map(|e| PlanRow { content: e.content, status: e.status }).collect();
                sh.emit(tag, ChatEv::Plan { turn, rows }).await;
            }
        }
        Event::HistoryTrimmed { dropped_messages, kept_turns, reason } => {
            sh.emit(tag, ChatEv::HistoryTrimmed { dropped: dropped_messages, kept: kept_turns, reason }).await;
        }
        Event::Done { turn, full_response, usage } => {
            flushed(sh, tag, co).await;
            suppress.store(false, Ordering::Relaxed);
            sh.add_usage(&usage);
            super::usage::record(&tag.session, &usage);
            if hushed {
                sh.emit(tag, ChatEv::Aborted { turn }).await;
            } else {
                sh.emit(tag, ChatEv::Done { turn, full_response, usage }).await;
            }
        }
        Event::Aborted { turn } => {
            flushed(sh, tag, co).await;
            suppress.store(false, Ordering::Relaxed);
            sh.emit(tag, ChatEv::Aborted { turn }).await;
        }
        Event::Error { turn, code, message, ends_turn } => {
            flushed(sh, tag, co).await;
            if ends_turn {
                suppress.store(false, Ordering::Relaxed);
            }
            sh.emit(tag, ChatEv::Error { turn, code, message, ends_turn }).await;
        }
        Event::TurnResumed { turn, .. } => {
            flushed(sh, tag, co).await;
            suppress.store(false, Ordering::Relaxed);
            sh.emit(tag, ChatEv::TurnResumed { turn }).await;
        }
        Event::TurnLost { turn } => {
            flushed(sh, tag, co).await;
            suppress.store(false, Ordering::Relaxed);
            sh.emit(tag, ChatEv::TurnLost { turn }).await;
        }
        Event::Unknown { .. } => {}
    }
}

/// Runs one command; false ends the connection.
async fn handle_cmd(sh: &Arc<Shared>, tag: &Tag, session: &ChatSession, cmd: SessionCmd, suppress: &Arc<AtomicBool>, pager: &Arc<std::sync::Mutex<Option<HistoryPager>>>) -> bool {
    match cmd {
        SessionCmd::Close => {
            session.close().await;
            return false;
        }
        SessionCmd::Send { text, images } => {
            let (sh, tag, session) = (sh.clone(), tag.clone(), session.clone());
            tokio::spawn(async move {
                if let Err(e) = session.send_user_message(&text, &images).await {
                    let error = sh.redact(&e.to_string());
                    sh.emit(&tag, ChatEv::SendFailed { text, error }).await;
                }
            });
        }
        SessionCmd::Approval { request_id, decision } => {
            if let Err(e) = session.respond_approval(&request_id, decision).await {
                sh.emit(tag, ChatEv::Notice(format!("The approval was not delivered: {}", sh.redact(&e.to_string())))).await;
            }
        }
        SessionCmd::Stop => {
            suppress.store(true, Ordering::Relaxed);
            let (sh, tag, session, suppress) = (sh.clone(), tag.clone(), session.clone(), suppress.clone());
            tokio::spawn(async move {
                match session.abort().await {
                    Ok(AbortOutcome::Aborted) => {}
                    Ok(AbortOutcome::NothingRunning) => suppress.store(false, Ordering::Relaxed),
                    Err(e) => {
                        suppress.store(false, Ordering::Relaxed);
                        sh.emit(&tag, ChatEv::Notice(format!("Stop failed: {}", sh.redact(&e.to_string())))).await;
                    }
                }
            });
        }
        SessionCmd::Reconnect => {
            let _ = session.reconnect().await;
        }
        SessionCmd::Raw(text) => {
            let sent = match serde_json::from_str::<ClientFrame>(&text) {
                Ok(ClientFrame::Message { content }) => session.send_text(&content).await,
                Ok(ClientFrame::ApprovalResponse { request_id, decision }) => session.respond_approval(&request_id, decision).await,
                Ok(ClientFrame::Connect { .. }) => Err(Error::Config("`connect` is sent once when the socket opens".into())),
                Err(e) => Err(Error::Config(format!("only `message` and `approval_response` frames can be sent: {e}"))),
            };
            let note = match sent {
                Ok(()) => "frame sent".to_string(),
                Err(e) => format!("frame not sent: {}", sh.redact(&e.to_string())),
            };
            sh.emit(tag, ChatEv::Notice(note)).await;
        }
        SessionCmd::History { page_size, more } => {
            if more {
                let page = pager.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(|p| {
                    let items = p.next_page();
                    page_event(items, p, false)
                });
                if let Some(ev) = page {
                    sh.emit(tag, ev).await;
                }
            } else {
                let (sh, tag, session, pager) = (sh.clone(), tag.clone(), session.clone(), pager.clone());
                tokio::spawn(async move {
                    match session.history(page_size, true).await {
                        Ok(mut fetched) => {
                            let items = fetched.next_page();
                            let ev = page_event(items, &fetched, true);
                            *pager.lock().unwrap_or_else(|e| e.into_inner()) = Some(fetched);
                            sh.emit(&tag, ev).await;
                        }
                        Err(e) => sh.emit(&tag, ChatEv::Notice(format!("History not loaded: {}", sh.redact(&e.to_string())))).await,
                    }
                });
            }
        }
    }
    true
}
