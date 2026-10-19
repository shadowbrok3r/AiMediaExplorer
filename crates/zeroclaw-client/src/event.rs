use std::time::Duration;

use serde_json::Value;

use crate::frames::{DoneFrame, PlanEntry, ServerFrame, error_code_ends_turn};

/// Token and cost figures reported by a `done` frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub context_window: Option<u64>,
    pub last_input_tokens: Option<u64>,
}

impl Usage {
    /// Share of the context window the last request used, 0.0 to 1.0, when both numbers are known.
    pub fn context_fraction(&self) -> Option<f32> {
        let used = self.last_input_tokens.or(self.input_tokens)?;
        let window = self.context_window.filter(|w| *w > 0)?;
        Some((used as f32 / window as f32).clamp(0.0, 1.0))
    }
}

impl From<&DoneFrame> for Usage {
    fn from(d: &DoneFrame) -> Self {
        Usage {
            input_tokens: d.input_tokens,
            output_tokens: d.output_tokens,
            total_tokens: d.tokens_used,
            cost_usd: d.cost_usd,
            model: d.model.clone(),
            provider: d.provider.clone(),
            context_window: d.max_context_tokens,
            last_input_tokens: d.last_input_tokens,
        }
    }
}

/// A tool call waiting for a human decision. Answer with `ChatSession::respond_approval`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApprovalRequest {
    pub turn: u64,
    pub request_id: String,
    pub tool: String,
    pub summary: String,
    pub arguments: Value,
    /// Seconds the gateway waits before it denies on its own.
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Frame received from the gateway.
    In,
    /// Frame sent to the gateway.
    Out,
}

/// A frame as it crossed the socket, for the debug tap.
#[derive(Debug, Clone, PartialEq)]
pub struct RawFrame {
    /// Milliseconds since the session started.
    pub t_ms: u64,
    pub dir: Direction,
    pub frame: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisconnectReason {
    /// The gateway closed the socket.
    Remote { code: Option<u16>, reason: String },
    Transport(String),
    /// No pong arrived within the configured timeout.
    PingTimeout,
    /// An inbound frame exceeded `Config::max_frame_bytes`.
    FrameTooLarge,
    /// The gateway reported an error after which it ends the connection.
    ServerError(String),
    /// Reconnecting is not possible or no longer allowed.
    Fatal(String),
    /// The caller closed the session or asked for a reconnect.
    Requested,
}

impl std::fmt::Display for DisconnectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DisconnectReason::Remote { code: Some(c), reason } => write!(f, "closed by gateway ({c}) {reason}"),
            DisconnectReason::Remote { code: None, reason } => write!(f, "closed by gateway {reason}"),
            DisconnectReason::Transport(e) => write!(f, "transport error: {e}"),
            DisconnectReason::PingTimeout => f.write_str("ping timeout"),
            DisconnectReason::FrameTooLarge => f.write_str("inbound frame too large"),
            DisconnectReason::ServerError(code) => write!(f, "gateway error {code}"),
            DisconnectReason::Fatal(e) => write!(f, "cannot reconnect: {e}"),
            DisconnectReason::Requested => f.write_str("closed by caller"),
        }
    }
}

/// What a chat session reports; turn events carry the turn number, `segment` and `index` order rows inside a turn.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The socket is open and the gateway named the session. `reconnect` is true after an automatic reconnect.
    Connected { session_id: String, resumed: bool, message_count: u64, name: Option<String>, reconnect: bool },
    /// The socket dropped. `will_retry` is false for the last event of a session.
    Disconnected { reason: DisconnectReason, will_retry: bool },
    /// A reconnect attempt is scheduled after `delay`; `last_error` is why the previous attempt failed.
    Reconnecting { attempt: u32, delay: Duration, last_error: Option<String> },
    TurnStarted { turn: u64 },
    /// Answer delta; `segment` counts the tool calls before it in this turn.
    Token { turn: u64, segment: u32, text: String },
    /// Reasoning delta.
    Thinking { turn: u64, segment: u32, text: String },
    ToolCall { turn: u64, index: u32, id: Option<String>, name: String, args: Value },
    ToolResult { turn: u64, index: u32, id: Option<String>, name: String, output: String },
    ApprovalRequest(ApprovalRequest),
    Plan { turn: u64, entries: Vec<PlanEntry> },
    HistoryTrimmed { dropped_messages: u64, kept_turns: u64, reason: String },
    /// The turn finished. `full_response` is authoritative over the summed tokens.
    Done { turn: u64, full_response: String, usage: Usage },
    Aborted { turn: u64 },
    /// A gateway error frame. `ends_turn` errors are followed by no `Done`.
    Error { turn: u64, code: Option<String>, message: String, ends_turn: bool },
    /// The gateway replays the running turn from its first frame; rebuild `turn` from the events that follow.
    TurnResumed { turn: u64, frames: usize },
    /// A turn open when the socket dropped ended while disconnected; reload history to see the outcome.
    TurnLost { turn: u64 },
    /// A frame this client does not model.
    Unknown { kind: String, raw: Value },
    /// Every frame that crossed the socket; emitted only when the raw tap is enabled.
    Raw(RawFrame),
}

impl Event {
    /// A flat JSON rendering for logs and `zcprobe --json`: `{"event": <kind>, ...fields}`.
    pub fn to_json(&self) -> Value {
        use serde_json::json;
        let mut v = match self {
            Event::Connected { session_id, resumed, message_count, name, reconnect } => json!({
                "session_id": session_id, "resumed": resumed, "message_count": message_count, "name": name, "reconnect": reconnect
            }),
            Event::Disconnected { reason, will_retry } => json!({"reason": reason.to_string(), "will_retry": will_retry}),
            Event::Reconnecting { attempt, delay, last_error } => {
                json!({"attempt": attempt, "delay_ms": delay.as_millis() as u64, "last_error": last_error})
            }
            Event::TurnStarted { turn } | Event::Aborted { turn } | Event::TurnLost { turn } => json!({"turn": turn}),
            Event::Token { turn, segment, text } | Event::Thinking { turn, segment, text } => {
                json!({"turn": turn, "segment": segment, "text": text})
            }
            Event::ToolCall { turn, index, id, name, args } => {
                json!({"turn": turn, "index": index, "id": id, "name": name, "args": args})
            }
            Event::ToolResult { turn, index, id, name, output } => {
                json!({"turn": turn, "index": index, "id": id, "name": name, "output": output})
            }
            Event::ApprovalRequest(a) => json!({
                "turn": a.turn, "request_id": a.request_id, "tool": a.tool, "summary": a.summary,
                "arguments": a.arguments, "timeout_secs": a.timeout_secs
            }),
            Event::Plan { turn, entries } => json!({
                "turn": turn,
                "entries": entries.iter().map(|e| json!({"content": e.content, "status": e.status})).collect::<Vec<_>>()
            }),
            Event::HistoryTrimmed { dropped_messages, kept_turns, reason } => {
                json!({"dropped_messages": dropped_messages, "kept_turns": kept_turns, "reason": reason})
            }
            Event::Done { turn, full_response, usage } => json!({
                "turn": turn, "full_response": full_response,
                "usage": {
                    "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens, "cost_usd": usage.cost_usd, "model": usage.model,
                    "provider": usage.provider, "context_window": usage.context_window,
                    "last_input_tokens": usage.last_input_tokens
                }
            }),
            Event::Error { turn, code, message, ends_turn } => {
                json!({"turn": turn, "code": code, "message": message, "ends_turn": ends_turn})
            }
            Event::TurnResumed { turn, frames } => json!({"turn": turn, "frames": frames}),
            Event::Unknown { kind, raw } => json!({"kind": kind, "raw": raw}),
            Event::Raw(r) => json!({
                "t_ms": r.t_ms,
                "dir": if r.dir == Direction::In { "in" } else { "out" },
                "frame": r.frame
            }),
        };
        if let Some(o) = v.as_object_mut() {
            o.insert("event".into(), Value::String(event_kind(self).into()));
        }
        v
    }
}

/// Maps gateway frames to [`Event`]s and tracks turn boundaries.
#[derive(Debug, Default)]
pub struct Translator {
    turn: u64,
    started: bool,
    tools: u32,
    calls: Vec<(String, u32)>,
}

impl Translator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of the current (or next) turn.
    pub fn turn(&self) -> u64 {
        self.turn
    }

    /// True between a turn's first frame and its terminal frame.
    pub fn turn_open(&self) -> bool {
        self.started
    }

    /// Ends a turn that finished while the socket was down.
    pub fn lose_turn(&mut self, out: &mut Vec<Event>) {
        if self.started {
            out.push(Event::TurnLost { turn: self.turn });
            self.end();
        }
    }

    fn begin(&mut self, out: &mut Vec<Event>) {
        if !self.started {
            self.started = true;
            out.push(Event::TurnStarted { turn: self.turn });
        }
    }

    fn end(&mut self) {
        self.turn += 1;
        self.started = false;
        self.tools = 0;
        self.calls.clear();
    }

    pub fn feed(&mut self, frame: ServerFrame, out: &mut Vec<Event>) {
        match frame {
            ServerFrame::SessionStart { .. }
            | ServerFrame::Connected { .. }
            | ServerFrame::AgentEnd { .. } => {}
            ServerFrame::AgentStart { .. } => self.begin(out),
            ServerFrame::Chunk { content } => {
                self.begin(out);
                out.push(Event::Token { turn: self.turn, segment: self.tools, text: content });
            }
            ServerFrame::Thinking { content } => {
                self.begin(out);
                out.push(Event::Thinking { turn: self.turn, segment: self.tools, text: content });
            }
            ServerFrame::ToolCall { id, name, args } => {
                self.begin(out);
                self.tools += 1;
                if let Some(id) = &id {
                    self.calls.push((id.clone(), self.tools));
                }
                out.push(Event::ToolCall { turn: self.turn, index: self.tools, id, name, args });
            }
            ServerFrame::ToolResult { id, name, output } => {
                self.begin(out);
                let index = id
                    .as_ref()
                    .and_then(|id| self.calls.iter().find(|(c, _)| c == id))
                    .map_or(self.tools, |(_, n)| *n);
                out.push(Event::ToolResult { turn: self.turn, index, id, name, output });
            }
            ServerFrame::ApprovalRequest { request_id, tool, arguments_summary, arguments, timeout_secs } => {
                self.begin(out);
                out.push(Event::ApprovalRequest(ApprovalRequest {
                    turn: self.turn,
                    request_id,
                    tool,
                    summary: arguments_summary,
                    arguments,
                    timeout_secs,
                }));
            }
            ServerFrame::Plan { entries } => {
                self.begin(out);
                out.push(Event::Plan { turn: self.turn, entries });
            }
            ServerFrame::HistoryTrimmed { dropped_messages, kept_turns, reason } => {
                out.push(Event::HistoryTrimmed { dropped_messages, kept_turns, reason });
            }
            ServerFrame::Done(done) => {
                self.begin(out);
                out.push(Event::Done {
                    turn: self.turn,
                    usage: Usage::from(&done),
                    full_response: done.full_response,
                });
                self.end();
            }
            ServerFrame::Aborted => {
                self.begin(out);
                out.push(Event::Aborted { turn: self.turn });
                self.end();
            }
            ServerFrame::Error { message, code } => {
                let ends_turn = code.as_deref().is_some_and(error_code_ends_turn);
                if ends_turn {
                    self.begin(out);
                }
                out.push(Event::Error { turn: self.turn, code, message, ends_turn });
                if ends_turn {
                    self.end();
                }
            }
            ServerFrame::TurnResume { frames } => {
                let nested = frames.len();
                self.started = true;
                self.tools = 0;
                self.calls.clear();
                out.push(Event::TurnResumed { turn: self.turn, frames: nested });
                for frame in frames {
                    self.feed(ServerFrame::from_value(frame), out);
                }
            }
            ServerFrame::Unknown(raw) => {
                let kind = raw.get("type").and_then(Value::as_str).unwrap_or("").to_string();
                out.push(Event::Unknown { kind, raw });
            }
        }
    }
}

/// Short stable name of an event variant, for logs and tests.
pub fn event_kind(e: &Event) -> &'static str {
    match e {
        Event::Connected { .. } => "connected",
        Event::Disconnected { .. } => "disconnected",
        Event::Reconnecting { .. } => "reconnecting",
        Event::TurnStarted { .. } => "turn_started",
        Event::Token { .. } => "token",
        Event::Thinking { .. } => "thinking",
        Event::ToolCall { .. } => "tool_call",
        Event::ToolResult { .. } => "tool_result",
        Event::ApprovalRequest(_) => "approval_request",
        Event::Plan { .. } => "plan",
        Event::HistoryTrimmed { .. } => "history_trimmed",
        Event::Done { .. } => "done",
        Event::Aborted { .. } => "aborted",
        Event::Error { .. } => "error",
        Event::TurnResumed { .. } => "turn_resumed",
        Event::TurnLost { .. } => "turn_lost",
        Event::Unknown { .. } => "unknown",
        Event::Raw(_) => "raw",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(frames: &[Value]) -> (Translator, Vec<Event>) {
        let mut t = Translator::new();
        let mut out = Vec::new();
        for f in frames {
            t.feed(ServerFrame::from_value(f.clone()), &mut out);
        }
        (t, out)
    }

    fn kinds(events: &[Event]) -> Vec<&'static str> {
        events.iter().map(event_kind).collect()
    }

    #[test]
    fn plain_turn_is_bracketed_and_numbered() {
        let (t, ev) = run(&[
            json!({"type":"session_start","session_id":"s"}),
            json!({"type":"connected","message":"ok"}),
            json!({"type":"thinking","content":"hm"}),
            json!({"type":"chunk","content":"Hel"}),
            json!({"type":"chunk","content":"lo"}),
            json!({"type":"done","full_response":"Hello","input_tokens":12,"max_context_tokens":120}),
        ]);
        assert_eq!(kinds(&ev), ["turn_started", "thinking", "token", "token", "done"]);
        assert!(matches!(ev[0], Event::TurnStarted { turn: 0 }));
        match &ev[4] {
            Event::Done { turn, full_response, usage } => {
                assert_eq!(*turn, 0);
                assert_eq!(full_response, "Hello");
                assert_eq!(usage.input_tokens, Some(12));
                assert_eq!(usage.context_fraction(), Some(0.1));
            }
            other => panic!("{other:?}"),
        }
        assert!(!t.turn_open());
        assert_eq!(t.turn(), 1);
    }

    #[test]
    fn second_turn_gets_the_next_number_and_a_fresh_start() {
        let (_, ev) = run(&[
            json!({"type":"chunk","content":"one"}),
            json!({"type":"done","full_response":"one"}),
            json!({"type":"chunk","content":"two"}),
            json!({"type":"done","full_response":"two"}),
        ]);
        assert_eq!(kinds(&ev), ["turn_started", "token", "done", "turn_started", "token", "done"]);
        assert!(matches!(ev[3], Event::TurnStarted { turn: 1 }));
        assert!(matches!(ev[4], Event::Token { turn: 1, .. }));
    }

    #[test]
    fn tool_calls_advance_the_segment_and_pair_results_by_id() {
        let (_, ev) = run(&[
            json!({"type":"chunk","content":"a"}),
            json!({"type":"tool_call","id":"c1","name":"search","args":{"q":1}}),
            json!({"type":"tool_call","id":"c2","name":"search","args":{"q":2}}),
            json!({"type":"tool_result","id":"c1","name":"search","output":"r1"}),
            json!({"type":"chunk","content":"b"}),
        ]);
        let segs: Vec<u32> = ev.iter().filter_map(|e| if let Event::Token { segment, .. } = e { Some(*segment) } else { None }).collect();
        assert_eq!(segs, [0, 2]);
        match &ev[4] {
            Event::ToolResult { index, id, .. } => {
                assert_eq!(*index, 1, "the result belongs to the first call");
                assert_eq!(id.as_deref(), Some("c1"));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(ev[2], Event::ToolCall { index: 1, .. }));
        assert!(matches!(ev[3], Event::ToolCall { index: 2, .. }));
    }

    #[test]
    fn approval_request_becomes_a_typed_event() {
        let (_, ev) = run(&[json!({"type":"approval_request","request_id":"r1","tool":"aimedia__image_edit_run","arguments_summary":"edit 1 image","arguments":{"n":1},"timeout_secs":120})]);
        match &ev[1] {
            Event::ApprovalRequest(a) => {
                assert_eq!(a.request_id, "r1");
                assert_eq!(a.tool, "aimedia__image_edit_run");
                assert_eq!(a.summary, "edit 1 image");
                assert_eq!(a.timeout_secs, 120);
                assert_eq!(a.arguments, json!({"n":1}));
                assert_eq!(a.turn, 0);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn only_listed_error_codes_end_the_turn() {
        let (t, ev) = run(&[
            json!({"type":"chunk","content":"x"}),
            json!({"type":"error","code":"EMPTY_CONTENT","message":"empty"}),
        ]);
        assert!(t.turn_open());
        assert!(matches!(&ev[2], Event::Error { ends_turn: false, .. }));
        let (t, ev) = run(&[
            json!({"type":"chunk","content":"x"}),
            json!({"type":"error","code":"PROVIDER_ERROR","message":"boom"}),
        ]);
        assert!(!t.turn_open());
        assert!(matches!(&ev[2], Event::Error { ends_turn: true, turn: 0, .. }));
    }

    #[test]
    fn error_outside_a_turn_does_not_open_one() {
        let (t, ev) = run(&[json!({"type":"error","code":"INVALID_JSON","message":"bad"})]);
        assert_eq!(kinds(&ev), ["error"]);
        assert!(!t.turn_open());
        assert_eq!(t.turn(), 0);
    }

    #[test]
    fn aborted_ends_the_turn() {
        let (t, ev) = run(&[json!({"type":"chunk","content":"x"}), json!({"type":"aborted"})]);
        assert_eq!(kinds(&ev), ["turn_started", "token", "aborted"]);
        assert!(!t.turn_open());
        assert_eq!(t.turn(), 1);
    }

    #[test]
    fn done_without_prior_frames_is_still_bracketed() {
        let (_, ev) = run(&[json!({"type":"done","full_response":"quick"})]);
        assert_eq!(kinds(&ev), ["turn_started", "done"]);
    }

    #[test]
    fn turn_resume_rewinds_the_open_turn_and_replays_its_frames() {
        let mut t = Translator::new();
        let mut out = Vec::new();
        for f in [
            json!({"type":"chunk","content":"Hel"}),
            json!({"type":"tool_call","id":"c1","name":"t","args":{}}),
        ] {
            t.feed(ServerFrame::from_value(f), &mut out);
        }
        out.clear();
        t.feed(
            ServerFrame::from_value(json!({"type":"turn_resume","frames":[
                {"type":"chunk","content":"Hello "},
                {"type":"tool_call","id":"c1","name":"t","args":{}},
                {"type":"tool_result","id":"c1","name":"t","output":"ok"}
            ]})),
            &mut out,
        );
        assert_eq!(kinds(&out), ["turn_resumed", "token", "tool_call", "tool_result"]);
        assert!(matches!(out[0], Event::TurnResumed { turn: 0, frames: 3 }));
        assert!(matches!(out[2], Event::ToolCall { turn: 0, index: 1, .. }), "numbering restarts with the replay");
        t.feed(ServerFrame::from_value(json!({"type":"chunk","content":"x"})), &mut out);
        assert!(matches!(out.last(), Some(Event::Token { turn: 0, segment: 1, .. })));
        assert!(t.turn_open());
    }

    #[test]
    fn turn_resume_with_no_prior_state_still_opens_the_turn() {
        let (t, ev) = run(&[json!({"type":"turn_resume","frames":[]})]);
        assert_eq!(kinds(&ev), ["turn_resumed"]);
        assert!(t.turn_open());
    }

    #[test]
    fn lose_turn_reports_once_and_closes_the_turn() {
        let mut t = Translator::new();
        let mut out = Vec::new();
        t.lose_turn(&mut out);
        assert!(out.is_empty());
        t.feed(ServerFrame::from_value(json!({"type":"chunk","content":"x"})), &mut out);
        out.clear();
        t.lose_turn(&mut out);
        assert_eq!(kinds(&out), ["turn_lost"]);
        assert!(!t.turn_open());
        t.lose_turn(&mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn unknown_frames_and_history_trimmed_do_not_open_a_turn() {
        let (t, ev) = run(&[
            json!({"type":"cron_result","job":"j"}),
            json!({"type":"history_trimmed","dropped_messages":4,"kept_turns":2,"reason":"limit"}),
        ]);
        assert_eq!(kinds(&ev), ["unknown", "history_trimmed"]);
        assert!(matches!(&ev[0], Event::Unknown { kind, .. } if kind == "cron_result"));
        assert!(!t.turn_open());
    }

    #[test]
    fn plan_updates_belong_to_the_turn() {
        let (_, ev) = run(&[json!({"type":"plan","entries":[{"content":"a","status":"in_progress"}]})]);
        assert_eq!(kinds(&ev), ["turn_started", "plan"]);
    }

    #[test]
    fn context_fraction_needs_both_numbers() {
        assert_eq!(Usage::default().context_fraction(), None);
        let u = Usage { last_input_tokens: Some(50), context_window: Some(100), input_tokens: Some(500), ..Default::default() };
        assert_eq!(u.context_fraction(), Some(0.5));
        let over = Usage { input_tokens: Some(500), context_window: Some(100), ..Default::default() };
        assert_eq!(over.context_fraction(), Some(1.0));
        assert_eq!(Usage { input_tokens: Some(5), context_window: Some(0), ..Default::default() }.context_fraction(), None);
    }

    #[test]
    fn to_json_tags_the_event_kind() {
        let v = Event::Token { turn: 2, segment: 1, text: "hi".into() }.to_json();
        assert_eq!(v, json!({"event":"token","turn":2,"segment":1,"text":"hi"}));
        let v = Event::Disconnected { reason: DisconnectReason::PingTimeout, will_retry: true }.to_json();
        assert_eq!(v["event"], "disconnected");
        assert_eq!(v["reason"], "ping timeout");
        let v = Event::Reconnecting { attempt: 2, delay: Duration::from_millis(250), last_error: None }.to_json();
        assert_eq!(v["delay_ms"], 250);
        let (_, ev) = run(&[json!({"type":"done","full_response":"ok","model":"m"})]);
        let done = ev.last().unwrap().to_json();
        assert_eq!(done["usage"]["model"], "m");
        assert_eq!(done["event"], "done");
    }

    #[test]
    fn disconnect_reasons_display_without_secrets() {
        assert_eq!(DisconnectReason::PingTimeout.to_string(), "ping timeout");
        assert!(DisconnectReason::Remote { code: Some(1011), reason: "x".into() }.to_string().contains("1011"));
    }
}
