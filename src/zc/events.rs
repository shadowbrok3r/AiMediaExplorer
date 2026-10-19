//! What the chat runtime reports to the interface: events tagged with the session and connection they belong to.

use std::time::Instant;

use serde_json::Value;
use zeroclaw_client::{ApprovalRequest, HistoryItem, Usage};

/// Where the chat socket is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkState {
    /// No session is open.
    Idle,
    Connecting,
    Open,
    Reconnecting,
    Closed,
    Failed,
}

impl LinkState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Connecting => "connecting",
            Self::Open => "open",
            Self::Reconnecting => "reconnecting",
            Self::Closed => "closed",
            Self::Failed => "failed",
        }
    }
}

/// One plan row of the agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanRow {
    pub content: String,
    pub status: Option<String>,
}

/// What happened on one chat connection.
#[derive(Clone, Debug)]
pub enum ChatEv {
    Link { state: LinkState, detail: String },
    Connected { resumed: bool, message_count: u64, name: Option<String>, reconnect: bool },
    TurnStarted { turn: u64 },
    Text { turn: u64, segment: u32, text: String },
    Thinking { turn: u64, segment: u32, text: String },
    ToolCall { turn: u64, index: u32, id: Option<String>, name: String, args: Value },
    ToolResult { turn: u64, index: u32, id: Option<String>, name: String, output: String },
    Approval(ApprovalRequest),
    Plan { turn: u64, rows: Vec<PlanRow> },
    HistoryTrimmed { dropped: u64, kept: u64, reason: String },
    Done { turn: u64, full_response: String, usage: Usage },
    Aborted { turn: u64 },
    Error { turn: u64, code: Option<String>, message: String, ends_turn: bool },
    /// The gateway replays the running turn; what was drawn for `turn` is rebuilt from the events that follow.
    TurnResumed { turn: u64 },
    /// A turn ended while the socket was down; the stored transcript has its outcome.
    TurnLost { turn: u64 },
    /// A message could not be sent; `text` is the draft to restore.
    SendFailed { text: String, error: String },
    /// A page of the stored transcript, oldest first.
    History { items: Vec<HistoryItem>, has_more: bool, total: usize, initial: bool },
    /// The gateway refused the token; the person has to pair again.
    AuthRefused,
    Notice(String),
}

/// An event with the session and connection it belongs to; the interface drops events of any other.
#[derive(Clone, Debug)]
pub struct Tagged {
    pub session: String,
    /// Connection number, counted by the runtime for every `connect`.
    pub epoch: u64,
    pub ev: ChatEv,
}

/// One entry of the connection state history.
#[derive(Clone, Debug)]
pub struct LinkStep {
    pub at_ms: u64,
    pub state: LinkState,
    pub detail: String,
}

/// The chat socket as the runtime last saw it.
#[derive(Clone, Debug)]
pub struct Link {
    pub state: LinkState,
    pub session: Option<String>,
    pub epoch: u64,
    pub since: Instant,
    pub detail: String,
    /// Recent transitions, oldest first.
    pub steps: Vec<LinkStep>,
}

impl Default for Link {
    fn default() -> Self {
        Self { state: LinkState::Idle, session: None, epoch: 0, since: Instant::now(), detail: String::new(), steps: Vec::new() }
    }
}

/// Transitions kept for the Devtools window.
const STEP_CAP: usize = 60;

impl Link {
    pub fn set(&mut self, state: LinkState, detail: impl Into<String>) {
        let detail = detail.into();
        if self.state != state || self.detail != detail {
            if self.state != state {
                self.since = Instant::now();
            }
            self.steps.push(LinkStep { at_ms: now_ms(), state, detail: detail.clone() });
            if self.steps.len() > STEP_CAP {
                self.steps.remove(0);
            }
        }
        self.state = state;
        self.detail = detail;
    }
}

/// Unix time in milliseconds.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Token and cost figures summed over the finished turns of this run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageTotals {
    pub turns: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    /// Per model: turns, input, output, cost.
    pub by_model: Vec<(String, u64, u64, u64, f64)>,
    pub last: Option<Usage>,
}

impl UsageTotals {
    pub fn add(&mut self, usage: &Usage) {
        self.turns += 1;
        let (input, output, cost) = (usage.input_tokens.unwrap_or(0), usage.output_tokens.unwrap_or(0), usage.cost_usd.unwrap_or(0.0));
        self.input_tokens += input;
        self.output_tokens += output;
        self.cost_usd += cost;
        let model = usage.model.clone().unwrap_or_else(|| "unknown".to_string());
        match self.by_model.iter_mut().find(|row| row.0 == model) {
            Some(row) => {
                row.1 += 1;
                row.2 += input;
                row.3 += output;
                row.4 += cost;
            }
            None => self.by_model.push((model, 1, input, output, cost)),
        }
        self.last = Some(usage.clone());
    }
}

/// A one-shot job running in the background; poll it from the frame loop.
pub struct Pending<T> {
    rx: crossbeam::channel::Receiver<T>,
    done: Option<T>,
}

impl<T> Pending<T> {
    pub fn new(rx: crossbeam::channel::Receiver<T>) -> Self {
        Self { rx, done: None }
    }

    /// The result once it has arrived.
    pub fn poll(&mut self) -> Option<&T> {
        if self.done.is_none() {
            self.done = self.rx.try_recv().ok();
        }
        self.done.as_ref()
    }

    pub fn is_done(&mut self) -> bool {
        self.poll().is_some()
    }

    /// Takes the result when it has arrived.
    pub fn take(&mut self) -> Option<T> {
        self.poll();
        self.done.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_link_keeps_a_bounded_history_of_changes_only() {
        let mut link = Link::default();
        link.set(LinkState::Connecting, "x");
        link.set(LinkState::Connecting, "x");
        link.set(LinkState::Open, "");
        assert_eq!(link.steps.len(), 2);
        for i in 0..100 {
            link.set(LinkState::Reconnecting, format!("attempt {i}"));
        }
        assert_eq!(link.steps.len(), STEP_CAP);
        assert_eq!(link.state, LinkState::Reconnecting);
    }

    #[test]
    fn usage_totals_add_up_per_model() {
        let mut totals = UsageTotals::default();
        let usage = |model: &str, i: u64, o: u64, c: f64| Usage { input_tokens: Some(i), output_tokens: Some(o), cost_usd: Some(c), model: Some(model.into()), ..Default::default() };
        totals.add(&usage("a", 10, 5, 0.5));
        totals.add(&usage("a", 1, 2, 0.25));
        totals.add(&usage("b", 3, 3, 0.0));
        totals.add(&Usage::default());
        assert_eq!((totals.turns, totals.input_tokens, totals.output_tokens), (4, 14, 10));
        assert!((totals.cost_usd - 0.75).abs() < 1e-9);
        assert_eq!(totals.by_model[0], ("a".to_string(), 2, 11, 7, 0.75));
        assert_eq!(totals.by_model.len(), 3);
    }

    #[test]
    fn a_pending_job_reports_once_and_can_be_taken() {
        let (tx, rx) = crossbeam::channel::bounded(1);
        let mut job = Pending::new(rx);
        assert!(job.poll().is_none());
        tx.send(7).unwrap();
        assert_eq!(job.poll(), Some(&7));
        assert!(job.is_done());
        assert_eq!(job.take(), Some(7));
        assert!(job.take().is_none());
    }
}
