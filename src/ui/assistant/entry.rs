//! The transcript model: typed entries, the mapping from chat events and stored rows onto them, and the rows a finished turn leaves in the cache.

use serde_json::{Value, json};
use zeroclaw_client::{HistoryItem, HistoryKind, ImageRef, Usage};

use crate::database::assistant_chat::AssistantMessage;
use crate::database::assistant_store::NewRow;
use crate::zc::events::{ChatEv, PlanRow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

/// What the button of a notice does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeAction {
    RePair,
    OpenSettings,
    Retry,
    ReloadHistory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallState {
    Running,
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Pending,
    Approved,
    Always,
    Denied,
    /// The time ran out; the gateway denied on its own.
    Expired,
}

impl Decision {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "waiting",
            Self::Approved => "approved",
            Self::Always => "always approved",
            Self::Denied => "denied",
            Self::Expired => "expired",
        }
    }

    fn parse(text: &str) -> Self {
        match text {
            "approved" => Self::Approved,
            "always" => Self::Always,
            "denied" => Self::Denied,
            _ => Self::Expired,
        }
    }

    fn stored(self) -> &'static str {
        match self {
            Self::Approved => "approved",
            Self::Always => "always",
            Self::Denied => "denied",
            Self::Pending | Self::Expired => "expired",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    User { text: String, attachments: Vec<String>, chips: Vec<String> },
    Assistant { text: String, streaming: bool },
    Reasoning { text: String, streaming: bool },
    ToolCall { id: Option<String>, name: String, args: Value, state: CallState },
    ToolResult { id: Option<String>, name: String, output: String },
    Approval { request_id: String, tool: String, summary: String, args: Value, deadline_ms: u64, decision: Decision },
    Plan { rows: Vec<PlanRow> },
    Notice { level: Level, text: String, action: Option<NoticeAction> },
}

impl Kind {
    /// The `kind` column of a stored row.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::Reasoning { .. } => "reasoning",
            Self::ToolCall { .. } => "tool_call",
            Self::ToolResult { .. } => "tool_result",
            Self::Approval { .. } => "approval",
            Self::Plan { .. } => "plan",
            Self::Notice { .. } => "notice",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// `session:epoch:turn:...`; unique within a session.
    pub key: String,
    pub turn: Option<u64>,
    pub at_ms: u64,
    /// Already saved to (or loaded from) the cache.
    pub persisted: bool,
    pub kind: Kind,
}

impl Entry {
    fn new(key: String, turn: Option<u64>, at_ms: u64, kind: Kind) -> Self {
        Self { key, turn, at_ms, persisted: false, kind }
    }

    /// The text a copy button copies, when the entry has text.
    pub fn text(&self) -> Option<&str> {
        match &self.kind {
            Kind::User { text, .. } | Kind::Assistant { text, .. } | Kind::Reasoning { text, .. } | Kind::Notice { text, .. } => Some(text),
            Kind::ToolResult { output, .. } => Some(output),
            _ => None,
        }
    }
}

/// What applying an event asks the panel to do besides redrawing.
#[derive(Debug, Default)]
pub struct Applied {
    /// Rows to save to the cache.
    pub persist: Vec<NewRow>,
    /// A draft to put back in the composer after a failed send.
    pub restore_draft: Option<String>,
    pub reload_history: bool,
    /// A turn finished normally with this answer.
    pub finished: Option<String>,
    pub auth_refused: bool,
}

#[derive(Debug, Default)]
pub struct Transcript {
    pub entries: Vec<Entry>,
    /// Usage of the last finished turn.
    pub usage: Option<Usage>,
    /// A turn is running.
    pub running: bool,
    pub has_more: bool,
    pub total: usize,
}

fn tool_label(name: &str) -> &str {
    name.rsplit("__").next().unwrap_or(name)
}

impl Transcript {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn find(&mut self, key: &str) -> Option<&mut Entry> {
        self.entries.iter_mut().rev().find(|e| e.key == key)
    }

    fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    /// Adds what the person typed; the returned row is the one to save.
    pub fn push_user(&mut self, key: String, text: &str, attachments: Vec<String>, chips: Vec<String>, at_ms: u64) -> NewRow {
        let mut entry = Entry::new(key, None, at_ms, Kind::User { text: text.to_string(), attachments, chips });
        entry.persisted = true;
        let row = to_row(&entry).unwrap_or_default();
        self.push(entry);
        self.running = true;
        row
    }

    /// Adds a notice that belongs to no turn and is not saved.
    pub fn notice(&mut self, key: String, level: Level, text: impl Into<String>, action: Option<NoticeAction>, at_ms: u64) {
        let text = text.into();
        if self.entries.last().is_some_and(|e| matches!(&e.kind, Kind::Notice { text: t, .. } if *t == text)) {
            return;
        }
        let mut entry = Entry::new(key, None, at_ms, Kind::Notice { level, text, action });
        entry.persisted = true;
        self.push(entry);
    }

    /// Marks the entries of `turn` that are still streaming as finished.
    fn settle(&mut self, turn: u64) {
        for e in self.entries.iter_mut().filter(|e| e.turn == Some(turn)) {
            match &mut e.kind {
                Kind::Assistant { streaming, .. } | Kind::Reasoning { streaming, .. } => *streaming = false,
                Kind::ToolCall { state, .. } => *state = CallState::Done,
                Kind::Approval { decision, .. } if *decision == Decision::Pending => *decision = Decision::Expired,
                _ => {}
            }
        }
    }

    /// The rows of the turn's entries not saved yet, marking them saved.
    fn drain_turn(&mut self, turn: u64) -> Vec<NewRow> {
        let mut rows = Vec::new();
        for e in self.entries.iter_mut().filter(|e| e.turn == Some(turn) && !e.persisted) {
            e.persisted = true;
            if let Some(row) = to_row(e) {
                rows.push(row);
            }
        }
        rows
    }

    /// Makes the gateway's full response the text of the turn's answer.
    fn finish_answer(&mut self, scope: &str, turn: u64, full: &str, now_ms: u64) {
        if full.trim().is_empty() {
            return;
        }
        let idxs: Vec<usize> = self.entries.iter().enumerate().filter(|(_, e)| e.turn == Some(turn) && matches!(e.kind, Kind::Assistant { .. })).map(|(i, _)| i).collect();
        let text_at = |entries: &[Entry], i: usize| match &entries[i].kind {
            Kind::Assistant { text, .. } => text.clone(),
            _ => String::new(),
        };
        let set = |entries: &mut Vec<Entry>, i: usize| {
            if let Kind::Assistant { text, .. } = &mut entries[i].kind {
                *text = full.to_string();
            }
        };
        match idxs.as_slice() {
            [] => self.push(Entry::new(format!("{scope}:{turn}:a0"), Some(turn), now_ms, Kind::Assistant { text: full.to_string(), streaming: false })),
            [only] => set(&mut self.entries, *only),
            [first, .., last] => {
                let joined: String = idxs.iter().map(|i| text_at(&self.entries, *i)).collect();
                if joined == full {
                    return;
                }
                if full.contains(text_at(&self.entries, *first).trim()) {
                    set(&mut self.entries, *first);
                    for i in idxs.iter().skip(1).rev() {
                        self.entries.remove(*i);
                    }
                } else {
                    set(&mut self.entries, *last);
                }
            }
        }
    }

    /// Applies one event of the connection `scope` (`session:epoch`) at `now_ms`.
    pub fn apply(&mut self, scope: &str, ev: &ChatEv, now_ms: u64) -> Applied {
        let mut out = Applied::default();
        match ev {
            ChatEv::Link { .. } | ChatEv::Connected { .. } => {}
            ChatEv::TurnStarted { .. } => self.running = true,
            ChatEv::Text { turn, segment, text } => {
                self.running = true;
                let key = format!("{scope}:{turn}:a{segment}");
                match self.find(&key) {
                    Some(Entry { kind: Kind::Assistant { text: t, .. }, .. }) => t.push_str(text),
                    _ => self.push(Entry::new(key, Some(*turn), now_ms, Kind::Assistant { text: text.clone(), streaming: true })),
                }
            }
            ChatEv::Thinking { turn, segment, text } => {
                self.running = true;
                let key = format!("{scope}:{turn}:r{segment}");
                match self.find(&key) {
                    Some(Entry { kind: Kind::Reasoning { text: t, .. }, .. }) => t.push_str(text),
                    _ => self.push(Entry::new(key, Some(*turn), now_ms, Kind::Reasoning { text: text.clone(), streaming: true })),
                }
            }
            ChatEv::ToolCall { turn, index, id, name, args } => {
                self.running = true;
                let kind = Kind::ToolCall { id: id.clone(), name: name.clone(), args: args.clone(), state: CallState::Running };
                self.push(Entry::new(format!("{scope}:{turn}:c{index}"), Some(*turn), now_ms, kind));
            }
            ChatEv::ToolResult { turn, index, id, name, output } => {
                if let Some(Entry { kind: Kind::ToolCall { state, .. }, .. }) = self.find(&format!("{scope}:{turn}:c{index}")) {
                    *state = CallState::Done;
                }
                let kind = Kind::ToolResult { id: id.clone(), name: name.clone(), output: output.clone() };
                self.push(Entry::new(format!("{scope}:{turn}:t{index}"), Some(*turn), now_ms, kind));
            }
            ChatEv::Approval(req) => {
                let kind = Kind::Approval {
                    request_id: req.request_id.clone(),
                    tool: req.tool.clone(),
                    summary: req.summary.clone(),
                    args: req.arguments.clone(),
                    deadline_ms: now_ms + req.timeout_secs * 1000,
                    decision: Decision::Pending,
                };
                self.push(Entry::new(format!("{scope}:{}:p{}", req.turn, req.request_id), Some(req.turn), now_ms, kind));
            }
            ChatEv::Plan { turn, rows } => {
                let key = format!("{scope}:{turn}:plan");
                match self.find(&key) {
                    Some(Entry { kind: Kind::Plan { rows: r }, .. }) => *r = rows.clone(),
                    _ => self.push(Entry::new(key, Some(*turn), now_ms, Kind::Plan { rows: rows.clone() })),
                }
            }
            ChatEv::HistoryTrimmed { dropped, kept, reason } => {
                let why = if reason.is_empty() { String::new() } else { format!(" ({reason})") };
                self.notice(format!("{scope}:trim:{now_ms}"), Level::Info, format!("The gateway dropped {dropped} older messages and kept {kept} turns{why}."), None, now_ms);
            }
            ChatEv::Done { turn, full_response, usage } => {
                self.running = false;
                self.finish_answer(scope, *turn, full_response, now_ms);
                self.settle(*turn);
                out.persist = self.drain_turn(*turn);
                self.usage = Some(usage.clone());
                out.finished = Some(full_response.clone());
            }
            ChatEv::Aborted { turn } => {
                self.running = false;
                self.settle(*turn);
                self.push(Entry::new(format!("{scope}:{turn}:stopped"), Some(*turn), now_ms, Kind::Notice { level: Level::Info, text: "Stopped.".into(), action: None }));
                out.persist = self.drain_turn(*turn);
            }
            ChatEv::Error { turn, code, message, ends_turn } => {
                let code = code.as_deref().map(|c| format!("{c}: ")).unwrap_or_default();
                let kind = Kind::Notice { level: Level::Error, text: format!("{code}{message}"), action: None };
                if *ends_turn {
                    self.running = false;
                    self.settle(*turn);
                    self.push(Entry::new(format!("{scope}:{turn}:error"), Some(*turn), now_ms, kind));
                    out.persist = self.drain_turn(*turn);
                } else {
                    let mut entry = Entry::new(format!("{scope}:{turn}:error{now_ms}"), None, now_ms, kind);
                    entry.persisted = true;
                    self.push(entry);
                }
            }
            ChatEv::TurnResumed { turn } => {
                let prefix = format!("{scope}:{turn}:");
                self.entries.retain(|e| !e.key.starts_with(&prefix));
                self.running = true;
            }
            ChatEv::TurnLost { turn } => {
                self.running = false;
                let prefix = format!("{scope}:{turn}:");
                self.entries.retain(|e| !e.key.starts_with(&prefix));
                self.notice(format!("{scope}:{turn}:lost"), Level::Warn, "The reply finished while the connection was down. Loading it from the gateway.", Some(NoticeAction::ReloadHistory), now_ms);
                out.reload_history = true;
            }
            ChatEv::SendFailed { text, error } => {
                self.running = false;
                if matches!(self.entries.last(), Some(Entry { kind: Kind::User { text: t, .. }, .. }) if t == text) {
                    self.entries.pop();
                }
                self.notice(format!("{scope}:send:{now_ms}"), Level::Error, format!("The message was not sent: {error}"), None, now_ms);
                out.restore_draft = Some(text.clone());
            }
            ChatEv::AuthRefused => {
                self.running = false;
                self.notice(format!("{scope}:auth"), Level::Error, "The gateway refused the token. It is not retried; pair again.", Some(NoticeAction::RePair), now_ms);
                out.auth_refused = true;
            }
            ChatEv::Notice(text) => self.notice(format!("{scope}:note:{now_ms}"), Level::Info, text.clone(), None, now_ms),
            ChatEv::History { items, has_more, total, initial } => {
                let loaded: Vec<Entry> = items.iter().filter_map(|i| entry_from_history(scope, i)).collect();
                self.has_more = *has_more;
                self.total = *total;
                if *initial {
                    self.merge_history(loaded);
                } else {
                    let mut older = loaded;
                    older.append(&mut self.entries);
                    self.entries = older;
                }
            }
        }
        out
    }

    /// Joins the stored transcript of the gateway with the entries already shown: the gateway wins where it has history, older cached entries stay before it and the turn in progress stays after it.
    pub fn merge_history(&mut self, server: Vec<Entry>) {
        if server.is_empty() {
            return;
        }
        let live: Vec<Entry> = self.entries.iter().filter(|e| e.turn.is_some() && !e.persisted).cloned().collect();
        let cutoff = server.first().map(|e| e.at_ms).filter(|t| *t > 0);
        let older: Vec<Entry> = match cutoff {
            Some(cutoff) => self.entries.iter().filter(|e| e.persisted && e.at_ms > 0 && e.at_ms < cutoff).cloned().collect(),
            None => Vec::new(),
        };
        self.entries = older.into_iter().chain(server).chain(live).collect();
    }

    /// Marks approvals whose time ran out; true when any changed.
    pub fn expire_approvals(&mut self, now_ms: u64) -> bool {
        let mut changed = false;
        for e in &mut self.entries {
            if let Kind::Approval { decision, deadline_ms, .. } = &mut e.kind {
                if *decision == Decision::Pending && now_ms >= *deadline_ms {
                    *decision = Decision::Expired;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Records the answer given to an approval card.
    pub fn decide(&mut self, request_id: &str, answer: Decision) {
        for e in &mut self.entries {
            if let Kind::Approval { request_id: id, decision, .. } = &mut e.kind {
                if id == request_id && *decision == Decision::Pending {
                    *decision = answer;
                    e.persisted = false;
                }
            }
        }
    }

    pub fn pending_approvals(&self) -> usize {
        self.entries.iter().filter(|e| matches!(e.kind, Kind::Approval { decision: Decision::Pending, .. })).count()
    }

    /// The text turns before the last user message, for providers that need the conversation: oldest first, as (is_user, text).
    pub fn conversation(&self, max_chars: usize) -> Vec<(bool, String)> {
        let mut turns: Vec<(bool, String)> = self
            .entries
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::User { text, .. } => Some((true, text.clone())),
                Kind::Assistant { text, .. } if !text.trim().is_empty() => Some((false, text.clone())),
                _ => None,
            })
            .collect();
        let mut total = 0usize;
        let mut keep = turns.len();
        for (i, (_, text)) in turns.iter().enumerate().rev() {
            total += text.chars().count();
            if total > max_chars {
                break;
            }
            keep = i;
        }
        turns.drain(..keep);
        turns
    }

    pub fn last_user_text(&self) -> Option<&str> {
        self.entries.iter().rev().find_map(|e| match &e.kind {
            Kind::User { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Text-bearing entries of the first exchange, for naming the session.
    pub fn first_exchange(&self) -> Option<(String, String)> {
        let user = self.entries.iter().find_map(|e| match &e.kind {
            Kind::User { text, .. } => Some(text.clone()),
            _ => None,
        })?;
        let answer = self.entries.iter().find_map(|e| match &e.kind {
            Kind::Assistant { text, .. } => Some(text.clone()),
            _ => None,
        })?;
        Some((user, answer))
    }
}

/// An entry for a stored gateway message; `None` for roles the transcript does not show.
pub fn entry_from_history(scope: &str, item: &HistoryItem) -> Option<Entry> {
    let at_ms = item.created_at.as_deref().and_then(zeroclaw_client::text::epoch_utc).map_or(0, |s| (s.max(0) as u64) * 1000);
    let key = format!("{scope}:{}", item.id);
    let kind = match item.kind {
        HistoryKind::User => Kind::User { text: with_image_notes(&item.text, &item.images), attachments: Vec::new(), chips: Vec::new() },
        HistoryKind::Assistant => Kind::Assistant { text: with_image_notes(&item.text, &item.images), streaming: false },
        HistoryKind::Reasoning => Kind::Reasoning { text: item.text.clone(), streaming: false },
        HistoryKind::ToolCall => Kind::ToolCall { id: item.call_id.clone(), name: item.name.clone().unwrap_or_default(), args: item.args.clone().unwrap_or(Value::Null), state: CallState::Done },
        HistoryKind::ToolResult => Kind::ToolResult { id: item.call_id.clone(), name: item.name.clone().unwrap_or_default(), output: item.text.clone() },
        HistoryKind::Other => return None,
    };
    let mut entry = Entry::new(key, None, at_ms, kind);
    entry.persisted = true;
    Some(entry)
}

fn with_image_notes(text: &str, images: &[ImageRef]) -> String {
    if images.is_empty() {
        return text.to_string();
    }
    let note = format!("[{} image{}]", images.len(), if images.len() == 1 { "" } else { "s" });
    if text.trim().is_empty() { note } else { format!("{text}\n{note}") }
}

/// The row that stores `entry` in the cache; entries that are not worth keeping give `None`.
pub fn to_row(entry: &Entry) -> Option<NewRow> {
    let role = match entry.kind {
        Kind::User { .. } => "user",
        _ => "assistant",
    };
    let (content, meta, attachments) = match &entry.kind {
        Kind::User { text, attachments, chips } => (text.clone(), json!({ "chips": chips }), (!attachments.is_empty()).then(|| attachments.clone())),
        Kind::Assistant { text, .. } | Kind::Reasoning { text, .. } => {
            if text.trim().is_empty() {
                return None;
            }
            (text.clone(), Value::Null, None)
        }
        Kind::ToolCall { id, name, args, .. } => (name.clone(), json!({ "id": id, "args": args }), None),
        Kind::ToolResult { id, name, output } => (output.clone(), json!({ "id": id, "name": name }), None),
        Kind::Approval { request_id, tool, summary, args, decision, .. } => (summary.clone(), json!({ "request_id": request_id, "tool": tool, "args": args, "decision": decision.stored() }), None),
        Kind::Plan { rows } => (rows.iter().map(|r| r.content.as_str()).collect::<Vec<_>>().join("\n"), json!({ "rows": rows.iter().map(|r| json!({ "content": r.content, "status": r.status })).collect::<Vec<_>>() }), None),
        Kind::Notice { level, text, action } => (text.clone(), json!({ "level": level_text(*level), "action": action.map(action_text) }), None),
    };
    Some(NewRow {
        role: role.to_string(),
        kind: entry.kind.tag().to_string(),
        content,
        turn: entry.turn.map(|t| t as i64),
        meta: (!meta.is_null()).then(|| meta.to_string()),
        attachments,
    })
}

fn level_text(level: Level) -> &'static str {
    match level {
        Level::Info => "info",
        Level::Warn => "warn",
        Level::Error => "error",
    }
}

fn action_text(action: NoticeAction) -> &'static str {
    match action {
        NoticeAction::RePair => "re_pair",
        NoticeAction::OpenSettings => "settings",
        NoticeAction::Retry => "retry",
        NoticeAction::ReloadHistory => "reload",
    }
}

fn action_from(text: &str) -> Option<NoticeAction> {
    match text {
        "re_pair" => Some(NoticeAction::RePair),
        "settings" => Some(NoticeAction::OpenSettings),
        "retry" => Some(NoticeAction::Retry),
        "reload" => Some(NoticeAction::ReloadHistory),
        _ => None,
    }
}

/// The entry a stored row stands for. Rows from before kinds existed read as plain messages.
pub fn entry_from_row(scope: &str, index: usize, row: &AssistantMessage) -> Entry {
    let meta: Value = row.meta.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(Value::Null);
    let at_ms = row.created.as_ref().map_or(0, |d| chrono::DateTime::<chrono::Utc>::from(d.clone()).timestamp_millis().max(0) as u64);
    let text = row.content.clone();
    let strings = |key: &str| meta.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect::<Vec<_>>()).unwrap_or_default();
    let id = || meta.get("id").and_then(Value::as_str).map(str::to_string);
    let kind = match row.kind.as_deref().unwrap_or(if row.role == "user" { "user" } else { "assistant" }) {
        "user" => Kind::User { text, attachments: row.attachments.clone().unwrap_or_default(), chips: strings("chips") },
        "reasoning" => Kind::Reasoning { text, streaming: false },
        "tool_call" => Kind::ToolCall { id: id(), name: text, args: meta.get("args").cloned().unwrap_or(Value::Null), state: CallState::Done },
        "tool_result" => Kind::ToolResult { id: id(), name: meta.get("name").and_then(Value::as_str).unwrap_or("").to_string(), output: text },
        "approval" => Kind::Approval {
            request_id: meta.get("request_id").and_then(Value::as_str).unwrap_or("").to_string(),
            tool: meta.get("tool").and_then(Value::as_str).unwrap_or("").to_string(),
            summary: text,
            args: meta.get("args").cloned().unwrap_or(Value::Null),
            deadline_ms: 0,
            decision: Decision::parse(meta.get("decision").and_then(Value::as_str).unwrap_or("")),
        },
        "plan" => Kind::Plan {
            rows: meta
                .get("rows")
                .and_then(Value::as_array)
                .map(|rows| rows.iter().map(|r| PlanRow { content: r.get("content").and_then(Value::as_str).unwrap_or("").to_string(), status: r.get("status").and_then(Value::as_str).map(str::to_string) }).collect())
                .unwrap_or_default(),
        },
        "notice" => Kind::Notice {
            level: match meta.get("level").and_then(Value::as_str) {
                Some("error") => Level::Error,
                Some("warn") => Level::Warn,
                _ => Level::Info,
            },
            text,
            action: meta.get("action").and_then(Value::as_str).and_then(action_from),
        },
        _ => Kind::Assistant { text, streaming: false },
    };
    let mut entry = Entry::new(format!("{scope}:row{index}"), row.turn.map(|t| t as u64), at_ms, kind);
    entry.persisted = true;
    entry
}

/// Name of a tool for display: the part after the gateway's `server__` prefix.
pub fn tool_name(name: &str) -> &str {
    tool_label(name)
}

#[cfg(test)]
#[path = "entry_tests.rs"]
mod tests;
