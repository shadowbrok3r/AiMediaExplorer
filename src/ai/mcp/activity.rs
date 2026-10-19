//! Records of tool calls, connections and request counts for the Devtools window.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

const ACTIVITY_CAP: usize = 300;
const EVENT_CAP: usize = 300;
const ARGS_CAP: usize = 160;
const SESSION_CAP: usize = 64;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A string cut to `max` characters with "..." appended when it was longer.
pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
}

/// Single-line summary of tool arguments.
pub fn args_summary(args: &serde_json::Value) -> String {
    clip(&args.to_string().replace(['\n', '\r'], " "), ARGS_CAP)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActivityEntry {
    pub at_ms: u64,
    pub client: String,
    pub tool: String,
    pub args: String,
    pub ok: bool,
    pub error_kind: Option<String>,
    pub duration_ms: u64,
    pub result_bytes: usize,
}

/// The most recent tool calls.
#[derive(Default)]
pub struct ActivityLog {
    entries: Mutex<VecDeque<ActivityEntry>>,
    total: AtomicU64,
}

impl ActivityLog {
    pub fn push(&self, entry: ActivityEntry) {
        let mut entries = lock(&self.entries);
        entries.push_back(entry);
        while entries.len() > ACTIVITY_CAP {
            entries.pop_front();
        }
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Newest first.
    pub fn recent(&self, n: usize) -> Vec<ActivityEntry> {
        lock(&self.entries).iter().rev().take(n).cloned().collect()
    }

    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Opened,
    Closed,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConnectionEvent {
    pub at_ms: u64,
    pub kind: EventKind,
    pub remote: String,
    pub session: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClientInfo {
    pub session: String,
    pub remote: String,
    pub user_agent: String,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub requests: u64,
}

/// Client sessions seen by the guard and the events around them.
#[derive(Default)]
pub struct ConnectionLog {
    events: Mutex<VecDeque<ConnectionEvent>>,
    sessions: Mutex<HashMap<String, ClientInfo>>,
}

impl ConnectionLog {
    fn event(&self, at_ms: u64, kind: EventKind, remote: &str, session: Option<&str>, detail: String) {
        let mut events = lock(&self.events);
        events.push_back(ConnectionEvent { at_ms, kind, remote: remote.to_string(), session: session.map(str::to_string), detail });
        while events.len() > EVENT_CAP {
            events.pop_front();
        }
    }

    /// Notes an accepted request of `session`; returns true when the session is new.
    pub fn touch(&self, at_ms: u64, session: &str, remote: &str, user_agent: &str) -> bool {
        let mut sessions = lock(&self.sessions);
        let new = !sessions.contains_key(session);
        let info = sessions.entry(session.to_string()).or_insert_with(|| ClientInfo {
            session: session.to_string(),
            remote: remote.to_string(),
            user_agent: clip(user_agent, 80),
            first_seen_ms: at_ms,
            last_seen_ms: at_ms,
            requests: 0,
        });
        info.last_seen_ms = at_ms;
        info.requests += 1;
        if sessions.len() > SESSION_CAP {
            if let Some(oldest) = sessions.values().min_by_key(|c| c.last_seen_ms).map(|c| c.session.clone()) {
                sessions.remove(&oldest);
            }
        }
        drop(sessions);
        if new {
            self.event(at_ms, EventKind::Opened, remote, Some(session), format!("client session started ({})", clip(user_agent, 60)));
        }
        new
    }

    /// Forgets `session` after its client ended it.
    pub fn close(&self, at_ms: u64, session: &str, remote: &str) {
        if lock(&self.sessions).remove(session).is_some() {
            self.event(at_ms, EventKind::Closed, remote, Some(session), "client session ended".to_string());
        }
    }

    pub fn reject(&self, at_ms: u64, remote: &str, reason: &str) {
        self.event(at_ms, EventKind::Rejected, remote, None, reason.to_string());
    }

    /// Sessions that sent a request within `ttl_ms` of `now_ms`, newest first.
    pub fn clients(&self, now_ms: u64, ttl_ms: u64) -> Vec<ClientInfo> {
        let mut live: Vec<ClientInfo> = lock(&self.sessions).values().filter(|c| now_ms.saturating_sub(c.last_seen_ms) <= ttl_ms).cloned().collect();
        live.sort_by_key(|c| std::cmp::Reverse(c.last_seen_ms));
        live
    }

    /// Newest first.
    pub fn events(&self, n: usize) -> Vec<ConnectionEvent> {
        lock(&self.events).iter().rev().take(n).cloned().collect()
    }
}

/// Request counters of the running server.
#[derive(Default)]
pub struct ServerStats {
    pub requests: AtomicU64,
    pub rejected_auth: AtomicU64,
    pub rejected_host: AtomicU64,
    pub rate_limited: AtomicU64,
    pub tool_calls: AtomicU64,
    pub started_ms: AtomicU64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StatsSnapshot {
    pub requests: u64,
    pub rejected_auth: u64,
    pub rejected_host: u64,
    pub rate_limited: u64,
    pub tool_calls: u64,
    pub started_ms: u64,
}

impl ServerStats {
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset(&self, started_ms: u64) {
        for c in [&self.requests, &self.rejected_auth, &self.rejected_host, &self.rate_limited, &self.tool_calls] {
            c.store(0, Ordering::Relaxed);
        }
        self.started_ms.store(started_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        StatsSnapshot {
            requests: get(&self.requests),
            rejected_auth: get(&self.rejected_auth),
            rejected_host: get(&self.rejected_host),
            rate_limited: get(&self.rate_limited),
            tool_calls: get(&self.tool_calls),
            started_ms: get(&self.started_ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tool: &str) -> ActivityEntry {
        ActivityEntry { at_ms: 1, client: "c".into(), tool: tool.into(), args: "{}".into(), ok: true, error_kind: None, duration_ms: 2, result_bytes: 3 }
    }

    #[test]
    fn the_activity_log_keeps_the_newest_entries_first_and_counts_all() {
        let log = ActivityLog::default();
        for i in 0..(ACTIVITY_CAP + 10) {
            log.push(entry(&format!("t{i}")));
        }
        assert_eq!(log.total(), (ACTIVITY_CAP + 10) as u64);
        let recent = log.recent(3);
        assert_eq!(recent.iter().map(|e| e.tool.as_str()).collect::<Vec<_>>(), vec!["t309", "t308", "t307"]);
        assert_eq!(log.recent(10_000).len(), ACTIVITY_CAP);
    }

    #[test]
    fn long_arguments_are_cut_and_flattened() {
        let args = serde_json::json!({"text": "x".repeat(500), "n": 1});
        let s = args_summary(&args);
        assert_eq!(s.chars().count(), ARGS_CAP);
        assert!(s.ends_with("..."));
        assert_eq!(args_summary(&serde_json::json!({"a": "line1\nline2"})), r#"{"a":"line1\nline2"}"#);
        assert_eq!(clip("short", 10), "short");
    }

    #[test]
    fn sessions_open_once_expire_by_age_and_close() {
        let log = ConnectionLog::default();
        assert!(log.touch(1000, "s1", "127.0.0.1:5000", "zeroclaw/0.8"));
        assert!(!log.touch(2000, "s1", "127.0.0.1:5000", "zeroclaw/0.8"));
        assert!(log.touch(3000, "s2", "127.0.0.1:5001", "curl/8"));
        let live = log.clients(3500, 10_000);
        assert_eq!(live.iter().map(|c| (c.session.as_str(), c.requests)).collect::<Vec<_>>(), vec![("s2", 1), ("s1", 2)]);
        assert_eq!(log.clients(12_500, 10_000).iter().map(|c| c.session.as_str()).collect::<Vec<_>>(), vec!["s2"]);
        log.close(4000, "s2", "127.0.0.1:5001");
        log.close(4000, "s2", "127.0.0.1:5001");
        assert!(log.clients(4000, 10_000).iter().all(|c| c.session != "s2"));
        let kinds: Vec<EventKind> = log.events(10).into_iter().map(|e| e.kind).collect();
        assert_eq!(kinds, vec![EventKind::Closed, EventKind::Opened, EventKind::Opened]);
    }

    #[test]
    fn rejections_are_logged_without_a_session() {
        let log = ConnectionLog::default();
        log.reject(5, "10.0.0.9:1234", "bearer token rejected");
        let e = &log.events(1)[0];
        assert_eq!((e.kind, e.session.as_deref(), e.detail.as_str()), (EventKind::Rejected, None, "bearer token rejected"));
    }

    #[test]
    fn the_session_table_is_bounded() {
        let log = ConnectionLog::default();
        for i in 0..(SESSION_CAP as u64 + 20) {
            log.touch(i, &format!("s{i}"), "r", "ua");
        }
        assert_eq!(log.clients(u64::MAX, u64::MAX).len(), SESSION_CAP);
    }

    #[test]
    fn counters_reset_and_snapshot() {
        let stats = ServerStats::default();
        ServerStats::bump(&stats.requests);
        ServerStats::bump(&stats.requests);
        ServerStats::bump(&stats.rate_limited);
        let s = stats.snapshot();
        assert_eq!((s.requests, s.rate_limited, s.tool_calls), (2, 1, 0));
        stats.reset(42);
        let s = stats.snapshot();
        assert_eq!((s.requests, s.rate_limited, s.started_ms), (0, 0, 42));
    }
}
