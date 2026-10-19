use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::attachments::InlineImage;
use crate::error::{Error, Result};
use crate::text::epoch_utc;

/// Largest `system` plus `prompt` text, in bytes, that `/api/complete` accepts.
pub const COMPLETE_INPUT_CAP: usize = 64 * 1024;
/// Hint used when a request names none.
pub const DEFAULT_COMPLETE_HINT: &str = "quick";

/// `GET /health` (unauthenticated).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Health {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub paired: bool,
    #[serde(default)]
    pub require_pairing: bool,
    #[serde(default)]
    pub runtime: Value,
}

/// `GET /api/config/agent-options`: the alias lists the gateway is configured with.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct AgentOptions {
    #[serde(default)]
    pub agents: Vec<String>,
    #[serde(default)]
    pub model_providers: Vec<String>,
    #[serde(default)]
    pub risk_profiles: Vec<String>,
    #[serde(default)]
    pub mcp_bundles: Vec<String>,
    #[serde(default)]
    pub channels: Vec<String>,
}

/// One row of `GET /api/sessions`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    #[serde(default)]
    pub session_key: Option<String>,
    #[serde(default)]
    pub agent_alias: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub message_count: u64,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub last_activity: Option<String>,
    #[serde(default)]
    pub channel_id: Option<String>,
}

impl SessionInfo {
    pub fn key(&self) -> &str {
        self.session_key.as_deref().unwrap_or(&self.session_id)
    }
}

/// One conversation; a channel transcript and its `gw_` twin store share a row.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRow {
    pub id: String,
    pub name: Option<String>,
    /// Every store held under this id.
    pub keys: Vec<String>,
    pub message_count: u64,
    pub last_activity: Option<String>,
    /// `last_activity` as epoch seconds.
    pub last_activity_epoch: Option<i64>,
}

/// Rows for one agent (`None` keeps all), newest activity first, without the empty shells a bare connect leaves.
pub fn group_sessions(agent: Option<&str>, sessions: &[SessionInfo]) -> Vec<SessionRow> {
    let mut rows: Vec<SessionRow> = Vec::new();
    for s in sessions {
        if agent.is_some_and(|a| s.agent_alias.as_deref() != Some(a)) || s.message_count == 0 {
            continue;
        }
        let name = s.name.as_deref().map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
        let key = s.key().to_string();
        match rows.iter_mut().find(|r| r.id == s.session_id) {
            Some(r) => {
                r.message_count += s.message_count;
                if s.last_activity > r.last_activity {
                    r.last_activity = s.last_activity.clone();
                }
                if r.name.is_none() {
                    r.name = name;
                }
                if !r.keys.contains(&key) {
                    r.keys.push(key);
                }
            }
            None => rows.push(SessionRow {
                id: s.session_id.clone(),
                name,
                keys: vec![key],
                message_count: s.message_count,
                last_activity: s.last_activity.clone(),
                last_activity_epoch: None,
            }),
        }
    }
    for r in &mut rows {
        r.last_activity_epoch = r.last_activity.as_deref().and_then(epoch_utc);
    }
    rows.sort_by(|a, b| b.last_activity.cmp(&a.last_activity));
    rows
}

/// `GET /api/sessions/{id}/state`.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnState {
    /// `idle`, `running` or `error`.
    pub state: String,
    pub running: bool,
    pub turn_id: Option<String>,
    /// Epoch seconds the running turn began.
    pub started: Option<i64>,
}

impl TurnState {
    pub fn parse(v: &Value) -> Result<Self> {
        let state = v
            .get("state")
            .and_then(Value::as_str)
            .filter(|s| matches!(*s, "running" | "idle" | "error"))
            .ok_or_else(|| Error::Decode { path: "/api/sessions/{id}/state".into(), message: "unknown session state".into() })?;
        let running = state == "running";
        Ok(TurnState {
            state: state.to_string(),
            running,
            turn_id: v.get("turn_id").and_then(Value::as_str).map(str::to_string),
            started: v
                .get("turn_started_at")
                .and_then(Value::as_str)
                .and_then(epoch_utc)
                .filter(|_| running),
        })
    }
}

/// Outcome of `POST /api/sessions/{id}/abort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortOutcome {
    /// The running turn was cancelled.
    Aborted,
    /// Nothing was running.
    NothingRunning,
}

/// `POST /api/complete` request. One model call through a `[[model_routes]]` hint: no agent, tools or session.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompleteRequest {
    pub hint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

impl CompleteRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { hint: DEFAULT_COMPLETE_HINT.into(), system: None, prompt: prompt.into(), temperature: None }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = hint.into();
        self
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Appends an inline image marker to the prompt, refusing one that would break the size cap.
    pub fn with_image(mut self, image: &InlineImage) -> Result<Self> {
        self.prompt = crate::attachments::compose_message(&self.prompt, std::slice::from_ref(image));
        self.validate()?;
        Ok(self)
    }

    /// Bytes counted against [`COMPLETE_INPUT_CAP`].
    pub fn input_len(&self) -> usize {
        self.prompt.len() + self.system.as_deref().map_or(0, str::len)
    }

    /// The gateway's checks, run before sending.
    pub fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() {
            return Err(Error::EmptyMessage);
        }
        let size = self.input_len();
        if size > COMPLETE_INPUT_CAP {
            return Err(Error::TooLarge { size, limit: COMPLETE_INPUT_CAP });
        }
        if self.temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
            return Err(Error::Config("temperature must be between 0.0 and 2.0".into()));
        }
        if self.hint.trim().is_empty() {
            return Err(Error::Config("hint is empty".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CompleteResponse {
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub hint: String,
}

/// One memory entry as `GET /api/memory` lists it; fields beyond these stay in `extra`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MemoryEntry {
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, deserialize_with = "category_string")]
    pub category: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Categories serialize as a plain string or as `{"custom": "name"}`.
fn category_string<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    Ok(match Value::deserialize(d)? {
        Value::String(s) => s,
        Value::Object(o) => o.values().find_map(|v| v.as_str().map(str::to_string)).unwrap_or_default(),
        _ => String::new(),
    })
}

/// Body of `POST /api/memory`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryStore {
    pub key: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

impl MemoryStore {
    pub fn new(key: impl Into<String>, content: impl Into<String>) -> Self {
        Self { key: key.into(), content: content.into(), category: None, agent: None }
    }
}

/// Filters for `GET /api/memory`. With a query (or a time range) the gateway recalls at most 50 entries; without, it lists.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryQuery {
    pub agent: Option<String>,
    pub query: Option<String>,
    pub category: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
}

/// One JSON Patch operation for `PATCH /api/config`; this crate only builds the body and never sends it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchOp {
    pub op: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

impl PatchOp {
    pub fn add(path: impl Into<String>, value: Value) -> Self {
        Self { op: "add".into(), path: path.into(), value: Some(value) }
    }

    pub fn replace(path: impl Into<String>, value: Value) -> Self {
        Self { op: "replace".into(), path: path.into(), value: Some(value) }
    }

    pub fn remove(path: impl Into<String>) -> Self {
        Self { op: "remove".into(), path: path.into(), value: None }
    }
}

/// A JSON Patch document (RFC 6902) for the gateway config; serializes as a bare array.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConfigPatch(pub Vec<PatchOp>);

impl ConfigPatch {
    pub fn push(mut self, op: PatchOp) -> Self {
        self.0.push(op);
        self
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!([]))
    }
}

/// `POST /pair` success body. The token is secret: `Debug` hides it.
#[derive(Clone, Deserialize)]
pub struct PairResult {
    #[serde(default)]
    pub paired: bool,
    pub token: String,
}

impl std::fmt::Debug for PairResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairResult").field("paired", &self.paired).field("token", &"<redacted>").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: &str, key: &str, agent: &str, n: u64, at: &str, name: Option<&str>) -> SessionInfo {
        SessionInfo {
            session_id: id.into(),
            session_key: Some(key.into()),
            agent_alias: Some(agent.into()),
            name: name.map(str::to_string),
            message_count: n,
            created_at: None,
            last_activity: Some(at.into()),
            channel_id: None,
        }
    }

    #[test]
    fn session_rows_skip_empty_shells_and_other_agents_newest_first() {
        let rows = group_sessions(
            Some("companion"),
            &[
                info("old", "gw_old", "companion", 2, "2026-09-09T12:00:00+00:00", None),
                info("empty", "gw_empty", "companion", 0, "2026-09-11T11:59:00+00:00", None),
                info("other", "gw_other", "voice", 9, "2026-09-11T11:00:00+00:00", None),
                info("new", "gw_new", "companion", 4, "2026-09-11T10:00:00+00:00", Some("Home Assistant")),
            ],
        );
        assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["new", "old"]);
        assert_eq!(rows[0].name.as_deref(), Some("Home Assistant"));
        assert!(rows[0].last_activity_epoch.is_some());
    }

    #[test]
    fn twin_stores_collapse_into_one_row_with_both_keys() {
        let rows = group_sessions(
            None,
            &[
                info("discord_x", "gw_discord_x", "comfy", 8, "2026-09-12T02:20:29+00:00", None),
                info("discord_x", "discord_x", "comfy", 915, "2026-09-11T05:42:11+00:00", Some("thread")),
            ],
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].message_count, 923);
        assert_eq!(rows[0].keys, ["gw_discord_x", "discord_x"]);
        assert_eq!(rows[0].name.as_deref(), Some("thread"));
        assert_eq!(rows[0].last_activity.as_deref(), Some("2026-09-12T02:20:29+00:00"));
    }

    #[test]
    fn session_info_parses_with_missing_fields() {
        let s: SessionInfo = serde_json::from_str(r#"{"session_id":"a"}"#).unwrap();
        assert_eq!(s.key(), "a");
        assert_eq!(s.message_count, 0);
        let s: SessionInfo = serde_json::from_str(r#"{"session_id":"a","session_key":"gw_a","extra":1}"#).unwrap();
        assert_eq!(s.key(), "gw_a");
    }

    #[test]
    fn turn_state_reads_running_and_when_it_began() {
        let v = json!({"state":"running","turn_id":"t","turn_started_at":"2026-09-13T02:58:13.313658155+00:00"});
        let s = TurnState::parse(&v).unwrap();
        assert!(s.running);
        assert_eq!(s.started, Some(1_789_268_293));
        assert_eq!(s.turn_id.as_deref(), Some("t"));
        let idle = TurnState::parse(&json!({"state":"idle","turn_started_at":"2026-09-13T02:58:13+00:00"})).unwrap();
        assert!(!idle.running);
        assert_eq!(idle.started, None);
        assert!(TurnState::parse(&json!({})).is_err());
        assert!(TurnState::parse(&json!({"state":"paused"})).is_err());
        assert_eq!(TurnState::parse(&json!({"state":"error"})).unwrap().state, "error");
    }

    #[test]
    fn complete_request_validation_matches_the_gateway() {
        assert!(CompleteRequest::new("ls -la").validate().is_ok());
        assert!(matches!(CompleteRequest::new("   ").validate(), Err(Error::EmptyMessage)));
        let at_cap = "x".repeat(COMPLETE_INPUT_CAP);
        assert!(CompleteRequest::new(at_cap.clone()).validate().is_ok());
        assert!(matches!(
            CompleteRequest::new(format!("{at_cap}x")).validate(),
            Err(Error::TooLarge { size, limit }) if size == COMPLETE_INPUT_CAP + 1 && limit == COMPLETE_INPUT_CAP
        ));
        let system_counts = CompleteRequest::new("p").with_system("s".repeat(COMPLETE_INPUT_CAP));
        assert!(system_counts.validate().is_err());
        assert!(CompleteRequest::new("p").with_temperature(2.5).validate().is_err());
        assert!(CompleteRequest::new("p").with_temperature(2.0).validate().is_ok());
        assert!(CompleteRequest::new("p").with_hint(" ").validate().is_err());
    }

    #[test]
    fn complete_request_serializes_without_unset_fields() {
        let v = serde_json::to_value(CompleteRequest::new("hello")).unwrap();
        assert_eq!(v, json!({"hint":"quick","prompt":"hello"}));
        let v = serde_json::to_value(CompleteRequest::new("p").with_hint("slow").with_system("s").with_temperature(0.2)).unwrap();
        assert_eq!(v, json!({"hint":"slow","system":"s","prompt":"p","temperature":0.2}));
    }

    #[test]
    fn image_marker_counts_against_the_complete_cap() {
        let small = InlineImage::from_bytes("a.png", b"\x89PNG\r\n\x1a\nrest").unwrap();
        let req = CompleteRequest::new("describe").with_image(&small).unwrap();
        assert!(req.prompt.starts_with("describe [IMAGE:data:image/png;base64,"));
        let bytes: Vec<u8> = b"\x89PNG\r\n\x1a\n".iter().copied().chain(std::iter::repeat_n(7u8, 60_000)).collect();
        let big = InlineImage::from_bytes("b.png", &bytes).unwrap();
        assert!(matches!(CompleteRequest::new("describe").with_image(&big), Err(Error::TooLarge { .. })));
    }

    #[test]
    fn memory_entry_accepts_string_and_custom_categories() {
        let e: MemoryEntry = serde_json::from_value(json!({"key":"k","content":"c","category":"core","id":"1"})).unwrap();
        assert_eq!(e.category, "core");
        assert_eq!(e.extra.get("id"), Some(&json!("1")));
        let e: MemoryEntry = serde_json::from_value(json!({"key":"k","content":"c","category":{"custom":"trips"}})).unwrap();
        assert_eq!(e.category, "trips");
    }

    #[test]
    fn config_patch_serializes_as_a_json_patch_array() {
        let patch = ConfigPatch::default()
            .push(PatchOp::replace("/agents/media/risk_profile", json!("media")))
            .push(PatchOp::remove("/mcp/servers/0"));
        assert_eq!(
            patch.to_json(),
            json!([
                {"op":"replace","path":"/agents/media/risk_profile","value":"media"},
                {"op":"remove","path":"/mcp/servers/0"}
            ])
        );
        let back: ConfigPatch = serde_json::from_value(patch.to_json()).unwrap();
        assert_eq!(back, patch);
    }

    #[test]
    fn pair_result_hides_the_token() {
        let p: PairResult = serde_json::from_str(r#"{"paired":true,"persisted":true,"token":"zc_secret"}"#).unwrap();
        assert!(p.paired);
        assert!(!format!("{p:?}").contains("zc_secret"));
    }

    #[test]
    fn health_and_agent_options_tolerate_missing_fields() {
        let h: Health = serde_json::from_str(r#"{"status":"ok","paired":true,"require_pairing":true,"runtime":{"x":1}}"#).unwrap();
        assert!(h.require_pairing && h.paired);
        let h: Health = serde_json::from_str("{}").unwrap();
        assert_eq!(h.status, "");
        let a: AgentOptions = serde_json::from_str(r#"{"agents":["a","b"],"unknown_list":[1]}"#).unwrap();
        assert_eq!(a.agents, ["a", "b"]);
    }
}
