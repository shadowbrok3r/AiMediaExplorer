use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Decision carried by an `approval_response` frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalDecision {
    Approve,
    /// Approve and stop asking for this tool.
    Always,
    Deny,
}

/// Frames the client sends on `/ws/chat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    Connect {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_name: Option<String>,
        #[serde(default)]
        capabilities: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
    /// Starts a turn; while a turn is running the gateway treats it as steering.
    Message { content: String },
    ApprovalResponse { request_id: String, decision: ApprovalDecision },
}

impl ClientFrame {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn kind(&self) -> &'static str {
        match self {
            ClientFrame::Connect { .. } => "connect",
            ClientFrame::Message { .. } => "message",
            ClientFrame::ApprovalResponse { .. } => "approval_response",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PlanEntry {
    #[serde(default, deserialize_with = "lossy_string")]
    pub content: String,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DoneFrame {
    #[serde(default, deserialize_with = "lossy_string")]
    pub full_response: String,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub tokens_used: Option<u64>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub max_context_tokens: Option<u64>,
    #[serde(default)]
    pub last_input_tokens: Option<u64>,
}

/// Frames the gateway sends on `/ws/chat`; unparseable or unknown frames become [`ServerFrame::Unknown`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    SessionStart {
        #[serde(deserialize_with = "lossy_string")]
        session_id: String,
        #[serde(default)]
        resumed: bool,
        #[serde(default)]
        message_count: u64,
        #[serde(default)]
        name: Option<String>,
    },
    Connected {
        #[serde(default, deserialize_with = "lossy_string")]
        message: String,
    },
    /// The running turn so far, sent to a socket that joins mid-turn. Entries are raw frames.
    TurnResume {
        #[serde(default)]
        frames: Vec<Value>,
    },
    AgentStart {
        #[serde(default)]
        model_provider: Option<String>,
        #[serde(default)]
        model: Option<String>,
    },
    AgentEnd {
        #[serde(default)]
        model_provider: Option<String>,
        #[serde(default)]
        model: Option<String>,
    },
    Thinking {
        #[serde(default, deserialize_with = "lossy_string")]
        content: String,
    },
    Chunk {
        #[serde(default, deserialize_with = "lossy_string")]
        content: String,
    },
    ToolCall {
        #[serde(default, deserialize_with = "opt_lossy_string")]
        id: Option<String>,
        #[serde(default, alias = "tool", deserialize_with = "lossy_string")]
        name: String,
        #[serde(default, alias = "arguments")]
        args: Value,
    },
    ToolResult {
        #[serde(default, deserialize_with = "opt_lossy_string")]
        id: Option<String>,
        #[serde(default, alias = "tool", deserialize_with = "lossy_string")]
        name: String,
        #[serde(default, alias = "result", alias = "content", deserialize_with = "lossy_string")]
        output: String,
    },
    ApprovalRequest {
        #[serde(deserialize_with = "lossy_string")]
        request_id: String,
        #[serde(default, alias = "tool_name", deserialize_with = "lossy_string")]
        tool: String,
        #[serde(default, deserialize_with = "lossy_string")]
        arguments_summary: String,
        #[serde(default)]
        arguments: Value,
        #[serde(default = "default_approval_timeout")]
        timeout_secs: u64,
    },
    Plan {
        #[serde(default)]
        entries: Vec<PlanEntry>,
    },
    HistoryTrimmed {
        #[serde(default)]
        dropped_messages: u64,
        #[serde(default)]
        kept_turns: u64,
        #[serde(default, deserialize_with = "lossy_string")]
        reason: String,
    },
    Done(DoneFrame),
    Aborted,
    Error {
        #[serde(default, deserialize_with = "lossy_string")]
        message: String,
        #[serde(default, deserialize_with = "opt_lossy_string")]
        code: Option<String>,
    },
    /// Any frame this client does not model, kept verbatim. Non-JSON text is kept as a JSON string.
    #[serde(skip)]
    Unknown(Value),
}

/// Default seconds the gateway waits for an approval before denying.
pub const DEFAULT_APPROVAL_TIMEOUT_SECS: u64 = 300;

fn default_approval_timeout() -> u64 {
    DEFAULT_APPROVAL_TIMEOUT_SECS
}

impl ServerFrame {
    pub fn parse(text: &str) -> ServerFrame {
        match serde_json::from_str::<Value>(text) {
            Ok(v) => ServerFrame::from_value(v),
            Err(_) => ServerFrame::Unknown(Value::String(text.to_string())),
        }
    }

    pub fn from_value(v: Value) -> ServerFrame {
        if !v.is_object() {
            return ServerFrame::Unknown(v);
        }
        match ServerFrame::deserialize(&v) {
            Ok(frame) => frame,
            Err(_) => ServerFrame::Unknown(v),
        }
    }

    /// The wire `type` of this frame (the raw one for `Unknown`).
    pub fn kind(&self) -> String {
        match self {
            ServerFrame::SessionStart { .. } => "session_start".into(),
            ServerFrame::Connected { .. } => "connected".into(),
            ServerFrame::TurnResume { .. } => "turn_resume".into(),
            ServerFrame::AgentStart { .. } => "agent_start".into(),
            ServerFrame::AgentEnd { .. } => "agent_end".into(),
            ServerFrame::Thinking { .. } => "thinking".into(),
            ServerFrame::Chunk { .. } => "chunk".into(),
            ServerFrame::ToolCall { .. } => "tool_call".into(),
            ServerFrame::ToolResult { .. } => "tool_result".into(),
            ServerFrame::ApprovalRequest { .. } => "approval_request".into(),
            ServerFrame::Plan { .. } => "plan".into(),
            ServerFrame::HistoryTrimmed { .. } => "history_trimmed".into(),
            ServerFrame::Done(_) => "done".into(),
            ServerFrame::Aborted => "aborted".into(),
            ServerFrame::Error { .. } => "error".into(),
            ServerFrame::Unknown(v) => v.get("type").and_then(Value::as_str).unwrap_or("").to_string(),
        }
    }

    /// The nested frames of a `turn_resume`, parsed.
    pub fn resume_frames(&self) -> Vec<ServerFrame> {
        match self {
            ServerFrame::TurnResume { frames } => frames.iter().cloned().map(ServerFrame::from_value).collect(),
            _ => Vec::new(),
        }
    }
}

/// Error codes after which the gateway ends the connection.
pub fn error_code_is_fatal(code: &str) -> bool {
    matches!(code, "AGENT_INIT_FAILED" | "INVALID_CWD" | "NEEDS_ONBOARDING")
}

/// Error codes that end the running turn but leave the connection usable.
pub fn error_code_ends_turn(code: &str) -> bool {
    matches!(code, "PROVIDER_ERROR" | "AUTH_ERROR" | "AGENT_ERROR")
}

/// A string field that tolerates null, numbers and other scalars instead of failing the frame.
fn lossy_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(opt_lossy_string(d)?.unwrap_or_default())
}

fn opt_lossy_string<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    match Value::deserialize(d)? {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::Bool(b) => Ok(Some(b.to_string())),
        other @ (Value::Array(_) | Value::Object(_)) => {
            Err(de::Error::custom(format!("expected a scalar, got {}", kind_of(&other))))
        }
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn client_frames_serialize_to_the_documented_shapes() {
        let connect = ClientFrame::Connect {
            device_name: Some("ame".into()),
            capabilities: vec!["text".into()],
            cwd: None,
        };
        assert_eq!(
            serde_json::from_str::<Value>(&connect.to_json()).unwrap(),
            json!({"type":"connect","device_name":"ame","capabilities":["text"]})
        );
        assert_eq!(
            serde_json::from_str::<Value>(&ClientFrame::Message { content: "hi".into() }.to_json()).unwrap(),
            json!({"type":"message","content":"hi"})
        );
        let approve = ClientFrame::ApprovalResponse { request_id: "r1".into(), decision: ApprovalDecision::Always };
        assert_eq!(
            serde_json::from_str::<Value>(&approve.to_json()).unwrap(),
            json!({"type":"approval_response","request_id":"r1","decision":"always"})
        );
    }

    #[test]
    fn client_frames_round_trip() {
        let frames = [
            ClientFrame::Connect { device_name: None, capabilities: vec![], cwd: Some("/w".into()) },
            ClientFrame::Message { content: "x".into() },
            ClientFrame::ApprovalResponse { request_id: "7".into(), decision: ApprovalDecision::Deny },
        ];
        for f in frames {
            let back: ClientFrame = serde_json::from_str(&f.to_json()).unwrap();
            assert_eq!(back, f);
        }
    }

    #[test]
    fn every_documented_server_frame_parses_typed() {
        let cases = [
            (r#"{"type":"session_start","session_id":"s","resumed":true,"message_count":4,"name":"n"}"#, "session_start"),
            (r#"{"type":"connected","message":"Connection established"}"#, "connected"),
            (r#"{"type":"turn_resume","frames":[{"type":"chunk","content":"a"}]}"#, "turn_resume"),
            (r#"{"type":"agent_start","model_provider":"p","model":"m"}"#, "agent_start"),
            (r#"{"type":"agent_end","model_provider":"p","model":"m"}"#, "agent_end"),
            (r#"{"type":"thinking","content":"hm"}"#, "thinking"),
            (r#"{"type":"chunk","content":"x"}"#, "chunk"),
            (r#"{"type":"tool_call","id":"c1","name":"t","args":{"a":1}}"#, "tool_call"),
            (r#"{"type":"tool_result","id":"c1","name":"t","output":"ok"}"#, "tool_result"),
            (r#"{"type":"approval_request","request_id":"r1","tool":"t","arguments_summary":"s","arguments":{},"timeout_secs":300}"#, "approval_request"),
            (r#"{"type":"plan","entries":[{"content":"a","status":"pending"}]}"#, "plan"),
            (r#"{"type":"history_trimmed","dropped_messages":3,"kept_turns":2,"reason":"r"}"#, "history_trimmed"),
            (r#"{"type":"done","full_response":"ok","input_tokens":10,"output_tokens":5,"tokens_used":15,"cost_usd":0.5,"model":"m","provider":"p","max_context_tokens":1000,"last_input_tokens":9}"#, "done"),
            (r#"{"type":"aborted"}"#, "aborted"),
            (r#"{"type":"error","message":"m","code":"PROVIDER_ERROR"}"#, "error"),
        ];
        for (text, kind) in cases {
            let frame = ServerFrame::parse(text);
            assert!(!matches!(frame, ServerFrame::Unknown(_)), "{text} parsed as unknown");
            assert_eq!(frame.kind(), kind);
        }
    }

    #[test]
    fn typed_fields_are_read() {
        match ServerFrame::parse(r#"{"type":"done","full_response":"ok","input_tokens":10,"cost_usd":0.25,"model":"m","max_context_tokens":1000}"#) {
            ServerFrame::Done(d) => {
                assert_eq!(d.full_response, "ok");
                assert_eq!(d.input_tokens, Some(10));
                assert_eq!(d.output_tokens, None);
                assert_eq!(d.cost_usd, Some(0.25));
                assert_eq!(d.model.as_deref(), Some("m"));
                assert_eq!(d.max_context_tokens, Some(1000));
            }
            other => panic!("{other:?}"),
        }
        match ServerFrame::parse(r#"{"type":"approval_request","request_id":"r","tool":"t"}"#) {
            ServerFrame::ApprovalRequest { timeout_secs, arguments, .. } => {
                assert_eq!(timeout_secs, DEFAULT_APPROVAL_TIMEOUT_SECS);
                assert_eq!(arguments, Value::Null);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn tool_frames_accept_the_alias_spellings() {
        match ServerFrame::parse(r#"{"type":"tool_call","tool":"shell","arguments":{"cmd":"ls"}}"#) {
            ServerFrame::ToolCall { id, name, args } => {
                assert_eq!(id, None);
                assert_eq!(name, "shell");
                assert_eq!(args, json!({"cmd":"ls"}));
            }
            other => panic!("{other:?}"),
        }
        match ServerFrame::parse(r#"{"type":"tool_result","id":7,"name":"shell","result":"done"}"#) {
            ServerFrame::ToolResult { id, output, .. } => {
                assert_eq!(id.as_deref(), Some("7"));
                assert_eq!(output, "done");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unknown_types_and_garbage_pass_through() {
        let v = json!({"type":"cron_result","job":"j","ok":true});
        match ServerFrame::from_value(v.clone()) {
            ServerFrame::Unknown(raw) => assert_eq!(raw, v),
            other => panic!("{other:?}"),
        }
        assert_eq!(ServerFrame::parse(&v.to_string()).kind(), "cron_result");
        assert_eq!(ServerFrame::parse("not json"), ServerFrame::Unknown(Value::String("not json".into())));
        assert_eq!(ServerFrame::parse("[1,2]"), ServerFrame::Unknown(json!([1, 2])));
        assert_eq!(ServerFrame::parse("{}").kind(), "");
        assert_eq!(ServerFrame::parse(r#"{"type":7}"#).kind(), "");
    }

    #[test]
    fn drifted_known_frames_fall_back_to_unknown_not_an_error() {
        let drifted = json!({"type":"approval_request"});
        assert_eq!(ServerFrame::from_value(drifted.clone()), ServerFrame::Unknown(drifted));
        let nested = json!({"type":"chunk","content":{"text":"x"}});
        assert_eq!(ServerFrame::from_value(nested.clone()), ServerFrame::Unknown(nested));
        let bad_count = json!({"type":"session_start","session_id":"s","message_count":"many"});
        assert!(matches!(ServerFrame::from_value(bad_count), ServerFrame::Unknown(_)));
    }

    #[test]
    fn extra_fields_are_ignored_and_scalars_coerced() {
        match ServerFrame::parse(r#"{"type":"chunk","content":"x","fork_extra":{"a":1}}"#) {
            ServerFrame::Chunk { content } => assert_eq!(content, "x"),
            other => panic!("{other:?}"),
        }
        match ServerFrame::parse(r#"{"type":"chunk","content":null}"#) {
            ServerFrame::Chunk { content } => assert_eq!(content, ""),
            other => panic!("{other:?}"),
        }
        match ServerFrame::parse(r#"{"type":"approval_request","request_id":42,"tool":"t"}"#) {
            ServerFrame::ApprovalRequest { request_id, .. } => assert_eq!(request_id, "42"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn turn_resume_exposes_nested_frames() {
        let frame = ServerFrame::parse(
            r#"{"type":"turn_resume","frames":[{"type":"thinking","content":"a"},{"type":"mystery"},{"type":"chunk","content":"b"}]}"#,
        );
        let nested = frame.resume_frames();
        assert_eq!(nested.len(), 3);
        assert!(matches!(nested[0], ServerFrame::Thinking { .. }));
        assert!(matches!(nested[1], ServerFrame::Unknown(_)));
        assert!(matches!(nested[2], ServerFrame::Chunk { .. }));
        assert!(ServerFrame::parse(r#"{"type":"chunk","content":"x"}"#).resume_frames().is_empty());
    }

    #[test]
    fn error_code_classes() {
        for c in ["PROVIDER_ERROR", "AUTH_ERROR", "AGENT_ERROR"] {
            assert!(error_code_ends_turn(c) && !error_code_is_fatal(c));
        }
        for c in ["AGENT_INIT_FAILED", "INVALID_CWD", "NEEDS_ONBOARDING"] {
            assert!(error_code_is_fatal(c) && !error_code_ends_turn(c));
        }
        for c in ["EMPTY_CONTENT", "INVALID_JSON", "STEERING_QUEUE_FULL", ""] {
            assert!(!error_code_is_fatal(c) && !error_code_ends_turn(c));
        }
    }
}
