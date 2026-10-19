use serde::Deserialize;
use serde_json::Value;

use crate::attachments::{ImageRef, split_images};

/// One stored message as `GET /api/sessions/{id}/messages` returns it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StoredMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub created_at: Option<String>,
}

/// A tool call recovered from a stored assistant envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredCall {
    pub id: Option<String>,
    pub name: String,
    pub args: Value,
}

/// What one stored message unpacks to.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stored {
    pub text: String,
    pub reasoning: Option<String>,
    pub calls: Vec<StoredCall>,
    /// Set on a tool-result envelope.
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

/// Unpacks the JSON envelope the gateway stores assistant and tool messages in; other text is returned as is.
pub fn unwrap_stored(role: &str, raw: &str) -> Stored {
    let plain = || Stored { text: raw.to_string(), ..Stored::default() };
    if role == "user" || !raw.trim_start().starts_with('{') {
        return plain();
    }
    let Ok(Value::Object(o)) = serde_json::from_str::<Value>(raw) else { return plain() };
    if !o.contains_key("content") {
        return plain();
    }
    let text = o.get("content").and_then(Value::as_str).unwrap_or("").to_string();
    let non_empty = |key: &str| {
        o.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
    };
    let calls = o
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|c| {
                    let name = c.get("name").or_else(|| c.pointer("/function/name")).and_then(Value::as_str)?;
                    let args = match c.get("arguments").or_else(|| c.pointer("/function/arguments")) {
                        Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.clone())),
                        Some(v) => v.clone(),
                        None => Value::Null,
                    };
                    Some(StoredCall {
                        id: c.get("id").and_then(Value::as_str).map(str::to_string),
                        name: name.to_string(),
                        args,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Stored {
        text,
        reasoning: non_empty("reasoning_content"),
        calls,
        tool_call_id: non_empty("tool_call_id"),
        tool_name: non_empty("tool_name"),
    }
}

/// Drops the `[CURRENT DATE & TIME: ...]` or `[YYYY-MM-DD HH:MM:SS +00:00]` preamble the gateway prepends to user messages.
pub fn strip_date_stamp(s: &str) -> String {
    let t = s.trim_start();
    let Some(rest) = t.strip_prefix('[') else { return s.to_string() };
    let labelled = rest.starts_with("CURRENT DATE & TIME:");
    let dated = rest.len() > 4 && rest.as_bytes()[..4].iter().all(u8::is_ascii_digit) && rest.as_bytes()[4] == b'-';
    if !labelled && !dated {
        return s.to_string();
    }
    match rest.find(']') {
        Some(i) => rest[i + 1..].trim_start().to_string(),
        None => s.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    User,
    Assistant,
    Reasoning,
    ToolCall,
    ToolResult,
    /// A role this client does not model (for example `system`).
    Other,
}

/// One transcript row derived from stored messages.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryItem {
    /// Stable per message index: `h{index}`, with `:r`/`:c{n}` suffixes for rows split out of one message.
    pub id: String,
    pub kind: HistoryKind,
    pub text: String,
    /// Tool name for `ToolCall` and `ToolResult` rows, the raw role for `Other`.
    pub name: Option<String>,
    pub args: Option<Value>,
    /// Tool call id for `ToolCall` and `ToolResult` rows.
    pub call_id: Option<String>,
    pub images: Vec<ImageRef>,
    pub created_at: Option<String>,
}

fn row(id: String, kind: HistoryKind, created_at: &Option<String>) -> HistoryItem {
    HistoryItem {
        id,
        kind,
        text: String::new(),
        name: None,
        args: None,
        call_id: None,
        images: Vec::new(),
        created_at: created_at.clone(),
    }
}

/// Rows for one stored message; `index` is its position in the full transcript.
pub fn items_for(index: usize, m: &StoredMessage) -> Vec<HistoryItem> {
    let id = format!("h{index}");
    let stored = unwrap_stored(&m.role, &m.content);
    let mut out = Vec::new();
    if let Some(reasoning) = stored.reasoning {
        out.push(HistoryItem { text: reasoning, ..row(format!("{id}:r"), HistoryKind::Reasoning, &m.created_at) });
    }
    for (j, call) in stored.calls.into_iter().enumerate() {
        out.push(HistoryItem {
            name: Some(call.name),
            args: Some(call.args),
            call_id: call.id,
            ..row(format!("{id}:c{j}"), HistoryKind::ToolCall, &m.created_at)
        });
    }
    let (text, images) = split_images(&strip_date_stamp(&stored.text));
    if text.is_empty() && images.is_empty() {
        return out;
    }
    let (kind, name) = match m.role.as_str() {
        "user" => (HistoryKind::User, None),
        "assistant" => (HistoryKind::Assistant, None),
        "tool" => (HistoryKind::ToolResult, stored.tool_name),
        other => (HistoryKind::Other, Some(other.to_string())),
    };
    out.push(HistoryItem {
        text,
        name,
        call_id: stored.tool_call_id,
        images,
        ..row(id, kind, &m.created_at)
    });
    out
}

/// Rows for a whole transcript, oldest first.
pub fn parse_history(messages: &[StoredMessage]) -> Vec<HistoryItem> {
    messages.iter().enumerate().flat_map(|(i, m)| items_for(i, m)).collect()
}

/// Interleave several stores of one conversation by stamp, keeping each store's own order on ties.
pub fn merge_by_time(mut messages: Vec<StoredMessage>) -> Vec<StoredMessage> {
    messages.sort_by(|a, b| a.created_at.as_deref().unwrap_or("").cmp(b.created_at.as_deref().unwrap_or("")));
    messages
}

/// Pager over a fetched transcript: the newest page first, older pages on demand.
#[derive(Debug, Clone)]
pub struct HistoryPager {
    messages: Vec<StoredMessage>,
    /// Index of the oldest message already handed out; equals `messages.len()` before the first page.
    cursor: usize,
    page_size: usize,
}

impl HistoryPager {
    /// `page_size` counts messages, not rows (one message can yield several rows); minimum 1.
    pub fn new(messages: Vec<StoredMessage>, page_size: usize) -> Self {
        let cursor = messages.len();
        Self { messages, cursor, page_size: page_size.max(1) }
    }

    pub fn total_messages(&self) -> usize {
        self.messages.len()
    }

    /// Messages not yet handed out.
    pub fn remaining(&self) -> usize {
        self.cursor
    }

    pub fn has_more(&self) -> bool {
        self.cursor > 0
    }

    /// The next older page, rows in chronological order. Empty when everything was returned.
    pub fn next_page(&mut self) -> Vec<HistoryItem> {
        let end = self.cursor;
        let start = end.saturating_sub(self.page_size);
        self.cursor = start;
        (start..end).flat_map(|i| items_for(i, &self.messages[i])).collect()
    }

    /// Every remaining page at once, rows in chronological order.
    pub fn rest(&mut self) -> Vec<HistoryItem> {
        let end = self.cursor;
        self.cursor = 0;
        (0..end).flat_map(|i| items_for(i, &self.messages[i])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(role: &str, content: &str) -> StoredMessage {
        StoredMessage { role: role.into(), content: content.into(), created_at: None }
    }

    #[test]
    fn user_and_plain_assistant_messages_stay_text() {
        let s = unwrap_stored("user", "{\"content\":\"looks like an envelope\"}");
        assert_eq!(s.text, "{\"content\":\"looks like an envelope\"}");
        let s = unwrap_stored("assistant", "hello");
        assert_eq!(s, Stored { text: "hello".into(), ..Stored::default() });
        let s = unwrap_stored("assistant", "{\"no_content_key\":1}");
        assert_eq!(s.text, "{\"no_content_key\":1}");
        let s = unwrap_stored("assistant", "{not json");
        assert_eq!(s.text, "{not json");
    }

    #[test]
    fn assistant_envelope_unpacks_reasoning_calls_and_reply() {
        let raw = r#"{"content":"*Wild, huh?*","reasoning_content":"Which character?","tool_calls":[{"arguments":"{\"limit\":10}","id":"x","name":"memory_recall"}]}"#;
        let s = unwrap_stored("assistant", raw);
        assert_eq!(s.text, "*Wild, huh?*");
        assert_eq!(s.reasoning.as_deref(), Some("Which character?"));
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].name, "memory_recall");
        assert_eq!(s.calls[0].id.as_deref(), Some("x"));
        assert_eq!(s.calls[0].args, json!({"limit": 10}));
    }

    #[test]
    fn openai_style_function_calls_and_unparsable_arguments_are_kept() {
        let raw = r#"{"content":null,"tool_calls":[{"function":{"name":"shell","arguments":"{\"cmd\":\"ls\"}"}},{"name":"raw","arguments":"not json"},{"arguments":"{}"}]}"#;
        let s = unwrap_stored("assistant", raw);
        assert_eq!(s.text, "");
        assert_eq!(s.calls.len(), 2, "a call without a name is dropped");
        assert_eq!(s.calls[0].args, json!({"cmd":"ls"}));
        assert_eq!(s.calls[1].args, json!("not json"));
    }

    #[test]
    fn tool_envelope_unpacks_output_and_ids() {
        let s = unwrap_stored("tool", r#"{"content":"review: OK","tool_call_id":"abc","tool_name":"critic"}"#);
        assert_eq!(s.text, "review: OK");
        assert_eq!(s.tool_call_id.as_deref(), Some("abc"));
        assert_eq!(s.tool_name.as_deref(), Some("critic"));
        assert_eq!(unwrap_stored("tool", "{\"ok\":true}").text, "{\"ok\":true}");
    }

    #[test]
    fn date_stamps_are_stripped_in_both_spellings() {
        assert_eq!(strip_date_stamp("[CURRENT DATE & TIME: 2026-09-11 21:51:33 +00:00]\n\nhello"), "hello");
        assert_eq!(strip_date_stamp("[2026-09-11 05:36:28 +00:00] Do more poses"), "Do more poses");
        assert_eq!(strip_date_stamp("plain text"), "plain text");
        assert_eq!(strip_date_stamp("[IMAGE:/tmp/a.png]"), "[IMAGE:/tmp/a.png]");
        assert_eq!(strip_date_stamp("[note] see above"), "[note] see above");
        assert_eq!(strip_date_stamp("[2026"), "[2026");
        assert_eq!(strip_date_stamp("[2026-09-11 unterminated"), "[2026-09-11 unterminated");
    }

    #[test]
    fn rows_split_out_of_one_assistant_message() {
        let m = msg(
            "assistant",
            r#"{"content":"done","reasoning_content":"think","tool_calls":[{"id":"c","name":"t","arguments":"{}"}]}"#,
        );
        let rows = items_for(3, &m);
        let kinds: Vec<_> = rows.iter().map(|r| (r.id.as_str(), r.kind)).collect();
        assert_eq!(kinds, [("h3:r", HistoryKind::Reasoning), ("h3:c0", HistoryKind::ToolCall), ("h3", HistoryKind::Assistant)]);
        assert_eq!(rows[1].name.as_deref(), Some("t"));
        assert_eq!(rows[1].call_id.as_deref(), Some("c"));
        assert_eq!(rows[2].text, "done");
    }

    #[test]
    fn empty_and_tool_only_messages_yield_no_text_row() {
        assert!(items_for(0, &msg("user", "   ")).is_empty());
        let calls_only = msg("assistant", r#"{"content":null,"tool_calls":[{"name":"shell","arguments":"{}"}]}"#);
        let rows = items_for(0, &calls_only);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, HistoryKind::ToolCall);
    }

    #[test]
    fn roles_map_to_kinds() {
        let rows = parse_history(&[
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("tool", r#"{"content":"out","tool_call_id":"c","tool_name":"t"}"#),
            msg("system", "unmodelled"),
        ]);
        let kinds: Vec<_> = rows.iter().map(|r| r.kind).collect();
        assert_eq!(kinds, [HistoryKind::User, HistoryKind::Assistant, HistoryKind::ToolResult, HistoryKind::Other]);
        assert_eq!(rows[2].name.as_deref(), Some("t"));
        assert_eq!(rows[3].name.as_deref(), Some("system"));
    }

    #[test]
    fn user_images_become_refs_and_stamp_is_dropped() {
        let rows = parse_history(&[
            msg("user", "[2026-09-18 10:00:00 +00:00] [IMAGE:data:image/jpeg;base64,BBBB]"),
            msg("user", "look [IMAGE:data:image/png;base64,AAAA] please"),
        ]);
        assert_eq!(rows[0].text, "");
        assert_eq!(rows[0].images.len(), 1);
        assert_eq!(rows[1].text, "look please");
    }

    #[test]
    fn merge_interleaves_by_stamp_and_keeps_tie_order() {
        let at = |c: &str, t: &str| StoredMessage { role: "user".into(), content: c.into(), created_at: Some(t.into()) };
        let merged = merge_by_time(vec![
            at("b", "2026-09-12T01:00:00+00:00"),
            at("d", "2026-09-12T03:00:00+00:00"),
            at("a", "2026-09-11T05:00:00+00:00"),
            at("c", "2026-09-12T01:00:00+00:00"),
        ]);
        let order: Vec<_> = merged.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(order, ["a", "b", "c", "d"]);
    }

    #[test]
    fn pager_serves_newest_page_first_then_older_ones() {
        let msgs: Vec<_> = (0..10).map(|i| msg("user", &format!("m{i}"))).collect();
        let mut pager = HistoryPager::new(msgs, 4);
        assert_eq!(pager.total_messages(), 10);
        assert!(pager.has_more());
        let p1 = pager.next_page();
        assert_eq!(p1.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["m6", "m7", "m8", "m9"]);
        assert_eq!(p1[0].id, "h6", "ids keep the real message index");
        let p2 = pager.next_page();
        assert_eq!(p2.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["m2", "m3", "m4", "m5"]);
        assert_eq!(pager.remaining(), 2);
        let p3 = pager.next_page();
        assert_eq!(p3.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["m0", "m1"]);
        assert!(!pager.has_more());
        assert!(pager.next_page().is_empty());
    }

    #[test]
    fn pager_rest_returns_everything_older_and_handles_empty() {
        let msgs: Vec<_> = (0..5).map(|i| msg("user", &format!("m{i}"))).collect();
        let mut pager = HistoryPager::new(msgs, 2);
        pager.next_page();
        let rest = pager.rest();
        assert_eq!(rest.len(), 3);
        assert_eq!(rest[0].text, "m0");
        assert!(!pager.has_more());
        let mut empty = HistoryPager::new(vec![], 0);
        assert!(!empty.has_more());
        assert!(empty.next_page().is_empty());
    }
}
