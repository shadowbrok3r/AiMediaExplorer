//! Tests of the transcript model: event mapping, rows, history merging.

use super::*;
use zeroclaw_client::ApprovalRequest;

const SCOPE: &str = "s1:1";

fn text(turn: u64, segment: u32, t: &str) -> ChatEv {
    ChatEv::Text { turn, segment, text: t.into() }
}

fn done(turn: u64, full: &str) -> ChatEv {
    ChatEv::Done { turn, full_response: full.into(), usage: Usage { input_tokens: Some(10), context_window: Some(100), last_input_tokens: Some(10), ..Default::default() } }
}

fn apply(t: &mut Transcript, ev: ChatEv) -> Applied {
    t.apply(SCOPE, &ev, 1_000)
}

fn kinds(t: &Transcript) -> Vec<&'static str> {
    t.entries.iter().map(|e| e.kind.tag()).collect()
}

#[test]
fn streamed_text_grows_one_entry_per_segment() {
    let mut t = Transcript::default();
    apply(&mut t, text(0, 0, "Hel"));
    apply(&mut t, text(0, 0, "lo"));
    apply(&mut t, text(0, 1, "after the tool"));
    assert_eq!(kinds(&t), ["assistant", "assistant"]);
    assert!(matches!(&t.entries[0].kind, Kind::Assistant { text, streaming: true } if text == "Hello"));
    assert!(t.running);
}

#[test]
fn done_finishes_the_turn_with_the_authoritative_text_and_usage() {
    let mut t = Transcript::default();
    t.push_user(format!("{SCOPE}:u0"), "hi", vec![], vec![], 1);
    apply(&mut t, ChatEv::Thinking { turn: 0, segment: 0, text: "hmm".into() });
    apply(&mut t, text(0, 0, "Hel"));
    let applied = apply(&mut t, done(0, "Hello there"));
    assert!(!t.running);
    assert!(matches!(&t.entries[2].kind, Kind::Assistant { text, streaming: false } if text == "Hello there"));
    assert!(matches!(&t.entries[1].kind, Kind::Reasoning { streaming: false, .. }));
    assert_eq!(applied.finished.as_deref(), Some("Hello there"));
    assert_eq!(applied.persist.iter().map(|r| r.kind.as_str()).collect::<Vec<_>>(), ["reasoning", "assistant"], "the user row was saved when it was sent");
    assert_eq!(t.usage.as_ref().and_then(|u| u.context_fraction()), Some(0.1));
    assert!(apply(&mut t, done(0, "Hello there")).persist.is_empty(), "rows are saved once");
}

#[test]
fn a_turn_with_only_a_done_frame_still_shows_the_answer() {
    let mut t = Transcript::default();
    apply(&mut t, done(0, "full answer"));
    assert!(matches!(&t.entries[0].kind, Kind::Assistant { text, .. } if text == "full answer"));
}

#[test]
fn a_tool_call_runs_then_completes_with_its_result() {
    let mut t = Transcript::default();
    apply(&mut t, ChatEv::ToolCall { turn: 0, index: 1, id: Some("c1".into()), name: "aimedia__library_stats".into(), args: json!({"a":1}) });
    assert!(matches!(&t.entries[0].kind, Kind::ToolCall { state: CallState::Running, .. }));
    apply(&mut t, ChatEv::ToolResult { turn: 0, index: 1, id: Some("c1".into()), name: "aimedia__library_stats".into(), output: "12 items".into() });
    assert!(matches!(&t.entries[0].kind, Kind::ToolCall { state: CallState::Done, .. }));
    assert!(matches!(&t.entries[1].kind, Kind::ToolResult { output, .. } if output == "12 items"));
    assert_eq!(tool_name("aimedia__library_stats"), "library_stats");
    assert_eq!(tool_name("plain"), "plain");
}

#[test]
fn an_approval_card_counts_down_and_expires() {
    let mut t = Transcript::default();
    let req = ApprovalRequest { turn: 0, request_id: "r1".into(), tool: "aimedia__describe_image".into(), summary: "describe a.jpg".into(), arguments: json!({}), timeout_secs: 300 };
    apply(&mut t, ChatEv::Approval(req));
    assert_eq!(t.pending_approvals(), 1);
    assert!(!t.expire_approvals(1_000 + 299_000));
    assert!(t.expire_approvals(1_000 + 300_000));
    assert_eq!(t.pending_approvals(), 0);
    assert!(matches!(&t.entries[0].kind, Kind::Approval { decision: Decision::Expired, .. }));
}

#[test]
fn an_approval_decision_sticks() {
    let mut t = Transcript::default();
    let req = ApprovalRequest { turn: 0, request_id: "r1".into(), tool: "t".into(), summary: "s".into(), arguments: Value::Null, timeout_secs: 60 };
    apply(&mut t, ChatEv::Approval(req));
    t.decide("r1", Decision::Denied);
    t.decide("r1", Decision::Approved);
    assert!(matches!(&t.entries[0].kind, Kind::Approval { decision: Decision::Denied, .. }), "the first answer stands");
}

#[test]
fn a_turn_ending_error_settles_the_turn_and_is_kept() {
    let mut t = Transcript::default();
    apply(&mut t, text(0, 0, "partial"));
    let applied = apply(&mut t, ChatEv::Error { turn: 0, code: Some("PROVIDER_ERROR".into()), message: "model unavailable".into(), ends_turn: true });
    assert!(!t.running);
    assert!(matches!(&t.entries[0].kind, Kind::Assistant { streaming: false, .. }));
    assert!(matches!(&t.entries[1].kind, Kind::Notice { level: Level::Error, text, .. } if text == "PROVIDER_ERROR: model unavailable"));
    assert_eq!(applied.persist.len(), 2);
    let soft = apply(&mut t, ChatEv::Error { turn: 1, code: None, message: "busy".into(), ends_turn: false });
    assert!(soft.persist.is_empty());
}

#[test]
fn a_resumed_turn_is_rebuilt_from_the_replay() {
    let mut t = Transcript::default();
    t.push_user(format!("{SCOPE}:u0"), "hi", vec![], vec![], 1);
    apply(&mut t, text(0, 0, "Hello "));
    apply(&mut t, ChatEv::TurnResumed { turn: 0 });
    assert_eq!(kinds(&t), ["user"], "what was drawn for the turn is dropped, the question stays");
    apply(&mut t, text(0, 0, "Hello "));
    apply(&mut t, text(0, 0, "world"));
    apply(&mut t, done(0, "Hello world"));
    assert!(matches!(&t.entries[1].kind, Kind::Assistant { text, .. } if text == "Hello world"));
}

#[test]
fn a_lost_turn_drops_its_partial_entries_and_asks_for_the_history() {
    let mut t = Transcript::default();
    t.push_user(format!("{SCOPE}:u0"), "hi", vec![], vec![], 1);
    apply(&mut t, text(0, 0, "partial answer"));
    let applied = apply(&mut t, ChatEv::TurnLost { turn: 0 });
    assert!(applied.reload_history && !t.running);
    assert_eq!(kinds(&t), ["user", "notice"], "the partial answer comes back from the gateway's history, not twice");
    assert!(matches!(&t.entries[1].kind, Kind::Notice { action: Some(NoticeAction::ReloadHistory), .. }));
}

#[test]
fn a_failed_send_gives_the_draft_back_and_removes_the_question() {
    let mut t = Transcript::default();
    t.push_user(format!("{SCOPE}:u0"), "my question", vec![], vec![], 1);
    let applied = apply(&mut t, ChatEv::SendFailed { text: "my question".into(), error: "not connected".into() });
    assert_eq!(applied.restore_draft.as_deref(), Some("my question"));
    assert_eq!(kinds(&t), ["notice"]);
    assert!(!t.running);
}

#[test]
fn a_refused_token_shows_a_re_pair_card_once() {
    let mut t = Transcript::default();
    let first = apply(&mut t, ChatEv::AuthRefused);
    apply(&mut t, ChatEv::AuthRefused);
    assert!(first.auth_refused);
    assert_eq!(t.entries.len(), 1);
    assert!(matches!(&t.entries[0].kind, Kind::Notice { action: Some(NoticeAction::RePair), .. }));
}

#[test]
fn a_stopped_turn_keeps_what_streamed_and_says_so() {
    let mut t = Transcript::default();
    apply(&mut t, text(0, 0, "half an ans"));
    let applied = apply(&mut t, ChatEv::Aborted { turn: 0 });
    assert!(!t.running);
    assert_eq!(kinds(&t), ["assistant", "notice"]);
    assert_eq!(applied.persist.len(), 2);
}

#[test]
fn plans_update_in_place() {
    let mut t = Transcript::default();
    let row = |c: &str, s: &str| PlanRow { content: c.into(), status: Some(s.into()) };
    apply(&mut t, ChatEv::Plan { turn: 0, rows: vec![row("a", "pending")] });
    apply(&mut t, ChatEv::Plan { turn: 0, rows: vec![row("a", "done"), row("b", "pending")] });
    assert_eq!(t.entries.len(), 1);
    assert!(matches!(&t.entries[0].kind, Kind::Plan { rows } if rows.len() == 2));
}

#[test]
fn rows_round_trip_every_kind() {
    let req = ApprovalRequest { turn: 0, request_id: "r1".into(), tool: "t".into(), summary: "sum".into(), arguments: json!({"x":1}), timeout_secs: 60 };
    let mut t = Transcript::default();
    t.push_user(format!("{SCOPE}:u"), "q", vec!["/a.jpg".into()], vec!["@selection".into()], 5);
    apply(&mut t, ChatEv::Thinking { turn: 0, segment: 0, text: "think".into() });
    apply(&mut t, ChatEv::ToolCall { turn: 0, index: 1, id: Some("c".into()), name: "n".into(), args: json!({"k":"v"}) });
    apply(&mut t, ChatEv::ToolResult { turn: 0, index: 1, id: Some("c".into()), name: "n".into(), output: "out".into() });
    apply(&mut t, ChatEv::Approval(req));
    t.decide("r1", Decision::Always);
    apply(&mut t, ChatEv::Plan { turn: 0, rows: vec![PlanRow { content: "p".into(), status: None }] });
    apply(&mut t, text(0, 1, "answer"));
    apply(&mut t, done(0, "answer"));
    for (i, entry) in t.entries.iter().enumerate() {
        let row = to_row(entry).unwrap_or_else(|| panic!("{:?} has no row", entry.kind));
        assert!(crate::database::assistant_store::KINDS.contains(&row.kind.as_str()));
        let stored = AssistantMessage {
            id: surrealdb::types::RecordId::new("assistant_messages", "x"),
            session_ref: surrealdb::types::RecordId::new("assistant_sessions", "s"),
            role: row.role.clone(),
            content: row.content.clone(),
            attachments: row.attachments.clone(),
            attachments_refs: None,
            created: None,
            kind: Some(row.kind.clone()),
            turn: row.turn,
            meta: row.meta.clone(),
        };
        let back = entry_from_row("s", i, &stored);
        assert_eq!(back.kind.tag(), entry.kind.tag());
        match (&entry.kind, &back.kind) {
            (Kind::Approval { decision: a, .. }, Kind::Approval { decision: b, request_id, .. }) => assert_eq!((a, request_id.as_str()), (b, "r1")),
            (Kind::ToolCall { args: a, name: an, .. }, Kind::ToolCall { args: b, name: bn, .. }) => assert_eq!((a, an), (b, bn)),
            (Kind::User { attachments: a, chips: ac, .. }, Kind::User { attachments: b, chips: bc, .. }) => assert_eq!((a, ac), (b, bc)),
            (Kind::Plan { rows: a }, Kind::Plan { rows: b }) => assert_eq!(a, b),
            (a, b) => assert_eq!(a.tag(), b.tag()),
        }
    }
}

#[test]
fn a_legacy_row_without_a_kind_reads_as_a_plain_message() {
    let row = |role: &str| AssistantMessage {
        id: surrealdb::types::RecordId::new("assistant_messages", "x"),
        session_ref: surrealdb::types::RecordId::new("assistant_sessions", "s"),
        role: role.into(),
        content: "old".into(),
        attachments: None,
        attachments_refs: None,
        created: None,
        kind: None,
        turn: None,
        meta: None,
    };
    assert!(matches!(entry_from_row("s", 0, &row("user")).kind, Kind::User { .. }));
    assert!(matches!(entry_from_row("s", 1, &row("assistant")).kind, Kind::Assistant { .. }));
}

#[test]
fn history_items_become_entries_and_merge_over_the_cache() {
    let item = |id: &str, kind: HistoryKind, text: &str, at: &str| HistoryItem { id: id.into(), kind, text: text.into(), name: None, args: None, call_id: None, images: vec![], created_at: Some(at.into()) };
    let mut t = Transcript::default();
    let mut old = Entry::new("s:old".into(), None, 1_000, Kind::Assistant { text: "cached older".into(), streaming: false });
    old.persisted = true;
    let mut dup = Entry::new("s:dup".into(), None, 5_000_000_000_000, Kind::Assistant { text: "cached dup".into(), streaming: false });
    dup.persisted = true;
    t.entries = vec![old, dup, Entry::new("s:live".into(), Some(2), 9, Kind::Assistant { text: "in progress".into(), streaming: true })];
    let server = vec![
        item("h0", HistoryKind::User, "hello", "2026-09-11T01:00:00+00:00"),
        item("h1", HistoryKind::Assistant, "hi", "2026-09-11T01:00:05+00:00"),
        item("h2", HistoryKind::Other, "system", "2026-09-11T01:00:06+00:00"),
    ];
    t.apply(SCOPE, &ChatEv::History { items: server, has_more: true, total: 40, initial: true }, 1);
    let texts: Vec<&str> = t.entries.iter().filter_map(Entry::text).collect();
    assert_eq!(texts, ["cached older", "hello", "hi", "in progress"], "the cached copy of server history is replaced, the live turn stays last");
    assert!(t.has_more && t.total == 40);
    let older = vec![item("h-1", HistoryKind::User, "before", "2026-09-10T01:00:00+00:00")];
    t.apply(SCOPE, &ChatEv::History { items: older, has_more: false, total: 40, initial: false }, 1);
    assert_eq!(t.entries[0].text(), Some("before"));
    assert!(!t.has_more);
}

#[test]
fn history_images_are_noted_not_inlined() {
    let item = HistoryItem {
        id: "h0".into(),
        kind: HistoryKind::User,
        text: "look".into(),
        name: None,
        args: None,
        call_id: None,
        images: vec![ImageRef::Inline { mime: "image/png".into(), base64_len: 4 }],
        created_at: None,
    };
    let e = entry_from_history(SCOPE, &item).unwrap();
    assert_eq!(e.text(), Some("look\n[1 image]"));
}

#[test]
fn the_conversation_for_a_provider_is_bounded_and_ordered() {
    let mut t = Transcript::default();
    for i in 0..6 {
        t.push_user(format!("u{i}"), &format!("question {i}"), vec![], vec![], i);
        apply(&mut t, text(i, 0, &format!("answer {i}")));
        apply(&mut t, done(i, &format!("answer {i}")));
    }
    let all = t.conversation(10_000);
    assert_eq!(all.len(), 12);
    assert!(all[0].0 && !all[1].0);
    let few = t.conversation(40);
    assert!(few.len() < 12 && few.last().map(|c| c.1.as_str()) == Some("answer 5"), "{few:?}");
    assert_eq!(t.last_user_text(), Some("question 5"));
    assert_eq!(t.first_exchange(), Some(("question 0".to_string(), "answer 0".to_string())));
}
