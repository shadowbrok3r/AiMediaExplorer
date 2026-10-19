mod common;

use common::*;
use zeroclaw_client::event::event_kind;
use zeroclaw_client::mock::{ConnScript, MockGateway};
use zeroclaw_client::replay::Recording;
use zeroclaw_client::{ConnectOptions, Direction, Event, ServerFrame, Translator};

const PLAIN: &str = include_str!("fixtures/plain_turn.jsonl");
const TOOL_APPROVAL: &str = include_str!("fixtures/tool_approval.jsonl");
const RESUME: &str = include_str!("fixtures/resume_mid_turn.jsonl");
const ERRORS: &str = include_str!("fixtures/errors_and_drift.jsonl");
const ABORTED: &str = include_str!("fixtures/aborted_turn.jsonl");

fn recording(text: &str) -> Recording {
    Recording::parse(text).expect("fixture parses")
}

/// Event kinds the pure translator produces for a recording, without any socket.
fn offline_kinds(rec: &Recording) -> Vec<&'static str> {
    let mut t = Translator::new();
    let mut out = Vec::new();
    for frame in rec.server_frames() {
        t.feed(ServerFrame::from_value(frame.clone()), &mut out);
    }
    out.iter().map(event_kind).collect()
}

#[test]
fn every_fixture_parses_and_round_trips() {
    for (name, text) in [("plain", PLAIN), ("tool_approval", TOOL_APPROVAL), ("resume", RESUME), ("errors", ERRORS), ("aborted", ABORTED)] {
        let rec = recording(text);
        assert!(rec.entries.len() >= 5, "{name}");
        assert_eq!(Recording::parse(&rec.to_jsonl()).unwrap(), rec, "{name}");
        assert!(rec.entries.windows(2).all(|w| w[0].t_ms <= w[1].t_ms), "{name}: timestamps must not go backwards");
        assert_eq!(rec.entries[0].dir, Direction::In, "{name}: a gateway starts with session_start");
        assert_eq!(rec.entries[0].frame["type"], "session_start", "{name}");
    }
}

#[test]
fn documented_frames_in_the_fixtures_parse_typed_and_only_the_drift_fixture_has_unknowns() {
    for (name, text, expected_unknown) in [
        ("plain", PLAIN, 0),
        ("tool_approval", TOOL_APPROVAL, 0),
        ("resume", RESUME, 0),
        ("aborted", ABORTED, 0),
        ("errors", ERRORS, 3),
    ] {
        let unknown = recording(text)
            .server_frames()
            .filter(|f| matches!(ServerFrame::from_value((*f).clone()), ServerFrame::Unknown(_)))
            .count();
        assert_eq!(unknown, expected_unknown, "{name}");
    }
}

#[test]
fn offline_translation_of_each_fixture() {
    assert_eq!(offline_kinds(&recording(PLAIN)), ["turn_started", "thinking", "thinking", "token", "token", "done"]);
    assert_eq!(
        offline_kinds(&recording(TOOL_APPROVAL)),
        ["turn_started", "tool_call", "tool_result", "tool_call", "approval_request", "tool_result", "plan", "token", "done"]
    );
    assert_eq!(
        offline_kinds(&recording(RESUME)),
        ["history_trimmed", "turn_resumed", "thinking", "tool_call", "tool_result", "token", "token", "done"]
    );
    assert_eq!(offline_kinds(&recording(ABORTED)), ["turn_started", "thinking", "token", "aborted"]);
    assert_eq!(offline_kinds(&recording(ERRORS)), ["error", "turn_started", "token", "unknown", "unknown", "unknown", "error"]);
}

async fn replay_plain(speed: f64) {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::replay(&recording(PLAIN), speed));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("find sunsets").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "thinking", "thinking", "token", "token", "done"]);
    assert_eq!(text_of(&got), "Found 12 sunsets.");
    match got.last().unwrap() {
        Event::Done { full_response, usage, .. } => {
            assert_eq!(full_response, "Found 12 sunsets.");
            assert_eq!(usage.input_tokens, Some(1234));
            assert_eq!(usage.output_tokens, Some(18));
            assert_eq!(usage.cost_usd, None);
            assert_eq!(usage.model.as_deref(), Some("qwen3.8-27b"));
            assert_eq!(usage.context_window, Some(131_072));
        }
        other => panic!("{other:?}"),
    }
    let sent: Vec<_> = mock.received().iter().map(|r| r.frame["type"].as_str().unwrap().to_string()).collect();
    assert_eq!(sent, ["connect", "message"]);
    assert_eq!(mock.received_of_type("message")[0]["content"], "find sunsets");
}

#[tokio::test]
async fn replay_plain_turn_instantly() {
    replay_plain(0.0).await;
}

#[tokio::test]
async fn replay_plain_turn_at_recorded_pace() {
    replay_plain(20.0).await;
}

#[tokio::test]
async fn replay_tool_calls_and_an_approval() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::replay(&recording(TOOL_APPROVAL), 0.0));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("edit the sunset shoot").await.unwrap();
    let before = collect_until(&mut events, |e| matches!(e, Event::ApprovalRequest(_))).await;
    assert_eq!(kinds(&before), ["turn_started", "tool_call", "tool_result", "tool_call", "approval_request"]);
    let Event::ApprovalRequest(req) = before.last().unwrap() else { unreachable!() };
    assert_eq!(req.request_id, "r1");
    assert_eq!(req.tool, "aimedia__image_edit_run");
    assert_eq!(req.arguments["prompt"], "warmer");
    session.approve(&req.request_id).await.unwrap();
    let after = until_done(&mut events).await;
    assert_eq!(kinds(&after), ["tool_result", "plan", "token", "done"]);
    assert!(matches!(&after[0], Event::ToolResult { index: 2, output, .. } if output == "job 7 started"));
    assert!(matches!(&after[1], Event::Plan { entries, .. } if entries.len() == 2 && entries[1].status.as_deref() == Some("in_progress")));
    assert_eq!(
        mock.received_of_type("approval_response")[0],
        serde_json::json!({"type":"approval_response","request_id":"r1","decision":"approve"})
    );
}

#[tokio::test]
async fn replay_turn_resume_after_reconnect() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::replay(&recording(RESUME), 0.0));
    let opts = ConnectOptions { session_id: Some("fixture-session".into()), ..ConnectOptions::default() };
    let (_session, mut events) = zeroclaw_client::ChatSession::connect(mock.config(), opts).await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(
        turn_kinds(&got),
        ["history_trimmed", "turn_resumed", "thinking", "tool_call", "tool_result", "token", "token", "done"]
    );
    assert!(matches!(&got[0], Event::Connected { resumed: true, message_count: 4, name: Some(n), .. } if n == "Sunset shoot"));
    assert!(matches!(&got[1], Event::HistoryTrimmed { dropped_messages: 6, kept_turns: 3, .. }));
    assert!(matches!(&got[2], Event::TurnResumed { turn: 0, frames: 4 }));
    assert_eq!(text_of(&got), "The library has 1042 items.");
}

#[tokio::test]
async fn replay_errors_and_unknown_frames() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::replay(&recording(ERRORS), 0.0));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("go").await.unwrap();
    let got = collect_until(&mut events, |e| matches!(e, Event::Error { ends_turn: true, .. })).await;
    assert_eq!(kinds(&got), ["error", "turn_started", "token", "unknown", "unknown", "unknown", "error"]);
    assert!(matches!(&got[0], Event::Error { code: Some(c), ends_turn: false, .. } if c == "STEERING_QUEUE_FULL"));
    assert!(matches!(&got[6], Event::Error { code: Some(c), message, ends_turn: true, .. } if c == "PROVIDER_ERROR" && message.contains("502")));
}

#[tokio::test]
async fn replay_aborted_turn() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::replay(&recording(ABORTED), 0.0));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("write a long essay").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "thinking", "token", "aborted"]);
}

#[tokio::test]
async fn a_live_recording_replays_to_the_same_events() {
    // Record a scripted exchange through the raw tap, then replay the recording and compare.
    let source = MockGateway::start().await;
    source.script_turn(
        zeroclaw_client::mock::Turn::new().thinking("hm").tool_call("c1", "t", serde_json::json!({})).tool_result("c1", "t", "ok").chunk("done").done("done"),
    );
    let opts = ConnectOptions { raw_tap: true, ..ConnectOptions::default() };
    let (session, mut events) = zeroclaw_client::ChatSession::connect(source.config(), opts).await.unwrap();
    session.send_text("go").await.unwrap();
    let live = collect_until(&mut events, |e| matches!(e, Event::Done { .. })).await;
    let raws: Vec<_> = live.iter().filter_map(|e| if let Event::Raw(r) = e { Some(r.clone()) } else { None }).collect();
    let recorded = Recording::from_raw(&raws);
    let live_kinds: Vec<_> = live.iter().filter(|e| !matches!(e, Event::Raw(_) | Event::Connected { .. })).map(event_kind).collect();

    let replay = MockGateway::start().await;
    replay.script_connection(ConnScript::replay(&Recording::parse(&recorded.to_jsonl()).unwrap(), 0.0));
    let (session, mut events) = connect(replay.config()).await;
    session.send_text("go").await.unwrap();
    let again = until_done(&mut events).await;
    assert_eq!(kinds(&again), live_kinds);
}
