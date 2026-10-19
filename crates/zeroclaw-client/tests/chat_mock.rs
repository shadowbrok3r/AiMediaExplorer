mod common;

use std::time::Duration;

use common::*;
use serde_json::json;
use zeroclaw_client::mock::{AuthUsed, ConnScript, MockAuth, MockGateway, MockOptions, Step, Turn};
use zeroclaw_client::replay::Recording;
use zeroclaw_client::{
    AbortOutcome, ChatSession, ConnState, ConnectOptions, Direction, DisconnectReason, Error, Event, InlineImage, WsAuth,
};

const PNG: &[u8] = b"\x89PNG\r\n\x1a\nrest";

fn agent_start() -> serde_json::Value {
    json!({"type":"agent_start","model_provider":"mock","model":"mock-model"})
}

#[tokio::test]
async fn plain_turn_streams_typed_events() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().frame(agent_start()).thinking("hm").chunk("Hello ").chunk("world").done("Hello world"));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("hi").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "thinking", "token", "token", "done"]);
    assert_eq!(text_of(&got), "Hello world");
    match got.last().unwrap() {
        Event::Done { turn, full_response, usage } => {
            assert_eq!(*turn, 0);
            assert_eq!(full_response, "Hello world");
            assert_eq!(usage.input_tokens, Some(100));
            assert_eq!(usage.model.as_deref(), Some("mock-model"));
            assert_eq!(usage.context_fraction(), Some(0.1));
        }
        other => panic!("{other:?}"),
    }
    let connect_frame = &mock.received_of_type("connect")[0];
    assert_eq!(connect_frame["device_name"], "zeroclaw-client");
    assert_eq!(connect_frame["capabilities"], json!(["text"]));
    assert_eq!(mock.received_of_type("message")[0]["content"], "hi");
    let conn = &mock.connections()[0];
    assert_eq!(conn.agent.as_deref(), Some("default"));
    assert_eq!(conn.auth, AuthUsed::Header);
    assert_eq!(session.state(), ConnState::Open);
}

#[tokio::test]
async fn unscripted_messages_are_echoed_and_turns_are_numbered() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    for (i, text) in ["one", "two"].into_iter().enumerate() {
        session.send_text(text).await.unwrap();
        let got = until_done(&mut events).await;
        assert_eq!(text_of(&got), format!("echo: {text}"));
        assert!(matches!(got[0], Event::TurnStarted { turn } if turn == i as u64));
    }
}

#[tokio::test]
async fn tool_calls_and_results_are_display_events_with_indexes() {
    let mock = MockGateway::start().await;
    mock.script_turn(
        Turn::new()
            .chunk("Looking. ")
            .tool_call("c1", "aimedia__library_search_semantic", json!({"query":"sunset"}))
            .tool_result("c1", "aimedia__library_search_semantic", "5 results")
            .chunk("Found 5.")
            .done("Looking. Found 5."),
    );
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("find sunsets").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "token", "tool_call", "tool_result", "token", "done"]);
    assert!(matches!(&got[2], Event::ToolCall { index: 1, name, args, .. }
        if name == "aimedia__library_search_semantic" && args["query"] == "sunset"));
    assert!(matches!(&got[3], Event::ToolResult { index: 1, output, .. } if output == "5 results"));
    assert!(matches!(&got[4], Event::Token { segment: 1, .. }));
}

#[tokio::test]
async fn approval_round_trip_for_each_decision() {
    let mock = MockGateway::start().await;
    for id in ["r1", "r2", "r3"] {
        mock.script_turn(Turn::new().approval(id, "aimedia__image_edit_run", "edit a.jpg").chunk("ok").done("ok"));
    }
    let (session, mut events) = connect(mock.config()).await;
    for (i, expected) in ["approve", "always", "deny"].into_iter().enumerate() {
        session.send_text("edit").await.unwrap();
        let before = collect_until(&mut events, |e| matches!(e, Event::ApprovalRequest(_))).await;
        let Event::ApprovalRequest(req) = before.last().unwrap() else { unreachable!() };
        assert_eq!(req.request_id, format!("r{}", i + 1));
        assert_eq!(req.tool, "aimedia__image_edit_run");
        assert_eq!(req.summary, "edit a.jpg");
        assert_eq!(req.timeout_secs, 300);
        match expected {
            "approve" => session.approve(&req.request_id).await.unwrap(),
            "always" => session.approve_always(&req.request_id).await.unwrap(),
            _ => session.deny(&req.request_id).await.unwrap(),
        }
        let after = until_done(&mut events).await;
        assert_eq!(kinds(&after), ["token", "done"]);
    }
    let responses = mock.received_of_type("approval_response");
    let decisions: Vec<_> = responses.iter().map(|r| (r["request_id"].as_str().unwrap(), r["decision"].as_str().unwrap())).collect();
    assert_eq!(decisions, [("r1", "approve"), ("r2", "always"), ("r3", "deny")]);
}

#[tokio::test]
async fn abort_goes_over_rest_and_reports_the_outcome() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().frame(agent_start()).chunk("Working").delay_ms(30_000));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("long task").await.unwrap();
    collect_until(&mut events, |e| matches!(e, Event::Token { .. })).await;
    assert_eq!(session.abort().await.unwrap(), AbortOutcome::Aborted);
    assert_eq!(session.abort().await.unwrap(), AbortOutcome::NothingRunning);
    let abort_calls: Vec<_> = mock.requests().into_iter().filter(|r| r.path.ends_with("/abort")).collect();
    assert_eq!(abort_calls.len(), 2);
    assert!(abort_calls.iter().all(|r| r.method == "POST" && r.authorized));
    assert!(abort_calls[0].path.contains(session.session_id()));
}

#[tokio::test]
async fn aborted_frame_ends_the_turn() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("Work").aborted());
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("go").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "token", "aborted"]);
    session.send_text("again").await.unwrap();
    let got = until_done(&mut events).await;
    assert!(matches!(got[0], Event::TurnStarted { turn: 1 }));
}

#[tokio::test]
async fn errors_end_the_turn_only_for_the_listed_codes() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().error("STEERING_QUEUE_FULL", "full").chunk("x").error("PROVIDER_ERROR", "boom"));
    mock.script_turn(Turn::reply("next"));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("go").await.unwrap();
    let got = collect_until(&mut events, |e| matches!(e, Event::Error { ends_turn: true, .. })).await;
    assert_eq!(kinds(&got), ["error", "turn_started", "token", "error"]);
    assert!(matches!(&got[0], Event::Error { code: Some(c), ends_turn: false, .. } if c == "STEERING_QUEUE_FULL"));
    session.send_text("again").await.unwrap();
    let got = until_done(&mut events).await;
    assert!(matches!(got[0], Event::TurnStarted { turn: 1 }), "the failed turn was numbered and closed");
}

#[tokio::test]
async fn unknown_and_drifted_frames_pass_through_as_events() {
    let mock = MockGateway::start().await;
    mock.script_turn(
        Turn::new()
            .chunk("a")
            .frame(json!({"type":"cron_result","job":"nightly"}))
            .frame(json!({"type":"tool_call","id":{"nested":1},"name":"x","args":{}}))
            .frame(json!({"type":"fork_feature","payload":[1,2,3]}))
            .done("a"),
    );
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("go").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(kinds(&got), ["turn_started", "token", "unknown", "unknown", "unknown", "done"]);
    let unknown: Vec<_> = got.iter().filter_map(|e| if let Event::Unknown { kind, .. } = e { Some(kind.as_str()) } else { None }).collect();
    assert_eq!(unknown, ["cron_result", "tool_call", "fork_feature"]);
    match &got[3] {
        Event::Unknown { raw, .. } => assert_eq!(raw["id"], json!({"nested":1}), "drifted frames keep their raw payload"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn bad_token_is_unauthorized_and_not_retried() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.set_token("wrong");
    let err = ChatSession::connect(cfg.clone(), ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Unauthorized), "{err:?}");
    assert!(err.is_auth() && !err.is_retryable());
    let attempts = mock.connections();
    assert_eq!(attempts.len(), 2, "header attempt, then the query fallback");
    assert!(attempts.iter().all(|c| !c.accepted && c.status == 401));

    cfg.ws_auth = WsAuth::Header;
    let before = mock.connections().len();
    assert!(matches!(ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap(), Error::Unauthorized));
    assert_eq!(mock.connections().len(), before + 1, "no fallback when the mode is pinned");
}

#[tokio::test]
async fn missing_token_is_unauthorized_without_leaking_anything() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.set_token("");
    let err = ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Unauthorized));
    assert_eq!(mock.connections()[0].auth, AuthUsed::Anonymous);
}

#[tokio::test]
async fn forbidden_and_other_rejections_are_typed() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::new().reject(403, "nope"));
    let err = ChatSession::connect(mock.config(), ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Forbidden), "{err:?}");

    mock.script_connection(ConnScript::new().reject(503, "overloaded"));
    let err = ChatSession::connect(mock.config(), ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(&err, Error::Rejected { status: 503, message } if message == "overloaded"), "{err:?}");
    assert!(err.is_retryable());
}

#[tokio::test]
async fn unknown_or_missing_agent_is_a_clear_error() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.agent = "ghost".into();
    let err = ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(&err, Error::Rejected { status: 400, message } if message.contains("Unknown agent")), "{err:?}");
    assert!(!err.is_retryable());

    let mut cfg = mock.config();
    cfg.agent = " ".into();
    let before = mock.connections().len();
    let err = ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Config(_)));
    assert_eq!(mock.connections().len(), before, "rejected before any network use");
}

#[tokio::test]
async fn header_stripping_proxy_falls_back_to_query_and_remembers_it() {
    let mock = MockGateway::start_with(MockOptions { ws_auth: MockAuth::QueryOnly, ..MockOptions::default() }).await;
    let (_session, mut events) = connect(mock.config()).await;
    let seen: Vec<_> = mock.connections().iter().map(|c| (c.auth, c.status)).collect();
    assert_eq!(seen, [(AuthUsed::Anonymous, 401), (AuthUsed::Query, 101)]);

    mock.kick();
    let got = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(matches!(got.last(), Some(Event::Connected { reconnect: true, .. })));
    let seen: Vec<_> = mock.connections().iter().map(|c| c.auth).collect();
    assert_eq!(seen, [AuthUsed::Anonymous, AuthUsed::Query, AuthUsed::Query], "the reconnect went straight to the working mode");
}

#[tokio::test]
async fn no_token_gateway_connects_anonymously() {
    let mock = MockGateway::start_with(MockOptions { token: None, ..MockOptions::default() }).await;
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("hi").await.unwrap();
    until_done(&mut events).await;
    assert_eq!(mock.connections()[0].auth, AuthUsed::Anonymous);
}

#[tokio::test]
async fn session_ids_resume_and_mismatches_are_refused() {
    let mock = MockGateway::start().await;
    let (first, _events) = connect(mock.config()).await;
    let id = first.session_id().to_string();
    first.close().await;

    let opts = ConnectOptions { session_id: Some(id.clone()), ..ConnectOptions::default() };
    let (second, mut events) = ChatSession::connect(mock.config(), opts).await.unwrap();
    assert_eq!(second.session_id(), id);
    let got = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(matches!(got.last(), Some(Event::Connected { resumed: true, session_id, .. }) if *session_id == id));
    assert_eq!(mock.connections().last().unwrap().requested_session.as_deref(), Some(id.as_str()));

    mock.script_connection(ConnScript::new().wrong_session("someone-else"));
    let opts = ConnectOptions { session_id: Some("wanted".into()), ..ConnectOptions::default() };
    let err = ChatSession::connect(mock.config(), opts).await.err().unwrap();
    assert!(
        matches!(&err, Error::SessionMismatch { requested, received } if requested == "wanted" && received == "someone-else"),
        "{err:?}"
    );
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn mid_turn_disconnect_reconnects_and_replays_the_turn() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().frame(agent_start()).chunk("Hel").chunk("lo ").drop_connection());
    let (session, mut events) = connect(mock.config()).await;
    mock.script_connection(ConnScript::new().resume(
        vec![json!({"type":"chunk","content":"Hello "})],
        Turn::new().chunk("world").done("Hello world"),
    ));
    session.send_text("hi").await.unwrap();
    let got = until_done(&mut events).await;
    assert_eq!(
        kinds(&got),
        [
            "turn_started",
            "token",
            "token",
            "disconnected",
            "reconnecting",
            "connected",
            "turn_resumed",
            "token",
            "token",
            "done"
        ]
    );
    assert!(matches!(&got[3], Event::Disconnected { will_retry: true, .. }));
    assert!(matches!(&got[5], Event::Connected { reconnect: true, resumed: true, .. }));
    // Clears the rendered text at each TurnResumed.
    let mut rendered = String::new();
    for e in &got {
        match e {
            Event::TurnResumed { .. } => rendered.clear(),
            Event::Token { text, .. } => rendered.push_str(text),
            _ => {}
        }
    }
    assert_eq!(rendered, "Hello world");
    let conns = mock.connections();
    assert_eq!(conns.len(), 2);
    assert_eq!(conns[1].requested_session.as_deref(), Some(session.session_id()), "the reconnect resumed the same session");
    assert_eq!(mock.received_of_type("message").len(), 1, "a prompt is never resent on reconnect");
    assert_eq!(session.state(), ConnState::Open);
}

#[tokio::test]
async fn turn_that_finished_while_disconnected_is_reported_lost() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().frame(agent_start()).chunk("partial").drop_connection());
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("hi").await.unwrap();
    collect_until(&mut events, |e| matches!(e, Event::Disconnected { .. })).await;
    mock.set_state(session.session_id(), "idle", None);
    let got = collect_until(&mut events, |e| matches!(e, Event::TurnLost { .. })).await;
    assert_eq!(kinds(&got), ["reconnecting", "connected", "turn_lost"]);
    assert!(matches!(got.last(), Some(Event::TurnLost { turn: 0 })));
    mock.script_turn(Turn::reply("fresh"));
    session.send_text("again").await.unwrap();
    let got = until_done(&mut events).await;
    assert!(matches!(got[0], Event::TurnStarted { turn: 1 }));
}

#[tokio::test]
async fn reconnect_gives_up_after_the_configured_attempts() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.reconnect.max_attempts = Some(2);
    let (session, mut events) = connect(cfg).await;
    drop(mock);
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["disconnected", "reconnecting", "reconnecting", "disconnected"]);
    assert!(matches!(&got[0], Event::Disconnected { will_retry: true, .. }));
    assert!(matches!(&got[1], Event::Reconnecting { attempt: 1, last_error: None, .. }));
    assert!(matches!(&got[2], Event::Reconnecting { attempt: 2, last_error: Some(_), .. }));
    assert!(matches!(&got[3], Event::Disconnected { will_retry: false, reason: DisconnectReason::Fatal(_) }));
    assert_eq!(session.state(), ConnState::Closed);
}

#[tokio::test]
async fn reconnect_can_be_disabled() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.reconnect.enabled = false;
    let (_session, mut events) = connect(cfg).await;
    mock.kick();
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["disconnected"]);
    assert!(matches!(&got[0], Event::Disconnected { will_retry: false, .. }));
    assert_eq!(mock.connections().len(), 1);
}

#[tokio::test]
async fn permanent_failure_during_reconnect_stops_retrying() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    mock.script_connection(ConnScript::new().reject(403, "revoked"));
    mock.kick();
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["disconnected", "reconnecting", "disconnected"]);
    assert!(matches!(&got[2], Event::Disconnected { will_retry: false, reason: DisconnectReason::Fatal(text) } if text.contains("403")));
    assert_eq!(mock.connections().len(), 2);
    assert_eq!(session.state(), ConnState::Closed);
}

#[tokio::test]
async fn gateway_errors_that_end_the_connection_are_not_retried() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::new().after_connect(
        Turn::new().error("NEEDS_ONBOARDING", "finish onboarding first").step(Step::Close),
    ));
    let (_session, mut events) = connect(mock.config()).await;
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["error", "disconnected"]);
    assert!(matches!(&got[0], Event::Error { code: Some(c), ends_turn: false, .. } if c == "NEEDS_ONBOARDING"));
    assert!(matches!(
        &got[1],
        Event::Disconnected { will_retry: false, reason: DisconnectReason::ServerError(code) } if code == "NEEDS_ONBOARDING"
    ));
    assert_eq!(mock.connections().len(), 1);
}

#[tokio::test]
async fn oversized_frame_is_reported_and_not_retried() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().frame(agent_start()).chunk("a").step(Step::Oversize(100_000)));
    let mut cfg = mock.config();
    cfg.max_frame_bytes = 20_000;
    let (session, mut events) = connect(cfg).await;
    session.send_text("go").await.unwrap();
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["turn_started", "token", "error", "disconnected"]);
    assert!(matches!(&got[2], Event::Error { code: Some(c), .. } if c == "FRAME_TOO_LARGE"));
    assert!(matches!(&got[3], Event::Disconnected { reason: DisconnectReason::FrameTooLarge, will_retry: false }));
    assert_eq!(mock.connections().len(), 1);
}

#[tokio::test]
async fn slow_upgrade_times_out_with_a_typed_error() {
    let mock = MockGateway::start().await;
    mock.script_connection(ConnScript::new().upgrade_delay(Duration::from_millis(800)));
    let mut cfg = mock.config();
    cfg.connect_timeout = Duration::from_millis(100);
    let err = ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Timeout("connect")), "{err:?}");
    assert!(err.is_retryable());
}

#[tokio::test]
async fn slow_streaming_is_delivered_incrementally() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("a").delay_ms(150).chunk("b").delay_ms(150).chunk("c").done("abc"));
    let (session, mut events) = connect(mock.config()).await;
    session.send_text("go").await.unwrap();
    let start = tokio::time::Instant::now();
    collect_until(&mut events, |e| matches!(e, Event::Token { text, .. } if text == "a")).await;
    assert!(start.elapsed() < Duration::from_millis(120), "the first token must not wait for the rest");
    let rest = until_done(&mut events).await;
    assert_eq!(text_of(&rest), "bc");
    assert!(start.elapsed() >= Duration::from_millis(250));
}

#[tokio::test]
async fn silent_server_is_detected_by_ping_timeout_and_reconnected() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().stall());
    let mut cfg = mock.config();
    cfg.ping_interval = Duration::from_millis(40);
    cfg.pong_timeout = Duration::from_millis(120);
    let (session, mut events) = connect(cfg).await;
    session.send_text("go").await.unwrap();
    let got = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(
        matches!(&got[0], Event::Disconnected { reason: DisconnectReason::PingTimeout, will_retry: true }),
        "{:?}",
        kinds(&got)
    );
    assert!(matches!(got.last(), Some(Event::Connected { reconnect: true, .. })));
}

#[tokio::test]
async fn healthy_server_answers_pings_so_an_idle_session_stays_up() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.ping_interval = Duration::from_millis(30);
    cfg.pong_timeout = Duration::from_millis(100);
    let (session, mut events) = connect(cfg).await;
    let idle = tokio::time::timeout(Duration::from_millis(500), events.next()).await;
    assert!(idle.is_err(), "unexpected event while idle: {idle:?}");
    session.send_text("still there").await.unwrap();
    assert_eq!(text_of(&until_done(&mut events).await), "echo: still there");
}

#[tokio::test]
async fn slow_consumer_gets_every_event_in_order() {
    let mock = MockGateway::start().await;
    let mut turn = Turn::new();
    let expected: String = (0..1500).map(|i| format!("{i},")).collect();
    for i in 0..1500 {
        turn = turn.chunk(&format!("{i},"));
    }
    mock.script_turn(turn.done(&expected));
    let mut cfg = mock.config();
    cfg.event_buffer = 16;
    let (session, mut events) = connect(cfg).await;
    session.send_text("count").await.unwrap();
    let mut text = String::new();
    let mut n = 0;
    loop {
        let e = tokio::time::timeout(WAIT, events.next()).await.expect("event").expect("open");
        match e {
            Event::Token { text: t, .. } => {
                text.push_str(&t);
                n += 1;
                if n % 100 == 0 {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            }
            Event::Done { full_response, .. } => {
                assert_eq!(full_response, expected);
                break;
            }
            _ => {}
        }
    }
    assert_eq!(text, expected);
}

#[tokio::test]
async fn a_stalled_consumer_does_not_look_like_a_dead_connection() {
    let mock = MockGateway::start().await;
    let mut turn = Turn::new();
    for _ in 0..3000 {
        turn = turn.chunk("x");
    }
    mock.script_turn(turn.done("done"));
    let mut cfg = mock.config();
    cfg.event_buffer = 16;
    cfg.ping_interval = Duration::from_millis(20);
    cfg.pong_timeout = Duration::from_millis(60);
    let (session, mut events) = connect(cfg).await;
    session.send_text("go").await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let got = until_done(&mut events).await;
    assert!(!got.iter().any(|e| matches!(e, Event::Disconnected { .. })), "reads were paused, not the connection lost");
    assert_eq!(text_of(&got).len(), 3000);
    assert_eq!(mock.connections().len(), 1);
}

#[tokio::test]
async fn sends_validate_and_never_queue_across_a_reconnect() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.max_outbound_bytes = 1000;
    cfg.reconnect.initial = Duration::from_millis(400);
    cfg.reconnect.max = Duration::from_millis(400);
    let (session, mut events) = connect(cfg).await;
    assert!(matches!(session.send_text("   ").await, Err(Error::EmptyMessage)));
    let big = "x".repeat(2000);
    assert!(matches!(session.send_text(&big).await, Err(Error::TooLarge { size: 2000, limit: 1000 })));
    assert!(mock.received_of_type("message").is_empty());

    mock.kick();
    collect_until(&mut events, |e| matches!(e, Event::Reconnecting { .. })).await;
    assert!(matches!(session.send_text("while down").await, Err(Error::NotConnected)));
    collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(mock.received_of_type("message").is_empty(), "the failed send was not replayed after the reconnect");
    session.send_text("after").await.unwrap();
    assert_eq!(text_of(&until_done(&mut events).await), "echo: after");
}

#[tokio::test]
async fn a_send_during_a_reconnect_attempt_fails_at_once_instead_of_going_out_late() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    mock.script_connection(ConnScript::new().upgrade_delay(Duration::from_millis(600)));
    mock.kick();
    collect_until(&mut events, |e| matches!(e, Event::Reconnecting { .. })).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    let started = tokio::time::Instant::now();
    assert!(matches!(session.send_text("late").await, Err(Error::NotConnected)));
    assert!(started.elapsed() < Duration::from_millis(300), "the refusal must not wait for the socket");
    collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(mock.received_of_type("message").is_empty(), "the refused message was sent after the reconnect");
}

#[tokio::test]
async fn close_during_a_reconnect_attempt_ends_the_session_promptly() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    mock.script_connection(ConnScript::new().upgrade_delay(Duration::from_secs(5)));
    mock.kick();
    collect_until(&mut events, |e| matches!(e, Event::Reconnecting { .. })).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = tokio::time::Instant::now();
    session.close().await;
    let got = collect_until(&mut events, |_| false).await;
    assert!(matches!(got.last(), Some(Event::Disconnected { reason: DisconnectReason::Requested, will_retry: false })));
    assert!(started.elapsed() < Duration::from_secs(2), "close waited for the pending connection attempt");
    assert_eq!(session.state(), ConnState::Closed);
}

#[tokio::test]
async fn images_ride_inside_the_message_as_markers() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    let image = InlineImage::from_bytes("a.png", PNG).unwrap();
    session.send_user_message("what is this", std::slice::from_ref(&image)).await.unwrap();
    until_done(&mut events).await;
    assert_eq!(mock.received_of_type("message")[0]["content"], format!("what is this {}", image.marker()));
    session.send_user_message("", std::slice::from_ref(&image)).await.unwrap();
    until_done(&mut events).await;
    assert_eq!(mock.received_of_type("message")[1]["content"], image.marker());
    assert!(matches!(session.send_user_message("", &[]).await, Err(Error::EmptyMessage)));
}

#[tokio::test]
async fn raw_tap_exposes_both_directions_as_a_replayable_recording() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("hi").done("hi"));
    let opts = ConnectOptions { raw_tap: true, ..ConnectOptions::default() };
    let (session, mut events) = ChatSession::connect(mock.config(), opts).await.unwrap();
    session.send_text("hello").await.unwrap();
    let got = collect_until(&mut events, |e| matches!(e, Event::Done { .. })).await;
    let raws: Vec<_> = got.iter().filter_map(|e| if let Event::Raw(r) = e { Some(r.clone()) } else { None }).collect();
    let seq: Vec<_> = raws.iter().map(|r| (r.dir, r.frame["type"].as_str().unwrap().to_string())).collect();
    assert_eq!(seq[0], (Direction::Out, "connect".into()));
    assert_eq!(seq[1], (Direction::In, "session_start".into()));
    assert!(seq.contains(&(Direction::In, "connected".into())));
    assert!(seq.contains(&(Direction::Out, "message".into())));
    assert_eq!(seq.last().unwrap(), &(Direction::In, "done".into()));
    assert!(raws.windows(2).all(|w| w[0].t_ms <= w[1].t_ms));
    let jsonl = Recording::from_raw(&raws).to_jsonl();
    assert_eq!(Recording::parse(&jsonl).unwrap().entries.len(), raws.len());
}

#[tokio::test]
async fn close_ends_the_stream_and_the_socket() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    wait_until("socket open", || mock.live_connections() == 1).await;
    session.close().await;
    let got = collect_until(&mut events, |_| false).await;
    assert_eq!(kinds(&got), ["disconnected"]);
    assert!(matches!(&got[0], Event::Disconnected { reason: DisconnectReason::Requested, will_retry: false }));
    assert_eq!(session.state(), ConnState::Closed);
    wait_until("socket closed", || mock.live_connections() == 0).await;
    assert!(matches!(session.send_text("late").await, Err(Error::Closed)));
}

#[tokio::test]
async fn dropping_the_last_session_handle_closes_the_socket() {
    let mock = MockGateway::start().await;
    let (session, mut events) = connect(mock.config()).await;
    let clone = session.clone();
    drop(session);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(mock.live_connections(), 1, "a clone still holds the session");
    drop(clone);
    wait_until("socket closed", || mock.live_connections() == 0).await;
    let got = collect_until(&mut events, |_| false).await;
    assert!(matches!(got.last(), Some(Event::Disconnected { will_retry: false, .. })));
}

#[tokio::test]
async fn forced_reconnect_skips_backoff_and_keeps_the_session() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.reconnect.initial = Duration::from_secs(30);
    cfg.reconnect.max = Duration::from_secs(30);
    let (session, mut events) = connect(cfg).await;
    session.reconnect().await.unwrap();
    let got = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert_eq!(kinds(&got), ["disconnected", "reconnecting", "connected"]);
    assert!(matches!(&got[0], Event::Disconnected { reason: DisconnectReason::Requested, will_retry: true }));
    session.send_text("back").await.unwrap();
    assert_eq!(text_of(&until_done(&mut events).await), "echo: back");
    assert_eq!(mock.connections().len(), 2);
}

#[tokio::test]
async fn named_sessions_and_their_history_are_reachable_from_the_session() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("Hello"));
    let opts = ConnectOptions { name: Some("Sunset shoot".into()), ..ConnectOptions::default() };
    let (session, mut events) = ChatSession::connect(mock.config(), opts).await.unwrap();
    let first = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(matches!(first.last(), Some(Event::Connected { name: Some(n), .. }) if n == "Sunset shoot"));
    assert_eq!(mock.connections()[0].name.as_deref(), Some("Sunset shoot"));

    session.send_text("hi there").await.unwrap();
    until_done(&mut events).await;
    let mut pager = session.history(10, false).await.unwrap();
    let rows = pager.next_page();
    assert_eq!(rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["hi there", "Hello"]);
    let info = session.rest().session(session.session_id()).await.unwrap().expect("listed");
    assert_eq!(info.name.as_deref(), Some("Sunset shoot"));
    assert_eq!(info.message_count, 2);
    assert!(session.rest().session("nope").await.unwrap().is_none());
}

#[tokio::test]
async fn reconnect_schedule_runs_on_a_paused_clock() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.reconnect = zeroclaw_client::ReconnectPolicy {
        enabled: true,
        initial: Duration::from_secs(1),
        max: Duration::from_secs(8),
        multiplier: 2.0,
        jitter: 0.0,
        max_attempts: Some(5),
    };
    let (_session, mut events) = connect(cfg).await;
    drop(mock);
    tokio::time::pause();
    let wall = std::time::Instant::now();
    let virtual_start = tokio::time::Instant::now();
    let got = collect_until(&mut events, |_| false).await;
    let delays: Vec<u64> = got
        .iter()
        .filter_map(|e| if let Event::Reconnecting { delay, .. } = e { Some(delay.as_secs()) } else { None })
        .collect();
    assert_eq!(delays, [1, 2, 4, 8, 8]);
    assert!(matches!(got.last(), Some(Event::Disconnected { will_retry: false, .. })));
    assert!(
        tokio::time::Instant::now() - virtual_start >= Duration::from_secs(23),
        "every backoff sleep must have elapsed on the paused clock"
    );
    assert!(wall.elapsed() < Duration::from_secs(5), "the paused clock must have skipped the real waiting");
}

#[tokio::test]
async fn state_watch_follows_the_connection() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.reconnect.initial = Duration::from_millis(200);
    cfg.reconnect.max = Duration::from_millis(200);
    let (session, mut events) = connect(cfg).await;
    assert_eq!(session.state(), ConnState::Open);
    mock.kick();
    collect_until(&mut events, |e| matches!(e, Event::Reconnecting { .. })).await;
    assert_eq!(session.state(), ConnState::Reconnecting);
    collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    wait_until("open again", || session.state() == ConnState::Open).await;
}
