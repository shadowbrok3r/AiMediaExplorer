//! Runtime tests against the loopback mock gateway: a turn, tools and approvals, errors, resume, refusal, stop, switching and pairing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::runtime::Handle;
use zeroclaw_client::mock::{MockGateway, MockOptions, Turn};
use zeroclaw_client::{ApprovalDecision, CompleteRequest};

use super::events::{ChatEv, LinkState, Tagged};
use super::health::{Health, SWEEP};
use super::profile::ZcProfile;
use super::runtime::ZcRuntime;
use super::secrets::TokenStore;

/// A fresh, empty folder path for `name`.
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = temp_dir_path(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The folder path `temp_dir(name)` handed out, untouched.
pub(crate) fn temp_dir_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ame-zc-rt-{name}-{}", std::process::id()))
}

pub(crate) fn runtime_for(mock: &MockGateway, agent: &str, name: &str) -> Arc<ZcRuntime> {
    let rt = ZcRuntime::new(Handle::current(), Some(TokenStore::at(temp_dir(name))));
    rt.save_token(mock.token().unwrap_or("t")).unwrap();
    rt.set_profile(ZcProfile { host: mock.url(), agent: agent.into(), ..Default::default() });
    rt
}

pub(crate) async fn until<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    for _ in 0..400 {
        if let Some(v) = f() {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

/// Collects events until one satisfies `done`; returns all collected.
pub(crate) async fn collect_until(rt: &ZcRuntime, what: &str, done: impl Fn(&ChatEv) -> bool) -> Vec<Tagged> {
    let mut all = Vec::new();
    until(what, || {
        all.extend(rt.drain(256));
        all.iter().any(|t| done(&t.ev)).then_some(())
    })
    .await;
    all
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_turn_streams_and_finishes_with_usage() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("Hello from the mock gateway"));
    let rt = runtime_for(&mock, "default", "plain");
    until("online", || rt.health().is_online().then_some(())).await;
    let epoch = rt.connect("sess-plain", Some("Plain"));
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    rt.send("hi".into(), vec![]);
    let events = collect_until(&rt, "done", |e| matches!(e, ChatEv::Done { .. })).await;
    assert!(events.iter().all(|t| t.session == "sess-plain" && t.epoch == epoch));
    let text: String = events.iter().filter_map(|t| if let ChatEv::Text { text, .. } = &t.ev { Some(text.as_str()) } else { None }).collect();
    assert_eq!(text, "Hello from the mock gateway");
    match events.iter().find_map(|t| if let ChatEv::Done { full_response, usage, .. } = &t.ev { Some((full_response.clone(), usage.clone())) } else { None }) {
        Some((full, usage)) => {
            assert_eq!(full, "Hello from the mock gateway");
            assert_eq!(usage.input_tokens, Some(100));
        }
        None => panic!("no done event"),
    }
    assert_eq!(rt.usage().turns, 1);
    assert!(rt.with_frames(|f| f.total_in) > 3 && rt.with_frames(|f| f.total_out) >= 2);
    assert_eq!(rt.link().state, LinkState::Open);
    rt.disconnect();
    assert_eq!(rt.link().state, LinkState::Idle);
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_a_tool_call_and_an_approval_flow_through_in_order() {
    let mock = MockGateway::start().await;
    mock.script_turn(
        Turn::new()
            .thinking("let me look")
            .tool_call("c1", "aimedia__library_stats", serde_json::json!({}))
            .approval("r1", "aimedia__library_stats", "count the library")
            .tool_result("c1", "aimedia__library_stats", "12 items")
            .chunk("There are 12 items.")
            .done("There are 12 items."),
    );
    let rt = runtime_for(&mock, "default", "tools");
    rt.connect("sess-tools", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    rt.send("stats?".into(), vec![]);
    let mut seen = Vec::new();
    let events = {
        let mut all = Vec::new();
        until("approval", || {
            all.extend(rt.drain(256));
            all.iter().any(|t| matches!(t.ev, ChatEv::Approval(_))).then_some(())
        })
        .await;
        let request = all.iter().find_map(|t| if let ChatEv::Approval(r) = &t.ev { Some(r.clone()) } else { None }).unwrap();
        assert_eq!((request.request_id.as_str(), request.tool.as_str()), ("r1", "aimedia__library_stats"));
        assert!(rt.approve("r1", ApprovalDecision::Approve));
        all.extend(collect_until(&rt, "done", |e| matches!(e, ChatEv::Done { .. })).await);
        all
    };
    for t in &events {
        seen.push(match &t.ev {
            ChatEv::Thinking { .. } => "thinking",
            ChatEv::ToolCall { .. } => "call",
            ChatEv::Approval(_) => "approval",
            ChatEv::ToolResult { .. } => "result",
            ChatEv::Text { .. } => "text",
            ChatEv::Done { .. } => "done",
            _ => continue,
        });
    }
    assert_eq!(seen, vec!["thinking", "call", "approval", "result", "text", "done"]);
    let response = mock.received_of_type("approval_response");
    assert_eq!(response[0]["decision"], "approve");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_error_frame_ends_the_turn_and_the_session_stays_usable() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("partial ").error("PROVIDER_ERROR", "model unavailable"));
    mock.script_turn(Turn::reply("second turn works"));
    let rt = runtime_for(&mock, "default", "error");
    rt.connect("sess-err", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    rt.send("one".into(), vec![]);
    let events = collect_until(&rt, "error", |e| matches!(e, ChatEv::Error { .. })).await;
    assert!(events.iter().any(|t| matches!(&t.ev, ChatEv::Error { code: Some(c), ends_turn: true, .. } if c == "PROVIDER_ERROR")));
    rt.send("two".into(), vec![]);
    let events = collect_until(&rt, "done", |e| matches!(e, ChatEv::Done { .. })).await;
    assert!(events.iter().any(|t| matches!(&t.ev, ChatEv::Done { turn: 1, .. })), "the failed turn used up number 0");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_mid_turn_drop_reconnects_and_the_turn_is_resumed() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("Half ").delay_ms(30).drop_connection());
    let rt = runtime_for(&mock, "default", "resume");
    rt.connect("sess-resume", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    mock.script_connection(zeroclaw_client::mock::ConnScript::new().resume(
        vec![serde_json::json!({"type":"chunk","content":"Half "})],
        Turn::new().chunk("done").done("Half done"),
    ));
    rt.send("go".into(), vec![]);
    let events = collect_until(&rt, "done", |e| matches!(e, ChatEv::Done { .. })).await;
    let resumed = events.iter().position(|t| matches!(t.ev, ChatEv::TurnResumed { .. })).expect("a resume marker");
    let reconnected = events.iter().position(|t| matches!(t.ev, ChatEv::Connected { reconnect: true, .. })).expect("a reconnect");
    assert!(reconnected < resumed);
    assert!(events.iter().any(|t| matches!(&t.ev, ChatEv::Done { full_response, .. } if full_response == "Half done")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_token_marks_health_and_is_not_retried() {
    let _serial = super::health::serial();
    let mock = MockGateway::start().await;
    let rt = ZcRuntime::new(Handle::current(), Some(TokenStore::at(temp_dir("refused"))));
    rt.set_publish(true);
    rt.save_token("wrong-token").unwrap();
    rt.set_profile(ZcProfile { host: mock.url(), agent: "default".into(), ..Default::default() });
    until("refusal", || (rt.health() == Health::Unauthorized).then_some(())).await;
    rt.connect("sess-401", None);
    let mut events = Vec::new();
    until("the refusal and the failed link", || {
        events.extend(rt.drain(256));
        let refused = events.iter().any(|t| matches!(t.ev, ChatEv::AuthRefused));
        let failed = events.iter().any(|t| matches!(&t.ev, ChatEv::Link { state: LinkState::Failed, .. }));
        (refused && failed).then_some(())
    })
    .await;
    let ws_attempts = mock.connections().len();
    let probes = mock.requests().iter().filter(|r| r.path == "/api/config/agent-options").count();
    rt.tick_at(Instant::now() + SWEEP * 2);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(mock.connections().len(), ws_attempts, "no socket retry after a 401");
    assert_eq!(mock.requests().iter().filter(|r| r.path == "/api/config/agent-options").count(), probes, "the sweep leaves a refused token alone");
    assert_eq!(rt.health(), Health::Unauthorized);
    assert!(crate::tasks::with(|r| r.attention()).iter().any(|i| i.key == "zeroclaw-token"));
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_aborts_the_turn_and_ignores_late_frames() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("one ").chunk("two ").delay_ms(200).chunk("late ").done("late"));
    let rt = runtime_for(&mock, "default", "stop");
    mock.set_state("sess-stop", "running", None);
    rt.connect("sess-stop", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    rt.send("long".into(), vec![]);
    collect_until(&rt, "first text", |e| matches!(e, ChatEv::Text { .. })).await;
    assert!(rt.stop());
    tokio::time::sleep(Duration::from_millis(400)).await;
    let rest = rt.drain(256);
    assert!(!rest.iter().any(|t| matches!(&t.ev, ChatEv::Text { text, .. } if text.contains("late"))), "frames after Stop are suppressed");
    assert!(mock.requests().iter().any(|r| r.path.ends_with("/abort")), "the abort endpoint was called");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_switch_makes_older_events_stale() {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", "switch");
    let first = rt.connect("sess-a", None);
    let second = rt.connect("sess-b", None);
    assert!(second > first);
    assert_eq!(rt.live_session().as_deref(), Some("sess-b"));
    let events = collect_until(&rt, "connected b", |e| matches!(e, ChatEv::Connected { .. })).await;
    assert!(events.iter().filter(|t| matches!(t.ev, ChatEv::Connected { .. })).all(|t| t.session == "sess-b" && t.epoch == second));
    assert!(!events.iter().any(|t| t.session == "sess-a" && matches!(t.ev, ChatEv::Connected { .. })), "the superseded connection reports nothing");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_send_without_a_connection_comes_back_as_a_failed_send() {
    let rt = ZcRuntime::new(Handle::current(), None);
    rt.send("draft text".into(), vec![]);
    let events = rt.drain(8);
    assert!(matches!(&events[0].ev, ChatEv::SendFailed { text, .. } if text == "draft text"));
}

#[tokio::test(flavor = "multi_thread")]
async fn complete_goes_through_the_gateway_and_errors_hide_the_token() {
    let mock = MockGateway::start().await;
    mock.set_complete_reply("A Title");
    let rt = runtime_for(&mock, "default", "complete");
    let mut job = rt.complete(CompleteRequest::new("name this chat"));
    let reply = until("complete", || job.take()).await.unwrap();
    assert_eq!(reply.response, "A Title");
    let rt2 = ZcRuntime::new(Handle::current(), None);
    let mut failing = rt2.complete(CompleteRequest::new("x"));
    let err = until("complete error", || failing.take()).await.unwrap_err();
    assert!(err.contains("not set") || err.contains("token"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn pairing_stores_the_token_and_brings_the_gateway_online() {
    let mock = MockGateway::start().await;
    let rt = ZcRuntime::new(Handle::current(), Some(TokenStore::at(temp_dir("pair"))));
    assert!(!rt.has_token());
    let mut bad = rt.pair(&mock.url(), "000000");
    assert!(until("bad pair", || bad.take()).await.is_err());
    assert!(!rt.has_token());
    let mut good = rt.pair(&mock.url(), "123456");
    until("pair", || good.take()).await.unwrap();
    assert!(rt.has_token());
    let stored_in = temp_dir_path("pair");
    rt.set_profile(ZcProfile { host: mock.url(), agent: "default".into(), ..Default::default() });
    until("online", || rt.health().is_online().then_some(())).await;
    assert!(TokenStore::at(stored_in).load().is_some(), "the token is in the token file");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_address_closes_the_socket_and_rechecks() {
    let mock = MockGateway::start_with(MockOptions { agents: vec!["a".into(), "b".into()], ..Default::default() }).await;
    let rt = runtime_for(&mock, "a", "moved");
    until("online", || rt.health().is_online().then_some(())).await;
    rt.connect("sess-m", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    let mut p = rt.profile();
    p.agent = "ghost".into();
    assert!(rt.set_profile(p.clone()));
    assert!(!rt.set_profile(p), "an unchanged profile changes nothing");
    assert!(rt.live_session().is_none());
    until("unknown agent", || matches!(rt.health(), Health::UnknownAgent { .. }).then_some(())).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_token_never_appears_in_notes_frames_or_debug_output() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("ok"));
    let rt = runtime_for(&mock, "default", "leak");
    let token = mock.token().unwrap().to_string();
    rt.connect("sess-leak", None);
    collect_until(&rt, "connected", |e| matches!(e, ChatEv::Connected { .. })).await;
    rt.send("hello".into(), vec![]);
    collect_until(&rt, "done", |e| matches!(e, ChatEv::Done { .. })).await;
    rt.note(format!("the token {token} must be hidden"));
    let notes = rt.notes().iter().map(|n| n.1.clone()).collect::<Vec<_>>().join("\n");
    assert!(!notes.contains(&token), "{notes}");
    let frames = rt.with_frames(|f| f.frames().map(|fr| fr.frame.to_string()).collect::<Vec<_>>().join("\n"));
    assert!(!frames.contains(&token));
    assert!(!format!("{:?}", rt.profile()).contains(&token));
    assert!(!format!("{:?}", rt.link()).contains(&token));
    assert!(!format!("{:?}", rt.health()).contains(&token));
}

/// Serves a scripted mock gateway on loopback until `<dir>/stop` appears (`ZC_MOCK_DIR=<dir>`).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "serves a loopback mock gateway for manual runs"]
async fn manual_mock_gateway() {
    let dir = PathBuf::from(std::env::var("ZC_MOCK_DIR").expect("set ZC_MOCK_DIR to a scratch folder"));
    std::fs::create_dir_all(&dir).unwrap();
    let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into(), "default".into()], ..Default::default() }).await;
    std::fs::write(dir.join("token"), mock.token().unwrap()).unwrap();
    std::fs::write(dir.join("url"), mock.url()).unwrap();
    mock.script_turn(
        Turn::new()
            .thinking("The person wants to know how big the library is. I should look at its statistics first.")
            .tool_call("c1", "aimedia__library_stats", serde_json::json!({"top": 5}))
            .tool_result("c1", "aimedia__library_stats", "{\"items\": 12408, \"tagged\": 9120, \"folders\": 37}")
            .tool_call("c2", "aimedia__describe_image", serde_json::json!({"path": "/home/user/Pictures/sunset.jpg", "dry_run": true}))
            .approval("r1", "aimedia__describe_image", "Describe /home/user/Pictures/sunset.jpg with the local vision model")
            .tool_result("c2", "aimedia__describe_image", "A sunset over the sea with orange clouds.")
            .chunk("Your library holds **12,408** items in 37 folders, and 9,120 of them are tagged.\n\n")
            .chunk("```text\nitems   12408\ntagged   9120\nfolders    37\n```\n")
            .chunk("The sunset picture shows orange clouds over the sea.")
            .done("Your library holds 12,408 items in 37 folders, and 9,120 of them are tagged. The sunset picture shows orange clouds over the sea."),
    );
    mock.script_turn(Turn::reply("Anything else you would like to know about the library?"));
    mock.script_turn(Turn::reply("Here is another answer from the mock gateway."));
    for _ in 0..20_000 {
        if dir.join("stop").exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
