//! Smoke run of the client surface against the mock, and against a real gateway with `--ignored` (see the README).

mod common;

use common::*;
use zeroclaw_client::mock::{MockGateway, Turn};
use zeroclaw_client::{AbortOutcome, AuthState, ChatSession, CompleteRequest, Config, ConnectOptions, Event, RestClient};

async fn exercise(cfg: Config, complete: bool) {
    let rest = RestClient::new(&cfg).expect("client");
    let probe = rest.probe().await.expect("health");
    assert_eq!(probe.auth, AuthState::Accepted, "token must be accepted");
    assert!(probe.agents.iter().any(|a| a == cfg.agent_name()), "agent {} not among {:?}", cfg.agent_name(), probe.agents);

    let (session, mut events) = connect(cfg.clone()).await;
    session.send_text("Reply with the single word: pong").await.expect("send");
    let turn = until_done(&mut events).await;
    let Some(Event::Done { full_response, .. }) = turn.last() else { panic!("no done: {:?}", kinds(&turn)) };
    assert!(!full_response.trim().is_empty(), "empty reply");
    println!("reply: {full_response}");

    let id = session.session_id().to_string();
    session.close().await;
    collect_until(&mut events, |_| false).await;

    let opts = ConnectOptions { session_id: Some(id.clone()), ..ConnectOptions::default() };
    let (again, mut events) = ChatSession::connect(cfg.clone(), opts).await.expect("resume");
    let first = collect_until(&mut events, |e| matches!(e, Event::Connected { .. })).await;
    assert!(matches!(first.last(), Some(Event::Connected { resumed: true, session_id, .. }) if *session_id == id), "resume minted a new session");

    let mut pager = again.history(20, false).await.expect("history");
    assert!(!pager.next_page().is_empty(), "history is empty after a turn");
    assert_eq!(again.abort().await.expect("abort"), AbortOutcome::NothingRunning);

    if complete {
        let reply = rest.complete(&CompleteRequest::new("Reply with the single word: ok")).await.expect("complete");
        assert!(!reply.response.trim().is_empty());
    }
    again.close().await;
}

#[tokio::test]
async fn smoke_run_against_the_mock() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("pong"));
    exercise(mock.config(), true).await;
}

#[tokio::test]
#[ignore = "needs ZC_URL, ZC_TOKEN or ZC_TOKEN_FILE, ZC_AGENT"]
async fn live_gateway() {
    let url = std::env::var("ZC_URL").expect("ZC_URL");
    let token = match std::env::var("ZC_TOKEN") {
        Ok(t) => t,
        Err(_) => std::fs::read_to_string(std::env::var("ZC_TOKEN_FILE").expect("ZC_TOKEN or ZC_TOKEN_FILE")).expect("token file"),
    };
    let agent = std::env::var("ZC_AGENT").expect("ZC_AGENT");
    exercise(Config::new(url, token).with_agent(agent), std::env::var("ZC_SKIP_COMPLETE").is_err()).await;
}
