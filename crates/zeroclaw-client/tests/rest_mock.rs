mod common;

use std::time::Duration;

use serde_json::json;
use zeroclaw_client::mock::{MockGateway, MockOptions};
use zeroclaw_client::types::{COMPLETE_INPUT_CAP, PairResult};
use zeroclaw_client::{
    AbortOutcome, AuthState, CompleteRequest, Config, Error, HistoryKind, InlineImage, MemoryQuery, MemoryStore, RestClient,
    group_sessions,
};

fn client(mock: &MockGateway) -> RestClient {
    RestClient::new(&mock.config()).unwrap()
}

#[tokio::test]
async fn health_needs_no_token_and_sends_none() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.set_token("");
    let health = RestClient::new(&cfg).unwrap().health().await.unwrap();
    assert_eq!(health.status, "ok");
    assert!(health.require_pairing && health.paired);

    client(&mock).health().await.unwrap();
    assert!(mock.requests().iter().all(|r| r.path == "/health" && !r.authorized), "health must not carry the bearer token");
}

#[tokio::test]
async fn probe_reports_reachability_auth_and_agents() {
    let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into(), "coder".into()], ..MockOptions::default() }).await;
    let probe = client(&mock).probe().await.unwrap();
    assert_eq!(probe.auth, AuthState::Accepted);
    assert_eq!(probe.agents, ["media", "coder"]);

    let mut bad = mock.config();
    bad.set_token("nope");
    let probe = RestClient::new(&bad).unwrap().probe().await.unwrap();
    assert_eq!(probe.auth, AuthState::Rejected(401));
    assert!(probe.agents.is_empty());
    assert_eq!(probe.health.status, "ok");

    let mut none = mock.config();
    none.set_token("");
    assert_eq!(RestClient::new(&none).unwrap().probe().await.unwrap().auth, AuthState::NotChecked);
}

#[tokio::test]
async fn unreachable_gateway_is_a_connect_error() {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let cfg = Config::new(format!("127.0.0.1:{port}"), "t");
    let err = RestClient::new(&cfg).unwrap().health().await.unwrap_err();
    assert!(matches!(err, Error::Connect(_)), "{err:?}");
    assert!(err.is_retryable());
    assert_eq!(err.exit_class().code(), 4);
}

#[tokio::test]
async fn wrong_token_maps_to_unauthorized_and_every_call_is_authenticated() {
    let mock = MockGateway::start().await;
    let mut bad = mock.config();
    bad.set_token("wrong");
    let err = RestClient::new(&bad).unwrap().sessions().await.unwrap_err();
    assert!(matches!(err, Error::Unauthorized), "{err:?}");
    assert_eq!(err.exit_class().code(), 3);

    let c = client(&mock);
    c.sessions().await.unwrap();
    c.agents().await.unwrap();
    assert!(mock.requests().iter().skip(1).all(|r| r.authorized));
}

#[tokio::test]
async fn sessions_group_by_agent_and_skip_empty_shells() {
    let mock = MockGateway::start().await;
    mock.add_session("old", "media", 2, "2026-09-09T12:00:00+00:00");
    mock.add_session("empty", "media", 0, "2026-09-11T11:59:00+00:00");
    mock.add_session("other", "voice", 9, "2026-09-11T11:00:00+00:00");
    mock.add_session("new", "media", 4, "2026-09-11T10:00:00+00:00");
    let c = client(&mock);
    let all = c.sessions().await.unwrap();
    assert_eq!(all.len(), 4);
    let rows = group_sessions(Some("media"), &all);
    assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["new", "old"]);
    assert_eq!(rows[0].keys, ["gw_new"]);
}

#[tokio::test]
async fn running_and_state_and_abort() {
    let mock = MockGateway::start().await;
    mock.add_session("s1", "media", 2, "2026-09-11T10:00:00+00:00");
    mock.set_state("s1", "running", Some("2026-09-13T02:58:13+00:00"));
    let c = client(&mock);
    assert_eq!(c.running().await.unwrap(), ["s1"]);
    let state = c.session_state("s1").await.unwrap();
    assert!(state.running);
    assert_eq!(state.started, Some(1_789_268_293));
    assert_eq!(c.abort("s1").await.unwrap(), AbortOutcome::Aborted);
    assert!(!c.session_state("s1").await.unwrap().running);
    assert_eq!(c.abort("s1").await.unwrap(), AbortOutcome::NothingRunning);
    assert!(c.running().await.unwrap().is_empty());
    let err = c.session_state("missing").await.unwrap_err();
    assert!(matches!(err, Error::Http { status: 404, .. }), "{err:?}");
}

#[tokio::test]
async fn rename_and_delete() {
    let mock = MockGateway::start().await;
    mock.add_session("s1", "media", 2, "2026-09-11T10:00:00+00:00");
    let c = client(&mock);
    c.rename_session("s1", "Sunset shoot").await.unwrap();
    assert_eq!(c.sessions().await.unwrap()[0].name.as_deref(), Some("Sunset shoot"));
    assert!(matches!(c.rename_session("s1", "  ").await, Err(Error::Http { status: 400, .. })));
    assert!(matches!(c.rename_session("gone", "x").await, Err(Error::Http { status: 404, .. })));
    assert!(c.delete_session("s1").await.unwrap());
    assert!(!c.delete_session("s1").await.unwrap(), "an already deleted session is not an error");
    assert!(c.sessions().await.unwrap().is_empty());
}

#[tokio::test]
async fn session_ids_are_percent_encoded_in_paths() {
    let mock = MockGateway::start().await;
    let c = client(&mock);
    let _ = c.session_state("discord.a b/c").await;
    assert_eq!(mock.requests().last().unwrap().path, "/api/sessions/discord.a%20b%2Fc/state");
}

#[tokio::test]
async fn history_pages_lazily_from_the_newest_end() {
    let mock = MockGateway::start().await;
    let rows: Vec<_> = (0..25)
        .map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("m{i}"), "created_at": format!("2026-09-11T10:{i:02}:00+00:00")}))
        .collect();
    mock.set_messages("gw_s1", rows);
    let c = client(&mock);
    let mut pager = c.history("s1", 10, false).await.unwrap();
    assert_eq!(pager.total_messages(), 25);
    let newest = pager.next_page();
    assert_eq!(newest.len(), 10);
    assert_eq!(newest[0].text, "m15");
    assert_eq!(newest[9].text, "m24");
    assert_eq!(newest[9].kind, HistoryKind::User);
    let older = pager.next_page();
    assert_eq!(older[0].text, "m5");
    assert_eq!(pager.rest().len(), 5);
    assert!(!pager.has_more());
    assert_eq!(mock.requests().iter().filter(|r| r.path.ends_with("/messages")).count(), 1, "one fetch serves every page");
}

#[tokio::test]
async fn tool_context_flag_reaches_the_query_and_envelopes_unpack() {
    let mock = MockGateway::start().await;
    mock.set_messages(
        "gw_s1",
        vec![
            json!({"role":"user","content":"[CURRENT DATE & TIME: 2026-09-11 21:51:33 +00:00]\n\nsearch"}),
            json!({"role":"assistant","content":"{\"content\":\"done\",\"reasoning_content\":\"hm\",\"tool_calls\":[{\"id\":\"c\",\"name\":\"t\",\"arguments\":\"{}\"}]}"}),
        ],
    );
    let c = client(&mock);
    let mut pager = c.history("s1", 50, true).await.unwrap();
    assert!(mock.requests().last().unwrap().query.contains("tool_context=true"));
    let items = pager.next_page();
    let kinds: Vec<_> = items.iter().map(|i| i.kind).collect();
    assert_eq!(kinds, [HistoryKind::User, HistoryKind::Reasoning, HistoryKind::ToolCall, HistoryKind::Assistant]);
    assert_eq!(items[0].text, "search");
}

#[tokio::test]
async fn merged_messages_interleave_twin_stores_and_tolerate_one_failure() {
    let mock = MockGateway::start().await;
    mock.set_messages("gw_x", vec![json!({"role":"user","content":"b","created_at":"2026-09-12T01:00:00+00:00"})]);
    mock.set_messages("x", vec![json!({"role":"user","content":"a","created_at":"2026-09-11T01:00:00+00:00"})]);
    let c = client(&mock);
    let merged = c.messages_merged(&["gw_x".into(), "x".into()], false).await.unwrap();
    assert_eq!(merged.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(), ["a", "b"]);

    mock.override_rest("GET", "/api/sessions/x/messages", 500, json!({"error":"boom"}), Duration::ZERO);
    let partial = c.messages_merged(&["gw_x".into(), "x".into()], false).await.unwrap();
    assert_eq!(partial.len(), 1);
    mock.override_rest("GET", "/api/sessions/gw_x/messages", 500, json!({"error":"boom"}), Duration::ZERO);
    assert!(c.messages_merged(&["gw_x".into(), "x".into()], false).await.is_err());
}

#[tokio::test]
async fn media_returns_bytes_and_missing_files_are_404() {
    let mock = MockGateway::start().await;
    mock.set_media("s1", "/out/a b.png", vec![1, 2, 3, 4]);
    let c = client(&mock);
    assert_eq!(c.media("s1", "/out/a b.png").await.unwrap(), [1, 2, 3, 4]);
    let err = c.media("s1", "/missing.png").await.unwrap_err();
    assert!(matches!(&err, Error::Http { status: 404, message, .. } if message == "image_file_unavailable"), "{err:?}");
    assert!(mock.requests()[0].query.contains("path=%2Fout%2Fa%20b.png") || mock.requests()[0].query.contains("path=%2Fout%2Fa+b.png"));
}

#[tokio::test]
async fn complete_round_trip_and_server_side_errors() {
    let mock = MockGateway::start().await;
    mock.set_complete_reply("tidy caption");
    let c = client(&mock);
    let reply = c.complete(&CompleteRequest::new("caption this").with_system("be brief").with_temperature(0.2)).await.unwrap();
    assert_eq!(reply.response, "tidy caption");
    assert_eq!(reply.model, "mock-model");
    assert_eq!(reply.hint, "quick");
    let sent = mock.requests().last().unwrap().body.clone();
    assert_eq!(sent, json!({"hint":"quick","system":"be brief","prompt":"caption this","temperature":0.2}));

    let err = c.complete(&CompleteRequest::new("x").with_hint("slow")).await.unwrap_err();
    assert!(matches!(&err, Error::Http { status: 400, message, .. } if message.contains("no [[model_routes]] entry has hint `slow`")), "{err:?}");
}

#[tokio::test]
async fn complete_enforces_the_input_cap_before_touching_the_network() {
    let mock = MockGateway::start().await;
    let c = client(&mock);
    let over = CompleteRequest::new("x".repeat(COMPLETE_INPUT_CAP + 1));
    assert!(matches!(c.complete(&over).await, Err(Error::TooLarge { .. })));
    assert!(matches!(c.complete(&CompleteRequest::new("  ")).await, Err(Error::EmptyMessage)));
    assert!(c.complete(&CompleteRequest::new("x").with_temperature(9.0)).await.is_err());
    assert!(mock.requests().is_empty(), "invalid requests must not be sent");
    assert!(c.complete(&CompleteRequest::new("x".repeat(COMPLETE_INPUT_CAP))).await.is_ok(), "exactly the cap is accepted");
}

#[tokio::test]
async fn complete_carries_an_image_marker_inside_the_prompt() {
    let mock = MockGateway::start().await;
    let c = client(&mock);
    let image = InlineImage::from_bytes("a.png", b"\x89PNG\r\n\x1a\nrest").unwrap();
    let req = CompleteRequest::new("what colour").with_image(&image).unwrap();
    c.complete(&req).await.unwrap();
    let prompt = mock.requests().last().unwrap().body["prompt"].as_str().unwrap().to_string();
    assert!(prompt.starts_with("what colour [IMAGE:data:image/png;base64,"));
}

#[tokio::test]
async fn memory_store_list_recall_delete() {
    let mock = MockGateway::start().await;
    let c = client(&mock);
    c.memory_store(&MemoryStore { category: Some("core".into()), agent: Some("media".into()), ..MemoryStore::new("style", "Tags are lowercase singular nouns") })
        .await
        .unwrap();
    c.memory_store(&MemoryStore::new("note", "Prefers warm tones")).await.unwrap();
    let all = c.memory(&MemoryQuery::default()).await.unwrap();
    assert_eq!(all.len(), 2);
    let media = c.memory(&MemoryQuery { agent: Some("media".into()), ..MemoryQuery::default() }).await.unwrap();
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].key, "style");
    assert_eq!(media[0].category, "core");
    let recalled = c.memory(&MemoryQuery { query: Some("WARM".into()), ..MemoryQuery::default() }).await.unwrap();
    assert_eq!(recalled.iter().map(|e| e.key.as_str()).collect::<Vec<_>>(), ["note"]);
    assert!(c.memory_delete("note", None).await.unwrap());
    assert!(!c.memory_delete("note", None).await.unwrap());
    assert_eq!(mock.memory().len(), 1);
}

#[tokio::test]
async fn pairing_exchanges_a_code_for_a_token_without_credentials() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.set_token("");
    let paired: PairResult = RestClient::pair(&cfg, "123456").await.unwrap();
    assert!(paired.paired);
    assert_eq!(paired.token, "test-token");
    assert!(!format!("{paired:?}").contains("test-token"));
    let err = RestClient::pair(&cfg, "000000").await.unwrap_err();
    assert!(matches!(err, Error::Forbidden), "{err:?}");
    assert!(mock.requests().iter().all(|r| r.path == "/pair"));
}

#[tokio::test]
async fn config_get_is_read_only_json() {
    let mock = MockGateway::start().await;
    let v = client(&mock).config_get().await.unwrap();
    assert_eq!(v["gateway"]["paired_tokens"], "***");
}

#[tokio::test]
async fn slow_responses_time_out_and_oversized_ones_are_refused() {
    let mock = MockGateway::start().await;
    let mut cfg = mock.config();
    cfg.request_timeout = Duration::from_millis(150);
    cfg.max_rest_body_bytes = 1000;
    let c = RestClient::new(&cfg).unwrap();
    mock.override_rest("GET", "/api/sessions", 200, json!({"sessions":[]}), Duration::from_millis(800));
    assert!(matches!(c.sessions().await, Err(Error::Timeout(_))));
    mock.override_rest("GET", "/api/sessions/running", 200, json!({"blob":"x".repeat(5000)}), Duration::ZERO);
    let err = c.running().await.unwrap_err();
    assert!(matches!(err, Error::ResponseTooLarge { limit: 1000, .. }), "{err:?}");
}

#[tokio::test]
async fn server_errors_surface_the_message_and_never_the_token() {
    let mock = MockGateway::start().await;
    let c = client(&mock);
    mock.override_rest("GET", "/api/sessions", 502, json!({"error":"upstream said test-token is bad"}), Duration::ZERO);
    let err = c.sessions().await.unwrap_err();
    let text = err.to_string();
    assert!(text.contains("502") && text.contains("upstream said"), "{text}");
    assert!(!text.contains("test-token"), "{text}");
    assert!(err.is_retryable());

    mock.override_rest("GET", "/api/memory", 200, json!("not an object"), Duration::ZERO);
    assert!(c.memory(&MemoryQuery::default()).await.unwrap().is_empty());
    mock.override_rest("GET", "/api/sessions/running", 200, json!({}), Duration::ZERO);
    assert!(c.running().await.unwrap().is_empty());
}

#[tokio::test]
async fn redirects_are_not_followed_so_credentials_stay_on_the_origin() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let elsewhere = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let elsewhere_addr = elsewhere.local_addr().unwrap();
    let origin = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin_addr = origin.local_addr().unwrap();
    let (hit_tx, mut hit_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = elsewhere.accept().await {
            let mut buf = vec![0u8; 4096];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let _ = hit_tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
            let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await;
        }
    });
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = origin.accept().await {
            let mut buf = vec![0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let reply = format!(
                "HTTP/1.1 302 Found\r\nLocation: http://{elsewhere_addr}/api/sessions\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let _ = socket.write_all(reply.as_bytes()).await;
        }
    });
    let cfg = Config::new(origin_addr.to_string(), "secret-token");
    let err = RestClient::new(&cfg).unwrap().sessions().await.unwrap_err();
    assert!(matches!(err, Error::Http { status: 302, .. }), "{err:?}");
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(hit_rx.try_recv().is_err(), "the redirect target must never be contacted");
}

#[tokio::test]
async fn scripted_complete_answers_are_served_in_order_then_the_echo_returns() {
    let mock = MockGateway::start().await;
    mock.script_complete_reply("first");
    mock.script_complete(503, json!({"error": "busy"}), Duration::from_millis(30));
    let rest = client(&mock);
    assert_eq!(rest.complete(&CompleteRequest::new("a")).await.unwrap().response, "first");
    let err = rest.complete(&CompleteRequest::new("b")).await.unwrap_err();
    assert!(matches!(err, Error::Http { status: 503, .. }) && err.is_retryable(), "{err:?}");
    assert_eq!(rest.complete(&CompleteRequest::new("c")).await.unwrap().response, "echo: c");
}
