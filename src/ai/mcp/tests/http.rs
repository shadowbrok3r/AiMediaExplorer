use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde_json::json;

use crate::ai::mcp::activity::{ConnectionLog, EventKind};
use crate::ai::mcp::config::McpConfig;
use crate::ai::mcp::server::{Hub, MCP_STATUS, ServerState};
use crate::ai::mcp::test_support::*;
use crate::status::{GlobalStatusIndicator, StatusState};

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(super) struct Served {
    pub(super) hub: Arc<Hub>,
    pub(super) f: Fixture,
    pub(super) addr: SocketAddr,
    pub(super) token: String,
    _dir: tempfile::TempDir,
    _lock: tokio::sync::MutexGuard<'static, ()>,
}

impl Served {
    pub(super) fn url(&self) -> String {
        format!("http://{}/mcp", self.addr)
    }

    pub(super) fn client(&self) -> reqwest::Client {
        reqwest::Client::builder().timeout(Duration::from_secs(10)).build().unwrap()
    }

    pub(super) async fn post(&self, token: Option<&str>, body: &str) -> reqwest::Response {
        let mut req = self.client().post(self.url()).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(body.to_string());
        if let Some(t) = token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        req.send().await.unwrap()
    }

    pub(super) fn rmcp_transport(&self, token: &str) -> StreamableHttpClientTransport<reqwest::Client> {
        StreamableHttpClientTransport::from_config(StreamableHttpClientTransportConfig::with_uri(self.url()).auth_header(token.to_string()))
    }
}

pub(super) const INIT: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;

pub(super) async fn served_with(tweak: impl FnOnce(&mut McpConfig)) -> Served {
    let lock = SERIAL.lock().await;
    let f = fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = McpConfig { enabled: true, port: 0, ..Default::default() };
    tweak(&mut cfg);
    let hub = Hub::new(f.ctx.clone(), Some(dir.path().to_path_buf()), cfg, f.activity.clone(), f.stats.clone(), Arc::new(ConnectionLog::default()), f.ctx.clock.clone());
    let addr = hub.start().await.unwrap();
    let token = hub.token().unwrap();
    Served { hub, f, addr, token, _dir: dir, _lock: lock }
}

pub(super) async fn served() -> Served {
    served_with(|_| {}).await
}

#[tokio::test]
async fn requests_need_the_right_bearer_token() {
    let s = served().await;
    let none = s.post(None, INIT).await;
    assert_eq!(none.status(), 401);
    assert_eq!(none.headers().get("www-authenticate").unwrap(), "Bearer");
    assert_eq!(s.post(Some("wrong"), INIT).await.status(), 401);
    assert_eq!(s.post(Some(&format!("{}x", &s.token[..42])), INIT).await.status(), 401);
    let basic = s.client().post(s.url()).header("Authorization", format!("Basic {}", s.token)).body(INIT).send().await.unwrap();
    assert_eq!(basic.status(), 401, "the scheme must be Bearer");
    let ok = s.post(Some(&s.token), INIT).await;
    assert_eq!(ok.status(), 200);
    assert!(ok.headers().get("mcp-session-id").is_some());
    let stats = s.hub.stats.snapshot();
    assert_eq!((stats.rejected_auth, stats.requests), (4, 5));
    let events = s.hub.conns.events(20);
    assert_eq!(events.iter().filter(|e| e.kind == EventKind::Rejected).count(), 4);
    s.hub.stop().await;
}

#[tokio::test]
async fn the_server_listens_on_loopback_only_by_default() {
    let s = served().await;
    assert!(s.addr.ip().is_loopback());
    assert_eq!(s.hub.url(), format!("http://127.0.0.1:{}/mcp", s.addr.port()));
    s.hub.stop().await;
}

#[tokio::test]
async fn other_host_names_are_turned_away_even_with_the_token() {
    let s = served().await;
    for host in ["evil.example.com", "evil.example.com:9871", "127.0.0.1.evil.com", "192.168.1.9"] {
        let res = s.client().post(s.url()).header("Host", host).header("Authorization", format!("Bearer {}", s.token)).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(INIT).send().await.unwrap();
        assert_eq!(res.status(), 403, "{host}");
    }
    assert_eq!(s.hub.stats.snapshot().rejected_host, 4);
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200, "the real host still works");
    s.hub.stop().await;
}

#[tokio::test]
async fn browser_origins_are_turned_away() {
    let s = served().await;
    let res = s.client().post(s.url()).header("Origin", "https://evil.example.com").header("Authorization", format!("Bearer {}", s.token)).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(INIT).send().await.unwrap();
    assert_eq!(res.status(), 403);
    let null_origin = s.client().post(s.url()).header("Origin", "null").header("Authorization", format!("Bearer {}", s.token)).body(INIT).send().await.unwrap();
    assert_eq!(null_origin.status(), 403);
    s.hub.stop().await;
}

#[tokio::test]
async fn a_listed_origin_is_accepted() {
    let s = served_with(|c| c.allowed_origins = vec!["https://app.example.com".into()]).await;
    let res = s.client().post(s.url()).header("Origin", "https://app.example.com").header("Authorization", format!("Bearer {}", s.token)).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(INIT).send().await.unwrap();
    assert_eq!(res.status(), 200);
    s.hub.stop().await;
}

#[tokio::test]
async fn oversized_bodies_are_refused() {
    let s = served_with(|c| c.max_body_bytes = 2000).await;
    let big = format!(r#"{{"jsonrpc":"2.0","id":1,"method":"ping","params":{{"pad":"{}"}}}}"#, "x".repeat(5000));
    assert_eq!(s.post(Some(&s.token), &big).await.status(), 413);
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200);
    s.hub.stop().await;
}

#[tokio::test]
async fn too_many_requests_get_429_until_the_bucket_refills() {
    let s = served_with(|c| {
        c.rate_burst = 3;
        c.rate_per_minute = 60;
    })
    .await;
    for _ in 0..3 {
        assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200);
    }
    let limited = s.post(Some(&s.token), INIT).await;
    assert_eq!(limited.status(), 429);
    assert!(limited.headers().get("retry-after").unwrap().to_str().unwrap().parse::<u64>().unwrap() >= 1);
    assert_eq!(s.hub.stats.snapshot().rate_limited, 1);
    s.f.clock.advance(Duration::from_secs(2));
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200);
    s.hub.stop().await;
}

#[tokio::test]
async fn repeated_wrong_tokens_are_throttled() {
    let s = served().await;
    let mut codes = Vec::new();
    for _ in 0..40 {
        codes.push(s.post(Some("guess"), INIT).await.status().as_u16());
    }
    assert!(codes[..30].iter().all(|c| *c == 401));
    assert!(codes[30..].iter().all(|c| *c == 429), "{codes:?}");
    s.hub.stop().await;
}

#[tokio::test]
async fn an_mcp_client_can_list_and_call_tools() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.expect("the client connects");
    let info = client.peer_info().expect("server info");
    assert_eq!(info.server_info.as_ref().map(|i| i.name.as_str()), Some("smart-media"));
    let instructions = info.instructions.clone().unwrap_or_default();
    assert!(instructions.contains("library_search") && instructions.contains("metadata_propose"), "{instructions}");
    assert!(info.capabilities.tools.is_some());

    let tools = client.list_all_tools().await.unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(names.len() >= 19, "{names:?}");
    assert!(names.contains(&"library_search") && names.contains(&"describe_image") && names.contains(&"ui_open_scope"));
    assert!(names.iter().all(|n| !n.contains('.')));
    let search = tools.iter().find(|t| t.name == "library_search").unwrap();
    assert_eq!(search.input_schema.get("type"), Some(&json!("object")));
    assert!(search.input_schema.get("properties").unwrap().get("query").is_some());

    let ping = client.call_tool(CallToolRequestParams::new("util_ping").with_arguments(json!({"message": "hello"}).as_object().unwrap().clone())).await.unwrap();
    assert_ne!(ping.is_error, Some(true));
    assert_eq!(serde_json::to_value(&ping.content).unwrap()[0]["text"], "pong: hello");
    assert_eq!(ping.structured_content.as_ref().unwrap()["message"], "hello");

    let stats = client.call_tool(CallToolRequestParams::new("library_stats")).await.unwrap();
    assert_eq!(stats.structured_content.as_ref().unwrap()["items"], 5);

    let found = client.call_tool(CallToolRequestParams::new("library_search").with_arguments(json!({"query": "forest"}).as_object().unwrap().clone())).await.unwrap();
    assert_eq!(found.structured_content.as_ref().unwrap()["total"], 1);

    let preview = client.call_tool(CallToolRequestParams::new("library_get_preview").with_arguments(json!({"path": s.f.path("photos/beach1.png")}).as_object().unwrap().clone())).await.unwrap();
    let blocks = serde_json::to_value(&preview.content).unwrap();
    assert!(blocks.as_array().unwrap().iter().any(|b| b["type"] == "image" && b["mimeType"] == "image/jpeg"));

    let denied = client.call_tool(CallToolRequestParams::new("describe_image").with_arguments(json!({"path": s.f.path("outside_library/secret.png")}).as_object().unwrap().clone())).await.unwrap();
    assert_eq!(denied.is_error, Some(true));
    assert_eq!(denied.structured_content.as_ref().unwrap()["error"]["kind"], "denied");
    assert_eq!(s.f.vision.calls.load(std::sync::atomic::Ordering::SeqCst), 0);

    let propose = client.call_tool(CallToolRequestParams::new("metadata_propose").with_arguments(json!({"path": s.f.path("photos/city1.png"), "tags_add": ["neon"]}).as_object().unwrap().clone())).await.unwrap();
    assert_eq!(propose.structured_content.as_ref().unwrap()["written"], false);

    let unknown = client.call_tool(CallToolRequestParams::new("media.search")).await;
    assert!(unknown.is_err(), "an unknown tool is a protocol error");
    let invalid = client.call_tool(CallToolRequestParams::new("library_search").with_arguments(json!({"limit": "many"}).as_object().unwrap().clone())).await;
    assert!(invalid.is_err(), "bad arguments are a protocol error");

    let log = s.hub.activity.recent(50);
    assert!(log.iter().any(|e| e.tool == "util_ping" && e.ok && e.client.contains("@127.0.0.1")), "{log:?}");
    assert!(s.hub.stats.snapshot().tool_calls >= 7);

    let clients = s.hub.clients();
    assert_eq!(clients.len(), 1, "{clients:?}");
    assert!(clients[0].requests >= 5);
    assert!(s.hub.conns.events(10).iter().any(|e| e.kind == EventKind::Opened));
    assert_eq!(MCP_STATUS.snapshot().state, StatusState::Running);
    assert!(MCP_STATUS.snapshot().detail.contains("1 client"), "{}", MCP_STATUS.snapshot().detail);

    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn a_client_with_the_wrong_token_cannot_connect() {
    let s = served().await;
    let attempt = ().serve(s.rmcp_transport("not-the-token")).await;
    assert!(attempt.is_err());
    assert!(s.hub.clients().is_empty());
    s.hub.stop().await;
}

#[tokio::test]
async fn the_server_stops_and_starts_again() {
    let s = served().await;
    assert!(matches!(s.hub.state(), ServerState::Running { .. }));
    assert!(s.hub.start().await.unwrap_err().contains("already running"));
    s.hub.stop().await;
    assert_eq!(s.hub.state(), ServerState::Stopped);
    assert_eq!(MCP_STATUS.snapshot().state, StatusState::Idle);
    assert!(s.client().post(s.url()).body(INIT).send().await.is_err(), "nothing listens after stop");
    let again = s.hub.start().await.unwrap();
    assert!(matches!(s.hub.state(), ServerState::Running { addr, .. } if addr == again));
    let res = s.client().post(format!("http://{again}/mcp")).header("Authorization", format!("Bearer {}", s.token)).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(INIT).send().await.unwrap();
    assert_eq!(res.status(), 200);
    s.hub.stop().await;
    s.hub.stop().await;
}

#[tokio::test]
async fn rotating_the_token_locks_out_the_old_one() {
    let s = served().await;
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200);
    let fresh = s.hub.rotate_token().unwrap();
    assert_ne!(fresh, s.token);
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 401);
    assert_eq!(s.post(Some(&fresh), INIT).await.status(), 200);
    assert_eq!(std::fs::read_to_string(crate::ai::mcp::auth::token_path(s.hub.dir().unwrap())).unwrap(), fresh);
    s.hub.stop().await;
}

#[tokio::test]
async fn a_port_in_use_is_a_failed_start_not_a_panic() {
    let _lock = SERIAL.lock().await;
    let f = fixture().await;
    let blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = blocker.local_addr().unwrap().port();
    let dir = tempfile::tempdir().unwrap();
    let cfg = McpConfig { enabled: true, port, ..Default::default() };
    let hub = Hub::new(f.ctx.clone(), Some(dir.path().to_path_buf()), cfg, f.activity.clone(), f.stats.clone(), Arc::new(ConnectionLog::default()), f.ctx.clock.clone());
    let err = hub.start().await.unwrap_err();
    assert!(err.contains("cannot listen"), "{err}");
    assert!(matches!(hub.state(), ServerState::Failed(_)));
    assert_eq!(MCP_STATUS.snapshot().state, StatusState::Error);
    drop(blocker);
    hub.start().await.unwrap();
    assert!(matches!(hub.state(), ServerState::Running { .. }));
    hub.stop().await;
}

#[tokio::test]
async fn tailnet_settings_without_hosts_do_not_start() {
    let _lock = SERIAL.lock().await;
    let f = fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let cfg = McpConfig { enabled: true, port: 0, allow_tailnet: true, bind_addr: "127.0.0.1".into(), ..Default::default() };
    let hub = Hub::new(f.ctx.clone(), Some(dir.path().to_path_buf()), cfg, f.activity.clone(), f.stats.clone(), Arc::new(ConnectionLog::default()), f.ctx.clock.clone());
    let err = hub.start().await.unwrap_err();
    assert!(err.contains("allowed host"), "{err}");
    assert!(matches!(hub.state(), ServerState::Failed(_)));
}

#[tokio::test]
async fn an_allowed_extra_host_works_when_tailnet_access_is_on() {
    let s = served_with(|c| {
        c.allow_tailnet = true;
        c.bind_addr = "127.0.0.1".into();
        c.allowed_hosts = vec!["desk.tail.example".into()];
    })
    .await;
    let named = s.client().post(s.url()).header("Host", "desk.tail.example:9871").header("Authorization", format!("Bearer {}", s.token)).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json").body(INIT).send().await.unwrap();
    assert_eq!(named.status(), 200);
    let other = s.client().post(s.url()).header("Host", "other.example").header("Authorization", format!("Bearer {}", s.token)).body(INIT).send().await.unwrap();
    assert_eq!(other.status(), 403);
    s.hub.stop().await;
}

#[tokio::test]
async fn the_gateway_snippet_names_the_url_and_a_token_placeholder() {
    let s = served().await;
    let snippet = s.hub.gateway_snippet();
    assert!(snippet.contains(&format!("url = \"http://127.0.0.1:{}/mcp\"", s.addr.port())), "{snippet}");
    assert!(snippet.contains("Bearer <paste the token"));
    assert!(!snippet.contains(&s.token), "the real token is never part of the snippet");
    assert!(snippet.contains("\"aimedia__library_search\"") && snippet.contains("always_ask = [\"aimedia__describe_image\", \"aimedia__clip_embed\", \"aimedia__image_edit_start\", \"aimedia__taxonomy_propose_rename\"]"));
    assert!(!snippet.contains("auto_approve = [\"aimedia__describe_image"));
    assert!(snippet.contains("pinned_resources = [\"library://guide\"]") && snippet.contains("[agents.media]"));
    s.hub.stop().await;
}
