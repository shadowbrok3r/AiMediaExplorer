//! The write tools, the edit job tools and the guide resource through the HTTP server with a real MCP client.

use std::time::{Duration, Instant};

use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult, ReadResourceRequestParams, ResourceContents};
use serde_json::{Value, json};

use super::http::{INIT, served};
use crate::ai::mcp::guide::{DEFAULT_GUIDE, GUIDE_URI};

fn args(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().unwrap().clone()
}

async fn call(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, name: &str, a: Value) -> Result<CallToolResult, rmcp::ServiceError> {
    client.call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(args(a))).await
}

fn data(r: &CallToolResult) -> Value {
    r.structured_content.clone().expect("structured content")
}

#[tokio::test]
async fn the_new_tools_are_listed_with_schemas_and_hints() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.unwrap();
    let tools = client.list_all_tools().await.unwrap();
    let find = |n: &str| tools.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("{n} is not listed"));
    for name in ["taxonomy_propose_rename", "taxonomy_list_proposals", "image_edit_start", "image_edit_status", "image_edit_cancel"] {
        let t = find(name);
        assert!(name.len() <= 40 && format!("aimedia__{name}").len() <= 64);
        assert_eq!(t.input_schema.get("type"), Some(&json!("object")), "{name}");
        assert!(t.input_schema.get("properties").is_some_and(Value::is_object), "{name}");
        assert!(t.description.as_ref().is_some_and(|d| d.len() > 60), "{name}");
        let hints = t.annotations.as_ref().expect("annotations");
        assert_eq!(hints.destructive_hint, Some(false), "{name} is never destructive");
    }
    assert_eq!(find("taxonomy_list_proposals").annotations.as_ref().unwrap().read_only_hint, Some(true));
    assert_eq!(find("image_edit_status").annotations.as_ref().unwrap().read_only_hint, Some(true));
    assert_eq!(find("taxonomy_propose_rename").annotations.as_ref().unwrap().read_only_hint, Some(false));
    let schema = Value::Object(find("taxonomy_propose_rename").input_schema.as_ref().clone()).to_string();
    assert!(schema.contains("\"tag\"") && schema.contains("\"category\""), "the kind choices are in the schema: {schema}");
    let required = find("image_edit_start").input_schema.get("required").unwrap().to_string();
    assert!(required.contains("path") && required.contains("prompt"), "{required}");
    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn taxonomy_proposals_work_over_the_wire_and_write_nothing() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.unwrap();
    let filed = call(&client, "taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach", "sea"], "to": "coast", "reason": "same place"})).await.unwrap();
    assert_ne!(filed.is_error, Some(true));
    let d = data(&filed);
    assert_eq!((d["status"].clone(), d["written"].clone(), d["preview"]["items"].clone()), (json!("pending"), json!(false), json!(2)));
    assert!(d["source"].as_str().unwrap().contains("@127.0.0.1"), "the client is named by user agent and address: {d}");
    let tags: Vec<Option<Vec<String>>> = s.f.db.query("SELECT VALUE tags FROM thumbnails WHERE filename = 'beach1.png'").await.unwrap().take(0).unwrap();
    assert!(tags[0].clone().unwrap().contains(&"beach".to_string()), "the library is untouched");
    let listed = call(&client, "taxonomy_list_proposals", json!({"status": "pending"})).await.unwrap();
    assert_eq!(data(&listed)["total"], 1);
    let missing = call(&client, "taxonomy_propose_rename", json!({"kind": "tag", "from": ["no-such-tag"], "to": "x"})).await.unwrap();
    assert_eq!(missing.is_error, Some(true));
    assert_eq!(data(&missing)["error"]["kind"], "not_found");
    assert!(call(&client, "taxonomy_propose_rename", json!({"kind": "folder", "from": ["a"], "to": "b"})).await.is_err(), "an unknown kind is a protocol error");
    assert!(call(&client, "taxonomy_propose_rename", json!({"kind": "tag", "from": [], "to": "b"})).await.is_err(), "an empty from is a protocol error");
    let many: Vec<String> = (0..11).map(|i| format!("t{i}")).collect();
    assert!(call(&client, "taxonomy_propose_rename", json!({"kind": "tag", "from": many, "to": "b"})).await.is_err(), "more than ten names");
    let accepted = s.f.ctx.agent.taxonomy.accept(filed.structured_content.as_ref().unwrap()["id"].as_str().unwrap()).await.unwrap();
    assert_eq!(accepted.status.as_str(), "accepted");
    let after = call(&client, "library_search", json!({"tags": ["coast"]})).await.unwrap();
    assert_eq!(data(&after)["total"], 2, "the review, not the agent, changed the library");
    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn an_image_edit_runs_over_the_wire_and_its_result_waits_for_the_user() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.unwrap();
    let original = std::fs::read(s.f.path("photos/forest1.png")).unwrap();
    let denied = call(&client, "image_edit_start", json!({"path": s.f.path("outside_library/secret.png"), "prompt": "x"})).await.unwrap();
    assert_eq!((denied.is_error, data(&denied)["error"]["kind"].clone()), (Some(true), json!("denied")));
    assert!(call(&client, "image_edit_start", json!({"path": s.f.path("photos/forest1.png"), "prompt": "x", "steps": 99})).await.is_err(), "a step count over the limit is a protocol error");
    assert!(call(&client, "image_edit_start", json!({"path": s.f.path("photos/forest1.png")})).await.is_err(), "the prompt is required");
    let started = call(&client, "image_edit_start", json!({"path": s.f.path("photos/forest1.png"), "prompt": "make it black and white", "steps": 2})).await.unwrap();
    let id = data(&started)["job_id"].as_str().unwrap().to_string();
    let end = Instant::now() + Duration::from_secs(20);
    let done = loop {
        let status = call(&client, "image_edit_status", json!({"job_id": id})).await.unwrap();
        if data(&status)["state"] == "done" {
            break status;
        }
        assert!(Instant::now() < end, "never finished: {}", data(&status));
        tokio::time::sleep(Duration::from_millis(30)).await;
    };
    assert_eq!(data(&done)["review"], "pending");
    let with_preview = call(&client, "image_edit_status", json!({"job_id": id, "include_preview": true})).await.unwrap();
    let blocks = serde_json::to_value(&with_preview.content).unwrap();
    assert!(blocks.as_array().unwrap().iter().any(|b| b["type"] == "image" && b["mimeType"] == "image/jpeg"), "{blocks}");
    assert_eq!(std::fs::read(s.f.path("photos/forest1.png")).unwrap(), original, "the original is untouched");
    assert!(call(&client, "image_edit_cancel", json!({"job_id": id})).await.is_err(), "a finished job cannot be cancelled");
    let unknown = call(&client, "image_edit_cancel", json!({"job_id": "enope"})).await.unwrap();
    assert_eq!((unknown.is_error, data(&unknown)["error"]["kind"].clone()), (Some(true), json!("not_found")));
    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn the_guide_is_a_resource_that_follows_the_settings() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.unwrap();
    assert!(client.peer_info().unwrap().capabilities.resources.is_some(), "the server advertises resources");
    let listed = client.list_all_resources().await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!((listed[0].uri.as_str(), listed[0].mime_type.as_deref()), (GUIDE_URI, Some("text/markdown")));
    let read = client.read_resource(ReadResourceRequestParams::new(GUIDE_URI)).await.unwrap();
    let ResourceContents::TextResourceContents { uri, text, mime_type, .. } = &read.contents[0] else { panic!("the guide is text") };
    assert_eq!((uri.as_str(), text.as_str(), mime_type.as_deref()), (GUIDE_URI, DEFAULT_GUIDE, Some("text/markdown")));
    s.f.guide.0.lock().unwrap().set_text("# Mine\nNo emoji tags.");
    let edited = client.read_resource(ReadResourceRequestParams::new(GUIDE_URI)).await.unwrap();
    let ResourceContents::TextResourceContents { text, .. } = &edited.contents[0] else { panic!("text") };
    assert_eq!(text, "# Mine\nNo emoji tags.");
    assert!(client.read_resource(ReadResourceRequestParams::new("library://secrets")).await.is_err(), "unknown resources are refused");
    assert!(client.read_resource(ReadResourceRequestParams::new("file:///etc/passwd")).await.is_err());
    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn resources_need_the_token_like_everything_else() {
    let s = served().await;
    let list = r#"{"jsonrpc":"2.0","id":2,"method":"resources/list"}"#;
    assert_eq!(s.post(None, list).await.status(), 401);
    assert_eq!(s.post(Some("wrong"), list).await.status(), 401);
    assert_eq!(s.post(Some(&s.token), INIT).await.status(), 200);
    s.hub.stop().await;
}

#[tokio::test]
async fn the_instructions_point_at_the_guide_and_the_new_tools() {
    let s = served().await;
    let client = ().serve(s.rmcp_transport(&s.token)).await.unwrap();
    let text = client.peer_info().unwrap().instructions.clone().unwrap_or_default();
    for word in ["library://guide", "taxonomy_propose_rename", "image_edit_start", "image_edit_status"] {
        assert!(text.contains(word), "{word} in {text}");
    }
    client.cancel().await.ok();
    s.hub.stop().await;
}

#[tokio::test]
async fn the_snippet_test_pings_the_server_through_its_own_address() {
    let s = served().await;
    let ping = crate::zc::snippet::ping_server(&s.url(), &s.token).await;
    assert!(ping.ok, "{ping:?}");
    assert!(ping.detail.contains("pong: snippet test"), "{}", ping.detail);
    let log = s.hub.activity.recent(10);
    assert!(log.iter().any(|e| e.tool == "util_ping" && e.ok), "the call went through the tool registry: {log:?}");
    let wrong = crate::zc::snippet::ping_server(&s.url(), "not-the-token").await;
    assert!(!wrong.ok && wrong.detail.contains("401"), "{wrong:?}");
    assert!(!wrong.detail.contains("not-the-token"));
    s.hub.stop().await;
    let stopped = crate::zc::snippet::ping_server(&s.url(), &s.token).await;
    assert!(!stopped.ok && stopped.detail.contains("did not answer"), "{stopped:?}");
}
