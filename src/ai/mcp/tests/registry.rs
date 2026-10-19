use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::registry::{MAX_RESULT_BYTES, ToolClass, ToolRegistry, entry};
use crate::ai::mcp::server::{call_result, error_result};
use crate::ai::mcp::test_support::*;

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
struct Wait {
    millis: u64,
    size: Option<usize>,
}

async fn wait(_ctx: Arc<ToolCtx>, _call: CallInfo, args: Wait) -> Result<ToolOutput, ToolError> {
    tokio::time::sleep(Duration::from_millis(args.millis)).await;
    Ok(ToolOutput::new("waited", json!({"blob": "x".repeat(args.size.unwrap_or(0))})))
}

fn with_slow_tool(f: &Fixture, timeout: Duration) -> ToolRegistry {
    ToolRegistry::new(vec![entry("wait", ToolClass::Read, timeout, "Wait", "Sleeps for the given time, then answers; used by the tests.", wait)], f.activity.clone(), f.stats.clone())
}

fn call_info() -> CallInfo {
    CallInfo { client: "tester".into(), cancel: CancellationToken::new() }
}

#[tokio::test]
async fn unknown_tools_and_bad_arguments_are_reported_plainly() {
    let f = fixture().await;
    let unknown = f.call("media.search", json!({})).await.unwrap_err();
    assert_eq!(unknown.kind(), "not_found");
    assert!(unknown.message().contains("media.search"));
    let bad = f.call("library_stats", json!("not an object")).await.unwrap_err();
    assert_eq!(bad.kind(), "invalid_arguments");
    let missing = f.call("library_get_item", json!({})).await.unwrap_err();
    assert!(missing.message().contains("path"), "{}", missing.message());
    assert!(f.call("library_stats", serde_json::Value::Null).await.is_ok(), "no arguments is an empty object");
}

#[tokio::test]
async fn a_call_over_its_time_limit_times_out() {
    let f = fixture().await;
    let registry = with_slow_tool(&f, Duration::from_millis(40));
    let started = std::time::Instant::now();
    let err = registry.call(f.ctx.clone(), call_info(), "wait", json!({"millis": 5000})).await.unwrap_err();
    assert_eq!(err.kind(), "timeout");
    assert!(err.message().contains("wait"));
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(registry.call(f.ctx.clone(), call_info(), "wait", json!({"millis": 1})).await.is_ok());
}

#[tokio::test]
async fn a_cancelled_call_returns_at_once() {
    let f = fixture().await;
    let registry = with_slow_tool(&f, Duration::from_secs(30));
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        trigger.cancel();
    });
    let err = registry.call(f.ctx.clone(), CallInfo { client: "t".into(), cancel }, "wait", json!({"millis": 10_000})).await.unwrap_err();
    assert_eq!(err, ToolError::Cancelled);
}

#[tokio::test]
async fn oversized_results_are_refused() {
    let f = fixture().await;
    let registry = with_slow_tool(&f, Duration::from_secs(5));
    let err = registry.call(f.ctx.clone(), call_info(), "wait", json!({"millis": 0, "size": MAX_RESULT_BYTES + 10})).await.unwrap_err();
    assert_eq!(err.kind(), "failed");
    assert!(err.message().contains("fewer items"));
    assert!(registry.call(f.ctx.clone(), call_info(), "wait", json!({"millis": 0, "size": 1000})).await.is_ok());
}

#[tokio::test]
async fn every_call_is_logged_with_its_outcome() {
    let f = fixture().await;
    f.call_as("util_ping", json!({"message": "hello"}), "zeroclaw@10.0.0.2", CancellationToken::new()).await.unwrap();
    f.call("library_get_item", json!({"path": f.path("photos/none.png")})).await.unwrap_err();
    f.call("nope", json!({})).await.unwrap_err();
    let log = f.activity.recent(10);
    assert_eq!(log.len(), 3);
    assert_eq!((log[2].tool.as_str(), log[2].client.as_str(), log[2].ok), ("util_ping", "zeroclaw@10.0.0.2", true));
    assert!(log[2].args.contains("hello"));
    assert_eq!((log[1].tool.as_str(), log[1].ok, log[1].error_kind.as_deref()), ("library_get_item", false, Some("not_found")));
    assert_eq!(log[0].error_kind.as_deref(), Some("not_found"));
    assert!(log[2].result_bytes > 0);
    assert_eq!(f.stats.snapshot().tool_calls, 3);
    assert_eq!(log[2].at_ms, f.ctx.clock.unix_ms());
}

#[tokio::test]
async fn ping_echoes_a_cleaned_message() {
    let f = fixture().await;
    let out = f.call("util_ping", json!({"message": "hi\u{7}\nthere"})).await.unwrap();
    assert_eq!(out.text, "pong: hi there");
    assert_eq!(out.data["server"], env!("CARGO_PKG_NAME"));
    assert_eq!(out.data["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(f.call("util_ping", json!({})).await.unwrap().text, "pong: ");
}

#[tokio::test]
async fn tool_results_carry_text_json_and_images_on_the_wire() {
    let f = fixture().await;
    let out = f.call("library_get_preview", json!({"path": f.path("photos/beach1.png")})).await.unwrap();
    let result = call_result(out);
    assert_ne!(result.is_error, Some(true));
    let blocks = serde_json::to_value(&result.content).unwrap();
    assert_eq!(blocks[0]["type"], "text");
    assert!(blocks[0]["text"].as_str().unwrap().starts_with("preview of"));
    assert_eq!(blocks[1]["type"], "text");
    assert!(serde_json::from_str::<serde_json::Value>(blocks[1]["text"].as_str().unwrap()).unwrap()["width"].is_number());
    assert_eq!(blocks[2]["type"], "image");
    assert_eq!(blocks[2]["mimeType"], "image/jpeg");
    assert_eq!(result.structured_content.as_ref().unwrap()["height"], 8);
}

#[tokio::test]
async fn tool_errors_become_error_results_with_a_kind() {
    let result = error_result(&ToolError::Denied("outside the folders".into()));
    assert_eq!(result.is_error, Some(true));
    let blocks = serde_json::to_value(&result.content).unwrap();
    assert_eq!(blocks[0]["text"], "denied: outside the folders");
    assert_eq!(result.structured_content.unwrap()["error"], json!({"kind": "denied", "message": "outside the folders"}));
}

#[tokio::test]
async fn the_catalog_lists_every_registered_tool_with_its_class() {
    let f = fixture().await;
    let catalog = f.registry.catalog();
    let class = |name: &str| catalog.iter().find(|r| r.name == name).map(|r| r.class);
    assert_eq!(class("library_search"), Some(ToolClass::Read));
    assert_eq!(class("ui_run"), Some(ToolClass::Ui));
    assert_eq!(class("describe_image"), Some(ToolClass::Gpu));
    assert_eq!(class("metadata_propose"), Some(ToolClass::Propose));
    assert_eq!(f.registry.tools().len(), catalog.len());
    assert!(f.registry.get("util_ping").is_some() && f.registry.get("util.ping").is_none());
}
