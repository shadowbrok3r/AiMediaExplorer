use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::ai::mcp::gpu::GpuLimits;
use crate::ai::mcp::proposals::Status;
use crate::ai::mcp::test_support::*;

#[tokio::test]
async fn describing_is_a_dry_run_unless_told_otherwise() {
    let f = fixture().await;
    let out = f.call("describe_image", json!({"path": f.path("photos/forest1.png")})).await.unwrap();
    assert_eq!(out.data["dry_run"], true);
    assert_eq!(out.data["stored"], false);
    assert_eq!(out.data["caption"], "forest1 caption");
    assert_eq!(out.data["tags"], json!(["forest1", "generated"]));
    assert!(out.data["gpu"]["job_id"].as_u64().is_some());
    assert!(out.data.get("proposal_id").is_none());
    assert_eq!(f.vision.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.ctx.proposals.counts().await.unwrap().pending, 0);
}

#[tokio::test]
async fn describing_without_a_dry_run_files_a_proposal_instead_of_writing() {
    let f = fixture().await;
    let out = f.call("describe_image", json!({"path": f.path("photos/sub/beach3.png"), "dry_run": false})).await.unwrap();
    assert_eq!(out.data["stored"], true);
    let id = out.data["proposal_id"].as_str().unwrap();
    let proposal = f.ctx.proposals.get(id).await.unwrap().unwrap();
    assert_eq!(proposal.status, Status::Pending);
    assert_eq!(proposal.changes.caption.as_deref(), Some("beach3 caption"));
    assert!(proposal.changes.tags_add.contains(&"generated".to_string()));
    assert!(proposal.source.starts_with("describe_image:"));
    let rows: Vec<Option<String>> = f.db.query("SELECT VALUE caption FROM thumbnails WHERE filename = 'beach3.png'").await.unwrap().take(0).unwrap();
    assert_eq!(rows.into_iter().next().flatten(), None, "the library row is unchanged until the proposal is accepted");
}

#[tokio::test]
async fn describing_a_file_outside_the_library_returns_the_text_but_files_nothing() {
    let f = fixture().await;
    std::fs::copy(f.path("photos/beach1.png"), f.path("photos/loose.png")).unwrap();
    let out = f.call("describe_image", json!({"path": f.path("photos/loose.png"), "dry_run": false})).await.unwrap();
    assert_eq!(out.data["stored"], false);
    assert!(out.data["note"].as_str().unwrap().contains("not in the library"));
    assert_eq!(f.ctx.proposals.counts().await.unwrap().pending, 0);
}

#[tokio::test]
async fn gpu_tools_only_read_files_inside_the_allowed_folders() {
    let f = fixture().await;
    for tool in ["describe_image", "clip_embed", "library_get_preview"] {
        let outside = f.call(tool, json!({"path": f.path("outside_library/secret.png")})).await.unwrap_err();
        assert_eq!(outside.kind(), "denied", "{tool}");
        let traversal = f.call(tool, json!({"path": format!("{}/../outside_library/secret.png", f.path("photos"))})).await.unwrap_err();
        assert_eq!(traversal.kind(), "invalid_arguments", "{tool}");
        let missing = f.call(tool, json!({"path": f.path("photos/missing.png")})).await.unwrap_err();
        assert_eq!(missing.kind(), "not_found", "{tool}");
        let text = f.call(tool, json!({"path": f.path("photos/notes.txt")})).await.unwrap_err();
        assert_eq!(text.kind(), "invalid_arguments", "{tool}");
        let relative = f.call(tool, json!({"path": "photos/beach1.png"})).await.unwrap_err();
        assert_eq!(relative.kind(), "invalid_arguments", "{tool}");
    }
    assert_eq!(f.vision.calls.load(Ordering::SeqCst) + f.clip.image_calls.load(Ordering::SeqCst), 0, "no model ran for a refused path");
    assert_eq!(f.ctx.gpu.status().completed, 0);
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_inside_a_root_cannot_reach_a_file_outside() {
    let f = fixture().await;
    std::os::unix::fs::symlink(f.path("outside_library/secret.png"), f.path("photos/link.png")).unwrap();
    for tool in ["describe_image", "clip_embed", "library_get_preview"] {
        assert_eq!(f.call(tool, json!({"path": f.path("photos/link.png")})).await.unwrap_err().kind(), "denied", "{tool}");
    }
}

#[tokio::test]
async fn embedding_reports_the_vector_and_stores_it_only_on_request() {
    let f = fixture().await;
    let dry = f.call("clip_embed", json!({"path": f.path("photos/sub/beach3.png")})).await.unwrap();
    assert_eq!(dry.data["dimensions"], 1024);
    assert!((dry.data["norm"].as_f64().unwrap() - 1.0).abs() < 0.001);
    assert_eq!(dry.data["stored"], false);
    let count = || row_count(&f.db, "clip_embeddings");
    assert_eq!(count().await, 4);
    let stored = f.call("clip_embed", json!({"path": f.path("photos/sub/beach3.png"), "dry_run": false})).await.unwrap();
    assert_eq!(stored.data["stored"], true);
    assert_eq!(count().await, 5);
    f.call("clip_embed", json!({"path": f.path("photos/sub/beach3.png"), "dry_run": false})).await.unwrap();
    assert_eq!(count().await, 5, "a second run updates the same row");
    let item = f.call("library_get_item", json!({"path": f.path("photos/sub/beach3.png")})).await.unwrap();
    assert_eq!(item.data["clip_embedding"]["stored"], true);
}

#[tokio::test]
async fn an_embedding_for_a_file_outside_the_library_is_not_stored() {
    let f = fixture().await;
    std::fs::copy(f.path("photos/beach1.png"), f.path("photos/loose.png")).unwrap();
    let out = f.call("clip_embed", json!({"path": f.path("photos/loose.png"), "dry_run": false})).await.unwrap();
    assert_eq!(out.data["stored"], false);
    assert!(out.data["note"].as_str().unwrap().contains("not in the library"));
}

#[tokio::test]
async fn previews_come_back_as_jpeg_image_content() {
    let f = fixture().await;
    let out = f.call("library_get_preview", json!({"path": f.path("photos/beach1.png"), "max_px": 64})).await.unwrap();
    assert_eq!(out.images.len(), 1);
    assert_eq!(out.images[0].mime, "image/jpeg");
    assert_eq!((out.data["width"].as_u64(), out.data["height"].as_u64()), (Some(8), Some(8)));
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD.decode(&out.images[0].data_base64).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
}

#[tokio::test]
async fn model_jobs_never_overlap() {
    let f = fixture().await;
    *f.vision.delay.lock().unwrap() = Duration::from_millis(40);
    *f.clip.delay.lock().unwrap() = Duration::from_millis(40);
    let calls = [
        f.call("describe_image", json!({"path": f.path("photos/beach1.png")})),
        f.call("clip_embed", json!({"path": f.path("photos/beach2.png")})),
        f.call("describe_image", json!({"path": f.path("photos/forest1.png")})),
        f.call("library_semantic_search", json!({"query": "beach"})),
    ];
    let clock = f.clock.clone();
    let ticker = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            clock.advance(Duration::from_millis(5));
        }
    });
    let results = futures_util::future::join_all(calls).await;
    ticker.abort();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    assert_eq!(f.vision.max_active.load(Ordering::SeqCst), 1);
    assert_eq!(f.clip.max_active.load(Ordering::SeqCst), 1);
    let status = f.ctx.gpu.status();
    assert_eq!(status.completed, 4);
    let waits: Vec<u64> = results.iter().map(|r| r.as_ref().unwrap().data["gpu"]["waited_ms"].as_u64().unwrap()).collect();
    assert!(waits.iter().filter(|w| **w >= 30).count() >= 2, "later jobs waited for the earlier ones: {waits:?}");
    let mut depths: Vec<u64> = results.iter().map(|r| r.as_ref().unwrap().data["gpu"]["queue_depth_at_submit"].as_u64().unwrap()).collect();
    depths.sort();
    assert_eq!(depths.first(), Some(&0));
    assert!(depths.last().copied() >= Some(2), "jobs queued behind each other: {depths:?}");
}

#[tokio::test]
async fn a_full_queue_answers_busy_and_recovers() {
    let f = fixture_with(GpuLimits { max_queued: 1, queue_timeout: Duration::from_secs(30), job_timeout: Duration::from_secs(30) }).await;
    *f.vision.delay.lock().unwrap() = Duration::from_millis(120);
    let a = f.call("describe_image", json!({"path": f.path("photos/beach1.png")}));
    let b = f.call("describe_image", json!({"path": f.path("photos/beach2.png")}));
    let c = f.call("describe_image", json!({"path": f.path("photos/forest1.png")}));
    let (a, b, c) = tokio::join!(a, b, c);
    let results = [a, b, c];
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 2);
    let busy = results.iter().find_map(|r| r.as_ref().err()).expect("one call is turned away");
    assert_eq!(busy.kind(), "busy");
    assert!(busy.message().contains("queue is full"));
    let again = f.call("describe_image", json!({"path": f.path("photos/forest1.png")})).await;
    assert!(again.is_ok(), "the queue accepts jobs again once it drains");
}

#[tokio::test]
async fn a_cancelled_call_stops_its_job_and_frees_the_gpu() {
    let f = fixture().await;
    *f.vision.delay.lock().unwrap() = Duration::from_secs(5);
    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        canceller.cancel();
    });
    let started = std::time::Instant::now();
    let err = f.call_as("describe_image", json!({"path": f.path("photos/beach1.png")}), "tester", token).await.unwrap_err();
    assert_eq!(err.kind(), "cancelled");
    assert!(started.elapsed() < Duration::from_secs(2));
    *f.vision.delay.lock().unwrap() = Duration::ZERO;
    assert!(f.call("describe_image", json!({"path": f.path("photos/beach2.png")})).await.is_ok());
    assert_eq!(f.ctx.gpu.status().cancelled, 1);
}

#[tokio::test]
async fn the_queue_status_tool_reports_limits_and_history() {
    let f = fixture().await;
    f.call("clip_embed", json!({"path": f.path("photos/beach1.png")})).await.unwrap();
    let out = f.call("gpu_queue_status", json!({})).await.unwrap();
    assert_eq!(out.data["max_queued"], 8);
    assert_eq!(out.data["completed"], 1);
    assert_eq!(out.data["recent"][0]["kind"], "clip_image");
    assert_eq!(out.data["recent"][0]["state"], "done");
    assert!(out.data["running"].is_null());
    assert_eq!(out.data["job_timeout_secs"], 30);
}

#[tokio::test]
async fn model_failures_surface_as_tool_errors() {
    let f = fixture().await;
    f.clip.fail.store(true, Ordering::SeqCst);
    let err = f.call("clip_embed", json!({"path": f.path("photos/beach1.png")})).await.unwrap_err();
    assert_eq!(err.kind(), "failed");
    assert!(err.message().contains("broken"));
    f.clip.fail.store(false, Ordering::SeqCst);
    assert!(f.call("clip_embed", json!({"path": f.path("photos/beach1.png")})).await.is_ok());
}
