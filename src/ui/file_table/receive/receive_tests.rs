use crate::ScanEnvelope;
use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::utilities::scan::ScanMsg;
use crate::utilities::types::{FoundFile, MediaKind};
use eframe::egui::Context;
use std::path::PathBuf;

const SCAN_ID: u64 = 4_242;

fn explorer(recursive: bool, page_size: usize) -> FileExplorer {
    let mut ex = FileExplorer::new(true);
    ex.current_scan_id = Some(SCAN_ID);
    ex.owning_scan_id = Some(SCAN_ID);
    ex.recursive_scan = recursive;
    ex.recursive_page_size = page_size;
    ex
}

fn found(i: usize) -> FoundFile {
    FoundFile {
        path: PathBuf::from(format!("/photos/dir{}/img_{i}.jpg", i % 31)),
        modified: Some(chrono::Local::now()),
        created: None,
        size: Some(50_000 + i as u64),
        kind: MediaKind::Image,
        thumb_data: None,
    }
}

fn send(ex: &FileExplorer, msg: ScanMsg) {
    ex.scan_tx.send(ScanEnvelope { scan_id: SCAN_ID, msg }).unwrap();
}

fn send_batch(ex: &FileExplorer, n: usize) {
    send(ex, ScanMsg::FoundBatch((0..n).map(found).collect()));
}

fn payload(tag: &str, i: usize) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(format!("{tag}-{i}"))
}

fn send_thumb(ex: &FileExplorer, i: usize, tag: &str) {
    send(ex, ScanMsg::UpdateThumb { path: found(i).path, thumb: payload(tag, i) });
}

fn drain_all(ex: &mut FileExplorer, ctx: &Context) -> usize {
    let mut frames = 0;
    while !ex.scan_rx.is_empty() && frames < 10_000 {
        ex.receive_scan(ctx);
        frames += 1;
    }
    frames
}

fn thumb_of(rows: &[Thumbnail], i: usize) -> Option<String> {
    let p = found(i).path.display().to_string();
    rows.iter().find(|r| r.path == p).and_then(|r| r.thumbnail_b64.clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn thumbs_reach_table_snapshot_and_scheduling_set() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 40);
    ex.receive_scan(&ctx);
    assert_eq!(ex.table.len(), 40);
    assert_eq!(ex.last_scan_rows.len(), 40);
    assert!(ex.thumb_scheduled.contains(&found(7).path.display().to_string()));
    for i in [0usize, 7, 39] {
        send_thumb(&ex, i, "t");
    }
    ex.receive_scan(&ctx);
    for i in [0usize, 7, 39] {
        assert_eq!(thumb_of(&ex.table, i), Some(payload("t", i)));
        assert_eq!(thumb_of(&ex.last_scan_rows, i), Some(payload("t", i)));
    }
    assert_eq!(thumb_of(&ex.table, 8), None);
    assert_eq!(ex.pending_thumb_rows.len(), 3);
    assert_eq!(ex.table_index.len(), ex.table.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn off_page_thumbs_update_only_the_snapshot() {
    let ctx = Context::default();
    let mut ex = explorer(true, 10);
    send_batch(&ex, 35);
    ex.receive_scan(&ctx);
    assert_eq!(ex.last_scan_rows.len(), 35);
    assert!(ex.table.len() <= 10);
    send_thumb(&ex, 30, "off");
    ex.receive_scan(&ctx);
    assert_eq!(thumb_of(&ex.last_scan_rows, 30), Some(payload("off", 30)));
    assert_eq!(thumb_of(&ex.table, 30), None);
    assert!(ex.viewer.thumb_cache.contains_key(&found(30).path.display().to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn filtered_snapshot_is_updated_even_after_replacement() {
    let ctx = Context::default();
    let mut ex = explorer(true, 10);
    send_batch(&ex, 25);
    ex.receive_scan(&ctx);
    ex.recursive_filtered_rows = ex.last_scan_rows.clone();
    send_thumb(&ex, 12, "f1");
    ex.receive_scan(&ctx);
    assert_eq!(thumb_of(&ex.recursive_filtered_rows, 12), Some(payload("f1", 12)));
    ex.recursive_filtered_rows.reverse();
    send_thumb(&ex, 3, "f2");
    ex.receive_scan(&ctx);
    assert_eq!(thumb_of(&ex.recursive_filtered_rows, 3), Some(payload("f2", 3)));
    assert_eq!(thumb_of(&ex.last_scan_rows, 3), Some(payload("f2", 3)));
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_table_index_is_repaired_after_external_row_removal() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 30);
    ex.receive_scan(&ctx);
    let removed: std::collections::HashSet<String> = (0..10).map(|i| found(i).path.display().to_string()).collect();
    ex.table.retain(|r| !removed.contains(&r.path));
    send_thumb(&ex, 20, "r");
    ex.receive_scan(&ctx);
    assert_eq!(thumb_of(&ex.table, 20), Some(payload("r", 20)));
    assert_eq!(ex.table.iter().filter(|r| r.thumbnail_b64.is_some()).count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn scan_messages_are_drained_within_the_frame_budget() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 3_000);
    ex.receive_scan(&ctx);
    for i in 0..3_000 {
        send_thumb(&ex, i, "b");
    }
    ex.receive_scan(&ctx);
    assert!(!ex.scan_rx.is_empty(), "one frame must not drain 3000 messages");
    let frames = drain_all(&mut ex, &ctx);
    assert!(frames >= 2);
    assert!(ex.scan_rx.is_empty());
    assert_eq!(ex.table.iter().filter(|r| r.thumbnail_b64.is_some()).count(), 3_000);
    assert_eq!(ex.last_scan_rows.iter().filter(|r| r.thumbnail_b64.is_some()).count(), 3_000);
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_thumbnails_are_recorded_with_their_error() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 3);
    send(&ex, ScanMsg::ThumbFailed { path: found(1).path, error: "raw: rawkit panicked".into() });
    ex.recursive_prefetch_inflight = 5;
    ex.receive_scan(&ctx);
    assert_eq!(ex.failed_thumbs.len(), 1);
    assert_eq!(ex.failed_thumbs[0].1, "raw: rawkit panicked");
    assert_eq!(ex.recursive_prefetch_inflight, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn thumbnail_channel_is_drained_with_a_budget() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 300);
    ex.receive_scan(&ctx);
    for i in 0..300 {
        let mut t = Thumbnail::new("x");
        t.path = found(i).path.display().to_string();
        t.filename = format!("img_{i}.jpg");
        t.thumbnail_b64 = Some(payload("sel", i));
        ex.thumbnail_tx.send(t).unwrap();
    }
    ex.receive_thumbnail(&ctx);
    assert!(!ex.thumbnail_rx.is_empty(), "budget must leave work for later frames");
    let mut frames = 1;
    while !ex.thumbnail_rx.is_empty() && frames < 1_000 {
        ex.receive_thumbnail(&ctx);
        frames += 1;
    }
    assert!(frames >= 5, "frames={frames}");
    assert_eq!(ex.table.iter().filter(|r| r.thumbnail_b64.as_deref().is_some_and(|s| s.starts_with(&payload("sel", 0)[..4]))).count(), 300);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_receive_scan_thumb_updates() {
    let ctx = Context::default();
    for n in [5_000usize, 20_000, 100_000] {
        let mut ex = explorer(false, usize::MAX);
        send_batch(&ex, n);
        drain_all(&mut ex, &ctx);
        assert_eq!(ex.last_scan_rows.len(), n);
        for i in 0..n {
            send_thumb(&ex, i, "x");
        }
        let started = std::time::Instant::now();
        let mut worst = std::time::Duration::ZERO;
        let mut frames = 0;
        while !ex.scan_rx.is_empty() {
            let t = std::time::Instant::now();
            ex.receive_scan(&ctx);
            worst = worst.max(t.elapsed());
            frames += 1;
        }
        println!("bench receive_scan n={n}: {:?} total in {frames} frames, worst frame {:?}", started.elapsed(), worst);
        assert_eq!(ex.table.iter().filter(|r| r.thumbnail_b64.is_some()).count(), n);
    }
}
