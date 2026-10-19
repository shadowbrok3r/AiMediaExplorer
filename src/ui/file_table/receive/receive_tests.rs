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

fn send_ready(ex: &FileExplorer, i: usize) {
    send(ex, ScanMsg::ThumbReady { path: found(i).path, key: format!("key-{i}") });
}

fn drain_all(ex: &mut FileExplorer, ctx: &Context) -> usize {
    let mut frames = 0;
    while !ex.scan_rx.is_empty() && frames < 10_000 {
        ex.receive_scan(ctx);
        frames += 1;
    }
    frames
}

fn path_of(i: usize) -> String {
    found(i).path.display().to_string()
}

fn pixel_free(rows: &[Thumbnail]) -> bool {
    rows.iter().all(|r| r.thumbnail_b64.is_none())
}

fn real_jpeg_b64() -> String {
    use base64::Engine as _;
    let img: image::RgbImage = image::ImageBuffer::from_pixel(32, 24, image::Rgb([30, 160, 90]));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Jpeg).unwrap();
    base64::engine::general_purpose::STANDARD.encode(out)
}

async fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if cond() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread")]
async fn thumb_ready_queues_rows_for_persistence_and_rows_carry_no_pixels() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 40);
    ex.receive_scan(&ctx);
    assert_eq!(ex.table.len(), 40);
    assert_eq!(ex.last_scan_rows.len(), 40);
    for i in [0usize, 7, 39] {
        send_ready(&ex, i);
    }
    ex.receive_scan(&ctx);
    let queued: Vec<&str> = ex.pending_thumb_rows.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(queued, [path_of(0), path_of(7), path_of(39)]);
    assert!(pixel_free(ex.table.rows()) && pixel_free(&ex.last_scan_rows) && pixel_free(&ex.pending_thumb_rows));
    ex.table.check_invariants().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn off_page_thumb_ready_leaves_the_table_alone() {
    let ctx = Context::default();
    let mut ex = explorer(true, 10);
    send_batch(&ex, 35);
    ex.receive_scan(&ctx);
    assert_eq!(ex.last_scan_rows.len(), 35);
    assert!(ex.table.len() <= 10);
    let before = ex.table.len();
    send_ready(&ex, 30);
    ex.receive_scan(&ctx);
    assert_eq!(ex.table.len(), before);
    assert!(ex.pending_thumb_rows.is_empty());
    assert!(ex.table.iter().all(|r| r.path != path_of(30)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_thumb_for_a_path_missing_from_a_shallow_table_adds_a_placeholder_row() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_ready(&ex, 5);
    ex.receive_scan(&ctx);
    assert_eq!(ex.table.len(), 1);
    assert_eq!(ex.table.iter().next().unwrap().path, path_of(5));
    assert_eq!(ex.pending_thumb_rows.len(), 1);
    assert!(pixel_free(ex.table.rows()));
}

#[tokio::test(flavor = "multi_thread")]
async fn lookups_stay_exact_after_rows_are_removed_from_the_table() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 30);
    ex.receive_scan(&ctx);
    let removed: std::collections::HashSet<String> = (0..10).map(path_of).collect();
    ex.table.retain(|r| !removed.contains(&r.path));
    ex.table.check_invariants().unwrap();
    send_ready(&ex, 20);
    send_ready(&ex, 3);
    ex.receive_scan(&ctx);
    let queued: Vec<&str> = ex.pending_thumb_rows.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(queued.first().copied(), Some(path_of(20).as_str()));
    assert_eq!(ex.table.len(), 21, "a removed row that gets a thumbnail is re-added as a placeholder");
    ex.table.check_invariants().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn scan_messages_are_drained_within_the_frame_budget() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 3_000);
    ex.receive_scan(&ctx);
    for i in 0..3_000 {
        send_ready(&ex, i);
    }
    ex.receive_scan(&ctx);
    assert!(!ex.scan_rx.is_empty(), "one frame must not drain 3000 messages");
    let frames = drain_all(&mut ex, &ctx);
    assert!(frames >= 2);
    assert!(ex.scan_rx.is_empty());
    assert_eq!(ex.table.len(), 3_000);
    assert!(pixel_free(ex.table.rows()) && pixel_free(&ex.last_scan_rows));
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_thumbnails_are_recorded_with_their_error() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 3);
    send(&ex, ScanMsg::ThumbFailed { path: found(1).path, error: "raw: rawkit panicked".into() });
    ex.receive_scan(&ctx);
    assert_eq!(ex.failed_thumbs.len(), 1);
    assert_eq!(ex.failed_thumbs[0].1, "raw: rawkit panicked");
}

#[tokio::test(flavor = "multi_thread")]
async fn thumbnail_channel_is_drained_with_a_budget_and_pixels_leave_the_rows() {
    let ctx = Context::default();
    let mut ex = explorer(false, 5_000);
    send_batch(&ex, 300);
    ex.receive_scan(&ctx);
    for i in 0..300 {
        let mut t = Thumbnail::new("x");
        t.path = path_of(i);
        t.filename = format!("img_{i}.jpg");
        t.thumbnail_b64 = Some("not an image".into());
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
    assert!(pixel_free(ex.table.rows()));
    assert!(ex.current_thumb.thumbnail_b64.is_none(), "selection rows are stripped too");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_pixels_in_loaded_rows_are_moved_into_the_service() {
    let mut ex = explorer(false, 5_000);
    let service = ex.viewer.thumbs.clone();
    let rows: Vec<Thumbnail> = (0..3)
        .map(|i| {
            let mut t = Thumbnail::new("legacy");
            t.path = format!("/legacy/only_{}_{i}.jpg", std::process::id());
            t.thumbnail_b64 = Some(real_jpeg_b64());
            t
        })
        .collect();
    let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
    ex.set_rows(rows);
    assert!(pixel_free(&ex.table.iter().cloned().collect::<Vec<_>>()), "rows must not keep the text copy");
    assert!(wait_for(|| paths.iter().all(|p| service.has_thumb(p))).await, "legacy thumbnails become stored blobs");
    let key = crate::ui::file_table::thumbs::ThumbKey::from_path(&paths[0]);
    let mut stored = 0;
    for _ in 0..200 {
        stored = service.store().get_many(&[key]).await.unwrap().len();
        if stored == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert_eq!(stored, 1, "the converted blob reaches the store");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_receive_scan_thumb_ready() {
    let ctx = Context::default();
    for n in [5_000usize, 20_000, 100_000] {
        let mut ex = explorer(false, usize::MAX);
        send_batch(&ex, n);
        drain_all(&mut ex, &ctx);
        assert_eq!(ex.last_scan_rows.len(), n);
        for i in 0..n {
            send_ready(&ex, i);
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
    }
}
