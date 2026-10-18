//! Service behaviour against the in-memory store: batching, priorities, cancellation, negative cache, write-behind and texture lifetime.

use super::core::CoreConfig;
use super::key::{ThumbKey, Tier};
use super::service::{Producer, ServiceConfig, ThumbService, ThumbView};
use super::store::{Blob, BlobFormat, BlobStamp, MemStore, ThumbBlobStore};
use crate::utilities::thumbs::{EncodedThumb, ThumbFormat};
use egui::Context;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SCOPE: u32 = 7;

#[derive(Default)]
struct TestProducer {
    requests: Mutex<Vec<String>>,
    refuse: AtomicBool,
}

impl TestProducer {
    fn requested(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Producer for TestProducer {
    fn request(&self, path: &Path) -> bool {
        if self.refuse.load(Ordering::Relaxed) {
            return false;
        }
        self.requests.lock().unwrap().push(path.display().to_string());
        true
    }
}

fn jpeg_bytes(w: u32, h: u32, shade: u8) -> Vec<u8> {
    let img: image::RgbImage = image::ImageBuffer::from_fn(w, h, |x, y| image::Rgb([shade, (x % 251) as u8, (y % 241) as u8]));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Jpeg).unwrap();
    out
}

fn blob(shade: u8) -> Blob {
    Blob { format: BlobFormat::Jpeg, w: 256, h: 171, bytes: jpeg_bytes(256, 171, shade) }
}

fn encoded(shade: u8) -> EncodedThumb {
    EncodedThumb { bytes: jpeg_bytes(128, 96, shade), format: ThumbFormat::Jpeg, width: 128, height: 96 }
}

fn config() -> ServiceConfig {
    ServiceConfig {
        core: CoreConfig { claim_budget: 200, ..CoreConfig::default() },
        decode_threads: 2,
        write_retry_ms: 15,
        write_retries: 10,
        sweep_ms: 30,
        ..ServiceConfig::default()
    }
}

struct Fixture {
    svc: Arc<ThumbService>,
    store: Arc<MemStore>,
    producer: Arc<TestProducer>,
    ctx: Context,
}

fn fixture_with(cfg: ServiceConfig) -> Fixture {
    let store = MemStore::new();
    let producer = Arc::new(TestProducer::default());
    let svc = ThumbService::new(store.clone(), tokio::runtime::Handle::current(), cfg, producer.clone());
    Fixture { svc, store, producer, ctx: Context::default() }
}

fn fixture() -> Fixture {
    fixture_with(config())
}

impl Fixture {
    fn seed(&self, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                let path = format!("/seed/img_{i:04}.jpg");
                self.store.insert(&path, blob(i as u8));
                path
            })
            .collect()
    }

    fn view(&self, path: &str) -> ThumbView {
        self.svc.request(SCOPE, path, Tier::Thumb, 0.0)
    }

    /// Polls like a frame loop, running `frame` before each poll, until `done` or the timeout.
    async fn run(&self, mut frame: impl FnMut(&Fixture), mut done: impl FnMut(&Fixture) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            frame(self);
            self.svc.poll_now(&self.ctx);
            if done(self) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
        false
    }

    async fn until_ready(&self, paths: &[String]) -> bool {
        self.run(|f| paths.iter().for_each(|p| drop(f.view(p))), |f| paths.iter().all(|p| f.svc.texture_for(p, Tier::Thumb).is_ready())).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drawn_cell_is_fetched_decoded_and_served_from_the_cache() {
    let f = fixture();
    let paths = f.seed(1);
    assert!(f.until_ready(&paths).await);
    let ThumbView::Ready { size, .. } = f.view(&paths[0]) else { panic!("not ready") };
    assert_eq!((size.x as u32, size.y as u32), (256, 171));
    assert_eq!(f.store.get_batches(), vec![1], "exactly one store read for one drawn cell");
    for _ in 0..30 {
        f.svc.poll_now(&f.ctx);
        assert!(f.view(&paths[0]).is_ready());
    }
    assert_eq!(f.store.get_batches(), vec![1], "steady state costs no store traffic");
    let s = f.svc.snapshot();
    assert_eq!(s.core.wants_created, 1);
    assert!(s.cache.hits >= 30);
    assert_eq!(s.core.decode_ok, 1);
    assert_eq!(s.entries, 1);
    assert!(s.used > 0 && s.used <= s.budget);
}

#[tokio::test(flavor = "multi_thread")]
async fn undrawn_cells_cause_no_store_traffic() {
    let f = fixture();
    let paths = f.seed(5);
    assert!(f.until_ready(&paths[..1]).await);
    assert_eq!(f.store.get_batches().iter().sum::<usize>(), 1);
    assert!(!f.svc.texture_for(&paths[3], Tier::Thumb).is_ready());
    assert_eq!(f.svc.snapshot().core.wants_created, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_screenful_is_read_with_one_batched_query() {
    let f = fixture();
    let paths = f.seed(50);
    paths.iter().for_each(|p| drop(f.view(p)));
    f.svc.poll_now(&f.ctx);
    assert!(f.until_ready(&paths).await);
    assert_eq!(f.store.get_batches(), vec![50]);
    assert_eq!(f.svc.snapshot().core.decode_ok, 50);
}

#[tokio::test(flavor = "multi_thread")]
async fn large_screens_are_split_into_capped_batches() {
    let f = fixture();
    let paths = f.seed(150);
    paths.iter().for_each(|p| drop(f.view(p)));
    assert!(f.until_ready(&paths).await);
    let batches = f.store.get_batches();
    assert_eq!(batches.iter().sum::<usize>(), 150);
    assert!(batches.iter().all(|b| *b <= 64), "{batches:?}");
    assert!(batches.len() >= 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scope_change_drops_in_flight_work() {
    let f = fixture();
    let paths = f.seed(10);
    f.svc.gate().set_delay_ms(150);
    paths.iter().for_each(|p| drop(f.view(p)));
    f.svc.poll_now(&f.ctx);
    f.svc.new_scope(SCOPE);
    f.svc.gate().set_delay_ms(0);
    for _ in 0..60 {
        f.svc.poll_now(&f.ctx);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let s = f.svc.snapshot();
    assert_eq!(s.entries, 0, "nothing from the old scope reaches the cache");
    assert_eq!(s.core.cancelled_scope, 10);
    assert_eq!(s.claims_in_flight, 0);
    assert_eq!(s.fetch_inflight, 0);
    assert!(f.until_ready(&paths).await, "the new scope can ask again");
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_that_leave_the_viewport_are_not_decoded() {
    let f = fixture();
    let paths = f.seed(4);
    f.svc.gate().set_delay_ms(120);
    paths.iter().for_each(|p| drop(f.view(p)));
    f.svc.poll_now(&f.ctx);
    let keep = &paths[0];
    let ok = f.run(|f| drop(f.view(keep)), |f| f.svc.texture_for(keep, Tier::Thumb).is_ready()).await;
    assert!(ok);
    let s = f.svc.snapshot();
    assert!(s.core.cancelled_stale >= 3, "{s:?}");
    assert_eq!(s.core.decode_ok, 1);
    for p in &paths[1..] {
        assert!(!f.svc.texture_for(p, Tier::Thumb).is_ready());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_database_never_blocks_the_frame() {
    let f = fixture();
    let paths = f.seed(20);
    f.svc.gate().set_delay_ms(400);
    let mut worst = Duration::ZERO;
    for _ in 0..20 {
        let t = Instant::now();
        paths.iter().for_each(|p| drop(f.view(p)));
        f.svc.poll_now(&f.ctx);
        worst = worst.max(t.elapsed());
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(worst < Duration::from_millis(200), "worst frame {worst:?}");
    f.svc.gate().set_delay_ms(0);
    assert!(f.until_ready(&paths).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn store_errors_retry_with_backoff_and_recover() {
    let f = fixture();
    let paths = f.seed(2);
    f.svc.gate().set_failing(true);
    let failed = f.run(|f| paths.iter().for_each(|p| drop(f.view(p))), |f| f.svc.snapshot().core.fetch_errors >= 1).await;
    assert!(failed);
    assert!(!f.svc.texture_for(&paths[0], Tier::Thumb).is_ready());
    f.svc.gate().set_failing(false);
    assert!(f.until_ready(&paths).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_blob_is_negative_cached_for_files_that_cannot_be_generated() {
    let f = fixture();
    let virtual_path = "zip:///data/a.zip!/inner.jpg".to_string();
    let missing = f.run(|f| drop(f.view(&virtual_path)), |f| f.view(&virtual_path) == ThumbView::Missing).await;
    assert!(missing);
    let batches_after_miss = f.store.get_batches().len();
    for _ in 0..40 {
        f.svc.poll_now(&f.ctx);
        assert_eq!(f.view(&virtual_path), ThumbView::Missing);
    }
    assert_eq!(f.store.get_batches().len(), batches_after_miss, "repeat frames must not query the store");
    assert!(f.producer.requested().is_empty(), "virtual paths are never generated from disk");
    assert!(f.svc.snapshot().core.negative_hits >= 40);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_miss_on_a_real_path_goes_to_the_producer_and_the_result_is_drawn_and_stored() {
    let f = fixture();
    let path = "/photos/new_shot.jpg".to_string();
    let asked = f.run(|f| drop(f.view(&path)), |f| !f.producer.requested().is_empty()).await;
    assert!(asked);
    assert_eq!(f.producer.requested(), vec![path.clone()], "one request, not one per frame");
    for _ in 0..10 {
        f.svc.poll_now(&f.ctx);
        drop(f.view(&path));
    }
    assert_eq!(f.producer.requested().len(), 1);
    let key = f.svc.sink().accept(Path::new(&path), encoded(90));
    assert_eq!(key, ThumbKey::from_path(&path).hex());
    assert!(f.until_ready(std::slice::from_ref(&path)).await);
    let stored = f.run(|_| {}, |f| f.store.contains(&path)).await;
    assert!(stored, "the write-behind queue reaches the store");
    assert!(f.svc.has_thumb(&path));
    assert_eq!(f.svc.snapshot().generate_inflight, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_generation_request_is_retried() {
    let f = fixture();
    f.producer.refuse.store(true, Ordering::Relaxed);
    let path = "/photos/busy_queue.jpg".to_string();
    let rejected = f.run(|f| drop(f.view(&path)), |f| f.svc.snapshot().core.generate_started >= 2).await;
    assert!(rejected, "a full producer queue is retried on later frames");
    f.producer.refuse.store(false, Ordering::Relaxed);
    assert!(f.run(|f| drop(f.view(&path)), |f| !f.producer.requested().is_empty()).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn producer_failures_become_a_visible_missing_state() {
    let f = fixture();
    let path = "/photos/corrupt.jpg".to_string();
    assert!(f.run(|f| drop(f.view(&path)), |f| !f.producer.requested().is_empty()).await);
    f.svc.sink().failed(Path::new(&path), "decode error");
    assert!(f.run(|f| drop(f.view(&path)), |f| f.view(&path) == ThumbView::Missing).await);
    assert_eq!(f.producer.requested().len(), 1, "a failed file is not retried every frame");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_produced_thumbnail_is_readable_before_its_database_write_lands() {
    let f = fixture();
    f.svc.gate().set_delay_ms(500);
    let path = "/photos/fresh.jpg".to_string();
    f.svc.sink().accept(Path::new(&path), encoded(10));
    let t = Instant::now();
    assert!(f.until_ready(std::slice::from_ref(&path)).await);
    assert!(t.elapsed() < Duration::from_millis(400), "served from the encoded cache, not the slow store");
    assert!(!f.store.contains(&path), "the write is still queued");
    assert_eq!(f.svc.snapshot().pending_writes, 1);
    f.svc.gate().set_delay_ms(0);
    assert!(f.run(|_| {}, |f| f.store.contains(&path)).await);
    assert!(f.run(|_| {}, |f| f.svc.snapshot().pending_writes == 0).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn writes_are_batched() {
    let f = fixture();
    let sink = f.svc.sink();
    let sink2 = sink.clone();
    let producer = std::thread::spawn(move || {
        for i in 0..150 {
            sink2.accept(Path::new(&format!("/bulk/{i}.jpg")), encoded((i % 200) as u8));
        }
    });
    producer.join().unwrap();
    assert!(f.run(|_| {}, |f| f.store.len() == 150).await);
    let batches = f.store.put_batches();
    assert_eq!(batches.iter().sum::<usize>(), 150);
    assert!(batches.iter().all(|b| *b <= 64), "{batches:?}");
    assert!(batches.len() < 150 / 4, "puts are grouped: {batches:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_writes_are_retried_then_succeed() {
    let f = fixture();
    f.svc.gate().set_failing(true);
    f.svc.sink().accept(Path::new("/retry/a.jpg"), encoded(1));
    assert!(f.run(|_| {}, |f| f.svc.snapshot().counters.write_errors >= 1).await);
    f.svc.gate().set_failing(false);
    assert!(f.run(|_| {}, |f| f.store.contains("/retry/a.jpg")).await);
    assert_eq!(f.svc.snapshot().counters.write_deferred, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn writes_that_fail_for_a_long_time_are_swept_up_later() {
    let mut cfg = config();
    cfg.write_retries = 1;
    let f = fixture_with(cfg);
    f.svc.gate().set_failing(true);
    let sink = f.svc.sink();
    for i in 0..5 {
        sink.accept(Path::new(&format!("/outage/{i}.jpg")), encoded(i));
    }
    assert!(f.run(|_| {}, |f| f.svc.snapshot().counters.write_deferred >= 5).await);
    let mid = f.svc.snapshot();
    assert_eq!(f.store.len(), 0);
    assert!(mid.recent_pinned >= 5, "unwritten thumbnails stay readable");
    assert!(f.until_ready(&["/outage/0.jpg".to_string()]).await, "and drawable while the store is down");
    f.svc.gate().set_failing(false);
    assert!(f.run(|_| {}, |f| f.store.len() == 5).await, "the sweeper writes them once the store recovers");
    assert!(f.run(|_| {}, |f| f.svc.snapshot().recent_unqueued == 0 && f.svc.snapshot().pending_writes == 0).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_full_write_queue_never_blocks_the_producer() {
    let f = fixture();
    f.svc.gate().set_delay_ms(300);
    let sink = f.svc.sink();
    let started = Instant::now();
    let worker = std::thread::spawn(move || {
        for i in 0..3000 {
            sink.accept(Path::new(&format!("/flood/{i}.jpg")), encoded((i % 200) as u8));
        }
    });
    worker.join().unwrap();
    assert!(started.elapsed() < Duration::from_secs(5), "producing must not wait for the store");
    assert!(f.svc.snapshot().counters.write_overflow > 0, "{:?}", f.svc.snapshot().counters);
    f.svc.gate().set_delay_ms(0);
    assert!(f.run(|_| {}, |f| f.store.len() == 3000).await, "overflowed writes are swept up");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_text_thumbnails_are_converted_once() {
    use base64::Engine as _;
    let f = fixture();
    let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg_bytes(64, 48, 40));
    let path = "/old/legacy_row.jpg".to_string();
    f.svc.ingest_legacy(&path, b64.clone());
    assert!(f.until_ready(std::slice::from_ref(&path)).await);
    assert!(f.run(|_| {}, |f| f.store.contains(&path)).await);
    f.svc.ingest_legacy(&path, b64);
    assert!(f.run(|_| {}, |f| f.svc.snapshot().counters.legacy_skipped >= 1).await);
    let s = f.svc.snapshot();
    assert_eq!((s.counters.legacy_ingested, s.counters.legacy_failed), (1, 0));
    assert!(f.producer.requested().is_empty(), "converted rows never fall back to generation");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_row_requested_while_its_legacy_payload_converts_does_not_trigger_generation() {
    use base64::Engine as _;
    let f = fixture();
    let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg_bytes(64, 48, 40));
    let path = "/old/racing_row.jpg".to_string();
    drop(f.view(&path));
    f.svc.ingest_legacy(&path, b64);
    assert!(f.until_ready(std::slice::from_ref(&path)).await);
    assert!(f.producer.requested().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_legacy_payloads_are_counted_and_do_not_poison_the_row() {
    let f = fixture();
    f.svc.ingest_legacy("/old/bad.jpg", "%%% not base64 %%%".into());
    assert!(f.run(|_| {}, |f| f.svc.snapshot().counters.legacy_failed == 1).await);
    assert!(!f.store.contains("/old/bad.jpg"));
}

#[tokio::test(flavor = "multi_thread")]
async fn presence_is_learned_from_the_store_for_loaded_pages() {
    let f = fixture();
    let paths = f.seed(30);
    f.svc.note_rows(paths.iter().cloned().chain(["/never/stored.jpg".to_string()]).collect());
    assert!(f.run(|_| {}, |f| f.svc.presence_len() == 30).await);
    assert!(f.svc.has_thumb(&paths[7]));
    assert!(!f.svc.has_thumb("/never/stored.jpg"));
    assert_eq!(f.store.exists_batches().iter().sum::<usize>(), 31);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_texture_budget_is_enforced_with_real_uploads() {
    let mut cfg = config();
    cfg.core.budget_bytes = 1_000_000;
    let f = fixture_with(cfg);
    let paths = f.seed(24);
    for chunk in paths.chunks(4) {
        assert!(f.until_ready(chunk).await);
        for _ in 0..3 {
            f.svc.poll_now(&f.ctx);
        }
    }
    let s = f.svc.snapshot();
    assert!(s.used <= 1_000_000, "used {} of {}", s.used, s.budget);
    assert!(s.cache.evictions >= 12, "{:?}", s.cache);
    assert!(s.entries <= 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_regenerated_thumbnail_updates_the_texture_in_place() {
    let f = fixture();
    let paths = f.seed(1);
    assert!(f.until_ready(&paths).await);
    let ThumbView::Ready { id: before, .. } = f.view(&paths[0]) else { panic!() };
    f.svc.sink().accept(Path::new(&paths[0]), encoded(200));
    let swapped = f
        .run(|f| drop(f.view(&paths[0])), |f| matches!(f.view(&paths[0]), ThumbView::Ready { size, .. } if size.x as u32 == 128))
        .await;
    assert!(swapped, "the new pixels arrive");
    let ThumbView::Ready { id: after, .. } = f.view(&paths[0]) else { panic!() };
    assert_eq!(before, after, "the same texture id is reused");
    assert!(f.svc.snapshot().cache.in_place_updates >= 1);
    assert_eq!(f.svc.snapshot().entries, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn flush_and_force_decode_rebuild_textures_from_the_store() {
    let f = fixture();
    let paths = f.seed(6);
    assert!(f.until_ready(&paths).await);
    f.svc.poll_now(&f.ctx);
    assert_eq!(f.svc.force_decode(3), 3);
    assert_eq!(f.svc.snapshot().entries, 3);
    assert!(f.until_ready(&paths).await);
    assert!(f.svc.snapshot().core.decode_ok >= 9);
    f.svc.flush();
    assert_eq!(f.svc.snapshot().entries, 0);
    assert_eq!(f.svc.snapshot().used, 0);
    assert!(f.until_ready(&paths).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_preview_tier_decodes_from_the_file_and_failures_are_remembered() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.png");
    image::RgbImage::from_pixel(2000, 1000, image::Rgb([9, 9, 9])).save(&path).unwrap();
    let p = path.display().to_string();
    let ok = f.run(|f| drop(f.svc.request(SCOPE, &p, Tier::Preview, 0.0)), |f| f.svc.texture_for(&p, Tier::Preview).is_ready()).await;
    assert!(ok);
    let ThumbView::Ready { size, .. } = f.svc.texture_for(&p, Tier::Preview) else { panic!() };
    assert_eq!((size.x as u32, size.y as u32), (1600, 800));
    assert!(f.store.get_batches().is_empty(), "previews never touch the blob store");
    let missing = dir.path().join("gone.png").display().to_string();
    let failed = f.run(|f| drop(f.svc.request(SCOPE, &missing, Tier::Preview, 0.0)), |f| f.svc.request(SCOPE, &missing, Tier::Preview, 0.0) == ThumbView::Missing).await;
    assert!(failed);
    assert_eq!(f.svc.snapshot().core.decode_failed, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn poll_runs_once_per_egui_pass() {
    let f = fixture();
    let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0))), ..Default::default() };
    f.ctx
        .run_ui(input.clone(), |ui| {
            f.svc.poll(ui.ctx());
            f.svc.poll(ui.ctx());
            f.svc.poll(ui.ctx());
        })
        .drop_without_applying_deltas();
    assert_eq!(f.svc.snapshot().timings.frame.samples, 1);
    f.ctx
        .run_ui(input, |ui| {
            f.svc.poll(ui.ctx());
        })
        .drop_without_applying_deltas();
    assert_eq!(f.svc.snapshot().timings.frame.samples, 2);
}

#[test]
fn sink_freshness_follows_the_source_file_stamp() {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let store = MemStore::new();
    let svc = ThumbService::new(store.clone(), rt.handle().clone(), config(), Arc::new(TestProducer::default()));
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("photo.jpg");
    std::fs::write(&file, b"original bytes").unwrap();
    let meta = std::fs::metadata(&file).unwrap();
    let secs = meta.modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let path = file.display().to_string();
    let sink = svc.sink();
    assert_eq!(sink.fresh_key(&file), None, "nothing stored yet");
    store.insert_stamped(&path, blob(1), BlobStamp { src_modified: None, src_size: None });
    assert_eq!(sink.fresh_key(&file), None, "a blob without a stamp is regenerated");
    store.insert_stamped(&path, blob(1), BlobStamp { src_modified: Some(secs), src_size: Some(meta.len()) });
    assert_eq!(sink.fresh_key(&file), Some(ThumbKey::from_path(&path).hex()));
    assert_eq!(svc.snapshot().counters.fresh_hits, 1);
    std::fs::write(&file, b"changed and longer bytes").unwrap();
    assert_eq!(sink.fresh_key(&file), None, "a modified file is regenerated");
    let accepted = sink.accept(&file, encoded(5));
    assert_eq!(accepted, ThumbKey::from_path(&path).hex());
    assert_eq!(sink.fresh_key(&file), Some(accepted), "a just-produced thumbnail is fresh before its write lands");
    drop(rt);
}

#[tokio::test(flavor = "multi_thread")]
async fn store_stats_are_read_on_request() {
    let f = fixture();
    f.seed(3);
    f.svc.request_store_stats();
    assert!(f.run(|_| {}, |f| f.svc.snapshot().store_stats.is_some()).await);
    assert_eq!(f.svc.snapshot().store_stats.unwrap().count, 3);
    let _ = f.store.stats().await.unwrap();
}
