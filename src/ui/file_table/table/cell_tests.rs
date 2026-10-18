//! Headless egui frames through the real table renderer: only cells that are drawn ask the thumbnail service for a texture.

use super::FileTableViewer;
use crate::Thumbnail;
use crate::ui::file_table::thumbs::store::{Blob, BlobFormat, MemStore};
use crate::ui::file_table::thumbs::service::{Producer, ServiceConfig, ThumbService};
use crate::ui::file_table::thumbs::{ThumbPresence, Tier};
use egui::{CentralPanel, Context, Pos2, RawInput, Rect, Widget, vec2};
use egui_data_table::{DataTable, Renderer};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

struct NoProducer;

impl Producer for NoProducer {
    fn request(&self, _path: &Path) -> bool {
        true
    }
}

fn jpeg() -> Blob {
    let img: image::RgbImage = image::ImageBuffer::from_fn(256, 171, |x, y| image::Rgb([(x * 7 + y * 3) as u8, (x ^ y) as u8, ((x * y) >> 3) as u8]));
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg).unwrap();
    Blob { format: BlobFormat::Jpeg, w: 256, h: 171, bytes }
}

fn path_of(i: usize) -> String {
    format!("/cells/img_{i:04}.jpg")
}

struct Rig {
    ctx: Context,
    svc: Arc<ThumbService>,
    store: Arc<MemStore>,
    table: DataTable<Thumbnail>,
    viewer: FileTableViewer,
}

fn rig(rows: usize) -> Rig {
    let store = MemStore::new();
    let svc = ThumbService::new(store.clone(), tokio::runtime::Handle::current(), ServiceConfig::default(), Arc::new(NoProducer));
    let mut table: DataTable<Thumbnail> = Default::default();
    let proto = jpeg();
    for i in 0..rows {
        let mut t = Thumbnail::new(&format!("img_{i:04}.jpg"));
        t.path = path_of(i);
        t.file_type = "jpg".into();
        store.insert(&t.path, proto.clone());
        table.push(t);
    }
    let (thumb_tx, _thumb_rx) = crossbeam::channel::unbounded();
    let (ai_tx, _ai_rx) = crossbeam::channel::unbounded();
    let (clip_tx, _clip_rx) = crossbeam::channel::unbounded();
    let mut viewer = FileTableViewer::new(thumb_tx, ai_tx, clip_tx);
    viewer.thumbs = svc.clone();
    viewer.thumb_cache = ThumbPresence::new(svc.clone());
    Rig { ctx: Context::default(), svc, store, table, viewer }
}

impl Rig {
    fn frame(&mut self) {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 600.0))), ..Default::default() };
        let (table, viewer, svc) = (&mut self.table, &mut self.viewer, self.svc.clone());
        self.ctx
            .run_ui(input, |ui| {
                svc.poll(ui.ctx());
                CentralPanel::default().show(ui, |ui| {
                    Renderer::new(table, viewer)
                        .with_style_modify(|s| {
                            s.table_row_height = Some(75.0);
                            s.auto_shrink = [false, false].into();
                        })
                        .ui(ui);
                });
            })
            .drop_without_applying_deltas();
    }

    fn tracked_rows(&self, rows: usize) -> Vec<usize> {
        (0..rows).filter(|i| self.svc.is_tracked(&path_of(*i), Tier::Thumb)).collect()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn only_drawn_cells_ask_for_a_thumbnail_and_each_asks_once() {
    let mut rig = rig(400);
    for _ in 0..40 {
        rig.frame();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let tracked = rig.tracked_rows(400);
    assert!(!tracked.is_empty(), "visible cells must request their thumbnails");
    assert!(tracked.len() <= 16, "a 600 px viewport shows about eight 75 px rows, got {}", tracked.len());
    assert_eq!(tracked, (0..tracked.len()).collect::<Vec<_>>(), "the visible window is the top of the table, nothing below it");
    let snap = rig.svc.snapshot();
    assert_eq!(snap.core.wants_created as usize, tracked.len(), "one request per drawn cell, none for undrawn rows");
    assert_eq!(rig.store.get_batches().iter().sum::<usize>(), tracked.len(), "the store was asked for exactly the drawn rows, once");
    assert_eq!(snap.core.decode_ok as usize, tracked.len());
    assert!(snap.core.requests as usize > tracked.len(), "cells re-ask every frame; only the first ask costs anything");
}

#[tokio::test(flavor = "multi_thread")]
async fn drawn_cells_end_up_with_textures_and_steady_frames_cost_nothing() {
    let mut rig = rig(50);
    for _ in 0..40 {
        rig.frame();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let tracked = rig.tracked_rows(50);
    assert!(tracked.iter().all(|i| rig.svc.texture_for(&path_of(*i), Tier::Thumb).is_ready()));
    let before = rig.svc.snapshot();
    for _ in 0..20 {
        rig.frame();
    }
    let after = rig.svc.snapshot();
    assert_eq!(after.core.wants_created, before.core.wants_created);
    assert_eq!(after.core.fetch_batches, before.core.fetch_batches);
    assert_eq!(after.core.decode_started, before.core.decode_started);
    assert_eq!(rig.store.get_batches().iter().sum::<usize>(), tracked.len());
    assert!(after.cache.hits >= before.cache.hits + 20 * tracked.len() as u64, "every steady frame is served from the cache");
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_without_thumbnail_data_show_a_missing_state_not_a_request_loop() {
    let mut rig = rig(3);
    for i in 0..3 {
        rig.store.insert(&path_of(i), jpeg());
    }
    let mut missing = Thumbnail::new("notes.txt");
    missing.path = "/cells/notes.txt".into();
    missing.file_type = "txt".into();
    rig.table.push(missing);
    for _ in 0..40 {
        rig.frame();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let snap = rig.svc.snapshot();
    assert!(snap.core.negative_hits > 0);
    assert_eq!(snap.core.fetch_missing, 1);
    let batches_before = rig.store.get_batches().len();
    for _ in 0..30 {
        rig.frame();
    }
    assert_eq!(rig.store.get_batches().len(), batches_before, "a known-missing thumbnail is not looked up again every frame");
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<f64>().ok())).map(|pages| pages * 4.0 / 1024.0).unwrap_or(0.0)
}

/// Prints frame-time and allocation figures for scrolling a large table; run with `--ignored --nocapture`, row count from `AME_BENCH_ROWS`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_table_body_scrolling() {
    use crate::ui::file_table::thumbs::perf::measure;
    use egui::{Event, Modifiers, MouseWheelSource, MouseWheelUnit, TouchPhase, pos2};
    let rows: usize = std::env::var("AME_BENCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let scroll = std::env::var("AME_BENCH_SCROLL").map(|v| v != "0").unwrap_or(true);
    let frames: usize = std::env::var("AME_BENCH_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let mut rig = rig(rows);
    let mut times = Vec::new();
    let mut allocs = Vec::new();
    let mut bytes = Vec::new();
    let mut polls = Vec::new();
    let poll_us = std::cell::Cell::new(0.0f64);
    let rss0 = rss_mb();
    for frame in 0..(frames + 30) {
        let mut events = vec![Event::PointerMoved(pos2(500.0, 300.0))];
        if scroll && frame >= 30 {
            events.push(Event::MouseWheel { unit: MouseWheelUnit::Point, delta: vec2(0.0, -600.0), phase: TouchPhase::Move, source: MouseWheelSource::Wheel, modifiers: Modifiers::NONE });
        }
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 600.0))), events, ..Default::default() };
        let (table, viewer, svc, ctx) = (&mut rig.table, &mut rig.viewer, rig.svc.clone(), &rig.ctx);
        let started = std::time::Instant::now();
        let (_, counts) = measure(|| {
            let mut out = ctx.run_ui(input, |ui| {
                let poll_started = std::time::Instant::now();
                svc.poll(ui.ctx());
                poll_us.set(poll_started.elapsed().as_secs_f64() * 1e6);
                CentralPanel::default().show(ui, |ui| {
                    Renderer::new(table, viewer)
                        .with_style_modify(|s| {
                            s.table_row_height = Some(75.0);
                            s.auto_shrink = [false, false].into();
                        })
                        .ui(ui);
                });
            });
            out.textures_delta.clear();
            let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
        });
        if frame >= 30 {
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            allocs.push(counts.allocs as f64);
            bytes.push(counts.bytes as f64);
            polls.push(poll_us.get());
        }
        tokio::time::sleep(Duration::from_millis(16)).await;
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let snap = rig.svc.snapshot();
    println!(
        "BENCH rows={rows} frames={frames} scroll={scroll}: frame ms p50 {:.2} p95 {:.2} max {:.2} | ui-thread allocs/frame mean {:.0} p95 {:.0} | bytes/frame mean {:.0} p95 {:.0} | cache {} entries {} hits {} fetches {} reuse {:.1}% evict {} | poll us mean {:.0} p95 {:.0} | rss {rss0:.0} -> {:.0} MB",
        pct(&mut times.clone(), 0.5),
        pct(&mut times.clone(), 0.95),
        pct(&mut times, 1.0),
        mean(&allocs),
        pct(&mut allocs.clone(), 0.95),
        mean(&bytes),
        pct(&mut bytes.clone(), 0.95),
        snap.used,
        snap.entries,
        snap.cache.hits,
        snap.cache.claims_granted,
        snap.cache.reuse_rate() * 100.0,
        snap.cache.evictions,
        mean(&polls),
        pct(&mut polls.clone(), 0.95),
        rss_mb(),
    );
}
