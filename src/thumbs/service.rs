//! Thumbnail service: the one entry point the UI uses to get textures. The pure policy lives in `core`; this file runs it with a store, a decode pool and a write-behind queue.

use super::cache::CacheStats;
use super::core::{Action, Core, CoreConfig, CoreStats, DecodeJob, DecodeResult, DecodeSrc, Depths, FetchOutcome, Lookup};
use super::decode::{self, Decoded};
use super::key::{CacheKey, ThumbKey, Tier};
use super::recent::{Presence, RECENT_HARD_BYTES, RECENT_SOFT_BYTES, Recent};
use super::stats::{Timings, TimingsSnapshot};
use super::store::{Blob, BlobFormat, BlobIn, BlobStamp, BlobStats, GatedStore, StoreGate, ThumbBlobStore};
use crate::utilities::thumbs::{EncodedThumb, ThumbFormat, ThumbSink};
use crate::utilities::types::{MediaKind, is_video};
use crossbeam::channel::{Receiver, Sender, bounded};
use egui::{ColorImage, Context, TextureHandle, TextureId, TextureOptions, Vec2};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::runtime::Handle;

const EVENT_QUEUE_CAP: usize = 4096;
const EVENTS_PER_POLL: usize = 4096;
const UPLOAD_TIME_BUDGET: Duration = Duration::from_millis(6);
const WRITE_QUEUE_CAP: usize = 2048;
const WRITE_GATHER: Duration = Duration::from_millis(25);
const PRESENCE_CHUNK: usize = 500;
const WAKE_DELAY: Duration = Duration::from_millis(8);
const IDLE_GAP_SECS: f32 = 0.25;

pub type ScopeId = u32;

static NEXT_SCOPE: AtomicU32 = AtomicU32::new(1);

/// Allocates an id for one independent set of requests, such as one explorer tab.
pub fn alloc_scope() -> ScopeId {
    NEXT_SCOPE.fetch_add(1, Ordering::Relaxed)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThumbView {
    Ready { id: TextureId, size: Vec2 },
    Pending,
    Missing,
}

impl ThumbView {
    pub fn is_ready(&self) -> bool {
        matches!(self, ThumbView::Ready { .. })
    }
}

/// Where standalone thumbnail generation requests go.
pub trait Producer: Send + Sync {
    fn request(&self, path: &Path) -> bool;
}

/// Producer backed by the scan thumbnail worker pool.
pub struct PoolProducer;

impl Producer for PoolProducer {
    fn request(&self, path: &Path) -> bool {
        let is_vid = path.extension().and_then(|e| e.to_str()).is_some_and(|e| is_video(&e.to_ascii_lowercase()));
        crate::utilities::scan::request_priority_thumb(path, if is_vid { MediaKind::Video } else { MediaKind::Image })
    }
}

#[derive(Clone)]
pub struct ServiceConfig {
    pub core: CoreConfig,
    pub upload_per_frame: usize,
    pub decode_threads: usize,
    pub recent_soft_bytes: usize,
    pub recent_hard_bytes: usize,
    pub write_batch: usize,
    pub write_retries: u32,
    pub write_retry_ms: u64,
    pub sweep_ms: u64,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        Self {
            core: CoreConfig::default(),
            upload_per_frame: 8,
            decode_threads: (cores / 4).clamp(2, 6),
            recent_soft_bytes: RECENT_SOFT_BYTES,
            recent_hard_bytes: RECENT_HARD_BYTES,
            write_batch: 64,
            write_retries: 3,
            write_retry_ms: 500,
            sweep_ms: 1500,
        }
    }
}

impl ServiceConfig {
    /// Defaults with the texture budget overridable through `AME_THUMB_CACHE_MB`.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(mb) = std::env::var("AME_THUMB_CACHE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            cfg.core.budget_bytes = mb.max(1) * 1024 * 1024;
        }
        cfg
    }
}

enum Event {
    Fetched { batch: u64, outcome: FetchOutcome },
    Decoded { key: CacheKey, result: Result<Decoded, String> },
    DecodeCancelled { key: CacheKey },
    Produced { key: ThumbKey, blob: Arc<Blob> },
    ProducerFailed { key: ThumbKey, error: String },
    GenRejected { key: ThumbKey },
    LegacyDone { key: ThumbKey },
}

#[derive(Default)]
struct Counters {
    writes_ok: AtomicU64,
    write_errors: AtomicU64,
    write_deferred: AtomicU64,
    write_overflow: AtomicU64,
    produced: AtomicU64,
    fresh_hits: AtomicU64,
    fresh_misses: AtomicU64,
    legacy_ingested: AtomicU64,
    legacy_skipped: AtomicU64,
    legacy_failed: AtomicU64,
    events: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CountersSnapshot {
    pub writes_ok: u64,
    pub write_errors: u64,
    pub write_deferred: u64,
    pub write_overflow: u64,
    pub produced: u64,
    pub fresh_hits: u64,
    pub fresh_misses: u64,
    pub legacy_ingested: u64,
    pub legacy_skipped: u64,
    pub legacy_failed: u64,
    pub events: u64,
}

struct Shared {
    store: Arc<dyn ThumbBlobStore>,
    gate: Arc<StoreGate>,
    rt: Handle,
    events_tx: Sender<Event>,
    recent: Arc<Mutex<Recent>>,
    presence: Arc<Mutex<Presence>>,
    writer_tx: tokio::sync::mpsc::Sender<BlobIn>,
    pending_writes: Arc<AtomicUsize>,
    ctx: OnceLock<Context>,
    timings: Arc<Mutex<Timings>>,
    ctr: Arc<Counters>,
    pool: rayon::ThreadPool,
    epoch: Instant,
    store_stats: Mutex<Option<BlobStats>>,
}

impl Shared {
    fn emit(&self, ev: Event) {
        self.ctr.events.fetch_add(1, Ordering::Relaxed);
        match self.events_tx.try_send(ev) {
            Ok(()) => {}
            Err(crossbeam::channel::TrySendError::Full(ev)) => {
                let tx = self.events_tx.clone();
                if Handle::try_current().is_ok() {
                    self.rt.spawn_blocking(move || {
                        let _ = tx.send(ev);
                    });
                } else {
                    let _ = tx.send(ev);
                }
            }
            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {}
        }
        self.wake();
    }

    fn wake(&self) {
        if let Some(ctx) = self.ctx.get() {
            ctx.request_repaint_after(WAKE_DELAY);
        }
    }
}

struct Staged {
    key: CacheKey,
    image: ColorImage,
    cost: usize,
}

struct PollState {
    staged: VecDeque<Staged>,
    actions: Vec<Action>,
    last_pass: u64,
    polled: bool,
    perf: Option<PerfLog>,
}

struct PerfLog {
    path: std::path::PathBuf,
    started: Instant,
    last: Instant,
}

pub struct ThumbService {
    core: Mutex<Core<TextureHandle>>,
    state: Mutex<PollState>,
    shared: Arc<Shared>,
    events_rx: Receiver<Event>,
    producer: Arc<dyn Producer>,
    upload_per_frame: AtomicUsize,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub core: CoreStats,
    pub cache: CacheStats,
    pub depths: Depths,
    pub budget: usize,
    pub used: usize,
    pub entries: usize,
    pub on_screen: usize,
    pub claims_in_flight: usize,
    pub claim_budget: usize,
    pub fetch_inflight: usize,
    pub decode_inflight: usize,
    pub generate_inflight: usize,
    pub staged: usize,
    pub upload_per_frame: usize,
    pub recent_bytes: usize,
    pub recent_entries: usize,
    pub recent_pinned: usize,
    pub recent_unqueued: usize,
    pub recent_dropped: u64,
    pub presence: usize,
    pub pending_writes: usize,
    pub counters: CountersSnapshot,
    pub timings: TimingsSnapshot,
    pub store_delay_ms: u64,
    pub store_failing: bool,
    pub store_stats: Option<BlobStats>,
}

fn file_stamp(path: &Path) -> Option<BlobStamp> {
    let meta = std::fs::metadata(path).ok()?;
    let secs = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    Some(BlobStamp { src_modified: Some(secs), src_size: Some(meta.len()) })
}

fn texture_options(tier: Tier) -> TextureOptions {
    match tier {
        Tier::Thumb => TextureOptions::LINEAR.with_mipmap_mode(Some(egui::TextureFilter::Linear)),
        Tier::Preview => TextureOptions::LINEAR,
    }
}

fn blob_from_encoded(thumb: EncodedThumb) -> Blob {
    Blob {
        format: match thumb.format {
            ThumbFormat::Jpeg => BlobFormat::Jpeg,
            ThumbFormat::Png => BlobFormat::Png,
        },
        w: thumb.width,
        h: thumb.height,
        bytes: thumb.bytes,
    }
}

impl ThumbService {
    pub fn new(store: Arc<dyn ThumbBlobStore>, rt: Handle, cfg: ServiceConfig, producer: Arc<dyn Producer>) -> Arc<Self> {
        let gate = Arc::new(StoreGate::default());
        let gated: Arc<dyn ThumbBlobStore> = Arc::new(GatedStore::new(store, gate.clone()));
        let (events_tx, events_rx) = bounded(EVENT_QUEUE_CAP);
        let (writer_tx, writer_rx) = tokio::sync::mpsc::channel(WRITE_QUEUE_CAP);
        let recent = Arc::new(Mutex::new(Recent::new(cfg.recent_soft_bytes, cfg.recent_hard_bytes)));
        let timings = Arc::new(Mutex::new(Timings::default()));
        let ctr = Arc::new(Counters::default());
        let pending_writes = Arc::new(AtomicUsize::new(0));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.decode_threads.max(1))
            .thread_name(|i| format!("thumb-decode-{i}"))
            .panic_handler(|p| log::error!("[thumbs] decode task panicked: {}", crate::utilities::guard::panic_message(p.as_ref())))
            .build()
            .expect("thumb decode pool");
        rt.spawn(writer_loop(WriterCtx {
            rx: writer_rx,
            store: gated.clone(),
            recent: recent.clone(),
            timings: timings.clone(),
            ctr: ctr.clone(),
            pending: pending_writes.clone(),
            batch: cfg.write_batch.max(1),
            retries: cfg.write_retries,
            retry_ms: cfg.write_retry_ms,
            sweep: Duration::from_millis(cfg.sweep_ms.max(1)),
        }));
        let shared = Arc::new(Shared {
            store: gated,
            gate,
            rt,
            events_tx,
            recent,
            presence: Arc::new(Mutex::new(Presence::default())),
            writer_tx,
            pending_writes,
            ctx: OnceLock::new(),
            timings,
            ctr,
            pool,
            epoch: Instant::now(),
            store_stats: Mutex::new(None),
        });
        let perf = std::env::var_os("AME_THUMB_PERF_LOG").map(|p| PerfLog { path: p.into(), started: Instant::now(), last: Instant::now() });
        Arc::new(Self {
            core: Mutex::new(Core::new(cfg.core.clone())),
            state: Mutex::new(PollState { staged: VecDeque::new(), actions: Vec::new(), last_pass: 0, polled: false, perf }),
            shared,
            events_rx,
            producer,
            upload_per_frame: AtomicUsize::new(cfg.upload_per_frame.max(1)),
        })
    }

    /// Result for a cell that is being drawn this frame; registers interest when the texture is not cached yet.
    pub fn request(&self, scope: ScopeId, path: &str, tier: Tier, priority: f32) -> ThumbView {
        match lock(&self.core).request(scope, path, tier, priority) {
            Lookup::Ready(h) => ThumbView::Ready { id: h.id(), size: h.size_vec2() },
            Lookup::Pending => ThumbView::Pending,
            Lookup::Missing | Lookup::Failed => ThumbView::Missing,
        }
    }

    /// Registers interest in a cell that is about to scroll into view without marking a cached texture as drawn; returns whether it is cached.
    pub fn request_ahead(&self, scope: ScopeId, path: &str, tier: Tier, priority: f32) -> bool {
        lock(&self.core).request_ahead(scope, path, tier, priority)
    }

    /// Cached texture for `path` without registering interest.
    pub fn texture_for(&self, path: &str, tier: Tier) -> ThumbView {
        match lock(&self.core).peek(path, tier) {
            Some(h) => ThumbView::Ready { id: h.id(), size: h.size_vec2() },
            None => ThumbView::Pending,
        }
    }

    /// Whether `path` has a pending request or a cached texture at `tier`.
    pub fn is_tracked(&self, path: &str, tier: Tier) -> bool {
        let core = lock(&self.core);
        core.is_wanted(path, tier) || core.is_cached(path, tier)
    }

    /// Records how long building the explorer UI took in one frame.
    pub fn record_ui_time(&self, took: Duration) {
        lock(&self.shared.timings).ui.push_duration(took);
    }

    /// Drops the pending work of one scope, such as when its table is replaced.
    pub fn new_scope(&self, scope: ScopeId) {
        lock(&self.core).new_scope(scope);
    }

    /// Per-frame step: advances the clock, applies finished work, uploads textures within the budget and starts new work. Runs once per egui pass.
    pub fn poll(&self, ctx: &Context) {
        let pass = ctx.cumulative_pass_nr();
        {
            let mut st = lock(&self.state);
            if st.polled && st.last_pass == pass {
                return;
            }
            st.polled = true;
            st.last_pass = pass;
        }
        self.poll_now(ctx);
    }

    /// `poll` without the once-per-pass guard.
    pub fn poll_now(&self, ctx: &Context) {
        let _ = self.shared.ctx.get_or_init(|| ctx.clone());
        let now_ms = self.shared.epoch.elapsed().as_millis() as u64;
        let dt = ctx.input(|i| i.unstable_dt);
        if dt < IDLE_GAP_SECS {
            lock(&self.shared.timings).frame.push(dt * 1000.0);
        }
        let mut st = lock(&self.state);
        let mut core = lock(&self.core);
        core.begin_frame(now_ms);
        self.drain_events(&mut core, &mut st);
        self.upload_staged(ctx, &mut core, &mut st);
        let mut actions = std::mem::take(&mut st.actions);
        core.plan(&mut actions);
        let more_now = core.claims_denied_this_frame() > 0 || !st.staged.is_empty();
        let waiting = core.wants_len() > 0;
        let frame = core.frame();
        drop(core);
        self.dispatch(&mut actions);
        st.actions = actions;
        if more_now {
            ctx.request_repaint();
        } else if waiting {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        if st.perf.is_some() && frame % 30 == 0 {
            self.write_perf_log(&mut st);
        }
    }

    fn drain_events(&self, core: &mut Core<TextureHandle>, st: &mut PollState) {
        for _ in 0..EVENTS_PER_POLL {
            let Ok(ev) = self.events_rx.try_recv() else { break };
            match ev {
                Event::Fetched { batch, outcome } => core.on_fetch_done(batch, outcome),
                Event::Decoded { key, result } => match result {
                    Ok(d) => st.staged.push_back(Staged { key, image: d.image, cost: d.cost }),
                    Err(e) => {
                        log::debug!("[thumbs] decode failed for {key:?}: {e}");
                        core.complete_decode(key, DecodeResult::Failed, |_| None);
                    }
                },
                Event::DecodeCancelled { key } => {
                    core.complete_decode(key, DecodeResult::Cancelled, |_| None);
                }
                Event::Produced { key, blob } => {
                    core.on_blob_ready(key, blob);
                    core.on_generation_finished(key, None);
                }
                Event::ProducerFailed { key, error } => core.on_generation_finished(key, Some(&error)),
                Event::GenRejected { key } => core.on_generation_rejected(key),
                Event::LegacyDone { key } => core.legacy_finished(key),
            }
        }
    }

    fn upload_staged(&self, ctx: &Context, core: &mut Core<TextureHandle>, st: &mut PollState) {
        let budget = self.upload_per_frame.load(Ordering::Relaxed);
        let started = Instant::now();
        let mut uploaded = 0usize;
        while uploaded < budget && started.elapsed() < UPLOAD_TIME_BUDGET {
            let Some(staged) = st.staged.pop_front() else { break };
            let t0 = Instant::now();
            let options = texture_options(staged.key.tier);
            let mut image = Some(staged.image);
            let mut used = false;
            core.complete_decode(staged.key, DecodeResult::Image { cost: staged.cost }, |existing| {
                let img = image.take()?;
                used = true;
                match existing {
                    Some(handle) => {
                        handle.set(img, options);
                        None
                    }
                    None => Some(ctx.load_texture(format!("thumb:{}", staged.key.key.hex()), img, options)),
                }
            });
            if used {
                uploaded += 1;
                lock(&self.shared.timings).upload.push_duration(t0.elapsed());
            }
        }
    }

    fn dispatch(&self, actions: &mut Vec<Action>) {
        for action in actions.drain(..) {
            match action {
                Action::Fetch { batch, keys } => self.spawn_fetch(batch, keys),
                Action::Decode(job) => self.spawn_decode(job),
                Action::Generate { key, path } => {
                    if !self.producer.request(Path::new(&*path)) {
                        self.shared.emit(Event::GenRejected { key });
                    }
                }
            }
        }
    }

    fn spawn_fetch(&self, batch: u64, keys: Vec<ThumbKey>) {
        let sh = self.shared.clone();
        self.shared.rt.spawn(async move {
            let started = Instant::now();
            let mut found: Vec<(ThumbKey, Arc<Blob>)> = Vec::with_capacity(keys.len());
            let mut need: Vec<ThumbKey> = Vec::new();
            {
                let recent = lock(&sh.recent);
                for k in &keys {
                    match recent.get(k) {
                        Some(b) => found.push((*k, b)),
                        None => need.push(*k),
                    }
                }
            }
            let outcome: FetchOutcome = if need.is_empty() {
                Ok((found, Vec::new()))
            } else {
                match sh.store.get_many(&need).await {
                    Ok(mut rows) => {
                        let mut missing = Vec::new();
                        let mut fetched: Vec<(ThumbKey, Arc<Blob>)> = Vec::new();
                        for k in need {
                            match rows.remove(&k) {
                                Some(b) => fetched.push((k, Arc::new(b))),
                                None => missing.push(k),
                            }
                        }
                        {
                            let mut recent = lock(&sh.recent);
                            let mut presence = lock(&sh.presence);
                            for (k, b) in &fetched {
                                recent.insert(*k, b.clone(), BlobStamp::default(), false);
                                presence.insert(*k);
                            }
                        }
                        found.extend(fetched);
                        Ok((found, missing))
                    }
                    Err(e) => Err(format!("{e:#}")),
                }
            };
            lock(&sh.timings).db_batch.push_duration(started.elapsed());
            sh.emit(Event::Fetched { batch, outcome });
        });
    }

    fn spawn_decode(&self, job: DecodeJob) {
        let sh = self.shared.clone();
        self.shared.pool.spawn(move || {
            if job.cancel.load(Ordering::Relaxed) {
                sh.emit(Event::DecodeCancelled { key: job.key });
                return;
            }
            let started = Instant::now();
            let result = match &job.src {
                DecodeSrc::Blob(blob) => decode::decode_blob(blob),
                DecodeSrc::File => decode::decode_preview(Path::new(&*job.path), Tier::PREVIEW_MAX_DIM),
            };
            lock(&sh.timings).decode.push_duration(started.elapsed());
            sh.emit(Event::Decoded { key: job.key, result });
        });
    }

    /// Hands a legacy text thumbnail to the service: it is converted once into a stored blob and then served like any other.
    pub fn ingest_legacy(&self, path: &str, thumb: String) {
        self.ingest_legacy_batch(vec![(path.to_string(), thumb)]);
    }

    /// Batch form of [`Self::ingest_legacy`]: one existence probe for the whole batch, conversions on the decode pool.
    pub fn ingest_legacy_batch(&self, items: Vec<(String, String)>) {
        let items: Vec<(ThumbKey, String, String)> = items.into_iter().filter(|(_, t)| !t.is_empty()).map(|(p, t)| (ThumbKey::from_path(&p), p, t)).collect();
        if items.is_empty() {
            return;
        }
        {
            let mut core = lock(&self.core);
            for (key, _, _) in &items {
                core.legacy_started(*key);
            }
        }
        let sh = self.shared.clone();
        self.shared.rt.spawn(async move {
            let known: Vec<bool> = {
                let presence = lock(&sh.presence);
                let recent = lock(&sh.recent);
                items.iter().map(|(k, _, _)| presence.contains(k) || recent.contains(k)).collect()
            };
            let unknown: Vec<ThumbKey> = items.iter().zip(&known).filter(|(_, k)| !**k).map(|((key, _, _), _)| *key).collect();
            let mut stored = std::collections::HashSet::new();
            for chunk in unknown.chunks(PRESENCE_CHUNK) {
                match sh.store.exists_many(chunk).await {
                    Ok(found) => stored.extend(found),
                    Err(e) => log::debug!("[thumbs] legacy probe failed: {e:#}"),
                }
            }
            for ((key, path, thumb), was_known) in items.into_iter().zip(known) {
                let skip = was_known || stored.contains(&key);
                let sh2 = sh.clone();
                sh.pool.spawn(move || {
                    if skip {
                        sh2.ctr.legacy_skipped.fetch_add(1, Ordering::Relaxed);
                        lock(&sh2.presence).insert(key);
                    } else {
                        match EncodedThumb::from_legacy_string(&thumb) {
                            Ok(enc) => {
                                let blob = Arc::new(blob_from_encoded(enc));
                                queue_write(&sh2, &path, key, blob.clone(), BlobStamp::default());
                                sh2.ctr.legacy_ingested.fetch_add(1, Ordering::Relaxed);
                                sh2.emit(Event::Produced { key, blob });
                            }
                            Err(e) => {
                                sh2.ctr.legacy_failed.fetch_add(1, Ordering::Relaxed);
                                log::debug!("[thumbs] legacy thumbnail for {path} unusable: {e}");
                            }
                        }
                    }
                    sh2.emit(Event::LegacyDone { key });
                });
            }
        });
    }

    /// Probes which of `paths` have stored thumbnails and records them in the presence set.
    pub fn note_rows(&self, paths: Vec<String>) {
        if paths.is_empty() {
            return;
        }
        let sh = self.shared.clone();
        self.shared.rt.spawn(async move {
            let keys: Vec<ThumbKey> = {
                let presence = lock(&sh.presence);
                paths.iter().map(|p| ThumbKey::from_path(p)).filter(|k| !presence.contains(k)).collect()
            };
            for chunk in keys.chunks(PRESENCE_CHUNK) {
                match sh.store.exists_many(chunk).await {
                    Ok(found) => {
                        let mut presence = lock(&sh.presence);
                        for k in found {
                            presence.insert(k);
                        }
                    }
                    Err(e) => {
                        log::debug!("[thumbs] presence probe failed: {e:#}");
                        return;
                    }
                }
            }
            sh.wake();
        });
    }

    /// True when a stored thumbnail is known for `path`.
    pub fn has_thumb(&self, path: &str) -> bool {
        lock(&self.shared.presence).contains(&ThumbKey::from_path(path))
    }

    pub fn presence_len(&self) -> usize {
        lock(&self.shared.presence).len()
    }

    /// Sink the producer pool reports finished thumbnails to.
    pub fn sink(&self) -> Arc<dyn ThumbSink> {
        Arc::new(ServiceSink { shared: self.shared.clone() })
    }

    pub fn gate(&self) -> &Arc<StoreGate> {
        &self.shared.gate
    }

    pub fn store(&self) -> Arc<dyn ThumbBlobStore> {
        self.shared.store.clone()
    }

    pub fn runtime(&self) -> &Handle {
        &self.shared.rt
    }

    pub fn set_budget_bytes(&self, bytes: usize) {
        lock(&self.core).set_budget(bytes);
    }

    pub fn set_claim_budget(&self, per_frame: usize) {
        lock(&self.core).set_claim_budget(per_frame);
    }

    pub fn set_upload_per_frame(&self, n: usize) {
        self.upload_per_frame.store(n.max(1), Ordering::Relaxed);
    }

    /// Drops every texture, the encoded cache entries already written, and the negative cache.
    pub fn flush(&self) {
        lock(&self.core).flush();
        lock(&self.shared.recent).clear_unpinned();
    }

    /// Drops up to `n` on-screen textures; they are fetched and decoded again.
    pub fn force_decode(&self, n: usize) -> usize {
        let dropped = lock(&self.core).drop_on_screen(n);
        if dropped > 0 {
            let mut recent = lock(&self.shared.recent);
            recent.clear_unpinned();
        }
        dropped
    }

    pub fn forget_path(&self, path: &str) {
        lock(&self.core).forget_path(path);
        let key = ThumbKey::from_path(path);
        lock(&self.shared.recent).remove(&key);
    }

    pub fn clear_negative(&self) {
        lock(&self.core).clear_negative();
    }

    /// Reads the stored blob count and size in the background.
    pub fn request_store_stats(&self) {
        let sh = self.shared.clone();
        self.shared.rt.spawn(async move {
            if let Ok(stats) = sh.store.stats().await {
                *lock(&sh.store_stats) = Some(stats);
            }
            sh.wake();
        });
    }

    pub fn snapshot(&self) -> Snapshot {
        let (core_stats, cache, depths, budget, used, entries, on_screen, claims, claim_budget, fetch, decode, generate) = {
            let core = lock(&self.core);
            (
                core.stats(),
                core.cache_stats(),
                core.depths(),
                core.budget(),
                core.cache_bytes(),
                core.cache_len(),
                core.on_screen_count(),
                core.claims_in_flight(),
                core.config().claim_budget,
                core.fetch_inflight(),
                core.decode_inflight(),
                core.generate_inflight(),
            )
        };
        let staged = lock(&self.state).staged.len();
        let (recent_bytes, recent_entries, recent_pinned, recent_unqueued, recent_dropped) = {
            let r = lock(&self.shared.recent);
            (r.bytes(), r.len(), r.pinned(), r.unqueued(), r.dropped_pinned())
        };
        let c = &self.shared.ctr;
        Snapshot {
            core: core_stats,
            cache,
            depths,
            budget,
            used,
            entries,
            on_screen,
            claims_in_flight: claims,
            claim_budget,
            fetch_inflight: fetch,
            decode_inflight: decode,
            generate_inflight: generate,
            staged,
            upload_per_frame: self.upload_per_frame.load(Ordering::Relaxed),
            recent_bytes,
            recent_entries,
            recent_pinned,
            recent_unqueued,
            recent_dropped,
            presence: self.presence_len(),
            pending_writes: self.shared.pending_writes.load(Ordering::Relaxed),
            counters: CountersSnapshot {
                writes_ok: c.writes_ok.load(Ordering::Relaxed),
                write_errors: c.write_errors.load(Ordering::Relaxed),
                write_deferred: c.write_deferred.load(Ordering::Relaxed),
                write_overflow: c.write_overflow.load(Ordering::Relaxed),
                produced: c.produced.load(Ordering::Relaxed),
                fresh_hits: c.fresh_hits.load(Ordering::Relaxed),
                fresh_misses: c.fresh_misses.load(Ordering::Relaxed),
                legacy_ingested: c.legacy_ingested.load(Ordering::Relaxed),
                legacy_skipped: c.legacy_skipped.load(Ordering::Relaxed),
                legacy_failed: c.legacy_failed.load(Ordering::Relaxed),
                events: c.events.load(Ordering::Relaxed),
            },
            timings: lock(&self.shared.timings).snapshot(),
            store_delay_ms: self.shared.gate.delay_ms(),
            store_failing: self.shared.gate.failing(),
            store_stats: *lock(&self.shared.store_stats),
        }
    }

    /// Changes whenever a texture is added, replaced or dropped.
    pub fn ready_epoch(&self) -> u64 {
        lock(&self.core).ready_epoch()
    }

    fn write_perf_log(&self, st: &mut PollState) {
        let Some(perf) = st.perf.as_mut() else { return };
        if perf.last.elapsed() < Duration::from_secs(2) {
            return;
        }
        perf.last = Instant::now();
        let t = lock(&self.shared.timings);
        let (p50, p95, max) = (t.ui.percentile(0.5), t.ui.percentile(0.95), t.ui.percentile(1.0));
        drop(t);
        let (entries, used, cache, requests) = {
            let c = lock(&self.core);
            (c.cache_len(), c.cache_bytes(), c.cache_stats(), c.stats().requests)
        };
        let rss_kb = std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<u64>().ok()))
            .map(|pages| pages * 4)
            .unwrap_or(0);
        let line = format!(
            "{:.1},{p50:.2},{p95:.2},{max:.2},{entries},{used},{},{},{},{rss_kb},{requests},{}\n",
            perf.started.elapsed().as_secs_f32(),
            cache.hits,
            cache.claims_granted,
            cache.evictions,
            cache.reuse_hits
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&perf.path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

// Queues the write without blocking; an entry the queue rejects stays pinned for the writer's idle sweep.
fn queue_write(sh: &Shared, path: &str, key: ThumbKey, blob: Arc<Blob>, stamp: BlobStamp) {
    lock(&sh.recent).insert_pending(key, path, blob.clone(), stamp);
    lock(&sh.presence).insert(key);
    sh.pending_writes.fetch_add(1, Ordering::Relaxed);
    match sh.writer_tx.try_send(BlobIn { path: path.to_string(), key, blob, stamp }) {
        Ok(()) => lock(&sh.recent).mark_queued(&key),
        Err(err) => {
            sh.pending_writes.fetch_sub(1, Ordering::Relaxed);
            if matches!(err, tokio::sync::mpsc::error::TrySendError::Full(_)) {
                sh.ctr.write_overflow.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

struct ServiceSink {
    shared: Arc<Shared>,
}

impl ThumbSink for ServiceSink {
    fn fresh_key(&self, path: &Path) -> Option<String> {
        let sh = &self.shared;
        let hit = (|| {
            let want = file_stamp(path)?;
            let key = ThumbKey::from_path(&path.to_string_lossy());
            if let Some(have) = lock(&sh.recent).stamp(&key) {
                return (have == want).then_some(key);
            }
            if Handle::try_current().is_ok() {
                return None;
            }
            let found = sh.rt.block_on(sh.store.stamps_many(&[key])).ok()?;
            let have = found.get(&key)?;
            if have.src_size.is_some() && *have == want {
                lock(&sh.presence).insert(key);
                Some(key)
            } else {
                None
            }
        })();
        let counter = if hit.is_some() { &sh.ctr.fresh_hits } else { &sh.ctr.fresh_misses };
        counter.fetch_add(1, Ordering::Relaxed);
        hit.map(|k| k.hex())
    }

    fn accept(&self, path: &Path, thumb: EncodedThumb) -> String {
        let sh = &self.shared;
        let path_str = path.to_string_lossy().to_string();
        let key = ThumbKey::from_path(&path_str);
        let blob = Arc::new(blob_from_encoded(thumb));
        let stamp = file_stamp(path).unwrap_or_default();
        queue_write(sh, &path_str, key, blob.clone(), stamp);
        sh.ctr.produced.fetch_add(1, Ordering::Relaxed);
        sh.emit(Event::Produced { key, blob });
        key.hex()
    }

    fn failed(&self, path: &Path, error: &str) {
        let key = ThumbKey::from_path(&path.to_string_lossy());
        self.shared.emit(Event::ProducerFailed { key, error: error.to_string() });
    }
}

struct WriterCtx {
    rx: tokio::sync::mpsc::Receiver<BlobIn>,
    store: Arc<dyn ThumbBlobStore>,
    recent: Arc<Mutex<Recent>>,
    timings: Arc<Mutex<Timings>>,
    ctr: Arc<Counters>,
    pending: Arc<AtomicUsize>,
    batch: usize,
    retries: u32,
    retry_ms: u64,
    sweep: Duration,
}

// Writes one batch with retries and backoff; entries that still fail return to the sweep pool.
async fn write_batch(w: &WriterCtx, batch: Vec<BlobIn>) -> bool {
    let n = batch.len();
    let keys: Vec<ThumbKey> = batch.iter().map(|e| e.key).collect();
    let started = Instant::now();
    let mut attempt = 0u32;
    loop {
        match w.store.put(batch.clone()).await {
            Ok(_) => {
                w.ctr.writes_ok.fetch_add(n as u64, Ordering::Relaxed);
                lock(&w.timings).store_put.push_duration(started.elapsed());
                lock(&w.recent).mark_written(&keys);
                return true;
            }
            Err(e) => {
                w.ctr.write_errors.fetch_add(1, Ordering::Relaxed);
                attempt += 1;
                if attempt > w.retries {
                    log::debug!("[thumbs] deferring {n} thumbnail writes: {e:#}");
                    w.ctr.write_deferred.fetch_add(n as u64, Ordering::Relaxed);
                    lock(&w.recent).mark_unqueued(&keys);
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(w.retry_ms << (attempt - 1).min(4))).await;
            }
        }
    }
}

async fn writer_loop(mut w: WriterCtx) {
    let mut next_sweep = tokio::time::Instant::now();
    let mut sweep_delay = w.sweep;
    loop {
        match tokio::time::timeout(w.sweep, w.rx.recv()).await {
            Ok(None) => break,
            Ok(Some(first)) => {
                let mut batch = vec![first];
                let deadline = tokio::time::Instant::now() + WRITE_GATHER;
                while batch.len() < w.batch {
                    match tokio::time::timeout_at(deadline, w.rx.recv()).await {
                        Ok(Some(entry)) => batch.push(entry),
                        _ => break,
                    }
                }
                let n = batch.len();
                if write_batch(&w, batch).await {
                    sweep_delay = w.sweep;
                }
                w.pending.fetch_sub(n, Ordering::Relaxed);
            }
            Err(_) => {
                if tokio::time::Instant::now() < next_sweep {
                    continue;
                }
                let taken = lock(&w.recent).take_unqueued(w.batch);
                if taken.is_empty() {
                    continue;
                }
                let batch: Vec<BlobIn> = taken.into_iter().map(|(key, path, blob, stamp)| BlobIn { path, key, blob, stamp }).collect();
                let n = batch.len();
                w.pending.fetch_add(n, Ordering::Relaxed);
                if write_batch(&w, batch).await {
                    sweep_delay = w.sweep;
                    next_sweep = tokio::time::Instant::now();
                } else {
                    sweep_delay = (sweep_delay * 2).min(Duration::from_secs(30));
                    next_sweep = tokio::time::Instant::now() + sweep_delay;
                }
                w.pending.fetch_sub(n, Ordering::Relaxed);
            }
        }
    }
}

static GLOBAL: OnceLock<Arc<ThumbService>> = OnceLock::new();
static PRIVATE_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn private_runtime() -> Handle {
    PRIVATE_RUNTIME
        .get_or_init(|| tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().thread_name("thumb-io").build().expect("thumb runtime"))
        .handle()
        .clone()
}

// Tests use a private runtime; non-test builds use the current one when there is one.
fn runtime_handle() -> Handle {
    if cfg!(test) {
        return private_runtime();
    }
    Handle::try_current().unwrap_or_else(|_| private_runtime())
}

/// The app-wide service, created on first use.
pub fn global() -> Arc<ThumbService> {
    GLOBAL
        .get_or_init(|| {
            #[cfg(test)]
            let store: Arc<dyn ThumbBlobStore> = super::store::MemStore::new();
            #[cfg(not(test))]
            let store: Arc<dyn ThumbBlobStore> = Arc::new(super::surreal_store::GlobalDbStore);
            ThumbService::new(store, runtime_handle(), ServiceConfig::from_env(), Arc::new(PoolProducer))
        })
        .clone()
}

/// Sink installed into the scan thumbnail workers.
pub fn default_sink() -> Arc<dyn ThumbSink> {
    global().sink()
}
