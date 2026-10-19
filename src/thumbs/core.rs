//! Deterministic thumbnail request state machine: wants, stages, priorities, cancellation, negative cache and planning. It performs no I/O; the runtime executes the planned actions and feeds results back.

use super::cache::{CacheStats, Claim, ThumbCache};
use super::key::{CacheKey, ThumbKey, Tier};
use super::store::Blob;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Retry delay after the producer failed on a file.
pub const GENERATION_FAILED_RETRY_MS: u64 = 300_000;

#[derive(Clone)]
pub struct CoreConfig {
    pub budget_bytes: usize,
    pub claim_budget: usize,
    pub fetch_batch_max: usize,
    pub fetch_inflight_max: usize,
    pub decode_inflight_max: usize,
    pub generate_inflight_max: usize,
    pub stale_frames: u64,
    pub missing_retry_ms: u64,
    pub failed_retry_ms: u64,
    pub fetch_backoff_ms: u64,
    pub negative_cap: usize,
    pub generatable: fn(&str) -> bool,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            budget_bytes: super::cache::DEFAULT_BUDGET_BYTES,
            claim_budget: super::cache::DEFAULT_CLAIM_BUDGET,
            fetch_batch_max: 64,
            fetch_inflight_max: 2,
            decode_inflight_max: 16,
            generate_inflight_max: 24,
            stale_frames: 3,
            missing_retry_ms: 60_000,
            failed_retry_ms: 30_000,
            fetch_backoff_ms: 250,
            negative_cap: 20_000,
            generatable: default_generatable,
        }
    }
}

/// True for real image or video files; virtual archive and category paths are never generated from disk.
pub fn default_generatable(path: &str) -> bool {
    if path.contains("://") {
        return false;
    }
    let Some((_, ext)) = path.rsplit_once('.') else { return false };
    if ext.len() > 8 || ext.contains(['/', '\\']) {
        return false;
    }
    let lower = ext.to_ascii_lowercase();
    crate::utilities::types::is_image(&lower) || crate::utilities::types::is_video(&lower)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Queued,
    Fetching(u64),
    DecodeReady,
    Decoding,
    GenWait,
    Generating,
    LegacyWait,
}

struct Want {
    path: Arc<str>,
    scope: u32,
    scope_gen: u64,
    prio: f32,
    last_wanted: u64,
    stage: Stage,
    blob: Option<Arc<Blob>>,
    cancel: Arc<AtomicBool>,
    retry_at: u64,
    refresh: bool,
    gen_ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegKind {
    Missing,
    Failed,
}

#[derive(Clone, Copy)]
struct Neg {
    kind: NegKind,
    until_ms: u64,
}

#[derive(Debug)]
pub enum Lookup<'a, V> {
    Ready(&'a V),
    Pending,
    Missing,
    Failed,
}

impl<V> Lookup<'_, V> {
    pub fn is_ready(&self) -> bool {
        matches!(self, Lookup::Ready(_))
    }
}

pub enum DecodeSrc {
    Blob(Arc<Blob>),
    File,
}

pub struct DecodeJob {
    pub key: CacheKey,
    pub path: Arc<str>,
    pub src: DecodeSrc,
    pub cancel: Arc<AtomicBool>,
    pub refresh: bool,
}

pub enum Action {
    Fetch { batch: u64, keys: Vec<ThumbKey> },
    Decode(DecodeJob),
    Generate { key: ThumbKey, path: Arc<str> },
}

pub type FetchOutcome = Result<(Vec<(ThumbKey, Arc<Blob>)>, Vec<ThumbKey>), String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeResult {
    Image { cost: usize },
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeEnd {
    Inserted,
    UpdatedInPlace,
    Dropped,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoreStats {
    pub requests: u64,
    pub ahead_requests: u64,
    pub negative_hits: u64,
    pub wants_created: u64,
    pub cancelled_stale: u64,
    pub cancelled_scope: u64,
    pub fetch_batches: u64,
    pub fetch_keys: u64,
    pub fetch_found: u64,
    pub fetch_missing: u64,
    pub fetch_errors: u64,
    pub decode_started: u64,
    pub decode_ok: u64,
    pub decode_failed: u64,
    pub decode_dropped: u64,
    pub generate_started: u64,
    pub generate_failed: u64,
    pub uploads_new: u64,
    pub uploads_in_place: u64,
    pub refreshes_started: u64,
    pub refreshes_forgotten: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Depths {
    pub queued: usize,
    pub fetching: usize,
    pub decode_ready: usize,
    pub decoding: usize,
    pub gen_wait: usize,
    pub generating: usize,
    pub legacy_wait: usize,
    pub negative: usize,
}

pub struct Core<V> {
    cfg: CoreConfig,
    cache: ThumbCache<CacheKey, V>,
    wants: HashMap<CacheKey, Want>,
    negative: HashMap<CacheKey, Neg>,
    scopes: HashMap<u32, u64>,
    legacy_pending: HashSet<ThumbKey>,
    generating: HashSet<ThumbKey>,
    frame: u64,
    now_ms: u64,
    next_batch: u64,
    fetch_inflight: usize,
    decode_inflight: usize,
    fetch_fail_streak: u32,
    claims_denied_this_frame: usize,
    ready_epoch: u64,
    scratch: Vec<(f32, CacheKey)>,
    stats: CoreStats,
}

impl<V> Core<V> {
    pub fn new(cfg: CoreConfig) -> Self {
        let mut cache = ThumbCache::new(cfg.budget_bytes);
        cache.set_claim_budget(cfg.claim_budget);
        Self {
            cfg,
            cache,
            wants: HashMap::new(),
            negative: HashMap::new(),
            scopes: HashMap::new(),
            legacy_pending: HashSet::new(),
            generating: HashSet::new(),
            frame: 0,
            now_ms: 0,
            next_batch: 1,
            fetch_inflight: 0,
            decode_inflight: 0,
            fetch_fail_streak: 0,
            claims_denied_this_frame: 0,
            ready_epoch: 0,
            scratch: Vec::new(),
            stats: CoreStats::default(),
        }
    }

    pub fn config(&self) -> &CoreConfig {
        &self.cfg
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn now_ms(&self) -> u64 {
        self.now_ms
    }

    pub fn stats(&self) -> CoreStats {
        self.stats
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache.bytes()
    }

    pub fn budget(&self) -> usize {
        self.cache.budget()
    }

    pub fn on_screen_count(&self) -> usize {
        self.cache.on_screen_count()
    }

    pub fn wants_len(&self) -> usize {
        self.wants.len()
    }

    pub fn claims_in_flight(&self) -> usize {
        self.cache.pending_len()
    }

    pub fn claims_denied_this_frame(&self) -> usize {
        self.claims_denied_this_frame
    }

    pub fn fetch_inflight(&self) -> usize {
        self.fetch_inflight
    }

    pub fn decode_inflight(&self) -> usize {
        self.decode_inflight
    }

    pub fn generate_inflight(&self) -> usize {
        self.generating.len()
    }

    /// Changes whenever a texture is added, replaced or dropped.
    pub fn ready_epoch(&self) -> u64 {
        self.ready_epoch
    }

    pub fn set_budget(&mut self, bytes: usize) {
        self.cache.set_budget(bytes);
    }

    pub fn set_claim_budget(&mut self, per_frame: usize) {
        self.cfg.claim_budget = per_frame;
        self.cache.set_claim_budget(per_frame);
    }

    pub fn depths(&self) -> Depths {
        let mut d = Depths { negative: self.negative.len(), ..Depths::default() };
        for w in self.wants.values() {
            match w.stage {
                Stage::Queued => d.queued += 1,
                Stage::Fetching(_) => d.fetching += 1,
                Stage::DecodeReady => d.decode_ready += 1,
                Stage::Decoding => d.decoding += 1,
                Stage::GenWait => d.gen_wait += 1,
                Stage::Generating => d.generating += 1,
                Stage::LegacyWait => d.legacy_wait += 1,
            }
        }
        d
    }

    pub fn is_wanted(&self, path: &str, tier: Tier) -> bool {
        self.wants.contains_key(&CacheKey::new(path, tier))
    }

    pub fn is_cached(&self, path: &str, tier: Tier) -> bool {
        self.cache.contains(&CacheKey::new(path, tier))
    }

    /// Starts a frame: advances the clock and drops wants that were not requested recently or whose scope changed.
    pub fn begin_frame(&mut self, now_ms: u64) {
        self.frame += 1;
        self.now_ms = now_ms.max(self.now_ms);
        self.claims_denied_this_frame = 0;
        self.cache.begin_frame();
        let stale_before = self.frame.saturating_sub(self.cfg.stale_frames);
        let Self { wants, cache, scopes, stats, .. } = self;
        wants.retain(|key, w| {
            let scope_gone = scopes.get(&w.scope).is_some_and(|g| *g != w.scope_gen);
            let stale = w.last_wanted < stale_before && !w.refresh;
            if !(scope_gone || stale) {
                return true;
            }
            w.cancel.store(true, Ordering::Relaxed);
            cache.release(key);
            if scope_gone {
                stats.cancelled_scope += 1;
            } else {
                stats.cancelled_stale += 1;
            }
            false
        });
    }

    /// Invalidates everything requested under `scope`: queued work is dropped and running work is flagged cancelled.
    pub fn new_scope(&mut self, scope: u32) {
        *self.scopes.entry(scope).or_insert(0) += 1;
        let current = self.scopes[&scope];
        let Self { wants, cache, stats, .. } = self;
        wants.retain(|key, w| {
            if w.scope != scope || w.scope_gen == current {
                return true;
            }
            w.cancel.store(true, Ordering::Relaxed);
            cache.release(key);
            stats.cancelled_scope += 1;
            false
        });
    }

    /// Looks up a texture for a cell that is being drawn this frame, creating or refreshing the want when it is not cached.
    pub fn request(&mut self, scope: u32, path: &str, tier: Tier, prio: f32) -> Lookup<'_, V> {
        self.stats.requests += 1;
        let key = CacheKey::new(path, tier);
        if self.cache.get(&key).is_some() {
            return match self.cache.peek(&key) {
                Some(v) => Lookup::Ready(v),
                None => Lookup::Pending,
            };
        }
        self.want_missing(key, scope, path, tier, prio)
    }

    /// Registers interest in a texture that is about to scroll into view; a cached texture is not stamped as drawn, so it stays evictable. Returns whether it is cached.
    pub fn request_ahead(&mut self, scope: u32, path: &str, tier: Tier, prio: f32) -> bool {
        self.stats.ahead_requests += 1;
        let key = CacheKey::new(path, tier);
        if self.cache.contains(&key) {
            return true;
        }
        let _ = self.want_missing(key, scope, path, tier, prio);
        false
    }

    // Creates or refreshes the want for a texture that is not cached.
    fn want_missing(&mut self, key: CacheKey, scope: u32, path: &str, tier: Tier, prio: f32) -> Lookup<'_, V> {
        let scope_gen = *self.scopes.entry(scope).or_insert(0);
        if let Some(w) = self.wants.get_mut(&key) {
            w.last_wanted = self.frame;
            w.prio = prio;
            w.scope = scope;
            w.scope_gen = scope_gen;
            return Lookup::Pending;
        }
        if let Some(neg) = self.negative.get(&key).copied() {
            if neg.until_ms > self.now_ms {
                self.stats.negative_hits += 1;
                return match neg.kind {
                    NegKind::Missing => Lookup::Missing,
                    NegKind::Failed => Lookup::Failed,
                };
            }
            self.negative.remove(&key);
        }
        match self.cache.claim(&key) {
            Claim::Granted => {}
            Claim::OverBudget => {
                self.claims_denied_this_frame += 1;
                return Lookup::Pending;
            }
            Claim::Cached | Claim::Pending => return Lookup::Pending,
        }
        let (stage, gen_ok) = match tier {
            Tier::Thumb => (Stage::Queued, (self.cfg.generatable)(path)),
            Tier::Preview => (Stage::DecodeReady, false),
        };
        self.wants.insert(
            key,
            Want {
                path: Arc::from(path),
                scope,
                scope_gen,
                prio,
                last_wanted: self.frame,
                stage,
                blob: None,
                cancel: Arc::new(AtomicBool::new(false)),
                retry_at: 0,
                refresh: false,
                gen_ok,
            },
        );
        self.stats.wants_created += 1;
        Lookup::Pending
    }

    /// Reads a cached texture without creating a want or stamping it as drawn.
    pub fn peek(&self, path: &str, tier: Tier) -> Option<&V> {
        self.cache.peek(&CacheKey::new(path, tier))
    }

    /// Plans the next batch of work within the in-flight limits, nearest-to-center first.
    pub fn plan(&mut self, out: &mut Vec<Action>) {
        self.plan_fetch(out);
        self.plan_decode(out);
        self.plan_generate(out);
    }

    fn collect_stage(&mut self, pred: impl Fn(&Want) -> bool) {
        self.scratch.clear();
        for (k, w) in &self.wants {
            if pred(w) {
                self.scratch.push((w.prio, *k));
            }
        }
        self.scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    fn plan_fetch(&mut self, out: &mut Vec<Action>) {
        if self.fetch_inflight >= self.cfg.fetch_inflight_max {
            return;
        }
        let now = self.now_ms;
        self.collect_stage(|w| w.stage == Stage::Queued && w.retry_at <= now);
        if self.scratch.is_empty() {
            return;
        }
        let batch = self.next_batch;
        self.next_batch += 1;
        let take = self.scratch.len().min(self.cfg.fetch_batch_max);
        let mut keys = Vec::with_capacity(take);
        for (_, ck) in &self.scratch[..take] {
            if let Some(w) = self.wants.get_mut(ck) {
                w.stage = Stage::Fetching(batch);
            }
            keys.push(ck.key);
        }
        self.fetch_inflight += 1;
        self.stats.fetch_batches += 1;
        self.stats.fetch_keys += take as u64;
        out.push(Action::Fetch { batch, keys });
    }

    fn plan_decode(&mut self, out: &mut Vec<Action>) {
        let free = self.cfg.decode_inflight_max.saturating_sub(self.decode_inflight);
        if free == 0 {
            return;
        }
        self.collect_stage(|w| w.stage == Stage::DecodeReady);
        let take = self.scratch.len().min(free);
        for i in 0..take {
            let ck = self.scratch[i].1;
            let Some(w) = self.wants.get_mut(&ck) else { continue };
            let src = match (&w.blob, ck.tier) {
                (Some(b), Tier::Thumb) => DecodeSrc::Blob(b.clone()),
                (None, Tier::Preview) => DecodeSrc::File,
                _ => continue,
            };
            w.stage = Stage::Decoding;
            self.decode_inflight += 1;
            self.stats.decode_started += 1;
            out.push(Action::Decode(DecodeJob { key: ck, path: w.path.clone(), src, cancel: w.cancel.clone(), refresh: w.refresh }));
        }
    }

    fn plan_generate(&mut self, out: &mut Vec<Action>) {
        let free = self.cfg.generate_inflight_max.saturating_sub(self.generating.len());
        if free == 0 {
            return;
        }
        self.collect_stage(|w| w.stage == Stage::GenWait);
        let take = self.scratch.len().min(free);
        for i in 0..take {
            let ck = self.scratch[i].1;
            let Some(w) = self.wants.get_mut(&ck) else { continue };
            if !self.generating.insert(ck.key) {
                w.stage = Stage::Generating;
                continue;
            }
            w.stage = Stage::Generating;
            self.stats.generate_started += 1;
            out.push(Action::Generate { key: ck.key, path: w.path.clone() });
        }
    }

    /// Applies the result of a store batch.
    pub fn on_fetch_done(&mut self, batch: u64, outcome: FetchOutcome) {
        self.fetch_inflight = self.fetch_inflight.saturating_sub(1);
        match outcome {
            Err(_) => {
                self.stats.fetch_errors += 1;
                self.fetch_fail_streak = self.fetch_fail_streak.saturating_add(1);
                let shift = (self.fetch_fail_streak - 1).min(5);
                let delay = (self.cfg.fetch_backoff_ms << shift).min(8_000);
                let retry_at = self.now_ms + delay;
                for w in self.wants.values_mut() {
                    if w.stage == Stage::Fetching(batch) {
                        w.stage = Stage::Queued;
                        w.retry_at = retry_at;
                    }
                }
            }
            Ok((found, missing)) => {
                self.fetch_fail_streak = 0;
                for (key, blob) in found {
                    self.stats.fetch_found += 1;
                    let ck = CacheKey { key, tier: Tier::Thumb };
                    if let Some(w) = self.wants.get_mut(&ck) {
                        if w.stage == Stage::Fetching(batch) {
                            w.blob = Some(blob);
                            w.stage = Stage::DecodeReady;
                        }
                    }
                }
                for key in missing {
                    self.stats.fetch_missing += 1;
                    self.on_blob_missing(key, batch);
                }
            }
        }
    }

    fn on_blob_missing(&mut self, key: ThumbKey, batch: u64) {
        let ck = CacheKey { key, tier: Tier::Thumb };
        let legacy = self.legacy_pending.contains(&key);
        let Some(w) = self.wants.get_mut(&ck) else { return };
        if w.stage != Stage::Fetching(batch) {
            return;
        }
        if legacy {
            w.stage = Stage::LegacyWait;
        } else if w.gen_ok {
            w.stage = Stage::GenWait;
        } else {
            self.drop_want_negative(ck, NegKind::Missing, self.cfg.missing_retry_ms);
        }
    }

    fn drop_want_negative(&mut self, ck: CacheKey, kind: NegKind, retry_ms: u64) {
        if let Some(w) = self.wants.remove(&ck) {
            w.cancel.store(true, Ordering::Relaxed);
        }
        self.cache.release(&ck);
        self.set_negative(ck, kind, retry_ms);
    }

    fn set_negative(&mut self, ck: CacheKey, kind: NegKind, retry_ms: u64) {
        if self.negative.len() >= self.cfg.negative_cap {
            let now = self.now_ms;
            self.negative.retain(|_, n| n.until_ms > now);
            if self.negative.len() >= self.cfg.negative_cap {
                self.negative.clear();
            }
        }
        self.negative.insert(ck, Neg { kind, until_ms: self.now_ms + retry_ms });
    }

    /// A thumbnail blob became available outside a fetch (producer or legacy migration).
    pub fn on_blob_ready(&mut self, key: ThumbKey, blob: Arc<Blob>) {
        let ck = CacheKey { key, tier: Tier::Thumb };
        self.negative.remove(&ck);
        self.legacy_pending.remove(&key);
        if let Some(w) = self.wants.get_mut(&ck) {
            if !matches!(w.stage, Stage::DecodeReady | Stage::Decoding) {
                w.blob = Some(blob);
                w.stage = Stage::DecodeReady;
            } else if w.stage == Stage::DecodeReady {
                w.blob = Some(blob);
            }
            return;
        }
        if !self.cache.contains(&ck) {
            return;
        }
        if self.cache.is_on_screen(&ck) {
            self.wants.insert(
                ck,
                Want {
                    path: Arc::from(""),
                    scope: 0,
                    scope_gen: self.scopes.get(&0).copied().unwrap_or(0),
                    prio: 0.0,
                    last_wanted: self.frame,
                    stage: Stage::DecodeReady,
                    blob: Some(blob),
                    cancel: Arc::new(AtomicBool::new(false)),
                    retry_at: 0,
                    refresh: true,
                    gen_ok: false,
                },
            );
            self.stats.refreshes_started += 1;
        } else {
            self.cache.forget(&ck);
            self.ready_epoch += 1;
            self.stats.refreshes_forgotten += 1;
        }
    }

    /// The producer finished a standalone generation request for `key`.
    pub fn on_generation_finished(&mut self, key: ThumbKey, error: Option<&str>) {
        let was_requested = self.generating.remove(&key);
        let ck = CacheKey { key, tier: Tier::Thumb };
        if let Some(msg) = error {
            self.stats.generate_failed += 1;
            let _ = msg;
            self.drop_want_negative(ck, NegKind::Failed, GENERATION_FAILED_RETRY_MS);
            return;
        }
        if was_requested {
            let now = self.now_ms;
            if let Some(w) = self.wants.get_mut(&ck) {
                if w.stage == Stage::Generating && w.blob.is_none() {
                    w.stage = Stage::Queued;
                    w.retry_at = now;
                }
            }
        }
    }

    /// The producer queue refused the job; try again on a later plan.
    pub fn on_generation_rejected(&mut self, key: ThumbKey) {
        self.generating.remove(&key);
        if let Some(w) = self.wants.get_mut(&CacheKey { key, tier: Tier::Thumb }) {
            if w.stage == Stage::Generating {
                w.stage = Stage::GenWait;
            }
        }
    }

    /// A legacy payload for `key` is being converted; a store miss for it must wait instead of generating.
    pub fn legacy_started(&mut self, key: ThumbKey) {
        self.legacy_pending.insert(key);
    }

    /// The legacy conversion for `key` ended; a want that was waiting on it asks the store again, which falls back to generation on a miss.
    pub fn legacy_finished(&mut self, key: ThumbKey) {
        self.legacy_pending.remove(&key);
        let now = self.now_ms;
        let ck = CacheKey { key, tier: Tier::Thumb };
        if let Some(w) = self.wants.get_mut(&ck) {
            if w.stage == Stage::LegacyWait {
                w.stage = Stage::Queued;
                w.retry_at = now;
            }
        }
    }

    /// Whether a decoded image for `key` is still wanted.
    pub fn decode_alive(&self, key: &CacheKey) -> bool {
        self.wants.get(key).is_some_and(|w| w.stage == Stage::Decoding && !w.cancel.load(Ordering::Relaxed))
    }

    /// Finishes a decode job: uploads the image through `upload` when still wanted, or drops it, and frees the decode slot.
    pub fn complete_decode(&mut self, key: CacheKey, result: DecodeResult, upload: impl FnOnce(Option<&mut V>) -> Option<V>) -> DecodeEnd {
        self.decode_inflight = self.decode_inflight.saturating_sub(1);
        let alive = self.decode_alive(&key);
        match result {
            DecodeResult::Cancelled => {
                self.stats.decode_dropped += 1;
                DecodeEnd::Dropped
            }
            DecodeResult::Failed => {
                self.stats.decode_failed += 1;
                if alive {
                    let retry = self.cfg.failed_retry_ms;
                    let refresh = self.wants.get(&key).is_some_and(|w| w.refresh);
                    if refresh {
                        self.wants.remove(&key);
                    } else {
                        self.drop_want_negative(key, NegKind::Failed, retry);
                    }
                }
                DecodeEnd::Failed
            }
            DecodeResult::Image { cost } => {
                if !alive {
                    self.stats.decode_dropped += 1;
                    return DecodeEnd::Dropped;
                }
                self.stats.decode_ok += 1;
                self.wants.remove(&key);
                self.ready_epoch += 1;
                if let Some(existing) = self.cache.get_mut(&key) {
                    let _ = upload(Some(existing));
                    self.cache.update_in_place(&key, cost, |_| {});
                    self.stats.uploads_in_place += 1;
                    DecodeEnd::UpdatedInPlace
                } else if let Some(v) = upload(None) {
                    self.cache.insert(key, v, cost);
                    self.stats.uploads_new += 1;
                    DecodeEnd::Inserted
                } else {
                    self.cache.release(&key);
                    DecodeEnd::Dropped
                }
            }
        }
    }

    /// Drops one path from every tier, the negative cache and the want list.
    pub fn forget_path(&mut self, path: &str) {
        for tier in [Tier::Thumb, Tier::Preview] {
            let ck = CacheKey::new(path, tier);
            self.negative.remove(&ck);
            if let Some(w) = self.wants.remove(&ck) {
                w.cancel.store(true, Ordering::Relaxed);
            }
            if self.cache.forget(&ck).is_some() {
                self.ready_epoch += 1;
            }
        }
    }

    /// Drops up to `n` on-screen textures; the next frame requests them again.
    pub fn drop_on_screen(&mut self, n: usize) -> usize {
        let mut victims: Vec<CacheKey> = Vec::new();
        for k in self.cache.keys() {
            if victims.len() >= n {
                break;
            }
            if k.tier == Tier::Thumb && self.cache.is_on_screen(k) {
                victims.push(*k);
            }
        }
        for k in &victims {
            self.cache.forget(k);
        }
        if !victims.is_empty() {
            self.ready_epoch += 1;
        }
        victims.len()
    }

    /// Drops every texture and negative entry; live wants keep their claims and continue.
    pub fn flush(&mut self) {
        self.wants.retain(|_, w| !w.refresh);
        self.cache.clear();
        let Self { wants, cache, .. } = self;
        for key in wants.keys() {
            cache.mark_claimed(*key);
        }
        self.negative.clear();
        self.ready_epoch += 1;
    }

    pub fn clear_negative(&mut self) {
        self.negative.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::store::BlobFormat;

    type TestCore = Core<u32>;

    fn blob() -> Arc<Blob> {
        Arc::new(Blob { format: BlobFormat::Jpeg, w: 8, h: 8, bytes: vec![0xFF, 0xD8, 0xFF] })
    }

    fn always(_: &str) -> bool {
        true
    }

    fn cfg() -> CoreConfig {
        CoreConfig { generatable: always, ..CoreConfig::default() }
    }

    fn core_with(cfg: CoreConfig) -> TestCore {
        let mut c = TestCore::new(cfg);
        c.begin_frame(1_000);
        c
    }

    fn plan(c: &mut TestCore) -> Vec<Action> {
        let mut out = Vec::new();
        c.plan(&mut out);
        out
    }

    fn fetch_keys(actions: &[Action]) -> Vec<ThumbKey> {
        actions
            .iter()
            .flat_map(|a| match a {
                Action::Fetch { keys, .. } => keys.clone(),
                _ => Vec::new(),
            })
            .collect()
    }

    fn batch_id(actions: &[Action]) -> u64 {
        actions
            .iter()
            .find_map(|a| match a {
                Action::Fetch { batch, .. } => Some(*batch),
                _ => None,
            })
            .expect("a fetch action")
    }

    fn decodes(actions: &[Action]) -> Vec<&DecodeJob> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Decode(j) => Some(j),
                _ => None,
            })
            .collect()
    }

    fn generates(actions: &[Action]) -> Vec<ThumbKey> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Generate { key, .. } => Some(*key),
                _ => None,
            })
            .collect()
    }

    fn finish_ok(c: &mut TestCore, job: &DecodeJob) -> DecodeEnd {
        c.complete_decode(job.key, DecodeResult::Image { cost: 1000 }, |_| Some(7))
    }

    fn next_frame(c: &mut TestCore, ms: u64) {
        let now = c.now_ms();
        c.begin_frame(now + ms);
    }

    #[test]
    fn a_drawn_cell_makes_exactly_one_want_and_one_fetch_across_frames() {
        let mut c = core_with(cfg());
        let mut fetched = 0;
        for _ in 0..30 {
            assert!(matches!(c.request(1, "/a/one.jpg", Tier::Thumb, 5.0), Lookup::Pending));
            fetched += fetch_keys(&plan(&mut c)).len();
            next_frame(&mut c, 16);
        }
        assert_eq!(fetched, 1);
        assert_eq!(c.stats().wants_created, 1);
        assert_eq!(c.stats().requests, 30);
        assert_eq!(c.claims_in_flight(), 1);
    }

    #[test]
    fn an_undrawn_path_never_reaches_the_store() {
        let mut c = core_with(cfg());
        c.request(1, "/a/drawn.jpg", Tier::Thumb, 0.0);
        let keys = fetch_keys(&plan(&mut c));
        assert_eq!(keys, vec![ThumbKey::from_path("/a/drawn.jpg")]);
        assert!(!c.is_wanted("/a/undrawn.jpg", Tier::Thumb));
        assert_eq!(c.stats().fetch_keys, 1);
    }

    #[test]
    fn fetches_are_batched_and_capped_in_flight() {
        let mut config = cfg();
        config.claim_budget = 1000;
        config.fetch_batch_max = 8;
        config.fetch_inflight_max = 2;
        let mut c = core_with(config);
        for i in 0..20 {
            c.request(1, &format!("/p/{i}.jpg"), Tier::Thumb, i as f32);
        }
        let first = plan(&mut c);
        assert_eq!(fetch_keys(&first).len(), 8);
        let second = plan(&mut c);
        assert_eq!(fetch_keys(&second).len(), 8);
        assert!(plan(&mut c).is_empty(), "two batches in flight is the cap");
        c.on_fetch_done(batch_id(&first), Ok((Vec::new(), Vec::new())));
        let third = plan(&mut c);
        assert_eq!(fetch_keys(&third).len(), 4);
        assert_eq!(c.stats().fetch_batches, 3);
    }

    #[test]
    fn nearest_to_center_is_fetched_and_decoded_first() {
        let mut config = cfg();
        config.claim_budget = 100;
        config.fetch_batch_max = 3;
        config.decode_inflight_max = 2;
        let mut c = core_with(config);
        for (i, prio) in [90.0f32, 10.0, 50.0, 30.0, 70.0].into_iter().enumerate() {
            c.request(1, &format!("/p/{i}.jpg"), Tier::Thumb, prio);
        }
        let actions = plan(&mut c);
        let order = fetch_keys(&actions);
        assert_eq!(order, vec![ThumbKey::from_path("/p/1.jpg"), ThumbKey::from_path("/p/3.jpg"), ThumbKey::from_path("/p/2.jpg")]);
        let found: Vec<_> = order.iter().map(|k| (*k, blob())).collect();
        c.on_fetch_done(batch_id(&actions), Ok((found, Vec::new())));
        let planned = plan(&mut c);
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 2, "decode slots cap the work released");
        assert_eq!(jobs[0].key.key, ThumbKey::from_path("/p/1.jpg"));
        assert_eq!(jobs[1].key.key, ThumbKey::from_path("/p/3.jpg"));
    }

    #[test]
    fn priority_updates_follow_the_latest_request() {
        let mut config = cfg();
        config.claim_budget = 100;
        config.fetch_batch_max = 1;
        let mut c = core_with(config);
        c.request(1, "/a.jpg", Tier::Thumb, 5.0);
        c.request(1, "/b.jpg", Tier::Thumb, 9.0);
        c.request(1, "/a.jpg", Tier::Thumb, 500.0);
        assert_eq!(fetch_keys(&plan(&mut c)), vec![ThumbKey::from_path("/b.jpg")]);
    }

    #[test]
    fn claims_are_capped_per_frame_and_the_rest_wait_for_the_next_frame() {
        let mut c = core_with(cfg());
        for i in 0..30 {
            assert!(matches!(c.request(1, &format!("/p/{i}.jpg"), Tier::Thumb, 0.0), Lookup::Pending));
        }
        assert_eq!(c.wants_len(), 12);
        assert_eq!(c.claims_denied_this_frame(), 18);
        next_frame(&mut c, 16);
        for i in 0..30 {
            c.request(1, &format!("/p/{i}.jpg"), Tier::Thumb, 0.0);
        }
        assert_eq!(c.wants_len(), 24);
        next_frame(&mut c, 16);
        for i in 0..30 {
            c.request(1, &format!("/p/{i}.jpg"), Tier::Thumb, 0.0);
        }
        assert_eq!(c.wants_len(), 30);
    }

    #[test]
    fn a_row_that_leaves_the_viewport_is_cancelled_before_decoding() {
        let mut config = cfg();
        config.stale_frames = 2;
        let mut c = core_with(config);
        c.request(1, "/leaves.jpg", Tier::Thumb, 0.0);
        c.request(1, "/stays.jpg", Tier::Thumb, 0.0);
        let actions = plan(&mut c);
        let found = vec![(ThumbKey::from_path("/leaves.jpg"), blob()), (ThumbKey::from_path("/stays.jpg"), blob())];
        c.on_fetch_done(batch_id(&actions), Ok((found, Vec::new())));
        for _ in 0..4 {
            next_frame(&mut c, 16);
            c.request(1, "/stays.jpg", Tier::Thumb, 0.0);
        }
        assert!(!c.is_wanted("/leaves.jpg", Tier::Thumb));
        assert!(c.is_wanted("/stays.jpg", Tier::Thumb));
        let planned = plan(&mut c);
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].key.key, ThumbKey::from_path("/stays.jpg"));
        assert_eq!(c.stats().cancelled_stale, 1);
        assert_eq!(c.claims_in_flight(), 1, "the dropped claim is released");
        assert!(matches!(c.request(1, "/leaves.jpg", Tier::Thumb, 0.0), Lookup::Pending), "and can be claimed again");
    }

    #[test]
    fn running_decodes_are_flagged_cancelled_and_their_results_dropped() {
        let mut config = cfg();
        config.stale_frames = 1;
        let mut c = core_with(config);
        c.request(1, "/gone.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path("/gone.jpg"), blob())], Vec::new())));
        let planned = plan(&mut c);
        let job = decodes(&planned)[0];
        assert!(!job.cancel.load(Ordering::Relaxed));
        assert_eq!(c.decode_inflight(), 1);
        next_frame(&mut c, 16);
        next_frame(&mut c, 16);
        next_frame(&mut c, 16);
        assert!(job.cancel.load(Ordering::Relaxed), "the worker can see the cancellation");
        let end = finish_ok(&mut c, job);
        assert_eq!(end, DecodeEnd::Dropped);
        assert_eq!(c.decode_inflight(), 0, "the slot comes back");
        assert!(c.peek("/gone.jpg", Tier::Thumb).is_none(), "nothing is uploaded for a dead want");
    }

    #[test]
    fn a_scope_change_cancels_only_that_scope() {
        let mut c = core_with(cfg());
        c.request(1, "/one/a.jpg", Tier::Thumb, 0.0);
        c.request(1, "/one/b.jpg", Tier::Thumb, 0.0);
        c.request(2, "/two/a.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.new_scope(1);
        assert_eq!(c.wants_len(), 1);
        assert_eq!(c.stats().cancelled_scope, 2);
        c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path("/one/a.jpg"), blob()), (ThumbKey::from_path("/two/a.jpg"), blob())], Vec::new())));
        let planned = plan(&mut c);
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].key.key, ThumbKey::from_path("/two/a.jpg"));
        assert!(matches!(c.request(1, "/one/a.jpg", Tier::Thumb, 0.0), Lookup::Pending));
        assert!(c.is_wanted("/one/a.jpg", Tier::Thumb), "the new scope may ask again");
    }

    #[test]
    fn a_decoded_image_is_uploaded_and_then_served_from_the_cache() {
        let mut c = core_with(cfg());
        c.request(1, "/a.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path("/a.jpg"), blob())], Vec::new())));
        let planned = plan(&mut c);
        let job = decodes(&planned)[0];
        assert_eq!(finish_ok(&mut c, job), DecodeEnd::Inserted);
        assert!(matches!(c.request(1, "/a.jpg", Tier::Thumb, 0.0), Lookup::Ready(7)));
        assert_eq!(c.wants_len(), 0);
        assert_eq!(c.cache_len(), 1);
        assert_eq!(c.cache_bytes(), 1000);
        assert_eq!(c.claims_in_flight(), 0);
        assert!(c.ready_epoch() > 0);
        assert!(plan(&mut c).is_empty());
    }

    #[test]
    fn a_store_miss_on_a_missing_file_is_negative_cached_until_it_expires() {
        let mut config = cfg();
        config.generatable = |_| false;
        config.missing_retry_ms = 5_000;
        let mut c = core_with(config);
        c.request(1, "/virtual/x.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![ThumbKey::from_path("/virtual/x.jpg")])));
        assert_eq!(c.wants_len(), 0);
        assert_eq!(c.claims_in_flight(), 0);
        for _ in 0..50 {
            next_frame(&mut c, 50);
            if c.now_ms() >= 1_000 + 5_000 {
                break;
            }
            assert!(matches!(c.request(1, "/virtual/x.jpg", Tier::Thumb, 0.0), Lookup::Missing));
            assert!(plan(&mut c).is_empty(), "a negative entry costs no store traffic");
        }
        assert!(c.stats().negative_hits >= 10);
        c.begin_frame(7_000);
        assert!(matches!(c.request(1, "/virtual/x.jpg", Tier::Thumb, 0.0), Lookup::Pending), "expired entries are retried");
        assert_eq!(fetch_keys(&plan(&mut c)).len(), 1);
    }

    #[test]
    fn a_miss_on_a_real_file_asks_the_producer_once_and_decodes_what_it_makes() {
        let mut c = core_with(cfg());
        c.request(1, "/real/img.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/real/img.jpg");
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        let planned = plan(&mut c);
        assert_eq!(generates(&planned), vec![key]);
        assert!(plan(&mut c).is_empty(), "the request is not repeated while it is in flight");
        assert_eq!(c.generate_inflight(), 1);
        c.on_blob_ready(key, blob());
        c.on_generation_finished(key, None);
        assert_eq!(c.generate_inflight(), 0);
        let planned = plan(&mut c);
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 1);
        assert_eq!(finish_ok(&mut c, jobs[0]), DecodeEnd::Inserted);
    }

    #[test]
    fn generation_slots_cap_producer_requests_nearest_first() {
        let mut config = cfg();
        config.claim_budget = 100;
        config.generate_inflight_max = 2;
        let mut c = core_with(config);
        let paths = ["/g/0.jpg", "/g/1.jpg", "/g/2.jpg", "/g/3.jpg"];
        for (p, prio) in paths.iter().zip([40.0f32, 10.0, 30.0, 20.0]) {
            c.request(1, p, Tier::Thumb, prio);
        }
        let a = plan(&mut c);
        let missing: Vec<_> = paths.iter().map(|p| ThumbKey::from_path(p)).collect();
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), missing)));
        let g = generates(&plan(&mut c));
        assert_eq!(g, vec![ThumbKey::from_path("/g/1.jpg"), ThumbKey::from_path("/g/3.jpg")]);
        c.on_generation_finished(ThumbKey::from_path("/g/1.jpg"), None);
        let g = generates(&plan(&mut c));
        assert_eq!(g, vec![ThumbKey::from_path("/g/2.jpg")]);
    }

    #[test]
    fn a_failed_generation_becomes_a_long_negative_entry() {
        let mut c = core_with(cfg());
        c.request(1, "/bad/img.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/bad/img.jpg");
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        plan(&mut c);
        c.on_generation_finished(key, Some("decode error"));
        assert_eq!(c.wants_len(), 0);
        assert_eq!(c.claims_in_flight(), 0);
        assert!(matches!(c.request(1, "/bad/img.jpg", Tier::Thumb, 0.0), Lookup::Failed));
        c.begin_frame(1_000 + GENERATION_FAILED_RETRY_MS - 1);
        assert!(matches!(c.request(1, "/bad/img.jpg", Tier::Thumb, 0.0), Lookup::Failed));
        c.begin_frame(1_000 + GENERATION_FAILED_RETRY_MS + 1);
        assert!(matches!(c.request(1, "/bad/img.jpg", Tier::Thumb, 0.0), Lookup::Pending));
    }

    #[test]
    fn a_ready_blob_clears_the_negative_entry() {
        let mut config = cfg();
        config.generatable = |_| false;
        let mut c = core_with(config);
        c.request(1, "/zip/a.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/zip/a.jpg");
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        assert!(matches!(c.request(1, "/zip/a.jpg", Tier::Thumb, 0.0), Lookup::Missing));
        c.on_blob_ready(key, blob());
        assert!(matches!(c.request(1, "/zip/a.jpg", Tier::Thumb, 0.0), Lookup::Pending));
        let planned = plan(&mut c);
        assert_eq!(fetch_keys(&planned).len(), 1, "after readiness the cell asks again");
    }

    #[test]
    fn a_blob_arriving_mid_fetch_wins_over_the_fetch_result() {
        let mut c = core_with(cfg());
        c.request(1, "/race.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/race.jpg");
        let a = plan(&mut c);
        c.on_blob_ready(key, blob());
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        let planned = plan(&mut c);
        assert_eq!(decodes(&planned).len(), 1);
        assert!(generates(&planned).is_empty(), "the late miss must not trigger generation");
    }

    #[test]
    fn legacy_conversion_in_flight_holds_back_generation() {
        let mut c = core_with(cfg());
        c.request(1, "/old/img.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/old/img.jpg");
        c.legacy_started(key);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        let planned = plan(&mut c);
        assert!(generates(&planned).is_empty());
        assert_eq!(c.depths().legacy_wait, 1);
        c.on_blob_ready(key, blob());
        assert_eq!(decodes(&plan(&mut c)).len(), 1);
    }

    #[test]
    fn legacy_conversion_that_fails_falls_back_to_generation() {
        let mut c = core_with(cfg());
        c.request(1, "/old/bad.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/old/bad.jpg");
        c.legacy_started(key);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), vec![key])));
        plan(&mut c);
        c.legacy_finished(key);
        let b = plan(&mut c);
        assert_eq!(fetch_keys(&b), vec![key], "the want asks the store once more");
        c.on_fetch_done(batch_id(&b), Ok((Vec::new(), vec![key])));
        assert_eq!(generates(&plan(&mut c)), vec![key]);
    }

    #[test]
    fn store_errors_back_off_and_retry_without_losing_the_want() {
        let mut c = core_with(cfg());
        c.request(1, "/e.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Err("db down".into()));
        assert_eq!(c.stats().fetch_errors, 1);
        assert!(plan(&mut c).is_empty(), "no retry before the backoff");
        c.request(1, "/e.jpg", Tier::Thumb, 0.0);
        let before = c.now_ms();
        c.begin_frame(before + 100);
        assert!(plan(&mut c).is_empty());
        c.begin_frame(before + 400);
        c.request(1, "/e.jpg", Tier::Thumb, 0.0);
        let b = plan(&mut c);
        assert_eq!(fetch_keys(&b).len(), 1);
        c.on_fetch_done(batch_id(&b), Err("still down".into()));
        c.begin_frame(before + 400 + 300);
        c.request(1, "/e.jpg", Tier::Thumb, 0.0);
        assert!(plan(&mut c).is_empty(), "the second failure doubles the delay");
    }

    #[test]
    fn preview_tier_skips_the_store_and_decodes_from_the_file() {
        let mut c = core_with(cfg());
        assert!(matches!(c.request(1, "/big.jpg", Tier::Preview, 0.0), Lookup::Pending));
        let planned = plan(&mut c);
        assert!(fetch_keys(&planned).is_empty());
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 1);
        assert!(matches!(jobs[0].src, DecodeSrc::File));
        assert_eq!(jobs[0].key.tier, Tier::Preview);
        assert_eq!(c.complete_decode(jobs[0].key, DecodeResult::Image { cost: 5_000_000 }, |_| Some(1)), DecodeEnd::Inserted);
        assert!(matches!(c.request(1, "/big.jpg", Tier::Preview, 0.0), Lookup::Ready(1)));
        assert!(!c.is_cached("/big.jpg", Tier::Thumb), "tiers are separate entries");
    }

    #[test]
    fn a_failed_decode_is_negative_cached_and_releases_the_claim() {
        let mut c = core_with(cfg());
        c.request(1, "/broken.jpg", Tier::Preview, 0.0);
        let planned = plan(&mut c);
        let job = decodes(&planned)[0];
        assert_eq!(c.complete_decode(job.key, DecodeResult::Failed, |_| None), DecodeEnd::Failed);
        assert_eq!(c.claims_in_flight(), 0);
        assert!(matches!(c.request(1, "/broken.jpg", Tier::Preview, 0.0), Lookup::Failed));
        assert_eq!(c.decode_inflight(), 0);
        assert_eq!(c.stats().decode_failed, 1);
    }

    #[test]
    fn a_regenerated_thumbnail_updates_a_drawn_texture_in_place() {
        let mut c = core_with(cfg());
        let key = ThumbKey::from_path("/regen.jpg");
        c.request(1, "/regen.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(key, blob())], Vec::new())));
        let job_planned = plan(&mut c);
        finish_ok(&mut c, decodes(&job_planned)[0]);
        c.request(1, "/regen.jpg", Tier::Thumb, 0.0);
        c.on_blob_ready(key, blob());
        let planned = plan(&mut c);
        let jobs = decodes(&planned);
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].refresh);
        let mut touched = false;
        let end = c.complete_decode(jobs[0].key, DecodeResult::Image { cost: 2000 }, |existing| {
            touched = existing.is_some();
            if let Some(v) = existing {
                *v = 99;
            }
            None
        });
        assert_eq!(end, DecodeEnd::UpdatedInPlace);
        assert!(touched);
        assert!(matches!(c.request(1, "/regen.jpg", Tier::Thumb, 0.0), Lookup::Ready(99)));
        assert_eq!(c.cache_bytes(), 2000);
        assert_eq!(c.stats().uploads_in_place, 1);
    }

    #[test]
    fn a_regenerated_thumbnail_for_an_undrawn_texture_just_drops_the_old_one() {
        let mut c = core_with(cfg());
        let key = ThumbKey::from_path("/regen.jpg");
        c.request(1, "/regen.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(key, blob())], Vec::new())));
        let p = plan(&mut c);
        finish_ok(&mut c, decodes(&p)[0]);
        for _ in 0..5 {
            next_frame(&mut c, 16);
        }
        c.on_blob_ready(key, blob());
        assert!(!c.is_cached("/regen.jpg", Tier::Thumb));
        assert_eq!(c.stats().refreshes_forgotten, 1);
        assert_eq!(c.cache_bytes(), 0);
    }

    #[test]
    fn eviction_keeps_the_budget_and_counts_thrash_on_refetch() {
        let mut config = cfg();
        config.budget_bytes = 2_500;
        config.claim_budget = 100;
        let mut c = core_with(config);
        for i in 0..4 {
            let p = format!("/b/{i}.jpg");
            c.request(1, &p, Tier::Thumb, 0.0);
            let a = plan(&mut c);
            c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path(&p), blob())], Vec::new())));
            let planned = plan(&mut c);
            finish_ok(&mut c, decodes(&planned)[0]);
            next_frame(&mut c, 16);
            next_frame(&mut c, 16);
        }
        assert!(c.cache_bytes() <= 2_500);
        assert!(c.cache_stats().evictions >= 2);
        assert!(matches!(c.request(1, "/b/0.jpg", Tier::Thumb, 0.0), Lookup::Pending));
        assert_eq!(c.cache_stats().thrash, 1);
    }

    #[test]
    fn an_ahead_request_wants_an_uncached_texture_without_a_second_want() {
        let mut c = core_with(cfg());
        assert!(!c.request_ahead(1, "/ahead.jpg", Tier::Thumb, 900.0));
        assert!(!c.request_ahead(1, "/ahead.jpg", Tier::Thumb, 400.0));
        assert!(c.is_wanted("/ahead.jpg", Tier::Thumb));
        assert_eq!(c.stats().wants_created, 1);
        assert_eq!(c.stats().ahead_requests, 2);
        assert_eq!(c.stats().requests, 0, "ahead requests are counted apart from draws");
        assert_eq!(fetch_keys(&plan(&mut c)), vec![ThumbKey::from_path("/ahead.jpg")]);
    }

    #[test]
    fn an_ahead_request_does_not_mark_a_cached_texture_as_drawn() {
        let mut c = core_with(cfg());
        c.request(1, "/c.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path("/c.jpg"), blob())], Vec::new())));
        let planned = plan(&mut c);
        finish_ok(&mut c, decodes(&planned)[0]);
        for _ in 0..3 {
            next_frame(&mut c, 16);
        }
        assert_eq!(c.on_screen_count(), 0, "nothing drew it for a while");
        assert!(c.request_ahead(1, "/c.jpg", Tier::Thumb, 10.0), "it is cached");
        assert_eq!(c.on_screen_count(), 0, "an ahead request leaves it evictable");
        assert!(matches!(c.request(1, "/c.jpg", Tier::Thumb, 0.0), Lookup::Ready(_)));
        assert_eq!(c.on_screen_count(), 1, "drawing stamps it");
    }

    #[test]
    fn ahead_wants_expire_like_any_other_when_the_viewport_moves_on() {
        let mut config = cfg();
        config.stale_frames = 2;
        let mut c = core_with(config);
        c.request_ahead(1, "/soon.jpg", Tier::Thumb, 100.0);
        for _ in 0..4 {
            next_frame(&mut c, 16);
        }
        assert!(!c.is_wanted("/soon.jpg", Tier::Thumb));
        assert_eq!(c.stats().cancelled_stale, 1);
    }

    #[test]
    fn forget_path_drops_textures_wants_and_negative_entries() {
        let mut c = core_with(cfg());
        c.request(1, "/f.jpg", Tier::Preview, 0.0);
        let planned = plan(&mut c);
        c.complete_decode(decodes(&planned)[0].key, DecodeResult::Image { cost: 10 }, |_| Some(3));
        c.request(1, "/f.jpg", Tier::Thumb, 0.0);
        c.forget_path("/f.jpg");
        assert_eq!(c.cache_len(), 0);
        assert_eq!(c.wants_len(), 0);
        assert_eq!(c.claims_in_flight(), 0);
    }

    #[test]
    fn flush_empties_the_cache_and_keeps_live_wants_going() {
        let mut c = core_with(cfg());
        c.request(1, "/keep.jpg", Tier::Thumb, 0.0);
        let key = ThumbKey::from_path("/keep.jpg");
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(key, blob())], Vec::new())));
        let p = plan(&mut c);
        finish_ok(&mut c, decodes(&p)[0]);
        c.request(1, "/other.jpg", Tier::Thumb, 0.0);
        let a = plan(&mut c);
        c.on_fetch_done(batch_id(&a), Ok((vec![(ThumbKey::from_path("/other.jpg"), blob())], Vec::new())));
        c.flush();
        assert_eq!(c.cache_len(), 0);
        assert_eq!(c.cache_bytes(), 0);
        assert_eq!(c.wants_len(), 1);
        assert_eq!(c.claims_in_flight(), 1);
        let planned = plan(&mut c);
        assert_eq!(decodes(&planned).len(), 1);
        assert_eq!(finish_ok(&mut c, decodes(&planned)[0]), DecodeEnd::Inserted);
        assert!(matches!(c.request(1, "/keep.jpg", Tier::Thumb, 0.0), Lookup::Pending), "flushed textures are fetched again on demand");
    }

    #[test]
    fn negative_cache_is_bounded() {
        let mut config = cfg();
        config.generatable = |_| false;
        config.claim_budget = 10_000;
        config.negative_cap = 50;
        config.fetch_batch_max = 10_000;
        let mut c = core_with(config);
        for i in 0..200 {
            c.request(1, &format!("/n/{i}.jpg"), Tier::Thumb, 0.0);
        }
        let a = plan(&mut c);
        let keys = fetch_keys(&a);
        c.on_fetch_done(batch_id(&a), Ok((Vec::new(), keys)));
        assert!(c.depths().negative <= 50);
    }

    fn sim_frame(c: &mut TestCore, first: usize, visible: usize) {
        let now = c.now_ms();
        c.begin_frame(now + 16);
        for i in first..first + visible {
            c.request(1, &format!("/scroll/{i}.jpg"), Tier::Thumb, (i - first) as f32);
        }
        let mut actions = Vec::new();
        c.plan(&mut actions);
        let mut decodes_to_run = Vec::new();
        for a in actions {
            match a {
                Action::Fetch { batch, keys } => c.on_fetch_done(batch, Ok((keys.iter().map(|k| (*k, blob())).collect(), Vec::new()))),
                Action::Decode(job) => decodes_to_run.push(job),
                Action::Generate { .. } => {}
            }
        }
        let mut more = Vec::new();
        c.plan(&mut more);
        decodes_to_run.extend(more.into_iter().filter_map(|a| if let Action::Decode(j) = a { Some(j) } else { None }));
        for job in decodes_to_run {
            c.complete_decode(job.key, DecodeResult::Image { cost: 1000 }, |_| Some(1));
        }
    }

    #[test]
    fn scrolling_back_over_recent_rows_hits_the_cache_and_older_rows_are_fetched_again() {
        let mut config = cfg();
        config.budget_bytes = 60_000;
        config.claim_budget = 100;
        let mut c = core_with(config);
        let visible = 9;
        for first in (0..300).step_by(3) {
            sim_frame(&mut c, first, visible);
        }
        let down = c.cache_stats();
        assert_eq!(down.reuse_hits, 0, "a first pass over new rows reuses nothing");
        assert!(c.cache_bytes() <= 60_000);
        let fetched_down = down.claims_granted;
        assert!(fetched_down >= 300, "every row was fetched once: {fetched_down}");
        for first in (200..=290).rev().step_by(3) {
            sim_frame(&mut c, first, visible);
        }
        let back = c.cache_stats();
        let reused = back.reuse_hits - down.reuse_hits;
        let refetched = back.claims_granted - down.claims_granted;
        assert!(reused >= 40, "rows still inside the budget come back from the cache: {reused}");
        assert!(refetched > 0 && refetched < 100, "older rows are fetched again: {refetched}");
        assert!(back.reuse_rate() > 0.0);
    }

    #[test]
    fn default_generatable_accepts_media_files_and_rejects_virtual_paths() {
        assert!(default_generatable("/a/b/photo.JPG"));
        assert!(default_generatable("C:\\pics\\clip.mp4"));
        assert!(!default_generatable("zip:///a.zip!/x.jpg"));
        assert!(!default_generatable("cat://nature"));
        assert!(!default_generatable("/a/b/readme.txt"));
        assert!(!default_generatable("/a/b/noext"));
        assert!(!default_generatable("/dir.v2/noext"));
    }
}
