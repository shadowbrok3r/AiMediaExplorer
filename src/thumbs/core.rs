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
mod tests;
