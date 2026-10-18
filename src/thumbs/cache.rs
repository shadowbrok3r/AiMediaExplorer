//! Byte-budgeted cache policy: insertion-order eviction with a second chance for entries drawn this frame or the last, single-shot fetch claims with a per-frame budget, and in-place value updates.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub const DEFAULT_BUDGET_BYTES: usize = 256 * 1024 * 1024;
pub const DEFAULT_CLAIM_BUDGET: usize = 12;
const EVICTED_MEMORY: usize = 2048;
const THRASH_FRAMES: u64 = 240;
const ORDER_SLACK: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub reuse_hits: u64,
    pub inserts: u64,
    pub in_place_updates: u64,
    pub evictions: u64,
    pub second_chances: u64,
    pub claims_granted: u64,
    pub claims_duplicate: u64,
    pub claims_over_budget: u64,
    pub thrash: u64,
}

impl CacheStats {
    /// Cached reads over cached reads plus fetches started.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.claims_granted;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }

    /// Share of cells that reappeared after being off screen and found their texture still cached.
    pub fn reuse_rate(&self) -> f32 {
        let total = self.reuse_hits + self.claims_granted;
        if total == 0 { 0.0 } else { self.reuse_hits as f32 / total as f32 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    Granted,
    Cached,
    Pending,
    OverBudget,
}

struct Entry<V> {
    value: V,
    cost: usize,
    last_seen: u64,
    stamp: u64,
    spared_pass: u64,
}

pub struct ThumbCache<K, V> {
    map: HashMap<K, Entry<V>>,
    order: VecDeque<(K, u64)>,
    pending: HashMap<K, ()>,
    evicted: HashMap<K, u64>,
    bytes: usize,
    budget: usize,
    frame: u64,
    claim_budget: usize,
    claims_this_frame: usize,
    next_stamp: u64,
    pass: u64,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone, V> Default for ThumbCache<K, V> {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET_BYTES)
    }
}

impl<K: Eq + Hash + Clone, V> ThumbCache<K, V> {
    pub fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            pending: HashMap::new(),
            evicted: HashMap::new(),
            bytes: 0,
            budget,
            frame: 0,
            claim_budget: DEFAULT_CLAIM_BUDGET,
            claims_this_frame: 0,
            next_stamp: 1,
            pass: 0,
            stats: CacheStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn claim_budget(&self) -> usize {
        self.claim_budget
    }

    pub fn set_claim_budget(&mut self, per_frame: usize) {
        self.claim_budget = per_frame;
    }

    pub fn claims_left(&self) -> usize {
        self.claim_budget.saturating_sub(self.claims_this_frame)
    }

    /// Starts a frame: entries read this frame or the previous one count as on screen, and the claim budget refills.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.claims_this_frame = 0;
    }

    /// Reads an entry and stamps it as drawn this frame.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let e = self.map.get_mut(key)?;
        if self.frame.wrapping_sub(e.last_seen) > 1 {
            self.stats.reuse_hits += 1;
        }
        e.last_seen = self.frame;
        self.stats.hits += 1;
        Some(&e.value)
    }

    /// Reads an entry without stamping it or counting a hit.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|e| &e.value)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Whether `key` was drawn this frame or the last.
    pub fn is_on_screen(&self, key: &K) -> bool {
        self.map.get(key).is_some_and(|e| self.on_screen(e.last_seen))
    }

    fn on_screen(&self, last_seen: u64) -> bool {
        self.frame.wrapping_sub(last_seen) <= 1
    }

    pub fn on_screen_count(&self) -> usize {
        self.map.values().filter(|e| self.on_screen(e.last_seen)).count()
    }

    /// Claims the right to fetch `key`: granted once until released, inserted or forgotten, and at most the per-frame budget of times per frame.
    pub fn claim(&mut self, key: &K) -> Claim {
        if self.map.contains_key(key) {
            return Claim::Cached;
        }
        if self.pending.contains_key(key) {
            self.stats.claims_duplicate += 1;
            return Claim::Pending;
        }
        if self.claims_this_frame >= self.claim_budget {
            self.stats.claims_over_budget += 1;
            return Claim::OverBudget;
        }
        self.claims_this_frame += 1;
        self.pending.insert(key.clone(), ());
        self.stats.claims_granted += 1;
        if self.evicted.remove(key).is_some_and(|at| self.frame.wrapping_sub(at) <= THRASH_FRAMES) {
            self.stats.thrash += 1;
        }
        Claim::Granted
    }

    /// Records a claim without spending the per-frame budget.
    pub fn mark_claimed(&mut self, key: K) {
        self.pending.insert(key, ());
    }

    pub fn is_claimed(&self, key: &K) -> bool {
        self.pending.contains_key(key)
    }

    /// Drops a claim so the key can be claimed again.
    pub fn release(&mut self, key: &K) -> bool {
        self.pending.remove(key).is_some()
    }

    /// Drops every claim.
    pub fn reset_pending(&mut self) {
        self.pending.clear();
    }

    /// Inserts a value, releasing its claim; an existing entry keeps its place in the eviction order and takes the new cost.
    pub fn insert(&mut self, key: K, value: V, cost: usize) {
        self.pending.remove(&key);
        if let Some(e) = self.map.get_mut(&key) {
            self.bytes = self.bytes.saturating_sub(e.cost) + cost;
            e.cost = cost;
            e.value = value;
            e.last_seen = self.frame;
            self.stats.in_place_updates += 1;
        } else {
            let stamp = self.next_stamp;
            self.next_stamp += 1;
            self.order.push_back((key.clone(), stamp));
            self.map.insert(key.clone(), Entry { value, cost, last_seen: self.frame, stamp, spared_pass: 0 });
            self.bytes += cost;
            self.stats.inserts += 1;
        }
        self.evict(Some(&key));
    }

    /// Runs `update` on the live value and re-costs it, returning false when the key is absent.
    pub fn update_in_place(&mut self, key: &K, cost: usize, update: impl FnOnce(&mut V)) -> bool {
        let Some(e) = self.map.get_mut(key) else { return false };
        update(&mut e.value);
        self.bytes = self.bytes.saturating_sub(e.cost) + cost;
        e.cost = cost;
        e.last_seen = self.frame;
        self.stats.in_place_updates += 1;
        self.evict(Some(key));
        true
    }

    /// Mutable access to a live value, stamped as drawn.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let frame = self.frame;
        self.map.get_mut(key).map(|e| {
            e.last_seen = frame;
            &mut e.value
        })
    }

    /// Changes the budget and evicts down to it.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.evict(None);
    }

    /// Drops one entry and its claim.
    pub fn forget(&mut self, key: &K) -> Option<V> {
        self.pending.remove(key);
        let e = self.map.remove(key)?;
        self.bytes = self.bytes.saturating_sub(e.cost);
        self.compact_order();
        Some(e.value)
    }

    /// Drops every entry and claim matching `pred`.
    pub fn forget_where(&mut self, mut pred: impl FnMut(&K) -> bool) -> usize {
        self.pending.retain(|k, _| !pred(k));
        let mut freed = 0usize;
        let mut dropped = 0usize;
        self.map.retain(|k, e| {
            if pred(k) {
                freed += e.cost;
                dropped += 1;
                false
            } else {
                true
            }
        });
        self.bytes = self.bytes.saturating_sub(freed);
        self.compact_order();
        dropped
    }

    /// Drops everything, claims included.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.pending.clear();
        self.bytes = 0;
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    fn compact_order(&mut self) {
        if self.order.len() > 2 * self.map.len() + ORDER_SLACK {
            let map = &self.map;
            self.order.retain(|(k, stamp)| map.get(k).is_some_and(|e| e.stamp == *stamp));
        }
    }

    fn remember_eviction(&mut self, key: K) {
        if self.evicted.len() >= EVICTED_MEMORY {
            let frame = self.frame;
            self.evicted.retain(|_, at| frame.wrapping_sub(*at) <= THRASH_FRAMES);
            if self.evicted.len() >= EVICTED_MEMORY {
                self.evicted.clear();
            }
        }
        self.evicted.insert(key, self.frame);
    }

    // Evicts oldest first, requeueing each on-screen entry once per pass and never evicting the newest.
    fn evict(&mut self, newest: Option<&K>) {
        if self.bytes <= self.budget {
            return;
        }
        self.pass = self.pass.wrapping_add(1);
        let pass = self.pass;
        let mut steps = 2 * self.order.len() + 2;
        while self.bytes > self.budget && self.map.len() > 1 && steps > 0 {
            steps -= 1;
            let Some((key, stamp)) = self.order.pop_front() else { break };
            let frame = self.frame;
            let Some(entry) = self.map.get_mut(&key) else { continue };
            if entry.stamp != stamp {
                continue;
            }
            let is_newest = newest.is_some_and(|n| *n == key);
            let drawn = frame.wrapping_sub(entry.last_seen) <= 1;
            if is_newest || (drawn && entry.spared_pass != pass) {
                if !is_newest {
                    entry.spared_pass = pass;
                    self.stats.second_chances += 1;
                }
                self.order.push_back((key, stamp));
                continue;
            }
            if let Some(gone) = self.map.remove(&key) {
                self.bytes = self.bytes.saturating_sub(gone.cost);
                self.stats.evictions += 1;
                self.remember_eviction(key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: usize = 64 * 1024 * 1024;

    fn cache() -> ThumbCache<String, u32> {
        ThumbCache::new(BUDGET)
    }

    #[test]
    fn claim_is_single_shot() {
        let mut c = cache();
        assert_eq!(c.claim(&"a#320".to_string()), Claim::Granted);
        assert_eq!(c.claim(&"a#320".to_string()), Claim::Pending);
        c.reset_pending();
        assert_eq!(c.claim(&"a#320".to_string()), Claim::Granted);
        c.insert("a#320".into(), 1, 10);
        assert_eq!(c.claim(&"a#320".to_string()), Claim::Cached);
        assert_eq!(c.pending_len(), 0, "inserting releases the claim");
    }

    #[test]
    fn claims_are_capped_per_frame_and_refill_next_frame() {
        let mut c = cache();
        c.begin_frame();
        let granted = (0..20).filter(|i| c.claim(&format!("k{i}")) == Claim::Granted).count();
        assert_eq!(granted, DEFAULT_CLAIM_BUDGET);
        assert_eq!(c.claims_left(), 0);
        assert_eq!(c.claim(&"k19".to_string()), Claim::OverBudget);
        assert!(!c.is_claimed(&"k19".to_string()), "a denied claim leaves no state behind");
        c.begin_frame();
        assert_eq!(c.claim(&"k19".to_string()), Claim::Granted);
        assert_eq!(c.claim(&"k0".to_string()), Claim::Pending, "earlier claims survive the frame boundary");
        assert_eq!(c.stats().claims_over_budget, 9);
    }

    #[test]
    fn release_makes_a_key_claimable_again() {
        let mut c = cache();
        assert_eq!(c.claim(&"x".to_string()), Claim::Granted);
        assert!(c.release(&"x".to_string()));
        assert!(!c.release(&"x".to_string()));
        assert_eq!(c.claim(&"x".to_string()), Claim::Granted);
    }

    #[test]
    fn a_visible_tile_outlives_one_that_scrolled_away() {
        let mut c = cache();
        let third = BUDGET / 3 + 1;
        c.insert("gone".into(), 1, third);
        c.insert("onscreen".into(), 2, third);
        c.begin_frame();
        c.begin_frame();
        assert!(c.get(&"onscreen".to_string()).is_some());
        c.insert("new".into(), 3, third);
        assert!(c.get(&"onscreen".to_string()).is_some(), "a tile being drawn must not be evicted");
        assert!(c.get(&"gone".to_string()).is_none(), "the tile that scrolled away pays");
        assert!(c.get(&"new".to_string()).is_some());
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn a_drawn_tile_survives_a_whole_drain_of_inserts() {
        let mut c = cache();
        let tile = BUDGET / 16;
        c.begin_frame();
        for i in 0..12 {
            c.insert(format!("stale{i}"), 0, tile);
        }
        for _ in 0..4 {
            c.begin_frame();
        }
        for i in 0..4 {
            c.insert(format!("live{i}"), 0, tile);
        }
        for i in 0..4 {
            assert!(c.get(&format!("live{i}")).is_some());
        }
        c.begin_frame();
        for i in 0..8 {
            c.insert(format!("new{i}"), 0, tile);
        }
        for i in 0..4 {
            assert!(c.peek(&format!("live{i}")).is_some(), "live{i} was evicted mid-drain");
        }
        assert!(c.peek(&"stale0".to_string()).is_none(), "scrolled-away tiles pay");
        assert!(c.bytes() <= BUDGET);
    }

    #[test]
    fn a_tile_stops_being_protected_once_it_leaves_the_screen() {
        let mut c = cache();
        let third = BUDGET / 3 + 1;
        c.begin_frame();
        c.insert("stale".into(), 0, third);
        assert!(c.get(&"stale".to_string()).is_some());
        for _ in 0..4 {
            c.begin_frame();
        }
        c.insert("a".into(), 0, third);
        c.insert("b".into(), 0, third);
        assert!(c.peek(&"stale".to_string()).is_none());
    }

    #[test]
    fn eviction_terminates_when_everything_is_on_screen() {
        let mut c = cache();
        let third = BUDGET / 3 + 1;
        c.insert("a".into(), 0, third);
        c.insert("b".into(), 0, third);
        assert!(c.get(&"a".to_string()).is_some());
        assert!(c.get(&"b".to_string()).is_some());
        c.insert("c".into(), 0, third);
        assert_eq!(c.len(), 2, "the budget still has to be enforced");
        assert!(c.peek(&"c".to_string()).is_some(), "the entry just inserted is never the one thrown away");
        assert!(c.bytes() <= BUDGET);
    }

    #[test]
    fn eviction_is_by_bytes_not_count() {
        let mut c = cache();
        for i in 0..10 {
            c.insert(format!("k{i}"), 0, 4 * 1024 * 1024);
        }
        assert_eq!(c.len(), 10);
        c.insert("big".into(), 0, BUDGET);
        assert!(c.len() < 11);
        assert!(c.peek(&"big".to_string()).is_some(), "the newest entry must survive");
        assert!(c.peek(&"k0".to_string()).is_none(), "the oldest entry goes first");
    }

    #[test]
    fn forget_drops_matching_entries_and_claims() {
        let mut c = cache();
        c.insert("u1/a/1.png#320".into(), 0, 1024);
        c.insert("u1/a/1.png#1024".into(), 0, 4096);
        c.insert("u1/a/10.png#320".into(), 0, 1024);
        c.claim(&"u1/a/1.png#512".to_string());
        let dropped = c.forget_where(|k| k.starts_with("u1/a/1.png#"));
        assert_eq!(dropped, 2);
        assert!(c.peek(&"u1/a/1.png#320".to_string()).is_none());
        assert!(c.peek(&"u1/a/10.png#320".to_string()).is_some());
        assert_eq!(c.bytes(), 1024, "freed bytes return to the budget");
        assert_eq!(c.claim(&"u1/a/1.png#512".to_string()), Claim::Granted);
    }

    #[test]
    fn forget_one_entry_returns_its_bytes() {
        let mut c = cache();
        c.insert("a".into(), 7, 100);
        c.insert("b".into(), 8, 200);
        assert_eq!(c.forget(&"a".to_string()), Some(7));
        assert_eq!(c.forget(&"a".to_string()), None);
        assert_eq!(c.bytes(), 200);
    }

    #[test]
    fn reinsert_does_not_leak_budget() {
        let mut c = cache();
        for i in 0..50 {
            c.insert("same".into(), i, 4 * 1024 * 1024);
        }
        assert_eq!(c.len(), 1);
        assert_eq!(c.bytes(), 4 * 1024 * 1024);
        assert_eq!(c.peek(&"same".to_string()), Some(&49));
        c.insert("same".into(), 0, 1024);
        assert_eq!(c.bytes(), 1024, "a replacement takes the new cost");
    }

    #[test]
    fn update_in_place_keeps_the_value_and_recosts_it() {
        let mut c = cache();
        c.insert("a".into(), 1, 1000);
        c.insert("b".into(), 1, 1000);
        assert!(c.update_in_place(&"a".to_string(), 5000, |v| *v += 10));
        assert_eq!(c.peek(&"a".to_string()), Some(&11));
        assert_eq!(c.bytes(), 6000);
        assert!(!c.update_in_place(&"missing".to_string(), 1, |_| {}));
        assert_eq!(c.stats().in_place_updates, 1);
        assert_eq!(c.stats().inserts, 2);
    }

    #[test]
    fn growing_in_place_can_evict_others_but_never_itself() {
        let mut c: ThumbCache<String, u32> = ThumbCache::new(1000);
        c.insert("a".into(), 0, 400);
        c.insert("b".into(), 0, 400);
        assert!(c.update_in_place(&"b".to_string(), 900, |_| {}));
        assert!(c.peek(&"a".to_string()).is_none());
        assert!(c.peek(&"b".to_string()).is_some());
        assert_eq!(c.bytes(), 900);
    }

    #[test]
    fn shrinking_the_budget_evicts_immediately() {
        let mut c = cache();
        for i in 0..8 {
            c.insert(format!("k{i}"), 0, 1000);
        }
        c.set_budget(3500);
        assert_eq!(c.len(), 3);
        assert!(c.bytes() <= 3500);
        assert!(c.peek(&"k7".to_string()).is_some());
    }

    #[test]
    fn counters_track_hits_reuse_fetches_and_evictions() {
        let mut c: ThumbCache<String, u32> = ThumbCache::new(2000);
        c.insert("a".into(), 0, 1000);
        c.insert("b".into(), 0, 1000);
        assert!(c.get(&"a".to_string()).is_some());
        assert!(c.get(&"zzz".to_string()).is_none());
        c.begin_frame();
        c.begin_frame();
        assert!(c.get(&"a".to_string()).is_some());
        c.insert("c".into(), 0, 1000);
        assert_eq!(c.claim(&"d".to_string()), Claim::Granted);
        let s = c.stats();
        assert_eq!((s.hits, s.reuse_hits, s.inserts, s.evictions, s.claims_granted), (2, 1, 3, 1, 1));
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-6);
        assert!((s.reuse_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_cell_drawn_every_frame_is_one_reuse_at_most() {
        let mut c = cache();
        c.insert("a".into(), 0, 10);
        for _ in 0..20 {
            c.begin_frame();
            assert!(c.get(&"a".to_string()).is_some());
        }
        assert_eq!(c.stats().hits, 20);
        assert_eq!(c.stats().reuse_hits, 0, "continuous drawing is not reuse");
        for _ in 0..5 {
            c.begin_frame();
        }
        assert!(c.get(&"a".to_string()).is_some());
        assert_eq!(c.stats().reuse_hits, 1);
    }

    #[test]
    fn evicting_then_claiming_again_counts_as_thrash() {
        let mut c: ThumbCache<String, u32> = ThumbCache::new(1000);
        c.insert("a".into(), 0, 600);
        c.begin_frame();
        c.begin_frame();
        c.insert("b".into(), 0, 600);
        assert!(c.peek(&"a".to_string()).is_none());
        c.begin_frame();
        assert_eq!(c.claim(&"a".to_string()), Claim::Granted);
        assert_eq!(c.stats().thrash, 1);
        for _ in 0..(THRASH_FRAMES + 2) {
            c.begin_frame();
        }
        c.insert("c".into(), 0, 600);
        c.begin_frame();
        c.begin_frame();
        c.insert("d".into(), 0, 600);
        for _ in 0..(THRASH_FRAMES + 2) {
            c.begin_frame();
        }
        assert_eq!(c.claim(&"c".to_string()), Claim::Granted);
        assert_eq!(c.stats().thrash, 1, "a long-evicted key is not thrash");
    }

    #[test]
    fn churn_does_not_grow_the_order_queue_without_bound() {
        let mut c: ThumbCache<u64, u32> = ThumbCache::new(10_000);
        for i in 0..5_000u64 {
            c.insert(i, 0, 100);
            if i % 3 == 0 {
                c.forget(&(i.saturating_sub(5)));
            }
        }
        assert!(c.order.len() <= 2 * c.len() + ORDER_SLACK + 200, "order={} len={}", c.order.len(), c.len());
        assert!(c.bytes() <= 10_000);
    }

    #[test]
    fn without_frame_calls_eviction_degrades_to_insertion_order() {
        let mut c: ThumbCache<String, u32> = ThumbCache::new(2500);
        for i in 0..5 {
            c.insert(format!("k{i}"), 0, 1000);
        }
        assert_eq!(c.len(), 2);
        assert!(c.peek(&"k4".to_string()).is_some());
        assert!(c.peek(&"k3".to_string()).is_some());
    }

    #[test]
    fn on_screen_covers_this_frame_and_the_last_only() {
        let mut c = cache();
        c.insert("a".into(), 0, 10);
        assert!(c.is_on_screen(&"a".to_string()));
        c.begin_frame();
        assert!(c.is_on_screen(&"a".to_string()), "still on screen one frame later");
        c.begin_frame();
        assert!(!c.is_on_screen(&"a".to_string()));
        assert_eq!(c.on_screen_count(), 0);
        assert!(c.get(&"a".to_string()).is_some());
        assert!(c.is_on_screen(&"a".to_string()));
        assert_eq!(c.on_screen_count(), 1);
    }
}
