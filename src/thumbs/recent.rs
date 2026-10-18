//! Encoded thumbnails that were just produced or fetched: serves reads while database writes are still queued, and remembers which keys are known to exist.

use super::key::ThumbKey;
use super::store::{Blob, BlobStamp};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const RECENT_SOFT_BYTES: usize = 32 * 1024 * 1024;
pub const RECENT_HARD_BYTES: usize = 128 * 1024 * 1024;
const PRESENCE_CAP: usize = 2_000_000;
const INSERT_EVICT_STEPS: usize = 64;

struct RecentEntry {
    blob: Arc<Blob>,
    stamp: BlobStamp,
    path: String,
    pinned: bool,
    queued: bool,
    seq: u64,
}

/// Byte-bounded map of encoded blobs. Entries not yet written to the store are pinned and only dropped past the hard limit.
pub struct Recent {
    map: HashMap<ThumbKey, RecentEntry>,
    order: VecDeque<(ThumbKey, u64)>,
    bytes: usize,
    soft: usize,
    hard: usize,
    seq: u64,
    dropped_pinned: u64,
}

impl Recent {
    pub fn new(soft: usize, hard: usize) -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), bytes: 0, soft, hard: hard.max(soft), seq: 0, dropped_pinned: 0 }
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

    pub fn pinned(&self) -> usize {
        self.map.values().filter(|e| e.pinned).count()
    }

    pub fn dropped_pinned(&self) -> u64 {
        self.dropped_pinned
    }

    pub fn get(&self, key: &ThumbKey) -> Option<Arc<Blob>> {
        self.map.get(key).map(|e| e.blob.clone())
    }

    pub fn stamp(&self, key: &ThumbKey) -> Option<BlobStamp> {
        self.map.get(key).map(|e| e.stamp)
    }

    pub fn contains(&self, key: &ThumbKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn insert(&mut self, key: ThumbKey, blob: Arc<Blob>, stamp: BlobStamp, pinned: bool) {
        self.insert_entry(key, String::new(), blob, stamp, pinned);
    }

    /// Inserts a blob whose database write is still owed; it stays readable and is handed out by [`Self::take_unqueued`] until written.
    pub fn insert_pending(&mut self, key: ThumbKey, path: &str, blob: Arc<Blob>, stamp: BlobStamp) {
        self.insert_entry(key, path.to_string(), blob, stamp, true);
    }

    fn insert_entry(&mut self, key: ThumbKey, path: String, blob: Arc<Blob>, stamp: BlobStamp, pinned: bool) {
        self.seq += 1;
        let cost = blob.bytes.len();
        if let Some(old) = self.map.insert(key, RecentEntry { blob, stamp, path, pinned, queued: false, seq: self.seq }) {
            self.bytes = self.bytes.saturating_sub(old.blob.bytes.len());
        }
        self.bytes += cost;
        self.order.push_back((key, self.seq));
        self.evict(INSERT_EVICT_STEPS);
    }

    /// Records that the write for `key` sits in the writer queue.
    pub fn mark_queued(&mut self, key: &ThumbKey) {
        if let Some(e) = self.map.get_mut(key) {
            e.queued = true;
        }
    }

    /// Returns failed writes to the pool [`Self::take_unqueued`] draws from.
    pub fn mark_unqueued(&mut self, keys: &[ThumbKey]) {
        for k in keys {
            if let Some(e) = self.map.get_mut(k) {
                e.queued = false;
            }
        }
    }

    /// Pending writes that are not in the writer queue, oldest first, marked queued.
    pub fn take_unqueued(&mut self, limit: usize) -> Vec<(ThumbKey, String, Arc<Blob>, BlobStamp)> {
        let mut out = Vec::new();
        for (key, seq) in &self.order {
            if out.len() >= limit {
                break;
            }
            if let Some(e) = self.map.get_mut(key) {
                if e.seq == *seq && e.pinned && !e.queued && !e.path.is_empty() {
                    e.queued = true;
                    out.push((*key, e.path.clone(), e.blob.clone(), e.stamp));
                }
            }
        }
        out
    }

    pub fn unqueued(&self) -> usize {
        self.map.values().filter(|e| e.pinned && !e.queued && !e.path.is_empty()).count()
    }

    /// Unpins entries whose database write finished.
    pub fn mark_written(&mut self, keys: &[ThumbKey]) {
        for k in keys {
            if let Some(e) = self.map.get_mut(k) {
                e.pinned = false;
            }
        }
        self.evict(usize::MAX);
    }

    pub fn remove(&mut self, key: &ThumbKey) {
        if let Some(e) = self.map.remove(key) {
            self.bytes = self.bytes.saturating_sub(e.blob.bytes.len());
        }
    }

    pub fn clear_unpinned(&mut self) {
        let mut freed = 0usize;
        self.map.retain(|_, e| {
            if e.pinned {
                true
            } else {
                freed += e.blob.bytes.len();
                false
            }
        });
        self.bytes = self.bytes.saturating_sub(freed);
        self.order.clear();
        for (k, e) in &self.map {
            self.order.push_back((*k, e.seq));
        }
    }

    fn evict(&mut self, max_steps: usize) {
        if self.order.len() > 2 * self.map.len() + 64 {
            let map = &self.map;
            self.order.retain(|(k, seq)| map.get(k).is_some_and(|e| e.seq == *seq));
        }
        let mut steps = max_steps.min(self.order.len() + 1);
        while self.bytes > self.soft && steps > 0 {
            steps -= 1;
            let Some((key, seq)) = self.order.pop_front() else { break };
            let Some(entry) = self.map.get(&key) else { continue };
            if entry.seq != seq {
                continue;
            }
            if entry.pinned && self.bytes <= self.hard {
                self.order.push_back((key, seq));
                continue;
            }
            if entry.pinned {
                self.dropped_pinned += 1;
            }
            if let Some(gone) = self.map.remove(&key) {
                self.bytes = self.bytes.saturating_sub(gone.blob.bytes.len());
            }
        }
    }
}

/// Keys known to have a stored thumbnail.
#[derive(Default)]
pub struct Presence {
    keys: HashSet<ThumbKey>,
}

impl Presence {
    pub fn insert(&mut self, key: ThumbKey) -> bool {
        if self.keys.len() >= PRESENCE_CAP {
            self.keys.clear();
        }
        self.keys.insert(key)
    }

    pub fn contains(&self, key: &ThumbKey) -> bool {
        self.keys.contains(key)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::store::BlobFormat;

    fn blob(len: usize) -> Arc<Blob> {
        Arc::new(Blob { format: BlobFormat::Jpeg, w: 1, h: 1, bytes: vec![0; len] })
    }

    fn key(i: u32) -> ThumbKey {
        ThumbKey::from_path(&format!("/p/{i}.jpg"))
    }

    #[test]
    fn unpinned_entries_are_evicted_oldest_first_past_the_soft_limit() {
        let mut r = Recent::new(1000, 4000);
        for i in 0..5 {
            r.insert(key(i), blob(400), BlobStamp::default(), false);
        }
        assert!(r.bytes() <= 1000);
        assert!(!r.contains(&key(0)));
        assert!(r.contains(&key(4)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn pinned_entries_survive_the_soft_limit_until_written() {
        let mut r = Recent::new(1000, 100_000);
        for i in 0..5 {
            r.insert(key(i), blob(400), BlobStamp::default(), true);
        }
        assert_eq!(r.len(), 5, "pending writes must stay readable");
        assert_eq!(r.pinned(), 5);
        r.mark_written(&[key(0), key(1), key(2)]);
        assert!(r.bytes() <= 1000 || r.pinned() == r.len(), "bytes={}", r.bytes());
        assert!(r.contains(&key(3)) && r.contains(&key(4)), "still pinned");
        assert!(!r.contains(&key(0)));
    }

    #[test]
    fn the_hard_limit_drops_pinned_entries_oldest_first() {
        let mut r = Recent::new(500, 1200);
        for i in 0..6 {
            r.insert(key(i), blob(400), BlobStamp::default(), true);
        }
        assert!(r.bytes() <= 1200);
        assert!(r.dropped_pinned() >= 3);
        assert!(r.contains(&key(5)));
        assert!(!r.contains(&key(0)));
    }

    #[test]
    fn replacing_a_key_does_not_double_count() {
        let mut r = Recent::new(10_000, 20_000);
        r.insert(key(1), blob(300), BlobStamp::default(), true);
        r.insert(key(1), blob(500), BlobStamp { src_modified: Some(1), src_size: Some(2) }, true);
        assert_eq!(r.bytes(), 500);
        assert_eq!(r.len(), 1);
        assert_eq!(r.stamp(&key(1)).unwrap().src_size, Some(2));
        r.remove(&key(1));
        assert_eq!(r.bytes(), 0);
    }

    #[test]
    fn order_queue_stays_bounded_under_churn() {
        let mut r = Recent::new(100_000, 200_000);
        for i in 0..10_000u32 {
            r.insert(key(i % 50), blob(100), BlobStamp::default(), false);
        }
        assert!(r.order.len() <= 2 * r.len() + 66, "order={}", r.order.len());
    }

    #[test]
    fn pending_writes_are_handed_out_once_until_returned() {
        let mut r = Recent::new(10_000, 20_000);
        r.insert_pending(key(1), "/p/1.jpg", blob(100), BlobStamp::default());
        r.insert_pending(key(2), "/p/2.jpg", blob(100), BlobStamp::default());
        r.insert_pending(key(3), "/p/3.jpg", blob(100), BlobStamp::default());
        r.mark_queued(&key(2));
        assert_eq!(r.unqueued(), 2);
        let first = r.take_unqueued(1);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].1, "/p/1.jpg", "oldest first");
        let rest = r.take_unqueued(10);
        assert_eq!(rest.iter().map(|e| e.1.as_str()).collect::<Vec<_>>(), ["/p/3.jpg"]);
        assert!(r.take_unqueued(10).is_empty());
        r.mark_unqueued(&[key(1), key(2)]);
        assert_eq!(r.unqueued(), 2);
        r.mark_written(&[key(1)]);
        assert_eq!(r.unqueued(), 1, "written entries are not offered again");
    }

    #[test]
    fn unpinned_and_pathless_entries_are_never_offered() {
        let mut r = Recent::new(10_000, 20_000);
        r.insert(key(1), blob(10), BlobStamp::default(), true);
        r.insert(key(2), blob(10), BlobStamp::default(), false);
        assert!(r.take_unqueued(10).is_empty());
    }

    #[test]
    fn clear_unpinned_keeps_pending_writes() {
        let mut r = Recent::new(10_000, 20_000);
        r.insert(key(1), blob(100), BlobStamp::default(), true);
        r.insert(key(2), blob(100), BlobStamp::default(), false);
        r.clear_unpinned();
        assert!(r.contains(&key(1)) && !r.contains(&key(2)));
        assert_eq!(r.bytes(), 100);
    }

    #[test]
    fn presence_is_bounded_and_answers_membership() {
        let mut p = Presence::default();
        assert!(p.insert(key(1)));
        assert!(!p.insert(key(1)));
        assert!(p.contains(&key(1)) && !p.contains(&key(2)));
        assert_eq!(p.len(), 1);
        p.clear();
        assert!(p.is_empty());
    }
}
