//! Numbers about the focused view, recomputed only when its change signature moves.

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shortest time between recomputes while rows are still streaming in.
pub const STREAMING_THROTTLE: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    Size,
    Dates,
    SkipIcons,
    Search,
    HiddenTypes,
    Other,
}

/// An active filter narrowing the view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterChip {
    pub kind: FilterKind,
    pub label: String,
}

/// Counts and sizes of the focused view.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewStats {
    /// Rows the view shows.
    pub items: usize,
    /// Rows the filters hide.
    pub hidden: usize,
    pub images: usize,
    pub videos: usize,
    pub dirs: usize,
    pub total_bytes: u64,
    pub selected: usize,
    pub selected_bytes: u64,
    pub filters: Vec<FilterChip>,
}

/// Cheap fingerprint of everything [`ViewStats`] depends on.
pub type Signature = u64;

/// The last computed stats with the signature they were computed for.
#[derive(Clone, Default)]
pub struct ViewStatsCache {
    signature: Option<Signature>,
    at: Option<Instant>,
    stats: Arc<ViewStats>,
    /// How often the stats were recomputed; for tests and the Perf window.
    pub computes: u64,
}

/// What the cache answered.
pub struct CacheAnswer {
    pub stats: Arc<ViewStats>,
    /// Frame to repaint after when the answer is older than the view.
    pub repaint_after: Option<Duration>,
}

impl ViewStatsCache {
    /// The stats for `signature`; `compute` runs only when the signature changed (and, while `streaming`, not more often than [`STREAMING_THROTTLE`]).
    pub fn get(&mut self, now: Instant, signature: Signature, streaming: bool, compute: impl FnOnce() -> ViewStats) -> CacheAnswer {
        let current = self.signature == Some(signature);
        if current {
            return CacheAnswer { stats: self.stats.clone(), repaint_after: None };
        }
        if streaming {
            if let Some(at) = self.at {
                let age = now.saturating_duration_since(at);
                if age < STREAMING_THROTTLE {
                    return CacheAnswer { stats: self.stats.clone(), repaint_after: Some(STREAMING_THROTTLE - age) };
                }
            }
        }
        self.stats = Arc::new(compute());
        self.signature = Some(signature);
        self.at = Some(now);
        self.computes += 1;
        CacheAnswer { stats: self.stats.clone(), repaint_after: None }
    }

    pub fn stats(&self) -> &Arc<ViewStats> {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn stats(items: usize) -> ViewStats {
        ViewStats { items, ..Default::default() }
    }

    #[test]
    fn an_unchanged_signature_never_recomputes() {
        let mut cache = ViewStatsCache::default();
        let runs = Cell::new(0);
        let t0 = Instant::now();
        for frame in 0..50 {
            let answer = cache.get(t0 + Duration::from_millis(frame), 7, false, || {
                runs.set(runs.get() + 1);
                stats(3)
            });
            assert_eq!(answer.stats.items, 3);
        }
        assert_eq!(runs.get(), 1);
        assert_eq!(cache.computes, 1);
    }

    #[test]
    fn a_new_signature_is_one_recompute() {
        let mut cache = ViewStatsCache::default();
        let t0 = Instant::now();
        cache.get(t0, 1, false, || stats(1));
        let answer = cache.get(t0, 2, false, || stats(2));
        assert_eq!(answer.stats.items, 2);
        cache.get(t0, 2, false, || panic!("same signature again"));
        assert_eq!(cache.computes, 2);
    }

    #[test]
    fn streaming_rows_are_recomputed_at_most_four_times_a_second() {
        let mut cache = ViewStatsCache::default();
        let t0 = Instant::now();
        let mut shown = Vec::new();
        for frame in 0..100u64 {
            let at = t0 + Duration::from_millis(frame * 10);
            shown.push(cache.get(at, frame, true, || stats(frame as usize)).stats.items);
        }
        assert!(cache.computes <= 5, "{} computes in one second", cache.computes);
        assert!(shown.windows(2).all(|w| w[1] >= w[0]), "stale answers never go backwards");
    }

    #[test]
    fn a_throttled_answer_asks_for_a_later_repaint() {
        let mut cache = ViewStatsCache::default();
        let t0 = Instant::now();
        cache.get(t0, 1, true, || stats(1));
        let answer = cache.get(t0 + Duration::from_millis(100), 2, true, || panic!("throttled"));
        assert_eq!(answer.stats.items, 1);
        assert_eq!(answer.repaint_after, Some(Duration::from_millis(150)));
        let answer = cache.get(t0 + Duration::from_millis(260), 2, true, || stats(2));
        assert_eq!((answer.stats.items, answer.repaint_after), (2, None));
    }

    #[test]
    fn the_first_answer_is_never_throttled() {
        let mut cache = ViewStatsCache::default();
        let answer = cache.get(Instant::now(), 9, true, || stats(4));
        assert_eq!(answer.stats.items, 4);
    }

    #[test]
    fn a_settled_view_catches_up_even_when_the_last_change_was_throttled() {
        let mut cache = ViewStatsCache::default();
        let t0 = Instant::now();
        cache.get(t0, 1, true, || stats(1));
        cache.get(t0 + Duration::from_millis(50), 2, true, || stats(2));
        let answer = cache.get(t0 + Duration::from_millis(400), 2, false, || stats(2));
        assert_eq!(answer.stats.items, 2);
    }
}
