//! Facts read from the file itself for the Info tab: existence, dates, picture and video dimensions.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::{Instant, SystemTime};

use chrono::{DateTime, Local};

/// Entries kept per cache.
pub const CACHE_CAP: usize = 96;
/// Age after which a cached entry is read again.
const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoFacts {
    pub duration_secs: Option<f64>,
    pub codec: Option<String>,
    pub fps: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileFacts {
    /// False when the file is gone; true for paths inside archives, which cannot be checked cheaply.
    pub exists: bool,
    pub size: Option<u64>,
    pub modified: Option<SystemTime>,
    pub created: Option<SystemTime>,
    pub dims: Option<(u32, u32)>,
    pub video: Option<VideoFacts>,
}

/// True for paths that name something inside an archive or a virtual folder.
pub fn is_virtual(path: &str) -> bool {
    path.contains("://")
}

/// Reads the facts of `path` from disk; blocking.
pub fn read_facts(path: &str, is_image: bool, is_video: bool) -> FileFacts {
    if is_virtual(path) {
        return FileFacts { exists: true, ..Default::default() };
    }
    let p = Path::new(path);
    let Ok(meta) = std::fs::metadata(p) else { return FileFacts::default() };
    let mut facts = FileFacts { exists: true, size: Some(meta.len()), modified: meta.modified().ok(), created: meta.created().ok(), ..Default::default() };
    if is_image {
        facts.dims = image::ImageReader::open(p).ok().and_then(|r| r.with_guessed_format().ok()).and_then(|r| r.into_dimensions().ok());
    }
    if is_video {
        if let Ok(info) = crate::video::probe(p) {
            let (w, h) = info.display_size();
            if w > 0 && h > 0 {
                facts.dims = Some((w, h));
            }
            facts.video = Some(VideoFacts { duration_secs: info.duration_secs, codec: info.video_codec.clone(), fps: info.fps });
        }
    }
    facts
}

/// A date for the Info tab in local time.
pub fn format_time(time: SystemTime) -> String {
    DateTime::<Local>::from(time).format("%Y-%m-%d %H:%M").to_string()
}

/// A duration as `m:ss` or `h:mm:ss`.
pub fn format_duration(secs: f64) -> String {
    let total = if secs.is_finite() { secs.max(0.0) as u64 } else { 0 };
    if total >= 3600 { format!("{}:{:02}:{:02}", total / 3600, (total / 60) % 60, total % 60) } else { format!("{}:{:02}", total / 60, total % 60) }
}

#[derive(Debug)]
enum Entry<T> {
    Loading(Instant),
    Ready(T, Instant),
}

/// Small cache of per-path lookups that run in the background; tells the caller when to start one.
#[derive(Debug)]
pub struct LookupCache<T> {
    map: HashMap<String, Entry<T>>,
    order: VecDeque<String>,
    cap: usize,
}

impl<T> Default for LookupCache<T> {
    fn default() -> Self {
        Self::with_capacity(CACHE_CAP)
    }
}

impl<T> LookupCache<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), cap: cap.max(1) }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The result for `path`, if loaded.
    pub fn get(&self, path: &str) -> Option<&T> {
        match self.map.get(path) {
            Some(Entry::Ready(v, _)) => Some(v),
            _ => None,
        }
    }

    pub fn is_loading(&self, path: &str) -> bool {
        matches!(self.map.get(path), Some(Entry::Loading(_)))
    }

    /// Whether the caller must start a lookup for `path` now; a stale result stays readable until the new one arrives.
    pub fn want(&mut self, path: &str, now: Instant) -> bool {
        match self.map.get_mut(path) {
            None => {
                self.insert(path, Entry::Loading(now));
                true
            }
            Some(Entry::Loading(since)) if now.saturating_duration_since(*since) > STALE_AFTER * 2 => {
                *since = now;
                true
            }
            Some(Entry::Ready(_, at)) if now.saturating_duration_since(*at) > STALE_AFTER => {
                *at = now;
                true
            }
            Some(_) => false,
        }
    }

    /// Stores the result of a lookup.
    pub fn put(&mut self, path: &str, value: T, now: Instant) {
        self.insert(path, Entry::Ready(value, now));
    }

    /// Stores a result that is looked up again after `ttl` instead of the usual staleness.
    pub fn put_for(&mut self, path: &str, value: T, now: Instant, ttl: std::time::Duration) {
        let at = now.checked_sub(STALE_AFTER.saturating_sub(ttl)).unwrap_or(now);
        self.insert(path, Entry::Ready(value, at));
    }

    /// Forgets `path` so the next [`want`](Self::want) starts a new lookup.
    pub fn invalidate(&mut self, path: &str) {
        if self.map.remove(path).is_some() {
            self.order.retain(|p| p != path);
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn insert(&mut self, path: &str, entry: Entry<T>) {
        if self.map.insert(path.to_string(), entry).is_none() {
            self.order.push_back(path.to_string());
        }
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_lookup_starts_once_and_again_only_after_it_went_stale() {
        let mut cache: LookupCache<u32> = LookupCache::default();
        let t0 = Instant::now();
        assert!(cache.want("/a", t0), "first request starts a lookup");
        assert!(cache.is_loading("/a"));
        assert!(!cache.want("/a", t0 + Duration::from_secs(1)), "one lookup at a time");
        cache.put("/a", 7, t0 + Duration::from_secs(2));
        assert_eq!(cache.get("/a"), Some(&7));
        assert!(!cache.want("/a", t0 + Duration::from_secs(10)));
        assert!(cache.want("/a", t0 + Duration::from_secs(60)), "an old result is read again");
        assert_eq!(cache.get("/a"), Some(&7), "the old result stays readable while the new lookup runs");
        assert!(!cache.want("/a", t0 + Duration::from_secs(61)));
    }

    #[test]
    fn a_short_lived_result_is_looked_up_again_after_its_ttl() {
        let mut cache: LookupCache<u32> = LookupCache::default();
        let t0 = Instant::now() + Duration::from_secs(100);
        cache.put_for("/a", 1, t0, Duration::from_secs(3));
        assert_eq!(cache.get("/a"), Some(&1));
        assert!(!cache.want("/a", t0 + Duration::from_secs(2)));
        assert!(cache.want("/a", t0 + Duration::from_secs(4)));
        assert_eq!(cache.get("/a"), Some(&1), "the short-lived value stays readable meanwhile");
    }

    #[test]
    fn a_lookup_that_never_answers_is_retried() {
        let mut cache: LookupCache<u32> = LookupCache::default();
        let t0 = Instant::now();
        assert!(cache.want("/a", t0));
        assert!(!cache.want("/a", t0 + Duration::from_secs(30)));
        assert!(cache.want("/a", t0 + Duration::from_secs(90)));
    }

    #[test]
    fn invalidating_makes_the_next_request_start_a_lookup() {
        let mut cache: LookupCache<u32> = LookupCache::default();
        let t0 = Instant::now();
        cache.put("/a", 1, t0);
        cache.invalidate("/a");
        assert!(cache.get("/a").is_none());
        assert!(cache.want("/a", t0));
        cache.invalidate("/missing");
    }

    #[test]
    fn the_cache_forgets_the_oldest_entries_beyond_its_capacity() {
        let mut cache: LookupCache<u32> = LookupCache::with_capacity(3);
        let t0 = Instant::now();
        for (i, p) in ["/a", "/b", "/c", "/d"].iter().enumerate() {
            cache.put(p, i as u32, t0);
        }
        assert_eq!(cache.len(), 3);
        assert!(cache.get("/a").is_none());
        assert_eq!(cache.get("/d"), Some(&3));
        cache.put("/d", 9, t0);
        assert_eq!(cache.len(), 3, "replacing an entry does not grow the cache");
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn facts_of_a_missing_file_say_it_is_gone_and_virtual_paths_are_assumed_present() {
        assert!(!read_facts("/definitely/not/here.jpg", true, false).exists);
        assert!(read_facts("zip:///a.zip!/b.jpg", true, false).exists);
        assert!(is_virtual("tar:///x!/y"));
        assert!(!is_virtual("/plain/path"));
    }

    #[test]
    fn facts_of_a_picture_carry_size_and_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.png");
        image::RgbImage::from_pixel(30, 20, image::Rgb([1, 2, 3])).save(&path).unwrap();
        let facts = read_facts(&path.display().to_string(), true, false);
        assert!(facts.exists);
        assert_eq!(facts.dims, Some((30, 20)));
        assert!(facts.size.is_some_and(|s| s > 0));
        assert!(facts.modified.is_some());
        let text = dir.path().join("t.txt");
        std::fs::write(&text, "hello").unwrap();
        let facts = read_facts(&text.display().to_string(), true, false);
        assert_eq!((facts.exists, facts.size, facts.dims), (true, Some(5), None), "a file that is not a picture has no dimensions");
    }

    #[test]
    fn durations_and_dates_are_formatted_for_reading() {
        assert_eq!(format_duration(65.2), "1:05");
        assert_eq!(format_duration(3725.0), "1:02:05");
        assert_eq!(format_duration(f64::NAN), "0:00");
        assert_eq!(format_duration(-3.0), "0:00");
        assert_eq!(format_time(SystemTime::UNIX_EPOCH).len(), "1970-01-01 00:00".len());
    }
}
