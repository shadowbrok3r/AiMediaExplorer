//! Files whose thumbnail could not be made, shared by every tab; the Failed thumbnails smart view lists them.

use std::sync::Mutex;

/// Most failures remembered; the oldest are dropped first.
pub const CAPACITY: usize = 5_000;

static FAILED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

fn failures<R>(f: impl FnOnce(&mut Vec<(String, String)>) -> R) -> R {
    f(&mut FAILED.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Notes that the thumbnail of `path` failed with `error`, replacing an earlier note for the same path.
pub fn record(path: &str, error: &str) {
    failures(|all| {
        all.retain(|(p, _)| p != path);
        all.push((path.to_string(), error.to_string()));
        if all.len() > CAPACITY {
            let excess = all.len() - CAPACITY;
            all.drain(..excess);
        }
    });
}

/// Forgets the failure of `path` after a thumbnail was made for it.
pub fn clear_path(path: &str) {
    failures(|all| all.retain(|(p, _)| p != path));
}

pub fn clear() {
    failures(|all| all.clear());
}

/// `(path, error)` of every failure, oldest first.
pub fn snapshot() -> Vec<(String, String)> {
    failures(|all| all.clone())
}

pub fn count() -> usize {
    failures(|all| all.len())
}

#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_are_kept_per_path_and_cleared_when_a_thumbnail_is_made() {
        let _serial = serial();
        clear();
        record("/a.jpg", "bad header");
        record("/b.jpg", "truncated");
        record("/a.jpg", "still bad");
        assert_eq!(snapshot(), vec![("/b.jpg".to_string(), "truncated".to_string()), ("/a.jpg".to_string(), "still bad".to_string())]);
        clear_path("/b.jpg");
        assert_eq!(count(), 1);
        clear_path("/never-failed.jpg");
        assert_eq!(count(), 1);
        clear();
        assert!(snapshot().is_empty());
    }

    #[test]
    fn the_oldest_failures_are_dropped_past_the_capacity() {
        let _serial = serial();
        clear();
        for i in 0..CAPACITY + 10 {
            record(&format!("/f/{i}.jpg"), "e");
        }
        assert_eq!(count(), CAPACITY);
        assert_eq!(snapshot()[0].0, "/f/10.jpg");
        clear();
    }
}
