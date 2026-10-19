//! Per-row badge: whether a file is in the library and whether an indexed file is still on disk.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender};

/// How long a disk check is trusted.
pub const RECHECK_AFTER: Duration = Duration::from_secs(60);
/// Disk checks in flight at once.
const MAX_PENDING: usize = 64;

/// What the library knows about a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Badge {
    /// In the library and on disk.
    Indexed,
    /// On disk only.
    NotIndexed,
    /// In the library but missing on disk.
    Offline,
}

impl Badge {
    pub fn label(self) -> &'static str {
        match self {
            Self::Indexed => "Indexed",
            Self::NotIndexed => "Not indexed",
            Self::Offline => "Offline",
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            Self::Indexed => "In the library",
            Self::NotIndexed => "Not in the library yet",
            Self::Offline => "In the library, but the file is missing on disk",
        }
    }
}

#[derive(Default)]
struct Inner {
    indexed: HashSet<String>,
    exists: HashMap<String, (bool, Instant)>,
    pending: HashSet<String>,
    requested: usize,
}

/// The badges of one tab's rows.
pub struct Badges {
    inner: Mutex<Inner>,
    tx: Sender<(String, bool)>,
    rx: Receiver<(String, bool)>,
}

impl Default for Badges {
    fn default() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self { inner: Mutex::new(Inner::default()), tx, rx }
    }
}

impl Badges {
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Forgets everything, as when the tab shows another scope.
    pub fn clear(&self) {
        *self.inner() = Inner::default();
        while self.rx.try_recv().is_ok() {}
    }

    /// Notes that these files are in the library.
    pub fn mark_indexed<'a>(&self, paths: impl IntoIterator<Item = &'a str>) {
        let mut inner = self.inner();
        inner.indexed.extend(paths.into_iter().map(str::to_string));
    }

    pub fn is_indexed(&self, path: &str) -> bool {
        self.inner().indexed.contains(path)
    }

    pub fn indexed_len(&self) -> usize {
        self.inner().indexed.len()
    }

    /// Takes in the answers of finished disk checks.
    pub fn poll(&self) -> usize {
        let mut inner = self.inner();
        let mut applied = 0;
        while let Ok((path, exists)) = self.rx.try_recv() {
            inner.pending.remove(&path);
            inner.exists.insert(path, (exists, Instant::now()));
            applied += 1;
        }
        applied
    }

    /// Number of disk checks that were started, for the debug window.
    pub fn requested(&self) -> usize {
        self.inner().requested
    }

    pub fn pending(&self) -> usize {
        self.inner().pending.len()
    }

    /// The badge of `path`. A row in the library counts as indexed until a disk check says the file is gone; the check starts here, once per path.
    pub fn badge(&self, path: &str) -> Badge {
        let mut inner = self.inner();
        if !inner.indexed.contains(path) {
            return Badge::NotIndexed;
        }
        if path.contains("://") {
            return Badge::Indexed;
        }
        let stale = match inner.exists.get(path) {
            Some((exists, at)) => {
                if at.elapsed() < RECHECK_AFTER {
                    return if *exists { Badge::Indexed } else { Badge::Offline };
                }
                Some(*exists)
            }
            None => None,
        };
        if !inner.pending.contains(path) && inner.pending.len() < MAX_PENDING {
            inner.pending.insert(path.to_string());
            inner.requested += 1;
            self.start_check(path.to_string());
        }
        match stale {
            Some(false) => Badge::Offline,
            _ => Badge::Indexed,
        }
    }

    fn start_check(&self, path: String) {
        let tx = self.tx.clone();
        let check = move || {
            let exists = std::path::Path::new(&path).exists();
            let _ = tx.send((path, exists));
            crate::command::request_repaint();
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(check);
            }
            Err(_) => check(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(badges: &Badges, path: &str, want: Badge) -> bool {
        for _ in 0..200 {
            badges.poll();
            if badges.badge(path) == want {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn a_file_outside_the_library_is_not_indexed_and_costs_no_disk_check() {
        let badges = Badges::default();
        assert_eq!(badges.badge("/anything.jpg"), Badge::NotIndexed);
        assert_eq!(badges.requested(), 0);
    }

    #[test]
    fn an_indexed_file_on_disk_stays_indexed_after_its_one_check() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.jpg");
        std::fs::write(&p, b"x").unwrap();
        let path = p.display().to_string();
        let badges = Badges::default();
        badges.mark_indexed([path.as_str()]);
        assert_eq!(badges.badge(&path), Badge::Indexed, "indexed until the check says otherwise");
        assert!(settle(&badges, &path, Badge::Indexed));
        for _ in 0..50 {
            badges.badge(&path);
        }
        assert_eq!(badges.requested(), 1, "the answer is remembered");
    }

    #[test]
    fn an_indexed_file_missing_on_disk_becomes_offline() {
        let badges = Badges::default();
        badges.mark_indexed(["/definitely/not/here.jpg"]);
        assert!(settle(&badges, "/definitely/not/here.jpg", Badge::Offline));
    }

    #[test]
    fn virtual_paths_are_never_checked_on_disk() {
        let badges = Badges::default();
        badges.mark_indexed(["zip:///a.zip!/x.png"]);
        assert_eq!(badges.badge("zip:///a.zip!/x.png"), Badge::Indexed);
        assert_eq!(badges.requested(), 0);
    }

    #[test]
    fn checks_in_flight_are_capped_and_never_repeated_for_a_path() {
        let badges = Badges::default();
        let paths: Vec<String> = (0..MAX_PENDING * 3).map(|i| format!("/nope/{i}.jpg")).collect();
        badges.mark_indexed(paths.iter().map(String::as_str));
        for p in &paths {
            badges.badge(p);
            badges.badge(p);
        }
        assert!(badges.requested() <= MAX_PENDING * 3);
        assert!(badges.pending() <= MAX_PENDING);
        assert!(settle(&badges, &paths[0], Badge::Offline));
    }

    #[test]
    fn clearing_forgets_the_library_and_the_answers() {
        let badges = Badges::default();
        badges.mark_indexed(["/a.jpg"]);
        assert_eq!(badges.indexed_len(), 1);
        badges.clear();
        assert_eq!((badges.indexed_len(), badges.badge("/a.jpg")), (0, Badge::NotIndexed));
    }

    #[test]
    fn badges_have_words_for_the_user() {
        for b in [Badge::Indexed, Badge::NotIndexed, Badge::Offline] {
            assert!(!b.label().is_empty() && !b.detail().is_empty());
            assert!(b.label().is_ascii());
        }
    }
}
