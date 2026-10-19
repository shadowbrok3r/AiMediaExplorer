//! Facts about files that need the file system: whether a file is still there and whether an image is icon-sized.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Largest side of an image that counts as an icon.
pub const TINY_EDGE: u32 = 64;
/// Jobs the background worker takes per batch.
const BATCH: usize = 256;

/// What probing one path found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probe {
    pub exists: bool,
    /// Whether the image size was read.
    pub dims_checked: bool,
    /// Both sides of the image are at most [`TINY_EDGE`].
    pub tiny: bool,
}

/// One path to probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeJob {
    pub path: String,
    /// Also read the image size.
    pub dims: bool,
}

/// Probes `path` now; blocks on the file system.
pub fn probe_file(path: &str, dims: bool) -> Probe {
    let exists = match std::fs::metadata(path) {
        Ok(_) => true,
        Err(e) => e.kind() == std::io::ErrorKind::PermissionDenied,
    };
    if !(exists && dims) {
        return Probe { exists, dims_checked: false, tiny: false };
    }
    let size = image::ImageReader::open(path).ok().and_then(|r| r.with_guessed_format().ok()).and_then(|r| r.into_dimensions().ok());
    Probe { exists, dims_checked: true, tiny: size.is_some_and(|(w, h)| w <= TINY_EDGE && h <= TINY_EDGE) }
}

type Waker = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Shared {
    map: RwLock<HashMap<String, Probe>>,
    queue: Mutex<Vec<ProbeJob>>,
    running: AtomicBool,
    generation: AtomicU64,
    waker: Mutex<Option<Waker>>,
}

/// Probe results gathered on a background thread and read by the filter without blocking.
#[derive(Clone, Default)]
pub struct ProbeCache {
    shared: Arc<Shared>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ProbeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calls `waker` after each batch of results arrives.
    pub fn set_waker(&self, waker: impl Fn() + Send + Sync + 'static) {
        *lock(&self.shared.waker) = Some(Arc::new(waker));
    }

    pub fn has_waker(&self) -> bool {
        lock(&self.shared.waker).is_some()
    }

    /// The result for `path`, when it was probed.
    pub fn get(&self, path: &str) -> Option<Probe> {
        self.shared.map.read().unwrap_or_else(|e| e.into_inner()).get(path).copied()
    }

    /// Whether the cache already answers `job`.
    pub fn satisfies(&self, job: &ProbeJob) -> bool {
        self.get(&job.path).is_some_and(|p| !job.dims || p.dims_checked)
    }

    /// Changes whenever results were added or dropped.
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        self.shared.map.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Jobs still waiting or being worked on.
    pub fn pending(&self) -> usize {
        lock(&self.shared.queue).len() + usize::from(self.shared.running.load(Ordering::Acquire))
    }

    /// True when no job is queued or running.
    pub fn is_idle(&self) -> bool {
        lock(&self.shared.queue).is_empty() && !self.shared.running.load(Ordering::Acquire)
    }

    pub fn clear(&self) {
        lock(&self.shared.queue).clear();
        self.shared.map.write().unwrap_or_else(|e| e.into_inner()).clear();
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Stores a result without touching the file system.
    pub fn insert(&self, path: &str, probe: Probe) {
        self.shared.map.write().unwrap_or_else(|e| e.into_inner()).insert(path.to_string(), probe);
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Queues `jobs` for the worker thread, which starts when none is running.
    pub fn request(&self, jobs: Vec<ProbeJob>) {
        let jobs: Vec<ProbeJob> = jobs.into_iter().filter(|j| !self.satisfies(j)).collect();
        if jobs.is_empty() {
            return;
        }
        lock(&self.shared.queue).extend(jobs);
        if !self.shared.running.swap(true, Ordering::AcqRel) {
            let shared = Arc::clone(&self.shared);
            let spawned = std::thread::Builder::new().name("filter-probe".into()).spawn(move || work(shared));
            if spawned.is_err() {
                self.shared.running.store(false, Ordering::Release);
            }
        }
    }

    /// Waits until the queue is drained; for tests and scripts.
    pub fn wait_idle(&self, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while !self.is_idle() {
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        true
    }
}

fn work(shared: Arc<Shared>) {
    loop {
        let batch: Vec<ProbeJob> = {
            let mut queue = lock(&shared.queue);
            let n = queue.len().min(BATCH);
            queue.drain(..n).collect()
        };
        if batch.is_empty() {
            shared.running.store(false, Ordering::Release);
            let more = !lock(&shared.queue).is_empty();
            if more && !shared.running.swap(true, Ordering::AcqRel) {
                continue;
            }
            return;
        }
        let results: Vec<(String, Probe)> = batch
            .into_iter()
            .map(|j| {
                let probe = probe_file(&j.path, j.dims);
                (j.path, probe)
            })
            .collect();
        {
            let mut map = shared.map.write().unwrap_or_else(|e| e.into_inner());
            map.extend(results);
        }
        shared.generation.fetch_add(1, Ordering::AcqRel);
        let waker = lock(&shared.waker).clone();
        if let Some(wake) = waker {
            wake();
        }
    }
}
