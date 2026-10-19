use crate::utilities::types::{Filters, FoundFile, MediaKind, DirItem, is_image, is_video, is_archive};
use crate::utilities::thumbs::{EncodedThumb, ThumbSink, thumb_sink};
use crate::utilities::filtering::FiltersExt;
use crate::utilities::guard::{guarded, panic_message};
use std::sync::atomic::{AtomicU64, Ordering};
use jwalk::{WalkDirGeneric, Parallelism};
use crossbeam::channel::{Sender, Receiver, bounded, unbounded};
use chrono::{DateTime, Local};
use once_cell::sync::Lazy;
use std::time::SystemTime;
use std::path::Path;
use std::time::{Duration, Instant};
use rayon::{ThreadPool, ThreadPoolBuildError};
use std::sync::{Arc, OnceLock};

// Track cancelled scan ids
static CANCELLED_SCANS: Lazy<std::sync::Mutex<std::collections::HashSet<u64>>> = Lazy::new(|| std::sync::Mutex::new(Default::default()));
static SCAN_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
/// send a progress update every N scanned paths in recursive mode
const PROGRESS_EVERY: usize = 5_000;

/// Scan id of thumbnail jobs queued outside a scan; they have no delivery route.
pub const STANDALONE_SCAN_ID: u64 = 0;

pub fn next_scan_id() -> u64 { SCAN_ID_COUNTER.fetch_add(1, Ordering::Relaxed) }

pub fn cancel_scan(id: u64) {
    if id == 0 { return; }
    if let Ok(mut set) = CANCELLED_SCANS.lock() { set.insert(id); }
}

fn is_cancelled(id: u64) -> bool {
    CANCELLED_SCANS.lock().map(|s| s.contains(&id)).unwrap_or(false)
}

fn clear_cancel(id: u64) {
    if let Ok(mut set) = CANCELLED_SCANS.lock() { let _ = set.remove(&id); }
}

/// A scan whose walk or queued thumbnail jobs have not finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveScan {
    pub id: u64,
    pub queued_thumbs: usize,
    pub walk_done: bool,
    pub cancelled: bool,
}

/// Scans that still have a delivery route, oldest first.
pub fn active_scans() -> Vec<ActiveScan> {
    let mut scans: Vec<ActiveScan> = SCAN_ROUTERS
        .lock()
        .map(|m| m.iter().map(|(id, r)| ActiveScan { id: *id, queued_thumbs: r.pending, walk_done: r.walk_done, cancelled: is_cancelled(*id) }).collect())
        .unwrap_or_default();
    scans.sort_by_key(|s| s.id);
    scans
}

// --- Thumbnail pipeline: unbounded job queue drained by a rayon worker pool ---
#[derive(Clone, Debug)]
pub struct ThumbJob {
    pub path: std::path::PathBuf,
    pub kind: MediaKind,
    pub scan_id: u64,
}

// Number of panics swallowed by pool panic handlers.
static POOL_PANICS: AtomicU64 = AtomicU64::new(0);

fn try_build_pool(name: &'static str, workers: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(move |i| format!("{name}-{i}"))
        .panic_handler(move |p| {
            POOL_PANICS.fetch_add(1, Ordering::Relaxed);
            log::error!("[{name}] pool task panicked: {}", panic_message(p.as_ref()));
        })
        .build()
}

fn thumb_worker_count() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    logical.saturating_sub(2).clamp(2, 12)
}

static THUMB_POOL: Lazy<ThreadPool> = Lazy::new(|| try_build_pool("thumb", thumb_worker_count()).expect("thumb pool"));

static THUMB_TX: OnceLock<Sender<ThumbJob>> = OnceLock::new();
static THUMB_PRIORITY_TX: OnceLock<Sender<ThumbJob>> = OnceLock::new();
const PRIORITY_QUEUE_CAP: usize = 512;

// Per-scan delivery route; removed once the walk finished and every queued thumbnail job completed.
struct ScanRoute {
    tx: Sender<ScanEnvelope>,
    pending: usize,
    walk_done: bool,
}

static SCAN_ROUTERS: Lazy<std::sync::Mutex<std::collections::HashMap<u64, ScanRoute>>> =
    Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

// Registers the delivery route; earlier scans still active on the same channel are superseded and cancelled.
fn register_route(scan_id: u64, tx: Sender<ScanEnvelope>) {
    let superseded: Vec<u64> = SCAN_ROUTERS
        .lock()
        .map(|m| m.iter().filter(|(id, r)| **id != scan_id && r.tx.same_channel(&tx)).map(|(id, _)| *id).collect())
        .unwrap_or_default();
    for id in superseded { cancel_scan(id); }
    if let Ok(mut map) = SCAN_ROUTERS.lock() { map.insert(scan_id, ScanRoute { tx, pending: 0, walk_done: false }); }
}

fn route_pending(scan_id: u64) -> usize {
    SCAN_ROUTERS.lock().ok().and_then(|m| m.get(&scan_id).map(|r| r.pending)).unwrap_or(0)
}

fn route_pending_add(scan_id: u64) {
    if let Ok(mut map) = SCAN_ROUTERS.lock() {
        if let Some(r) = map.get_mut(&scan_id) { r.pending += 1; }
    }
}

fn route_send(scan_id: u64, msg: ScanMsg) {
    let tx = SCAN_ROUTERS.lock().ok().and_then(|m| m.get(&scan_id).map(|r| r.tx.clone()));
    if let Some(tx) = tx { let _ = tx.send(ScanEnvelope { scan_id, msg }); }
}

// Drops the route and the cancel flag once nothing is left to deliver for the scan.
fn route_release_if_idle(scan_id: u64) {
    let idle = SCAN_ROUTERS.lock().map(|mut m| {
        let idle = m.get(&scan_id).map(|r| r.walk_done && r.pending == 0).unwrap_or(true);
        if idle { m.remove(&scan_id); }
        idle
    }).unwrap_or(false);
    if idle { clear_cancel(scan_id); }
}

fn route_job_done(scan_id: u64) {
    if let Ok(mut map) = SCAN_ROUTERS.lock() {
        if let Some(r) = map.get_mut(&scan_id) { r.pending = r.pending.saturating_sub(1); }
    }
    route_release_if_idle(scan_id);
}

fn route_walk_done(scan_id: u64) {
    if let Ok(mut map) = SCAN_ROUTERS.lock() {
        if let Some(r) = map.get_mut(&scan_id) { r.walk_done = true; }
    }
    route_release_if_idle(scan_id);
}

pub fn init_thumbnail_workers() {
    if THUMB_TX.get().is_some() { return; }
    let (tx, rx): (Sender<ThumbJob>, Receiver<ThumbJob>) = unbounded();
    let (prio_tx, prio_rx): (Sender<ThumbJob>, Receiver<ThumbJob>) = bounded(PRIORITY_QUEUE_CAP);
    if THUMB_TX.set(tx).is_err() { return; }
    let _ = THUMB_PRIORITY_TX.set(prio_tx);
    start_thumbnail_workers(rx, prio_rx);
}

fn start_thumbnail_workers(rx: Receiver<ThumbJob>, prio_rx: Receiver<ThumbJob>) {
    let sink = thumb_sink();
    for _ in 0..THUMB_POOL.current_num_threads() {
        let (rx, prio_rx, sink) = (rx.clone(), prio_rx.clone(), sink.clone());
        THUMB_POOL.spawn(move || run_thumb_worker(rx, prio_rx, generate_thumb, sink));
    }
}

type ThumbGenerator = fn(&Path, &MediaKind) -> Result<EncodedThumb, String>;

// Next job, preferring the priority queue; None once the scan queue is closed.
fn next_thumb_job(rx: &Receiver<ThumbJob>, prio_rx: &Receiver<ThumbJob>) -> Option<ThumbJob> {
    if let Ok(job) = prio_rx.try_recv() { return Some(job); }
    let mut select = crossbeam::channel::Select::new();
    let prio_index = select.recv(prio_rx);
    let scan_index = select.recv(rx);
    let op = select.select();
    if op.index() == prio_index {
        match op.recv(prio_rx) {
            Ok(job) => Some(job),
            Err(_) => rx.recv().ok(),
        }
    } else {
        op.recv(rx).ok()
    }
}

fn run_thumb_worker(rx: Receiver<ThumbJob>, prio_rx: Receiver<ThumbJob>, generate: ThumbGenerator, sink: Arc<dyn ThumbSink>) {
    while let Some(job) = next_thumb_job(&rx, &prio_rx) {
        if !is_cancelled(job.scan_id) {
            let msg = produce_thumb(&job, generate, sink.as_ref());
            if job.scan_id != STANDALONE_SCAN_ID { route_send(job.scan_id, msg); }
        }
        if job.scan_id != STANDALONE_SCAN_ID { route_job_done(job.scan_id); }
    }
}

// Reuses a stored thumbnail that is still valid, otherwise generates one and hands the bytes to the sink.
fn produce_thumb(job: &ThumbJob, generate: ThumbGenerator, sink: &dyn ThumbSink) -> ScanMsg {
    let path = job.path.clone();
    if let Some(key) = sink.fresh_key(&job.path) {
        return ScanMsg::ThumbReady { path, key };
    }
    match guarded(|| generate(&job.path, &job.kind), |msg| format!("thumbnail worker panicked: {msg}")) {
        Ok(thumb) => ScanMsg::ThumbReady { key: sink.accept(&job.path, thumb), path },
        Err(error) => {
            sink.failed(&job.path, &error);
            ScanMsg::ThumbFailed { path, error }
        }
    }
}

fn enqueue_thumb_job(path: &Path, kind: MediaKind, scan_id: u64) {
    let Some(tx) = THUMB_TX.get() else { return };
    route_pending_add(scan_id);
    if tx.send(ThumbJob { path: path.to_path_buf(), kind, scan_id }).is_err() {
        route_job_done(scan_id);
    }
}

// Queues a generation outside any scan, ahead of the scan backlog; false when the priority queue is full.
pub fn request_priority_thumb(path: &Path, kind: MediaKind) -> bool {
    init_thumbnail_workers();
    let Some(tx) = THUMB_PRIORITY_TX.get() else { return false };
    tx.try_send(ThumbJob { path: path.to_path_buf(), kind, scan_id: STANDALONE_SCAN_ID }).is_ok()
}

fn generate_thumb(path: &Path, kind: &MediaKind) -> Result<EncodedThumb, String> {
    match kind {
        MediaKind::Image => crate::utilities::thumbs::generate_image_thumb(path),
        MediaKind::Video => crate::utilities::thumbs::generate_video_thumb(path),
        _ => Err("unsupported kind".to_string()),
    }
}

#[derive(Debug)]
pub enum ScanMsg {
    Found(FoundFile),
    FoundBatch(Vec<FoundFile>),
    FoundDir(DirItem),
    FoundDirBatch(Vec<DirItem>),
    /// A thumbnail for `path` is stored under blob `key`; the pixels never travel through this message.
    ThumbReady {
        path: std::path::PathBuf,
        key: String,
    },
    ThumbFailed {
        path: std::path::PathBuf,
        error: String,
    },
    /// Report encrypted archives encountered during scans so the UI can prompt once at the end.
    EncryptedArchives(Vec<String>),
    Progress {
        scanned: usize,
        total: usize,
    },
    Error(String),
    Done,
}

fn systemtime_to_local(st: SystemTime) -> DateTime<Local> {
    DateTime::<Local>::from(st)
}

fn is_media_kind(path: &Path) -> MediaKind {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
        if is_image(&ext) { return MediaKind::Image; }
        if is_video(&ext) { return MediaKind::Video; }
        if is_archive(&ext) { return MediaKind::Archive; }
    }
    MediaKind::Other
}

// Global scanning channel (stable for entire app lifetime) so we don't swap Receivers in and out of scopes.
// We wrap each message with a scan generation id allowing us to ignore stale messages from previous scans.
#[derive(Debug)]
pub struct ScanEnvelope {
    pub scan_id: u64,
    pub msg: ScanMsg,
}

// Main entry used by UI resource: offloads whole scan onto blocking thread pool.
pub async fn spawn_scan(filters: Filters, tx: Sender<ScanEnvelope>, recursive: bool, scan_id: u64) {
    // Ensure thumbnail workers are initialized with this scan channel
    init_thumbnail_workers();
    // Register router for this scan id
    register_route(scan_id, tx.clone());
    let done_tx = tx.clone();
    // Perform the blocking filesystem traversal on a blocking thread.
    if let Err(e) = tokio::task::spawn_blocking(move || perform_scan_blocking(filters, tx, recursive, scan_id)).await {
        log::warn!("spawn_blocking scan join error: {e:?}");
        let _ = done_tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
        route_walk_done(scan_id);
    }
}

fn perform_scan_blocking(filters: Filters, tx: Sender<ScanEnvelope>, recursive: bool, scan_id: u64) {
    // Optional timing instrumentation: enabled in tests or with env var SMART_MEDIA_SCAN_TIMING=1
    let timing_enabled = cfg!(test) || std::env::var("SMART_MEDIA_SCAN_TIMING").ok().as_deref() == Some("1");
    let mut t_all = Instant::now();
    let t_last_phase = &mut t_all;
    if is_cancelled(scan_id) {
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
        route_walk_done(scan_id);
        return;
    }
    
    let root = if filters.root.as_os_str().is_empty() {
        match std::env::current_dir().and_then(|p| std::path::absolute(p)) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Error(e.to_string()) });
                let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
                route_walk_done(scan_id);
                return;
            }
        }
    } else {
        // Normalize WSL UNC on Windows to improve exists()/is_dir() stability
        let p = &filters.root;
        let s = p.to_string_lossy().to_string();
        std::path::PathBuf::from(crate::utilities::windows::normalize_wsl_unc(&s))
    };

    if !root.exists() {
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Error(format!("Root does not exist: {}", root.display())) });
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
        route_walk_done(scan_id);
        return;
    }
    if timing_enabled { 
        eprintln!("scan[{scan_id}]: setup done in {:?}", t_last_phase.elapsed()); 
        *t_last_phase = Instant::now(); 
    }

    // For recursive scans allow override date range
    let after = if recursive { filters.recursive_modified_after.as_ref().or(filters.modified_after.as_ref()) } else { filters.modified_after.as_ref() }
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(0,0,0).unwrap());

    let before = if recursive { filters.recursive_modified_before.as_ref().or(filters.modified_before.as_ref()) } else { filters.modified_before.as_ref() }
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(23,59,59).unwrap());

    let mut scanned = 0usize; // shallow-only: number of file paths processed at root level
    let total = if recursive { 0 } else { estimate_total_shallow(&root, &filters) };
    let _ = tx.send(ScanEnvelope { 
        scan_id, 
        msg: ScanMsg::Progress { scanned: 0, total } 
    });

    // Clone once for the walker closure; keep original `filters` for later processing
    let filters_for_walker = filters.clone();
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16);
    let mut batch: Vec<FoundFile> = vec![];
    let found_batch_max: usize = crate::database::settings::SETTINGS_CACHE
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .and_then(|s| s.scan_found_batch_max)
        .unwrap_or(128)
        .clamp(16, 4096);
    log::info!("Found batch size: {found_batch_max}");
    let mut scanned_paths: usize = 0;
    let walk_threads = std::cmp::max(8, std::cmp::min(logical, 16));
    let parallelism = match try_build_pool("walk", walk_threads) {
        Ok(pool) => Parallelism::RayonExistingPool { pool: Arc::new(pool), busy_timeout: None },
        Err(_) => Parallelism::RayonNewPool(walk_threads),
    };

    let excluded_dirs = filters.recursive_excluded_dirs.clone();
    let excluded_terms = filters.excluded_terms.clone();

    // Use jwalk for both recursive and shallow traversal.
    let walker = WalkDirGeneric::<((), Option<u64>)>::new(&root)
        .skip_hidden(false)
        .follow_links(false)
        .parallelism(parallelism)
        .max_depth(if recursive { usize::MAX } else { 1 })
        .process_read_dir(move |_depth, dir_path, _state, entries| {
            // Helper function to check if a path matches any excluded directory
            let path_matches_excluded = |path: &std::path::Path| -> bool {
                // Create lowercase component vectors for case-insensitive comparison on Windows
                let path_comps: Vec<String> = path
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().to_string().to_ascii_lowercase())
                    .collect();
                excluded_dirs.iter().any(|excluded| {
                    let ex_path = std::path::Path::new(excluded);
                    let ex_comps: Vec<String> = ex_path
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string().to_ascii_lowercase())
                        .collect();
                    if ex_comps.is_empty() { return false; }
                    if ex_path.is_absolute() {
                        if ex_comps.len() > path_comps.len() { return false; }
                        // Component-wise prefix compare (case-insensitive)
                        for i in 0..ex_comps.len() {
                            if path_comps[i] != ex_comps[i] { return false; }
                        }
                        true
                    } else {
                        // Relative pattern: match anywhere in the path (component-wise)
                        if ex_comps.len() > path_comps.len() { return false; }
                        for start in 0..=(path_comps.len() - ex_comps.len()) {
                            let mut all_eq = true;
                            for j in 0..ex_comps.len() {
                                if path_comps[start + j] != ex_comps[j] { all_eq = false; break; }
                            }
                            if all_eq { return true; }
                        }
                        false
                    }
                })
            };

            // Only enforce recursive excludes during recursive scans.
            if recursive {
                if path_matches_excluded(dir_path) {
                    entries.clear();
                    return;
                }
            }

            // Filter & optionally sort entries in-place.
            entries.retain(|res| {
                if let Ok(entry) = res {
                    let p = entry.path();
                    // Early substring filter on full lowercased path when configured
                    if !excluded_terms.is_empty() {
                        let lp = p.to_string_lossy().to_ascii_lowercase();
                        if excluded_terms.iter().any(|t| lp.contains(t)) { return false; }
                    }
                    if entry.file_type.is_dir() {
                        // In recursive mode, skip excluded directories entirely (and their children).
                        if recursive && path_matches_excluded(&p) {
                            return false;
                        }
                        return true; // keep directory (walker decides whether to descend based on max_depth)
                    }
                    if entry.file_type.is_file() {
                        if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                            // Quick pre-filter via trait: includes toggles + recursive excludes
                            if !filters_for_walker.ext_allowed(&ext, recursive) { return false; }
                            // In shallow scans we optionally include archives (.zip) in listing
                            if !recursive && is_archive(ext.as_str()) && !filters_for_walker.include_archives { return false; }
                            return true;
                        } else { return false; }
                    }
                    return false;
                }
                true
            });
            if !recursive {
                entries.sort_by(|a,b| match (a,b) { (Ok(ae), Ok(be)) => ae.file_name.cmp(&be.file_name), _ => std::cmp::Ordering::Equal });
            }
        });
    if timing_enabled { 
        eprintln!("scan[{scan_id}]: walker constructed in {:?}", t_last_phase.elapsed()); 
        *t_last_phase = Instant::now(); 
    }

    // For shallow scans we also collect the immediate subdirectories to send in a single batch at the end.
    let mut dir_batch: Vec<DirItem> = if recursive { Vec::new() } else { Vec::with_capacity(128) };

    let mut idx: usize = 0;
    // Collect videos for deferred thumbnailing
    let mut videos_for_later: Vec<std::path::PathBuf> = Vec::new();
    // (Removed) encrypted archive probing: now done lazily when navigating into an archive
    // Track per-progress timing in recursive scans to spot slowdowns
    let mut last_progress_tick: Option<Instant> = if recursive { Some(Instant::now()) } else { None };
    for dir_entry_result in walker {
        if is_cancelled(scan_id) {
            let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
            route_walk_done(scan_id);
            return;
        }
        match dir_entry_result {
            Ok(entry) => {
                if entry.file_type.is_dir() {
                    // Only in shallow mode, we expose the immediate subdirectories (not the root itself).
                    if !recursive {
                        if entry.path() != root { dir_batch.push(DirItem { path: entry.path() }); }
                    }
                    continue;
                }
                if entry.file_type.is_file() {
                    let path_buf = entry.path();
                    let path = path_buf.as_path();
                    if let Some(found) = process_path_collect(path, &filters, after, before) {
                        batch.push(found);
                        if batch.len() >= found_batch_max {
                            let _ = tx.send(ScanEnvelope { 
                                scan_id, 
                                msg: ScanMsg::FoundBatch(std::mem::take(&mut batch)) 
                            });
                        }
                        // Schedule image thumbnails immediately; defer videos
                        match is_media_kind(path) {
                            MediaKind::Image => enqueue_thumb_job(path, MediaKind::Image, scan_id),
                            MediaKind::Video => videos_for_later.push(path.to_path_buf()),
                            _ => {}
                        }
                    }
                    if recursive {
                        scanned_paths += 1;
                        if scanned_paths % PROGRESS_EVERY == 0 {
                            if timing_enabled {
                                if let Some(start) = last_progress_tick.take() {
                                    let dt = start.elapsed();
                                    eprintln!(
                                        "scan[{scan_id}]: progress {} files — last {} took {:?} (avg {:?}/1k)",
                                        scanned_paths,
                                        PROGRESS_EVERY,
                                        dt,
                                        dt / (PROGRESS_EVERY as u32 / 1000)
                                    );
                                }
                                last_progress_tick = Some(Instant::now());
                            }
                            let _ = tx.send(ScanEnvelope { 
                                scan_id, 
                                msg: ScanMsg::Progress { scanned: scanned_paths, total: 0 } 
                            });
                        }
                    } else {
                        scanned += 1;
                        let display_scanned = if total > 0 { scanned.min(total) } else { scanned };
                        if scanned % 25 == 0 {
                            let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: display_scanned, total } });
                        }
                        if scanned % 200 == 0 { std::thread::yield_now(); }
                    }
                }
            }
            Err(err) => {
                let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Error(format!("walk error: {err}")) });
            }
        }
        if idx % 1000 == 0 { std::thread::yield_now(); }
        idx += 1;
    }

    // Flush remaining batches and final progress per mode
    if timing_enabled { 
        eprintln!("scan[{scan_id}]: walking loop finished in {:?}", t_last_phase.elapsed()); 
        *t_last_phase = Instant::now(); 
    }
    if !recursive {
        if !dir_batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundDirBatch(dir_batch) }); }
        if !batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let final_scanned = if total > 0 { scanned.min(total) } else { scanned };
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: final_scanned, total } });
    } else {
        if !batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: scanned_paths, total: 0 } });
    }
    if timing_enabled { 
        eprintln!("scan[{scan_id}]: flush + final progress in {:?}", t_last_phase.elapsed()); 
        *t_last_phase = Instant::now(); 
    }

    // Defer video thumbnails until image thumbnails are complete
    // Wait briefly for this scan's queued thumbnails to drain (respect cancellation)
    let mut spins = 0usize;
    while route_pending(scan_id) > 0 {
        if is_cancelled(scan_id) { break; }
        std::thread::sleep(Duration::from_millis(15));
        spins += 1;
        if spins > 2_000 { break; } // ~30s safety cap
    }
    if timing_enabled {
        eprintln!("scan[{scan_id}]: wait for image thumbnails drained in {:?}", t_last_phase.elapsed());
        *t_last_phase = Instant::now();
    }
    for vpath in videos_for_later.into_iter() {
        enqueue_thumb_job(&vpath, MediaKind::Video, scan_id);
    }
    if timing_enabled { 
        eprintln!("scan[{scan_id}]: enqueue video thumbnails in {:?}", t_last_phase.elapsed()); 
        *t_last_phase = Instant::now(); 
    }

    finish(&tx, scan_id);
    if timing_enabled { eprintln!("scan[{scan_id}]: total elapsed {:?}", t_all.elapsed()); }
}

fn estimate_total_shallow(root: &Path, filters: &Filters) -> usize {
    if let Ok(rd) = std::fs::read_dir(root) {
        let mut count = 0usize;
        for dent in rd.flatten() {
            if let Ok(ft) = dent.file_type() { if !ft.is_file() { continue; } } else { continue; }
            if let Some(ext) = dent.path().extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()) {
                if (filters.include_images && is_image(&ext)) || (filters.include_videos && is_video(&ext)) { count += 1; }
            }
        }
        count
    } else { 0 }
}

fn finish(tx: &Sender<ScanEnvelope>, scan_id: u64) {
    let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
    // The route stays registered until the last queued thumbnail of this scan has been delivered
    route_walk_done(scan_id);
}

fn process_path_collect(
    path: &Path,
    filters: &Filters,
    after: Option<chrono::NaiveDateTime>,
    before: Option<chrono::NaiveDateTime>,
) -> Option<FoundFile> {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) { if name.starts_with("._") { return None; } }
    // Guard against excluded terms again (extra safety in case of path changes between read_dir and metadata fetch)
    if !filters.excluded_terms.is_empty() {
        let lp = path.to_string_lossy().to_ascii_lowercase();
        if filters.excluded_terms.iter().any(|t| lp.contains(t)) { return None; }
    }
    let kind = is_media_kind(path);
    // Permit archives when explicitly requested for shallow scans (represented as MediaKind::Other)
    let is_archive_file = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).map(|e| is_archive(e.as_str())).unwrap_or(false);
    if !filters.kind_allowed(&kind, is_archive_file) { return None; }
    let md = match path.metadata() {
        Ok(m) => m,
        Err(_) => return None,
    };
    let modified = md.modified().ok().map(systemtime_to_local);
    let created = md.created().ok().map(systemtime_to_local);
    if !filters.date_ok(modified, created, /*recursive*/ after.is_some() || before.is_some()) { return None; }
    let size_val = md.len();
    if !filters.skip_icons_heuristic_allows(path, size_val) { return None; }
    if !filters.size_ok(size_val) { return None; }
    let size = Some(size_val);
    let item = FoundFile {
        path: path.to_path_buf(),
        modified,
        created,
        size,
        kind: kind.clone(),
        thumb_data: None,
    };
    Some(item)
}

fn _is_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| crate::is_image(s.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel::unbounded;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

    fn recv(rx: &Receiver<ScanEnvelope>) -> ScanEnvelope {
        rx.recv_timeout(Duration::from_secs(60)).expect("scan message")
    }

    fn route_gone(scan_id: u64) -> bool {
        for _ in 0..500 {
            if !SCAN_ROUTERS.lock().unwrap().contains_key(&scan_id) { return true; }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    fn stub_thumb() -> EncodedThumb {
        EncodedThumb { bytes: vec![0xFF, 0xD8, 0xFF, 0xD9], format: crate::utilities::thumbs::ThumbFormat::Jpeg, width: 2, height: 2 }
    }

    fn scripted_generator(path: &Path, _kind: &MediaKind) -> Result<EncodedThumb, String> {
        match path.file_name().and_then(|n| n.to_str()) {
            Some("panic.jpg") => panic!("decoder exploded"),
            Some("fail.jpg") => Err("plain failure".into()),
            _ => Ok(stub_thumb()),
        }
    }

    #[derive(Default)]
    struct TestSink {
        accepted: std::sync::Mutex<Vec<PathBuf>>,
        failed: std::sync::Mutex<Vec<(PathBuf, String)>>,
        fresh: std::sync::Mutex<HashSet<PathBuf>>,
    }

    impl ThumbSink for TestSink {
        fn fresh_key(&self, path: &Path) -> Option<String> {
            self.fresh.lock().unwrap().contains(path).then(|| "fresh-key".to_string())
        }

        fn accept(&self, path: &Path, _thumb: EncodedThumb) -> String {
            self.accepted.lock().unwrap().push(path.to_path_buf());
            format!("key:{}", path.display())
        }

        fn failed(&self, path: &Path, error: &str) {
            self.failed.lock().unwrap().push((path.to_path_buf(), error.to_string()));
        }
    }

    fn no_priority() -> Receiver<ThumbJob> {
        crossbeam::channel::never()
    }

    fn queue_jobs(scan_id: u64, names: &[&str]) -> Receiver<ThumbJob> {
        let (jtx, jrx) = unbounded::<ThumbJob>();
        for name in names {
            route_pending_add(scan_id);
            jtx.send(ThumbJob { path: PathBuf::from(name), kind: MediaKind::Image, scan_id }).unwrap();
        }
        jrx
    }

    fn tiny_jpeg() -> Vec<u8> {
        let img: image::RgbImage = image::ImageBuffer::from_pixel(8, 8, image::Rgb([200, 40, 40]));
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Jpeg).unwrap();
        out
    }

    #[test]
    fn worker_survives_panicking_generator() {
        let scan_id = next_scan_id();
        let (utx, urx) = unbounded::<ScanEnvelope>();
        register_route(scan_id, utx);
        let jrx = queue_jobs(scan_id, &["a.jpg", "panic.jpg", "fail.jpg", "b.jpg"]);
        let pool = try_build_pool("test-worker", 1).unwrap();
        let sink = Arc::new(TestSink::default());
        let worker_sink = sink.clone();
        pool.spawn(move || run_thumb_worker(jrx, no_priority(), scripted_generator, worker_sink));
        let msgs: Vec<ScanMsg> = (0..4).map(|_| recv(&urx).msg).collect();
        assert!(matches!(&msgs[0], ScanMsg::ThumbReady { path, key } if path.ends_with("a.jpg") && key == "key:a.jpg"));
        assert!(matches!(&msgs[1], ScanMsg::ThumbFailed { error, .. } if error.contains("thumbnail worker panicked: decoder exploded")));
        assert!(matches!(&msgs[2], ScanMsg::ThumbFailed { error, .. } if error == "plain failure"));
        assert!(matches!(&msgs[3], ScanMsg::ThumbReady { path, .. } if path.ends_with("b.jpg")));
        assert_eq!(sink.accepted.lock().unwrap().len(), 2);
        assert_eq!(sink.failed.lock().unwrap().len(), 2, "failures reach the sink too");
        route_walk_done(scan_id);
        assert!(route_gone(scan_id));
    }

    #[test]
    fn a_fresh_stored_thumbnail_skips_generation() {
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn counting(_: &Path, _: &MediaKind) -> Result<EncodedThumb, String> {
            CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(stub_thumb())
        }
        let scan_id = next_scan_id();
        let (utx, urx) = unbounded::<ScanEnvelope>();
        register_route(scan_id, utx);
        let jrx = queue_jobs(scan_id, &["cached.jpg", "new.jpg"]);
        let sink = Arc::new(TestSink::default());
        sink.fresh.lock().unwrap().insert(PathBuf::from("cached.jpg"));
        let worker_sink = sink.clone();
        try_build_pool("test-fresh", 1).unwrap().spawn(move || run_thumb_worker(jrx, no_priority(), counting, worker_sink));
        let first = recv(&urx).msg;
        let second = recv(&urx).msg;
        assert!(matches!(&first, ScanMsg::ThumbReady { key, .. } if key == "fresh-key"));
        assert!(matches!(&second, ScanMsg::ThumbReady { key, .. } if key == "key:new.jpg"));
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(sink.accepted.lock().unwrap().len(), 1);
        route_walk_done(scan_id);
        assert!(route_gone(scan_id));
    }

    #[test]
    fn priority_jobs_run_ahead_of_the_scan_backlog() {
        let (scan_tx, scan_rx) = unbounded::<ThumbJob>();
        let (prio_tx, prio_rx) = bounded::<ThumbJob>(8);
        let job = |name: &str, id: u64| ThumbJob { path: PathBuf::from(name), kind: MediaKind::Image, scan_id: id };
        for n in ["s1.jpg", "s2.jpg", "s3.jpg"] {
            scan_tx.send(job(n, 77)).unwrap();
        }
        prio_tx.send(job("p1.jpg", STANDALONE_SCAN_ID)).unwrap();
        prio_tx.send(job("p2.jpg", STANDALONE_SCAN_ID)).unwrap();
        let order: Vec<String> = (0..5).filter_map(|_| next_thumb_job(&scan_rx, &prio_rx)).map(|j| j.path.display().to_string()).collect();
        assert_eq!(order, ["p1.jpg", "p2.jpg", "s1.jpg", "s2.jpg", "s3.jpg"]);
        drop(scan_tx);
        assert!(next_thumb_job(&scan_rx, &prio_rx).is_none(), "a closed scan queue ends the worker");
    }

    #[test]
    fn standalone_jobs_reach_the_sink_without_a_route() {
        let (jtx, jrx) = unbounded::<ThumbJob>();
        jtx.send(ThumbJob { path: PathBuf::from("solo.jpg"), kind: MediaKind::Image, scan_id: STANDALONE_SCAN_ID }).unwrap();
        drop(jtx);
        let sink = Arc::new(TestSink::default());
        run_thumb_worker(jrx, no_priority(), scripted_generator, sink.clone());
        assert_eq!(sink.accepted.lock().unwrap().as_slice(), [PathBuf::from("solo.jpg")]);
    }

    #[test]
    fn unguarded_pool_panic_does_not_abort_the_process() {
        let before = POOL_PANICS.load(Ordering::Relaxed);
        let pool = try_build_pool("test-panic", 2).unwrap();
        pool.spawn(|| panic!("unguarded task panic"));
        let (tx, rx) = unbounded::<u32>();
        pool.spawn(move || tx.send(7).unwrap());
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), 7);
        for _ in 0..300 {
            if POOL_PANICS.load(Ordering::Relaxed) > before { break; }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(POOL_PANICS.load(Ordering::Relaxed) > before);
    }

    #[test]
    fn cancelled_scan_jobs_are_skipped_and_route_is_released() {
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn counting(_: &Path, _: &MediaKind) -> Result<EncodedThumb, String> {
            CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(stub_thumb())
        }
        let scan_id = next_scan_id();
        let (utx, urx) = unbounded::<ScanEnvelope>();
        register_route(scan_id, utx);
        let jrx = queue_jobs(scan_id, &["1.jpg", "2.jpg", "3.jpg"]);
        cancel_scan(scan_id);
        try_build_pool("test-cancel", 1).unwrap().spawn(move || run_thumb_worker(jrx, no_priority(), counting, Arc::new(TestSink::default())));
        route_walk_done(scan_id);
        assert!(route_gone(scan_id));
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
        assert!(urx.try_recv().is_err());
        assert!(!is_cancelled(scan_id));
    }

    #[test]
    fn shallow_scan_lists_only_real_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.jpg"), tiny_jpeg()).unwrap();
        let (tx, rx) = unbounded::<ScanEnvelope>();
        let scan_id = next_scan_id();
        let mut filters = Filters::default();
        filters.root = dir.path().to_path_buf();
        perform_scan_blocking(filters, tx, false, scan_id);
        let mut dirs = Vec::new();
        while let Ok(env) = rx.try_recv() {
            if let ScanMsg::FoundDirBatch(d) = env.msg { dirs.extend(d.into_iter().map(|d| d.path)); }
        }
        assert_eq!(dirs, vec![dir.path().join("sub")]);
    }

    #[test]
    fn new_scan_on_the_same_channel_cancels_the_previous_one() {
        let (tx, _rx) = unbounded::<ScanEnvelope>();
        let (other_tx, _other_rx) = unbounded::<ScanEnvelope>();
        let (first, second, unrelated) = (next_scan_id(), next_scan_id(), next_scan_id());
        register_route(first, tx.clone());
        register_route(unrelated, other_tx);
        assert!(!is_cancelled(first));
        register_route(second, tx);
        assert!(is_cancelled(first), "previous scan of the same explorer is superseded");
        assert!(!is_cancelled(second));
        assert!(!is_cancelled(unrelated), "scans of other explorers are untouched");
        for id in [first, second, unrelated] {
            route_walk_done(id);
        }
    }

    #[test]
    fn route_lives_until_walk_is_done_and_all_jobs_completed() {
        let scan_id = next_scan_id();
        let (utx, _urx) = unbounded::<ScanEnvelope>();
        register_route(scan_id, utx);
        route_pending_add(scan_id);
        route_pending_add(scan_id);
        assert_eq!(route_pending(scan_id), 2);
        route_walk_done(scan_id);
        assert!(SCAN_ROUTERS.lock().unwrap().contains_key(&scan_id));
        route_job_done(scan_id);
        assert!(SCAN_ROUTERS.lock().unwrap().contains_key(&scan_id));
        route_job_done(scan_id);
        assert!(!SCAN_ROUTERS.lock().unwrap().contains_key(&scan_id));
    }

    #[test]
    fn scan_delivers_every_thumbnail_without_dropping_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let jpeg = tiny_jpeg();
        let images = 700usize;
        for i in 0..images {
            std::fs::write(dir.path().join(format!("img_{i:04}.jpg")), &jpeg).unwrap();
        }
        let mut cr3 = vec![0u8; 2048];
        cr3.extend_from_slice(&jpeg);
        std::fs::write(dir.path().join("shot.cr3"), &cr3).unwrap();
        std::fs::write(dir.path().join("broken.arw"), [0x49u8, 0x49, 0x2A, 0x00, 0x08, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        std::fs::write(dir.path().join("clip_a.mp4"), b"not a video").unwrap();
        std::fs::write(dir.path().join("clip_b.mp4"), b"not a video").unwrap();

        init_thumbnail_workers();
        let (tx, rx) = unbounded::<ScanEnvelope>();
        let scan_id = next_scan_id();
        register_route(scan_id, tx.clone());
        let mut filters = Filters::default();
        filters.root = dir.path().to_path_buf();
        filters.include_images = true;
        filters.include_videos = true;
        perform_scan_blocking(filters, tx, false, scan_id);

        let (mut found, mut done) = (0usize, false);
        let (mut updated, mut failed) = (HashSet::new(), HashMap::new());
        let expected = images + 1 + 3;
        let started = Instant::now();
        while updated.len() + failed.len() < expected && started.elapsed() < Duration::from_secs(60) {
            let Ok(env) = rx.recv_timeout(Duration::from_millis(200)) else { continue };
            match env.msg {
                ScanMsg::FoundBatch(b) => found += b.len(),
                ScanMsg::ThumbReady { path, .. } => { updated.insert(path); }
                ScanMsg::ThumbFailed { path, error } => { failed.insert(path, error); }
                ScanMsg::Done => done = true,
                _ => {}
            }
        }
        assert!(done);
        assert_eq!(found, images + 4);
        assert_eq!(updated.len(), images + 1, "every image and the RAW preview must get a thumbnail");
        let broken = failed.iter().find(|(p, _)| p.ends_with("broken.arw")).map(|(_, e)| e.clone()).expect("broken.arw reported");
        assert!(broken.starts_with("raw: "), "{broken}");
        assert_eq!(failed.len(), 3);
        assert!(route_gone(scan_id));
    }

    fn real_arw() -> Option<PathBuf> {
        std::env::var_os("SMART_MEDIA_TEST_ARW").map(PathBuf::from).filter(|p| p.exists())
    }

    #[test]
    fn arw_through_thumb_pool_survives() {
        let Some(arw) = real_arw() else { return };
        init_thumbnail_workers();
        let (tx, rx) = unbounded::<ScanEnvelope>();
        let scan_id = next_scan_id();
        register_route(scan_id, tx);
        enqueue_thumb_job(&arw, MediaKind::Image, scan_id);
        match recv(&rx).msg {
            ScanMsg::ThumbReady { key, .. } => eprintln!("real ARW thumbnail stored as {key}"),
            other => panic!("expected a thumbnail, got {other:?}"),
        }
        route_walk_done(scan_id);
        assert!(route_gone(scan_id));
    }

    #[test]
    fn arw_without_preview_through_thumb_pool_fails_cleanly() {
        let Some(arw) = real_arw() else { return };
        let mut bytes = std::fs::read(&arw).unwrap();
        for i in 0..bytes.len().saturating_sub(3) {
            if bytes[i..i + 3] == [0xFF, 0xD8, 0xFF] { bytes[i] = 0; }
        }
        let dir = tempfile::tempdir().unwrap();
        let broken = dir.path().join("no_preview.arw");
        std::fs::write(&broken, bytes).unwrap();
        init_thumbnail_workers();
        let (tx, rx) = unbounded::<ScanEnvelope>();
        let scan_id = next_scan_id();
        register_route(scan_id, tx);
        enqueue_thumb_job(&broken, MediaKind::Image, scan_id);
        match recv(&rx).msg {
            ScanMsg::ThumbFailed { error, .. } => {
                eprintln!("rawkit failure surfaced as: {error}");
                assert!(error.contains("rawkit panicked"), "{error}");
            }
            other => panic!("expected a failed thumbnail, got {other:?}"),
        }
        enqueue_thumb_job(&arw, MediaKind::Image, scan_id);
        assert!(matches!(recv(&rx).msg, ScanMsg::ThumbReady { .. }), "pool must keep working after the failure");
    }

    #[cfg(windows)]
    #[test]
    fn timing_recursive_progress() {
        // Use C:\ as the scan root by default; allow override via env
        let root_path: std::path::PathBuf = std::env::var("SMART_MEDIA_SCAN_TEST_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("C:\\"));

        let mut filters = Filters::default();
        filters.root = root_path.clone();
        // Include common media and archives to keep a representative subset of files
        filters.include_images = true;
        filters.include_videos = true;
        filters.include_archives = true;

        let (tx, rx) = unbounded::<ScanEnvelope>();
        let scan_id = next_scan_id();

        // Spawn receiver to timestamp messages as they arrive
        let recv_handle = std::thread::spawn(move || {
            let mut last_progress: Option<Instant> = None;
            let t0 = Instant::now();
            let mut found_batches = 0usize;
            let mut progress_events = 0usize;
            let mut cancelled = false;
            while let Ok(env) = rx.recv() {
                match env.msg {
                    ScanMsg::Progress { scanned, total } => {
                        progress_events += 1;
                        let now = Instant::now();
                        if let Some(lp) = last_progress.replace(now) {
                            let dt = now.duration_since(lp);
                            eprintln!("test-scan[{scan_id}]: progress scanned={scanned} total={total} dt={:?}", dt);
                        } else {
                            eprintln!("test-scan[{scan_id}]: first progress at {:?}", now.duration_since(t0));
                        }
                        // After a few ticks, cancel to bound test runtime
                        if progress_events >= 3 && !cancelled {
                            cancelled = true;
                            super::cancel_scan(scan_id);
                            eprintln!("test-scan[{scan_id}]: cancelling after {progress_events} progress events");
                        }
                        // Time budget safety: cancel after ~20s
                        if !cancelled && t0.elapsed() > Duration::from_secs(20) {
                            cancelled = true;
                            super::cancel_scan(scan_id);
                            eprintln!("test-scan[{scan_id}]: cancelling due to time budget");
                        }
                    }
                    ScanMsg::FoundBatch(_) => { found_batches += 1; }
                    ScanMsg::Done => {
                        let total = t0.elapsed();
                        eprintln!("test-scan[{scan_id}]: done; found_batches={} progress_events={} total={:?}", found_batches, progress_events, total);
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Run the blocking scanner on this thread
        perform_scan_blocking(filters, tx, true, scan_id);

        // Ensure receiver completes
        recv_handle.join().unwrap();
    }
}
