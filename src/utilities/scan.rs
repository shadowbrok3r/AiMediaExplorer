use crate::utilities::types::{Filters, FoundFile, MediaKind, DirItem, is_image, is_video, is_archive};
use crate::utilities::filtering::FiltersExt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use jwalk::{WalkDirGeneric, Parallelism};
use crossbeam::channel::{Sender, Receiver, bounded};
use chrono::{DateTime, Local}; 
use once_cell::sync::Lazy;
use std::time::SystemTime;
use std::path::Path;
use std::time::Duration;
use rayon::ThreadPool;
use std::sync::OnceLock;

// Track cancelled scan ids
static CANCELLED_SCANS: Lazy<std::sync::Mutex<std::collections::HashSet<u64>>> = Lazy::new(|| std::sync::Mutex::new(Default::default()));
static SCAN_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
/// send a progress update every N scanned paths in recursive mode
const PROGRESS_EVERY: usize = 5_000; 

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

// --- Dedicated thumbnail pipeline (bounded queue + small worker pool) ---
#[derive(Clone, Debug)]
pub struct ThumbJob {
    pub path: std::path::PathBuf,
    pub kind: MediaKind,
    pub scan_id: u64,
}

static THUMB_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let workers = logical.clamp(2, 4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|i| format!("thumb-{}", i))
        .build()
        .expect("thumb pool")
});

static THUMB_TX: OnceLock<Sender<ThumbJob>> = OnceLock::new();
static IMG_INFLIGHT: AtomicUsize = AtomicUsize::new(0);
static VID_INFLIGHT: AtomicUsize = AtomicUsize::new(0);

// Route scan updates to the correct UI (per scan_id)
static SCAN_ROUTERS: Lazy<std::sync::Mutex<std::collections::HashMap<u64, Sender<ScanEnvelope>>>> =
    Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

pub fn init_thumbnail_workers() {
    if THUMB_TX.get().is_some() { return; }
    let (tx, rx): (Sender<ThumbJob>, Receiver<ThumbJob>) = bounded(256);
    let _ = THUMB_TX.set(tx.clone());
    start_thumbnail_workers(rx);
}

fn start_thumbnail_workers(rx: Receiver<ThumbJob>) {
    let workers = 3usize.min(THUMB_POOL.current_num_threads());
    for _ in 0..workers {
        let rx = rx.clone();
        THUMB_POOL.spawn(move || {
            while let Ok(job) = rx.recv() {
                if is_cancelled(job.scan_id) { 
                    // decrement the proper counter if we had incremented on enqueue but got cancelled before work
                    match job.kind { 
                        MediaKind::Image => { IMG_INFLIGHT.fetch_sub(1, Ordering::Relaxed); },
                        MediaKind::Video => { VID_INFLIGHT.fetch_sub(1, Ordering::Relaxed); },
                        _ => {}
                    }
                    continue; 
                }
                if let Some(thumb_b64) = generate_thumb(&job.path, &job.kind) {
                    // Route thumbnail to the UI that owns this scan_id
                    if let Ok(map) = SCAN_ROUTERS.lock() {
                        if let Some(tx) = map.get(&job.scan_id) {
                            let _ = tx.send(ScanEnvelope {
                                scan_id: job.scan_id,
                                msg: ScanMsg::UpdateThumb { path: job.path.clone(), thumb: thumb_b64 },
                            });
                        }
                    }
                }
                match job.kind {
                    MediaKind::Image => { IMG_INFLIGHT.fetch_sub(1, Ordering::Relaxed); },
                    MediaKind::Video => { VID_INFLIGHT.fetch_sub(1, Ordering::Relaxed); },
                    _ => {}
                }
            }
        });
    }
}

fn enqueue_thumb_job(path: &Path, kind: MediaKind, scan_id: u64) {
    if let Some(tx) = THUMB_TX.get() {
        // increment inflight on success; bounded send applies back-pressure
        let job = ThumbJob { path: path.to_path_buf(), kind: kind.clone(), scan_id };
        if tx.send(job).is_ok() {
            match kind { 
                MediaKind::Image => { IMG_INFLIGHT.fetch_add(1, Ordering::Relaxed); },
                MediaKind::Video => { VID_INFLIGHT.fetch_add(1, Ordering::Relaxed); },
                _ => {}
            }
        }
    }
}

fn generate_thumb(path: &Path, kind: &MediaKind) -> Option<String> {
    match kind {
        MediaKind::Image => crate::utilities::thumbs::generate_image_thumb_data(path).ok(),
        MediaKind::Video => {
            #[cfg(windows)]
            { return crate::utilities::thumbs::generate_video_thumb_data(path).ok(); }
            #[cfg(not(windows))]
            { return None; }
        }
        _ => None,
    }
}

#[derive(Debug)]
pub enum ScanMsg {
    Found(FoundFile),
    FoundBatch(Vec<FoundFile>),
    FoundDir(DirItem),
    FoundDirBatch(Vec<DirItem>),
    UpdateThumb {
        path: std::path::PathBuf,
        thumb: String,
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
    if let Ok(mut map) = SCAN_ROUTERS.lock() { map.insert(scan_id, tx.clone()); }
    // Perform the blocking filesystem traversal on a blocking thread.
    if let Err(e) = tokio::task::spawn_blocking(move || perform_scan_blocking(filters, tx, recursive, scan_id)).await {
        log::warn!("spawn_blocking scan join error: {e:?}");
    }
}

fn perform_scan_blocking(filters: Filters, tx: Sender<ScanEnvelope>, recursive: bool, scan_id: u64) {
    if is_cancelled(scan_id) {
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
        clear_cancel(scan_id);
        return;
    }
    
    let root = if filters.root.as_os_str().is_empty() {
        match std::env::current_dir().and_then(|p| std::path::absolute(p)) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Error(e.to_string()) });
                let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
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
        return;
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
    let mut scanned_paths: usize = 0;
    let parallelism = Parallelism::RayonNewPool(std::cmp::max(8, std::cmp::min(logical, 16)));

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
                excluded_dirs.iter().any(|excluded| {
                    // Check if the path components match, ignoring drive letters
                    let excluded_path = std::path::Path::new(excluded);
                    let path_components: Vec<_> = path.components().collect();
                    let excluded_components: Vec<_> = excluded_path.components().collect();
                    
                    // If excluded path is absolute, try both exact match and drive-agnostic match
                    if excluded_path.is_absolute() {
                        // First try exact match
                        if path.starts_with(excluded) {
                            return true;
                        }
                        // Then try drive-agnostic match - skip the first component if it's a drive
                        if let (Some(std::path::Component::Prefix(_)), Some(std::path::Component::Prefix(_))) 
                            = (path_components.first(), excluded_components.first()) {
                            // Both have drive prefixes, compare without them
                            if path_components.len() >= excluded_components.len() {
                                let path_no_drive = &path_components[1..];
                                let excluded_no_drive = &excluded_components[1..];
                                return path_no_drive.starts_with(excluded_no_drive);
                            }
                        }
                    } else {
                        // Relative path - check if any suffix of the path matches
                        if path_components.len() >= excluded_components.len() {
                            for start_idx in 0..=(path_components.len() - excluded_components.len()) {
                                let path_slice = &path_components[start_idx..start_idx + excluded_components.len()];
                                if path_slice == excluded_components.as_slice() {
                                    return true;
                                }
                            }
                        }
                    }
                    false
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

    // For shallow scans we also collect the immediate subdirectories to send in a single batch at the end.
    let mut dir_batch: Vec<DirItem> = if recursive { Vec::new() } else { Vec::with_capacity(128) };

    let mut idx: usize = 0;
    // Collect videos for deferred thumbnailing
    let mut videos_for_later: Vec<std::path::PathBuf> = Vec::new();
    // Encrypted archives found during this scan
    let mut encrypted_archives: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for dir_entry_result in walker {
        if is_cancelled(scan_id) {
            let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
            clear_cancel(scan_id);
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
                    // Detect if this is an encrypted ZIP so we can prompt later
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                        if ext == "zip" {
                            if let Ok(file) = std::fs::File::open(path) {
                                if let Ok(mut z) = zip::ZipArchive::new(file) {
                                    let mut found = false;
                                    let to_check = z.len().min(8);
                                    for i in 0..to_check {
                                        if let Ok(entry) = z.by_index(i) {
                                            let enc = entry.encrypted();
                                            if enc { found = true; break; }
                                        }
                                    }
                                    if found { encrypted_archives.insert(path.to_string_lossy().to_string()); }
                                }
                            }
                        }
                    }
                    if let Some(found) = process_path_collect(path, &filters, after, before) {
                        batch.push(found);
                        let _ = tx.send(ScanEnvelope { 
                            scan_id, 
                            msg: ScanMsg::FoundBatch(std::mem::take(&mut batch)) 
                        });
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
    if !recursive {
        if !dir_batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundDirBatch(dir_batch) }); }
        if !batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let final_scanned = if total > 0 { scanned.min(total) } else { scanned };
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: final_scanned, total } });
    } else {
        if !batch.is_empty() { let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: scanned_paths, total: 0 } });
    }

    // Defer video thumbnails until image thumbnails are complete
    // Wait briefly for image inflight to drain (respect cancellation)
    let mut spins = 0usize;
    while IMG_INFLIGHT.load(Ordering::Relaxed) > 0 {
        if is_cancelled(scan_id) { break; }
        std::thread::sleep(Duration::from_millis(15));
        spins += 1;
        if spins > 2_000 { break; } // ~30s safety cap
    }
    for vpath in videos_for_later.into_iter() {
        enqueue_thumb_job(&vpath, MediaKind::Video, scan_id);
    }

    // If we found encrypted archives, notify UI before Done
    if !encrypted_archives.is_empty() {
        let _ = tx.send(ScanEnvelope { scan_id, msg: ScanMsg::EncryptedArchives(encrypted_archives.into_iter().collect()) });
    }
    finish(&tx, scan_id);
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
    // Unregister router mapping for this scan
    if let Ok(mut map) = SCAN_ROUTERS.lock() { let _ = map.remove(&scan_id); }
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
