
use crate::utilities::types::{DateField, Filters, FoundFile, MediaKind, DirItem, is_image, is_video};
use std::sync::atomic::{AtomicU64, Ordering};
use jwalk::{WalkDirGeneric, Parallelism};
use crossbeam::channel::Sender;
use chrono::{DateTime, Local}; 
use once_cell::sync::Lazy;
use std::time::SystemTime;
use std::path::Path;

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
    // Perform the blocking filesystem traversal on a blocking thread.
    if let Err(e) = tokio::task::spawn_blocking(move || perform_scan_blocking(filters, tx, recursive, scan_id)).await {
        log::warn!("spawn_blocking scan join error: {e:?}");
    }
}

fn perform_scan_blocking(filters: Filters, tx: Sender<ScanEnvelope>, recursive: bool, scan_id: u64) {
    if is_cancelled(scan_id) {
        let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
        clear_cancel(scan_id);
        return;
    }
    
    let root = if filters.root.as_os_str().is_empty() {
        match std::env::current_dir().and_then(|p| std::path::absolute(p)) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Error(e.to_string()) });
                let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
                return;
            }
        }
    } else { filters.root.clone() };
    if !root.exists() {
        let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Error(format!("Root does not exist: {}", root.display())) });
        let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
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
    let _ = tx.try_send(ScanEnvelope { 
        scan_id, 
        msg: ScanMsg::Progress { scanned: 0, total } 
    });

    let filters_clone = filters.clone();
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16);
    let mut batch: Vec<FoundFile> = vec![];
    let mut scanned_paths: usize = 0;
    let parallelism = Parallelism::RayonNewPool(std::cmp::max(8, std::cmp::min(logical, 16)));

    let excluded_dirs = filters_clone.recursive_excluded_dirs.clone();
    let excluded_exts = filters_clone.recursive_excluded_exts.clone();

    // Precompute lowercase excluded extensions into a small HashSet for faster matching.
    use std::collections::HashSet as _HashSet;
    let excluded_exts_set: _HashSet<String> = excluded_exts.iter().map(|s| s.to_ascii_lowercase()).collect();

    // Use jwalk for both recursive and shallow traversal.
    let walker = WalkDirGeneric::<((), Option<u64>)>::new(&root)
        .skip_hidden(false)
        .follow_links(false)
        .parallelism(parallelism)
        // Depth semantics: 0 = root only, 1 = root children only.
        // For shallow scans we want the immediate children, not just the root dir itself.
        .max_depth(if recursive { usize::MAX } else { 1 })
        .process_read_dir(move |_depth, dir_path, _state, entries| {
            // Only enforce recursive excludes during recursive scans.
            if recursive {
                if excluded_dirs.iter().any(|d| dir_path.starts_with(d)) {
                    entries.clear();
                    return;
                }
            }

            // Filter & optionally sort entries in-place.
            entries.retain(|res| {
                if let Ok(entry) = res {
                    let p = entry.path();
                    if entry.file_type.is_dir() {
                        // In recursive mode, skip excluded directories entirely (and their children).
                        if recursive && excluded_dirs.iter().any(|d| p.starts_with(d)) {
                            return false;
                        }
                        return true; // keep directory (walker decides whether to descend based on max_depth)
                    }
                    if entry.file_type.is_file() {
                        if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                            // Quick pre-filter: only keep potentially supported media.
                            if !crate::is_supported_media_ext(ext.as_str()) { return false; }
                            // Only apply extension excludes in recursive mode (as per Filters contract).
                            if recursive && excluded_exts_set.contains(&ext) { return false; }
                            return true;
                        } else { return false; }
                    }
                    return false;
                }
                true
            });
            entries.sort_by(|a,b| match (a,b) { (Ok(ae), Ok(be)) => ae.file_name.cmp(&be.file_name), _ => std::cmp::Ordering::Equal });
        });

    // For shallow scans we also collect the immediate subdirectories to send in a single batch at the end.
    let mut dir_batch: Vec<DirItem> = if recursive { Vec::new() } else { Vec::with_capacity(128) };

    let mut idx: usize = 0;
    for dir_entry_result in walker {
        if is_cancelled(scan_id) {
            let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Done });
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
                    if let Some(found) = process_path_collect(path, &filters_clone, after, before) {
                        batch.push(found);
                        let _ = tx.try_send(ScanEnvelope { 
                            scan_id, 
                            msg: ScanMsg::FoundBatch(std::mem::take(&mut batch)) 
                        });
                    }
                    if recursive {
                        scanned_paths += 1;
                        if scanned_paths % PROGRESS_EVERY == 0 {
                            let _ = tx.try_send(ScanEnvelope { 
                                scan_id, 
                                msg: ScanMsg::Progress { scanned: scanned_paths, total: 0 } 
                            });
                        }
                    } else {
                        scanned += 1;
                        let display_scanned = if total > 0 { scanned.min(total) } else { scanned };
                        if scanned % 25 == 0 {
                            let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: display_scanned, total } });
                        }
                        if scanned % 200 == 0 { std::thread::yield_now(); }
                    }
                }
            }
            Err(err) => {
                let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Error(format!("walk error: {err}")) });
            }
        }
        if idx % 1000 == 0 { std::thread::yield_now(); }
        idx += 1;
    }

    // Flush remaining batches and final progress per mode
    if !recursive {
        if !dir_batch.is_empty() { let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::FoundDirBatch(dir_batch) }); }
        if !batch.is_empty() { let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let final_scanned = if total > 0 { scanned.min(total) } else { scanned };
        let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: final_scanned, total } });
    } else {
        if !batch.is_empty() { let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::FoundBatch(batch) }); }
        let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Progress { scanned: scanned_paths, total: 0 } });
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
    let _ = tx.try_send(ScanEnvelope { scan_id, msg: ScanMsg::Done }); 
}

fn process_path_collect(
    path: &Path,
    filters: &Filters,
    after: Option<chrono::NaiveDateTime>,
    before: Option<chrono::NaiveDateTime>,
) -> Option<FoundFile> {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) { if name.starts_with("._") { return None; } }
    let kind = is_media_kind(path);
    if !(
        (filters.include_images && kind == MediaKind::Image) 
        || (filters.include_videos && kind == MediaKind::Video)
    ) {
        return None;
    }
    let md = match path.metadata() {
        Ok(m) => m,
        Err(_) => return None,
    };
    let modified = md.modified().ok().map(systemtime_to_local);
    let created = md.created().ok().map(systemtime_to_local);
    match filters.date_field {
        DateField::Modified => {
            if let Some(m) = modified {
                if let Some(a) = after {
                    if m.naive_local() < a {
                        return None;
                    }
                }
                if let Some(b) = before {
                    if m.naive_local() > b {
                        return None;
                    }
                }
            }
        }
        DateField::Created => {
            if let Some(c) = created {
                if let Some(a) = after {
                    if c.naive_local() < a {
                        return None;
                    }
                }
                if let Some(b) = before {
                    if c.naive_local() > b {
                        return None;
                    }
                }
            }
        }
    }
    let size_val = md.len();
    // Heuristic: skip likely icons/asset images
    if filters.skip_icons {
        // 1) Common icon extensions
        let is_icon_ext = path.extension().and_then(|e| e.to_str()).map(|s| s.eq_ignore_ascii_case("ico")).unwrap_or(false);
        // 2) Very small file size (<= min_size_bytes if provided, else 10 KB)
        let tiny_thresh = filters.min_size_bytes.unwrap_or(10 * 1024);
        let is_tiny_file = size_val <= tiny_thresh;
        // 3) Very small image dimensions (<= 64 px on both sides)
        let is_tiny_image = if is_image_ext(path) {
            // Try cheap decode to get dimensions only; ignore errors
            image::ImageReader::open(path)
                .ok()
                .and_then(|r| r.with_guessed_format().ok())
                .and_then(|r| r.into_dimensions().ok())
                .map(|(w,h)| w <= 64 && h <= 64)
                .unwrap_or(false)
        } else { false };
        if is_icon_ext || (is_tiny_file && is_tiny_image) {
            return None;
        }
    }
    if let Some(minb) = filters.min_size_bytes { if size_val < minb { return None; } }
    if let Some(maxb) = filters.max_size_bytes { if size_val > maxb { return None; } }
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

fn is_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| crate::is_image(s.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}
