use crate::utilities::types::{FoundFile, MediaKind, IMAGE_EXTS, VIDEO_EXTS};
use crossbeam::channel::{unbounded, Receiver, Sender};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Local};
use std::time::Instant;
use log;

#[derive(Debug)]
pub enum FilesScanMsg {
    Found(FoundFile),
    FoundBatch(Vec<FoundFile>),
    Progress(usize), // scanned so far
    UpdateThumb { path: PathBuf, thumb: String },
    Done,
}

pub struct Files {
    pub current_path: PathBuf,
    pub scan_results: Vec<FoundFile>,
    pub scanning: bool,
    pub scan_rx: Option<Receiver<FilesScanMsg>>,
    pub scanned_count: usize,
    pub started_at: Instant,
    pub total_enumerated: usize,
    pub last_ui_len: usize, // number of items already published to UI
}

impl Files {
    pub fn new() -> Self {
        let current_path = std::path::absolute(std::env::current_dir().unwrap()).unwrap_or_else(|_| PathBuf::from("."));
        Self {
            current_path,
            scan_results: Vec::new(),
            scanning: false,
            scan_rx: None,
            scanned_count: 0,
            started_at: Instant::now(),
            total_enumerated: 0,
            last_ui_len: 0,
        }
    }

    pub fn begin_scan(&mut self, root: PathBuf, include_images: bool, include_videos: bool) {
        self.current_path = root.clone();
        self.scan_results.clear();
        self.scanning = true;
        self.scanned_count = 0;
        self.started_at = Instant::now();
        self.last_ui_len = 0;
        let (tx, rx) = unbounded::<FilesScanMsg>();
        self.scan_rx = Some(rx);
            start_scan_thread(root, include_images, include_videos, tx, None);
    }

    // Begin scan but skip any paths present in 'skip' (already cached) and pre-seed scan_results with provided preloaded list.
    pub fn begin_scan_with_skip(&mut self, root: PathBuf, include_images: bool, include_videos: bool, preloaded: Vec<FoundFile>, skip: HashSet<PathBuf>) {
        self.current_path = root.clone();
        self.scan_results.clear();
        // Pre-seed
        self.scan_results.extend(preloaded.into_iter());
        self.last_ui_len = self.scan_results.len();
        self.scanning = true;
        self.scanned_count = 0; // will count only newly found
        self.started_at = Instant::now();
        let (tx, rx) = unbounded::<FilesScanMsg>();
        self.scan_rx = Some(rx);
        start_scan_thread(root, include_images, include_videos, tx, Some(skip));
    }

    // Returns Some(new_items) if there are newly enumerated files since last poll.
    pub fn poll_scan(&mut self) -> Option<Vec<FoundFile>> {
        let mut had_new = false;
        let mut drained_msgs = 0usize;
        let mut found_cnt = 0usize;
        let mut batch_cnt = 0usize;
        let mut thumb_cnt = 0usize;
        if let Some(rx) = &self.scan_rx {
            let backlog = rx.len();
            log::info!("[poll_scan] start backlog={backlog} current_ui_len={} scanned_count={} scanning={}", self.last_ui_len, self.scanned_count, self.scanning);
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    FilesScanMsg::Found(f) => {
                        self.scanned_count += 1; // counting accepted media files
                        self.scan_results.push(f);
                        had_new = true;
                        found_cnt += 1;
                    }
                    FilesScanMsg::FoundBatch(mut batch) => {
                        self.scanned_count += batch.len();
                        self.scan_results.append(&mut batch);
                        had_new = true;
                        batch_cnt += 1;
                    }
                    FilesScanMsg::Progress(p) => { log::info!("[poll_scan] progress msg scanned={p}"); }
                    FilesScanMsg::UpdateThumb { path, thumb } => {
                        if let Some(found) = self.scan_results.iter_mut().find(|f| f.path == path) { found.thumb_data = Some(thumb); }
                        thumb_cnt += 1;
                    }
                    FilesScanMsg::Done => {
                        self.scanning = false;
                        log::info!("[poll_scan] received Done. total_scanned={} final_len={} elapsed_ms={}", self.scanned_count, self.scan_results.len(), self.started_at.elapsed().as_millis());
                        break;
                    }
                }
                drained_msgs += 1;
            }
        }
        log::info!("[poll_scan] drained_msgs={drained_msgs} found_msgs={found_cnt} batch_msgs={batch_cnt} thumb_updates={thumb_cnt} total_len={} had_new={}", self.scan_results.len(), had_new);
        if had_new && self.scan_results.len() > self.last_ui_len {
            let delta = self.scan_results[self.last_ui_len..].to_vec();
            self.last_ui_len = self.scan_results.len();
            log::info!("[poll_scan] publishing delta_len={} new_last_ui_len={}", delta.len(), self.last_ui_len);
            return Some(delta);
        }
        None
    }

    pub fn go_up(&mut self) {
        if let Some(parent) = self.current_path.parent() {
            self.current_path = parent.to_path_buf();
        }
    }
}

// Helper: convert a FoundFile (basic scan result) into a minimal Thumbnail row for persistence
pub fn file_to_thumbnail(f: &crate::utilities::types::FoundFile) -> Option<crate::Thumbnail> {
    use chrono::Utc;
    let file_type = f.path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
    let ft_string = if let Some(ext) = file_type.clone() {
        if crate::utilities::types::IMAGE_EXTS.iter().any(|e| *e == ext) { "image".to_string() }
        else if crate::utilities::types::VIDEO_EXTS.iter().any(|e| *e == ext) { "video".to_string() }
        else { ext }
    } else { "other".into() };
    let md = std::fs::metadata(&f.path).ok();
    let modified = md.as_ref().and_then(|m| m.modified().ok()).map(|st| chrono::DateTime::<chrono::Utc>::from(st));
    Some(crate::Thumbnail {
        id: None,
        db_created: Utc::now().into(),
        path: f.path.display().to_string(),
        filename: f.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
        file_type: ft_string,
        size: f.size.unwrap_or(0),
        description: None,
        caption: None,
        tags: Vec::new(),
        category: None,
        embedding: None,
        thumbnail_b64: f.thumb_data.clone(), // may be None; raw data URL string stored earlier version
        modified: if let Some(date) = modified { Some(date.into()) } else { Some(Utc::now().into()) },
        hash: None,
    })
}

fn systemtime_to_local(st: std::time::SystemTime) -> DateTime<Local> { DateTime::<Local>::from(st) }

fn start_scan_thread(root: PathBuf, include_images: bool, include_videos: bool, sender: Sender<FilesScanMsg>, skip_paths: Option<HashSet<PathBuf>>) {
    std::thread::spawn(move || {
        use jwalk::{WalkDir, Parallelism};
    use std::time::Instant;
    let t_start = Instant::now();
    log::info!("[scan] thread started root={} images={} videos={}", root.display(), include_images, include_videos);
        let skip_paths = skip_paths.unwrap_or_default();
        // Early filtering of directory entries to only descend/keep media candidates
        fn process_by(
            _depth: Option<usize>,
            _dir_path: &Path,
            _state: &mut (),
            children: &mut Vec<Result<jwalk::DirEntry<((), ())>, jwalk::Error>>,
            include_images: bool,
            include_videos: bool,
        ) {
            children.retain(|res| match res {
                Ok(entry) => {
                    if entry.file_type().is_dir() { return true; }
                    let ext = entry.path().extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                    if let Some(ext) = ext {
                        if include_images && IMAGE_EXTS.iter().any(|e| *e == ext) { return true; }
                        if include_videos && VIDEO_EXTS.iter().any(|e| *e == ext) { return true; }
                    }
                    false
                }
                Err(_) => false,
            });
        }

        let inc_img = include_images;
        let inc_vid = include_videos;
        let mut collected: Vec<FoundFile> = Vec::new();
        let mut scanned = 0usize;
        let t_walker_ready = Instant::now();
        log::info!("[scan] walker constructed in {:.2?}", t_walker_ready.duration_since(t_start));

    // Batch buffer to reduce channel send overhead & UI churn
    let mut batch: Vec<FoundFile> = Vec::with_capacity(128);
    let mut entries_seen: u64 = 0;

    for entry in WalkDir::new(&root)
        .follow_links(false)
        .skip_hidden(false)
        .process_read_dir(move |d,p,s,children| process_by(d,p,s,children, inc_img, inc_vid))
        .parallelism(Parallelism::RayonNewPool(16))
        .into_iter() 
    {
        if let Ok(e) = entry {
            entries_seen += 1;
            if e.file_type().is_dir() { continue; }
            let p = e.path();
            if skip_paths.contains(&p) { continue; }
            // Single extension extraction & classification (avoid extra metadata/syscalls)
            let mut kind_opt: Option<MediaKind> = None;
            if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                if include_images && IMAGE_EXTS.iter().any(|e| *e == ext) { kind_opt = Some(MediaKind::Image); }
                else if include_videos && VIDEO_EXTS.iter().any(|e| *e == ext) { kind_opt = Some(MediaKind::Video); }
            }
            if let Some(kind) = kind_opt {
                if let Ok(md) = e.metadata() { // use jwalk provided metadata
                    let modified = md.modified().ok().map(systemtime_to_local);
                    let created = md.created().ok().map(systemtime_to_local);
                    let size = Some(md.len());
                    let f = FoundFile { path: p.to_path_buf(), modified, created, size, kind, thumb_data: None };
                    batch.push(f.clone());
                    collected.push(f);
                    scanned += 1;
                    if scanned % 50 == 0 { let _ = sender.send(FilesScanMsg::Progress(scanned)); }
                    if batch.len() >= 128 { let _ = sender.send(FilesScanMsg::FoundBatch(std::mem::take(&mut batch))); }
                    if scanned % 5000 == 0 {
                        let elapsed = t_start.elapsed().as_secs_f32();
                        let rate = if elapsed > 0.0 { scanned as f32 / elapsed } else { 0.0 };
                        log::info!("[scan] enumerated {} media files (entries_seen={}) in {:.2?} ({:.1}/s)", scanned, entries_seen, t_start.elapsed(), rate);
                    }
                }
            }
            if entries_seen % 10000 == 0 { log::info!("[scan] entries_seen={} scanned_media={} batch_pending={} elapsed_ms={}", entries_seen, scanned, batch.len(), t_start.elapsed().as_millis()); }
        }
    }
    if !batch.is_empty() { let _ = sender.send(FilesScanMsg::FoundBatch(batch)); }
        let enum_elapsed = t_start.elapsed();
        let rate = if enum_elapsed.as_secs_f32() > 0.0 { scanned as f32 / enum_elapsed.as_secs_f32() } else { 0.0 };
        log::info!("[scan] enumeration complete: {} media files (entries_seen={}) in {:.2?} ({:.1}/s)", scanned, entries_seen, enum_elapsed, rate);
        let _ = sender.send(FilesScanMsg::Done);
    });
}
