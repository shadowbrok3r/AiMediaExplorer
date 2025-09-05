use std::{path::PathBuf, sync::Arc, collections::HashMap};
use tokio::sync::Mutex;

use crate::database::DB;

impl super::AISearchEngine {
    /// Hydrate minimal metadata (no thumbnails/base64/embeddings) for a set of directory file paths.
    /// Returns number of rows inserted. Only paths already known in cached_paths and not yet in self.files are hydrated.
    pub async fn hydrate_directory_paths(&self, dir_paths: &[String]) -> usize {
        if dir_paths.is_empty() { return 0; }
        // Snapshot existing in-memory paths into a HashSet for fast exclusion
        let existing: std::collections::HashSet<String> = {
            let files = self.files.lock().await;
            files.iter().map(|f| f.path.clone()).collect()
        };
        // Filter to those we know have cached rows but are not loaded yet.
        let cached_guard = self.cached_paths.lock().await;
        let mut need: Vec<String> = Vec::new();
        need.reserve(dir_paths.len());
        for p in dir_paths.iter() {
            if !existing.contains(p) && cached_guard.contains(p) {
                need.push(p.clone());
            }
        }
        drop(cached_guard);
        if need.is_empty() { return 0; }

        let mut inserted = 0usize;
        // Chunk membership queries to avoid overly large parameter arrays.
        const CHUNK: usize = 256;
        for chunk in need.chunks(CHUNK) {
            let chunk_vec: Vec<String> = chunk.iter().cloned().collect();
            // Only select minimal fields. (thumbnail_b64 intentionally excluded)
            let sql = "SELECT path, filename, file_type, size, modified, hash, description, caption, tags, category FROM thumbnails WHERE array::find($paths, path) != NONE";
            match crate::database::DB.query(sql).bind(("paths", chunk_vec)).await {
                Ok(mut resp) => {
                    let rows: Result<Vec<serde_json::Value>, _> = resp.take(0);
                    match rows {
                        Ok(vals) => {
                            let mut files_guard = self.files.lock().await;
                            for v in vals.into_iter() {
                                if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
                                    if files_guard.iter().any(|f| f.path == p) { continue; }
                                    // Convert modified -> chrono::Local if present
                                    let modified_local = v.get("modified").and_then(|m| m.as_str()).and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()).map(|dt| dt.with_timezone(&chrono::Local));
                                    let fm = super::FileMetadata {
                                        id: None,
                                        path: p.to_string(),
                                        filename: v.get("filename").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                                        file_type: v.get("file_type").and_then(|x| x.as_str()).unwrap_or("other").to_string(),
                                        size: v.get("size").and_then(|x| x.as_i64()).unwrap_or(0) as u64,
                                        modified: modified_local,
                                        created: modified_local,
                                        thumbnail_path: None,
                                        thumb_b64: None, // defer heavy field
                                        hash: v.get("hash").and_then(|x| x.as_str()).map(|s| s.to_string()),
                                        description: v.get("description").and_then(|x| x.as_str()).map(|s| s.to_string()),
                                        caption: v.get("caption").and_then(|x| x.as_str()).map(|s| s.to_string()),
                                        tags: v.get("tags").and_then(|x| x.as_array()).map(|arr| arr.iter().filter_map(|e| e.as_str().map(|s| s.to_string())).collect()).unwrap_or_default(),
                                        category: v.get("category").and_then(|x| x.as_str()).map(|s| s.to_string()),
                                        embedding: None, // not loaded yet
                                        similarity_score: None,
                                        clip_embedding: None,
                                        clip_similarity_score: None,
                                    };
                                    files_guard.push(fm);
                                    inserted += 1;
                                }
                            }
                        }
                        Err(e) => { log::warn!("[AI] hydrate_directory_paths parse error: {e}"); }
                    }
                }
                Err(e) => { log::warn!("[AI] hydrate_directory_paths query failed: {e}"); }
            }
        }
        if inserted > 0 { log::info!("[AI] Hydrated {} minimal rows for directory", inserted); }
        inserted
    }
    pub fn new() -> Self {
        Self {
            files: Arc::new(Mutex::new(Vec::new())),
            path_to_id: Arc::new(Mutex::new(HashMap::new())),
            indexing_in_progress: Arc::new(Mutex::new(HashMap::new())),
            cached_paths: Arc::new(Mutex::new(std::collections::HashSet::new())),
            auto_descriptions_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_clip_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            index_tx: Arc::new(Mutex::new(None)),
            index_queue_len: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            index_active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            index_completed: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            clip_engine: Arc::new(Mutex::new(None)),
            active_vision_path: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Apply a fully generated Thumbnail (vision metadata) to in-memory metadata & persist without triggering generation.
    pub async fn apply_vision_description(&self, path: &str, vd: &crate::ai::generate::VisionDescription) -> anyhow::Result<()> {
        // Fast path: try to update if already present.
        {
            let mut files = self.files.lock().await;
            if let Some(f) = files.iter_mut().find(|f| f.path == path) {
                 f.description = Some(vd.description.clone());
                 f.caption = Some(vd.caption.clone());
                 f.tags = vd.tags.clone();
                 f.category = if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) };
            } else {
                // Race: indexing not finished yet. Attempt to recover from DB thumbnail row.
                drop(files); // release lock before async DB call
                if let Some(restored) = self.try_restore_file_metadata(path).await? {
                    log::info!("[AI] Restored metadata stub for {} before applying vision description", path);
                    let mut files2 = self.files.lock().await;
                    files2.push(restored);
                } else {
                    log::warn!("[AI] apply_vision_description: metadata missing and restore failed for {} (creating minimal stub)", path);
                    let ftype = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).map(|ext| {
                        if crate::is_image(&ext) { "image" } else if crate::is_video(&ext) { "video" } else { "other" }
                    }).unwrap_or("other").to_string();
                    let stub = super::FileMetadata {
                        id: None,
                        path: path.to_string(),
                        filename: std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
                        file_type: ftype,
                        size: 0,
                        modified: None,
                        created: None,
                        thumbnail_path: None,
                        thumb_b64: None,
                        hash: None,
                        description: Some(vd.description.clone()),
                        caption: Some(vd.caption.clone()),
                        tags: vd.tags.clone(),
                        category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                        embedding: None,
                        similarity_score: None,
                        clip_embedding: None,
                        clip_similarity_score: None,
                    };
                    let mut files3 = self.files.lock().await;
                    files3.push(stub);
                }
                // After restoration/insertion, update fields (avoid duplication above by separate block)
                let mut files4 = self.files.lock().await;
                if let Some(f) = files4.iter_mut().find(|f| f.path == path) {
                    f.description = Some(vd.description.clone());
                    f.caption = Some(vd.caption.clone());
                    f.tags = vd.tags.clone();
                    f.category = if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) };
                }
            }
        }
        if let Some(meta) = self.get_file_metadata(path).await {
            log::info!("Found file meta");
            self.cache_thumbnail_and_metadata(&meta).await?;
        } else {
            log::error!("No file meta found for {path}");
        }
        Ok(())
    }

    // Attempt to restore FileMetadata from DB cached thumbnail row (best-effort)
    async fn try_restore_file_metadata(&self, path: &str) -> anyhow::Result<Option<super::FileMetadata>> {
        // Single-row lookup via UNIQUE index helper.
        let Some(row) = crate::database::get_thumbnail_by_path(path).await? else { return Ok(None); };
        // Build FileMetadata (leave hash/clip/embedding empty; they will be filled later by indexing)
        // Convert surrealdb::sql::Datetime -> chrono::DateTime<Local>
        fn conv_dt(dt: &Option<surrealdb::sql::Datetime>) -> Option<chrono::DateTime<chrono::Local>> {
            dt.as_ref().and_then(|d| chrono::DateTime::parse_from_rfc3339(&d.to_string()).ok()).map(|dt_fixed| dt_fixed.with_timezone(&chrono::Local))
        }
        let fm = super::FileMetadata {
            id: row.id.as_ref().map(|rid| rid.to_string()),
            path: row.path.clone(),
            filename: std::path::Path::new(&row.path).file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
            file_type: row.file_type.clone(),
            size: row.size,
            modified: conv_dt(&row.modified),
            created: None, // thumbnails row doesn't store a created time compatible with chrono Local directly
            thumbnail_path: None,
            thumb_b64: row.thumbnail_b64.clone(),
            hash: row.hash.clone(),
            description: row.description.clone(),
            caption: row.caption.clone(),
            tags: row.tags.clone(),
            category: row.category.clone(),
            embedding: row.embedding.clone(),
            similarity_score: None,
            clip_embedding: None,
            clip_similarity_score: None,
        };
        Ok(Some(fm))
    }

    pub async fn ensure_vision_model(
        &self,
    ) -> Result<(), anyhow::Error> {
        if crate::ai::joycaption_adapter::is_enabled() {
            if let Err(e) = crate::ai::joycaption_adapter::ensure_loaded().await { log::warn!("[AI] joycaption ensure_loaded failed: {e}"); }
            return Ok(());
        }
        Ok(())
    }

    // Background enrichment: generate descriptions for any previously indexed images that are missing one.
    pub async fn enrich_missing_descriptions(&self) -> usize {
        // Collect snapshot of paths needing enrichment.
        let snapshot: Vec<std::path::PathBuf> = {
            let files = self.files.lock().await;
            files.iter().filter(|f| {
                f.file_type == "image" && (f.description.is_none() || f.description.as_ref().map(|d| d.trim().len() < 12).unwrap_or(true))
            }).map(|f| std::path::PathBuf::from(&f.path)).collect()
        };
        if snapshot.is_empty() { return 0; }
        log::info!("[AI] Scheduling enrichment for {} images", snapshot.len());
        // Ensure model once (async wait) before spawning individual tasks; if this fails we return 0 scheduled.
        if let Err(e) = self.ensure_vision_model().await {
            log::error!("Failed to ensure vision model before scheduling enrichment: {}", e);
            return 0;
        }
        let arc_self = std::sync::Arc::new(self.clone());
        for pb in snapshot.iter() { arc_self.clone().spawn_generate_vision_description(pb.clone()); }
        snapshot.len()
    }

    // Count images lacking a sufficiently descriptive caption.
    pub async fn count_missing_descriptions(&self) -> usize {
        let files = self.files.lock().await;
        files
            .iter()
            .filter(|f| {
                f.file_type == "image"
                    && (f.description.is_none()
                        || f.description
                            .as_ref()
                            .map(|d| d.trim().len() < 12)
                            .unwrap_or(true))
            })
            .count()
    }

    pub fn compute_file_hash(&self, path: &PathBuf) -> anyhow::Result<String, std::io::Error> {
        use std::io::Read;
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            ));
        }
        let mut file = std::fs::File::open(path)?;
        let mut hasher = blake3::Hasher::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
    
    // Return up to 'limit' thumbnail/cache rows directly from Surreal for debug view.
    pub async fn list_thumbnail_rows(&self, limit: usize) -> Vec<crate::Thumbnail> {
        if limit == 0 { return Vec::new(); }
        // Use an ordered, bounded query. If no index for ordering, we still constrain rows via LIMIT.
        let sql = "SELECT * FROM thumbnails LIMIT $limit"; // stable order not guaranteed; only for debug
        match DB.query(sql).bind(("limit", limit as i64)).await {
            Ok(mut resp) => {
                let rows: Result<Vec<crate::Thumbnail>, _> = resp.take(0);
                rows.unwrap_or_else(|e| { log::warn!("Debug list_thumbnail_rows take failed: {e}"); Vec::new() })
            }
            Err(e) => { log::warn!("Debug list_thumbnail_rows query failed: {e}"); Vec::new() }
        }
    }


    /// Update (or insert) a description for a file already tracked in self.files.
    /// Also persists (best-effort) to the cached thumbnail/metadata row if full metadata can be retrieved.
    pub async fn set_file_description(&self, path: &str, desc: &str) -> anyhow::Result<(), anyhow::Error> {
        {
            let mut files = self.files.lock().await;
            if let Some(entry) = files.iter_mut().find(|f| f.path == path) {
                entry.description = Some(desc.to_string());
            } else {
                // If we don't have it yet, create a minimal placeholder so enrichment won't re-trigger.
                files.push(super::FileMetadata {
                    id: None,
                    path: path.to_string(),
                    filename: std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
                    file_type: "other".into(),
                    size: 0,
                    modified: None,
                    created: None,
                    thumbnail_path: None,
                    thumb_b64: None,
                    hash: None,
                    description: Some(desc.to_string()),
                    caption: None,
                    tags: Vec::new(),
                    category: None,
                    embedding: None,
                    similarity_score: None,
                    clip_embedding: None,
                    clip_similarity_score: None,
                });
            }
        }
        // Persist updated row if full metadata available.
        if let Some(updated) = self.get_file_metadata(path).await {
            self.cache_thumbnail_and_metadata(&updated).await?;
        }
        Ok(())
    }
}

impl super::AISearchEngine {
    /// Spawn a background task to generate a vision description for an image path.
    /// On success, updates in-memory metadata & persists (best-effort) without blocking caller.
    pub fn spawn_generate_vision_description(self: std::sync::Arc<Self>, path: std::path::PathBuf) {
        // Only spawn for existing image files.
        if !path.exists() { return; }
        // Fire-and-forget task.
        tokio::spawn(async move {
            let p_str = path.to_string_lossy().to_string();
            // Check cached_paths inside async context.
            if self.cached_paths.lock().await.contains(&p_str) { return; }
            {
                let mut act = self.active_vision_path.lock().await; *act = Some(p_str.clone());
            }
            let start = std::time::Instant::now();
            match self.generate_vision_description(&path).await {
                Some(vd) => {
                    // Update metadata
                    {
                        let mut files = self.files.lock().await;
                        if let Some(f) = files.iter_mut().find(|f| f.path == p_str) {
                            f.description = Some(vd.description.clone());
                            f.caption = Some(vd.caption.clone());
                            f.tags = vd.tags.clone();
                            f.category = if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) };
                        }
                    }
                    if let Some(meta) = self.get_file_metadata(&p_str).await {
                        if let Err(e) = self.cache_thumbnail_and_metadata(&meta).await {
                            log::warn!("[AI] spawn persist failed for {}: {}", p_str, e);
                        }
                    }
                    log::info!("[AI] spawn vision description ok {} in {}ms", p_str, start.elapsed().as_millis());
                }
                None => {
                    log::warn!("[AI] spawn vision description returned None for {} ({}ms)", p_str, start.elapsed().as_millis());
                }
            }
        });
    }

    /// Stream a vision description for the given image path. Invokes `on_update` with (interim_text, final_opt)
    /// where `final_opt` is Some(final VisionDescription) only once when parsed & applied.
    /// Falls back to full generation if streaming not available.
    #[allow(unused_variables)]
    pub async fn stream_vision_description<F>(&self, path: &std::path::Path, prompt_template: &str, mut on_update: F)
        where F: FnMut(&str, Option<&crate::ai::generate::VisionDescription>) + Send + 'static
    {
        if !path.exists() { return; }
        // Ensure model first (best-effort)
        {
            let mut act = self.active_vision_path.lock().await; *act = Some(path.to_string_lossy().to_string());
        }
        if let Err(e) = self.ensure_vision_model().await { log::warn!("[AI] ensure_vision_model failed: {e}"); }
        let prompt = if prompt_template.trim().is_empty() {
            "Analyze the supplied image and return ONLY JSON with keys: description, caption, tags (array), category.".to_string()
        } else { prompt_template.to_string() };

        // joycaption streaming path (feature gated)
        #[cfg(feature="joycaption")]
        {
            if crate::ai::joycaption_adapter::is_enabled() {
                if let Ok(bytes) = tokio::fs::read(path).await {
                    // ---------------- Incremental streaming parse ----------------------
                    #[derive(Default, Debug, Clone)]
                    struct PartialVD { description: Option<String>, caption: Option<String>, tags: Option<Vec<String>>, category: Option<String> }
                    impl PartialVD {
                        fn is_complete(&self) -> bool {
                            self.description.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false)
                                && self.caption.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false)
                                && self.category.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false)
                                && self.tags.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
                        }
                        fn to_final(&self) -> Option<crate::ai::generate::VisionDescription> {
                            if !self.is_complete() { return None; }
                            Some(crate::ai::generate::VisionDescription {
                                description: self.description.clone().unwrap_or_default(),
                                caption: self.caption.clone().unwrap_or_default(),
                                tags: self.tags.clone().unwrap_or_default(),
                                category: self.category.clone().unwrap_or_else(|| "general".into()),
                            })
                        }
                    }

                    fn extract_string_field(buf: &str, key: &str) -> Option<String> {
                        let pattern = format!("\"{}\"", key);
                        let key_pos = buf.find(&pattern)?;
                        let mut i = key_pos + pattern.len();
                        let bytes = buf.as_bytes();
                        // Skip whitespace
                        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i+=1; }
                        if i >= bytes.len() || bytes[i] != b':' { return None; }
                        i+=1; // skip colon
                        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i+=1; }
                        if i >= bytes.len() || bytes[i] != b'"' { return None; }
                        i+=1; // opening quote
                        let start_val = i;
                        let mut escaped = false;
                        while i < bytes.len() {
                            let b = bytes[i];
                            if escaped { escaped = false; i+=1; continue; }
                            match b {
                                b'\\' => { escaped = true; i+=1; },
                                b'"' => {
                                    let slice = &buf[start_val..i];
                                    return Some(slice.to_string());
                                }
                                _ => i+=1
                            }
                        }
                        None // incomplete
                    }

                    fn extract_tags_field(buf: &str) -> Option<Vec<String>> {
                        let pattern = "\"tags\"";
                        let key_pos = buf.find(pattern)?;
                        let mut i = key_pos + pattern.len();
                        let bytes = buf.as_bytes();
                        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i+=1; }
                        if i >= bytes.len() || bytes[i] != b':' { return None; }
                        i+=1;
                        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i+=1; }
                        if i >= bytes.len() || bytes[i] != b'[' { return None; }
                        i+=1; // past '['
                        let mut items: Vec<String> = Vec::new();
                        let mut cur = String::new();
                        let mut in_string = false;
                        let mut escaped = false;
                        while i < bytes.len() {
                            let b = bytes[i];
                            if in_string {
                                if escaped { cur.push(b as char); escaped=false; i+=1; continue; }
                                match b {
                                    b'\\' => { escaped = true; i+=1; },
                                    b'"' => { in_string = false; items.push(cur.trim().to_string()); cur.clear(); i+=1; },
                                    _ => { cur.push(b as char); i+=1; }
                                }
                                continue;
                            }
                            match b {
                                b'"' => { in_string = true; i+=1; },
                                b',' => { i+=1; },
                                b']' => { return Some(items.into_iter().filter(|s| !s.is_empty()).collect()); },
                                b if b.is_ascii_whitespace() => { i+=1; },
                                _ => { // not expected yet (maybe partial)
                                    return None;
                                }
                            }
                        }
                        None
                    }

                    let partial = PartialVD::default();
                    let path_string = path.to_string_lossy().to_string();

                    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
                    let early_stop = std::sync::Arc::new(AtomicBool::new(false));
                    // Wrap mutable state in Arc<Mutex<>> for safe shared mutation inside callback
                    let shared_buffer = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
                    let shared_partial = std::sync::Arc::new(std::sync::Mutex::new(partial));
                    let last_emit = std::sync::Arc::new(std::sync::Mutex::new(0usize));
                    // Wrap callback in Arc<Mutex<>> to allow post-stream final emission
                    let shared_cb: std::sync::Arc<std::sync::Mutex<Box<dyn FnMut(&str, Option<&crate::ai::generate::VisionDescription>) + Send>>> = std::sync::Arc::new(std::sync::Mutex::new(Box::new(on_update)));
                    let cb_early = early_stop.clone();
                    let cb_buffer = shared_buffer.clone();
                    let cb_partial = shared_partial.clone();
                    let cb_last_emit = last_emit.clone();
                    let cb_shared_cb = shared_cb.clone();
                    let full_text = match crate::ai::joycaption_adapter::stream_describe_bytes_with_callback(bytes, &prompt, move |frag| {
                        if cb_early.load(AtomicOrdering::Relaxed) { return; }
                        let mut buf = cb_buffer.lock().unwrap();
                        buf.push_str(frag);
                        if !buf.contains('{') { return; }
                        let mut part = cb_partial.lock().unwrap().clone();
                        let before = part.clone();
                        if part.description.is_none() { if let Some(v) = extract_string_field(&buf, "description") { part.description = Some(v); } }
                        if part.caption.is_none() { if let Some(v) = extract_string_field(&buf, "caption") { part.caption = Some(v); } }
                        if part.category.is_none() { if let Some(v) = extract_string_field(&buf, "category") { part.category = Some(v); } }
                        if part.tags.is_none() { if let Some(v) = extract_tags_field(&buf) { part.tags = Some(v); } }
                        if part.is_complete() { cb_early.store(true, AtomicOrdering::Relaxed); }
                        let changed = before.description != part.description
                            || before.caption != part.caption
                            || before.category != part.category
                            || before.tags.as_ref().map(|v| v.len()).unwrap_or(0) != part.tags.as_ref().map(|v| v.len()).unwrap_or(0);
                        let mut le = cb_last_emit.lock().unwrap();
                        let newly = buf.len().saturating_sub(*le);
                        if changed || newly >= 64 {
                            *le = buf.len();
                            let interim_text = if let Some(d) = &part.description { d.clone() }
                                else if let Some(c) = &part.caption { c.clone() }
                                else { buf.clone() };
                            if let Ok(mut cb_guard) = cb_shared_cb.lock() { (cb_guard)(&interim_text, None); }
                        }
                        // write back updated partial
                        *cb_partial.lock().unwrap() = part;
                    }).await {
                        Ok(t) => t,
                        Err(e) => { 
                            log::warn!("[AI] streaming error: {e}"); 
                            shared_buffer.lock().unwrap().clone()
                        }
                    };
                    // Recover owned buffer & partial
                    let buffer = std::sync::Arc::try_unwrap(shared_buffer).unwrap_or_default().into_inner().unwrap();
                    let mut partial = std::sync::Arc::try_unwrap(shared_partial).unwrap_or_default().into_inner().unwrap();

                    // Attempt fenced code block JSON extraction first (```json ... ```)
                    fn extract_fenced_json(s: &str) -> Option<&str> {
                        let fence_start = s.find("```json")?;
                        // find newline after fence line
                        let after = &s[fence_start+6..];
                        let nl_pos = after.find('\n')?;
                        let body_start = fence_start + 6 + nl_pos + 1;
                        let fence_end = s[body_start..].find("```")? + body_start;
                        Some(&s[body_start..fence_end])
                    }
                    if let Some(fenced) = extract_fenced_json(&full_text) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(fenced) {
                            if let Ok(vd) = serde_json::from_value::<crate::ai::generate::VisionDescription>(val.clone()) {
                                partial.description.get_or_insert(vd.description.clone());
                                partial.caption.get_or_insert(vd.caption.clone());
                                if partial.tags.is_none() && !vd.tags.is_empty() { partial.tags = Some(vd.tags.clone()); }
                                partial.category.get_or_insert(vd.category.clone());
                            }
                        }
                    } else if let Some(start) = full_text.find('{') { if let Some(end) = full_text.rfind('}') { if end > start {
                        let slice = &full_text[start..=end];
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(slice) {
                            if let Ok(vd) = serde_json::from_value::<crate::ai::generate::VisionDescription>(val.clone()) {
                                partial.description.get_or_insert(vd.description.clone());
                                partial.caption.get_or_insert(vd.caption.clone());
                                if partial.tags.is_none() && !vd.tags.is_empty() { partial.tags = Some(vd.tags.clone()); }
                                partial.category.get_or_insert(vd.category.clone());
                            }
                        }
                    } } }

                    if let Some(final_vd) = partial.to_final() {
                        let _ = self.apply_vision_description(&path_string, &final_vd).await;
                        if let Ok(mut cb_guard) = shared_cb.lock() { (cb_guard)(&final_vd.description, Some(&final_vd)); }
                    } else {
                        let fallback_text = partial.description.clone().or(partial.caption.clone()).unwrap_or_else(|| full_text.clone());
                        if let Ok(mut cb_guard) = shared_cb.lock() { (cb_guard)(&fallback_text, None); }
                    }
                    // Prevent use-after-move by forgetting original on_update (consumed into shared_cb)
                    // (No action needed; moved)
                } else { log::warn!("[AI] failed to read file for streaming: {}", path.display()); }
                return; // streaming handled
            }
        }
        // Fallback: non-streaming generation
    if let Some(vd) = self.generate_vision_description(&path.to_path_buf()).await {
            let _ = self.apply_vision_description(&path.to_string_lossy(), &vd).await;
            on_update(&vd.description, Some(&vd));
        }
    }

    pub async fn get_active_vision_path(&self) -> Option<String> { self.active_vision_path.lock().await.clone() }
}

// Helper function to extract metadata from FoundFile
pub fn found_file_to_metadata(found_file: &crate::utilities::types::FoundFile) -> super::FileMetadata {
    super::FileMetadata {
        id: None,
        path: found_file.path.display().to_string(),
        filename: found_file
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string(),
        file_type: match found_file.kind {
            crate::utilities::types::MediaKind::Image => "image".to_string(),
            crate::utilities::types::MediaKind::Video => "video".to_string(),
            crate::utilities::types::MediaKind::Other => "other".to_string(),
        },
        size: found_file.size.unwrap_or(0),
        modified: found_file.modified,
        created: found_file.created,
        // Only set thumb_b64; do not misuse thumbnail_path for base64 data URLs.
        thumbnail_path: None,
        thumb_b64: found_file.thumb_data.clone(),
        hash: None,
        description: None,
    caption: None,
        tags: Vec::new(),
    category: None,
        embedding: None,    
        similarity_score: None,
        clip_embedding: None,
        clip_similarity_score: None,
    }
}
