use std::time::{Duration, Instant};
use crate::ui::file_table::rows::Upsert;

const SCAN_MSG_BUDGET: usize = 512;
const SCAN_TIME_BUDGET: Duration = Duration::from_millis(6);

impl crate::ui::file_table::FileExplorer {
    pub fn receive_scan(&mut self, ctx: &eframe::egui::Context) {
        // Drains scan messages in order until the per-frame message or time budget is spent.
        let started = Instant::now();
        let mut handled = 0usize;
        loop {
            if handled >= SCAN_MSG_BUDGET || started.elapsed() >= SCAN_TIME_BUDGET {
                if !self.scan_rx.is_empty() { ctx.request_repaint(); }
                break;
            }
            let Ok(env) = self.scan_rx.try_recv() else { break };
            handled += 1;
            // Only process messages for scans owned by this explorer. If we don't yet
            // have an owning_scan_id, we accept only when this scan is the actively
            // initiated one (current_scan_id). This prevents cross-tab leakage.
            let mut accept = false;
            if let Some(owner) = self.owning_scan_id {
                accept = env.scan_id == owner;
            } else if let Some(current) = self.current_scan_id {
                accept = env.scan_id == current;
                if accept {
                    // Bind ownership to this explorer for the lifetime of this scan
                    self.owning_scan_id = Some(current);
                }
            }
            if !accept {
                // Ignore stray messages for other tabs/scans, but continue draining the queue
                continue;
            }
            match env.msg {
                crate::utilities::scan::ScanMsg::FoundDir(dir) => {
                    if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&dir.path) {
                        let dp = row.path.clone();
                        if !self.last_scan_paths.contains(&dp) {
                            self.last_scan_rows.push(row.clone());
                            self.last_scan_paths.insert(dp.clone());
                            // Only add to visible table if within current page slice (when paginating)
                            if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                                self.update_recursive_total_pages();
                                let start = self.recursive_current_page * self.recursive_page_size;
                                let end = (start + self.recursive_page_size).min(self.last_scan_rows.len());
                                if self.last_scan_rows.len()-1 >= start && self.last_scan_rows.len()-1 < end {
                                    self.table.push(row);
                                }
                            } else {
                                self.table.push(row);
                            }
                            ctx.request_repaint();
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::FoundDirBatch(dirs) => {
                    if self.perf_scan_started.is_none() { self.perf_scan_started = Some(std::time::Instant::now()); self.perf_last_batch_at = self.perf_scan_started; }
                    {
                        for d in dirs.iter() {
                            if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&d.path) {
                                if !self.last_scan_paths.contains(&row.path) {
                                    self.last_scan_rows.push(row.clone());
                                    self.last_scan_paths.insert(row.path.clone());
                                    if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                                        self.update_recursive_total_pages();
                                        let start = self.recursive_current_page * self.recursive_page_size;
                                        let end = (start + self.recursive_page_size).min(self.last_scan_rows.len());
                                        if self.last_scan_rows.len()-1 >= start && self.last_scan_rows.len()-1 < end {
                                            self.table.push(row);
                                        }
                                    } else {
                                        self.table.push(row);
                                    }
                                }
                            }
                        }
                        ctx.request_repaint();
                    }
                }
                crate::utilities::scan::ScanMsg::Found(item) => {
                    // Every found file is kept; the tab's filter decides what is listed
                    if let Some(row0) = crate::file_to_thumbnail(&item) {
                        // Merge with DB if exists
                        let mut row = if let Some(db_row) = self.db_lookup.get(&row0.path).cloned() { db_row } else { row0 };
                        crate::ui::file_table::thumbs::adopt_legacy_one(&self.viewer.thumbs, &mut row);
                        // If archive, tag type for UI (file_to_thumbnail currently classifies only image/video/other)
                        if let Some(ext) = std::path::Path::new(&row.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                        {
                            if crate::utilities::types::is_archive(&ext) {
                                row.file_type = "<ARCHIVE>".into();
                            }
                        }
                        // Always snapshot first
                        let is_new = !self.last_scan_paths.contains(&row.path);
                        if is_new {
                            self.last_scan_rows.push(row.clone());
                            self.last_scan_paths.insert(row.path.clone());
                        }
                        if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                            self.update_recursive_total_pages();
                            // Only add to visible table if the new row falls on current page
                            let start = self.recursive_current_page * self.recursive_page_size;
                            let end = (start + self.recursive_page_size).min(self.last_scan_rows.len());
                            if self.last_scan_rows.len() - 1 >= start && self.last_scan_rows.len() - 1 < end {
                                self.table.push(row.clone());
                                ctx.request_repaint();
                            }
                        } else {
                            // Non paginated path or below threshold
                            self.table.upsert(row.clone());
                            ctx.request_repaint();
                        }
                        // Append to cached scan items buffer
                        if self.recursive_scan {
                            let path_str = row.path.clone();
                            let sid = self.owning_scan_id.unwrap_or(0);
                            tokio::spawn(async move {
                                if let Ok(Some(sc)) = crate::database::CachedScan::get_by_scan_id(sid).await {
                                    let _ = crate::database::CachedScanItem::add_many(&sc.id, vec![path_str]).await;
                                }
                            });
                        }
                        // Enqueue for AI indexing (images/videos) only if auto_save is enabled and a logical group is active.
                        let path_clone = item.path.to_string_lossy().to_string();
                        let parent_dir = item.path.parent().map(|p| p.to_string_lossy().to_string().clone()).unwrap_or_default();
                        // Use actual extension (lowercase) for file_type
                        let ftype = item
                            .path
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                            .unwrap_or_else(|| String::new());
                            let db_known = self.db_lookup.contains_key(&path_clone);
                            // Build metadata from DB row if present to preserve id and prior fields
                            let meta = if let Some(db_row) = self.db_lookup.get(&path_clone) {
                            let mut m = db_row.clone();
                            // Prefer fresh size/thumbnail if provided by scan
                            if let Some(sz) = item.size { m.size = sz; }
                            if item.thumb_data.is_some() { m.thumbnail_b64 = item.thumb_data.clone(); }
                            m
                        } else {
                            crate::database::Thumbnail {
                                id: crate::Thumbnail::new(
                                    &item
                                        .path
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("")
                                        .to_string(),
                                ).id,
                                db_created: None,
                                path: path_clone.clone(),
                                filename: item
                                    .path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("")
                                    .to_string(),
                                file_type: ftype,
                                size: item.size.unwrap_or(0),
                                modified: None,
                                thumbnail_b64: item.thumb_data.clone(),
                                hash: None,
                                description: None,
                                caption: None,
                                tags: Vec::new(),
                                category: None,
                                parent_dir: parent_dir,
                                logical_group: crate::LogicalGroup::default().id,
                            }
                        };

                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        // Only auto-save immediately when the row is already stored or arrived with a thumbnail (so failed thumbs don't get stored).
                        if auto_save && (db_known || meta.thumbnail_b64.is_some()) {
                            let do_index = self.viewer.ui_settings.auto_indexing;
                            self.queue_persist(vec![meta.clone()]);
                            if do_index {
                                tokio::spawn(async move {
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await;
                                });
                            }
                        }
                        // If auto CLIP is enabled but auto indexing is off, schedule CLIP embedding for images here
                        if self.viewer.ui_settings.auto_clip_embeddings && !self.viewer.ui_settings.auto_indexing {
                            if let Some(ext) = std::path::Path::new(&row.path)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                            {
                                if crate::is_image(ext.as_str()) {
                                    let path = row.path.clone();
                                    tokio::spawn(async move {
                                        // Ensure engine ready and generate embedding for this single path
                                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                        let _ = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path]).await;
                                        Ok::<(), anyhow::Error>(())
                                    });
                                }
                            }
                        }
                        // Incremental filtering used: no whole-table re-filter per item
                    }
                }
                crate::utilities::scan::ScanMsg::FoundBatch(batch) => {
                    // Batch timing measurement: gap since last batch arrival (recv) and UI processing time (ui)
                    let now = std::time::Instant::now();
                    if self.perf_scan_started.is_none() { self.perf_scan_started = Some(now); self.perf_last_batch_at = Some(now); }
                    let recv_gap = self.perf_last_batch_at.map(|prev| now.duration_since(prev)).unwrap_or_default();
                    self.perf_last_batch_at = Some(now);

                    let ui_start = std::time::Instant::now();
                    let mut to_index: Vec<crate::database::Thumbnail> = Vec::new();
                    let mut need_repaint = false;
                    let mut cached_paths: Vec<String> = Vec::new();

                    for item in batch.into_iter() {
                        // Every found file is kept; the tab's filter decides what is listed
                        if let Some(row0) = crate::file_to_thumbnail(&item) {
                            // Merge with DB if exists
                            let mut row = if let Some(db_row) = self.db_lookup.get(&row0.path).cloned() { db_row } else { row0 };
                            crate::ui::file_table::thumbs::adopt_legacy_one(&self.viewer.thumbs, &mut row);
                            if let Some(ext) = std::path::Path::new(&row.path)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                            {
                                if crate::utilities::types::is_archive(&ext) {
                                    row.file_type = "<ARCHIVE>".into();
                                }
                            }
                            // Snapshot first
                            let row_snapshot = row.clone();
                            let is_new_global = !self.last_scan_paths.contains(&row_snapshot.path);
                            if is_new_global {
                                self.last_scan_rows.push(row_snapshot.clone());
                                self.last_scan_paths.insert(row_snapshot.path.clone());
                            }
                            // Decide if row is visible on current page (if paginating)
                            let mut show_in_page = true;
                            if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                                self.update_recursive_total_pages();
                                let start = self.recursive_current_page * self.recursive_page_size;
                                let end = (start + self.recursive_page_size).min(self.last_scan_rows.len());
                                show_in_page = is_new_global && (self.last_scan_rows.len() - 1) >= start && (self.last_scan_rows.len() - 1) < end;
                            }
                            if show_in_page {
                                if let Upsert::Inserted(_) = self.table.upsert(row_snapshot.clone()) {
                                    if self.recursive_scan { cached_paths.push(row_snapshot.path.clone()); }
                                }
                                need_repaint = true;
                            }
                            // Prepare indexing metadata
                            // Use actual extension (lowercase) for file_type
                            let ftype = item
                                .path
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                                .unwrap_or_else(|| String::new());
                            let p_str = item.path.to_string_lossy().to_string();
                            let parent_dir = item.path.parent().map(|p| p.to_string_lossy().to_string().clone()).unwrap_or_default();
                            // If DB row exists reuse it else create new
                            if let Some(db_row) = self.db_lookup.get(&p_str) {
                                let mut m = db_row.clone();
                                if let Some(sz) = item.size { m.size = sz; }
                                if item.thumb_data.is_some() { m.thumbnail_b64 = item.thumb_data.clone(); }
                                // Rows already in the database were stored with a thumbnail
                                to_index.push(m);
                            } else {
                                let row_for_db = crate::database::Thumbnail {
                                    id: crate::Thumbnail::new(
                                        &item
                                            .path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("")
                                            .to_string(),
                                    ).id,
                                    db_created: None,
                                    path: p_str,
                                    filename: item
                                        .path
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    file_type: ftype,
                                    size: item.size.unwrap_or(0),
                                    modified: None,
                                    thumbnail_b64: item.thumb_data.clone(),
                                    hash: None,
                                    description: None,
                                    caption: None,
                                    tags: Vec::new(),
                                    category: None,
                                    parent_dir,
                                    logical_group: crate::LogicalGroup::default().id,
                                };
                                if row_for_db.thumbnail_b64.is_some() { to_index.push(row_for_db); }
                            }
                            // If auto CLIP is enabled but auto indexing is off, schedule CLIP embedding for images here
                            if self.viewer.ui_settings.auto_clip_embeddings && !self.viewer.ui_settings.auto_indexing {
                                if let Some(ext) = item
                                    .path
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .map(|s| s.to_ascii_lowercase())
                                {
                                    if crate::is_image(ext.as_str()) {
                                        let path = item.path.to_string_lossy().to_string();
                                        tokio::spawn(async move {
                                            let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                            let _ = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path]).await;
                                            Ok::<(), anyhow::Error>(())
                                        });
                                    }
                                }
                            }
                        }
                        self.flush_pending_rows(false);
                    }
                    if self.recursive_scan && !cached_paths.is_empty() {
                        let sid = self.owning_scan_id.unwrap_or(0);
                        tokio::spawn(async move {
                            if let Ok(Some(sc)) = crate::database::CachedScan::get_by_scan_id(sid).await {
                                let _ = crate::database::CachedScanItem::add_many(&sc.id, cached_paths).await;
                            }
                        });
                    }
                    // Incremental filtering used: no whole-table re-filter per batch
                    let ui_time = ui_start.elapsed();
                    // Record performance sample
                    self.perf_batches.push((self.table.len(), recv_gap, ui_time));
                    if self.perf_batches.len() > 2000 { self.perf_batches.drain(0..self.perf_batches.len()-2000); }
                    if need_repaint { ctx.request_repaint(); }
                    if !to_index.is_empty() {
                        let do_index = self.viewer.ui_settings.auto_indexing;
                        if self.viewer.ui_settings.auto_save_to_database {
                            self.queue_persist(to_index.clone());
                        }
                        if do_index {
                            tokio::spawn(async move {
                                for meta in to_index.into_iter() {
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await;
                                }
                            });
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::ThumbReady { path, .. } => {
                    let key = path.to_string_lossy().to_string();
                    if let Some(found) = self.table.get_by_path(&key) {
                        self.pending_thumb_rows.push(found.clone());
                    } else {
                        // Path not yet present (likely due to message reordering).
                        // Do NOT inject into table during recursive scans to avoid page bleed; let Found/Batch add rows.
                        if self.recursive_scan {
                            log::debug!("ThumbReady for unknown path during recursive scan; deferring: {}", key);
                        } else {
                            // For non-recursive views, create a placeholder row for immediate display.
                            let ext_lc = std::path::Path::new(&key)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                                .unwrap_or_default();
                            let is_img = crate::is_image(ext_lc.as_str());
                            let is_vid = crate::is_video(ext_lc.as_str());
                            if is_img || is_vid {
                                use chrono::Utc;
                                let filename = std::path::Path::new(&key)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("")
                                    .to_string();
                                let parent_dir = std::path::Path::new(&key)
                                    .parent()
                                    .map(|p| p.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                let row = crate::Thumbnail {
                                    id: crate::Thumbnail::new(&filename).id,
                                    db_created: Some(Utc::now().into()),
                                    path: key.clone(),
                                    filename,
                                    file_type: ext_lc.clone(),
                                    size: 0,
                                    description: None,
                                    caption: None,
                                    tags: Vec::new(),
                                    category: None,
                                    thumbnail_b64: None,
                                    modified: Some(Utc::now().into()),
                                    hash: None,
                                    parent_dir,
                                    logical_group: crate::LogicalGroup::default().id,
                                };
                                self.table.push(row.clone());
                                self.pending_thumb_rows.push(row);
                                log::debug!("Inserted placeholder row for missing ThumbReady: {}", key);
                            }
                        }
                    }
                    super::super::scope::failed::clear_path(&key);
                    self.flush_pending_rows(false);
                }
                crate::utilities::scan::ScanMsg::ThumbFailed { path, error } => {
                    let key = path.to_string_lossy().to_string();
                    super::super::scope::failed::record(&key, &error);
                    self.failed_thumbs.push((key.clone(), error.clone()));
                    log::warn!("Thumbnail failed: {} => {}", key, error);
                }
                crate::utilities::scan::ScanMsg::EncryptedArchives(zips) => {
                    // Queue unique archives; show modal after scan completes
                    for z in zips.into_iter() {
                        if !self.pending_zip_passwords.contains(&z) {
                            self.pending_zip_passwords.push_back(z);
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::Progress { scanned, total } => {
                    if scanned > 0 && total > 0 {
                        self.file_scan_progress = (scanned as f32) / (total as f32);
                    }
                }
                crate::utilities::scan::ScanMsg::Error(e) => {
                    log::error!("ScanMsg Error: {e:?}");
                }
                crate::utilities::scan::ScanMsg::Done => {
                    // Rebuild page view at end of scan to ensure correct slice size
                    if self.recursive_scan { self.rebuild_recursive_page(); }

                    log::info!("Done");
                    self.file_scan_progress = 1.0;
                    self.scan_done = true;
                    self.current_scan_id = None;
                    if self.recursive_scan && self.last_scan_rows.len() > self.recursive_page_size {
                        self.update_recursive_total_pages();
                        // Rebuild current page to ensure correct slice (in case of overflow on last batch)
                        self.rebuild_recursive_page();
                    }
                    // Record total elapsed for this scan
                    if let Some(start) = self.perf_scan_started.take() {
                        let total = start.elapsed();
                        self.perf_last_total = Some(total);
                        self.perf_last_batch_at = None;
                    }
                    
                    // If we have pending encrypted archives, open the modal now
                    if !self.pending_zip_passwords.is_empty() {
                        self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                        self.show_zip_modal = true;
                    }

                    // Mark cached scan finished with total count if available
                    if self.recursive_scan {
                        let total = self.table.len() as u64;
                        let sid = self.owning_scan_id.unwrap_or(0);
                        tokio::spawn(async move {
                            if let Ok(Some(sc)) = crate::database::CachedScan::get_by_scan_id(sid).await { let _ = sc.mark_finished(total).await; }
                        });
                    }
                    self.flush_pending_rows(true);
                    if super::super::dialogs::scan::take_library_save(self.last_scan_root.as_deref().unwrap_or_default()) {
                        self.save_scanned_to_library();
                    }
                    self.note_listing_done();
                    // Do not bulk-refresh CLIP presence here; per-thumbnail checks will update presence incrementally.
                }
            }
            ctx.request_repaint();
        }

    }
}