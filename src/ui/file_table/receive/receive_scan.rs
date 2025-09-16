use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use std::sync::Arc;
use crate::utilities::filtering::FiltersExt;

impl crate::ui::file_table::FileExplorer {
    pub fn receive_scan(&mut self, ctx: &eframe::egui::Context) {
        // Drain all pending scan messages this frame to preserve ordering (so Found/FoundBatch
        // are processed before their subsequent UpdateThumb events). This reduces the chance
        // of UpdateThumb arriving first and logging Not found repeatedly.
        while let Ok(env) = self.scan_rx.try_recv() {
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
                // Ignore stray messages for other tabs/scans
                return;
            }
            // Create or reuse a cached scan record when a recursive scan begins receiving
            if self.recursive_scan && self.owning_scan_id.is_some() && self.last_scan_root.is_none() {
                let root_str = self.current_path.clone();
                let title = format!("Scan: {}", root_str);
                let root_for_task = root_str.clone();
                let sid = self.owning_scan_id.unwrap_or(0);
                let tx = self.scan_tx.clone();
                tokio::spawn(async move {
                    if let Ok(_scan) = crate::database::CachedScan::create(&root_for_task, Some(title), sid).await {
                        // Notify UI thread with a no-op progress to store scan id (optional)
                        let _ = tx.try_send(crate::ScanEnvelope { scan_id: sid, msg: crate::utilities::scan::ScanMsg::Progress { scanned: 0, total: 0 } });
                        // Store scan.id on UI via subsequent messages referencing owning_scan_id; we'll fetch lazily when appending
                    }
                });
                self.last_scan_root = Some(root_str);
            }
            match env.msg {
                crate::utilities::scan::ScanMsg::FoundDir(dir) => {
                    if !self.viewer.types_show_dirs { ctx.request_repaint(); return; }
                    if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&dir.path) {
                        if let Some(&idx) = self.table_index.get(&row.path) {
                            // Update in-place to preserve selection state and caching
                            if let Some(existing) = self.table.get_mut(idx) {
                                let keep_thumb = existing.thumbnail_b64.clone();
                                *existing = row;
                                if existing.thumbnail_b64.is_none() { existing.thumbnail_b64 = keep_thumb; }
                            }
                        } else {
                            let idx = self.table.len();
                            self.table.push(row.clone());
                            self.table_index.insert(row.path.clone(), idx);
                        }
                        // Track in last scan snapshot
                        let dp = dir.path.to_string_lossy().to_string();
                        if !self.last_scan_paths.contains(&dp) {
                            if let Some(&i) = self.table_index.get(&dp) { if let Some(r) = self.table.get(i) {
                                self.last_scan_rows.push(r.clone());
                                self.last_scan_paths.insert(r.path.clone());
                            }}
                        }
                        ctx.request_repaint();
                    }
                }
                crate::utilities::scan::ScanMsg::FoundDirBatch(dirs) => {
                    // Start timing on first batch arrival if not already started
                    if self.perf_scan_started.is_none() { self.perf_scan_started = Some(std::time::Instant::now()); self.perf_last_batch_at = self.perf_scan_started; }
                    if self.viewer.types_show_dirs {
                        for d in dirs.iter() {
                            if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&d.path) {
                                let idx = self.table.len();
                                self.table.push(row.clone());
                                self.table_index.insert(row.path.clone(), idx);
                            }
                        }
                    }
                    ctx.request_repaint();
                }
                crate::utilities::scan::ScanMsg::Found(item) => {
                    // Found single file (incremental processing)
                    // (extension filtering handled during scan)
                    // Build UI filter parameters (no extra metadata reads on UI thread)
                    let mut filters = crate::Filters::default();
                    filters.include_images = self.viewer.types_show_images;
                    filters.include_videos = self.viewer.types_show_videos;
                    filters.include_archives = false; // archives not shown in file table view
                    filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                    filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                    filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                    filters.modified_after = self.viewer.ui_settings.filter_modified_after.clone();
                    filters.modified_before = self.viewer.ui_settings.filter_modified_before.clone();

                    let size_val = item.size.unwrap_or(0);
                    if let Some(row0) = if filters.allow_file_attrs(
                        &item.path, 
                        size_val, 
                        item.modified, 
                        item.created, 
                        false
                    ) { 
                        crate::file_to_thumbnail(&item) 
                    } else { 
                        None 
                    } {
                        // Merge with DB if exists
                        let mut row = if let Some(db_row) = self.db_lookup.get(&row0.path).cloned() { db_row } else { row0 };
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
                        // Insert or update via index
                        if let Some(&idx) = self.table_index.get(&row.path) {
                            if let Some(existing) = self.table.get_mut(idx) { *existing = row.clone(); }
                        } else {
                            let idx = self.table.len();
                            self.table.push(row.clone());
                            self.table_index.insert(row.path.clone(), idx);
                        }
                        // Snapshot rows to allow restore later
                        if !self.last_scan_paths.contains(&row.path) {
                            self.last_scan_rows.push(row.clone());
                            self.last_scan_paths.insert(row.path.clone());
                        }
                        ctx.request_repaint();
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
                        if auto_save {
                            let group_name = self.active_logical_group_name.clone();
                            let meta_clone = meta.clone();
                            let do_index = self.viewer.ui_settings.auto_indexing;
                            tokio::spawn(async move {
                                // Upsert row always when auto-save is enabled
                                match crate::database::upsert_rows_and_get_ids(vec![meta_clone.clone()]).await {
                                    Ok(ids) => {
                                        // Optionally add to group if one is active
                                        if let Some(group_name) = group_name {
                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                            }
                                        }
                                    }
                                    Err(e) => log::error!("Error upserting rows and getting ID's: {e:?}")
                                }
                                // Index only if enabled
                                if do_index {
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta_clone).await;
                                }
                            });
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
                    let mut newly_enqueued: Vec<String> = Vec::new();
                    let mut to_index: Vec<crate::database::Thumbnail> = Vec::new();
                    let mut need_repaint = false;
                    let excluded = if self.excluded_terms.is_empty() {
                        None
                    } else {
                        Some(
                            self.excluded_terms
                                .iter()
                                .map(|s| s.to_ascii_lowercase())
                                .collect::<Vec<_>>(),
                        )
                    };
                    let mut cached_paths: Vec<String> = Vec::new();
                    // Build UI filter parameters once for this batch (no extra metadata reads)
                    let mut filters = crate::Filters::default();
                    filters.include_images = self.viewer.types_show_images;
                    filters.include_videos = self.viewer.types_show_videos;
                    filters.include_archives = false; // archives not shown in file table view
                    filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
                    filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
                    filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
                    filters.modified_after = self.viewer.ui_settings.filter_modified_after.clone();
                    filters.modified_before = self.viewer.ui_settings.filter_modified_before.clone();

                    for item in batch.into_iter() {
                        // (extension filtering handled during scan)
                        if let Some(ref ex) = excluded {
                            let lp = item.path.to_string_lossy().to_ascii_lowercase();
                            if ex.iter().any(|t| lp.contains(t)) {
                                continue;
                            }
                        }
                        // Incremental filter check
                        let size_val = item.size.unwrap_or(0);
                        if let Some(row0) = if filters.allow_file_attrs(
                            &item.path, 
                            size_val, 
                            item.modified, 
                            item.created, 
                            false
                        ) { 
                            crate::file_to_thumbnail(&item) 
                        } else { 
                            None 
                        } {
                            // Merge with DB if exists
                            let mut row = if let Some(db_row) = self.db_lookup.get(&row0.path).cloned() { db_row } else { row0 };
                            if let Some(ext) = std::path::Path::new(&row.path)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                            {
                                if crate::utilities::types::is_archive(&ext) {
                                    row.file_type = "<ARCHIVE>".into();
                                }
                            }
                            // Clone for snapshot before potentially moving `row`
                            let row_snapshot = row.clone();
                            if let Some(&idx) = self.table_index.get(&row_snapshot.path) {
                                if let Some(existing) = self.table.get_mut(idx) {
                                    let keep_thumb = existing.thumbnail_b64.clone();
                                    *existing = row;
                                    if existing.thumbnail_b64.is_none() { existing.thumbnail_b64 = keep_thumb; }
                                }
                            } else {
                                let idx = self.table.len();
                                self.table.push(row_snapshot.clone());
                                self.table_index.insert(row_snapshot.path.clone(), idx);
                                if self.recursive_scan { cached_paths.push(row_snapshot.path.clone()); }
                            }
                            // snapshot
                            if !self.last_scan_paths.contains(&row_snapshot.path) {
                                self.last_scan_rows.push(row_snapshot.clone());
                                self.last_scan_paths.insert(row_snapshot.path.clone());
                            }
                            need_repaint = true;
                            // Schedule thumbnail generation if none yet
                            if item.thumb_data.is_none() {
                                let p_str = item.path.to_string_lossy().to_string();
                                if self.thumb_scheduled.insert(p_str.clone()) {
                                    newly_enqueued.push(p_str);
                                }
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
                                to_index.push(m);
                            } else {
                                to_index.push(crate::database::Thumbnail {
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
                                });
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
                        if self.pending_thumb_rows.len() >= 32 {
                            let batch = std::mem::take(&mut self.pending_thumb_rows);
                            let group_name = self.active_logical_group_name.clone();
                            if self.viewer.ui_settings.auto_save_to_database {
                                tokio::spawn(async move {
                                    let mut ids: Vec<surrealdb::RecordId> = Vec::new();
                                    for meta in batch.into_iter() {
                                        match crate::database::upsert_row_and_get_id(meta).await {
                                            Ok(Some(id)) => ids.push(id),
                                            Ok(None) => {},
                                            Err(e) => log::warn!("per-item upsert failed: {e}"),
                                        }
                                    }
                                    if !ids.is_empty() {
                                        if let Some(gname) = group_name {
                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&gname).await {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                            }
                                        }
                                    }
                                });
                            } else {
                                log::debug!("Skipping batch DB save: auto_save_to_database disabled");
                            }
                        }
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
                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        let group_name = self.active_logical_group_name.clone();
                        tokio::spawn(async move {
                            let mut ids: Vec<surrealdb::RecordId> = Vec::new();
                            if auto_save {
                                for meta in to_index.iter().cloned() {
                                    match crate::database::upsert_row_and_get_id(meta.clone()).await {
                                        Ok(Some(id)) => ids.push(id),
                                        Ok(None) => {},
                                        Err(e) => log::warn!("per-item upsert failed: {e}"),
                                    }
                                }
                                if !ids.is_empty() {
                                    if let Some(group_name) = group_name {
                                        if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                                            let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                        }
                                    }
                                }
                            }
                            if do_index {
                                for meta in to_index.into_iter() {
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await;
                                }
                            }
                        });
                    }
                    if !newly_enqueued.is_empty() {
                        let scan_tx = self.scan_tx.clone();
                        let sem = self.thumb_semaphore.clone();
                        // Route these updates to this explorer's owning scan id
                        let scan_id_for_updates = self.owning_scan_id.or(self.current_scan_id).unwrap_or(0);
                        for path_str in newly_enqueued.into_iter() {
                            let permit_fut = sem.clone().acquire_owned();
                            let tx_clone = scan_tx.clone();
                            tokio::spawn(async move {
                                let _permit = permit_fut.await.expect("semaphore closed");
                                let path_buf = std::path::PathBuf::from(&path_str);
                                let ext = path_buf
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .map(|s| s.to_ascii_lowercase());

                                let mut thumb: Option<String> = None;
                                if let Some(ext) = ext {
                                    let is_img = crate::is_image(ext.as_str());
                                    let is_vid = crate::is_video(ext.as_str());
                                    if is_img {
                                        thumb =crate::utilities::thumbs::generate_image_thumb_data(&path_buf).ok();
                                    } else if is_vid {
                                        thumb = crate::utilities::thumbs::generate_video_thumb_data(&path_buf).ok();
                                    }
                                }
                                if let Some(t) = thumb {
                                    let _ = tx_clone.try_send(crate::ScanEnvelope {
                                        scan_id: scan_id_for_updates,
                                        msg: crate::utilities::scan::ScanMsg::UpdateThumb {
                                            path: path_buf,
                                            thumb: t,
                                        },
                                    });
                                }
                            });
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::UpdateThumb { path, thumb } => {
                    let key = path.to_string_lossy().to_string();
                    if let Some(&idx) = self.table_index.get(&key) {
                        if let Some(found) = self.table.get_mut(idx) {
                            found.thumbnail_b64 = Some(thumb.clone());
                            self.pending_thumb_rows.push(found.clone());
                        }
                    } else {
                        // Path not yet present (likely due to message reordering). Create a
                        // placeholder row so the thumbnail appears immediately instead of
                        // spamming errors. We only insert if filters would normally allow it.
                        let ext_lc = std::path::Path::new(&key)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_ascii_lowercase())
                            .unwrap_or_default();
                        let is_img = crate::is_image(ext_lc.as_str());
                        let is_vid = crate::is_video(ext_lc.as_str());
                        if (is_img && self.viewer.types_show_images) || (is_vid && self.viewer.types_show_videos) {
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
                                thumbnail_b64: Some(thumb.clone()),
                                modified: Some(Utc::now().into()),
                                hash: None,
                                parent_dir,
                                logical_group: crate::LogicalGroup::default().id,
                            };
                            let idx = self.table.len();
                            self.table.push(row.clone());
                            self.table_index.insert(row.path.clone(), idx);
                            self.pending_thumb_rows.push(row);
                            log::debug!("Inserted placeholder row for missing UpdateThumb: {}", key);
                        } else {
                            log::debug!("Skipping placeholder for UpdateThumb (filtered out): {}", key);
                        }
                    }
                    if let Ok(decoded) = B64.decode(thumb.as_bytes()) {
                        let key = path.to_string_lossy().to_string();
                        self.viewer
                            .thumb_cache
                            .entry(key)
                            .or_insert_with(|| Arc::from(decoded.into_boxed_slice()));
                    }
                    if self.pending_thumb_rows.len() >= 32 {
                        let batch = std::mem::take(&mut self.pending_thumb_rows);
                        let group_name = self.active_logical_group_name.clone();
                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        tokio::spawn(async move {
                            if auto_save {
                                match crate::database::upsert_rows_and_get_ids(batch).await {
                                    Ok(ids) => {
                                        if let Some(gname) = group_name {
                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&gname).await {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                            }
                                        }
                                    }
                                    Err(e) => log::error!("thumb update batch upsert failed: {e}"),
                                }
                            } else {
                                log::debug!("Skipping batch DB save: auto_save_to_database disabled");
                            }
                        });
                    }
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
                    log::info!("Done");
                    self.file_scan_progress = 1.0;
                    self.scan_done = true;
                    self.current_scan_id = None;
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
                    if !self.pending_thumb_rows.is_empty() {
                        let batch = std::mem::take(&mut self.pending_thumb_rows);
                        let group_name = self.active_logical_group_name.clone();
                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        tokio::spawn(async move {
                            if auto_save {
                                match crate::database::upsert_rows_and_get_ids(batch).await {
                                    Ok(ids) => {
                                        if let Some(gname) = group_name {
                                            if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&gname).await {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(&g.id, &ids).await;
                                            }
                                        }
                                    }
                                    Err(e) => log::error!("final scan batch upsert failed: {e}"),
                                }
                            } else {
                                log::debug!("Skipping final batch DB save: auto_save_to_database disabled");
                            }
                        });
                    }
                    // Do not bulk-refresh CLIP presence here; per-thumbnail checks will update presence incrementally.
                }
            }
            ctx.request_repaint();
        }

    }
}