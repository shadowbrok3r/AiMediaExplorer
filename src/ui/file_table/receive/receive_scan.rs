use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use std::{path::PathBuf, sync::Arc};

impl crate::ui::file_table::FileExplorer {
    pub fn receive_scan(&mut self, ctx: &eframe::egui::Context) {
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
                continue;
            }
            match env.msg {
                crate::utilities::scan::ScanMsg::FoundDir(dir) => {
                    if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&dir.path) {
                        if let Some(existing) = self.table.iter_mut().find(|r| r.path == row.path) {
                            // Update in-place to preserve selection state and caching
                            let keep_thumb = existing.thumbnail_b64.clone();
                            *existing = row;
                            if existing.thumbnail_b64.is_none() { existing.thumbnail_b64 = keep_thumb; }
                        } else {
                            self.table.push(row);
                        }
                        // Track in last scan snapshot
                        let dp = dir.path.to_string_lossy().to_string();
                        if !self.last_scan_paths.contains(&dp) {
                            if let Some(r) = self.table.iter().find(|r| r.path == dp) {
                                self.last_scan_rows.push(r.clone());
                                self.last_scan_paths.insert(r.path.clone());
                            }
                        }
                        ctx.request_repaint();
                    }
                }
                crate::utilities::scan::ScanMsg::FoundDirBatch(dirs) => {
                    for d in dirs.iter() {
                        if let Some(row) = crate::ui::file_table::FileExplorer::directory_to_thumbnail(&d.path) {
                            self.table.push(row);
                        }
                    }
                    ctx.request_repaint();
                }
                crate::utilities::scan::ScanMsg::Found(item) => {
                    log::info!("Found");
                    // (extension filtering handled during scan)
                    if let Some(row0) = crate::file_to_thumbnail(&item) {
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
                        self.table.push(row.clone());
                        // Snapshot rows to allow restore later
                        if !self.last_scan_paths.contains(&row.path) {
                            self.last_scan_rows.push(row.clone());
                            self.last_scan_paths.insert(row.path.clone());
                        }
                        ctx.request_repaint();
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
                                id: None,
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
                                parent_dir: parent_dir
                            }
                        };

                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        if auto_save {
                            let group_name = self.active_logical_group_name.clone();
                            let meta_clone = meta.clone();
                            let do_index = self.viewer.ui_settings.auto_indexing;
                            tokio::spawn(async move {
                                // Upsert row always when auto-save is enabled
                                if let Ok(ids) = crate::database::upsert_rows_and_get_ids(vec![meta_clone.clone()]).await {
                                    // Optionally add to group if one is active
                                    if let Some(group_name) = group_name {
                                        if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                                            if let Some(gid) = g.id.as_ref() {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                            }
                                        }
                                    }
                                }
                                // Index only if enabled
                                if do_index {
                                    let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta_clone).await;
                                }
                            });
                        }
                    }
                }
                crate::utilities::scan::ScanMsg::FoundBatch(batch) => {
                    log::info!("FoundBatch");
                    let mut newly_enqueued: Vec<String> = Vec::new();
                    let mut to_index: Vec<crate::database::Thumbnail> = Vec::new();
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
                    for item in batch.into_iter() {
                        // (extension filtering handled during scan)
                        if let Some(ref ex) = excluded {
                            let lp = item.path.to_string_lossy().to_ascii_lowercase();
                            if ex.iter().any(|t| lp.contains(t)) {
                                continue;
                            }
                        }
                        if let Some(row0) = crate::file_to_thumbnail(&item) {
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
                            if let Some(existing) = self.table.iter_mut().find(|r| r.path == row_snapshot.path) {
                                let keep_thumb = existing.thumbnail_b64.clone();
                                *existing = row;
                                if existing.thumbnail_b64.is_none() { existing.thumbnail_b64 = keep_thumb; }
                            } else {
                                self.table.push(row_snapshot.clone());
                            }
                            // snapshot
                            if !self.last_scan_paths.contains(&row_snapshot.path) {
                                self.last_scan_rows.push(row_snapshot.clone());
                                self.last_scan_paths.insert(row_snapshot.path.clone());
                            }
                            ctx.request_repaint();
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
                                    id: None,
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
                                    parent_dir
                                });
                            }
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
                                                    if let Some(gid) = g.id.as_ref() {
                                                        let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => log::error!("scan batch upsert failed: {e}"),
                                    }
                                } else {
                                    log::debug!("Skipping batch DB save: auto_save_to_database disabled");
                                }
                            });
                        }
                    }
                    if !to_index.is_empty() {
                        let do_index = self.viewer.ui_settings.auto_indexing;
                        let auto_save = self.viewer.ui_settings.auto_save_to_database;
                        let group_name = self.active_logical_group_name.clone();
                        tokio::spawn(async move {
                            if auto_save {
                                if let Ok(ids) = crate::database::upsert_rows_and_get_ids(to_index.clone()).await {
                                    if let Some(group_name) = group_name {
                                        if let Ok(Some(g)) = crate::database::LogicalGroup::get_by_name(&group_name).await {
                                            if let Some(gid) = g.id.as_ref() {
                                                let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                            }
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
                    if let Some(found) = self
                        .table
                        .iter_mut()
                        .find(|f| PathBuf::from(f.path.clone()) == path)
                    {
                        found.thumbnail_b64 = Some(thumb.clone());
                        self.pending_thumb_rows.push(found.clone());
                    } else {
                        log::error!("Not found");
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
                                                if let Some(gid) = g.id.as_ref() {
                                                    let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                }
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
                    
                    // If we have pending encrypted archives, open the modal now
                    if !self.pending_zip_passwords.is_empty() {
                        self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                        self.show_zip_modal = true;
                    }

                    // Record the root associated with this scan for UX
                    self.last_scan_root = Some(self.current_path.clone());
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
                                                if let Some(gid) = g.id.as_ref() {
                                                    let _ = crate::database::LogicalGroup::add_thumbnails(gid, &ids).await;
                                                }
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
                    // Refresh CLIP presence for all visible rows now that thumbnails are ready
                    let rows: Vec<crate::database::Thumbnail> = self
                        .table
                        .iter()
                        .filter(|r| r.file_type != "<DIR>")
                        .cloned()
                        .collect();
                    if !rows.is_empty() {
                        let tx_clip = self.viewer.clip_embedding_tx.clone();
                        tokio::spawn(async move {
                            for r in rows.into_iter() {
                                let _ = tx_clip.try_send(r.get_embedding().await.unwrap_or_default());
                            }
                        });
                    }
                }
            }
            ctx.request_repaint();
        }

    }
}