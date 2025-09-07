use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use std::{path::PathBuf, sync::Arc};
use eframe::egui::Context;
use crate::{ScanEnvelope, next_scan_id, ui::file_table::AIUpdate};

impl super::FileExplorer {
    pub fn receive(&mut self, ctx: &Context) {
        while let Ok(thumbnail) = self.thumbnail_rx.try_recv() {
            ctx.request_repaint();
            if self.viewer.mode == super::viewer::ExplorerMode::FileSystem {
                if thumbnail.file_type == "<DIR>" {
                    // Navigate into directory
                    self.current_path = thumbnail.path.clone();
                    self.table.clear();
                    self.populate_current_directory();
                } else {
                    self.current_thumb = thumbnail;
                    if !self.current_thumb.filename.is_empty() {
                        self.open_preview_pane = true;
                    }
                }
            } else {
                // Database mode:
                // If this path already exists in the table, treat as a selection (open preview)
                if self.table.iter().any(|r| r.path == thumbnail.path) {
                    self.current_thumb = thumbnail.clone();
                    if !self.current_thumb.filename.is_empty() {
                        self.open_preview_pane = true;
                    }
                } else {
                    // New row arriving from async DB load
                    self.table.push(thumbnail.clone());
                    // Update paging tracking
                    self.db_last_batch_len += 1;
                }
            }
        }
        
        // If we were loading a DB page and all rows have arrived (channel drained for now), finalize batch.
        if self.viewer.mode == super::viewer::ExplorerMode::Database && self.db_loading {
            // Heuristic: if last batch len < limit then no more pages.
            if self.db_last_batch_len > 0 {
                // Completed a page
                self.db_offset += self.db_last_batch_len;
            }
            self.db_loading = false;
            // After loading a DB page, hydrate minimal AI metadata for these paths into the engine
            // so that CLIP searches have in-memory candidates.
            let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
            let paths: Vec<String> = self
                .table
                .iter()
                .filter(|r| r.file_type != "<DIR>")
                .map(|r| r.path.clone())
                .collect();
            if !paths.is_empty() {
                tokio::spawn(async move {
                    let _ = engine.hydrate_directory_paths(&paths).await;
                });
            }
        }

        while let Ok(env) = self.scan_rx.try_recv() {
            match env.msg {
                crate::utilities::scan::ScanMsg::Found(item) => {
                    log::info!("Found");
                    // (extension filtering handled during scan)
                    if let Some(row) = crate::file_to_thumbnail(&item) {
                        let mut row = row;
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
                        self.table.push(row);
                        ctx.request_repaint();
                        // Enqueue for AI indexing (images/videos). Use lightweight Thumbnail conversion.
                        let path_clone = item.path.to_string_lossy().to_string();
                        let ftype = {
                            if let Some(ext) = item
                                .path
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                            {
                                if crate::is_image(ext.as_str()) {
                                    "image".to_string()
                                } else if crate::is_video(ext.as_str()) {
                                    "video".to_string()
                                } else {
                                    "other".to_string()
                                }
                            } else {
                                "other".to_string()
                            }
                        };

                        let meta = crate::database::Thumbnail {
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
                            thumb_b64: item.thumb_data.clone(),
                            hash: None,
                            description: None,
                            caption: None,
                            tags: Vec::new(),
                            category: None,
                        };

                        tokio::spawn(async move {
                            let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await;
                        });
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
                        if let Some(row) = crate::file_to_thumbnail(&item) {
                            let mut row = row;
                            if let Some(ext) = std::path::Path::new(&row.path)
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                            {
                                if crate::utilities::types::is_archive(&ext) {
                                    row.file_type = "<ARCHIVE>".into();
                                }
                            }
                            self.table.push(row);
                            ctx.request_repaint();
                            // Schedule thumbnail generation if none yet
                            if item.thumb_data.is_none() {
                                let p_str = item.path.to_string_lossy().to_string();
                                if self.thumb_scheduled.insert(p_str.clone()) {
                                    newly_enqueued.push(p_str);
                                }
                            }
                            // Prepare indexing metadata
                            let ftype = {
                                if let Some(ext) = item
                                    .path
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .map(|s| s.to_ascii_lowercase())
                                {
                                    if crate::is_image(ext.as_str()) {
                                        "image".to_string()
                                    } else if crate::is_video(ext.as_str()) {
                                        "video".to_string()
                                    } else {
                                        "other".to_string()
                                    }
                                } else {
                                    "other".to_string()
                                }
                            };
                            to_index.push(crate::database::Thumbnail {
                                id: None,
                                db_created: None,
                                path: item.path.to_string_lossy().to_string(),
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
                                thumb_b64: item.thumb_data.clone(),
                                hash: None,
                                description: None,
                                caption: None,
                                tags: Vec::new(),
                                category: None,
                            });
                        }
                        if self.pending_thumb_rows.len() >= 32 {
                            let batch = std::mem::take(&mut self.pending_thumb_rows);
                            tokio::spawn(async move {
                                if let Err(e) = crate::database::save_thumbnail_batch(batch).await {
                                    log::error!("scan batch persistence failed: {e}");
                                }
                            });
                        }
                    }
                    if !to_index.is_empty() {
                        tokio::spawn(async move {
                            for meta in to_index.into_iter() {
                                let _ = crate::ai::GLOBAL_AI_ENGINE.enqueue_index(meta).await;
                            }
                        });
                    }
                    if !newly_enqueued.is_empty() {
                        let scan_tx = self.scan_tx.clone();
                        let sem = self.thumb_semaphore.clone();
                        let scan_id_for_updates = next_scan_id();
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
                                    let _ = tx_clone.try_send(ScanEnvelope {
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
                        tokio::spawn(async move {
                            if let Err(e) = crate::database::save_thumbnail_batch(batch).await {
                                log::error!("thumb update persistence failed: {e}");
                            }
                        });
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
                    if !self.pending_thumb_rows.is_empty() {
                        let batch = std::mem::take(&mut self.pending_thumb_rows);
                        tokio::spawn(async move {
                            if let Err(e) = crate::database::save_thumbnail_batch(batch).await {
                                log::error!("final scan batch persistence failed: {e}");
                            }
                        });
                    }
                }
            }
            ctx.request_repaint();
        }

        // Process AI streaming updates
        while let Ok(update) = self.ai_update_rx.try_recv() {
            ctx.request_repaint();
            match update {
                AIUpdate::Interim { path, text } => {
                    // Auto-follow: always advance to the currently streaming image if enabled.
                    if self.follow_active_vision && !path.is_empty() {
                        if self.current_thumb.path != path {
                            if let Some(row) = self.table.iter().find(|r| r.path == path) {
                                self.current_thumb = row.clone();
                                self.open_preview_pane = true; // ensure visible
                            }
                        }
                    }
                    self.streaming_interim.insert(path, text);
                }
                AIUpdate::Final {
                    path,
                    description,
                    caption,
                    category,
                    tags,
                } => {
                    if self.follow_active_vision && !path.is_empty() {
                        if self.current_thumb.path != path {
                            if let Some(row) = self.table.iter().find(|r| r.path == path) {
                                self.current_thumb = row.clone();
                                self.open_preview_pane = true;
                            }
                        }
                    }
                    self.streaming_interim.remove(&path);
                    // Update counters: a pending item finished.
                    if self.vision_pending > 0 {
                        self.vision_pending -= 1;
                    }
                    self.vision_completed += 1;
                    let desc_clone_for_row = description.clone();
                    let caption_clone_for_row = caption.clone();
                    let category_clone_for_row = category.clone();
                    let tags_clone_for_row = tags.clone();
                    if let Some(row) = self.table.iter_mut().find(|r| r.path == path) {
                        if !desc_clone_for_row.trim().is_empty() {
                            row.description = Some(desc_clone_for_row.clone());
                        }
                        if let Some(c) = caption_clone_for_row.clone() {
                            if !c.trim().is_empty() {
                                row.caption = Some(c);
                            }
                        }
                        if let Some(cat) = category_clone_for_row.clone() {
                            if !cat.trim().is_empty() {
                                row.category = Some(cat);
                            }
                        }
                        if !tags_clone_for_row.is_empty() {
                            row.tags = tags_clone_for_row.clone();
                        }
                    }
                    if self.current_thumb.path == path {
                        if !description.trim().is_empty() {
                            self.current_thumb.description = Some(description.clone());
                        }
                        if let Some(c) = caption.clone() {
                            if !c.trim().is_empty() {
                                self.current_thumb.caption = Some(c);
                            }
                        }
                        if let Some(cat) = category.clone() {
                            if !cat.trim().is_empty() {
                                self.current_thumb.category = Some(cat);
                            }
                        }
                        if !tags.is_empty() {
                            self.current_thumb.tags = tags.clone();
                        }
                    }
                    // Defensive persistence: ensure final AI metadata is saved to DB (idempotent if already saved by engine)
                    {
                        let persist_path = path.clone();
                        let description_clone = description.clone();
                        let caption_clone = caption.clone();
                        let category_clone = category.clone();
                        let tags_clone = tags.clone();
                        // Prefer current_thumb if active, else table row, else the cloned incoming values.
                        let (desc_final, cap_final, cat_final, tags_final) =
                            if self.current_thumb.path == persist_path {
                                (
                                    self.current_thumb.description.clone().unwrap_or_default(),
                                    self.current_thumb.caption.clone().unwrap_or_default(),
                                    self.current_thumb
                                        .category
                                        .clone()
                                        .unwrap_or_else(|| "general".into()),
                                    self.current_thumb.tags.clone(),
                                )
                            } else if let Some(row) =
                                self.table.iter().find(|r| r.path == persist_path)
                            {
                                (
                                    row.description.clone().unwrap_or_default(),
                                    row.caption.clone().unwrap_or_default(),
                                    row.category.clone().unwrap_or_else(|| "general".into()),
                                    row.tags.clone(),
                                )
                            } else {
                                (
                                    description_clone,
                                    caption_clone.unwrap_or_default(),
                                    category_clone.unwrap_or_else(|| "general".into()),
                                    tags_clone,
                                )
                            };
                        tokio::spawn(async move {
                            let vd = crate::ai::VisionDescription {
                                description: desc_final,
                                caption: cap_final,
                                category: cat_final,
                                tags: tags_final,
                            };
                            if let Err(e) = crate::ai::GLOBAL_AI_ENGINE
                                .apply_vision_description(&persist_path, &vd)
                                .await
                            {
                                log::warn!(
                                    "[AI] UI final persist failed for {}: {}",
                                    persist_path,
                                    e
                                );
                            }
                        });
                    }
                }
                AIUpdate::SimilarResults {
                    origin_path,
                    results,
                } => {
                    self.similar_origin = Some(origin_path.clone());
                    self.similar_results = results;
                    if self.similar_results.is_empty() {
                        // keep modal open showing placeholder
                        self.show_similar_modal = true;
                    } else {
                        self.show_similar_modal = true; // ensure visible
                    }
                }
            }
        }

        // Process metadata update messages
        while let Ok(update) = self.meta_rx.try_recv() {
            ctx.request_repaint();
            let path = update.path.clone();
            let description = update.description.clone();
            let caption = update.caption.clone();
            let category = update.category.clone();
            let tags = update.tags.clone();
            // Update in-memory table row(s)
            if let Some(row) = self.table.iter_mut().find(|r| r.path == path) {
                if let Some(desc) = description.clone() {
                    row.description = Some(desc);
                }
                if let Some(cap) = caption.clone() {
                    row.caption = Some(cap);
                }
                if let Some(cat) = category.clone() {
                    row.category = Some(cat);
                }
                row.tags = tags.clone();
            }
            // Update current_thumb if it matches
            if self.current_thumb.path == path {
                if let Some(desc) = description {
                    self.current_thumb.description = Some(desc);
                }
                if let Some(cap) = caption {
                    self.current_thumb.caption = Some(cap);
                }
                if let Some(cat) = category {
                    self.current_thumb.category = Some(cat);
                }
                self.current_thumb.tags = tags.clone();
            }
        }

        // Process clip presence updates (embedding rows)
        while let Ok(clip_embedding) = self.clip_embedding_rx.try_recv() {
            let has = !clip_embedding.embedding.is_empty();
            if has {
                self.clip_presence.insert(clip_embedding.path.clone());
            } else {
                self.clip_presence.remove(&clip_embedding.path);
            }
            ctx.request_repaint();
        }
    }
}
