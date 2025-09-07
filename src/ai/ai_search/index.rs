use crate::ai::Thumbnail;
 
impl crate::ai::AISearchEngine {
    // Internal generalized indexer with optional force flag (bypass hash/description skip logic for reindex only).
    // NOTE: Description & embedding generation now rely solely on the corresponding auto_* atomic flags
    // (currently only auto_descriptions_enabled) and no longer use `force` to override.
    pub(crate) async fn index_file_internal(
        &self,
        mut metadata: crate::Thumbnail,
        force: bool,
    ) -> anyhow::Result<(), anyhow::Error> {
        // Reentrancy / duplicate guard
        {
            let mut guard = self.indexing_in_progress.lock().await;
            if let Some(count) = guard.get_mut(&metadata.path) {
                *count += 1;
                log::warn!(
                    "[AI] Skipping duplicate indexing request for {} (active reentry count={})",
                    metadata.path,
                    count
                );
                return Ok(());
            } else {
                guard.insert(metadata.path.clone(), 1);
            }
        }
        let path = std::path::PathBuf::from(&metadata.path);
        log::debug!(
            "Indexing file: {} (type: {})",
            metadata.path,
            metadata.file_type
        );
        // Compute hash to detect changes
        metadata.hash = self.compute_file_hash(&path).ok();
        if !force {
            if let Some(existing) = self.get_file_metadata(&metadata.path).await {
                let clip_embedding = existing.clone().get_embedding().await?;
                if existing.hash.is_some() && existing.hash == metadata.hash {
                    // Determine whether we can safely skip: only skip if (a) non-image OR (b) image already has description AND either clip embedding present OR auto clip disabled.
                    let auto_clip = self
                        .auto_clip_enabled
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let is_image = std::path::Path::new(&metadata.path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                        .map(|ext| crate::is_image(ext.as_str()))
                        .unwrap_or(false);
                    let can_skip = if !is_image {
                        true
                    } else {
                        let has_desc = existing.description.is_some();
                        let has_clip = !clip_embedding.embedding.is_empty();
                        has_desc && (has_clip || !auto_clip)
                    };
                    if can_skip {
                        log::info!(
                            "[AI] Skipping re-index (unchanged hash) for {}",
                            metadata.path
                        );
                        // Still, if auto_clip enabled and clip missing, schedule embedding before returning.
                        if is_image && auto_clip
                            && clip_embedding.embedding.is_empty()
                        {
                            let arc_self = std::sync::Arc::new(self.clone());
                            let p_clone = metadata.path.clone();
                            tokio::spawn(async move {
                                match arc_self.clip_generate_for_paths(&[p_clone.clone()]).await {
                                    Ok(added) => log::info!("[CLIP] Manual per-item generation: added {added} for {p_clone}"),
                                    Err(e) => log::error!("engine.clip_generate_for_paths: {e:?}")
                                }
                            });
                        }
                        return Ok(());
                    }
                }
            }
        }

        if metadata.thumbnail_b64.is_none() {
            metadata.thumbnail_b64 = metadata.thumbnail_b64.clone();
        }
        if let Err(e) = self.cache_thumbnail_and_metadata(&metadata).await {
            log::warn!("Thumbnail cache failed: {}", e);
        }

        // Store in memory (replace existing entry if same path) BEFORE scheduling async enrichment to avoid race (apply_vision_description needs it present)
        let mut files = self.files.lock().await;
        if let Some(existing_idx) = files.iter().position(|f| f.path == metadata.path) {
            files[existing_idx] = metadata.clone();
        } else {
            files.push(metadata.clone());
        }
        log::debug!("Finished indexing file");

        // Now that metadata is in-memory, schedule image enrichment (description + CLIP) if enabled.
        let is_image = std::path::Path::new(&metadata.path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .map(|ext| crate::is_image(ext.as_str()))
            .unwrap_or(false);
        if is_image && path.exists() {
            let auto_desc = self
                .auto_descriptions_enabled
                .load(std::sync::atomic::Ordering::Relaxed);
            let auto_clip = self
                .auto_clip_enabled
                .load(std::sync::atomic::Ordering::Relaxed);
            if auto_desc {
                let arc_self = std::sync::Arc::new(self.clone());
                let schedule_path = path.clone();
                log::info!(
                    "[AI] Scheduling async vision description for {} (post-insert)",
                    metadata.path
                );
                tokio::spawn(async move {
                    arc_self.spawn_generate_vision_description(schedule_path);
                });
            }
            if auto_clip {
                let arc_self2 = std::sync::Arc::new(self.clone());
                let clip_path = metadata.path.clone();
                tokio::spawn(async move {
                    // Respect overwrite flag: only regenerate if missing, unless overwrite is enabled
                    let overwrite = crate::database::settings::load_settings().clip_overwrite_embeddings;
                    let mut should_run = overwrite;
                    if !should_run {
                        if let Some(meta) = arc_self2.get_file_metadata(&clip_path).await {
                            if let Ok(emb) = meta.clone().get_embedding().await { should_run = emb.embedding.is_empty(); }
                        }
                    }
                    if should_run {
                        match arc_self2.clip_generate_for_paths(&[clip_path.clone()]).await {
                            Ok(added) => log::info!("[CLIP] Manual per-item generation: added {added} for {clip_path}"),
                            Err(e) => log::error!("engine.clip_generate_for_paths: {e:?}")
                        }
                    }
                });
            }
        }

        // Remove reentrancy marker
        {
            let mut guard = self.indexing_in_progress.lock().await;
            guard.remove(&metadata.path);
        }
        Ok(())
    }

    // Return list of currently loaded (cached) file paths.
    pub async fn list_indexed_paths(&self) -> Vec<String> {
        let files = self.files.lock().await;
        files.iter().map(|f| f.path.clone()).collect()
    }

    pub async fn index_file(&self, metadata: Thumbnail) -> anyhow::Result<(), anyhow::Error> {
        self.index_file_internal(metadata, false).await
    }

    pub async fn ensure_index_worker(&self) {
        let mut guard = self.index_tx.lock().await;
        if guard.is_some() {
            return;
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Thumbnail>();
        *guard = Some(tx);
        let engine = self.clone();
        tokio::spawn(async move {
            while let Some(meta) = rx.recv().await {
                let path = meta.path.clone();
                engine
                    .index_active
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(e) = engine.index_file(meta).await {
                    log::warn!("[AI] queue index failed for {}: {}", path, e);
                }
                engine
                    .index_active
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                engine
                    .index_completed
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Decrement queue len (saturating)
                engine
                    .index_queue_len
                    .fetch_update(
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                        |v| Some(v.saturating_sub(1)),
                    )
                    .ok();
            }
            log::info!("[AI] indexing worker channel closed");
        });
    }

    /// Enqueue a file metadata record for background indexing. Returns false if queue not ready yet.
    pub async fn enqueue_index(&self, meta: Thumbnail) -> bool {
        // Global gate: only allow enqueue when auto indexing is enabled
        let auto_on = crate::database::settings::load_settings().auto_indexing;
        if !auto_on {
            return false;
        }
        if self.index_tx.lock().await.is_none() {
            self.ensure_index_worker().await;
        }
        let sent = if let Some(tx) = self.index_tx.lock().await.as_ref() {
            tx.send(meta).is_ok()
        } else {
            false
        };
        if sent {
            self.index_queue_len
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        sent
    }

    pub fn cancel_bulk_descriptions(&self) {
        self.cancel_bulk
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn reset_bulk_cancel(&self) {
        self.cancel_bulk
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
