
// Limit concurrent per-thumbnail embedding lookups to avoid flooding
static EMBED_CHECK_SEM: once_cell::sync::Lazy<std::sync::Arc<tokio::sync::Semaphore>> =
    once_cell::sync::Lazy::new(|| std::sync::Arc::new(tokio::sync::Semaphore::new(8)));

impl crate::ui::file_table::FileExplorer {
    pub fn receive_thumbnail(&mut self, ctx: &eframe::egui::Context) {
        if let Ok(thumbnail) = self.thumbnail_rx.try_recv() {
            // No more per-row <PAGE_DONE> path; page loads come via db_preload_rx
            // Special-case: high-res preview thumbnails use a synthetic cache key ("preview::...")
            // These should only populate the in-memory cache and not affect table rows or selection.
            if thumbnail.path.starts_with("preview::") {
                // Insert into preview cache
                crate::ui::file_table::insert_thumbnail(&mut self.viewer.thumb_cache, thumbnail.clone());
                ctx.request_repaint();
                return;
            }
            log::info!("Received: {}", thumbnail.filename);
            ctx.request_repaint();
            // Special control message: password prompt request
            if thumbnail.file_type == "<PASSWORD_REQUIRED>" && !thumbnail.path.is_empty() {
                let archive_path = thumbnail.path.clone();
                if !self.pending_zip_passwords.contains(&archive_path) {
                    self.pending_zip_passwords.push_back(archive_path.clone());
                }
                self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                self.show_zip_modal = true;
            }

            // If this message carries image data, update the row thumbnail in-place
            if thumbnail.thumbnail_b64.is_some() {
                if let Some(existing) = self.table.iter_mut().find(|r| r.path == thumbnail.path) {
                    existing.thumbnail_b64 = thumbnail.thumbnail_b64.clone();
                    ctx.request_repaint();
                }
            }
            // If we receive fresh rows while showing similarity, clear that mode
            if self.viewer.showing_similarity {
                self.viewer.showing_similarity = false;
                self.viewer.similar_scores.clear();
            }

            log::info!("self.viewer.mode: {:?}", self.viewer.mode);

            if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::FileSystem {
                log::info!("FS mode: {}", thumbnail.file_type);
                if thumbnail.file_type == "<DIR>" {
                    // Navigate into directory
                    self.current_path = thumbnail.path.clone();
                    self.table.clear();
                    self.table_index.clear();
                    self.populate_current_directory();
                } else {
                    log::info!("Opening preview for: {}", thumbnail.filename);
                    self.current_thumb = thumbnail;
                    if !self.current_thumb.filename.is_empty() {
                        self.open_preview_pane = true;
                    }
                }
            } else {
                log::info!("DB mode: {}", thumbnail.file_type);
                // Database mode:
                // If new row arriving from async DB load
                if !self.table.iter().any(|r| r.path == thumbnail.path) {
                    // New row arriving from async DB load
                    self.table.push(thumbnail.clone());
                    let idx = self.table.len()-1;
                    self.table_index.insert(thumbnail.path.clone(), idx);
                    // Update paging tracking
                    self.db_last_batch_len += 1;
                    // Advance offset per item to allow incremental "Load More"
                    self.db_offset += 1;

                    // Kick off a lightweight, per-row CLIP embedding presence check for this item
                    if thumbnail.file_type != "<DIR>" {
                        let tx_clip = self.viewer.clip_embedding_tx.clone();
                        let thumb = thumbnail.clone();
                        // Skip if we already know we have an embedding for this path
                        if !self.viewer.clip_presence.contains(&thumb.path) {
                            tokio::spawn(async move {
                                // Concurrency guard
                                let _permit = EMBED_CHECK_SEM.clone().acquire_owned().await.ok();
                                let embedding = thumb.get_embedding().await.unwrap_or_default();
                                let _ = tx_clip.try_send(embedding);
                            });
                        }
                    }
                } else {
                    // Selection event: open preview
                    self.current_thumb = thumbnail.clone();
                    if !self.current_thumb.filename.is_empty() {
                        self.open_preview_pane = true;
                    }
                }
            }
        }
    }
}
        