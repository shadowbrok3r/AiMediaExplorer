use std::time::{Duration, Instant};

// Limit concurrent per-thumbnail embedding lookups to avoid flooding
static EMBED_CHECK_SEM: once_cell::sync::Lazy<std::sync::Arc<tokio::sync::Semaphore>> =
    once_cell::sync::Lazy::new(|| std::sync::Arc::new(tokio::sync::Semaphore::new(8)));

const THUMBNAIL_MSG_BUDGET: usize = 64;
const THUMBNAIL_TIME_BUDGET: Duration = Duration::from_millis(4);

impl crate::ui::file_table::FileExplorer {
    pub fn receive_thumbnail(&mut self, ctx: &eframe::egui::Context) {
        let started = Instant::now();
        let mut handled = 0usize;
        while handled < THUMBNAIL_MSG_BUDGET && started.elapsed() < THUMBNAIL_TIME_BUDGET {
            let Ok(thumbnail) = self.thumbnail_rx.try_recv() else { return };
            handled += 1;
            self.apply_thumbnail_message(ctx, thumbnail);
        }
        if !self.thumbnail_rx.is_empty() {
            ctx.request_repaint();
        }
    }

    fn apply_thumbnail_message(&mut self, ctx: &eframe::egui::Context, mut thumbnail: crate::Thumbnail) {
        // Pixels in the message go to the thumbnail service; the row itself carries none.
        crate::ui::file_table::thumbs::adopt_legacy_one(&self.viewer.thumbs, &mut thumbnail);
        log::debug!("Received: {}", thumbnail.filename);
        ctx.request_repaint();
        // Special control message: password prompt request
        if thumbnail.file_type == "<PASSWORD_REQUIRED>" && !thumbnail.path.is_empty() {
            let archive_path = thumbnail.path.clone();
            if !self.pending_zip_passwords.contains(&archive_path) {
                self.pending_zip_passwords.push_back(archive_path.clone());
            }
            self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
            self.show_zip_modal = true;
            return;
        }

        // If we receive fresh rows while showing similarity, clear that mode
        if self.viewer.showing_similarity {
            self.viewer.showing_similarity = false;
            self.viewer.similar_scores.clear();
        }

        if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::FileSystem {
            if thumbnail.file_type == "<DIR>" {
                // Navigate into directory
                self.current_path = thumbnail.path.clone();
                self.table.clear();
                self.populate_current_directory();
            }
        } else if !thumbnail.path.is_empty() && !self.table.contains_path(&thumbnail.path) {
            // Database mode: new row arriving from an async DB load
            self.table.push(thumbnail.clone());
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
        }
    }
}
