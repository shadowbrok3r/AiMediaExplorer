
impl crate::ui::file_table::FileExplorer {
    pub fn receive_thumbnail(&mut self, ctx: &eframe::egui::Context) {
        while let Ok(thumbnail) = self.thumbnail_rx.try_recv() {
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
                    // Update paging tracking
                    self.db_last_batch_len += 1;
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
        