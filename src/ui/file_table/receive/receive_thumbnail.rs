
impl crate::ui::file_table::FileExplorer {
    pub fn receive_thumbnail(&mut self, ctx: &eframe::egui::Context) {
        while let Ok(thumbnail) = self.thumbnail_rx.try_recv() {
            ctx.request_repaint();
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
    }
}
        