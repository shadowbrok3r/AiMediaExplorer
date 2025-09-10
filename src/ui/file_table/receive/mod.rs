pub mod receive_preload;
pub mod receive_scan;
pub mod receive_thumbnail;
pub mod receive_ai_update;
pub mod receive_clip;

impl super::FileExplorer {
    pub fn receive(&mut self, ctx: &eframe::egui::Context) {
        self.receive_ai_update(ctx);
        self.receive_clip(ctx);
        self.receive_preload(ctx);
        self.receive_thumbnail(ctx);
        self.receive_scan(ctx);

        // If we were loading a DB page and all rows have arrived (channel drained for now), finalize batch.
        if self.viewer.mode == super::table::ExplorerMode::Database && self.db_loading {
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
            // Also check CLIP embeddings for the visible rows and update presence column
            {
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
            ctx.request_repaint();
        }
    }
}
