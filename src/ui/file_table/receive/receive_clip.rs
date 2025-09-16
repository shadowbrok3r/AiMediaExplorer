impl crate::ui::file_table::FileExplorer {
    pub fn receive_clip(&mut self, ctx: &eframe::egui::Context) {
        // Process clip presence updates (embedding rows)
        if let Ok(clip_embedding) = self.clip_embedding_rx.try_recv() {
            let has = !clip_embedding.embedding.is_empty();
            if has {
                log::debug!("clip: embedding present for {}", clip_embedding.path);
                self.clip_presence.insert(clip_embedding.path.clone());
                self.viewer.clip_presence.insert(clip_embedding.path.clone());
                if let Some(h) = clip_embedding.hash.as_ref() {
                    self.viewer.clip_presence_hashes.insert(h.clone());
                }
            } else {
                log::debug!("clip: no embedding for {}", clip_embedding.path);
                self.clip_presence.remove(&clip_embedding.path);
                self.viewer.clip_presence.remove(&clip_embedding.path);
                if let Some(h) = clip_embedding.hash.as_ref() {
                    self.viewer.clip_presence_hashes.remove(h);
                }
            }
            ctx.request_repaint();
        }
    }
}