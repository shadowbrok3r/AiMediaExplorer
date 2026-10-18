use std::time::{Duration, Instant};

const CLIP_MSG_BUDGET: usize = 256;
const CLIP_TIME_BUDGET: Duration = Duration::from_millis(2);

impl crate::ui::file_table::FileExplorer {
    pub fn receive_clip(&mut self, ctx: &eframe::egui::Context) {
        // Process clip presence updates (embedding rows)
        let started = Instant::now();
        let mut handled = 0usize;
        while handled < CLIP_MSG_BUDGET && started.elapsed() < CLIP_TIME_BUDGET {
            let Ok(clip_embedding) = self.clip_embedding_rx.try_recv() else { return };
            handled += 1;
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
        if !self.clip_embedding_rx.is_empty() {
            ctx.request_repaint();
        }
    }
}
