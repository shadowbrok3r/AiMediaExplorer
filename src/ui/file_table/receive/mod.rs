pub mod receive_preload;
pub mod receive_scan;
pub mod receive_thumbnail;
pub mod receive_ai_update;
pub mod receive_clip;
#[cfg(test)]
mod receive_tests;

impl super::FileExplorer {
    pub fn receive(&mut self, ctx: &eframe::egui::Context) {
        self.receive_ai_update(ctx);
        self.receive_clip(ctx);
        self.receive_preload(ctx);
        self.receive_thumbnail(ctx);
        self.receive_scan(ctx);
        self.receive_scope(ctx);

        // Integrate any freshly loaded filter groups
        while let Ok(groups) = self.filter_groups_rx.try_recv() {
            self.filter_groups = groups;
        }

        // Integrate the collection list the Collections menu refreshed
        while let Ok(groups) = self.logical_groups_rx.try_recv() {
            self.logical_groups = groups;
        }
    }
}
