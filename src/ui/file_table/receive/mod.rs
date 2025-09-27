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
        // Keep global active group name snapshot in sync for viewer context menu
        {
            let guard = crate::ui::file_table::ACTIVE_GROUP_NAME.get_or_init(|| std::sync::Mutex::new(None));
            if let Ok(mut g) = guard.lock() { *g = self.active_logical_group_name.clone(); }
        }

        // Integrate any freshly loaded filter groups
        while let Ok(groups) = self.filter_groups_rx.try_recv() {
            self.filter_groups = groups;
        }

        // Integrate any freshly loaded logical groups
        while let Ok(groups) = self.logical_groups_rx.try_recv() {
            self.logical_groups = groups;
            if self.logical_groups.is_empty() {
                // Ensure at least Default exists, then refresh list
                let tx = self.logical_groups_tx.clone();
                tokio::spawn(async move {
                    let _ = crate::database::LogicalGroup::create("Default").await;
                    if let Ok(groups) = crate::database::LogicalGroup::list_all().await { let _ = tx.try_send(groups); }
                });

            } else if self.active_logical_group_name.is_none() {
                // Choose a sensible default: previously used from settings or first available
                let preferred = crate::ui::file_table::active_group_name();
                if let Some(name) = preferred {
                    if self.logical_groups.iter().any(|g| g.name == name) { self.active_logical_group_name = Some(name); }
                }
                if self.active_logical_group_name.is_none() {
                    self.active_logical_group_name = Some(self.logical_groups[0].name.clone());
                }
            }
        }
        
        // If we were loading a DB page or full view, finalize after preload integration
        if self.viewer.mode == super::table::ExplorerMode::Database && self.db_loading {
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
            ctx.request_repaint();
        }
    }
}
