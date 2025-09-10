
impl crate::ui::file_table::FileExplorer {
    pub fn receive_preload(&mut self, ctx: &eframe::egui::Context) {
        // First, integrate any preloaded DB rows for current path into a fast lookup by path
        while let Ok(rows) = self.db_preload_rx.try_recv() {
            self.db_lookup.clear();
            for r in rows.into_iter() {
                self.db_lookup.insert(r.path.clone(), r);
            }
            // Merge DB-backed fields into any existing table rows that match by path
            for row in self.table.iter_mut() {
                if row.file_type == "<DIR>" { continue; }
                if let Some(db_row) = self.db_lookup.get(&row.path) {
                    // Preserve any runtime-generated thumbnail; otherwise take DB one
                    if row.thumbnail_b64.is_none() && db_row.thumbnail_b64.is_some() {
                        row.thumbnail_b64 = db_row.thumbnail_b64.clone();
                    }
                    // Prefer DB id and metadata
                    row.id = db_row.id.clone();
                    row.db_created = db_row.db_created.clone();
                    row.hash = db_row.hash.clone();
                    row.description = db_row.description.clone();
                    row.caption = db_row.caption.clone();
                    row.tags = db_row.tags.clone();
                    row.category = db_row.category.clone();
                    row.file_type = db_row.file_type.clone();
                    if row.modified.is_none() { row.modified = db_row.modified.clone(); }
                    if row.size == 0 { row.size = db_row.size; }
                }
            }
            // Keep current_thumb in sync as well
            if !self.current_thumb.path.is_empty() {
                if let Some(db_row) = self.db_lookup.get(&self.current_thumb.path) {
                    if self.current_thumb.thumbnail_b64.is_none() && db_row.thumbnail_b64.is_some() {
                        self.current_thumb.thumbnail_b64 = db_row.thumbnail_b64.clone();
                    }
                    self.current_thumb.id = db_row.id.clone();
                    self.current_thumb.db_created = db_row.db_created.clone();
                    self.current_thumb.hash = db_row.hash.clone();
                    self.current_thumb.description = db_row.description.clone();
                    self.current_thumb.caption = db_row.caption.clone();
                    self.current_thumb.tags = db_row.tags.clone();
                    self.current_thumb.category = db_row.category.clone();
                    self.current_thumb.file_type = db_row.file_type.clone();
                    if self.current_thumb.modified.is_none() { self.current_thumb.modified = db_row.modified.clone(); }
                    if self.current_thumb.size == 0 { self.current_thumb.size = db_row.size; }
                }
            }
        
            ctx.request_repaint();
        }
    }
}