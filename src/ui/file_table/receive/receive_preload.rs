impl crate::ui::file_table::FileExplorer {
    pub fn receive_preload(&mut self, ctx: &eframe::egui::Context) {
        // Integrate the library rows of the current folder into a lookup by path and merge them into the rows shown
        let started = std::time::Instant::now();
        while let Ok(mut rows) = self.db_preload_rx.try_recv() {
            crate::ui::file_table::thumbs::adopt_legacy(&self.viewer.thumbs, &mut rows);
            self.viewer.thumbs.note_rows(rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect());
            self.viewer.badges.mark_indexed(rows.iter().map(|r| r.path.as_str()));
            super::super::scope::runner::spawn_embedded_check(rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect(), self.nav.tx.clone());
            self.db_lookup.clear();
            for r in rows.into_iter() {
                self.db_lookup.insert(r.path.clone(), r);
            }
            // Merge DB-backed fields into any existing table rows that match by path
            let lookup = &self.db_lookup;
            self.table.for_each_mut(|row| {
                if row.file_type == "<DIR>" { return; }
                if let Some(db_row) = lookup.get(&row.path) {
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
            });
            // Keep current_thumb in sync as well
            if !self.current_thumb.path.is_empty() {
                if let Some(db_row) = self.db_lookup.get(&self.current_thumb.path) {
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
            if started.elapsed() >= std::time::Duration::from_millis(8) { break; }
        }
    }
}
