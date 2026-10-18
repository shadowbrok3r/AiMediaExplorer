
impl crate::ui::file_table::FileExplorer {
    pub fn receive_preload(&mut self, ctx: &eframe::egui::Context) {
        // First, integrate any preloaded DB rows for current path into a fast lookup by path
        let started = std::time::Instant::now();
        while let Ok(mut rows) = self.db_preload_rx.try_recv() {
            crate::ui::file_table::thumbs::adopt_legacy(&self.viewer.thumbs, &mut rows);
            self.viewer.thumbs.note_rows(rows.iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect());
            // Page-aware apply: only apply to table if these rows belong to the active loading page
            let expecting_page = self.db_loading_page;
            let current_page = self.db_current_page;
            let rows_len = rows.len();
            if rows_len > 0 {
                // Determine which page these rows correspond to based on current offset when spawned
                // We treat any incoming rows as the ones requested for expecting_page
                if let Some(p) = expecting_page {
                    if p == current_page {
                        // Apply to table (fresh page swap)
                        self.viewer.thumbs.new_scope(self.viewer.scope);
                        self.table.clear();
                        self.table_index.clear();
                        // Clamp to page size for safety
                        for r in rows.iter().take(self.db_limit) {
                            let idx = self.table.len();
                            self.table_index.insert(r.path.clone(), idx);
                            self.table.push(r.clone());
                        }
                        self.db_last_batch_len = self.table.len();
                        self.db_offset = p * self.db_limit + self.db_last_batch_len;
                        // Cache page snapshot
                        self.cache_db_page(p, self.table.iter().cloned().collect());
                        self.db_max_loaded_page = self.db_max_loaded_page.max(p);
                        self.db_reached_end = rows_len < self.db_limit;
                        self.db_loading = false;
                        self.db_loading_page = None;
                    } else {
                        // Not the page we're currently viewing; cache only
                        self.cache_db_page(p, rows.clone());
                        self.db_max_loaded_page = self.db_max_loaded_page.max(p);
                        if rows_len < self.db_limit { self.db_reached_end = true; }
                        // Keep loading state as-is for the actual active page
                    }
                } else {
                    // No page is expected (shouldn't happen), ignore apply but keep lookup merge below
                }
            }
            self.db_lookup.clear();
            for r in rows.into_iter() {
                self.db_lookup.insert(r.path.clone(), r);
            }
            // Merge DB-backed fields into any existing table rows that match by path
            for row in self.table.iter_mut() {
                if row.file_type == "<DIR>" { continue; }
                if let Some(db_row) = self.db_lookup.get(&row.path) {
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