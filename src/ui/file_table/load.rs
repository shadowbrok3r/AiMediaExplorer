
impl super::FileExplorer {
    /// Load a virtual "folders" view where each folder is a distinct category
    pub fn load_virtual_categories_view(&mut self) {
        if self.db_loading { return; }
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.db_loading = true;
        self.table.clear();
        self.table_index.clear();
        self.thumb_scheduled.clear();
        self.pending_thumb_rows.clear();
        self.viewer.clip_presence.clear();
        self.viewer.clip_presence_hashes.clear();
        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            match crate::Thumbnail::list_distinct_categories().await {
                Ok(cats) => {
                    for cat in cats.into_iter() {
                        let mut row = crate::Thumbnail::default();
                        row.file_type = "<DIR>".to_string();
                        row.filename = cat.clone();
                        row.path = format!("cat://{cat}");
                        row.parent_dir = "cat://".to_string();
                        let _ = tx.try_send(row);
                    }
                }
                Err(e) => log::error!("Virtual categories load failed: {e:?}"),
            }
        });
        // Mark as not loading so UI can interact; rows will stream in via channel
        self.db_loading = false;
    }

    /// Load a virtual "folders" view where each folder is a distinct tag
    pub fn load_virtual_tags_view(&mut self) {
        if self.db_loading { return; }
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.db_loading = true;
        self.table.clear();
        self.table_index.clear();
        self.thumb_scheduled.clear();
        self.pending_thumb_rows.clear();
        self.viewer.clip_presence.clear();
        self.viewer.clip_presence_hashes.clear();
        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            match crate::Thumbnail::list_distinct_tags().await {
                Ok(tags) => {
                    for tag in tags.into_iter() {
                        let mut row = crate::Thumbnail::default();
                        row.file_type = "<DIR>".to_string();
                        row.filename = tag.clone();
                        row.path = format!("tag://{tag}");
                        row.parent_dir = "tag://".to_string();
                        let _ = tx.try_send(row);
                    }
                }
                Err(e) => log::error!("Virtual tags load failed: {e:?}"),
            }
        });
        self.db_loading = false;
    }

    // Load and display rows from a logical group by name
    pub fn load_logical_group_by_name(&mut self, name: String) {
        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            match crate::database::LogicalGroup::get_by_name(&name).await {
                Ok(Some(group)) => {
                    match crate::Thumbnail::fetch_by_logical_group_id(&group.id).await {
                        Ok(rows) => {
                            let count = rows.len();
                            for r in rows.into_iter() { let _ = tx.try_send(r); }
                            log::info!("[Groups] Loaded '{}' with {} rows", name, count);
                        }
                        Err(e) => log::error!("Failed to fetch rows for group '{}': {e:?}", name),
                    }
                }
                Ok(None) => log::warn!("Logical group '{}' not found", name),
                Err(e) => log::error!("get_by_name '{}' failed: {e:?}", name),
            }
        });
    }
    
    pub fn load_database_rows(&mut self) {
        if self.db_loading {
            return;
        }
        self.db_all_view = false;
        // Reset similarity state when (re)loading DB pages
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.db_loading = true;
        // When offset is zero we are (re)loading fresh page; clear existing rows
        if self.db_offset == 0 {
            self.table.clear();
            self.table_index.clear();
            self.thumb_scheduled.clear();
            self.pending_thumb_rows.clear();
            // Reset CLIP presence caches for a fresh page
            self.viewer.clip_presence.clear();
            self.viewer.clip_presence_hashes.clear();
        }
        let tx = self.thumbnail_tx.clone();
        let offset = self.db_offset;
        let _limit = self.db_limit; // reserved for future paging reuse
        let path = self.current_path.clone();
        tokio::spawn(async move {
            match crate::Thumbnail::get_all_thumbnails_from_directory(&path).await {
                Ok(rows) => {
                    for r in rows.iter() {
                        let _ = tx.try_send(r.clone());
                    }
                    // Send a synthetic zero-size row? Not needed; we will update state after join via channel? We'll rely on UI polling.
                    log::info!("[DB] Loaded page offset={} count={}", offset, rows.len());
                }
                Err(e) => {
                    log::error!("DB page load failed: {e}");
                }
            }
        });
    }

    // Load and display all rows from the entire database (no directory filter)
    pub fn load_all_database_rows(&mut self) {
        if self.db_loading {
            return;
        }
        // Reset similarity state and table caches
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.db_loading = true;
        self.db_offset = 0;
        self.db_all_view = true;
    self.table.clear();
    self.table_index.clear();
        self.thumb_scheduled.clear();
        self.pending_thumb_rows.clear();
        self.viewer.clip_presence.clear();
        self.viewer.clip_presence_hashes.clear();

        let tx = self.thumbnail_tx.clone();
        tokio::spawn(async move {
            match crate::Thumbnail::get_all_thumbnails().await {
                Ok(rows) => {
                    for r in rows.into_iter() {
                        let _ = tx.try_send(r);
                    }
                    log::info!("[DB] Loaded entire database");
                }
                Err(e) => log::error!("DB full load failed: {e}"),
            }
        });
    }

}