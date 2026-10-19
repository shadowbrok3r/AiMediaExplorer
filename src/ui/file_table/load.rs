use crate::command::Scope;

impl super::FileExplorer {
    /// Empties the view for a virtual folder list that streams in through the thumbnail channel.
    fn begin_virtual_view(&mut self) {
        self.stop_work();
        self.viewer.mode = super::table::ExplorerMode::Database;
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.table.clear();
        self.viewer.thumbs.new_scope(self.viewer.scope);
        self.pending_thumb_rows.clear();
        self.viewer.clip_presence.clear();
        self.viewer.clip_presence_hashes.clear();
        self.nav.more = None;
    }

    /// Load a virtual "folders" view where each folder is a distinct category
    pub fn load_virtual_categories_view(&mut self) {
        self.begin_virtual_view();
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
    }

    /// Load a virtual "folders" view where each folder is a distinct tag
    pub fn load_virtual_tags_view(&mut self) {
        self.begin_virtual_view();
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
    }

    /// Shows the members of a collection, read page by page.
    pub fn load_logical_group_by_name(&mut self, name: String) {
        if let Err(reason) = self.open_scope(Scope::Collection(name)) {
            log::warn!("[scope] {reason}");
        }
    }

    /// Shows a collection in this tab.
    pub fn open_collection(&mut self, name: &str) {
        self.load_logical_group_by_name(name.to_string());
    }
}
