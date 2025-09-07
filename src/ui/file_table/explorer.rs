use crate::{next_scan_id, ui::status::GlobalStatusIndicator, Thumbnail, DB};
use std::{path::{Path, PathBuf}, time::Duration};

impl super::FileExplorer {
    pub fn populate_current_directory(&mut self) {
        let path = PathBuf::from(&self.current_path);
        if !path.exists() {
            return;
        }
        // Reset similarity view when changing directory
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.table.clear();
        self.files.clear();
        let excluded: Vec<String> = self
            .excluded_terms
            .iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();
        // Build scan parameters up-front
        let mut filters = crate::utilities::types::Filters::default();
        filters.root = path.clone();
        filters.include_images = true;
        filters.include_videos = true;
        filters.excluded_terms = excluded;
        let tx = self.scan_tx.clone();
        let scan_id = next_scan_id();
        self.current_scan_id = Some(scan_id);
        // Preload DB rows first, then start the scan so IDs are available to merge
        let prefix = self.current_path.clone();
        let preload_tx = self.db_preload_tx.clone();
        tokio::spawn(async move {
            crate::utilities::scan::spawn_scan(filters, tx, false, scan_id).await;
            match crate::Thumbnail::get_all_thumbnails_from_directory(prefix.as_str()).await {
                Ok(rows) => { let _ = preload_tx.try_send(rows); },
                Err(e) => {
                    log::error!("DB preload failed: {e}");
                    if e.to_string().contains("uninitialised") {
                        crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Database still initializing..");
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        DB.wait_for(surrealdb::opt::WaitFor::Connection).await;
                        match crate::Thumbnail::get_all_thumbnails_from_directory(prefix.as_str()).await {
                            Ok(rows) => { let _ = preload_tx.try_send(rows); },
                            Err(e) => log::error!("DB preload still failed: {e:?}")
                        }
                    }
                },
            }
        });
    }

    pub(super) fn directory_to_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        t.file_type = "<DIR>".into();
        Some(t)
    }

    pub fn push_history(&mut self, new_path: String) {
        if self.current_path != new_path {
            // save to settings recent paths
            let mut s = crate::database::settings::load_settings();
            s.push_recent_path(new_path.clone());
            crate::database::settings::save_settings(&s);
            self.back_stack.push(self.current_path.clone());
            self.forward_stack.clear();
            self.current_path = new_path;
        }
    }

    pub fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            self.forward_stack.push(self.current_path.clone());
            self.current_path = prev;
            if self.viewer.mode == super::viewer::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else {
                self.populate_current_directory();
            }
        }
    }

    pub fn nav_forward(&mut self) {
        if let Some(next) = self.forward_stack.pop() {
            self.back_stack.push(self.current_path.clone());
            self.current_path = next;
            if self.viewer.mode == super::viewer::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else {
                self.populate_current_directory();
            }
        }
    }

    pub fn nav_up(&mut self) {
        if let Some(parent) = Path::new(&self.current_path).parent() {
            let p = parent.to_string_lossy().to_string();
            self.push_history(p);
            if self.viewer.mode == super::viewer::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else {
                self.populate_current_directory();
            }
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    pub fn nav_home(&mut self) {
        if let Some(ud) = directories::UserDirs::new() {
            let home = ud.home_dir();
            let hp = home.to_string_lossy().to_string();
            self.push_history(hp);
            if self.viewer.mode == super::viewer::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else {
                self.populate_current_directory();
            }
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    pub fn refresh(&mut self) {
        if self.viewer.mode == super::viewer::ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }

    pub fn set_path(&mut self, path: impl Into<String>) {
        self.current_path = path.into();
        if self.viewer.mode == super::viewer::ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }
}