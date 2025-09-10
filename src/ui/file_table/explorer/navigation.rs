use crate::Thumbnail;
use std::path::Path;

impl crate::ui::file_table::FileExplorer {
    pub fn directory_to_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        t.file_type = "<DIR>".into();
        Some(t)
    }

    pub fn push_history(&mut self, new_path: String) {
        if self.current_path != new_path {
            // save to settings recent paths (only if settings are loaded from DB)
            if let Some(mut s) = crate::database::settings::load_settings() {
                s.push_recent_path(new_path.clone());
                crate::database::settings::save_settings(&s);
            }
            self.back_stack.push(self.current_path.clone());
            self.forward_stack.clear();
            self.current_path = new_path;
        }
    }

    pub fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            self.forward_stack.push(self.current_path.clone());
            self.current_path = prev;
            if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
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
            if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else {
                self.populate_current_directory();
            }
        }
    }

    pub fn nav_up(&mut self) {
        // Generic handling for any supported virtual scheme path
        if let Some((scheme, archive_path, internal)) = super::parse_any_virtual_path(&self.current_path) {
            let trimmed = internal.trim_matches('/');
            if trimmed.is_empty() {
                // At root of archive: go up to the archive's parent directory in filesystem
                if let Some(p) = std::path::Path::new(&archive_path).parent().map(|p| p.to_string_lossy().to_string()) {
                    self.push_history(p);
                    self.populate_current_directory();
                }
            } else {
                let mut parts: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
                parts.pop();
                let new_internal = parts.join("/");
                let new_path = if new_internal.is_empty() {
                    format!("{}://{}!/", scheme, archive_path)
                } else {
                    format!("{}://{}!/{}/", scheme, archive_path, new_internal)
                };
                self.push_history(new_path);
                self.populate_current_directory();
            }
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
            return;
        }
        if let Some(parent) = Path::new(&self.current_path).parent() {
            let p = parent.to_string_lossy().to_string();
            self.push_history(p);
            if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
                self.db_offset = 0;
                self.db_last_batch_len = 0;
                self.load_database_rows();
            } else { self.populate_current_directory(); }
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    pub fn nav_home(&mut self) {
        if let Some(ud) = directories::UserDirs::new() {
            let home = ud.home_dir();
            let hp = home.to_string_lossy().to_string();
            self.push_history(hp);
            if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
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
        if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }

    pub fn set_path(&mut self, path: impl Into<String>) {
        let s: String = path.into();
        self.current_path = crate::utilities::windows::normalize_wsl_unc(&s);
        if self.viewer.mode == crate::ui::file_table::table::ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }
}