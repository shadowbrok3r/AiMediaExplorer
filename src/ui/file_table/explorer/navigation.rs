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
    
    pub fn refresh_wsl_cache(&mut self) {
        self.cached_wsl_distros = Some(crate::utilities::explorer::list_wsl_distros());
        self.cached_physical_drives = Some(crate::utilities::explorer::list_physical_drives());
    }

    /// Initialize this explorer to open a specific path in a new tab, optionally performing a recursive scan.
    /// This encapsulates internal state and avoids external code touching private fields.
    pub fn init_open_path(&mut self, path: &str, recursive: bool) {
        self.current_path = path.to_string();
        // reset view state
        self.viewer.showing_similarity = false;
        self.viewer.similar_scores.clear();
        self.table.clear();
        if recursive {
            self.recursive_scan = true;
            self.scan_done = false;
            self.file_scan_progress = 0.0;
            // Reset previous scan snapshot
            self.last_scan_rows.clear();
            self.last_scan_paths.clear();
            self.last_scan_root = Some(self.current_path.clone());
            // Spawn recursive scan with current filters
            let scan_id = crate::next_scan_id();
            self.owning_scan_id = Some(scan_id);
            let tx = self.scan_tx.clone();
            let mut filters = crate::Filters::default();
            filters.root = std::path::PathBuf::from(self.current_path.clone());
            filters.min_size_bytes = self.viewer.ui_settings.db_min_size_bytes;
            filters.max_size_bytes = self.viewer.ui_settings.db_max_size_bytes;
            filters.include_images = self.viewer.types_show_images;
            filters.include_videos = self.viewer.types_show_videos;
            filters.skip_icons = self.viewer.ui_settings.filter_skip_icons;
            filters.excluded_terms = self.excluded_terms.clone();
            // Add excluded directories from UI settings (normalize + absolute for robust matching)
            if let Some(ref excluded_dirs) = self.viewer.ui_settings.excluded_dirs {
                let mut set: std::collections::BTreeSet<std::path::PathBuf> = std::collections::BTreeSet::new();
                for s in excluded_dirs.iter() {
                    let norm = crate::utilities::windows::normalize_wsl_unc(s);
                    let pb = std::path::PathBuf::from(norm);
                    let abs = std::path::absolute(&pb).unwrap_or(pb);
                    let _ = set.insert(abs);
                }
                filters.recursive_excluded_dirs = set;
            }
            // Add excluded extensions (lowercase, without dot) from UI settings
            if let Some(ref exts) = self.viewer.ui_settings.db_excluded_exts {
                filters.recursive_excluded_exts = exts.iter().map(|s| s.to_ascii_lowercase()).collect();
            }
            self.current_scan_id = Some(scan_id);
            tokio::spawn(async move { crate::spawn_scan(filters, tx, true, scan_id).await; });
        } else {
            // Shallow scan / directory populate
            self.populate_current_directory();
        }
    }
    
    pub fn set_rows(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
    }
    
    // Set the table rows from DB results and switch to Database viewing mode
    pub fn set_rows_from_db(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.viewer.mode = crate::ui::file_table::table::ExplorerMode::Database;
        self.table.clear();
        for r in rows.into_iter() { self.table.push(r); }
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