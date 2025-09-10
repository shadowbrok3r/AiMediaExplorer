use crate::{next_scan_id, ui::status::GlobalStatusIndicator, Thumbnail, DB};
use crate::utilities::archive::{ArchiveRegistry, entry_to_thumbnail};
use std::{path::{Path, PathBuf}, time::Duration};

impl super::FileExplorer {
    pub fn populate_current_directory(&mut self) {
        // Support virtual archive navigation via scheme://<fs>!/<internal>
        if let Some((scheme, archive_path, internal)) = parse_any_virtual_path(&self.current_path) {
            self.viewer.showing_similarity = false;
            self.viewer.similar_scores.clear();
            self.table.clear();
            self.files.clear();

            let reg = ArchiveRegistry::default();
            if let Some(handler) = reg.by_scheme(&scheme) {
                // Use cached password if present for this archive path
                let pw_opt = self.archive_passwords.get(&archive_path).map(|s| s.as_str());
                let params = crate::utilities::archive::ListParams {
                    archive_fs_path: &archive_path,
                    internal_path: &internal,
                    password: pw_opt,
                };
                match handler.list(&params) {
                    Ok(entries) => {
                        for e in entries {
                            let t = entry_to_thumbnail(&scheme, &archive_path, &internal, e);
                            self.table.push(t);
                        }
                    }
                    Err(e) => {
                        log::error!("Failed listing {} archive '{}': {e:?}", scheme, archive_path);
                        // If likely a password issue or any error for 7z/zip, enqueue modal prompt
                        if scheme == "7z" || scheme == "zip" {
                            if !self.pending_zip_passwords.contains(&archive_path) {
                                self.pending_zip_passwords.push_back(archive_path.clone());
                            }
                            self.active_zip_prompt = self.pending_zip_passwords.front().cloned();
                            self.show_zip_modal = true;
                        }
                    },
                }
                return;
            }
        }
        // Normalize WSL UNC variants on Windows
        let normalized = crate::utilities::windows::normalize_wsl_unc(&self.current_path);
        let path = PathBuf::from(&normalized);
        if !path.exists() { return; }
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
    filters.include_archives = true; // show .zip alongside media in shallow scans
        filters.excluded_terms = excluded;
    let tx = self.scan_tx.clone();
        let scan_id = next_scan_id();
        self.current_scan_id = Some(scan_id);
        // Preload DB rows first, then start the scan so IDs are available to merge
        let prefix = self.current_path.clone();
        let preload_tx = self.db_preload_tx.clone();
        // Track this shallow scan as owned by this explorer instance
        self.owning_scan_id = Some(scan_id);
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

    // ZIP/TAR/7z specific functions replaced by unified registry listing

    // Tar bespoke listing removed (now via ArchiveRegistry)

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
        // Generic handling for any supported virtual scheme path
        if let Some((scheme, archive_path, internal)) = parse_any_virtual_path(&self.current_path) {
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
            if self.viewer.mode == super::viewer::ExplorerMode::Database {
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
        let s: String = path.into();
        self.current_path = crate::utilities::windows::normalize_wsl_unc(&s);
        if self.viewer.mode == super::viewer::ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }
}

// 7z bespoke helpers removed (now via ArchiveRegistry)

// Parse any supported virtual path: <scheme>://<archive>!/<internal>
fn parse_any_virtual_path(v: &str) -> Option<(String,String,String)> {
    if let Some((scheme, rest)) = v.split_once("://") {
        if let Some((archive, internal)) = rest.split_once('!') {
            return Some((scheme.to_string(), archive.trim_matches('/').to_string(), internal.trim_start_matches('/').to_string()));
        }
    }
    None
}

// end of impl super::FileExplorer in this file