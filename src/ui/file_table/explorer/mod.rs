use crate::{next_scan_id, ui::status::GlobalStatusIndicator, DB};
use crate::utilities::archive::{ArchiveRegistry, entry_to_thumbnail};
use std::{path::PathBuf, time::Duration};

pub mod navigation;
pub mod backup;

impl crate::ui::file_table::FileExplorer {
    pub fn populate_current_directory(&mut self) {
        // Support virtual archive navigation via scheme://<fs>!/<internal>
        if let Some((scheme, archive_path, internal)) = parse_any_virtual_path(&self.current_path) {
            self.viewer.showing_similarity = false;
            self.viewer.similar_scores.clear();
                self.table.clear();
                self.table_index.clear();
            self.files.clear();

            let reg = ArchiveRegistry::default();
            if let Some(handler) = reg.by_scheme(&scheme) {
                // Use cached password if present for this archive path
                let pw_opt = self.viewer.archive_passwords.get(&archive_path).map(|s| s.as_str());
                let params = crate::utilities::archive::ListParams {
                    archive_fs_path: &archive_path,
                    internal_path: &internal,
                    password: pw_opt,
                };
                match handler.list(&params) {
                    Ok(entries) => {
                        for e in entries {
                            let t = entry_to_thumbnail(&scheme, &archive_path, &internal, e.clone());
                                self.table.push(t.clone());
                                let idx = self.table.len()-1;
                                self.table_index.insert(t.path.clone(), idx);
                            
                            // For media files, schedule asynchronous thumbnail generation
                            let ext = std::path::Path::new(&e.name)
                                .extension()
                                .and_then(|x| x.to_str())
                                .map(|s| s.to_ascii_lowercase())
                                .unwrap_or_default();
                            if !e.is_dir && (crate::utilities::types::is_image(&ext) || crate::utilities::types::is_video(&ext)) {
                                let scheme_clone = scheme.clone();
                                let archive_path_clone = archive_path.clone();
                                let internal_clone = internal.clone();
                                let filename_clone = e.name.clone();
                                let password_clone = pw_opt.map(|s| s.to_string());
                                let thumbnail_tx = self.thumbnail_tx.clone();
                                
                                // Generate virtual path for this file to identify the thumbnail
                                let vpath = format!("{}://{}!/{}/{}",
                                    scheme,
                                    archive_path,
                                    normalize_prefix(&internal).trim_end_matches('/'),
                                    e.name
                                ).replace("//!", "/!").replace("!//", "!/");
                                
                                tokio::spawn(async move {
                                    log::info!("Starting thumbnail generation for archive file: {}", vpath);
                                    match crate::utilities::archive::extract_and_generate_thumbnail(
                                        &scheme_clone, &archive_path_clone, &internal_clone, &filename_clone, 
                                        password_clone.as_deref()
                                    ) {
                                        Ok(Some(thumb_b64)) => {
                                            log::info!("Generated thumbnail for archive file: {}", vpath);
                                            let mut thumb = crate::Thumbnail::default();
                                            thumb.path = vpath.clone();
                                            thumb.thumbnail_b64 = Some(thumb_b64);
                                            match thumbnail_tx.try_send(thumb) {
                                                Ok(_) => log::info!("Sent thumbnail update for: {}", vpath),
                                                Err(e) => log::error!("Failed to send thumbnail for {}: {:?}", vpath, e),
                                            }
                                        }
                                        Ok(None) => {
                                            log::warn!("No thumbnail generated for archive file: {}", vpath);
                                        }
                                        Err(err) => {
                                            let es = err.to_string();
                                            if es.contains("PasswordRequired") {
                                                log::warn!("Password required for archive: {}", archive_path_clone);
                                                // Send a control Thumbnail row with file_type marker to trigger UI modal
                                                let mut t = crate::Thumbnail::default();
                                                t.path = archive_path_clone.clone();
                                                t.file_type = "<PASSWORD_REQUIRED>".to_string();
                                                let _ = thumbnail_tx.try_send(t);
                                            } else {
                                                log::warn!("Failed to generate thumbnail for archive file: {}: {}", vpath, es);
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed listing {} archive '{}': {e:?}", scheme, archive_path);
                        // If likely a password issue or any error for 7z/zip, enqueue modal prompt
                        if e.to_string().contains("PasswordRequired") || scheme == "7z" || scheme == "zip" {
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
            self.table_index.clear();
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
}

// Helper function to normalize internal path to a directory prefix
fn normalize_prefix(internal: &str) -> String {
    let t = internal.trim_matches('/');
    if t.is_empty() { String::new() } else { format!("{}/", t) }
}

// Parse any supported virtual path: <scheme>://<archive>!/<internal>
pub fn parse_any_virtual_path(v: &str) -> Option<(String,String,String)> {
    if let Some((scheme, rest)) = v.split_once("://") {
        if let Some((archive, internal)) = rest.split_once('!') {
            return Some((scheme.to_string(), archive.trim_matches('/').to_string(), internal.trim_start_matches('/').to_string()));
        }
    }
    None
}