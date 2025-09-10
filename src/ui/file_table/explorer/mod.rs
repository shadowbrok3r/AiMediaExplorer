use crate::{next_scan_id, ui::status::GlobalStatusIndicator, DB};
use crate::utilities::archive::{ArchiveRegistry, entry_to_thumbnail};
use std::{path::PathBuf, time::Duration};
pub mod navigation;

impl crate::ui::file_table::FileExplorer {
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