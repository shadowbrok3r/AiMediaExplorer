use crate::{ScanEnvelope, Thumbnail, next_scan_id};
use std::{path::Path, path::PathBuf};
use jwalk::WalkDirGeneric;

impl super::FileExplorer {
    pub fn populate_current_directory(&mut self) {
        let path = PathBuf::from(&self.current_path);
        if !path.exists() {
            return;
        }
        self.table.clear();
        self.files.clear();
        let walker = WalkDirGeneric::<((), Option<u64>)>::new(&path)
            .parallelism(jwalk::Parallelism::RayonNewPool(16))
            .skip_hidden(false)
            .follow_links(false)
            .max_depth(1);
        let mut media_paths: Vec<PathBuf> = Vec::new();
        let excluded: Vec<String> = self
            .excluded_terms
            .iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();
        for entry in walker {
            if let Ok(e) = entry {
                let p = e.path();
                // Skip the root itself (depth 0) - jwalk root yields itself sometimes
                if p == path {
                    continue;
                }
                if e.file_type().is_dir() {
                    if let Some(row) = Self::directory_to_thumbnail(&p) {
                        self.table.push(row);
                    }
                    continue;
                }
                if e.file_type().is_file() {
                    if let Some(ext) = p
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                    {
                        if !crate::is_supported_media_ext(ext.as_str()) {
                            continue;
                        }
                        // Mark archives differently
                        if crate::utilities::types::is_archive(&ext) {
                            // Represent as a pseudo-directory row to allow future expansion (opening contents)
                            if let Some(mut row) = Self::file_to_minimal_thumbnail(&p) {
                                row.file_type = "<ARCHIVE>".into();
                                self.table.push(row);
                            }
                            continue; // don't schedule as media thumbnail
                        }
                    } else {
                        continue;
                    }
                    if !excluded.is_empty() {
                        let lower_path = p.to_string_lossy().to_ascii_lowercase();
                        if excluded.iter().any(|term| lower_path.contains(term)) {
                            continue;
                        }
                    }
                    // Reuse existing conversion helper if available
                    if let Some(row) = Self::file_to_minimal_thumbnail(&p) {
                        self.table.push(row);
                    }
                    media_paths.push(p.clone());
                }
            }
        }
        // Spawn shallow thumbnail generation for displayed media
        if !media_paths.is_empty() {
            let tx = self.scan_tx.clone();
            let scan_id = next_scan_id();
            std::thread::spawn(move || {
                for p in media_paths.into_iter() {
                    if let Some(ext) = p
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                    {
                        let is_img = crate::is_image(ext.as_str());
                        let is_vid = crate::is_video(ext.as_str());
                        if is_img {
                            if let Ok(thumb) =
                                crate::utilities::thumbs::generate_image_thumb_data(&p)
                            {
                                let _ = tx.try_send(ScanEnvelope {
                                    scan_id,
                                    msg: crate::utilities::scan::ScanMsg::UpdateThumb {
                                        path: p.clone(),
                                        thumb,
                                    },
                                });
                            }
                        } else if is_vid {
                            #[cfg(windows)]
                            if let Ok(thumb) =
                                crate::utilities::thumbs::generate_video_thumb_data(&p)
                            {
                                let _ = tx.try_send(ScanEnvelope {
                                    scan_id,
                                    msg: crate::utilities::scan::ScanMsg::UpdateThumb {
                                        path: p.clone(),
                                        thumb,
                                    },
                                });
                            }
                        }
                    }
                }
                let _ = tx.try_send(ScanEnvelope {
                    scan_id,
                    msg: crate::utilities::scan::ScanMsg::Done,
                });
            });
        }
        // After listing entries, fetch any existing AI metadata cached in DB and merge.
        // New per-directory minimal AI metadata hydration (path-only set + minimal fields).
        {
            let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
            let dir_file_paths: Vec<String> = self
                .table
                .iter()
                .filter(|t| t.file_type != "<DIR>")
                .map(|t| t.path.clone())
                .collect();
            if !dir_file_paths.is_empty() {
                tokio::spawn(async move {
                    let _ = engine.hydrate_directory_paths(&dir_file_paths).await; // count logged inside
                });
            }
        }
        // Legacy metadata fetch pathway retained for now (can be removed once UI fully reads from AI engine state if desired)
        self.fetch_directory_metadata();
    }

    fn directory_to_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        t.file_type = "<DIR>".into();
        Some(t)
    }

    fn file_to_minimal_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            t.file_type = ext.to_ascii_lowercase();
        }
        if let Ok(meta) = std::fs::metadata(p) {
            t.size = meta.len();
        }
        Some(t)
    }

    pub fn push_history(&mut self, new_path: String) {
        if self.current_path != new_path {
            self.back_stack.push(self.current_path.clone());
            self.forward_stack.clear();
            self.current_path = new_path;
        }
    }

    pub fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            self.forward_stack.push(self.current_path.clone());
            self.current_path = prev;
            self.populate_current_directory();
        }
    }

    pub fn nav_forward(&mut self) {
        if let Some(next) = self.forward_stack.pop() {
            self.back_stack.push(self.current_path.clone());
            self.current_path = next;
            self.populate_current_directory();
        }
    }

    pub fn nav_up(&mut self) {
        if let Some(parent) = Path::new(&self.current_path).parent() {
            let p = parent.to_string_lossy().to_string();
            self.push_history(p);
            self.populate_current_directory();
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    pub fn nav_home(&mut self) {
        if let Some(ud) = directories::UserDirs::new() {
            let home = ud.home_dir();
            let hp = home.to_string_lossy().to_string();
            self.push_history(hp);
            self.populate_current_directory();
            let _ = self.thumbnail_tx.try_send(self.current_thumb.clone());
        }
    }

    pub fn refresh(&mut self) {
        self.populate_current_directory();
    }

    pub fn set_path(&mut self, path: impl Into<String>) {
        self.current_path = path.into();
        self.populate_current_directory();
    }

    fn fetch_directory_metadata(&self) {
        // This leverages the UNIQUE index on thumbnails.path.
        let paths: Vec<String> = self.table.iter().map(|t| t.path.clone()).collect();
        if paths.is_empty() {
            return;
        }
        let tx = self.meta_tx.clone();
        // Chunk to keep query parameter size reasonable
        const CHUNK: usize = 400;
        for chunk in paths.chunks(CHUNK) {
            let chunk_vec: Vec<String> = chunk.iter().cloned().collect();
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                match crate::Thumbnail::find_thumbs_from_paths(chunk_vec).await {
                    Ok(rows) => {
                        for row in rows.into_iter() {
                            let _ = tx_clone.try_send(super::AIMetadataUpdate {
                                path: row.path.clone(),
                                description: row.description.clone(),
                                caption: row.caption.clone(),
                                category: row.category.clone(),
                                tags: row.tags.clone(),
                            });
                        }
                    }
                    Err(e) => {
                        log::warn!("Primary directory metadata query failed: {e}");
                        return;
                    }
                };
            });
        }
    }
}