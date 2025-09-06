use crate::get_thumbnail_paths;
use chrono::Utc;

impl crate::ai::AISearchEngine {
    // Cache thumbnail & AI metadata in surrealdb table `thumbnails` (id = path)
    pub async fn cache_thumbnail_and_metadata(
        &self,
        metadata: &crate::FileMetadata,
    ) -> Result<(), anyhow::Error> {
        // Prefer existing in-memory base64 thumbnail if present; else attempt to read from on-disk path.
        let thumb_b64 = if let Some(b64) = &metadata.thumb_b64 {
            Some(b64.clone().trim().to_string())
        } else if let Some(tp) = &metadata.thumbnail_path {
            if tp.starts_with("data:image") {
                Some(tp.clone())
            } else {
                use base64::Engine;
                match tokio::fs::read(tp).await {
                    Ok(bytes) => Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                    Err(_) => None,
                }
            }
        } else {
            None
        };

        crate::Thumbnail::get_thumbnail_by_path(&metadata.path)
            .await?
            .unwrap_or_else(|| crate::Thumbnail {
                id: None,
                db_created: Utc::now().into(),
                path: metadata.path.clone(),
                filename: metadata.filename.clone(),
                file_type: metadata.file_type.clone(),
                size: metadata.size,
                description: None,
                caption: None,
                tags: Vec::new(),
                category: None,
                embedding: None,
                thumbnail_b64: None,
                modified: metadata.modified.map(|dt| dt.to_utc().into()),
                hash: metadata.hash.clone(),
            })
            .update_or_create_thumbnail(metadata, thumb_b64)
            .await?;

        Ok(())
    }

    // Load previously cached thumbnail/meta rows from Surreal into memory so we don't
    // re-index (and especially don't re-run expensive vision description) every launch.
    // Returns number of records loaded.
    pub async fn load_cached(&self) -> usize {
        // Only pull the minimal data necessary to know which paths have any cached AI info.
        // This avoids deserializing large base64 thumbnails & embeddings at startup.
        let mut count = 0usize;
        match get_thumbnail_paths().await {
            Ok(vals) => {
                for val in vals.iter() {
                    let mut set_guard = self.cached_paths.lock().await;
                    set_guard.clear();
                    set_guard.insert(val.clone());
                    count += 1;
                }
            }
            Err(e) => {
                log::warn!("[AI] failed path-only cache load: {e}");
            }
        }
        count
    }
}
