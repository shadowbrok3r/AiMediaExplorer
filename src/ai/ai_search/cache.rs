use crate::get_thumbnail_paths;
use chrono::Utc;

impl crate::ai::AISearchEngine {
    // Cache thumbnail & AI metadata in surrealdb table `thumbnails` (id = path)
    pub async fn cache_thumbnail_and_metadata(
        &self,
        metadata: &crate::Thumbnail,
    ) -> Result<(), anyhow::Error> {
        // Prefer in-memory `thumb_b64`; else use already-persisted `thumbnail_b64`.
        let thumb_b64 = if let Some(b64) = &metadata.thumb_b64 {
            Some(b64.clone().trim().to_string())
        } else {
            metadata.thumbnail_b64.clone()
        };

        crate::Thumbnail::get_thumbnail_by_path(&metadata.path)
            .await?
            .unwrap_or_else(|| crate::Thumbnail {
                id: None,
                db_created: Some(Utc::now().into()),
                path: metadata.path.clone(),
                filename: metadata.filename.clone(),
                file_type: metadata.file_type.clone(),
                size: metadata.size,
                description: None,
                caption: None,
                tags: Vec::new(),
                category: None,
                thumbnail_b64: None,
                modified: metadata.modified.clone(),
                hash: metadata.hash.clone(),
                thumb_b64: None,
                similarity_score: None,
                clip_embedding: None,
                clip_similarity_score: None,
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
                // Replace the entire cached set in one lock scope to avoid clearing each iteration.
                let mut set_guard = self.cached_paths.lock().await;
                set_guard.clear();
                for val in vals.iter() {
                    set_guard.insert(val.clone());
                }
                count = set_guard.len();
            }
            Err(e) => {
                log::warn!("[AI] failed path-only cache load: {e}");
            }
        }
        count
    }
}
