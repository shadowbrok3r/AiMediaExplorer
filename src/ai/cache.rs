use crate::database::DB;
use chrono::Utc;


impl super::AISearchEngine {
    // Cache thumbnail & AI metadata in surrealdb table `thumbnails` (id = path)
    pub async fn cache_thumbnail_and_metadata(
        &self,
        metadata: &super::FileMetadata,
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

        // Attempt to load existing row (by path) to merge instead of overwriting newer AI data.
        let existing: Option<crate::Thumbnail> = DB
            .query("SELECT * FROM type::table($table) WHERE path = $path LIMIT 1;")
            .bind(("table", crate::database::THUMBNAILS))
            .bind(("path", metadata.path.clone()))
            .await?
            .take(0)?;

        // Field-wise merge: prefer new non-empty AI fields; retain existing where new is None/empty.
        let mut merged = existing.unwrap_or_else(|| crate::Thumbnail {
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
        });

        // Only update fields if new content present
        if let Some(desc) = &metadata.description { if desc.trim().len() > 0 { merged.description = Some(desc.clone()); } }
        if let Some(caption) = &metadata.caption { if caption.trim().len() > 0 { merged.caption = Some(caption.clone()); } }
        if !metadata.tags.is_empty() { merged.tags = metadata.tags.clone(); }
        if let Some(cat) = &metadata.category { if cat.trim().len() > 0 { merged.category = Some(cat.clone()); } }
        if let Some(embed) = &metadata.embedding { if !embed.is_empty() { merged.embedding = Some(embed.clone()); } }
        if thumb_b64.is_some() { merged.thumbnail_b64 = thumb_b64; }
        if let Some(mod_dt) = metadata.modified { merged.modified = Some(mod_dt.to_utc().into()); }
        if let Some(h) = &metadata.hash { if !h.is_empty() { merged.hash = Some(h.clone()); } }

        // Upsert strategy: use an UPDATE with WHERE path match; if none updated, INSERT new.
        let updated: Option<crate::Thumbnail> = DB
            .query("UPDATE thumbnails SET filename = $filename, file_type = $file_type, size = $size, description = $description, caption = $caption, tags = $tags, category = $category, embedding = $embedding, thumbnail_b64 = $thumbnail_b64, modified = $modified, hash = $hash WHERE path = $path")
            .bind(("table", crate::database::THUMBNAILS))
            .bind(("filename", merged.filename.clone()))
            .bind(("file_type", merged.file_type.clone()))
            .bind(("size", merged.size))
            .bind(("description", merged.description.clone()))
            .bind(("caption", merged.caption.clone()))
            .bind(("tags", merged.tags.clone()))
            .bind(("category", merged.category.clone()))
            .bind(("embedding", merged.embedding.clone()))
            .bind(("thumbnail_b64", merged.thumbnail_b64.clone()))
            .bind(("modified", merged.modified.clone()))
            .bind(("hash", merged.hash.clone()))
            .bind(("path", merged.path.clone()))
            .await?
            .take(0)?;

        log::info!("Cached data Is Some: {:?}", updated.is_some());
        if updated.is_none() {
            let _: Option<crate::Thumbnail> = DB
                .create(crate::database::THUMBNAILS)
                .content(merged)
                .await?;
        }
        Ok(())
    }

    // Load previously cached thumbnail/meta rows from Surreal into memory so we don't
    // re-index (and especially don't re-run expensive vision description) every launch.
    // Returns number of records loaded.
    pub async fn load_cached(&self) -> usize {
        // Only pull the minimal data necessary to know which paths have any cached AI info.
        // This avoids deserializing large base64 thumbnails & embeddings at startup.
        let sql = "SELECT path FROM thumbnails"; // path has a UNIQUE index for fast scan
        let mut count = 0usize;
        match DB.query(sql).await {
            Ok(mut resp) => {
                // Surreal returns one result set; take it as Vec<serde_json::Value> then extract paths.
                let rows: Result<Vec<serde_json::Value>, _> = resp.take(0);
                match rows {
                    Ok(vals) => {
                        let mut set_guard = self.cached_paths.lock().await;
                        set_guard.clear();
                        for v in vals.into_iter() {
                            if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
                                set_guard.insert(p.to_string());
                                count += 1;
                            }
                        }
                        log::info!("[AI] cached path-only hydration: {} paths", count);
                    }
                    Err(e) => {
                        log::warn!("[AI] failed to parse cached path list: {e}");
                    }
                }
            }
            Err(e) => {
                log::warn!("[AI] failed path-only cache load: {e}");
            }
        }
        count
    }

}