use crate::{database::DB, Thumbnail};
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
        let rows: Result<Vec<Thumbnail>, _> = DB.select("thumbnails").await;
        let mut loaded = 0usize;
        match rows {
            Ok(list) => {
                if list.is_empty() {
                    return 0;
                }
                // Preload clip embeddings (path -> embedding)
                let clip_map: std::collections::HashMap<String, (Vec<f32>, Option<String>)> = match crate::database::load_all_clip_embeddings().await {
                    Ok(rows) => rows.into_iter().map(|r| (r.path.clone(), (r.embedding.clone(), r.hash.clone()))).collect(),
                    Err(e) => { log::warn!("Failed loading clip_embeddings: {e}"); std::collections::HashMap::new() }
                };
                let mut files_guard = self.files.lock().await;
                for r in list.into_iter() {
                    // Attempt parse of modified timestamp
                    let modified_dt = r
                        .modified
                        .as_ref()
                        .and_then(|s| Some(s.with_timezone(&chrono::Local)));

                    let mut meta = super::FileMetadata {
                        id: None, // document id not restored (not needed for search mapping)
                        path: r.path.clone(),
                        filename: r.filename.clone(),
                        file_type: r.file_type.clone(),
                        size: r.size,
                        modified: modified_dt,
                        created: modified_dt,
                        // We persist only the base64 thumbnail (thumbnail_b64). Older rows may have stored
                        // a path in thumbnail_b64 erroneously if a previous bug existed; we detect a likely
                        // data URL by prefix. If it's not a data URL we keep it in thumbnail_path so later
                        // code can try to load & convert it.
                        thumbnail_path: r.thumbnail_b64.as_ref().and_then(|s| {
                            if s.starts_with("data:image") {
                                None
                            } else {
                                Some(s.clone())
                            }
                        }),
                        thumb_b64: r.thumbnail_b64.as_ref().and_then(|s| {
                            if s.starts_with("data:image") {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }),
                        hash: r.hash.clone(),
                        description: r.description.clone(),
                        caption: r.caption.clone(),
                        tags: r.tags.clone(),
                        category: r.category.clone(),
                        embedding: r.embedding.clone(),
                        similarity_score: None,
                        clip_embedding: None,
                        clip_similarity_score: None,
                    };
                    if let Some((emb, _hash)) = clip_map.get(&meta.path) { meta.clip_embedding = Some(emb.clone()); }
                    files_guard.push(meta);
                    loaded += 1;
                }
                log::info!("Loaded {} cached AI metadata rows", loaded);
            }
            Err(e) => {
                log::warn!("Failed to load cached AI metadata: {}", e);
            }
        }
        loaded
    }

}