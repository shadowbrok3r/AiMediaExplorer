use std::time::Instant;

use crate::DB;
use crate::database::{db_activity, db_set_progress, db_set_detail, db_set_error};
use surrealdb::RecordId;
use chrono::Utc;


impl crate::Thumbnail {
    /// Fetch all thumbnail rows across the entire database.
    pub async fn get_all_thumbnails() -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Load all thumbnails");
        db_set_detail("Loading all thumbnails".to_string());
        let resp: Vec<Self> = DB
            .query("SELECT * FROM thumbnails")
            .await?
            .take(0)?;
        Ok(resp)
    }
    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_thumbnail_by_path(
        path: &str,
    ) -> anyhow::Result<Option<Self>, anyhow::Error> {
    let _ga = db_activity(format!("Load thumbnail by path"));
    db_set_detail(format!("Loading thumbnail by path"));
        let resp: Option<Self> = DB
            .query("SELECT * FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        log::warn!("SELECT * FROM thumbnails WHERE path = $path: {:?}", resp.is_some());
        Ok(resp)
    }

    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_all_thumbnails_from_directory(
        path: &str,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let now = Instant::now();
    let _ga = db_activity("Load thumbnails by parent directory");
    db_set_detail("Loading thumbnails (by directory)".to_string());
        let resp: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WITH INDEX idx_parent_dir WHERE parent_dir = $parent_directory")
            .bind(("parent_directory", path.to_string()))
            .await?
            .take(0)?;
        log::warn!("get_all_thumbnails_from_directory is empty: {:?}\nTime for query: {:?}", resp.is_empty(), now.elapsed().as_secs_f32());
        Ok(resp)
    }

    pub async fn get_thumbnail_id_by_path(
        path: &str,
    ) -> anyhow::Result<Option<RecordId>, anyhow::Error> {
        log::info!("Getting thumbnail for path: {path:?}");
    let _ga = db_activity("Lookup thumbnail id by path");
    db_set_detail("Looking up thumbnail id".to_string());
        let resp: Option<RecordId> = DB
            .query("SELECT VALUE id FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        log::info!("Thumbnail for path is Some(): {resp:?}");
        Ok(resp)
    }

    pub async fn get_embedding(self) -> anyhow::Result<super::ClipEmbeddingRow, anyhow::Error> {
        log::info!("Checking embedding with thumb_ref == {:?}", self.id.clone());
    let _ga = db_activity("Load clip embedding by thumb_ref");
    db_set_detail("Loading clip embedding for thumbnail".to_string());
        let embedding: Option<super::ClipEmbeddingRow> = DB
            .query("SELECT * FROM clip_embeddings WITH INDEX clip_thumb_ref_idx WHERE thumb_ref = $id")
            .bind(("id", self.id.clone()))
            .await?
            .take(0)?;

        log::info!("Embedding for image is Some: {:?}", embedding.is_some());
        Ok(embedding.unwrap_or_default())
    }

    pub async fn find_thumbs_from_paths(
        chunk_vec: Vec<String>,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
    let _ga = db_activity("Load thumbnails by path set");
    db_set_detail("Loading thumbnails by set".to_string());
        let thumbs: Vec<Self> = DB
        .query("SELECT id, db_created, path, filename, file_type, size, modified, hash, description, caption, tags, category FROM thumbnails WHERE array::find($paths, path) != NONE")
        .bind(("paths", chunk_vec))
        .await?
        .take(0)?;

        Ok(thumbs)
    }

    // Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
    pub async fn save_thumbnail_row(self) -> anyhow::Result<(), anyhow::Error> {
        log::info!("SAVING: {:?}", self.path);
        let _ga = db_activity("Insert thumbnail row");
        db_set_detail("Inserting thumbnail".to_string());
        let _: Option<Self> = DB
            .create("thumbnails")
            .content::<Self>(self)
            .await
            .map_err(|e| {
                db_set_error(format!("Insert thumbnail failed: {e}"));
                e
            })?
            .take();
        Ok(())
    }

    pub async fn update_or_create_thumbnail(
        mut self,
        metadata: &super::Thumbnail,
        thumb_b64: Option<String>,
    ) -> anyhow::Result<(), anyhow::Error> {
        // Only update fields if new content present
    let _ga = db_activity("Upsert thumbnail by path");
    db_set_detail("Saving thumbnail".to_string());
        if let Some(desc) = &metadata.description {
            if desc.trim().len() > 0 {
                self.description = Some(desc.clone());
            }
        }
        if let Some(caption) = &metadata.caption {
            if caption.trim().len() > 0 {
                self.caption = Some(caption.clone());
            }
        }
        if !metadata.tags.is_empty() {
            self.tags = metadata.tags.clone();
        }
        if let Some(cat) = &metadata.category {
            if cat.trim().len() > 0 {
                self.category = Some(cat.clone());
            }
        }
    // embedding field on thumbnails removed; embeddings live in clip_embeddings table
        if thumb_b64.is_some() {
            self.thumbnail_b64 = thumb_b64;
        }
        if metadata.modified.is_some() {
            self.modified = metadata.modified.clone();
        }
        if let Some(h) = &metadata.hash {
            if !h.is_empty() {
                self.hash = Some(h.clone());
            }
        }

        // Upsert strategy: use an UPDATE with WHERE path match; if none updated, INSERT new.
        let updated: Option<super::Thumbnail> = DB
            .query(
                r#"
                    UPDATE thumbnails 
                        SET filename = $filename, 
                        file_type = $file_type, 
                        size = $size, 
                        description = $description, 
                        caption = $caption, 
                        tags = $tags, 
                        category = $category, 
                        thumbnail_b64 = $thumbnail_b64, 
                        modified = $modified, 
                        hash = $hash 
                    WHERE path = $path
                    "#,
            )
            .bind(("table", crate::database::THUMBNAILS))
            .bind(("filename", self.filename.clone()))
            .bind(("file_type", self.file_type.clone()))
            .bind(("size", self.size))
            .bind(("description", self.description.clone()))
            .bind(("caption", self.caption.clone()))
            .bind(("tags", self.tags.clone()))
            .bind(("category", self.category.clone()))
            .bind(("thumbnail_b64", self.thumbnail_b64.clone()))
            .bind(("modified", self.modified.clone()))
            .bind(("hash", self.hash.clone()))
            .bind(("path", self.path.clone()))
            .await
            .map_err(|e| {
                db_set_error(format!("Thumbnail upsert failed: {e}"));
                e
            })?
            .take(0)?;

        log::info!("Cached data Is Some: {:?}", updated.is_some());
        Ok(())
    }

}

pub async fn save_thumbnail_batch(
    thumbs: Vec<super::Thumbnail>,
) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _ga = db_activity("Insert thumbnail batch");
    db_set_detail(format!("Saving {} thumbnails", thumbs.len()));
    db_set_progress(0, thumbs.len() as u64);
    let _: Vec<super::Thumbnail> = DB
        .insert("thumbnails")
        .content::<Vec<super::Thumbnail>>(thumbs)
        .await
        .map_err(|e| {
            db_set_error(format!("Batch insert failed: {e}"));
            e
        })?;
    db_set_progress(1, 1);
    Ok(())
}

pub async fn get_thumbnail_paths() -> anyhow::Result<Vec<String>, anyhow::Error> {
    let _ga = db_activity("SELECT VALUE path FROM thumbnails");
    let paths: Vec<String> = DB.query("SELECT VALUE path FROM thumbnails").await?.take(0)?;
    Ok(paths)
}

// Upsert a list of thumbnail rows (by path) and return their RecordIds
pub async fn upsert_rows_and_get_ids(rows: Vec<super::Thumbnail>) -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
    let total = rows.len() as u64;
    let _ga = db_activity(format!("Upsert {} thumbnails", total));
    db_set_detail(format!("Saving thumbnails (0/{total})"));
    db_set_progress(0, total);
    let mut ids: Vec<RecordId> = Vec::new();
    for (i, meta) in rows.into_iter().enumerate() {
        db_set_detail(format!("Saving thumbnails ({} / {total})", i+1));
        let base = crate::Thumbnail::get_thumbnail_by_path(&meta.path)
            .await
            .map_err(|e| {
                db_set_error(format!("Load existing thumb failed: {e}"));
                e
            })?
            .unwrap_or_else(|| crate::Thumbnail {
                id: None,
                db_created: Some(Utc::now().into()),
                path: meta.path.clone(),
                filename: meta.filename.clone(),
                file_type: meta.file_type.clone(),
                size: meta.size,
                description: None,
                caption: None,
                tags: Vec::new(),
                category: None,
                thumbnail_b64: None,
                modified: meta.modified.clone(),
                hash: meta.hash.clone(),
                parent_dir: meta.parent_dir.clone(),
                logical_group: None,
            });
        // Prefer any provided thumbnail data in meta
        base.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await
            .map_err(|e| { db_set_error(format!("Update/create thumb failed: {e}")); e })?;
        if let Some(id) = crate::Thumbnail::get_thumbnail_id_by_path(&meta.path).await
            .map_err(|e| { db_set_error(format!("Fetch id after upsert failed: {e}")); e })?
        {
            if !ids.iter().any(|x| x == &id) {
                ids.push(id);
            }
        }
        db_set_progress((i as u64) + 1, total);
    }
    Ok(ids)
}

/// Delete thumbnails and their associated CLIP embeddings for a given set of file paths.
/// This operation only affects the database (does not delete files on disk).
/// Returns (deleted_embeddings_count, deleted_thumbnails_count).
pub async fn delete_thumbnails_and_embeddings_by_paths(
    paths: Vec<String>,
) -> anyhow::Result<(usize, usize), anyhow::Error> {
    if paths.is_empty() { return Ok((0, 0)); }
    let _ga = db_activity(format!("Delete {} rows from DB", paths.len()));
    db_set_detail("Deleting thumbnails and embeddings".to_string());
    // Use a single multi-statement query: collect thumbnail ids, delete embeddings by thumb_ref or path, then delete thumbnails by path.
    let mut resp = DB
        .query(
            r#"
            LET paths = $paths;
            LET ids = (
                SELECT VALUE id FROM thumbnails WHERE array::find(paths, path) != NONE
            );
            DELETE clip_embeddings WHERE array::find(paths, path) != NONE OR array::find(ids, thumb_ref) != NONE;
            DELETE thumbnails WHERE array::find(paths, path) != NONE;
            "#,
        )
        .bind(("paths", paths))
        .await
        .map_err(|e| { db_set_error(format!("DB delete query failed: {e}")); e })?;

    // Statement order: 0 LET, 1 LET, 2 DELETE embeddings, 3 DELETE thumbnails
    let deleted_embeddings: Vec<super::ClipEmbeddingRow> = resp
        .take(2)
        .map_err(|e| { db_set_error(format!("Read deleted embeddings failed: {e}")); e })?;
    let deleted_thumbs: Vec<super::Thumbnail> = resp
        .take(3)
        .map_err(|e| { db_set_error(format!("Read deleted thumbnails failed: {e}")); e })?;
    Ok((deleted_embeddings.len(), deleted_thumbs.len()))
}


