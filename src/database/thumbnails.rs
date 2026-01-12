use std::time::Instant;

use crate::{LogicalGroup, DB};
use crate::database::{db_activity, db_set_progress, db_set_detail, db_set_error};
use surrealdb::types::{RecordId, SurrealValue};
use chrono::Utc;
use crossbeam::channel::Sender as CrossbeamSender;

impl Default for crate::Thumbnail {
    fn default() -> Self {
        Self { 
            id: RecordId::new(super::THUMBNAILS, surrealdb_types::Uuid::new_v4().to_string()), 
            db_created: Default::default(), 
            path: Default::default(), 
            filename: Default::default(), 
            file_type: Default::default(), 
            size: Default::default(), 
            description: Default::default(), 
            caption: Default::default(), 
            tags: Default::default(), 
            category: Default::default(), 
            thumbnail_b64: Default::default(), 
            modified: Default::default(), 
            hash: Default::default(), 
            parent_dir: Default::default(), 
            logical_group: LogicalGroup::default().id
        }
    }
}

impl crate::Thumbnail {
    pub fn new(filename: &str) -> Self {
        // Previous implementation used the filename as the record key which caused ID collisions
        // whenever two files in different directories shared the same filename. That resulted in
        // SurrealDB errors like: "Database record `thumbnails:<name>` already exists" during
        // scanning auto-save, preventing additional rows from being written. We now always
        // generate a UUID-based id (same strategy as Default) and rely on the UNIQUE path index
        // plus update_or_create_thumbnail() logic to keep rows in sync. Existing rows with
        // filename-based IDs remain valid; new inserts will use UUIDs avoiding collisions.
        Self {
            id: RecordId::new(
                super::THUMBNAILS,
                surrealdb_types::Uuid::new_v4().to_string(),
            ),
            db_created: Default::default(),
            path: Default::default(),
            filename: filename.to_string(),
            file_type: Default::default(),
            size: Default::default(),
            description: Default::default(),
            caption: Default::default(),
            tags: Default::default(),
            category: Default::default(),
            thumbnail_b64: Default::default(),
            modified: Default::default(),
            hash: Default::default(),
            parent_dir: Default::default(),
            logical_group: crate::LogicalGroup::default().id,
        }
    }

    /// Fetch all thumbnail rows across the entire database in large batches.
    /// Uses SurrealDB pagination (LIMIT/START) to avoid huge single responses.
    pub async fn get_all_thumbnails() -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Load all thumbnails (batched)");
        const CHUNK: i64 = 5_000;
        let mut out: Vec<Self> = Vec::new();
        let mut start: i64 = 0;
        loop {
            db_set_detail(format!("Loading thumbnails ({}..{}]", start, start + CHUNK));
            let batch: Vec<Self> = DB
                .query("SELECT * FROM thumbnails LIMIT $limit START $start")
                .bind(("limit", CHUNK))
                .bind(("start", start))
                .await?
                .take(0)?;
            if batch.is_empty() {
                break;
            }
            let len = batch.len() as i64;
            out.extend(batch);
            start += len;
            if len < CHUNK { break; }
        }
        Ok(out)
    }

    /// Stream all thumbnails in batches to the provided channel. Sends an empty Vec at the end.
    pub async fn stream_all_thumbnails(tx: CrossbeamSender<Vec<Self>>) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Load all thumbnails (stream)");
        // Determine total rows for progress updates
        let total_rows: i64 = DB
            .query("SELECT VALUE count() FROM thumbnails")
            .await
            .and_then(|mut r| r.take::<Vec<i64>>(0))
            .ok()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(0);
        db_set_detail(format!("Loading all thumbnails (0/{total_rows})"));
        db_set_progress(0, total_rows as u64);
        const CHUNK: i64 = 5_000;
        let mut start: i64 = 0;
        loop {
            let batch: Vec<Self> = DB
                .query("SELECT * FROM thumbnails LIMIT $limit START $start")
                .bind(("limit", CHUNK))
                .bind(("start", start))
                .await?
                .take(0)?;
            let len = batch.len() as i64;
            if len == 0 {
                let _ = tx.try_send(Vec::new());
                break;
            }
            start += len;
            let _ = tx.try_send(batch);
            db_set_progress(start as u64, total_rows as u64);
            db_set_detail(format!("Loading all thumbnails ({} / {})", start, total_rows));
            if len < CHUNK { 
                let _ = tx.try_send(Vec::new());
                break; 
            }
        }
        Ok(())
    }

    /// Count all thumbnails across the entire database.
    pub async fn count_all_thumbnails() -> anyhow::Result<i64, anyhow::Error> {
        let _ga = db_activity("Count all thumbnails");
        db_set_detail("Counting all thumbnails".to_string());
        let count_vec: Vec<i64> = DB
            .query("SELECT VALUE count() FROM thumbnails")
            .await?
            .take(0)?;
        Ok(count_vec.first().copied().unwrap_or(0))
    }

    /// Fetch a single page of thumbnails across the entire database using LIMIT/START.
    pub async fn get_all_thumbnails_paged(limit: i64, start: i64) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Load all thumbnails (paged)");
        db_set_detail(format!("Loading all thumbnails ({}..{})", start, start + limit));
        let rows: Vec<Self> = DB
            .query("SELECT * FROM thumbnails LIMIT $limit START $start")
            .bind(("limit", limit))
            .bind(("start", start))
            .await?
            .take(0)?;
        Ok(rows)
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

    /// Count thumbnails in a given parent directory (exact match).
    pub async fn count_thumbnails_in_directory(path: &str) -> anyhow::Result<i64, anyhow::Error> {
        let _ga = db_activity("Count thumbnails by parent directory");
        db_set_detail("Counting thumbnails (by directory)".to_string());
        let count_vec: Vec<i64> = DB
            .query("SELECT VALUE count() FROM thumbnails WITH INDEX idx_parent_dir WHERE parent_dir = $parent_directory")
            .bind(("parent_directory", path.to_string()))
            .await?
            .take(0)?;
        Ok(count_vec.first().copied().unwrap_or(0))
    }

    /// Page through thumbnails in a parent directory using LIMIT/START for Database mode pagination.
    pub async fn get_thumbnails_from_directory_paged(
        path: &str,
        limit: i64,
        start: i64,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Load thumbnails by directory (paged)");
        db_set_detail(format!("Loading thumbnails ({}..{})", start, start + limit));
        let resp: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WITH INDEX idx_parent_dir WHERE parent_dir = $parent_directory LIMIT $limit START $start")
            .bind(("parent_directory", path.to_string()))
            .bind(("limit", limit))
            .bind(("start", start))
            .await?
            .take(0)?;
        Ok(resp)
    }

    pub async fn get_by_id(id: &RecordId) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let _ga = db_activity("Load thumbnail by id");
        db_set_detail("Loading thumbnail by id".to_string());
        let resp: Option<Self> = DB.select(id.clone()).await?;
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

    /// List distinct non-empty categories present across thumbnails.
    pub async fn list_distinct_categories() -> anyhow::Result<Vec<String>, anyhow::Error> {
        let _ga = db_activity("List distinct categories");
        db_set_detail("Loading unique categories".to_string());
        let cats: Vec<String> = DB
            .query("SELECT VALUE category FROM thumbnails WHERE category != NONE GROUP BY category")
            .await?
            .take(0)?;
        Ok(cats)
    }

    /// List distinct tag strings by flattening tags arrays across all thumbnails.
    pub async fn list_distinct_tags() -> anyhow::Result<Vec<String>, anyhow::Error> {
        let _ga = db_activity("List distinct tags");
        db_set_detail("Loading unique tags".to_string());
        let rows: Vec<Vec<String>> = DB
            .query("SELECT VALUE tags FROM thumbnails WHERE tags != NONE")
            .await?
            .take(0)?;
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for arr in rows.into_iter() {
            for t in arr.into_iter() {
                if !t.trim().is_empty() { set.insert(t); }
            }
        }
        Ok(set.into_iter().collect())
    }

    /// Fetch thumbnails that have an exact category match.
    pub async fn fetch_by_category(category: &str) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select thumbnails by category");
        db_set_detail(format!("Loading category '{category}'"));
        let rows: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WHERE category = $cat")
            .bind(("cat", category.to_string()))
            .await?
            .take(0)?;
        Ok(rows)
    }

    /// Fetch thumbnails that contain the given tag in their tags array.
    pub async fn fetch_by_tag(tag: &str) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select thumbnails by tag");
        db_set_detail(format!("Loading tag '{tag}'"));
        let rows: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WHERE array::contains(tags, $tag)")
            .bind(("tag", tag.to_string()))
            .await?
            .take(0)?;
        Ok(rows)
    }

    /// List categories with their counts.
    pub async fn list_category_counts() -> anyhow::Result<Vec<(String, i64)>, anyhow::Error> {
        #[derive(Debug, serde::Deserialize, SurrealValue)]
        struct Row { category: Option<String>, cnt: i64 }
        let _ga = db_activity("Count categories");
        db_set_detail("Counting categories".to_string());
        let rows: Vec<Row> = DB
            .query("SELECT category, count() AS cnt FROM thumbnails GROUP BY category")
            .await?
            .take(0)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.category.map(|c| (c, r.cnt)))
            .collect())
    }

    /// List tags with their counts (computed client-side).
    pub async fn list_tag_counts() -> anyhow::Result<Vec<(String, i64)>, anyhow::Error> {
        let _ga = db_activity("Count tags");
        db_set_detail("Counting tags".to_string());
        let rows: Vec<Vec<String>> = DB
            .query("SELECT VALUE tags FROM thumbnails WHERE tags != NONE")
            .await?
            .take(0)?;
        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for arr in rows.into_iter() {
            for t in arr.into_iter() {
                if t.trim().is_empty() { continue; }
                *map.entry(t).or_default() += 1;
            }
        }
        Ok(map.into_iter().collect())
    }

    /// Rename/merge a category across all rows, returns number of updated rows.
    pub async fn rename_category(old: &str, new: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Rename category");
        db_set_detail(format!("{old} -> {new}"));
        let updated: Vec<Self> = DB
            .query("UPDATE thumbnails SET category = $new WHERE category = $old")
            .bind(("old", old.to_string()))
            .bind(("new", new.to_string()))
            .await?
            .take(0)?;
        Ok(updated.len() as u64)
    }

    /// Rename/merge a tag across all rows, returns number of updated rows.
    pub async fn rename_tag(old: &str, new: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Rename tag");
        db_set_detail(format!("{old} -> {new}"));
        let updated: Vec<Self> = DB
            .query(
                r#"
                UPDATE thumbnails
                SET tags = array::distinct(array::union(array::remove(tags, $old), [$new]))
                WHERE array::contains(tags, $old)
                "#,
            )
            .bind(("old", old.to_string()))
            .bind(("new", new.to_string()))
            .await?
            .take(0)?;
        Ok(updated.len() as u64)
    }

    /// Delete a tag across all rows; returns number of updated rows.
    pub async fn delete_tag(tag: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Delete tag");
        db_set_detail(format!("{tag}"));
        let updated: Vec<Self> = DB
            .query(
                r#"
                UPDATE thumbnails
                SET tags = array::remove(tags, $tag)
                WHERE array::contains(tags, $tag)
                "#,
            )
            .bind(("tag", tag.to_string()))
            .await?
            .take(0)?;
        Ok(updated.len() as u64)
    }

    /// Limit the number of tags per row (truncate beyond limit). Returns number of updated rows.
    pub async fn prune_tags(limit: i64) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Prune tags per item");
        db_set_detail(format!("limit = {limit}"));
        let updated: Vec<Self> = DB
            .query(
                r#"
                UPDATE thumbnails
                SET tags = array::slice(tags, 0, $limit)
                WHERE array::len(tags) > $limit
                "#,
            )
            .bind(("limit", limit))
            .await?
            .take(0)?;
        Ok(updated.len() as u64)
    }

    // Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
    pub async fn save_thumbnail_row(self) -> anyhow::Result<(), anyhow::Error> {
        log::info!("SAVING: {:?}", self.path);
        let _ga = db_activity("Insert thumbnail row");
        db_set_detail("Inserting thumbnail".to_string());
        let _: Option<Self> = DB
            .create(super::THUMBNAILS)
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

        // If no existing row matched (nothing updated), INSERT a new record
        if updated.is_none() {
            // Insert new row. If another concurrent task already inserted a row with the same
            // (now UUID) id between our SELECT/UPDATE and this INSERT, we treat that as benign.
            match DB
                .create(super::THUMBNAILS)
                .content::<Self>(self.clone())
                .await
            {
                Ok(mut resp) => {
                    let _: Option<Self> = resp.take();
                }
                Err(e) => {
                    let es = e.to_string();
                    if es.contains("already exists") {
                        log::warn!("Thumbnail insert skipped (already exists) for path {}", self.path);
                        // Silent success: another task inserted it. We proceed.
                    } else {
                        db_set_error(format!("Thumbnail insert after update miss failed: {e}"));
                        return Err(e.into());
                    }
                }
            }
        }

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
        .insert(super::THUMBNAILS)
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

// Upsert a single thumbnail row (by path) and return its RecordId if present
pub async fn upsert_row_and_get_id(meta: super::Thumbnail) -> anyhow::Result<Option<RecordId>, anyhow::Error> {
    let _ga = db_activity("Upsert 1 thumbnail");
    db_set_detail("Saving thumbnail (1/1)".to_string());
    db_set_progress(0, 1);
    let base = crate::Thumbnail::get_thumbnail_by_path(&meta.path)
        .await
        .map_err(|e| {
            db_set_error(format!("Load existing thumb failed: {e}"));
            e
        })?
        .unwrap_or_else(|| crate::Thumbnail {
            id: crate::Thumbnail::new(&meta.filename).id,
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
            logical_group: crate::LogicalGroup::default().id,
        });
    base.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await
        .map_err(|e| { db_set_error(format!("Update/create thumb failed: {e}")); e })?;
    let id = crate::Thumbnail::get_thumbnail_id_by_path(&meta.path).await
        .map_err(|e| { db_set_error(format!("Fetch id after upsert failed: {e}")); e })?;
    db_set_progress(1, 1);
    Ok(id)
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
                id: crate::Thumbnail::new(&meta.filename).id,
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
                logical_group: crate::LogicalGroup::default().id,
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
    paths: Vec<RecordId>,
) -> anyhow::Result<(usize, usize), anyhow::Error> {
    if paths.is_empty() { return Ok((0, 0)); }
    let _ga = db_activity(format!("Delete {} rows from DB", paths.len()));
    db_set_detail("Deleting thumbnails and embeddings".to_string());
    let mut resp = DB
        .query(
            r#"
                LET $ids = (
                    SELECT VALUE id FROM thumbnails WHERE array::find($paths, path) != NONE
                );
                DELETE clip_embeddings WHERE array::find($paths, path) != NONE OR array::find($ids, thumb_ref) != NONE;
                DELETE thumbnails WHERE array::find($paths, path) != NONE;
            "#,
        )
        .bind(("paths", paths))
        .await
        .map_err(|e| { db_set_error(format!("DB delete query failed: {e}")); e })?;

    // Statement order: 0 LET, 1 LET, 2 DELETE embeddings, 3 DELETE thumbnails
    let deleted_embeddings: Vec<super::ClipEmbeddingRow> = resp
        .take(1)
        .map_err(|e| { db_set_error(format!("Read deleted embeddings failed: {e}")); e })?;
    let deleted_thumbs: Vec<super::Thumbnail> = resp
        .take(2)
        .map_err(|e| { db_set_error(format!("Read deleted thumbnails failed: {e}")); e })?;
    Ok((deleted_embeddings.len(), deleted_thumbs.len()))
}


