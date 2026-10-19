use std::time::Instant;


use crate::{LogicalGroup, DB};
use crate::database::{db_activity, db_set_progress, db_set_detail, db_set_error};
use crate::database::thumb_blobs::{self, ThumbBlobIn};
use surrealdb::types::{RecordId, SurrealValue};
use chrono::Utc;

pub(super) const COLS_META: &str = "id, db_created, path, filename, file_type, size, description, caption, tags, category, modified, hash, parent_dir, logical_group";
pub(super) const COLS_WITH_INLINE: &str = "id, db_created, path, filename, file_type, size, description, caption, tags, category, thumbnail_b64, modified, hash, parent_dir, logical_group";

/// Row columns for list queries; the inline base64 column is selected only while legacy rows may still hold it.
pub(crate) fn row_cols() -> &'static str {
    if super::thumb_migration::inline_may_exist() { COLS_WITH_INLINE } else { COLS_META }
}

/// Replaces the tag `$old` with `$new` on every row that has it.
pub(crate) const RENAME_TAG_SQL: &str = "UPDATE thumbnails SET tags = array::distinct(array::union(array::complement(tags, [$old]), [$new])) WHERE tags CONTAINS $old";

/// Removes the tag `$tag` from every row that has it.
pub(crate) const DELETE_TAG_SQL: &str = "UPDATE thumbnails SET tags = array::complement(tags, [$tag]) WHERE tags CONTAINS $tag";

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

    /// Fetch a single thumbnail row by exact path (UNIQUE index); `thumbnail_b64` is filled from the blob table when the row has no inline copy.
    pub async fn get_thumbnail_by_path(
        path: &str,
    ) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let mut row = Self::get_thumbnail_row_by_path(path).await?;
        if let Some(r) = row.as_mut() {
            r.hydrate_inline().await;
        }
        Ok(row)
    }

    /// Row lookup by path without blob hydration.
    pub async fn get_thumbnail_row_by_path(
        path: &str,
    ) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let _ga = db_activity(format!("Load thumbnail by path"));
        db_set_detail(format!("Loading thumbnail by path"));
        let resp: Option<Self> = DB
            .query(format!("SELECT {} FROM thumbnails WHERE path = $path", row_cols()))
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        Ok(resp)
    }

    /// Fill `thumbnail_b64` from the blob table when the row carries no inline copy.
    async fn hydrate_inline(&mut self) {
        self.hydrate_inline_with(&DB).await;
    }

    pub(super) async fn hydrate_inline_with<C: surrealdb::Connection>(&mut self, db: &surrealdb::Surreal<C>) {
        if self.thumbnail_b64.as_deref().is_some_and(|s| !s.is_empty()) {
            return;
        }
        let key = thumb_blobs::key_for_path(&self.path);
        match thumb_blobs::get_many_with(db, std::slice::from_ref(&key)).await {
            Ok(mut m) => {
                if let Some(blob) = m.remove(&key) {
                    self.thumbnail_b64 = Some(thumb_blobs::blob_to_thumb_string(&blob));
                }
            }
            Err(e) => log::warn!("[thumb] blob lookup failed for {}: {e}", self.path),
        }
    }

    #[cfg(test)]
    pub(super) async fn legacy_thumbnail_b64_with<C: surrealdb::Connection>(db: &surrealdb::Surreal<C>, path: &str) -> anyhow::Result<Option<String>, anyhow::Error> {
        let found: Option<String> = db
            .query("SELECT VALUE thumbnail_b64 FROM thumbnails WHERE path = $path AND thumbnail_b64 != NONE")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        Ok(found.filter(|s| !s.is_empty()))
    }

    #[cfg(test)]
    pub(super) async fn legacy_thumbnail_b64_many_with<C: surrealdb::Connection>(db: &surrealdb::Surreal<C>, paths: &[String]) -> anyhow::Result<std::collections::HashMap<String, String>, anyhow::Error> {
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { path: String, thumbnail_b64: Option<String> }
        let mut out = std::collections::HashMap::new();
        for chunk in paths.chunks(64) {
            let rows: Vec<Row> = db
                .query("SELECT path, thumbnail_b64 FROM thumbnails WHERE path IN $paths AND thumbnail_b64 != NONE")
                .bind(("paths", chunk.to_vec()))
                .await?
                .take(0)?;
            out.extend(rows.into_iter().filter_map(|r| r.thumbnail_b64.filter(|s| !s.is_empty()).map(|b| (r.path, b))));
        }
        Ok(out)
    }

    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_all_thumbnails_from_directory(
        path: &str,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let now = Instant::now();
    let _ga = db_activity("Load thumbnails by parent directory");
    db_set_detail("Loading thumbnails (by directory)".to_string());
        let resp: Vec<Self> = DB
            .query(format!("SELECT {} FROM thumbnails WITH INDEX idx_parent_dir WHERE parent_dir = $parent_directory", row_cols()))
            .bind(("parent_directory", path.to_string()))
            .await?
            .take(0)?;
        log::warn!("get_all_thumbnails_from_directory is empty: {:?}\nTime for query: {:?}", resp.is_empty(), now.elapsed().as_secs_f32());
        Ok(resp)
    }

    pub async fn get_by_id(id: &RecordId) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let _ga = db_activity("Load thumbnail by id");
        db_set_detail("Loading thumbnail by id".to_string());
        let mut resp: Option<Self> = DB.select(id.clone()).await?;
        if let Some(r) = resp.as_mut() {
            r.hydrate_inline().await;
        }
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
        .query(format!("SELECT {} FROM thumbnails WHERE array::find($paths, path) != NONE", row_cols()))
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

    /// Rename/merge a category across all rows, returns number of updated rows; the change is journaled.
    pub async fn rename_category(old: &str, new: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Rename category");
        db_set_detail(format!("{old} -> {new}"));
        super::journal::rename_category_with(&DB, old, new).await
    }

    /// Rename/merge a tag across all rows, returns number of updated rows; the change is journaled.
    pub async fn rename_tag(old: &str, new: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Rename tag");
        db_set_detail(format!("{old} -> {new}"));
        super::journal::rename_tag_with(&DB, old, new).await
    }

    /// Delete a tag across all rows; returns number of updated rows; the change is journaled.
    pub async fn delete_tag(tag: &str) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Delete tag");
        db_set_detail(format!("{tag}"));
        super::journal::delete_tag_with(&DB, tag).await
    }

    /// Limit the number of tags per row (truncate beyond limit). Returns number of updated rows; the change is journaled.
    pub async fn prune_tags(limit: i64) -> anyhow::Result<u64, anyhow::Error> {
        let _ga = db_activity("Prune tags per item");
        db_set_detail(format!("limit = {limit}"));
        super::journal::prune_tags_with(&DB, limit).await
    }

    /// Move the inline thumbnail string into a blob entry, leaving the row metadata-only.
    fn take_inline_blob(&mut self) -> Option<ThumbBlobIn> {
        let inline = self.thumbnail_b64.take().filter(|s| !s.trim().is_empty())?;
        match ThumbBlobIn::from_thumb_string(self.path.clone(), &inline) {
            Ok(b) => Some(b.with_source(self.modified.clone().map(Into::into), Some(self.size))),
            Err(e) => {
                log::warn!("[thumb] dropping unusable thumbnail for {}: {e}", self.path);
                None
            }
        }
    }

    pub async fn update_or_create_thumbnail(
        self,
        metadata: &super::Thumbnail,
        thumb_b64: Option<String>,
    ) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Upsert thumbnail by path");
        db_set_detail("Saving thumbnail".to_string());
        self.upsert_with(&DB, metadata, thumb_b64).await
    }

    /// Upsert by path on `db`: metadata goes to the row, a supplied thumbnail goes to the blob table.
    pub(super) async fn upsert_with<C: surrealdb::Connection>(
        mut self,
        db: &surrealdb::Surreal<C>,
        metadata: &super::Thumbnail,
        thumb_b64: Option<String>,
    ) -> anyhow::Result<(), anyhow::Error> {
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
        if metadata.modified.is_some() {
            self.modified = metadata.modified.clone();
        }
        if let Some(h) = &metadata.hash {
            if !h.is_empty() {
                self.hash = Some(h.clone());
            }
        }
        self.thumbnail_b64 = thumb_b64;
        let blob_stored = match self.take_inline_blob() {
            Some(blob) => {
                thumb_blobs::put_with(db, vec![blob]).await.map_err(|e| {
                    db_set_error(format!("Thumbnail blob save failed: {e}"));
                    e
                })?;
                true
            }
            None => false,
        };

        // Upsert strategy: use an UPDATE with WHERE path match; if none updated, INSERT new.
        let clear_inline = if blob_stored { ", thumbnail_b64 = NONE" } else { "" };
        let updated: Option<RecordId> = db
            .query(format!(
                r#"
                    UPDATE thumbnails 
                        SET filename = $filename, 
                        file_type = $file_type, 
                        size = $size, 
                        description = $description, 
                        caption = $caption, 
                        tags = $tags, 
                        category = $category, 
                        modified = $modified, 
                        hash = $hash{clear_inline} 
                    WHERE path = $path
                    RETURN VALUE id
                    "#
            ))
            .bind(("filename", self.filename.clone()))
            .bind(("file_type", self.file_type.clone()))
            .bind(("size", self.size))
            .bind(("description", self.description.clone()))
            .bind(("caption", self.caption.clone()))
            .bind(("tags", self.tags.clone()))
            .bind(("category", self.category.clone()))
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
            match db
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
    mut thumbs: Vec<super::Thumbnail>,
) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _ga = db_activity("Insert thumbnail batch");
    db_set_detail(format!("Saving {} thumbnails", thumbs.len()));
    db_set_progress(0, thumbs.len() as u64);
    let blobs: Vec<ThumbBlobIn> = thumbs.iter_mut().filter_map(|t| t.take_inline_blob()).collect();
    thumb_blobs::put(blobs).await?;
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

// Upsert a list of thumbnail rows (by path) and return their RecordIds
pub async fn upsert_rows_and_get_ids(rows: Vec<super::Thumbnail>) -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
    let total = rows.len() as u64;
    let _ga = db_activity(format!("Upsert {} thumbnails", total));
    db_set_detail(format!("Saving thumbnails (0/{total})"));
    db_set_progress(0, total);
    let mut ids: Vec<RecordId> = Vec::new();
    for (i, meta) in rows.into_iter().enumerate() {
        db_set_detail(format!("Saving thumbnails ({} / {total})", i+1));
        let base = crate::Thumbnail::get_thumbnail_row_by_path(&meta.path)
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


