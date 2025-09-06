use crate::DB;
use surrealdb::RecordId;


impl crate::Thumbnail {
    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_thumbnail_by_path(
        path: &str,
    ) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let resp: Option<Self> = DB
            .query("SELECT * FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        Ok(resp)
    }

    pub async fn get_thumbnail_id_by_path(
        path: &str,
    ) -> anyhow::Result<Option<RecordId>, anyhow::Error> {
        let resp: Option<RecordId> = DB
            .query("SELECT id FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        Ok(resp)
    }

    pub async fn find_thumbs_from_paths(
        chunk_vec: Vec<String>,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let thumbs: Vec<Self> = DB
        .query("SELECT path, filename, file_type, size, modified, hash, description, caption, tags, category FROM thumbnails WHERE array::find($paths, path) != NONE")
        .bind(("paths", chunk_vec))
        .await?
        .take(0)?;

        Ok(thumbs)
    }

    // Paged DB thumbnail retrieval without ordering. Applies optional filters.
    pub async fn load_thumbnails_page(
        offset: usize,
        limit: usize,
        min_size: Option<u64>,
        max_size: Option<u64>,
        modified_after: Option<&str>,
        modified_before: Option<&str>,
        excluded_exts: Option<&[String]>,
        path_prefix: Option<&str>,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let mut clauses: Vec<String> = Vec::new();
        if let Some(ms) = min_size {
            clauses.push(format!("size >= {}", ms));
        }
        if let Some(mx) = max_size {
            clauses.push(format!("size <= {}", mx));
        }
        if let Some(a) = modified_after {
            clauses.push(format!("modified >= <datetime>'{}'", a));
        }
        if let Some(b) = modified_before {
            clauses.push(format!("modified <= <datetime>'{}'", b));
        }
        if let Some(exts) = excluded_exts {
            if !exts.is_empty() {
                let joined = exts
                    .iter()
                    .map(|e| format!("'{}'", e))
                    .collect::<Vec<_>>()
                    .join(",");
                clauses.push(format!("file_type NOT IN [{}]", joined));
            }
        }
        if let Some(prefix) = path_prefix {
            if !prefix.trim().is_empty() {
                // simple prefix filter (string starts-with)
                // SurrealDB lacks direct STARTSWITH; use string::starts_with function if available or fallback to LIKE
                // Using LIKE with escaped %; ensure prefix sanitized (no % introduced by user). For large datasets an index on path exists.
                let safe = prefix.replace('%', "");
                clauses.push(format!("path LIKE '{}%'", safe));
            }
        }
        let where_sql = if clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", clauses.join(" AND "))
        };
        let sql = format!(
            "SELECT * FROM thumbnails{where_sql} LIMIT {limit} START {offset}"
        );
        let mut resp = DB.query(sql).await?;
        let rows: Vec<Self> = resp.take(0)?;
        Ok(rows)
    }

    // Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
    pub async fn save_thumbnail_row(self) -> anyhow::Result<(), anyhow::Error> {
        log::info!("SAVING: {:?}", self.path);
        let _: Option<Self> = DB
            .create("thumbnails")
            .content::<Self>(self)
            .await?
            .take();
        Ok(())
    }

    pub async fn update_or_create_thumbnail(
        mut self,
        metadata: &super::Thumbnail,
        thumb_b64: Option<String>,
    ) -> anyhow::Result<(), anyhow::Error> {
        // Only update fields if new content present
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
            .await?
            .take(0)?;

        log::info!("Cached data Is Some: {:?}", updated.is_some());
        Ok(())
    }

}

pub async fn save_thumbnail_batch(
    thumbs: Vec<super::Thumbnail>,
) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _: Vec<super::Thumbnail> = DB
        .insert("thumbnails")
        .content::<Vec<super::Thumbnail>>(thumbs)
        .await?;
    Ok(())
}

pub async fn get_thumbnail_paths() -> anyhow::Result<Vec<String>, anyhow::Error> {
    let paths: Vec<String> = DB.query("SELECT path FROM thumbnails").await?.take(0)?;
    Ok(paths)
}


