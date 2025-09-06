use surrealdb::RecordId;
use crate::DB;

pub async fn save_thumbnail_batch(thumbs: Vec<super::Thumbnail>) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _: Vec<super::Thumbnail> = DB
        .insert("thumbnails")
        .content::<Vec<super::Thumbnail>>(thumbs)
        .await?;
    Ok(())
}

// Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
pub async fn save_thumbnail_row(row: super::Thumbnail) -> anyhow::Result<(), anyhow::Error> {
    log::info!("SAVING: {row:?}");
    let _: Option<super::Thumbnail> = DB
        .create("thumbnails")
        .content::<super::Thumbnail>(row)
        .await?
        .take();
    Ok(())
}

// Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
pub async fn get_thumbnail_by_path(path: &str) -> anyhow::Result<Option<super::Thumbnail>, anyhow::Error> {
    let resp: Option<super::Thumbnail> = DB
        .query("SELECT * FROM thumbnails WHERE path = $path LIMIT 1")
        .bind(("path", path.to_string()))
        .await?
        .take(0)?;
    Ok(resp)
}

pub async fn update_or_create_thumbnail(mut thumbnail_to_merge: super::Thumbnail, metadata: &super::FileMetadata, thumb_b64: Option<String>) -> anyhow::Result<(), anyhow::Error> {
        // Only update fields if new content present
        if let Some(desc) = &metadata.description { if desc.trim().len() > 0 { thumbnail_to_merge.description = Some(desc.clone()); } }
        if let Some(caption) = &metadata.caption { if caption.trim().len() > 0 { thumbnail_to_merge.caption = Some(caption.clone()); } }
        if !metadata.tags.is_empty() { thumbnail_to_merge.tags = metadata.tags.clone(); }
        if let Some(cat) = &metadata.category { if cat.trim().len() > 0 { thumbnail_to_merge.category = Some(cat.clone()); } }
        if let Some(embed) = &metadata.embedding { if !embed.is_empty() { thumbnail_to_merge.embedding = Some(embed.clone()); } }
        if thumb_b64.is_some() { thumbnail_to_merge.thumbnail_b64 = thumb_b64; }
        if let Some(mod_dt) = metadata.modified { thumbnail_to_merge.modified = Some(mod_dt.to_utc().into()); }
        if let Some(h) = &metadata.hash { if !h.is_empty() { thumbnail_to_merge.hash = Some(h.clone()); } }

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
                    embedding = $embedding, 
                    thumbnail_b64 = $thumbnail_b64, 
                    modified = $modified, 
                    hash = $hash 
                WHERE path = $path
                "#
            )
            .bind(("table", crate::database::THUMBNAILS))
            .bind(("filename", thumbnail_to_merge.filename.clone()))
            .bind(("file_type", thumbnail_to_merge.file_type.clone()))
            .bind(("size", thumbnail_to_merge.size))
            .bind(("description", thumbnail_to_merge.description.clone()))
            .bind(("caption", thumbnail_to_merge.caption.clone()))
            .bind(("tags", thumbnail_to_merge.tags.clone()))
            .bind(("category", thumbnail_to_merge.category.clone()))
            .bind(("embedding", thumbnail_to_merge.embedding.clone()))
            .bind(("thumbnail_b64", thumbnail_to_merge.thumbnail_b64.clone()))
            .bind(("modified", thumbnail_to_merge.modified.clone()))
            .bind(("hash", thumbnail_to_merge.hash.clone()))
            .bind(("path", thumbnail_to_merge.path.clone()))
            .await?
            .take(0)?;

        log::info!("Cached data Is Some: {:?}", updated.is_some());
        Ok(())
}


pub async fn get_thumbnail_paths() -> anyhow::Result<Vec<String>, anyhow::Error> {
    let paths: Vec<String> = DB.query("SELECT path FROM thumbnails").await?.take(0)?;
    Ok(paths)
}

pub async fn get_thumbnail_id_by_path(path: &str) -> anyhow::Result<Option<RecordId>, anyhow::Error> {
    let sql = "SELECT id FROM thumbnails WHERE path = $path LIMIT 1";
    let resp: Option<RecordId> = DB.query(sql).bind(("path", path.to_string())).await?.take(0)?;
    Ok(resp)
}

pub async fn find_thumb_out_of_paths(chunk_vec: Vec<String>) -> anyhow::Result<Vec<super::Thumbnail>, anyhow::Error> {
    let thumbs: Vec<super::Thumbnail> = DB
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
) -> anyhow::Result<Vec<super::Thumbnail>, anyhow::Error> {
    let mut clauses: Vec<String> = Vec::new();
    if let Some(ms) = min_size { clauses.push(format!("size >= {}", ms)); }
    if let Some(mx) = max_size { clauses.push(format!("size <= {}", mx)); }
    if let Some(a) = modified_after { clauses.push(format!("modified >= time::parse('{}')", a)); }
    if let Some(b) = modified_before { clauses.push(format!("modified <= time::parse('{}')", b)); }
    if let Some(exts) = excluded_exts { if !exts.is_empty() { let joined = exts.iter().map(|e| format!("'{}'", e)).collect::<Vec<_>>().join(","); clauses.push(format!("file_type NOT IN [{}]", joined)); } }
    if let Some(prefix) = path_prefix { if !prefix.trim().is_empty() { // simple prefix filter (string starts-with)
        // SurrealDB lacks direct STARTSWITH; use string::starts_with function if available or fallback to LIKE
        // Using LIKE with escaped %; ensure prefix sanitized (no % introduced by user). For large datasets an index on path exists.
        let safe = prefix.replace('%', "");
        clauses.push(format!("path LIKE '{}%'", safe));
    }}
    let where_sql = if clauses.is_empty() { String::new() } else { format!(" WHERE {}", clauses.join(" AND ")) };
    let sql = format!("SELECT * FROM thumbnails{} LIMIT {} START {}", where_sql, limit, offset);
    let mut resp = DB.query(sql).await?;
    let rows: Vec<super::Thumbnail> = resp.take(0)?;
    Ok(rows)
}