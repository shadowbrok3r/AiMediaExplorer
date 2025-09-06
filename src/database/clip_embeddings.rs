impl super::ClipEmbeddingRow {
    // Load all clip embedding rows and return mapping path -> (embedding, hash, thumb_ref)
    pub async fn load_all_clip_embeddings()
    -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let rows: Vec<super::ClipEmbeddingRow> = super::DB.select("clip_embeddings").await?;
        Ok(rows)
    }

    pub async fn load_clip_embeddings_for_path(
        path: &str,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let rows: Vec<super::ClipEmbeddingRow> = super::DB
            .query("SELECT * FROM clip_embeddings WHERE path == $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;

        Ok(rows)
    }
}

// Upsert a clip embedding row (match on path). Optionally sets thumb_ref if missing.
pub async fn upsert_clip_embedding(
    path: &str,
    hash: Option<&str>,
    embedding: &[f32],
) -> anyhow::Result<(), anyhow::Error> {
    let thumbnail_id = crate::Thumbnail::get_thumbnail_id_by_path(path).await?;
    let _ = super::DB
        .query("UPDATE clip_embeddings SET embedding = $embedding, hash = $hash, updated = time::now() WHERE path = $path")
        .bind(("embedding", embedding.to_vec()))
        .bind(("hash", hash.map(|h| h.to_string())))
        .bind(("path", path.to_string()))
        .await?;

    // Check whether row exists; if not, create it
    let existing: Option<surrealdb::RecordId> = super::DB
        .query("SELECT id FROM clip_embeddings WHERE path = $path LIMIT 1")
        .bind(("path", path.to_string()))
        .await?
        .take(0)?;
    if existing.is_none() {
        let _: Option<super::ClipEmbeddingRow> = super::DB
            .create("clip_embeddings")
            .content(super::ClipEmbeddingRow {
                id: None,
                thumb_ref: thumbnail_id,
                path: path.to_string(),
                hash: hash.map(|h| h.to_string()),
                embedding: embedding.to_vec(),
                created: None,
                updated: None,
                similarity_score: None,
                clip_embedding: None,
                clip_similarity_score: None,
            })
            .await?;
    }
    Ok(())
}


