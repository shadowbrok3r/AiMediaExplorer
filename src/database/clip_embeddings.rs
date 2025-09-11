use chrono::Utc;
use surrealdb::RecordId;
use crate::database::{db_activity, db_set_detail, db_set_error};

#[derive(Debug, serde::Deserialize)]
pub struct SimilarHit {
    pub id: RecordId,
    pub thumb_ref: Option<crate::Thumbnail>,
    pub path: String,
    pub dist: f32, // cosine distance from Surreal (lower is better)

}

impl super::ClipEmbeddingRow {
    // Load all clip embedding rows and return mapping path -> (embedding, hash, thumb_ref)
    pub async fn load_all_clip_embeddings()
    -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select all clip_embeddings");
        db_set_detail("Loading clip embeddings".to_string());
        let rows: Vec<super::ClipEmbeddingRow> = super::DB.select("clip_embeddings").await?;
        Ok(rows)
    }

    pub async fn load_clip_embeddings_for_path(
        path: &str,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select clip_embeddings by path");
        db_set_detail("Loading clip embeddings for path".to_string());
        let rows: Vec<super::ClipEmbeddingRow> = super::DB
            .query("SELECT * FROM clip_embeddings WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;

        Ok(rows)
    }

    /// KNN search using the HNSW index on clip_embeddings.embedding
    /// Returns the top-K nearest neighbours to `query_vec`.
    pub async fn find_similar_by_embedding(
        query_vec: &[f32],
        k: usize,
        ef: usize,
    ) -> anyhow::Result<Vec<SimilarHit>, anyhow::Error> {
        log::info!("find_similar_by_embedding");
        let _ga = db_activity("KNN clip_embeddings");
        db_set_detail("Searching similar embeddings".to_string());
        // let ef = ef_for(top_n);
        let res: Vec<SimilarHit> = super::DB
            .query(
                r#"
                SELECT id, thumb_ref, path, vector::distance::knn() AS dist
                FROM clip_embeddings
                WHERE embedding <| 64, 64 |> $vec
                ORDER BY dist
                
                FETCH thumb_ref
                "#, // LIMIT $k
            )
            .bind(("vec", query_vec.to_vec()))
            .bind(("k", k as i64))
            .bind(("ef", ef as i64))
            .await?
            .take(0)?;

        log::info!("find_similar_by_embedding len: {}", res.len());
        Ok(res)
    }

}

fn _ef_for(k: usize) -> usize {
    // good starting heuristic for HNSW recall vs. latency
    (k * 4).clamp(32, 256)
}

// Upsert a clip embedding row (match on path). Optionally sets thumb_ref if missing.
pub async fn upsert_clip_embedding(
    path: &str,
    hash: Option<&str>,
    embedding: &[f32],
) -> anyhow::Result<(), anyhow::Error> {
    log::info!("upsert_clip_embedding");
    let _ga = db_activity("Upsert clip_embedding");
    db_set_detail("Saving CLIP embedding".to_string());
    let thumbnail_id = crate::Thumbnail::get_thumbnail_id_by_path(path).await?;
    let query = if let Some(id) = &thumbnail_id {
        let _ = super::DB.set("id", id.clone()).await;
        "SELECT VALUE id FROM clip_embeddings WITH INDEX clip_thumb_ref_idx WHERE thumb_ref = $id"
    } else {
        let _ = super::DB.set("path", path.to_string()).await;
        "SELECT VALUE id FROM clip_embeddings WHERE path = $path"
    };

    let existing: Option<surrealdb::RecordId> = super::DB.query(query).await
        .map_err(|e| { db_set_error(format!("Embedding lookup failed: {e}")); e })?
        .take(0)?;

    if let Some(id) = existing {
        log::info!("Existing clip_embeddings");
        let record: Option<crate::ClipEmbeddingRow> = super::DB
            .query("UPDATE clip_embeddings SET embedding = $embedding, hash = $hash, updated = time::now() WHERE id == $id")
            .bind(("embedding", embedding.to_vec()))
            .bind(("hash", hash.map(|h| h.to_string())))
            .bind(("path", path.to_string()))
            .bind(("id", id))
            .await
            .map_err(|e| { db_set_error(format!("Update embedding failed: {e}")); e })?
            .take(0)?;

        log::info!("Record is Some: {:?}", record.is_some());
    } else {
        log::error!("existing.is_none()");
        let new: Option<super::ClipEmbeddingRow> = super::DB
            .create("clip_embeddings")
            .content(super::ClipEmbeddingRow {
                id: None,
                thumb_ref: thumbnail_id,
                path: path.to_string(),
                hash: hash.map(|h| h.to_string()),
                embedding: embedding.to_vec(),
                created: Some(Utc::now().into()),
                updated: Some(Utc::now().into()),
                similarity_score: None,
                clip_similarity_score: None,
            })
            .await
            .map_err(|e| { db_set_error(format!("Insert embedding failed: {e}")); e })?;
        log::info!("New clip_embeddings: {:?}", new.is_some());
    }

    Ok(())
}


