use chrono::Utc;
use surrealdb::types::{RecordId, SurrealValue};
use crate::database::{db_activity, db_set_detail, db_set_error};

#[derive(Debug, serde::Deserialize, SurrealValue)]
pub struct SimilarHit {
    pub id: RecordId,
    pub thumb_ref: Option<crate::Thumbnail>,
    pub path: String,
    pub dist: f32, // cosine distance from Surreal (lower is better)
}

impl SimilarHit {
    /// Cosine similarity in -1..=1 (higher is better).
    pub fn similarity(&self) -> f32 {
        similarity_from_distance(self.dist)
    }
}

/// A KNN hit without the joined row.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, SurrealValue)]
pub struct SimilarPath {
    pub path: String,
    pub dist: f32,
}

impl SimilarPath {
    pub fn similarity(&self) -> f32 {
        similarity_from_distance(self.dist)
    }
}

/// Cosine similarity for a cosine distance (distance = 1 - similarity).
pub fn similarity_from_distance(dist: f32) -> f32 {
    1.0 - dist
}

/// Most neighbours one KNN query asks the HNSW index for.
pub const KNN_MAX_K: usize = 1000;

/// KNN query text: the index is asked for `start + k` neighbours with an `ef` of at least that, then paged by `$limit` and `$start`.
pub fn knn_query(k: usize, ef: usize, start: usize, fetch_rows: bool) -> String {
    let wanted = start.saturating_add(k).clamp(1, KNN_MAX_K);
    let ef = ef.clamp(wanted, KNN_MAX_K.max(wanted));
    knn_sql(wanted, ef, fetch_rows)
}

/// KNN query text for an index search of `search` neighbours with effort `effort`.
fn knn_sql(search: usize, effort: usize, fetch_rows: bool) -> String {
    let (cols, fetch) = if fetch_rows { ("id, thumb_ref, path", "FETCH thumb_ref") } else { ("path", "") };
    format!(
        "SELECT {cols}, vector::distance::knn() AS dist FROM clip_embeddings WHERE embedding <| {search}, {effort} |> $vec ORDER BY dist LIMIT $limit START $start {fetch}"
    )
}

impl super::ClipEmbeddingRow {

    /// KNN search using the HNSW index on clip_embeddings.embedding
    /// Returns `k` nearest neighbours of `query_vec` after skipping `start`.
    pub async fn find_similar_by_embedding(
        query_vec: &[f32],
        k: usize,
        ef: usize,
        start: usize,
    ) -> anyhow::Result<Vec<SimilarHit>, anyhow::Error> {
        let _ga = db_activity("KNN clip_embeddings");
        db_set_detail("Searching similar embeddings".to_string());
        let res = Self::find_similar_in(&super::DB, query_vec, k, ef, start).await?;
        log::info!("find_similar_by_embedding len: {}", res.len());
        Ok(res)
    }

    /// [`Self::find_similar_by_embedding`] on `db`.
    pub async fn find_similar_in<C: surrealdb::Connection>(
        db: &surrealdb::Surreal<C>,
        query_vec: &[f32],
        k: usize,
        ef: usize,
        start: usize,
    ) -> anyhow::Result<Vec<SimilarHit>, anyhow::Error> {
        let res: Vec<SimilarHit> = db
            .query(knn_query(k, ef, start, true))
            .bind(("vec", query_vec.to_vec()))
            .bind(("limit", k as i64))
            .bind(("start", start as i64))
            .await?
            .take(0)?;
        Ok(res)
    }

    /// KNN search returning only paths and distances.
    pub async fn find_similar_paths_in<C: surrealdb::Connection>(
        db: &surrealdb::Surreal<C>,
        query_vec: &[f32],
        k: usize,
        ef: usize,
        start: usize,
    ) -> anyhow::Result<Vec<SimilarPath>, anyhow::Error> {
        let res: Vec<SimilarPath> = db
            .query(knn_query(k, ef, start, false))
            .bind(("vec", query_vec.to_vec()))
            .bind(("limit", k as i64))
            .bind(("start", start as i64))
            .await?
            .take(0)?;
        Ok(res)
    }

}

/// Fewest neighbours one KNN query searches for.
pub const KNN_MIN: usize = 64;
/// Most neighbours one KNN query searches for.
pub const KNN_MAX: usize = 20_000;

/// KNN on any connection: the search covers `start + k` neighbours (at least [`KNN_MIN`], at most [`KNN_MAX`]) with an effort of at least `ef`, and the page is the `k` after the first `start`.
pub async fn knn_with<C: surrealdb::Connection>(db: &surrealdb::Surreal<C>, query_vec: &[f32], k: usize, ef: usize, start: usize) -> anyhow::Result<Vec<SimilarHit>, anyhow::Error> {
    let search = (start + k).clamp(KNN_MIN, KNN_MAX);
    let effort = ef.max(search).min(KNN_MAX);
    let res: Vec<SimilarHit> = db
        .query(knn_sql(search, effort, true))
        .bind(("vec", query_vec.to_vec()))
        .bind(("limit", k as i64))
        .bind(("start", start as i64))
        .await?
        .take(0)?;
    Ok(res)
}

/// Cosine similarity of a neighbour from its cosine distance.
pub fn similarity_of(distance: f32) -> f32 {
    similarity_from_distance(distance)
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

    let existing: Option<RecordId> = super::DB.query(query).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClipEmbeddingRow;
    use crate::database::tests::mem_db;

    #[derive(Clone, Debug, SurrealValue)]
    struct Seed {
        path: String,
        embedding: Vec<f32>,
    }

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    fn dense(seed: usize) -> Vec<f32> {
        unit((0..1024).map(|j| ((seed as f32 + 1.0) * 13.7 + j as f32 * 0.37 + (seed * j) as f32 * 0.001).sin()).collect())
    }

    async fn seeded(n: usize) -> surrealdb::Surreal<surrealdb::engine::local::Db> {
        let db = mem_db().await;
        let rows: Vec<Seed> = (0..n).map(|i| Seed { path: format!("/p/{i:03}.jpg"), embedding: dense(i) }).collect();
        db.query("INSERT INTO clip_embeddings $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
        db
    }

    #[test]
    fn the_query_asks_the_index_for_the_page_it_needs() {
        let q = knn_query(24, 64, 0, true);
        assert!(q.contains("<| 24, 64 |>"), "{q}");
        assert!(q.contains("FETCH thumb_ref") && q.contains("id, thumb_ref, path"));
        let paged = knn_query(24, 8, 10, false);
        assert!(paged.contains("<| 34, 34 |>"), "ef never drops below the neighbours asked for: {paged}");
        assert!(!paged.contains("FETCH"));
        assert!(knn_query(5000, 5000, 0, false).contains("<| 1000, 1000 |>"));
        assert!(knn_query(0, 0, 0, false).contains("<| 1, 1 |>"));
    }

    #[test]
    fn similarity_is_one_minus_the_cosine_distance() {
        assert_eq!(similarity_from_distance(0.0), 1.0);
        assert_eq!(similarity_from_distance(1.0), 0.0);
        assert_eq!(similarity_from_distance(2.0), -1.0);
        assert!((SimilarPath { path: "x".into(), dist: 0.25 }.similarity() - 0.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn more_than_64_neighbours_come_back() {
        let db = seeded(120).await;
        let hits = ClipEmbeddingRow::find_similar_paths_in(&db, &dense(5), 100, 200, 0).await.unwrap();
        assert_eq!(hits.len(), 100);
        assert_eq!(hits[0].path, "/p/005.jpg");
        assert!(hits[0].dist < 0.001 && hits[0].similarity() > 0.999, "{:?}", hits[0]);
        assert!(hits.windows(2).all(|w| w[0].dist <= w[1].dist + 1e-6), "results are ordered by distance");
    }

    #[tokio::test]
    async fn the_start_offset_skips_the_best_hits() {
        let db = seeded(120).await;
        let all = ClipEmbeddingRow::find_similar_paths_in(&db, &dense(7), 30, 200, 0).await.unwrap();
        let tail = ClipEmbeddingRow::find_similar_paths_in(&db, &dense(7), 10, 200, 20).await.unwrap();
        assert_eq!(tail.len(), 10);
        assert_eq!(all[0].path, "/p/007.jpg");
        assert!(tail.iter().all(|t| t.path != "/p/007.jpg"));
        assert!(tail[0].dist >= all[19].dist - 0.02, "the page starts where the first twenty end");
    }

    #[tokio::test]
    async fn orthogonal_and_opposite_vectors_have_the_expected_distances() {
        let db = mem_db().await;
        let mut e0 = vec![0.0f32; 1024];
        e0[0] = 1.0;
        let mut e1 = vec![0.0f32; 1024];
        e1[1] = 1.0;
        let opposite: Vec<f32> = e0.iter().map(|x| -x).collect();
        let rows = vec![Seed { path: "/same".into(), embedding: e0.clone() }, Seed { path: "/ortho".into(), embedding: e1 }, Seed { path: "/opposite".into(), embedding: opposite }];
        db.query("INSERT INTO clip_embeddings $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
        let hits = ClipEmbeddingRow::find_similar_paths_in(&db, &e0, 3, 64, 0).await.unwrap();
        let sim = |p: &str| hits.iter().find(|h| h.path == p).map(|h| h.similarity()).unwrap();
        assert!((sim("/same") - 1.0).abs() < 0.001);
        assert!(sim("/ortho").abs() < 0.001);
        assert!((sim("/opposite") + 1.0).abs() < 0.001);
        assert_eq!(hits.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(), vec!["/same", "/ortho", "/opposite"]);
    }

    #[tokio::test]
    async fn the_row_returning_variant_agrees_with_the_path_variant() {
        let db = seeded(40).await;
        let rows = ClipEmbeddingRow::find_similar_in(&db, &dense(3), 5, 64, 0).await.unwrap();
        let paths = ClipEmbeddingRow::find_similar_paths_in(&db, &dense(3), 5, 64, 0).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.path.clone()).collect::<Vec<_>>(), paths.iter().map(|p| p.path.clone()).collect::<Vec<_>>());
        assert!((rows[0].similarity() - paths[0].similarity()).abs() < 1e-6);
        assert!(rows[0].thumb_ref.is_none());
    }
}


