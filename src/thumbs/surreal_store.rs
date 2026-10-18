//! The only file binding the thumbnail service to SurrealDB; blob reads and writes go through `database::thumb_blobs`.

use super::key::ThumbKey;
use super::store::{Blob, BlobFormat, BlobIn, BlobStamp, BlobStats, ThumbBlobStore};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use surrealdb::engine::local::Db;
use surrealdb::{Connection, Surreal};

/// Store backed by the app-wide database handle.
pub struct GlobalDbStore;

/// Store backed by an explicit handle, for tests.
pub struct DbStore<C: Connection> {
    db: Surreal<C>,
}

impl<C: Connection> DbStore<C> {
    pub fn new(db: Surreal<C>) -> Self {
        Self { db }
    }
}

fn global() -> Result<&'static Surreal<Db>> {
    if !crate::database::is_connected() {
        anyhow::bail!("database not connected");
    }
    Ok(&crate::database::DB)
}

#[async_trait]
impl ThumbBlobStore for GlobalDbStore {
    async fn get_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>> {
        blobs::get_many(global()?, keys).await
    }

    async fn put(&self, entries: Vec<BlobIn>) -> Result<usize> {
        blobs::put(global()?, entries).await
    }

    async fn exists_many(&self, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>> {
        blobs::exists_many(global()?, keys).await
    }

    async fn stamps_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        blobs::stamps_many(global()?, keys).await
    }

    async fn delete_by_paths(&self, paths: &[String]) -> Result<usize> {
        blobs::delete_by_paths(global()?, paths).await
    }

    async fn stats(&self) -> Result<BlobStats> {
        blobs::stats(global()?).await
    }
}

#[async_trait]
impl<C: Connection> ThumbBlobStore for DbStore<C> {
    async fn get_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>> {
        blobs::get_many(&self.db, keys).await
    }

    async fn put(&self, entries: Vec<BlobIn>) -> Result<usize> {
        blobs::put(&self.db, entries).await
    }

    async fn exists_many(&self, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>> {
        blobs::exists_many(&self.db, keys).await
    }

    async fn stamps_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        blobs::stamps_many(&self.db, keys).await
    }

    async fn delete_by_paths(&self, paths: &[String]) -> Result<usize> {
        blobs::delete_by_paths(&self.db, paths).await
    }

    async fn stats(&self) -> Result<BlobStats> {
        blobs::stats(&self.db).await
    }
}

mod blobs {
    use super::*;
    use crate::database::thumb_blobs as db_blobs;
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use surrealdb::types::{Datetime, RecordId, SurrealValue};

    #[derive(Serialize, Deserialize, SurrealValue)]
    struct StampRow {
        key: String,
        src_modified: Option<Datetime>,
        src_size: Option<i64>,
    }

    fn id_of(key: &ThumbKey) -> RecordId {
        db_blobs::blob_id(&key.hex())
    }

    fn to_secs(d: &Datetime) -> i64 {
        d.timestamp()
    }

    pub async fn put<C: Connection>(db: &Surreal<C>, entries: Vec<BlobIn>) -> Result<usize> {
        let rows = entries
            .into_iter()
            .map(|e| {
                let modified = e.stamp.src_modified.and_then(|s| DateTime::<Utc>::from_timestamp(s, 0));
                db_blobs::ThumbBlobIn::new(e.path, e.blob.format.as_str(), e.blob.w, e.blob.h, e.blob.bytes.clone())
                    .with_source(modified, e.stamp.src_size)
            })
            .collect();
        db_blobs::put_with(db, rows).await
    }

    pub async fn get_many<C: Connection>(db: &Surreal<C>, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>> {
        let hexes: Vec<String> = keys.iter().map(ThumbKey::hex).collect();
        let found = db_blobs::get_many_with(db, &hexes).await?;
        Ok(found
            .into_iter()
            .filter_map(|(k, b)| Some((ThumbKey::from_hex(&k)?, Blob { format: BlobFormat::parse(&b.format)?, w: b.w, h: b.h, bytes: b.bytes })))
            .collect())
    }

    pub async fn exists_many<C: Connection>(db: &Surreal<C>, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>> {
        let hexes: Vec<String> = keys.iter().map(ThumbKey::hex).collect();
        let found = db_blobs::exists_many_with(db, &hexes).await?;
        Ok(found.iter().filter_map(|k| ThumbKey::from_hex(k)).collect())
    }

    pub async fn stamps_many<C: Connection>(db: &Surreal<C>, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }
        let ids: Vec<RecordId> = keys.iter().map(id_of).collect();
        let rows: Vec<StampRow> = db.query("SELECT key, src_modified, src_size FROM $ids").bind(("ids", ids)).await?.take(0)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let key = ThumbKey::from_hex(&r.key)?;
                Some((key, BlobStamp { src_modified: r.src_modified.as_ref().map(to_secs), src_size: r.src_size.map(|s| s.max(0) as u64) }))
            })
            .collect())
    }

    pub async fn delete_by_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> Result<usize> {
        db_blobs::delete_by_paths_with(db, paths).await
    }

    pub async fn stats<C: Connection>(db: &Surreal<C>) -> Result<BlobStats> {
        let s = db_blobs::stats_with(db).await?;
        Ok(BlobStats { count: s.count, total_bytes: s.total_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use surrealdb::engine::local::Mem;

    async fn store() -> DbStore<Db> {
        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("file_explorer").use_db("ai_search").await.unwrap();
        crate::database::define_schema(&db).await.unwrap();
        DbStore::new(db)
    }

    fn blob(n: u8, len: usize) -> Arc<Blob> {
        Arc::new(Blob { format: BlobFormat::Jpeg, w: 256, h: 171, bytes: (0..len).map(|i| (i as u8).wrapping_add(n)).collect() })
    }

    fn entry(path: &str, n: u8, stamp: BlobStamp) -> BlobIn {
        BlobIn { path: path.into(), key: ThumbKey::from_path(path), blob: blob(n, 4096), stamp }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blobs_round_trip_through_a_real_database() {
        let s = store().await;
        let stamp = BlobStamp { src_modified: Some(1_700_000_000), src_size: Some(12_345) };
        assert_eq!(s.put(vec![entry("/a/one.jpg", 1, stamp), entry("/a/two.jpg", 2, BlobStamp::default())]).await.unwrap(), 2);
        let (k1, k2, k3) = (ThumbKey::from_path("/a/one.jpg"), ThumbKey::from_path("/a/two.jpg"), ThumbKey::from_path("/a/none.jpg"));
        let got = s.get_many(&[k1, k2, k3]).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[&k1], *blob(1, 4096));
        assert_eq!(got[&k2].bytes, blob(2, 4096).bytes);
        assert_eq!((got[&k1].w, got[&k1].h, got[&k1].format), (256, 171, BlobFormat::Jpeg));
        assert_eq!(s.exists_many(&[k1, k2, k3]).await.unwrap(), HashSet::from([k1, k2]));
        let stamps = s.stamps_many(&[k1, k2, k3]).await.unwrap();
        assert_eq!(stamps[&k1], stamp);
        assert_eq!(stamps[&k2], BlobStamp::default());
        assert!(!stamps.contains_key(&k3));
        let stats = s.stats().await.unwrap();
        assert_eq!((stats.count, stats.total_bytes), (2, 8192));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn put_overwrites_in_place_and_delete_removes_by_path() {
        let s = store().await;
        s.put(vec![entry("/x.jpg", 1, BlobStamp::default())]).await.unwrap();
        s.put(vec![entry("/x.jpg", 9, BlobStamp { src_modified: Some(5), src_size: Some(6) })]).await.unwrap();
        let k = ThumbKey::from_path("/x.jpg");
        let got = s.get_many(&[k]).await.unwrap();
        assert_eq!(got[&k].bytes, blob(9, 4096).bytes);
        assert_eq!(s.stats().await.unwrap().count, 1);
        assert_eq!(s.delete_by_paths(&["/x.jpg".into(), "/missing.jpg".into()]).await.unwrap(), 1);
        assert!(s.get_many(&[k]).await.unwrap().is_empty());
        assert_eq!(s.stats().await.unwrap(), BlobStats::default());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batches_larger_than_one_transaction_chunk_all_land() {
        let s = store().await;
        let entries: Vec<BlobIn> = (0..150).map(|i| entry(&format!("/many/{i}.jpg"), i as u8, BlobStamp::default())).collect();
        let keys: Vec<ThumbKey> = entries.iter().map(|e| e.key).collect();
        assert_eq!(s.put(entries).await.unwrap(), 150);
        assert_eq!(s.get_many(&keys).await.unwrap().len(), 150);
        assert_eq!(s.exists_many(&keys).await.unwrap().len(), 150);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_inputs_are_no_ops() {
        let s = store().await;
        assert!(s.get_many(&[]).await.unwrap().is_empty());
        assert!(s.exists_many(&[]).await.unwrap().is_empty());
        assert_eq!(s.put(Vec::new()).await.unwrap(), 0);
        assert_eq!(s.delete_by_paths(&[]).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn the_global_store_reports_an_unconnected_database_instead_of_hanging() {
        if crate::database::is_connected() {
            return;
        }
        let err = GlobalDbStore.get_many(&[ThumbKey::from_path("/a.jpg")]).await.unwrap_err();
        assert!(err.to_string().contains("not connected"));
    }
}
