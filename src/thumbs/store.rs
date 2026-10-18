//! Blob store boundary used by the thumbnail service, with an in-memory fake and a latency/failure gate.

use super::key::ThumbKey;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobFormat {
    Jpeg,
    Png,
}

impl BlobFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            BlobFormat::Jpeg => "jpeg",
            BlobFormat::Png => "png",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "jpeg" => Some(BlobFormat::Jpeg),
            "png" => Some(BlobFormat::Png),
            _ => None,
        }
    }

    pub fn image_format(self) -> image::ImageFormat {
        match self {
            BlobFormat::Jpeg => image::ImageFormat::Jpeg,
            BlobFormat::Png => image::ImageFormat::Png,
        }
    }

    /// Format from leading magic bytes.
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some(BlobFormat::Jpeg)
        } else if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
            Some(BlobFormat::Png)
        } else {
            None
        }
    }
}

/// Encoded thumbnail as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blob {
    pub format: BlobFormat,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

/// Source file stamp recorded with a blob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlobStamp {
    pub src_modified: Option<i64>,
    pub src_size: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct BlobIn {
    pub path: String,
    pub key: ThumbKey,
    pub blob: Arc<Blob>,
    pub stamp: BlobStamp,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlobStats {
    pub count: u64,
    pub total_bytes: u64,
}

#[async_trait]
pub trait ThumbBlobStore: Send + Sync + 'static {
    async fn get_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>>;
    async fn put(&self, entries: Vec<BlobIn>) -> Result<usize>;
    async fn exists_many(&self, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>>;
    /// Stamps of the present blobs; absent blobs are omitted and unstamped legacy blobs carry `None` fields.
    async fn stamps_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        Ok(self.exists_many(keys).await?.into_iter().map(|k| (k, BlobStamp::default())).collect())
    }
    async fn delete_by_paths(&self, paths: &[String]) -> Result<usize>;
    async fn stats(&self) -> Result<BlobStats>;
}

#[derive(Default)]
struct MemInner {
    rows: HashMap<ThumbKey, (String, Blob, BlobStamp)>,
    get_batches: Vec<usize>,
    exists_batches: Vec<usize>,
    put_batches: Vec<usize>,
}

/// In-memory store for tests and offline runs.
#[derive(Default)]
pub struct MemStore {
    inner: Mutex<MemInner>,
}

impl MemStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn insert(&self, path: &str, blob: Blob) {
        self.insert_stamped(path, blob, BlobStamp::default());
    }

    pub fn insert_stamped(&self, path: &str, blob: Blob, stamp: BlobStamp) {
        if let Ok(mut g) = self.inner.lock() {
            g.rows.insert(ThumbKey::from_path(path), (path.to_string(), blob, stamp));
        }
    }

    pub fn contains(&self, path: &str) -> bool {
        self.inner.lock().map(|g| g.rows.contains_key(&ThumbKey::from_path(path))).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.rows.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_batches(&self) -> Vec<usize> {
        self.inner.lock().map(|g| g.get_batches.clone()).unwrap_or_default()
    }

    pub fn put_batches(&self) -> Vec<usize> {
        self.inner.lock().map(|g| g.put_batches.clone()).unwrap_or_default()
    }

    pub fn exists_batches(&self) -> Vec<usize> {
        self.inner.lock().map(|g| g.exists_batches.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl ThumbBlobStore for MemStore {
    async fn get_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>> {
        let mut g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        g.get_batches.push(keys.len());
        Ok(keys.iter().filter_map(|k| g.rows.get(k).map(|(_, b, _)| (*k, b.clone()))).collect())
    }

    async fn put(&self, entries: Vec<BlobIn>) -> Result<usize> {
        let mut g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        g.put_batches.push(entries.len());
        let n = entries.len();
        for e in entries {
            g.rows.insert(e.key, (e.path, (*e.blob).clone(), e.stamp));
        }
        Ok(n)
    }

    async fn exists_many(&self, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>> {
        let mut g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        g.exists_batches.push(keys.len());
        Ok(keys.iter().filter(|k| g.rows.contains_key(k)).copied().collect())
    }

    async fn stamps_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        let g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(keys.iter().filter_map(|k| g.rows.get(k).map(|(_, _, s)| (*k, *s))).collect())
    }

    async fn delete_by_paths(&self, paths: &[String]) -> Result<usize> {
        let mut g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut n = 0;
        for p in paths {
            if g.rows.remove(&ThumbKey::from_path(p)).is_some() {
                n += 1;
            }
        }
        Ok(n)
    }

    async fn stats(&self) -> Result<BlobStats> {
        let g = self.inner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(BlobStats { count: g.rows.len() as u64, total_bytes: g.rows.values().map(|(_, b, _)| b.bytes.len() as u64).sum() })
    }
}

/// Shared knobs applied by [`GatedStore`].
#[derive(Default)]
pub struct StoreGate {
    delay_ms: AtomicU64,
    fail: AtomicBool,
}

impl StoreGate {
    pub fn delay_ms(&self) -> u64 {
        self.delay_ms.load(Ordering::Relaxed)
    }

    pub fn set_delay_ms(&self, ms: u64) {
        self.delay_ms.store(ms, Ordering::Relaxed);
    }

    pub fn failing(&self) -> bool {
        self.fail.load(Ordering::Relaxed)
    }

    pub fn set_failing(&self, fail: bool) {
        self.fail.store(fail, Ordering::Relaxed);
    }
}

/// Wraps a store with an adjustable delay and failure switch.
pub struct GatedStore {
    inner: Arc<dyn ThumbBlobStore>,
    gate: Arc<StoreGate>,
}

impl GatedStore {
    pub fn new(inner: Arc<dyn ThumbBlobStore>, gate: Arc<StoreGate>) -> Self {
        Self { inner, gate }
    }

    async fn pass(&self) -> Result<()> {
        let ms = self.gate.delay_ms();
        if ms > 0 {
            tokio::time::sleep(Duration::from_millis(ms)).await;
        }
        if self.gate.failing() {
            anyhow::bail!("store failure injected from devtools");
        }
        Ok(())
    }
}

#[async_trait]
impl ThumbBlobStore for GatedStore {
    async fn get_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, Blob>> {
        self.pass().await?;
        self.inner.get_many(keys).await
    }

    async fn put(&self, entries: Vec<BlobIn>) -> Result<usize> {
        self.pass().await?;
        self.inner.put(entries).await
    }

    async fn exists_many(&self, keys: &[ThumbKey]) -> Result<HashSet<ThumbKey>> {
        self.pass().await?;
        self.inner.exists_many(keys).await
    }

    async fn stamps_many(&self, keys: &[ThumbKey]) -> Result<HashMap<ThumbKey, BlobStamp>> {
        self.pass().await?;
        self.inner.stamps_many(keys).await
    }

    async fn delete_by_paths(&self, paths: &[String]) -> Result<usize> {
        self.pass().await?;
        self.inner.delete_by_paths(paths).await
    }

    async fn stats(&self) -> Result<BlobStats> {
        self.inner.stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(n: u8) -> Blob {
        Blob { format: BlobFormat::Jpeg, w: 4, h: 4, bytes: vec![0xFF, 0xD8, 0xFF, n] }
    }

    #[tokio::test]
    async fn mem_store_round_trips_and_records_batches() {
        let store = MemStore::new();
        let a = ThumbKey::from_path("/a.jpg");
        let b = ThumbKey::from_path("/b.jpg");
        let n = store
            .put(vec![BlobIn { path: "/a.jpg".into(), key: a, blob: Arc::new(blob(1)), stamp: BlobStamp { src_modified: Some(5), src_size: Some(9) } }])
            .await
            .unwrap();
        assert_eq!(n, 1);
        let got = store.get_many(&[a, b]).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[&a], blob(1));
        assert_eq!(store.exists_many(&[a, b]).await.unwrap(), HashSet::from([a]));
        assert_eq!(store.stamps_many(&[a, b]).await.unwrap()[&a].src_size, Some(9));
        assert_eq!(store.get_batches(), vec![2]);
        assert_eq!(store.put_batches(), vec![1]);
        assert_eq!(store.stats().await.unwrap(), BlobStats { count: 1, total_bytes: 4 });
        assert_eq!(store.delete_by_paths(&["/a.jpg".into(), "/zzz".into()]).await.unwrap(), 1);
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn gate_injects_delay_and_failure() {
        let store = MemStore::new();
        let gate = Arc::new(StoreGate::default());
        let gated = GatedStore::new(store.clone(), gate.clone());
        let k = ThumbKey::from_path("/a.jpg");
        assert!(gated.get_many(&[k]).await.is_ok());
        gate.set_delay_ms(40);
        let t = std::time::Instant::now();
        gated.get_many(&[k]).await.unwrap();
        assert!(t.elapsed() >= Duration::from_millis(35));
        gate.set_delay_ms(0);
        gate.set_failing(true);
        assert!(gated.get_many(&[k]).await.is_err());
        assert!(gated.put(Vec::new()).await.is_err());
    }

    #[test]
    fn format_helpers() {
        assert_eq!(BlobFormat::parse("jpeg"), Some(BlobFormat::Jpeg));
        assert_eq!(BlobFormat::parse("webp"), None);
        assert_eq!(BlobFormat::sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(BlobFormat::Jpeg));
        assert_eq!(BlobFormat::sniff(&[0x89, b'P', b'N', b'G', 1]), Some(BlobFormat::Png));
        assert_eq!(BlobFormat::sniff(b"GIF89a"), None);
        assert_eq!(BlobFormat::Png.as_str(), "png");
    }
}
