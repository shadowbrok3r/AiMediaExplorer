//! Thumbnail blob table: JPEG/PNG bytes keyed by a path hash, stored apart from the metadata rows.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Mutex;

use anyhow::{Context, anyhow};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use chrono::{DateTime, Utc};
use image::{DynamicImage, GenericImageView, ImageFormat};
use surrealdb::types::{Bytes, Datetime, RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::{DB, db_activity, db_set_detail};

pub const TABLE: &str = "thumb_blobs";
pub const FORMAT_JPEG: &str = "jpeg";
pub const FORMAT_PNG: &str = "png";
pub const PUT_BATCH: usize = 64;
pub const GET_CHUNK: usize = 512;
pub const MAX_DIM: u32 = 512;
pub const JPEG_QUALITY: u8 = 85;

/// Lowercase hex of the first 16 bytes of blake3 over the thumbnail record's `path` string.
pub fn key_for_path(path: &str) -> String {
    let hash = blake3::hash(path.as_bytes());
    hash.as_bytes()[..16].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn blob_id(key: &str) -> RecordId {
    RecordId::new(TABLE, key)
}

/// One blob to store; `format` is `jpeg` or `png`.
#[derive(Clone, Debug)]
pub struct ThumbBlobIn {
    pub path: String,
    pub format: String,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
    pub src_modified: Option<DateTime<Utc>>,
    pub src_size: Option<u64>,
}

impl ThumbBlobIn {
    pub fn new(path: impl Into<String>, format: &str, w: u32, h: u32, bytes: Vec<u8>) -> Self {
        Self { path: path.into(), format: format.to_string(), w, h, bytes, src_modified: None, src_size: None }
    }

    /// Normalizes raw JPEG/PNG (or any decodable) bytes into a blob entry.
    pub fn from_image_bytes(path: impl Into<String>, raw: &[u8]) -> anyhow::Result<Self> {
        let n = normalize_thumb(raw)?;
        Ok(Self::new(path, n.format, n.w, n.h, n.bytes))
    }

    /// Normalizes a stored thumbnail string (raw base64 or a `data:` URL) into a blob entry.
    pub fn from_thumb_string(path: impl Into<String>, thumb: &str) -> anyhow::Result<Self> {
        Self::from_image_bytes(path, &decode_thumb_string(thumb)?)
    }

    pub fn with_source(mut self, modified: Option<DateTime<Utc>>, size: Option<u64>) -> Self {
        self.src_modified = modified;
        self.src_size = size;
        self
    }

    pub fn key(&self) -> String {
        key_for_path(&self.path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThumbBlob {
    pub key: String,
    pub path: String,
    pub format: String,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlobStats {
    pub count: u64,
    pub total_bytes: u64,
}

impl BlobStats {
    pub fn avg_bytes(&self) -> u64 {
        if self.count == 0 { 0 } else { self.total_bytes / self.count }
    }
}

/// Count plus a byte total that is exact or extrapolated from a sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuickStats {
    pub stats: BlobStats,
    pub estimated: bool,
}

#[derive(Clone, Debug, SurrealValue)]
struct BlobRowIn {
    id: RecordId,
    key: String,
    path: String,
    format: String,
    w: i64,
    h: i64,
    bytes: Bytes,
    src_modified: Option<Datetime>,
    src_size: Option<i64>,
}

#[derive(Debug, SurrealValue)]
struct BlobRowOut {
    key: String,
    path: String,
    format: String,
    w: i64,
    h: i64,
    bytes: Bytes,
}

#[derive(Debug, SurrealValue)]
struct StatsRow {
    count: i64,
    total: Option<i64>,
}

/// Blob storage used by the thumbnail service.
#[async_trait]
pub trait ThumbBlobStore: Send + Sync {
    async fn get_many(&self, keys: &[String]) -> anyhow::Result<HashMap<String, ThumbBlob>>;
    async fn put(&self, entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize>;
    async fn exists_many(&self, keys: &[String]) -> anyhow::Result<HashSet<String>>;
}

/// Store backed by the global SurrealDB handle.
#[derive(Clone, Copy, Debug, Default)]
pub struct SurrealThumbBlobStore;

#[async_trait]
impl ThumbBlobStore for SurrealThumbBlobStore {
    async fn get_many(&self, keys: &[String]) -> anyhow::Result<HashMap<String, ThumbBlob>> {
        get_many(keys).await
    }

    async fn put(&self, entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize> {
        put(entries).await
    }

    async fn exists_many(&self, keys: &[String]) -> anyhow::Result<HashSet<String>> {
        exists_many(keys).await
    }
}

/// In-memory store for tests.
#[derive(Default)]
pub struct MemThumbBlobStore {
    rows: Mutex<HashMap<String, ThumbBlob>>,
}

impl MemThumbBlobStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn remove_path(&self, path: &str) -> bool {
        self.lock().remove(&key_for_path(path)).is_some()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, ThumbBlob>> {
        self.rows.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait]
impl ThumbBlobStore for MemThumbBlobStore {
    async fn get_many(&self, keys: &[String]) -> anyhow::Result<HashMap<String, ThumbBlob>> {
        let rows = self.lock();
        Ok(keys.iter().filter_map(|k| rows.get(k).map(|b| (k.clone(), b.clone()))).collect())
    }

    async fn put(&self, entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize> {
        let mut rows = self.lock();
        let n = entries.len();
        for e in entries {
            let key = e.key();
            rows.insert(key.clone(), ThumbBlob { key, path: e.path, format: e.format, w: e.w, h: e.h, bytes: e.bytes });
        }
        Ok(n)
    }

    async fn exists_many(&self, keys: &[String]) -> anyhow::Result<HashSet<String>> {
        let rows = self.lock();
        Ok(keys.iter().filter(|k| rows.contains_key(*k)).cloned().collect())
    }
}

fn to_row(e: ThumbBlobIn) -> BlobRowIn {
    let key = e.key();
    BlobRowIn {
        id: blob_id(&key),
        key,
        path: e.path,
        format: e.format,
        w: e.w as i64,
        h: e.h as i64,
        bytes: Bytes::from(e.bytes),
        src_modified: e.src_modified.map(Datetime::from),
        src_size: e.src_size.map(|s| s as i64),
    }
}

const UPSERT_SQL: &str = "INSERT INTO thumb_blobs $rows ON DUPLICATE KEY UPDATE \
    path = $input.path, key = $input.key, bytes = $input.bytes, format = $input.format, w = $input.w, h = $input.h, \
    src_modified = $input.src_modified, src_size = $input.src_size, created = time::now() RETURN VALUE id";

fn is_conflict(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}").to_lowercase();
    msg.contains("can be retried") || msg.contains("conflict")
}

/// Runs `op` again after a short pause while it fails with a transaction conflict.
pub(super) async fn retry_conflicts<T, Fut>(mut op: impl FnMut() -> Fut) -> anyhow::Result<T>
where
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut attempt = 0u64;
    loop {
        match op().await {
            Err(e) if attempt < 4 && is_conflict(&e) => {
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_millis(25 * attempt)).await;
            }
            other => return other,
        }
    }
}

const INSERT_IF_ABSENT_SQL: &str = "INSERT IGNORE INTO thumb_blobs $rows RETURN VALUE id";

/// Keeps the last entry per path and writes the rows in `PUT_BATCH` statements; returns (distinct entries, ids reported by the statements).
async fn write_chunks<C: Connection>(db: &Surreal<C>, entries: Vec<ThumbBlobIn>, sql: &'static str) -> anyhow::Result<(usize, usize)> {
    let mut latest: HashMap<String, usize> = HashMap::with_capacity(entries.len());
    let mut order: Vec<ThumbBlobIn> = Vec::with_capacity(entries.len());
    for e in entries {
        match latest.get(&e.key()) {
            Some(&i) => order[i] = e,
            None => {
                latest.insert(e.key(), order.len());
                order.push(e);
            }
        }
    }
    let total = order.len();
    let mut reported = 0usize;
    let mut rest = order.into_iter();
    loop {
        let rows: Vec<BlobRowIn> = rest.by_ref().take(PUT_BATCH).map(to_row).collect();
        if rows.is_empty() {
            break;
        }
        reported += retry_conflicts(|| async {
            let ids: Vec<RecordId> = db.query(sql).bind(("rows", rows.clone())).await?.check().context("thumb_blobs write failed")?.take(0)?;
            Ok(ids.len())
        })
        .await?;
    }
    Ok((total, reported))
}

/// Batched upsert (one statement per `PUT_BATCH` entries); the last entry wins for a repeated path.
pub async fn put_with<C: Connection>(db: &Surreal<C>, entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize> {
    Ok(write_chunks(db, entries, UPSERT_SQL).await?.0)
}

/// Batched insert that leaves existing blobs untouched; returns how many were newly stored.
pub async fn put_if_absent_with<C: Connection>(db: &Surreal<C>, entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize> {
    Ok(write_chunks(db, entries, INSERT_IF_ABSENT_SQL).await?.1)
}

fn ids_for(keys: &[String]) -> Vec<RecordId> {
    keys.iter().map(|k| blob_id(k)).collect()
}

/// Blobs for the given keys in chunked point lookups; absent keys are omitted.
pub async fn get_many_with<C: Connection>(db: &Surreal<C>, keys: &[String]) -> anyhow::Result<HashMap<String, ThumbBlob>> {
    let mut out = HashMap::with_capacity(keys.len());
    for chunk in keys.chunks(GET_CHUNK) {
        let rows: Vec<BlobRowOut> = db
            .query("SELECT key, path, format, w, h, bytes FROM $ids")
            .bind(("ids", ids_for(chunk)))
            .await?
            .take(0)?;
        for r in rows {
            out.insert(
                r.key.clone(),
                ThumbBlob { key: r.key, path: r.path, format: r.format, w: r.w.max(0) as u32, h: r.h.max(0) as u32, bytes: r.bytes.into_inner().to_vec() },
            );
        }
    }
    Ok(out)
}

/// Subset of `keys` that has a stored blob.
pub async fn exists_many_with<C: Connection>(db: &Surreal<C>, keys: &[String]) -> anyhow::Result<HashSet<String>> {
    let mut out = HashSet::with_capacity(keys.len());
    for chunk in keys.chunks(GET_CHUNK) {
        let found: Vec<String> = db.query("SELECT VALUE key FROM $ids").bind(("ids", ids_for(chunk))).await?.take(0)?;
        out.extend(found);
    }
    Ok(out)
}

/// Deletes the blobs of the given thumbnail paths; returns how many existed.
pub async fn delete_by_paths_with<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<usize> {
    let mut deleted = 0usize;
    for chunk in paths.chunks(GET_CHUNK) {
        let keys: Vec<String> = chunk.iter().map(|p| key_for_path(p)).collect();
        let mut resp = db
            .query("LET $found = (SELECT VALUE key FROM $ids); DELETE $ids RETURN NONE; RETURN array::len($found);")
            .bind(("ids", ids_for(&keys)))
            .await?
            .check()?;
        let n: Option<i64> = resp.take(2)?;
        deleted += n.unwrap_or(0).max(0) as usize;
    }
    Ok(deleted)
}

/// Exact count and byte total (reads every blob record).
pub async fn stats_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<BlobStats> {
    let rows: Vec<StatsRow> = db
        .query("SELECT count() AS count, math::sum(bytes::len(bytes)) AS total FROM thumb_blobs GROUP ALL")
        .await?
        .take(0)?;
    Ok(rows.first().map(|r| BlobStats { count: r.count.max(0) as u64, total_bytes: r.total.unwrap_or(0).max(0) as u64 }).unwrap_or_default())
}

pub async fn count_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<u64> {
    let rows: Vec<StatsRow> = db.query("SELECT count() AS count FROM thumb_blobs GROUP ALL").await?.take(0)?;
    Ok(rows.first().map(|r| r.count.max(0) as u64).unwrap_or(0))
}

/// Count plus a byte total extrapolated from the first `sample` blobs; exact when the table is small.
pub async fn quick_stats_with<C: Connection>(db: &Surreal<C>, sample: usize) -> anyhow::Result<QuickStats> {
    let count = count_with(db).await?;
    if count == 0 {
        return Ok(QuickStats::default());
    }
    let lens: Vec<i64> = db
        .query("SELECT VALUE bytes::len(bytes) FROM thumb_blobs LIMIT $n")
        .bind(("n", sample.max(1) as i64))
        .await?
        .take(0)?;
    let sampled = lens.len() as u64;
    let sum: u64 = lens.iter().map(|l| (*l).max(0) as u64).sum();
    let exact = sampled >= count;
    let total_bytes = if exact || sampled == 0 { sum } else { (sum as f64 / sampled as f64 * count as f64) as u64 };
    Ok(QuickStats { stats: BlobStats { count, total_bytes }, estimated: !exact })
}

/// Random blobs for spot checks; offset scan, so intended for devtools and tests.
pub async fn sample_with<C: Connection>(db: &Surreal<C>, count: usize, start: u64) -> anyhow::Result<Vec<ThumbBlob>> {
    let rows: Vec<BlobRowOut> = db
        .query("SELECT key, path, format, w, h, bytes FROM thumb_blobs LIMIT $n START $s")
        .bind(("n", count.max(1) as i64))
        .bind(("s", start as i64))
        .await?
        .take(0)?;
    Ok(rows
        .into_iter()
        .map(|r| ThumbBlob { key: r.key, path: r.path, format: r.format, w: r.w.max(0) as u32, h: r.h.max(0) as u32, bytes: r.bytes.into_inner().to_vec() })
        .collect())
}

pub async fn put(entries: Vec<ThumbBlobIn>) -> anyhow::Result<usize> {
    if entries.is_empty() {
        return Ok(0);
    }
    let _ga = db_activity("Saving thumbnail blobs");
    db_set_detail(format!("Saving {} thumbnail blobs", entries.len()));
    put_with(&DB, entries).await
}

pub async fn get_many(keys: &[String]) -> anyhow::Result<HashMap<String, ThumbBlob>> {
    get_many_with(&DB, keys).await
}

pub async fn exists_many(keys: &[String]) -> anyhow::Result<HashSet<String>> {
    exists_many_with(&DB, keys).await
}

pub async fn delete_by_paths(paths: &[String]) -> anyhow::Result<usize> {
    if paths.is_empty() {
        return Ok(0);
    }
    delete_by_paths_with(&DB, paths).await
}

pub async fn stats() -> anyhow::Result<BlobStats> {
    stats_with(&DB).await
}

pub async fn quick_stats() -> anyhow::Result<QuickStats> {
    quick_stats_with(&DB, 256).await
}

/// Base64 payload of a thumbnail string with any `data:<mime>;base64,` prefix removed.
fn thumb_payload(thumb: &str) -> &str {
    thumb.strip_prefix("data:").and_then(|rest| rest.split_once(";base64,")).map(|(_, p)| p).unwrap_or(thumb)
}

pub fn decode_thumb_string(thumb: &str) -> anyhow::Result<Vec<u8>> {
    let payload = thumb_payload(thumb.trim());
    anyhow::ensure!(!payload.is_empty(), "empty thumbnail string");
    BASE64.decode(payload).map_err(|e| anyhow!("invalid base64: {e}"))
}

/// An encoded thumbnail that satisfies the blob table rules.
#[derive(Clone, Debug)]
pub struct NormalizedThumb {
    pub format: &'static str,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

fn has_transparency(img: &DynamicImage) -> bool {
    img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255)
}

fn encode_jpeg(img: &DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    match img.color() {
        image::ColorType::L8 | image::ColorType::L16 => enc.encode_image(&img.to_luma8())?,
        _ => enc.encode_image(&img.to_rgb8())?,
    }
    Ok(buf)
}

fn encode_png(img: &DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
    Ok(buf)
}

/// Decodes `raw`, caps the longest side at `MAX_DIM`, and returns JPEG q85 (PNG only for real transparency).
pub fn normalize_thumb(raw: &[u8]) -> anyhow::Result<NormalizedThumb> {
    let guessed = image::guess_format(raw).map_err(|e| anyhow!("unknown image format: {e}"))?;
    let img = image::load_from_memory_with_format(raw, guessed).map_err(|e| anyhow!("decode failed: {e}"))?;
    let (w, h) = img.dimensions();
    anyhow::ensure!(w > 0 && h > 0, "image has no pixels");
    let oversize = w.max(h) > MAX_DIM;
    if guessed == ImageFormat::Jpeg && !oversize {
        return Ok(NormalizedThumb { format: FORMAT_JPEG, w, h, bytes: raw.to_vec() });
    }
    let img = if oversize { img.thumbnail(MAX_DIM, MAX_DIM) } else { img };
    let (w, h) = img.dimensions();
    if has_transparency(&img) {
        return Ok(NormalizedThumb { format: FORMAT_PNG, w, h, bytes: encode_png(&img)? });
    }
    Ok(NormalizedThumb { format: FORMAT_JPEG, w, h, bytes: encode_jpeg(&img)? })
}

/// Thumbnail string for UI consumers: raw base64 for JPEG, a `data:` URL for PNG.
pub fn blob_to_thumb_string(blob: &ThumbBlob) -> String {
    let b64 = BASE64.encode(&blob.bytes);
    if blob.format == FORMAT_PNG { format!("data:image/png;base64,{b64}") } else { b64 }
}

/// Fully decodes a stored blob and checks it against its recorded format and size.
pub fn verify_blob(blob: &ThumbBlob) -> anyhow::Result<(u32, u32)> {
    let want = match blob.format.as_str() {
        FORMAT_JPEG => ImageFormat::Jpeg,
        FORMAT_PNG => ImageFormat::Png,
        other => anyhow::bail!("unknown format '{other}'"),
    };
    let img = image::load_from_memory_with_format(&blob.bytes, want).map_err(|e| anyhow!("decode failed: {e}"))?;
    let dims = img.dimensions();
    anyhow::ensure!(dims == (blob.w, blob.h), "size {}x{} differs from recorded {}x{}", dims.0, dims.1, blob.w, blob.h);
    Ok(dims)
}
