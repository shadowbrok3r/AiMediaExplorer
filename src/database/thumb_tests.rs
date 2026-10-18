//! Tests for the thumbnail blob table, the write path, the migration and the blob-aware exports.

use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem};
use surrealdb::types::{RecordId, SurrealValue};

use super::snapshot::{self, no_progress};
use super::surql::{self, ExportOptions, VerifyOptions, VerifyTarget};
use super::tests::{make_store, mem_db, now_fixed, plenty};
use super::thumb_blobs::{self, MemThumbBlobStore, ThumbBlob, ThumbBlobIn, ThumbBlobStore};
use super::thumb_migration::{self, CopyOptions, FinishOptions, MigrationState, Migrator};
use super::{DB_NAME, NS, THUMBNAILS};
use crate::Thumbnail;

fn gradient(w: u32, h: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| Rgb([(x as u8).wrapping_add(seed), (y as u8).wrapping_mul(2), seed.wrapping_mul(3).wrapping_add((x ^ y) as u8)])))
}

fn translucent(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_fn(w, h, |x, y| Rgba([200, (x % 255) as u8, (y % 255) as u8, if x < w / 2 { 0 } else { 255 }])))
}

fn encode(img: &DynamicImage, fmt: ImageFormat) -> Vec<u8> {
    let mut buf = Vec::new();
    let rgb;
    let img = if fmt == ImageFormat::Jpeg {
        rgb = DynamicImage::ImageRgb8(img.to_rgb8());
        &rgb
    } else {
        img
    };
    img.write_to(&mut Cursor::new(&mut buf), fmt).unwrap();
    buf
}

fn png_url(img: &DynamicImage) -> String {
    format!("data:image/png;base64,{}", BASE64.encode(encode(img, ImageFormat::Png)))
}

fn jpeg_b64(img: &DynamicImage) -> String {
    BASE64.encode(encode(img, ImageFormat::Jpeg))
}

fn blob_in(path: &str, bytes: Vec<u8>) -> ThumbBlobIn {
    ThumbBlobIn::new(path, thumb_blobs::FORMAT_JPEG, 1, 1, bytes)
}

fn patterned(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

struct HandleStore(Surreal<Db>);

#[async_trait]
impl ThumbBlobStore for HandleStore {
    async fn get_many(&self, keys: &[String]) -> Result<std::collections::HashMap<String, ThumbBlob>> {
        thumb_blobs::get_many_with(&self.0, keys).await
    }

    async fn put(&self, entries: Vec<ThumbBlobIn>) -> Result<usize> {
        thumb_blobs::put_with(&self.0, entries).await
    }

    async fn exists_many(&self, keys: &[String]) -> Result<HashSet<String>> {
        thumb_blobs::exists_many_with(&self.0, keys).await
    }
}

async fn store_contract<S: ThumbBlobStore>(store: &S) {
    let a = blob_in("/pics/a.jpg", vec![1, 2, 3]);
    let b = blob_in("/pics/b.jpg", vec![4, 5, 6, 7]);
    let (ka, kb, kc) = (thumb_blobs::key_for_path("/pics/a.jpg"), thumb_blobs::key_for_path("/pics/b.jpg"), thumb_blobs::key_for_path("/pics/c.jpg"));
    assert_eq!(store.put(vec![a, b]).await.unwrap(), 2);
    let got = store.get_many(&[ka.clone(), kb.clone(), kc.clone()]).await.unwrap();
    assert_eq!(got.len(), 2, "absent keys are omitted");
    assert_eq!(got[&ka].bytes, vec![1, 2, 3]);
    assert_eq!(got[&kb].path, "/pics/b.jpg");
    assert_eq!((got[&ka].format.as_str(), got[&ka].w, got[&ka].h), ("jpeg", 1, 1));
    let present = store.exists_many(&[ka.clone(), kb.clone(), kc.clone()]).await.unwrap();
    assert_eq!(present, HashSet::from([ka.clone(), kb.clone()]));
    assert!(store.exists_many(&[]).await.unwrap().is_empty());

    store.put(vec![blob_in("/pics/a.jpg", vec![9, 9])]).await.unwrap();
    let got = store.get_many(std::slice::from_ref(&ka)).await.unwrap();
    assert_eq!(got[&ka].bytes, vec![9, 9], "put replaces an existing blob");
}

#[tokio::test]
async fn the_fake_and_the_surreal_store_satisfy_the_same_contract() {
    store_contract(&MemThumbBlobStore::new()).await;
    store_contract(&HandleStore(mem_db().await)).await;
}

#[test]
fn keys_are_stable_lowercase_hex_of_the_path_hash() {
    let key = thumb_blobs::key_for_path("/pics/a.jpg");
    assert_eq!(key.len(), 32);
    assert!(key.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(key, blake3::hash(b"/pics/a.jpg").to_hex()[..32].to_string());
    assert_eq!(key, thumb_blobs::key_for_path("/pics/a.jpg"));
    assert_ne!(thumb_blobs::key_for_path("C:\\Users\\me\\a.jpg"), thumb_blobs::key_for_path("C:/Users/me/a.jpg"), "paths are hashed exactly as stored");
    assert_ne!(thumb_blobs::key_for_path("/pics/a.jpg"), thumb_blobs::key_for_path("/pics/A.jpg"));
    assert_eq!(thumb_blobs::key_for_path(""), blake3::hash(b"").to_hex()[..32].to_string());
    assert_eq!(thumb_blobs::key_for_path("C:\\Users\\me\\a.jpg"), "79a0976dbcd8b799069dc0bdf223b2fc");
    assert_eq!(thumb_blobs::key_for_path("/pics/a.jpg"), "e77dac4b4f6d49e0a8877544113e1f10");
}

#[tokio::test]
async fn blobs_round_trip_with_metadata_and_delete_by_path() {
    let db = mem_db().await;
    let when = chrono::DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z").unwrap().with_timezone(&chrono::Utc);
    let entry = ThumbBlobIn::new("/pics/x.png", "png", 40, 30, vec![7; 33]).with_source(Some(when), Some(4096));
    assert_eq!(thumb_blobs::put_with(&db, vec![entry]).await.unwrap(), 1);
    let key = thumb_blobs::key_for_path("/pics/x.png");
    let got = thumb_blobs::get_many_with(&db, std::slice::from_ref(&key)).await.unwrap();
    let blob = &got[&key];
    assert_eq!((blob.format.as_str(), blob.w, blob.h, blob.bytes.len()), ("png", 40, 30, 33));
    let (modified, size): (Vec<surrealdb::types::Datetime>, Vec<i64>) = {
        let mut r = db.query("SELECT VALUE src_modified FROM thumb_blobs; SELECT VALUE src_size FROM thumb_blobs").await.unwrap();
        (r.take(0).unwrap(), r.take(1).unwrap())
    };
    assert_eq!(chrono::DateTime::<chrono::Utc>::from(modified[0].clone()), when);
    assert_eq!(size, vec![4096]);

    let paths = vec!["/pics/x.png".to_string(), "/pics/never-stored.png".to_string()];
    assert_eq!(thumb_blobs::delete_by_paths_with(&db, &paths).await.unwrap(), 1);
    assert!(thumb_blobs::exists_many_with(&db, &[key]).await.unwrap().is_empty());
    assert_eq!(thumb_blobs::delete_by_paths_with(&db, &paths).await.unwrap(), 0);
}

#[tokio::test]
async fn large_batches_are_chunked_and_upserts_do_not_duplicate() {
    let db = mem_db().await;
    let entries: Vec<ThumbBlobIn> = (0..1500).map(|i| blob_in(&format!("/lib/{i}.jpg"), patterned(300 + i % 7, i as u64))).collect();
    assert_eq!(thumb_blobs::put_with(&db, entries.clone()).await.unwrap(), 1500);
    assert_eq!(thumb_blobs::count_with(&db).await.unwrap(), 1500);

    let keys: Vec<String> = entries.iter().map(|e| e.key()).collect();
    let got = thumb_blobs::get_many_with(&db, &keys).await.unwrap();
    assert_eq!(got.len(), 1500);
    for (e, k) in entries.iter().zip(&keys) {
        assert_eq!(got[k].bytes, e.bytes);
    }
    let mut with_missing = keys[..10].to_vec();
    with_missing.push(thumb_blobs::key_for_path("/lib/absent.jpg"));
    assert_eq!(thumb_blobs::exists_many_with(&db, &with_missing).await.unwrap().len(), 10);

    let again: Vec<ThumbBlobIn> = (0..100).map(|i| blob_in(&format!("/lib/{i}.jpg"), vec![i as u8; 5])).collect();
    thumb_blobs::put_with(&db, again).await.unwrap();
    assert_eq!(thumb_blobs::count_with(&db).await.unwrap(), 1500, "same paths update in place");
    let got = thumb_blobs::get_many_with(&db, &keys[..100]).await.unwrap();
    assert_eq!(got[&keys[42]].bytes, vec![42; 5]);

    let dup = vec![blob_in("/dup.jpg", vec![1]), blob_in("/dup.jpg", vec![2]), blob_in("/other.jpg", vec![3])];
    assert_eq!(thumb_blobs::put_with(&db, dup).await.unwrap(), 2);
    let got = thumb_blobs::get_many_with(&db, &[thumb_blobs::key_for_path("/dup.jpg")]).await.unwrap();
    assert_eq!(got.values().next().unwrap().bytes, vec![2], "last entry wins for a repeated path");
    assert_eq!(thumb_blobs::count_with(&db).await.unwrap(), 1502);
}

#[tokio::test]
async fn statistics_are_exact_or_extrapolated() {
    let db = mem_db().await;
    assert_eq!(thumb_blobs::stats_with(&db).await.unwrap(), thumb_blobs::BlobStats::default());
    assert_eq!(thumb_blobs::quick_stats_with(&db, 10).await.unwrap(), thumb_blobs::QuickStats::default());
    let entries: Vec<ThumbBlobIn> = (0..25).map(|i| blob_in(&format!("/s/{i}.jpg"), vec![0; 100])).collect();
    thumb_blobs::put_with(&db, entries).await.unwrap();
    let exact = thumb_blobs::stats_with(&db).await.unwrap();
    assert_eq!((exact.count, exact.total_bytes, exact.avg_bytes()), (25, 2500, 100));
    let quick = thumb_blobs::quick_stats_with(&db, 10).await.unwrap();
    assert!(quick.estimated);
    assert_eq!((quick.stats.count, quick.stats.total_bytes), (25, 2500));
    let full = thumb_blobs::quick_stats_with(&db, 100).await.unwrap();
    assert!(!full.estimated);
    let sample = thumb_blobs::sample_with(&db, 5, 20).await.unwrap();
    assert_eq!(sample.len(), 5);
}

#[tokio::test]
async fn insert_if_absent_never_replaces_an_existing_blob() {
    let db = mem_db().await;
    let first = vec![blob_in("/a.jpg", vec![1]), blob_in("/b.jpg", vec![2])];
    assert_eq!(thumb_blobs::put_if_absent_with(&db, first).await.unwrap(), 2);
    let second = vec![blob_in("/a.jpg", vec![9]), blob_in("/c.jpg", vec![3]), blob_in("/c.jpg", vec![4])];
    assert_eq!(thumb_blobs::put_if_absent_with(&db, second).await.unwrap(), 1, "only the new path is stored");
    let keys: Vec<String> = ["/a.jpg", "/b.jpg", "/c.jpg"].iter().map(|p| thumb_blobs::key_for_path(p)).collect();
    let got = thumb_blobs::get_many_with(&db, &keys).await.unwrap();
    assert_eq!((got[&keys[0]].bytes.clone(), got[&keys[1]].bytes.clone(), got[&keys[2]].bytes.clone()), (vec![1], vec![2], vec![4]));
    assert_eq!(thumb_blobs::count_with(&db).await.unwrap(), 3);
}

#[test]
fn thumbnails_are_normalized_to_capped_jpeg_unless_transparent() {
    let opaque = normalize_png(&gradient(300, 200, 1));
    assert_eq!((opaque.format, opaque.w, opaque.h), ("jpeg", 300, 200), "opaque PNG becomes JPEG");
    assert_eq!(image::guess_format(&opaque.bytes).unwrap(), ImageFormat::Jpeg);

    let keep = thumb_blobs::normalize_thumb(&encode(&translucent(64, 64), ImageFormat::Png)).unwrap();
    assert_eq!((keep.format, keep.w, keep.h), ("png", 64, 64), "real transparency keeps PNG");
    assert_eq!(image::guess_format(&keep.bytes).unwrap(), ImageFormat::Png);

    let big = normalize_png(&gradient(1200, 800, 2));
    assert_eq!((big.w, big.h), (512, 341), "longest side is capped at 512");
    let tall = thumb_blobs::normalize_thumb(&encode(&gradient(300, 1500, 3), ImageFormat::Jpeg)).unwrap();
    assert_eq!((tall.w, tall.h), (102, 512));
    assert_eq!(image::load_from_memory(&tall.bytes).unwrap().height(), 512);

    let jpeg = encode(&gradient(256, 171, 4), ImageFormat::Jpeg);
    let same = thumb_blobs::normalize_thumb(&jpeg).unwrap();
    assert_eq!(same.bytes, jpeg, "a JPEG within the cap is stored as is");

    assert!(thumb_blobs::normalize_thumb(b"definitely not an image").is_err());
    assert!(thumb_blobs::normalize_thumb(&[]).is_err());
    assert!(thumb_blobs::decode_thumb_string("").is_err());
    assert!(thumb_blobs::decode_thumb_string("data:image/png;base64,").is_err());
    assert!(thumb_blobs::decode_thumb_string("%%%").is_err());
}

fn normalize_png(img: &DynamicImage) -> thumb_blobs::NormalizedThumb {
    thumb_blobs::normalize_thumb(&encode(img, ImageFormat::Png)).unwrap()
}

#[test]
fn thumbnail_strings_accept_data_urls_and_raw_base64() {
    let img = gradient(100, 80, 5);
    let from_url = ThumbBlobIn::from_thumb_string("/a.png", &png_url(&img)).unwrap();
    let from_raw = ThumbBlobIn::from_thumb_string("/a.jpg", &jpeg_b64(&img)).unwrap();
    assert_eq!((from_url.format.as_str(), from_url.w, from_url.h), ("jpeg", 100, 80));
    assert_eq!((from_raw.format.as_str(), from_raw.w, from_raw.h), ("jpeg", 100, 80));
    let blob = ThumbBlob { key: String::new(), path: "/a.jpg".into(), format: "jpeg".into(), w: 100, h: 80, bytes: from_raw.bytes.clone() };
    assert_eq!(ThumbBlobIn::from_thumb_string("/a.jpg", &thumb_blobs::blob_to_thumb_string(&blob)).unwrap().bytes, from_raw.bytes);
    let png_blob = ThumbBlob { format: "png".into(), ..blob };
    assert!(thumb_blobs::blob_to_thumb_string(&png_blob).starts_with("data:image/png;base64,"));
    assert!(thumb_blobs::verify_blob(&ThumbBlob { key: String::new(), path: String::new(), format: "jpeg".into(), w: 100, h: 80, bytes: from_raw.bytes.clone() }).is_ok());
    assert!(thumb_blobs::verify_blob(&ThumbBlob { key: String::new(), path: String::new(), format: "jpeg".into(), w: 99, h: 80, bytes: from_raw.bytes.clone() }).is_err());
    assert!(thumb_blobs::verify_blob(&ThumbBlob { key: String::new(), path: String::new(), format: "jpeg".into(), w: 1, h: 1, bytes: vec![1, 2, 3] }).is_err());
}

#[tokio::test]
async fn the_blob_table_is_defined_and_keys_are_unique() {
    let db = mem_db().await;
    let info: surrealdb::types::Value = db.query("INFO FOR TABLE thumb_blobs").await.unwrap().take(0).unwrap();
    let info = info.into_json_value();
    let fields: Vec<String> = info["fields"].as_object().unwrap().keys().cloned().collect();
    for f in ["path", "key", "bytes", "format", "w", "h", "src_modified", "src_size", "created"] {
        assert!(fields.contains(&f.to_string()), "missing field {f}");
    }
    assert!(info["indexes"].as_object().unwrap().values().any(|d| d.as_str().unwrap_or("").contains("UNIQUE")));
    thumb_blobs::put_with(&db, vec![blob_in("/u.jpg", vec![1])]).await.unwrap();
    let clash = db
        .query("CREATE thumb_blobs:other SET path = '/u.jpg', key = $k, bytes = <bytes>'', format = 'jpeg', w = 1, h = 1")
        .bind(("k", thumb_blobs::key_for_path("/u.jpg")))
        .await
        .unwrap()
        .check();
    assert!(clash.is_err(), "a second record with the same key is rejected");
    let bad_format = db.query("CREATE thumb_blobs:f SET path = '/f', key = 'f', bytes = <bytes>'', format = 'gif', w = 1, h = 1").await.unwrap().check();
    assert!(bad_format.is_err(), "format is limited to jpeg and png");
}

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue)]
struct LegacySeed {
    id: RecordId,
    filename: String,
    path: String,
    file_type: String,
    parent_dir: String,
    size: i64,
    thumbnail_b64: Option<String>,
    tags: Option<Vec<String>>,
    description: Option<String>,
    logical_group: RecordId,
}

fn seed_row(i: usize, b64: Option<String>) -> LegacySeed {
    LegacySeed {
        id: RecordId::new(THUMBNAILS, format!("{i:08x}")),
        filename: format!("img{i}.jpg"),
        path: format!("/lib/img{i}.jpg"),
        file_type: "image".into(),
        parent_dir: "/lib".into(),
        size: 1000 + i as i64,
        thumbnail_b64: b64,
        tags: Some(vec!["t".into(), format!("n{}", i % 3)]),
        description: Some(format!("desc {i}")),
        logical_group: RecordId::new("logical_groups", "default"),
    }
}

async fn insert_rows(db: &Surreal<Db>, rows: Vec<LegacySeed>) {
    db.query("INSERT INTO thumbnails $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
}

async fn legacy_count(db: &Surreal<Db>) -> u64 {
    let rows: Vec<surrealdb::types::Value> = db.query("SELECT count() AS count FROM thumbnails WHERE thumbnail_b64 != NONE GROUP ALL").await.unwrap().take(0).unwrap();
    rows.first().map(|v| v.clone().into_json_value()["count"].as_u64().unwrap()).unwrap_or(0)
}

async fn rows_by_path(db: &Surreal<Db>, cols: &str, path: &str) -> Vec<Thumbnail> {
    db.query(format!("SELECT {cols} FROM thumbnails WHERE path = $p")).bind(("p", path.to_string())).await.unwrap().take(0).unwrap()
}

#[tokio::test]
async fn list_columns_leave_out_the_inline_thumbnail_but_keep_every_other_field() {
    let db = mem_db().await;
    insert_rows(&db, vec![seed_row(1, Some(jpeg_b64(&gradient(64, 48, 1)))), seed_row(2, None)]).await;
    let meta = rows_by_path(&db, super::thumbnails::COLS_META, "/lib/img1.jpg").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].thumbnail_b64, None);
    assert_eq!((meta[0].filename.as_str(), meta[0].size, meta[0].parent_dir.as_str()), ("img1.jpg", 1001, "/lib"));
    assert_eq!(meta[0].description.as_deref(), Some("desc 1"));
    assert_eq!(meta[0].tags, vec!["t".to_string(), "n1".to_string()]);
    let full = rows_by_path(&db, super::thumbnails::COLS_WITH_INLINE, "/lib/img1.jpg").await;
    assert!(full[0].thumbnail_b64.as_deref().is_some_and(|s| !s.is_empty()));
    let bare = rows_by_path(&db, super::thumbnails::COLS_META, "/lib/img2.jpg").await;
    assert_eq!(bare.len(), 1, "rows without optional fields still deserialize");
}

#[tokio::test]
async fn the_write_path_stores_metadata_on_the_row_and_the_image_as_a_blob() {
    let db = mem_db().await;
    let path = "/lib/new.jpg";
    let base = Thumbnail { path: path.into(), filename: "new.jpg".into(), file_type: "image".into(), size: 777, parent_dir: "/lib".into(), modified: Some(chrono::Utc::now().into()), ..Default::default() };
    let meta = Thumbnail { description: Some("a cat".into()), tags: vec!["pet".into()], ..base.clone() };
    base.clone().upsert_with(&db, &meta, Some(png_url(&gradient(256, 170, 9)))).await.unwrap();

    let rows = rows_by_path(&db, super::thumbnails::COLS_WITH_INLINE, path).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].thumbnail_b64, None, "new rows carry no inline thumbnail");
    assert_eq!((rows[0].description.as_deref(), rows[0].tags.clone(), rows[0].size), (Some("a cat"), vec!["pet".to_string()], 777));
    let key = thumb_blobs::key_for_path(path);
    let blobs = thumb_blobs::get_many_with(&db, std::slice::from_ref(&key)).await.unwrap();
    assert_eq!((blobs[&key].format.as_str(), blobs[&key].w, blobs[&key].h), ("jpeg", 256, 170));

    base.clone().upsert_with(&db, &Thumbnail { caption: Some("c".into()), ..base.clone() }, None).await.unwrap();
    assert_eq!(rows_by_path(&db, super::thumbnails::COLS_META, path).await.len(), 1, "update by path does not insert again");
    assert!(thumb_blobs::exists_many_with(&db, std::slice::from_ref(&key)).await.unwrap().contains(&key), "no thumbnail given leaves the blob alone");

    db.query("UPDATE thumbnails SET thumbnail_b64 = 'LEGACY' WHERE path = $p").bind(("p", path.to_string())).await.unwrap().check().unwrap();
    base.clone().upsert_with(&db, &base, Some("%%not-an-image%%".into())).await.unwrap();
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, path).await.unwrap().as_deref(), Some("LEGACY"), "an unusable thumbnail leaves the legacy value");
    base.clone().upsert_with(&db, &base, Some(String::new())).await.unwrap();
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, path).await.unwrap().as_deref(), Some("LEGACY"));

    base.clone().upsert_with(&db, &base, Some(jpeg_b64(&gradient(80, 60, 3)))).await.unwrap();
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, path).await.unwrap(), None, "a new thumbnail replaces and clears the legacy copy");
    let blobs = thumb_blobs::get_many_with(&db, std::slice::from_ref(&key)).await.unwrap();
    assert_eq!((blobs[&key].w, blobs[&key].h), (80, 60));
}

#[tokio::test]
async fn single_row_reads_fill_the_thumbnail_string_from_the_blob_table() {
    let db = mem_db().await;
    let plain = ThumbBlobIn::from_image_bytes("/lib/a.jpg", &encode(&gradient(64, 48, 1), ImageFormat::Jpeg)).unwrap();
    let keep_alpha = ThumbBlobIn::from_image_bytes("/lib/b.png", &encode(&translucent(32, 32), ImageFormat::Png)).unwrap();
    thumb_blobs::put_with(&db, vec![plain.clone(), keep_alpha]).await.unwrap();

    let mut jpeg_row = Thumbnail { path: "/lib/a.jpg".into(), ..Default::default() };
    jpeg_row.hydrate_inline_with(&db).await;
    assert_eq!(jpeg_row.thumbnail_b64, Some(BASE64.encode(&plain.bytes)), "JPEG blobs come back as raw base64");

    let mut png_row = Thumbnail { path: "/lib/b.png".into(), ..Default::default() };
    png_row.hydrate_inline_with(&db).await;
    assert!(png_row.thumbnail_b64.as_deref().is_some_and(|s| s.starts_with("data:image/png;base64,")));

    let mut legacy = Thumbnail { path: "/lib/a.jpg".into(), thumbnail_b64: Some("LEGACY".into()), ..Default::default() };
    legacy.hydrate_inline_with(&db).await;
    assert_eq!(legacy.thumbnail_b64.as_deref(), Some("LEGACY"), "an inline copy is never replaced");

    let mut absent = Thumbnail { path: "/lib/none.jpg".into(), ..Default::default() };
    absent.hydrate_inline_with(&db).await;
    assert_eq!(absent.thumbnail_b64, None);
}

#[tokio::test]
async fn legacy_accessors_return_only_rows_that_still_hold_inline_data() {
    let db = mem_db().await;
    insert_rows(&db, vec![seed_row(1, Some("AAAA".into())), seed_row(2, None), seed_row(3, Some(String::new()))]).await;
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, "/lib/img1.jpg").await.unwrap().as_deref(), Some("AAAA"));
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, "/lib/img2.jpg").await.unwrap(), None);
    assert_eq!(super::Thumbnail::legacy_thumbnail_b64_with(&db, "/lib/img3.jpg").await.unwrap(), None);
    let paths: Vec<String> = (1..=4).map(|i| format!("/lib/img{i}.jpg")).collect();
    let many = super::Thumbnail::legacy_thumbnail_b64_many_with(&db, &paths).await.unwrap();
    assert_eq!(many.len(), 1);
    assert_eq!(many["/lib/img1.jpg"], "AAAA");
}

fn migrator(db: &Surreal<Db>, store: &Path) -> std::sync::Arc<Migrator> {
    Migrator::new(db.clone(), store.to_path_buf())
}

fn quick_copy() -> CopyOptions {
    CopyOptions { require_snapshot: false, batch: 4, workers: 2, ..Default::default() }
}

fn quick_finish() -> FinishOptions {
    FinishOptions { require_snapshot: false, batch: 4, ..Default::default() }
}

struct Library {
    db: Surreal<Db>,
    tmp: tempfile::TempDir,
    store: PathBuf,
}

/// 10 decodable rows (PNG URL, raw JPEG, translucent PNG, oversize PNG), 2 undecodable, 3 without inline data.
async fn library() -> Library {
    let db = mem_db().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("db").join("ai_search");
    let mut rows = Vec::new();
    for i in 0..6 {
        rows.push(seed_row(i, Some(png_url(&gradient(256, 170, i as u8)))));
    }
    for i in 6..8 {
        rows.push(seed_row(i, Some(jpeg_b64(&gradient(200, 150, i as u8)))));
    }
    rows.push(seed_row(8, Some(png_url(&translucent(90, 60)))));
    rows.push(seed_row(9, Some(png_url(&gradient(1400, 900, 7)))));
    rows.push(seed_row(10, Some("this is not base64 of an image".into())));
    rows.push(seed_row(11, Some(String::new())));
    for i in 12..15 {
        rows.push(seed_row(i, None));
    }
    insert_rows(&db, rows).await;
    Library { db, tmp, store }
}

#[tokio::test]
async fn copy_writes_blobs_only_and_leaves_the_originals_untouched() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    let before = legacy_count(&lib.db).await;
    assert_eq!(before, 12);
    let report = m.run_copy(quick_copy()).await.unwrap();
    assert_eq!((report.migrated, report.failed, report.skipped_existing, report.cancelled), (10, 2, 0, false));
    assert!(report.bytes_out > 0 && report.bytes_in > report.bytes_out);
    assert_eq!(legacy_count(&lib.db).await, before, "copy never clears the originals");
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);

    let key = |i: usize| thumb_blobs::key_for_path(&format!("/lib/img{i}.jpg"));
    let keys: Vec<String> = (0..12).map(key).collect();
    let blobs = thumb_blobs::get_many_with(&lib.db, &keys).await.unwrap();
    assert_eq!(blobs[&key(0)].format, "jpeg");
    assert_eq!((blobs[&key(6)].format.as_str(), blobs[&key(6)].w), ("jpeg", 200));
    assert_eq!(blobs[&key(8)].format, "png", "transparency is preserved");
    assert_eq!((blobs[&key(9)].w, blobs[&key(9)].h), (512, 329), "oversize thumbnails are capped");
    assert!(!blobs.contains_key(&key(10)) && !blobs.contains_key(&key(11)));
    for b in blobs.values() {
        thumb_blobs::verify_blob(b).unwrap();
    }

    let state = m.state();
    assert!(state.copy_done && state.started.is_some() && state.finished.is_none());
    assert_eq!((state.migrated, state.failed, state.rows_total), (10, 2, Some(15)));
    assert_eq!(state.failures.len(), 2);
    assert!(thumb_migration::state_path(&lib.store).is_file());
    assert_eq!(thumb_migration::state_path(&lib.store).parent(), lib.store.parent());

    let progress = m.progress();
    assert_eq!((progress.phase, progress.migrated, progress.failed), (thumb_migration::Phase::CopyDone, 10, 2));
    assert!(!m.is_running());
}

#[tokio::test]
async fn copy_is_idempotent_and_never_overwrites_a_newer_blob() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    let fresh = ThumbBlobIn::new("/lib/img0.jpg", "jpeg", 2, 2, vec![0xAB; 10]);
    thumb_blobs::put_with(&lib.db, vec![fresh]).await.unwrap();
    m.run_copy(quick_copy()).await.unwrap();
    let key0 = thumb_blobs::key_for_path("/lib/img0.jpg");
    let got = thumb_blobs::get_many_with(&lib.db, std::slice::from_ref(&key0)).await.unwrap();
    assert_eq!(got[&key0].bytes, vec![0xAB; 10], "a blob written by the new pipeline wins over the legacy copy");
    assert_eq!(m.state().skipped_existing, 1);

    let again = m.run_copy(CopyOptions { restart: true, ..quick_copy() }).await.unwrap();
    assert_eq!((again.migrated, again.skipped_existing, again.failed), (0, 10, 2));
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);
}

#[tokio::test]
async fn copy_resumes_after_an_interruption_without_redoing_finished_batches() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    let first = m.run_copy(CopyOptions { max_batches: Some(2), ..quick_copy() }).await.unwrap();
    assert!(first.cancelled);
    assert_eq!(first.processed, 8, "two batches of four rows");
    let mid = m.state();
    assert!(!mid.copy_done);
    assert!(mid.last_key.is_some());
    let done_so_far = thumb_blobs::count_with(&lib.db).await.unwrap();
    assert!(done_so_far > 0 && done_so_far <= 8);

    let m2 = migrator(&lib.db, &lib.store);
    let rest = m2.run_copy(quick_copy()).await.unwrap();
    assert!(!rest.cancelled);
    assert_eq!(rest.processed, 4, "only the remaining legacy rows are visited");
    assert_eq!(m2.state().migrated, 10);
    assert!(m2.state().copy_done);
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);
}

#[tokio::test]
async fn cancel_and_pause_control_a_running_copy() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    assert!(m.spawn_copy(CopyOptions { batch: 1, ..quick_copy() }));
    assert!(!m.spawn_copy(quick_copy()), "a second run cannot start while one is active");
    m.pause();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    m.resume();
    m.cancel();
    for _ in 0..200 {
        if !m.is_running() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(!m.is_running());
    let rest = m.run_copy(quick_copy()).await.unwrap();
    assert!(rest.migrated + m.progress().skipped_existing <= 10);
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);
}

#[tokio::test]
async fn finish_refuses_until_the_copy_is_complete() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    assert!(m.run_finish(quick_finish()).await.is_err());
    assert_eq!(legacy_count(&lib.db).await, 12);
}

#[tokio::test]
async fn finish_blocks_on_undecodable_rows_unless_told_to_clear_them() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    m.run_copy(quick_copy()).await.unwrap();

    let blocked = m.run_finish(quick_finish()).await.unwrap();
    assert!(!blocked.ok);
    assert_eq!((blocked.legacy_rows, blocked.covered, blocked.undecodable_count, blocked.cleared), (12, 10, 2, 0));
    assert_eq!(legacy_count(&lib.db).await, 12, "nothing is cleared while undecodable rows block");
    assert!(m.state().finished.is_none());

    let done = m.run_finish(FinishOptions { clear_undecodable: true, ..quick_finish() }).await.unwrap();
    assert!(done.ok, "{}", done.summary);
    assert_eq!((done.cleared, done.remaining, done.sample_failures.len()), (12, 0, 0));
    assert_eq!(legacy_count(&lib.db).await, 0);
    assert!(m.state().finished.is_some());
    assert_eq!(m.progress().phase, thumb_migration::Phase::Finished);

    let kept = rows_by_path(&lib.db, super::thumbnails::COLS_META, "/lib/img3.jpg").await;
    assert_eq!((kept[0].filename.as_str(), kept[0].description.as_deref(), kept[0].size), ("img3.jpg", Some("desc 3"), 1003), "metadata survives the clear");
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);

    let missing = m.missing_paths(100).await.unwrap();
    assert_eq!(missing.scanned, 15);
    assert_eq!(missing.missing, 5, "two undecodable plus three rows that never had a thumbnail");
    let want: HashSet<String> = [10, 11, 12, 13, 14].iter().map(|i| format!("/lib/img{i}.jpg")).collect();
    assert_eq!(missing.paths.iter().cloned().collect::<HashSet<_>>(), want);
    assert_eq!(m.missing_paths(2).await.unwrap().paths.len(), 2);
}

#[tokio::test]
async fn finish_repairs_missing_blobs_from_the_inline_copy_before_clearing() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    m.run_copy(quick_copy()).await.unwrap();
    thumb_blobs::delete_by_paths_with(&lib.db, &["/lib/img2.jpg".to_string(), "/lib/img7.jpg".to_string()]).await.unwrap();
    let report = m.run_finish(FinishOptions { clear_undecodable: true, ..quick_finish() }).await.unwrap();
    assert!(report.ok, "{}", report.summary);
    assert_eq!(report.repaired, 2);
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10, "the deleted blobs were rebuilt");
    assert_eq!(legacy_count(&lib.db).await, 0);
}

#[tokio::test]
async fn finish_refuses_when_a_sampled_blob_does_not_decode() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    m.run_copy(quick_copy()).await.unwrap();
    let key = thumb_blobs::key_for_path("/lib/img4.jpg");
    lib.db
        .query("UPDATE type::record('thumb_blobs', $k) SET bytes = $b")
        .bind(("k", key))
        .bind(("b", surrealdb::types::Bytes::from(vec![1u8, 2, 3, 4])))
        .await
        .unwrap()
        .check()
        .unwrap();
    let report = m.run_finish(FinishOptions { clear_undecodable: true, sample: 100, ..quick_finish() }).await.unwrap();
    assert!(!report.ok);
    assert_eq!(report.sample_failures.len(), 1);
    assert_eq!(report.sample_failures[0].path, "/lib/img4.jpg");
    assert_eq!(report.cleared, 0);
    assert_eq!(legacy_count(&lib.db).await, 12, "a failed sample decode clears nothing");
}

#[tokio::test]
async fn finish_converts_rows_that_gained_inline_data_after_the_copy() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    m.run_copy(quick_copy()).await.unwrap();
    insert_rows(&lib.db, vec![seed_row(50, Some(png_url(&gradient(64, 64, 50))))]).await;
    let report = m.run_finish(FinishOptions { clear_undecodable: true, ..quick_finish() }).await.unwrap();
    assert!(report.ok);
    assert_eq!(report.repaired, 1, "rows added after the copy are repaired during verification");
    assert_eq!(legacy_count(&lib.db).await, 0);
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 11);
}

#[tokio::test]
async fn the_migration_takes_a_snapshot_before_the_first_write_and_reuses_a_fresh_one() {
    let lib = library().await;
    let store = make_store(lib.tmp.path(), "snap_store");
    let m = migrator(&lib.db, &store);
    let first = m.ensure_snapshot(no_progress()).await.unwrap();
    let info = snapshot::list_snapshots_for(&store);
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].path, PathBuf::from(&first));
    assert_eq!(info[0].label.as_deref(), Some("thumbs"));
    assert!(snapshot::read_manifest(&info[0].path).unwrap().consistent);
    let second = m.ensure_snapshot(no_progress()).await.unwrap();
    assert_eq!(first, second, "a snapshot from the last hour is reused");
    assert_eq!(snapshot::list_snapshots_for(&store).len(), 1);

    let report = m.run_copy(CopyOptions { require_snapshot: true, ..quick_copy() }).await.unwrap();
    assert_eq!(report.snapshot.as_deref(), Some(first.as_str()));
    assert_eq!(m.state().snapshot.as_deref(), Some(first.as_str()));
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 10);
}

#[tokio::test]
async fn no_blob_is_written_when_the_snapshot_cannot_be_taken() {
    let lib = library().await;
    let m = migrator(&lib.db, &lib.store);
    let err = m.run_copy(CopyOptions { require_snapshot: true, ..quick_copy() }).await.unwrap_err();
    assert!(format!("{err:#}").contains("not found") || format!("{err:#}").contains("Path"), "{err:#}");
    assert_eq!(thumb_blobs::count_with(&lib.db).await.unwrap(), 0);
    assert!(m.state().started.is_none());
    assert_eq!(m.progress().phase, thumb_migration::Phase::Failed);
    assert!(m.progress().error.is_some());
}

#[tokio::test]
async fn finish_demands_the_snapshot_to_still_exist() {
    let lib = library().await;
    let store = make_store(lib.tmp.path(), "snap_store");
    let m = migrator(&lib.db, &store);
    m.run_copy(CopyOptions { require_snapshot: true, ..quick_copy() }).await.unwrap();
    let snap = PathBuf::from(m.state().snapshot.unwrap());
    snapshot::delete_snapshot(&store, &snap).unwrap();
    assert!(!snap.exists());
    let report = m.run_finish(FinishOptions { clear_undecodable: true, batch: 4, ..Default::default() }).await.unwrap();
    assert!(report.ok);
    let fresh = PathBuf::from(m.state().snapshot.unwrap());
    assert!(fresh.is_dir(), "a new snapshot was taken before anything was cleared");
    assert_eq!(snapshot::list_snapshots_for(&store).len(), 1);
    assert!(snapshot::read_manifest(&fresh).is_some_and(|m| m.consistent));
}

#[test]
fn restoring_a_snapshot_resets_the_migration_state() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snap = snapshot::create_snapshot_blocking(&store, "manual", None, now_fixed(), &plenty, &no_progress()).unwrap();
    MigrationState { started: Some("t".into()), copy_done: true, finished: Some("t".into()), ..Default::default() }.save(&store).unwrap();
    assert!(thumb_migration::state_path(&store).is_file());
    snapshot::restore_snapshot_blocking(&snap.path, &store, now_fixed(), &plenty, &no_progress()).unwrap();
    assert!(!thumb_migration::state_path(&store).exists());
    assert_eq!(MigrationState::load(&store), MigrationState::default());
}

#[test]
fn the_state_file_round_trips_and_tolerates_missing_or_damaged_content() {
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("db").join("ai_search");
    assert_eq!(MigrationState::load(&store), MigrationState::default());
    let state = MigrationState { started: Some("s".into()), last_key: Some("k".into()), migrated: 5, failed: 1, copy_done: true, ..Default::default() };
    state.save(&store).unwrap();
    assert_eq!(MigrationState::load(&store), state);
    std::fs::write(thumb_migration::state_path(&store), "{ not json").unwrap();
    assert_eq!(MigrationState::load(&store), MigrationState::default());
    std::fs::write(thumb_migration::state_path(&store), r#"{"migrated": 3, "future_field": true}"#).unwrap();
    assert_eq!(MigrationState::load(&store).migrated, 3);
}

async fn blob_db_with_images(n: usize) -> Surreal<Db> {
    let db = mem_db().await;
    insert_rows(&db, (0..n).map(|i| seed_row(i, None)).collect()).await;
    let blobs: Vec<ThumbBlobIn> = (0..n).map(|i| ThumbBlobIn::from_image_bytes(format!("/lib/img{i}.jpg"), &encode(&gradient(64, 48, i as u8), ImageFormat::Jpeg)).unwrap()).collect();
    thumb_blobs::put_with(&db, blobs).await.unwrap();
    db
}

async fn text_of(db: &Surreal<Db>, dir: &Path, name: &str, include: bool) -> (PathBuf, String, u64) {
    let path = dir.join(name);
    let rep = surql::export_atomic_with(db, &path, &no_progress(), ExportOptions { include_thumb_blobs: include }).await.unwrap();
    let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned();
    (path, text, rep.bytes)
}

async fn fresh_mem() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    db
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exports_leave_out_thumbnail_blobs_by_default_but_keep_their_definition() {
    let tmp = tempfile::tempdir().unwrap();
    let src = blob_db_with_images(40).await;
    let big: Vec<ThumbBlobIn> = (0..30).map(|i| blob_in(&format!("/big/{i}.jpg"), patterned(60_000, i as u64))).collect();
    thumb_blobs::put_with(&src, big).await.unwrap();

    let (without_path, without, without_bytes) = text_of(&src, tmp.path(), "no-blobs.surql", false).await;
    let (_, with, with_bytes) = text_of(&src, tmp.path(), "blobs.surql", true).await;
    assert!(without_bytes < 60_000, "export without blobs is small ({without_bytes} bytes)");
    assert!(with_bytes > 3_000_000, "export with blobs carries the data ({with_bytes} bytes)");
    assert!(without.contains("DEFINE TABLE") && without.contains("thumb_blobs"), "the table definition is still exported");
    assert!(without.contains("DEFINE FIELD OVERWRITE bytes ON thumb_blobs") && without.contains("thumb_blobs_key_idx"));
    assert!(!without.contains("TABLE DATA: thumb_blobs") && !without.contains("b\"FFD8"), "no blob records");
    assert!(without.contains("TABLE DATA: thumbnails"), "other tables keep their records");
    assert!(with.contains("TABLE DATA: thumb_blobs"));
    assert!(!tmp.path().join("no-blobs.surql.partial").exists() && !tmp.path().join("no-blobs.surql.partial.schema").exists());

    let dst = fresh_mem().await;
    dst.import(&without_path).await.unwrap();
    let info: surrealdb::types::Value = dst.query("INFO FOR DB").await.unwrap().take(0).unwrap();
    assert!(info.into_json_value()["tables"].as_object().unwrap().contains_key("thumb_blobs"));
    assert_eq!(thumb_blobs::count_with(&dst).await.unwrap(), 0);
    assert_eq!(surql::count_rows(&dst, "thumbnails").await.unwrap(), 40);
    let a = surql::build_manifest_with(&src, &no_progress(), false).await.unwrap();
    let b = surql::build_manifest_with(&dst, &no_progress(), false).await.unwrap();
    assert!(!a.tables.contains_key("thumb_blobs"));
    assert!(surql::compare_manifests(&a, &b).iter().all(|t| t.ok));
    let full_dst = surql::build_manifest(&dst, &no_progress()).await.unwrap();
    assert!(full_dst.tables["thumb_blobs"].indexes.iter().any(|d| d.contains("UNIQUE")), "the key index was imported");
    assert!(full_dst.tables["thumb_blobs"].fields.contains(&"bytes".to_string()));
    thumb_blobs::put_with(&dst, vec![blob_in("/after-import.jpg", vec![1, 2])]).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verify_ignores_thumbnail_blobs_unless_asked_to_compare_them() {
    let tmp = tempfile::tempdir().unwrap();
    let src = blob_db_with_images(20).await;
    let (without, _, _) = text_of(&src, tmp.path(), "a.surql", false).await;
    let (with, _, _) = text_of(&src, tmp.path(), "b.surql", true).await;

    let plain = surql::verify_export(&without, VerifyTarget::Live(&src), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(plain.ok, "{:?}", plain.problems);
    assert!(!plain.imported.tables.contains_key("thumb_blobs") && !plain.tables.iter().any(|t| t.table == "thumb_blobs"));

    let strict = VerifyOptions { include_thumb_blobs: true, ..Default::default() };
    let full = surql::verify_export(&with, VerifyTarget::Live(&src), &strict, &no_progress()).await.unwrap();
    assert!(full.ok, "{:?}", full.problems);
    assert_eq!(full.imported.tables["thumb_blobs"].count, 20);
    let mismatch = surql::verify_export(&without, VerifyTarget::Live(&src), &strict, &no_progress()).await.unwrap();
    assert!(!mismatch.ok, "comparing blobs against a file that has none is a mismatch");

    let manifest = surql::build_manifest(&src, &no_progress()).await.unwrap();
    let saved = surql::verify_export(&without, VerifyTarget::Manifest(&manifest), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(saved.ok, "{:?}", saved.problems);

    let old = fresh_mem().await;
    super::define_schema(&old).await.unwrap();
    old.query("REMOVE TABLE thumb_blobs").await.unwrap().check().unwrap();
    insert_rows(&old, (0..20).map(|i| seed_row(i, None)).collect()).await;
    let (legacy_file, legacy_text, _) = text_of(&old, tmp.path(), "legacy.surql", false).await;
    assert!(!legacy_text.contains("thumb_blobs"));
    let against = surql::verify_export(&legacy_file, VerifyTarget::Live(&src), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(against.ok, "an export from before the blob table existed still verifies: {:?}", against.problems);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exported_blob_bytes_survive_the_surql_round_trip_exactly() {
    let tmp = tempfile::tempdir().unwrap();
    let src = mem_db().await;
    let all_values: Vec<u8> = (0..=255u8).cycle().take(256 * 4).collect();
    let one_mib = patterned(1024 * 1024, 77);
    let zeros = vec![0u8; 4096];
    let edge = vec![0u8, 255, 0, 255, 13, 10, 34, 39, 92, 59, 96, 0];
    let entries = vec![
        blob_in("/bytes/all-256.bin", all_values.clone()),
        blob_in("/bytes/1mib.bin", one_mib.clone()),
        blob_in("/bytes/zeros.bin", zeros.clone()),
        blob_in("/bytes/edge.bin", edge.clone()),
        blob_in("/bytes/one.bin", vec![200]),
    ];
    thumb_blobs::put_with(&src, entries.clone()).await.unwrap();
    let (path, text, bytes) = text_of(&src, tmp.path(), "bytes.surql", true).await;
    assert!(bytes > 2 * 1024 * 1024, "hex encoded 1 MiB blob is in the file");
    assert!(text.contains("TABLE DATA: thumb_blobs"));

    let dst = fresh_mem().await;
    dst.import(&path).await.unwrap();
    let keys: Vec<String> = entries.iter().map(|e| e.key()).collect();
    let got = thumb_blobs::get_many_with(&dst, &keys).await.unwrap();
    assert_eq!(got.len(), 5);
    for e in &entries {
        assert_eq!(got[&e.key()].bytes, e.bytes, "bytes of {} differ after the round trip", e.path);
        assert_eq!(got[&e.key()].path, e.path);
    }
    let a = surql::build_manifest(&src, &no_progress()).await.unwrap();
    let b = surql::build_manifest(&dst, &no_progress()).await.unwrap();
    assert!(surql::compare_manifests(&a, &b).iter().all(|t| t.ok));
    let report = surql::verify_export(&path, VerifyTarget::Live(&src), &VerifyOptions { include_thumb_blobs: true, ..Default::default() }, &no_progress()).await.unwrap();
    assert!(report.ok, "{:?}", report.problems);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_into_a_new_store_handles_exports_without_blobs() {
    let tmp = tempfile::tempdir().unwrap();
    let src = blob_db_with_images(15).await;
    let (file, _, _) = text_of(&src, tmp.path(), "n.surql", false).await;
    let target = tmp.path().join("new-store");
    let rep = surql::import_into_new_store(&file, &target, &no_progress()).await.unwrap();
    assert_eq!(rep.manifest.tables["thumbnails"].count, 15);
    assert_eq!(rep.manifest.tables["thumb_blobs"].count, 0);
    assert!(rep.manifest.tables["thumb_blobs"].indexes.iter().any(|d| d.contains("UNIQUE")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_export_honours_the_blob_setting() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let db = blob_db_with_images(10).await;
    let changed = snapshot::EngineChange::Changed { from: Some("3.2.4".into()), to: "3.3.4".into(), reason: "test".into() };
    let pre = |settings: snapshot::MaintenanceSettings| snapshot::PreOpen { change: changed.clone(), snapshot: None, note: None, settings, restored: None };

    let snapshot::AutoExport::Exported(plain, _) = snapshot::auto_export_if_changed(&db, &store, &pre(Default::default())).await else { panic!("expected export") };
    let text = std::fs::read_to_string(&plain).unwrap();
    assert!(!text.contains("TABLE DATA: thumb_blobs") && text.contains("thumb_blobs"));

    let with = snapshot::MaintenanceSettings { export_include_thumb_blobs: true, ..Default::default() };
    let snapshot::AutoExport::Exported(full, _) = snapshot::auto_export_if_changed(&db, &store, &pre(with)).await else { panic!("expected export") };
    assert!(std::fs::read_to_string(&full).unwrap().contains("TABLE DATA: thumb_blobs"));
    assert!(!snapshot::MaintenanceSettings::default().export_include_thumb_blobs);
    let parsed: snapshot::MaintenanceSettings = serde_json::from_str(r#"{"retention_keep": 3}"#).unwrap();
    assert!(!parsed.export_include_thumb_blobs, "settings files from before this option load with it off");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_thumbnail_storage_window_renders_headless() {
    let ctx = eframe::egui::Context::default();
    crate::ui::devtools::thumb_storage::set_open(&ctx, true);
    for _ in 0..3 {
        let mut out = ctx.run_ui(eframe::egui::RawInput::default(), |ui| crate::ui::devtools::thumb_storage::show(ui.ctx()));
        out.textures_delta.clear();
        assert!(!out.shapes.is_empty());
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

fn rss_mib() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) / 1024;
    (get("VmRSS"), get("VmHWM"))
}

/// Times a 5000-row list page with and without the inline column (AME_PROBE_DIR).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "probes an on-disk store; set AME_PROBE_DIR"]
async fn probe_list_page() {
    use std::time::Instant;
    use surrealdb::engine::local::SurrealKv;
    let store = std::env::var("AME_PROBE_DIR").expect("AME_PROBE_DIR");
    let db = Surreal::new::<SurrealKv>(store.as_str()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    for (label, cols) in [("explicit columns without inline", super::thumbnails::COLS_META), ("explicit columns with inline", super::thumbnails::COLS_WITH_INLINE), ("explicit columns without inline (repeat)", super::thumbnails::COLS_META)] {
        let t = Instant::now();
        let rows: Vec<Thumbnail> = db.query(format!("SELECT {cols} FROM thumbnails LIMIT 5000 START 10000")).await.unwrap().take(0).unwrap();
        let bytes: usize = rows.iter().map(|r| r.thumbnail_b64.as_ref().map_or(0, |s| s.len())).sum();
        println!("{label:<42} {:>5} rows {:>8.1} ms  inline bytes {:>10}  rss {:?}", rows.len(), t.elapsed().as_secs_f64() * 1000.0, bytes, rss_mib());
    }
    drop(db);
    snapshot::wait_released(Path::new(&store), std::time::Duration::from_secs(60)).await;
}

/// Times blob reads and writes against a migrated store (AME_PROBE_DIR).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "probes an on-disk store; set AME_PROBE_DIR"]
async fn probe_blob_io() {
    use std::time::Instant;
    use surrealdb::engine::local::SurrealKv;
    let store = std::env::var("AME_PROBE_DIR").expect("AME_PROBE_DIR");
    let db = Surreal::new::<SurrealKv>(store.as_str()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    let count = thumb_blobs::count_with(&db).await.unwrap();
    println!("blobs {count}");
    for window in [32usize, 100, 200, 500, 1000] {
        let mut times = Vec::new();
        for round in 0..5u64 {
            let start = thumb_migration::random_below(count.saturating_sub(window as u64).max(1)) + round;
            let keys: Vec<String> = thumb_blobs::sample_with(&db, window, start).await.unwrap().into_iter().map(|b| b.key).collect();
            let t = Instant::now();
            let got = thumb_blobs::get_many_with(&db, &keys).await.unwrap();
            times.push((t.elapsed().as_secs_f64() * 1000.0, got.values().map(|b| b.bytes.len()).sum::<usize>()));
        }
        times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("get_many {window:>5} keys: median {:>7.1} ms  ({} KiB)", times[times.len() / 2].0, times[times.len() / 2].1 / 1024);
    }
    let keys: Vec<String> = thumb_blobs::sample_with(&db, 500, 100).await.unwrap().into_iter().map(|b| b.key).collect();
    let t = Instant::now();
    let found = thumb_blobs::exists_many_with(&db, &keys).await.unwrap();
    println!("exists_many 500 keys: {:.1} ms ({} found)", t.elapsed().as_secs_f64() * 1000.0, found.len());
    let entries: Vec<ThumbBlobIn> = (0..64).map(|i| blob_in(&format!("/probe/{i}.jpg"), patterned(7000, i as u64))).collect();
    let t = Instant::now();
    thumb_blobs::put_with(&db, entries.clone()).await.unwrap();
    println!("put 64 x 7 KB: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    let t = Instant::now();
    thumb_blobs::put_with(&db, entries).await.unwrap();
    println!("put 64 x 7 KB again (upsert): {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    let paths: Vec<String> = (0..64).map(|i| format!("/probe/{i}.jpg")).collect();
    thumb_blobs::delete_by_paths_with(&db, &paths).await.unwrap();
    drop(db);
    snapshot::wait_released(Path::new(&store), std::time::Duration::from_secs(60)).await;
}

/// Times count queries against an existing store (AME_PROBE_DIR).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "probes an on-disk store; set AME_PROBE_DIR"]
async fn probe_count_queries() {
    use std::time::Instant;
    use surrealdb::engine::local::SurrealKv;
    let store = std::env::var("AME_PROBE_DIR").expect("AME_PROBE_DIR");
    let db = Surreal::new::<SurrealKv>(store.as_str()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    for (label, sql) in [
        ("count thumbnails group all", "SELECT count() AS count FROM thumbnails GROUP ALL"),
        ("count thumbnails again", "SELECT count() AS count FROM thumbnails GROUP ALL"),
        ("value count()", "SELECT VALUE count() FROM thumbnails"),
        ("array::len ids", "RETURN array::len((SELECT VALUE id FROM thumbnails))"),
        ("count legacy filter", "SELECT count() AS count FROM thumbnails WHERE thumbnail_b64 != NONE GROUP ALL"),
        ("count blobs group all", "SELECT count() AS count FROM thumb_blobs GROUP ALL"),
        ("sample blob lens", "SELECT VALUE bytes::len(bytes) FROM thumb_blobs LIMIT 256"),
    ] {
        let t = Instant::now();
        let v: surrealdb::types::Value = db.query(sql).await.unwrap().take(0).unwrap();
        let text: String = format!("{v:?}").chars().take(60).collect();
        println!("{label:<30} {:>9.1} ms  rss {:?}  -> {text}", t.elapsed().as_secs_f64() * 1000.0, rss_mib());
    }
    drop(db);
    snapshot::wait_released(Path::new(&store), std::time::Duration::from_secs(60)).await;
}

/// Times keyset-pagination query shapes against an existing store (AME_PROBE_DIR).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "probes an on-disk store; set AME_PROBE_DIR"]
async fn probe_keyset_queries() {
    use std::ops::Bound;
    use std::time::Instant;
    use surrealdb::engine::local::SurrealKv;
    use surrealdb::types::{RecordIdKey, RecordIdKeyRange};
    let store = std::env::var("AME_PROBE_DIR").expect("AME_PROBE_DIR");
    let db = Surreal::new::<SurrealKv>(store.as_str()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    let keys: Vec<String> = db.query("SELECT VALUE id FROM thumbnails LIMIT 1 START 15000").await.unwrap().take::<Vec<RecordId>>(0).unwrap().iter().map(|r| format!("{:?}", r.key)).collect();
    println!("sample key {keys:?} rss {:?}", rss_mib());
    for pos in [0usize, 7_500, 15_000, 29_000] {
        let after: Option<RecordId> = if pos == 0 { None } else { db.query("SELECT VALUE id FROM thumbnails LIMIT 1 START $s").bind(("s", pos as i64)).await.unwrap().take::<Vec<RecordId>>(0).unwrap().pop() };
        let after_key = after.as_ref().map(|r| match &r.key {
            RecordIdKey::String(s) => s.clone(),
            other => panic!("unexpected key {other:?}"),
        });
        let range_id = |k: &str| RecordId::new(THUMBNAILS, RecordIdKey::Range(Box::new(RecordIdKeyRange { start: Bound::Excluded(RecordIdKey::String(k.to_string())), end: Bound::Unbounded })));
        let shapes: Vec<(&str, String, Option<RecordId>, Option<RecordId>)> = vec![
            ("A where id> + order by", "SELECT id, path FROM thumbnails WHERE id > $after AND thumbnail_b64 != NONE ORDER BY id LIMIT 64".into(), after.clone(), None),
            ("B range scan + where", "SELECT id, path FROM $rid WHERE thumbnail_b64 != NONE LIMIT 64".into(), None, after_key.as_deref().map(range_id)),
            ("C where id> no order", "SELECT id, path FROM thumbnails WHERE id > $after AND thumbnail_b64 != NONE LIMIT 64".into(), after.clone(), None),
            ("D range scan no where", "SELECT id, path FROM $rid LIMIT 64".into(), None, after_key.as_deref().map(range_id)),
            ("E start offset", format!("SELECT id, path FROM thumbnails WHERE thumbnail_b64 != NONE LIMIT 64 START {pos}"), None, None),
        ];
        let after_path: Option<String> = if pos == 0 { None } else { db.query("SELECT VALUE path FROM thumbnails ORDER BY path LIMIT 1 START $s").bind(("s", pos as i64)).await.unwrap().take::<Vec<String>>(0).unwrap().pop() };
        let path_shapes: Vec<(&str, &str)> = vec![
            ("F path> order+filter", "SELECT id, path FROM thumbnails WHERE path > $p AND thumbnail_b64 != NONE ORDER BY path LIMIT 64"),
            ("G path> order", "SELECT id, path FROM thumbnails WHERE path > $p ORDER BY path LIMIT 64"),
            ("H path> index order", "SELECT id, path FROM thumbnails WITH INDEX path_idx WHERE path > $p ORDER BY path LIMIT 64"),
            ("I path> filter no order", "SELECT id, path FROM thumbnails WITH INDEX path_idx WHERE path > $p AND thumbnail_b64 != NONE LIMIT 64"),
        ];
        if let Some(p) = &after_path {
            for (label, sql) in path_shapes {
                let t = Instant::now();
                let rows: Vec<surrealdb::types::Value> = db.query(sql).bind(("p", p.clone())).await.unwrap().take(0).unwrap();
                println!("pos {pos:>6} {label:<26} {:>4} rows {:>8.1} ms  rss {:?}", rows.len(), t.elapsed().as_secs_f64() * 1000.0, rss_mib());
            }
        }
        for (label, sql, after_param, rid) in shapes {
            if (sql.contains("$after") && after_param.is_none()) || (sql.contains("$rid") && rid.is_none()) {
                continue;
            }
            let t = Instant::now();
            let mut q = db.query(sql.as_str());
            if let Some(a) = after_param {
                q = q.bind(("after", a));
            }
            if let Some(r) = rid {
                q = q.bind(("rid", r));
            }
            let rows: Vec<surrealdb::types::Value> = q.await.unwrap().take(0).unwrap();
            println!("pos {pos:>6} {label:<26} {:>4} rows {:>8.1} ms  rss {:?}", rows.len(), t.elapsed().as_secs_f64() * 1000.0, rss_mib());
        }
    }
    drop(db);
    snapshot::wait_released(Path::new(&store), std::time::Duration::from_secs(60)).await;
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn noisy_photo(seed: u64, amp: u8) -> DynamicImage {
    let noise = patterned(256 * 171 * 3, seed);
    DynamicImage::ImageRgb8(RgbImage::from_fn(256, 171, |x, y| {
        let i = ((y * 256 + x) * 3) as usize;
        let n = |k: usize| noise[i + k] % amp;
        let base = (x as f32 / 256.0 * 200.0) as u8;
        Rgb([base.wrapping_add(n(0)), (y as u8).wrapping_add(n(1)), (base / 2).wrapping_add(n(2)).wrapping_add(seed as u8)])
    }))
}

/// Builds an on-disk store of legacy rows: synthetic photos, or every image of AME_SYNTH_IMAGES when set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "builds a large on-disk store; set AME_SYNTH_DIR (and optionally AME_SYNTH_ROWS, AME_SYNTH_IMAGES, AME_SYNTH_BAD)"]
async fn build_legacy_store() {
    use surrealdb::engine::local::SurrealKv;
    let store = PathBuf::from(std::env::var("AME_SYNTH_DIR").expect("AME_SYNTH_DIR"));
    let rows = env_usize("AME_SYNTH_ROWS", 30_000);
    let bad = env_usize("AME_SYNTH_BAD", 0);
    std::fs::create_dir_all(&store).unwrap();
    let db = Surreal::new::<SurrealKv>(store.to_string_lossy().as_ref()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    super::define_schema(&db).await.unwrap();
    db.query("CREATE logical_groups:default SET name = 'Default'").await.unwrap().check().unwrap();

    let mut seeds: Vec<LegacySeed> = Vec::new();
    if let Ok(dir) = std::env::var("AME_SYNTH_IMAGES") {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
        files.sort();
        for (i, f) in files.iter().enumerate() {
            let Ok(img) = image::open(f) else { continue };
            let thumb = img.thumbnail(256, 256);
            let b64 = if i % 2 == 0 { png_url(&thumb) } else { jpeg_b64(&thumb) };
            let mut row = seed_row(i, Some(b64));
            row.path = f.to_string_lossy().into_owned();
            row.filename = f.file_name().unwrap().to_string_lossy().into_owned();
            row.parent_dir = f.parent().unwrap().to_string_lossy().into_owned();
            row.size = f.metadata().map(|m| m.len() as i64).unwrap_or(0);
            seeds.push(row);
        }
        for j in 0..bad {
            let i = seeds.len() + j;
            let mut row = seed_row(i, Some(if j % 2 == 0 { "bm90IGFuIGltYWdl".to_string() } else { String::new() }));
            row.path = format!("{dir}/broken-{j}.jpg");
            row.filename = format!("broken-{j}.jpg");
            row.parent_dir = dir.clone();
            seeds.push(row);
        }
    } else {
        let pool: Vec<String> = (0..64u64).map(|i| png_url(&noisy_photo(i + 1, env_usize("AME_SYNTH_AMP", 14) as u8))).collect();
        println!("pool: {} images, average base64 length {}", pool.len(), pool.iter().map(|p| p.len()).sum::<usize>() / pool.len());
        for i in 0..rows {
            let mut row = seed_row(i, Some(pool[i % pool.len()].clone()));
            row.path = format!("/synthetic/dir{}/img{i}.jpg", i % 97);
            row.parent_dir = format!("/synthetic/dir{}", i % 97);
            seeds.push(row);
        }
    }
    let total = seeds.len();
    let mut written = 0;
    while !seeds.is_empty() {
        let rest = seeds.split_off(seeds.len().min(200));
        let batch = std::mem::replace(&mut seeds, rest);
        written += batch.len();
        insert_rows(&db, batch).await;
        if written % 5000 == 0 {
            println!("inserted {written} of {total}");
        }
    }
    println!("inserted {written} legacy rows into {}", store.display());
    drop(db);
    snapshot::wait_released(&store, std::time::Duration::from_secs(120)).await;
}
