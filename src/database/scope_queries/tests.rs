use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::*;
use crate::database::collections::tests::db;

#[derive(Clone, Debug, SurrealValue)]
pub(crate) struct Seed {
    pub id: RecordId,
    pub filename: String,
    pub path: String,
    pub parent_dir: String,
    pub size: i64,
    pub file_type: String,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub description: Option<String>,
    pub hash: Option<String>,
    pub logical_group: RecordId,
}

pub(crate) fn seed(i: usize, path: &str) -> Seed {
    Seed {
        id: RecordId::new("thumbnails", format!("t{i:06}")),
        filename: path.rsplit('/').next().unwrap_or(path).to_string(),
        path: path.to_string(),
        parent_dir: path.rsplit_once('/').map_or("", |(d, _)| d).to_string(),
        size: 100 + i as i64,
        file_type: path.rsplit('.').next().unwrap_or("").to_string(),
        tags: Vec::new(),
        category: None,
        description: None,
        hash: None,
        logical_group: RecordId::new("logical_groups", "Default"),
    }
}

pub(crate) async fn insert(db: &Surreal<Db>, rows: Vec<Seed>) {
    for chunk in rows.chunks(400) {
        db.query("INSERT INTO thumbnails $rows").bind(("rows", chunk.to_vec())).await.unwrap().check().unwrap();
    }
}

pub(crate) fn path_of(i: usize) -> String {
    format!("/lib/d{}/img{i:05}.jpg", i % 7)
}

pub(crate) async fn library(db: &Surreal<Db>, n: usize) -> Vec<String> {
    let mut paths: Vec<String> = (0..n).map(path_of).collect();
    insert(db, paths.iter().enumerate().map(|(i, p)| seed(i, p)).collect()).await;
    paths.sort();
    paths
}

fn paths_of(rows: &[Thumbnail]) -> Vec<String> {
    rows.iter().map(|r| r.path.clone()).collect()
}

async fn all_pages(db: &Surreal<Db>, limit: usize, mut next: impl AsyncFnMut(&Surreal<Db>, Option<String>, usize) -> KeysetPage) -> Vec<Vec<String>> {
    let mut pages = Vec::new();
    let mut after = None;
    loop {
        let page = next(db, after.clone(), limit).await;
        pages.push(paths_of(&page.rows));
        match page.next {
            Some(n) => after = Some(n),
            None => return pages,
        }
    }
}

#[tokio::test]
async fn the_library_streams_every_row_once_in_path_order() {
    let db = db().await;
    let expected = library(&db, 23).await;
    assert_eq!(count_library(&db).await.unwrap(), 23);
    let pages = all_pages(&db, 10, async |db, after, limit| library_page(db, after.as_deref(), limit).await.unwrap()).await;
    assert_eq!(pages.iter().map(Vec::len).collect::<Vec<_>>(), vec![10, 10, 3]);
    assert_eq!(pages.concat(), expected);
}

#[tokio::test]
async fn a_library_that_fills_the_last_page_ends_with_one_empty_page() {
    let db = db().await;
    library(&db, 20).await;
    let pages = all_pages(&db, 10, async |db, after, limit| library_page(db, after.as_deref(), limit).await.unwrap()).await;
    assert_eq!(pages.iter().map(Vec::len).collect::<Vec<_>>(), vec![10, 10, 0]);
}

#[tokio::test]
async fn an_empty_library_has_one_empty_last_page() {
    let db = db().await;
    let page = library_page(&db, None, 10).await.unwrap();
    assert!(page.rows.is_empty() && page.is_last());
    assert_eq!(count_library(&db).await.unwrap(), 0);
}

#[tokio::test]
async fn tag_and_category_pages_hold_only_matching_rows() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..12).map(|i| seed(i, &path_of(i))).collect();
    for (i, r) in rows.iter_mut().enumerate() {
        if i % 3 == 0 {
            r.tags = vec!["sunset".into(), "beach".into()];
        } else if i % 3 == 1 {
            r.tags = vec!["sunrise".into()];
        }
        if i % 4 == 0 {
            r.category = Some("landscape".into());
        }
    }
    insert(&db, rows).await;
    let tagged = all_pages(&db, 3, async |db, after, limit| tag_page(db, "sunset", after.as_deref(), limit).await.unwrap()).await.concat();
    assert_eq!(tagged.len(), 4);
    assert!(tagged.windows(2).all(|w| w[0] < w[1]));
    let rows = tag_page(&db, "sunset", None, 100).await.unwrap().rows;
    assert!(rows.iter().all(|r| r.tags.contains(&"sunset".to_string())));
    assert!(tag_page(&db, "sun", None, 100).await.unwrap().rows.is_empty(), "tags match exactly");
    let categories = category_page(&db, "landscape", None, 100).await.unwrap().rows;
    assert_eq!(categories.len(), 3);
    assert!(category_page(&db, "nothing", None, 100).await.unwrap().rows.is_empty());
}

#[tokio::test]
async fn recent_rows_are_the_newest_first_and_capped() {
    let db = db().await;
    for i in 0..6 {
        db.query("CREATE thumbnails SET path = $p, filename = $f, parent_dir = '/r', size = 5, file_type = 'jpg', tags = [], logical_group = logical_groups:Default, db_created = $at")
            .bind(("p", format!("/r/{i}.jpg")))
            .bind(("f", format!("{i}.jpg")))
            .bind(("at", surrealdb::types::Datetime::from(chrono::DateTime::parse_from_rfc3339(&format!("2024-01-0{}T00:00:00Z", i + 1)).unwrap().with_timezone(&chrono::Utc))))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let rows = recent_rows(&db, 4).await.unwrap();
    assert_eq!(paths_of(&rows), vec!["/r/5.jpg", "/r/4.jpg", "/r/3.jpg", "/r/2.jpg"]);
}

#[tokio::test]
async fn untagged_and_undescribed_views_list_what_lacks_the_field() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..9).map(|i| seed(i, &path_of(i))).collect();
    rows[0].tags = vec!["a".into()];
    rows[1].tags = vec!["a".into()];
    rows[2].description = Some("a cat".into());
    rows[3].description = Some(String::new());
    rows.push(seed(100, "/lib/clip.mp4"));
    insert(&db, rows).await;
    let untagged = smart_page(&db, SmartView::Untagged, None, 100).await.unwrap();
    assert_eq!(untagged.rows.len(), 8, "videos count for tags");
    assert!(untagged.is_last());
    let undescribed = smart_page(&db, SmartView::Undescribed, None, 100).await.unwrap();
    assert_eq!(undescribed.rows.len(), 8, "one description is empty and counts as none; the video is not an image");
    assert!(!paths_of(&undescribed.rows).contains(&path_of(2)));
    assert!(paths_of(&undescribed.rows).contains(&path_of(3)));
}

#[tokio::test]
async fn the_no_clip_view_skips_embedded_images_and_videos() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..8).map(|i| seed(i, &path_of(i))).collect();
    rows.push(seed(50, "/lib/movie.mp4"));
    insert(&db, rows).await;
    for i in [1usize, 4] {
        db.query("CREATE clip_embeddings SET path = $p, embedding = $e").bind(("p", path_of(i))).bind(("e", vec![0.5f32; 1024])).await.unwrap().check().unwrap();
    }
    let page = smart_page(&db, SmartView::NoClip, None, 100).await.unwrap();
    assert_eq!(page.rows.len(), 6);
    assert!(page.rows.iter().all(|r| r.path != path_of(1) && r.path != path_of(4) && r.path != "/lib/movie.mp4"));
    assert!(page.is_last());
}

#[tokio::test]
async fn smart_views_continue_where_they_stopped() {
    let db = db().await;
    let n = PAGE * 2 + 120;
    let rows: Vec<Seed> = (0..n).map(|i| seed(i, &format!("/big/f{i:06}.jpg"))).collect();
    insert(&db, rows).await;
    let mut seen = Vec::new();
    let mut after = None;
    let mut calls = 0;
    loop {
        let page = smart_page(&db, SmartView::Untagged, after.as_deref(), 300).await.unwrap();
        seen.extend(paths_of(&page.rows));
        calls += 1;
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    assert_eq!(seen.len(), n);
    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted, seen, "each row once, in path order");
    assert!(calls >= 2, "{calls} calls");
}

#[tokio::test]
async fn views_that_are_not_row_queries_are_refused() {
    let db = db().await;
    assert!(smart_page(&db, SmartView::FailedThumbs, None, 10).await.is_err());
    assert!(smart_page(&db, SmartView::Duplicates, None, 10).await.is_err());
}

#[tokio::test]
async fn the_duplicate_scan_reads_paths_sizes_and_hashes_of_sized_rows() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..5).map(|i| seed(i, &path_of(i))).collect();
    rows[1].size = 0;
    rows[2].hash = Some("h2".into());
    insert(&db, rows).await;
    let (first, next) = size_rows_page(&db, None, 3).await.unwrap();
    assert_eq!(first.len(), 3);
    assert!(next.is_some());
    let (rest, next) = size_rows_page(&db, next.as_deref(), 3).await.unwrap();
    assert_eq!(rest.len(), 1, "the zero-sized row is left out");
    assert!(next.is_none());
    let all: Vec<SizeRow> = first.into_iter().chain(rest).collect();
    assert_eq!(all.iter().find(|r| r.path == path_of(2)).unwrap().hash.as_deref(), Some("h2"));
}

#[tokio::test]
async fn cached_scans_resolve_by_ui_id_with_the_newest_winning_and_page_their_paths() {
    let db = db().await;
    for (title, started, id) in [("old", "2024-01-01T00:00:00Z", 7u64), ("new", "2024-02-01T00:00:00Z", 7), ("other", "2024-03-01T00:00:00Z", 8)] {
        db.query("CREATE cached_scans SET root = '/r', title = $t, scan_id = $s, started = $at")
            .bind(("t", title.to_string()))
            .bind(("s", id))
            .bind(("at", surrealdb::types::Datetime::from(chrono::DateTime::parse_from_rfc3339(started).unwrap().with_timezone(&chrono::Utc))))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let found = cached_scan(&db, 7).await.unwrap().expect("scan 7");
    let title: Vec<String> = db.query("SELECT VALUE title FROM $id").bind(("id", found.clone())).await.unwrap().take(0).unwrap();
    assert_eq!(title, vec!["new".to_string()]);
    assert!(cached_scan(&db, 99).await.unwrap().is_none());
    let items: Vec<String> = (0..12).map(|i| format!("/s/{i:02}.jpg")).collect();
    for p in &items {
        db.query("CREATE cached_scan_items SET scan_ref = $r, path = $p").bind(("r", found.clone())).bind(("p", p.clone())).await.unwrap().check().unwrap();
    }
    let mut got = Vec::new();
    for offset in [0, 5, 10] {
        got.extend(cached_scan_paths(&db, &found, offset, 5).await.unwrap());
    }
    got.sort();
    assert_eq!(got, items);
    assert!(cached_scan_paths(&db, &found, 12, 5).await.unwrap().is_empty());
}

pub(crate) fn unit_vector(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..1024)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm = v.iter().map(|f| f * f).sum::<f32>().sqrt();
    v.iter_mut().for_each(|f| *f /= norm);
    v
}

pub(crate) async fn embedded_library(db: &Surreal<Db>, n: usize) {
    insert(db, (0..n).map(|i| seed(i, &path_of(i))).collect()).await;
    for i in 0..n {
        db.query("CREATE clip_embeddings SET path = $p, thumb_ref = $t, embedding = $e")
            .bind(("p", path_of(i)))
            .bind(("t", RecordId::new("thumbnails", format!("t{i:06}"))))
            .bind(("e", unit_vector(i as u64 + 1)))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
}

#[tokio::test]
async fn a_nearest_neighbour_search_can_return_more_than_the_old_limit_of_64() {
    let db = db().await;
    embedded_library(&db, 120).await;
    let query = unit_vector(1);
    let first = crate::database::clip_embeddings::knn_with(&db, &query, 100, 100, 0).await.unwrap();
    assert_eq!(first.len(), 100);
    assert_eq!(first[0].path, path_of(0), "the vector itself is nearest");
    assert!(first.windows(2).all(|w| w[0].dist <= w[1].dist), "nearest first");
    assert!((crate::database::clip_embeddings::similarity_of(first[0].dist) - 1.0).abs() < 1e-3);
}

#[tokio::test]
async fn nearest_neighbour_pages_follow_each_other_without_repeats() {
    let db = db().await;
    embedded_library(&db, 90).await;
    let query = unit_vector(5);
    let mut seen = Vec::new();
    for start in [0usize, 30, 60] {
        let page = crate::database::clip_embeddings::knn_with(&db, &query, 30, 64, start).await.unwrap();
        assert_eq!(page.len(), 30, "page at {start}");
        seen.extend(page.into_iter().map(|h| h.path));
    }
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 90, "{} distinct of {}", unique.len(), seen.len());
}

#[tokio::test]
async fn an_embedding_is_found_by_path() {
    let db = db().await;
    embedded_library(&db, 3).await;
    let found = embedding_of(&db, &path_of(2)).await.unwrap().expect("embedding");
    assert_eq!(found.len(), 1024);
    assert!(embedding_of(&db, "/nope.jpg").await.unwrap().is_none());
}

#[tokio::test]
async fn tag_queries_select_rename_and_delete_by_value() {
    use crate::database::thumbnails::{DELETE_TAG_SQL, RENAME_TAG_SQL};
    let db = db().await;
    let mut rows: Vec<Seed> = (0..4).map(|i| seed(i, &path_of(i))).collect();
    rows[0].tags = vec!["sun".into(), "sea".into()];
    rows[1].tags = vec!["sea".into()];
    rows[2].tags = vec!["sun".into(), "sand".into()];
    insert(&db, rows).await;

    let with_sun: Vec<Thumbnail> = db.query("SELECT * FROM thumbnails WHERE tags CONTAINS $tag").bind(("tag", "sun".to_string())).await.unwrap().take(0).unwrap();
    assert_eq!(with_sun.len(), 2);

    let renamed: Vec<Thumbnail> = db.query(RENAME_TAG_SQL).bind(("old", "sun".to_string())).bind(("new", "sea".to_string())).await.unwrap().take(0).unwrap();
    assert_eq!(renamed.len(), 2);
    let mut tags: Vec<Vec<String>> = renamed.into_iter().map(|r| { let mut t = r.tags; t.sort(); t }).collect();
    tags.sort();
    assert_eq!(tags, vec![vec!["sand".to_string(), "sea".to_string()], vec!["sea".to_string()]]);

    let deleted: Vec<Thumbnail> = db.query(DELETE_TAG_SQL).bind(("tag", "sea".to_string())).await.unwrap().take(0).unwrap();
    assert_eq!(deleted.len(), 3);
    let left: Vec<Thumbnail> = db.query("SELECT * FROM thumbnails WHERE tags CONTAINS $tag").bind(("tag", "sea".to_string())).await.unwrap().take(0).unwrap();
    assert!(left.is_empty());
}
