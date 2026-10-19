use std::sync::Mutex;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::*;
use crate::database::collections::tests::db;
use crate::database::scope_queries::tests::{Seed, embedded_library, insert, library, path_of, seed, unit_vector};

struct FixedEmbedder(Option<Vec<f32>>);

#[async_trait::async_trait]
impl TextEmbedder for FixedEmbedder {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        self.0.clone().ok_or_else(|| anyhow::anyhow!("no engine"))
    }
}

fn env_with(embedder: FixedEmbedder, progress: Arc<Mutex<Vec<(u64, u64)>>>) -> Env {
    Env {
        cancel: Arc::new(AtomicBool::new(false)),
        embedder: Arc::new(embedder),
        hasher: Arc::new(duplicates::file_hash),
        progress: Arc::new(move |done, total| progress.lock().unwrap().push((done, total))),
    }
}

fn env() -> Env {
    env_with(FixedEmbedder(None), Arc::default())
}

async fn load(db: &Surreal<Db>, req: Request, env: &Env) -> anyhow::Result<(Vec<RowBatch>, Outcome)> {
    let mut batches = Vec::new();
    let outcome = run(db, &req, env, &mut |b| {
        batches.push(b);
        true
    })
    .await?;
    Ok((batches, outcome))
}

async fn load_scope(db: &Surreal<Db>, scope: Scope) -> (Vec<RowBatch>, Outcome) {
    load(db, Request::first(scope), &env()).await.unwrap()
}

fn paths(batches: &[RowBatch]) -> Vec<String> {
    batches.iter().flat_map(|b| b.rows.iter().map(|r| r.path.clone())).collect()
}

#[tokio::test]
async fn the_library_streams_in_batches_with_progress_and_a_known_total() {
    let db = db().await;
    let expected = library(&db, BATCH * 2 + 203).await;
    let progress = Arc::default();
    let env = env_with(FixedEmbedder(None), Arc::clone(&progress));
    let (batches, out) = load(&db, Request::first(Scope::Library), &env).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.rows.len()).collect::<Vec<_>>(), vec![BATCH, BATCH, 203]);
    assert_eq!(paths(&batches), expected);
    assert_eq!((out.rows, out.batches, out.total, out.more, out.cancelled), (BATCH * 2 + 203, 3, Some(1203), false, false));
    assert!(out.first_batch.is_some());
    let seen = progress.lock().unwrap().clone();
    assert_eq!(seen.last(), Some(&(1203, 1203)));
    assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0));
}

#[tokio::test]
async fn pages_read_ahead_of_their_clip_checks_still_arrive_in_page_order() {
    let db = db().await;
    let expected = library(&db, BATCH * 4 + 77).await;
    for i in (0..BATCH * 4 + 77).step_by(97) {
        db.query("CREATE clip_embeddings SET path = $p, embedding = $e").bind(("p", path_of(i))).bind(("e", vec![0.25f32; 1024])).await.unwrap().check().unwrap();
    }
    let (batches, out) = load_scope(&db, Scope::Library).await;
    assert_eq!(paths(&batches), expected, "every row once, in path order");
    assert_eq!(batches.len(), 5);
    let embedded: usize = batches.iter().map(|b| b.embedded.len()).sum();
    assert_eq!(embedded, (0..BATCH * 4 + 77).step_by(97).count(), "each batch carries the presence of its own rows");
    for batch in &batches {
        assert!(batch.embedded.iter().all(|p| batch.rows.iter().any(|r| &r.path == p)));
    }
    assert!(out.stages.page > Duration::ZERO, "the page reads are timed");
    assert!(out.stages.embed > Duration::ZERO, "the CLIP checks are timed");
}

#[tokio::test]
async fn a_load_cancelled_with_checks_in_flight_ends_cleanly_and_a_new_one_works() {
    let db = db().await;
    library(&db, BATCH * 6).await;
    let env = env();
    let cancel = env.cancel.clone();
    let out = run(&db, &Request::first(Scope::Library), &env, &mut |_| {
        cancel.store(true, Ordering::Relaxed);
        true
    })
    .await
    .unwrap();
    assert!(out.cancelled && out.rows == BATCH);
    let (batches, again) = load_scope(&db, Scope::Library).await;
    assert_eq!((paths(&batches).len(), again.cancelled), (BATCH * 6, false));
}

#[tokio::test]
async fn the_no_clip_view_loads_without_asking_which_rows_have_embeddings() {
    let db = db().await;
    library(&db, 40).await;
    let (batches, out) = load_scope(&db, Scope::Smart(SmartView::NoClip)).await;
    assert_eq!(paths(&batches).len(), 40);
    assert!(batches.iter().all(|b| b.embedded.is_empty()));
    assert_eq!(out.stages.embed, Duration::ZERO, "no presence check ran");
}

#[tokio::test]
async fn an_empty_library_loads_nothing_without_failing() {
    let db = db().await;
    let (batches, out) = load_scope(&db, Scope::Library).await;
    assert!(batches.is_empty());
    assert_eq!((out.rows, out.total), (0, Some(0)));
}

#[tokio::test]
async fn a_cancel_stops_the_load_between_batches() {
    let db = db().await;
    library(&db, BATCH * 3).await;
    let env = env();
    let cancel = env.cancel.clone();
    let mut got = 0;
    let out = run(&db, &Request::first(Scope::Library), &env, &mut |b| {
        got += b.rows.len();
        cancel.store(true, Ordering::Relaxed);
        true
    })
    .await
    .unwrap();
    assert_eq!(got, BATCH);
    assert!(out.cancelled);
}

#[tokio::test]
async fn a_receiver_that_is_gone_stops_the_load() {
    let db = db().await;
    library(&db, BATCH * 3).await;
    let mut calls = 0;
    let out = run(&db, &Request::first(Scope::Library), &env(), &mut |_| {
        calls += 1;
        false
    })
    .await
    .unwrap();
    assert_eq!((calls, out.rows), (1, BATCH));
}

#[tokio::test]
async fn batches_say_which_rows_have_a_clip_embedding() {
    let db = db().await;
    library(&db, 6).await;
    for i in [0usize, 3] {
        db.query("CREATE clip_embeddings SET path = $p, embedding = $e").bind(("p", path_of(i))).bind(("e", vec![0.5f32; 1024])).await.unwrap().check().unwrap();
    }
    let (batches, _) = load_scope(&db, Scope::Library).await;
    let mut embedded = batches[0].embedded.clone();
    embedded.sort();
    assert_eq!(embedded, vec![path_of(0), path_of(3)]);
}

#[tokio::test]
async fn recent_lists_the_newest_rows_first() {
    let db = db().await;
    for i in 0..5 {
        db.query("CREATE thumbnails SET path = $p, filename = $f, parent_dir = '/r', size = 5, file_type = 'jpg', tags = [], logical_group = logical_groups:Default, db_created = $at")
            .bind(("p", format!("/r/{i}.jpg")))
            .bind(("f", format!("{i}.jpg")))
            .bind(("at", surrealdb::types::Datetime::from(chrono::DateTime::parse_from_rfc3339(&format!("2024-02-0{}T00:00:00Z", i + 1)).unwrap().with_timezone(&chrono::Utc))))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let (batches, out) = load_scope(&db, Scope::Recent).await;
    assert_eq!(paths(&batches), vec!["/r/4.jpg", "/r/3.jpg", "/r/2.jpg", "/r/1.jpg", "/r/0.jpg"]);
    assert_eq!(out.total, Some(5));
}

#[tokio::test]
async fn a_collection_lists_its_members_and_shows_unindexed_ones_from_disk() {
    let db = db().await;
    let dir = tempfile::tempdir().unwrap();
    let on_disk = dir.path().join("loose.png");
    std::fs::write(&on_disk, b"12345").unwrap();
    let on_disk = on_disk.display().to_string();
    insert(&db, (0..3).map(|i| seed(i, &path_of(i))).collect()).await;
    let c = collections::create(&db, "Trip").await.unwrap();
    let mut members: Vec<String> = (0..3).map(path_of).collect();
    members.push(on_disk.clone());
    members.push("/gone/ghost.jpg".to_string());
    collections::add_paths(&db, &c.id, &members).await.unwrap();
    let (batches, out) = load_scope(&db, Scope::Collection("trip".into())).await;
    assert_eq!(out.rows, 5);
    assert_eq!(out.total, Some(5));
    let rows: Vec<&Thumbnail> = batches.iter().flat_map(|b| b.rows.iter()).collect();
    let loose = rows.iter().find(|r| r.path == on_disk).expect("the unindexed file is listed");
    assert_eq!((loose.filename.as_str(), loose.size, loose.file_type.as_str()), ("loose.png", 5, "png"));
    let ghost = rows.iter().find(|r| r.path == "/gone/ghost.jpg").expect("missing files are listed too");
    assert_eq!(ghost.size, 0);
    let mut unindexed = batches[0].unindexed.clone();
    unindexed.sort();
    let mut expected = vec![on_disk, "/gone/ghost.jpg".to_string()];
    expected.sort();
    assert_eq!(unindexed, expected);
}

#[tokio::test]
async fn a_collection_larger_than_a_batch_streams_all_members() {
    let db = db().await;
    let rows: Vec<Seed> = (0..BATCH + 90).map(|i| seed(i, &format!("/c/f{i:05}.jpg"))).collect();
    let members: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
    insert(&db, rows).await;
    let c = collections::create(&db, "Big").await.unwrap();
    collections::add_paths(&db, &c.id, &members).await.unwrap();
    let (batches, out) = load_scope(&db, Scope::Collection("Big".into())).await;
    assert_eq!(batches.iter().map(|b| b.rows.len()).collect::<Vec<_>>(), vec![BATCH, 90]);
    assert_eq!(out.rows, BATCH + 90);
}

#[tokio::test]
async fn an_unknown_collection_is_an_error() {
    let db = db().await;
    let err = load(&db, Request::first(Scope::Collection("nope".into())), &env()).await.unwrap_err();
    assert!(err.to_string().contains("no collection named \"nope\""), "{err}");
}

#[tokio::test]
async fn tags_and_categories_load_only_their_rows() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..10).map(|i| seed(i, &path_of(i))).collect();
    for (i, r) in rows.iter_mut().enumerate() {
        if i % 2 == 0 {
            r.tags = vec!["even".into()];
            r.category = Some("evens".into());
        }
    }
    insert(&db, rows).await;
    let (tagged, _) = load_scope(&db, Scope::Tag("even".into())).await;
    assert_eq!(paths(&tagged).len(), 5);
    let (categorized, _) = load_scope(&db, Scope::Category("evens".into())).await;
    assert_eq!(paths(&categorized).len(), 5);
    let (none, out) = load_scope(&db, Scope::Tag("odd".into())).await;
    assert!(none.is_empty() && out.rows == 0);
}

#[tokio::test]
async fn a_cached_scan_loads_its_files_and_an_unknown_one_fails() {
    let db = db().await;
    insert(&db, (0..2).map(|i| seed(i, &path_of(i))).collect()).await;
    db.query("CREATE cached_scans SET root = '/r', scan_id = 12, started = time::now()").await.unwrap().check().unwrap();
    let record = queries::cached_scan(&db, 12).await.unwrap().unwrap();
    for p in [path_of(0), path_of(1), "/elsewhere/new.jpg".to_string()] {
        db.query("CREATE cached_scan_items SET scan_ref = $r, path = $p").bind(("r", record.clone())).bind(("p", p)).await.unwrap().check().unwrap();
    }
    let (batches, out) = load_scope(&db, Scope::Scan(12)).await;
    assert_eq!(out.rows, 3);
    assert_eq!(batches[0].unindexed, vec!["/elsewhere/new.jpg".to_string()]);
    let err = load(&db, Request::first(Scope::Scan(999)), &env()).await.unwrap_err();
    assert!(err.to_string().contains("no cached scan #999"), "{err}");
}

#[tokio::test]
async fn smart_views_list_what_the_names_say() {
    let db = db().await;
    let mut rows: Vec<Seed> = (0..6).map(|i| seed(i, &path_of(i))).collect();
    rows[0].tags = vec!["t".into()];
    rows[1].description = Some("d".into());
    insert(&db, rows).await;
    db.query("CREATE clip_embeddings SET path = $p, embedding = $e").bind(("p", path_of(2))).bind(("e", vec![0.5f32; 1024])).await.unwrap().check().unwrap();
    let (untagged, _) = load_scope(&db, Scope::Smart(SmartView::Untagged)).await;
    assert_eq!(paths(&untagged).len(), 5);
    let (undescribed, _) = load_scope(&db, Scope::Smart(SmartView::Undescribed)).await;
    assert_eq!(paths(&undescribed).len(), 5);
    let (no_clip, _) = load_scope(&db, Scope::Smart(SmartView::NoClip)).await;
    let listed = paths(&no_clip);
    assert_eq!(listed.len(), 5);
    assert!(!listed.contains(&path_of(2)));
    assert!(no_clip.iter().all(|b| b.embedded.is_empty()));
}

#[tokio::test]
async fn the_failed_thumbnails_view_lists_the_shared_failures() {
    let _serial = failed::serial();
    failed::clear();
    let db = db().await;
    failed::record("/f/bad.jpg", "corrupt");
    failed::record("/f/worse.png", "truncated");
    let ours = |batches: &[super::RowBatch]| batches.iter().flat_map(|b| b.rows.iter()).filter(|r| r.path.starts_with("/f/")).cloned().collect::<Vec<_>>();
    let (batches, _) = load_scope(&db, Scope::Smart(SmartView::FailedThumbs)).await;
    let listed = ours(&batches);
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|r| r.description.as_deref() == Some("Thumbnail failed: corrupt")));
    assert!(batches.iter().map(|b| b.unindexed.len()).sum::<usize>() >= 2);
    failed::clear();
    let (none, _) = load_scope(&db, Scope::Smart(SmartView::FailedThumbs)).await;
    assert!(ours(&none).is_empty());
}

#[tokio::test]
async fn the_duplicates_view_lists_files_with_equal_contents_marked_with_their_hash() {
    let db = db().await;
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        p.display().to_string()
    };
    let a = write("a.jpg", b"AAAA");
    let b = write("b.jpg", b"AAAA");
    let c = write("c.jpg", b"BBBB");
    let lone = write("lone.jpg", b"CCCCCC");
    let mut rows: Vec<Seed> = [&a, &b, &c, &lone].iter().enumerate().map(|(i, p)| seed(i, p)).collect();
    for r in &mut rows {
        r.size = std::fs::metadata(&r.path).unwrap().len() as i64;
    }
    insert(&db, rows).await;
    let (batches, out) = load_scope(&db, Scope::Smart(SmartView::Duplicates)).await;
    let mut listed = paths(&batches);
    listed.sort();
    let mut expected = vec![a.clone(), b.clone()];
    expected.sort();
    assert_eq!(listed, expected);
    assert_eq!(out.rows, 2);
    let hash = blake3::hash(b"AAAA").to_hex().to_string();
    assert!(batches[0].rows.iter().all(|r| r.hash.as_deref() == Some(hash.as_str())));
}

#[tokio::test]
async fn a_library_without_duplicates_gives_an_empty_view() {
    let db = db().await;
    library(&db, 8).await;
    let (batches, out) = load_scope(&db, Scope::Smart(SmartView::Duplicates)).await;
    assert!(batches.is_empty());
    assert_eq!(out.rows, 0);
}

#[tokio::test]
async fn a_semantic_search_pages_by_similarity_and_keeps_its_vector() {
    let db = db().await;
    embedded_library(&db, SEARCH_PAGE + 30).await;
    let query = unit_vector(1);
    let env = env_with(FixedEmbedder(Some(query.clone())), Arc::default());
    let (first, out) = load(&db, Request::first(Scope::Semantic("a cat".into())), &env).await.unwrap();
    assert_eq!(out.rows, SEARCH_PAGE);
    assert!(out.more);
    assert_eq!(out.next_start, SEARCH_PAGE);
    assert_eq!(first[0].rows[0].path, path_of(0), "the nearest file comes first");
    let scores: Vec<f32> = first[0].scores.iter().map(|(_, s)| *s).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "best match first");
    assert!((scores[0] - 1.0).abs() < 1e-3);
    assert!(first[0].embedded.len() == SEARCH_PAGE);

    let vector = out.vector.clone().expect("the vector comes back for the next page");
    let failing = env_with(FixedEmbedder(None), Arc::default());
    let (second, out2) = load(&db, Request { scope: Scope::Semantic("a cat".into()), start: out.next_start, vector: Some(vector) }, &failing).await.unwrap();
    assert_eq!(out2.rows, 30, "the next page does not embed the text again");
    assert!(!out2.more);
    let mut all = paths(&first);
    all.extend(paths(&second));
    all.sort();
    all.dedup();
    assert_eq!(all.len(), SEARCH_PAGE + 30);
}

#[tokio::test]
async fn a_search_fails_when_the_engine_cannot_embed() {
    let db = db().await;
    embedded_library(&db, 3).await;
    let err = load(&db, Request::first(Scope::Semantic("q".into())), &env()).await.unwrap_err();
    assert!(err.to_string().contains("no engine"));
}

#[tokio::test]
async fn similar_to_a_file_leaves_the_file_out_and_scores_the_rest() {
    let db = db().await;
    embedded_library(&db, 12).await;
    let (batches, out) = load_scope(&db, Scope::Similar(path_of(4))).await;
    let listed = paths(&batches);
    assert_eq!(listed.len(), 11);
    assert!(!listed.contains(&path_of(4)));
    assert!(!out.more);
    assert_eq!(batches[0].scores.len(), 11);
}

#[tokio::test]
async fn similar_to_a_file_without_an_embedding_says_what_to_do() {
    let db = db().await;
    library(&db, 2).await;
    let err = load(&db, Request::first(Scope::Similar(path_of(0))), &env()).await.unwrap_err();
    assert!(err.to_string().contains("has no CLIP embedding yet"), "{err}");
}

#[tokio::test]
async fn folders_and_archives_belong_to_the_scanner() {
    let db = db().await;
    for scope in [Scope::Folder("/x".into()), Scope::Archive("zip:///a.zip!/".into())] {
        assert!(load(&db, Request::first(scope), &env()).await.is_err());
    }
}

#[test]
fn rows_made_from_paths_carry_what_the_file_system_knows() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("Pic.JPG");
    std::fs::write(&p, b"abc").unwrap();
    let row = row_from_path(&p.display().to_string());
    assert_eq!((row.filename.as_str(), row.file_type.as_str(), row.size), ("Pic.JPG", "jpg", 3));
    assert!(row.modified.is_some());
    assert_eq!(row.parent_dir, dir.path().display().to_string());
    let folder = row_from_path(&dir.path().display().to_string());
    assert_eq!(folder.file_type, "<DIR>");
    let gone = row_from_path("/nowhere/x.png");
    assert_eq!((gone.size, gone.file_type.as_str(), gone.modified.is_none()), (0, "png", true));
    let inner = row_from_path("zip:///a.zip!/in/pic.png");
    assert_eq!((inner.size, inner.filename.as_str()), (0, "pic.png"));
}

#[tokio::test]
async fn waiting_for_a_database_that_never_connects_times_out_and_obeys_cancel() {
    if crate::database::is_ready() {
        return;
    }
    let cancel = AtomicBool::new(false);
    assert!(!wait_for_connection(&cancel, Duration::from_millis(150)).await);
    cancel.store(true, Ordering::Relaxed);
    let started = Instant::now();
    assert!(!wait_for_connection(&cancel, Duration::from_secs(30)).await);
    assert!(started.elapsed() < Duration::from_secs(5));
}
