use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::*;
use crate::database::tests::mem_db;

fn seed(path: &str) -> RowSeed {
    RowSeed {
        path: path.to_string(),
        filename: path.rsplit('/').next().unwrap_or(path).to_string(),
        file_type: "jpg".to_string(),
        size: 1234,
        parent_dir: path.rsplit_once('/').map_or("", |(d, _)| d).to_string(),
        hash: None,
        modified: None,
    }
}

fn tags(list: &[&str]) -> Vec<String> {
    list.iter().map(|t| t.to_string()).collect()
}

async fn row(db: &Surreal<Db>, path: &str) -> Option<Thumbnail> {
    db.query("SELECT id, db_created, path, filename, file_type, size, description, caption, tags, category, thumbnail_b64, modified, hash, parent_dir, logical_group FROM thumbnails WHERE path = $p")
        .bind(("p", path.to_string()))
        .await
        .unwrap()
        .take::<Vec<Thumbnail>>(0)
        .unwrap()
        .into_iter()
        .next()
}

async fn count(db: &Surreal<Db>) -> usize {
    db.query("SELECT VALUE path FROM thumbnails").await.unwrap().take::<Vec<String>>(0).unwrap().len()
}

fn patch(description: Option<&str>, caption: Option<&str>, category: Option<&str>, tags: Option<&[&str]>) -> MetaPatch {
    let text = |v: Option<&str>| Some(v.map(str::to_string));
    MetaPatch { description: text(description), caption: text(caption), category: text(category), tags: tags.map(|t| self::tags(t)) }
}

#[test]
fn tags_are_trimmed_collapsed_and_deduplicated_ignoring_case() {
    assert_eq!(normalize_tag("  sun   set "), Some("sun set".to_string()));
    assert_eq!(normalize_tag("   "), None);
    assert_eq!(normalize_tags(["Cat", "cat", " CAT ", "dog", ""]), tags(&["Cat", "dog"]));
}

#[test]
fn a_tag_change_removes_ignoring_case_then_adds_without_repeats() {
    let existing = tags(&["Sunset", "beach", "dog"]);
    let change = TagChange { add: tags(&["BEACH", "night"]), remove: tags(&["sunset"]) };
    assert_eq!(apply_tag_change(&existing, &change), tags(&["beach", "dog", "night"]));
    assert_eq!(apply_tag_change(&existing, &TagChange::default()), existing);
    assert_eq!(apply_tag_change(&[], &TagChange { add: tags(&["a", "A", " b "]), remove: vec![] }), tags(&["a", "b"]));
    let both = TagChange { add: tags(&["x"]), remove: tags(&["x"]) };
    assert_eq!(apply_tag_change(&tags(&["x"]), &both), tags(&["x"]), "a tag removed and added in one change ends up present");
}

#[test]
fn a_patch_reports_the_fields_it_writes_and_cleans_blank_text() {
    let p = patch(Some("  d  "), Some("   "), None, Some(&["a", "A", " b"]));
    assert_eq!(p.fields(), vec!["description", "caption", "category", "tags"]);
    let c = p.cleaned();
    assert_eq!(c.description, Some(Some("d".to_string())));
    assert_eq!(c.caption, Some(None), "blank text clears the field");
    assert_eq!(c.category, Some(None));
    assert_eq!(c.tags, Some(tags(&["a", "b"])));
    assert!(MetaPatch::default().is_empty());
    assert!(MetaPatch::default().fields().is_empty());
}

#[tokio::test]
async fn the_first_edit_of_a_file_without_a_row_creates_it() {
    let db = mem_db().await;
    let s = seed("/pics/new.jpg");
    assert!(row(&db, &s.path).await.is_none());
    let out = save_patch(&db, &s, &patch(Some("a cat"), Some("Cat"), Some("pets"), Some(&["cat", "pet"]))).await.unwrap();
    assert!(out.created);
    let r = row(&db, &s.path).await.expect("the row exists");
    assert_eq!((r.description.as_deref(), r.caption.as_deref(), r.category.as_deref()), (Some("a cat"), Some("Cat"), Some("pets")));
    assert_eq!(r.tags, tags(&["cat", "pet"]));
    assert_eq!((r.filename.as_str(), r.file_type.as_str(), r.size, r.parent_dir.as_str()), ("new.jpg", "jpg", 1234, "/pics"));
    assert_eq!(count(&db).await, 1);
}

#[tokio::test]
async fn a_second_save_updates_the_same_row_and_can_clear_fields() {
    let db = mem_db().await;
    let s = seed("/pics/a.jpg");
    save_patch(&db, &s, &patch(Some("d"), Some("c"), Some("k"), Some(&["t"]))).await.unwrap();
    let out = save_patch(&db, &s, &patch(None, None, None, None)).await.unwrap();
    assert!(!out.created);
    let r = row(&db, &s.path).await.unwrap();
    assert_eq!((r.description, r.caption, r.category), (None, None, None), "cleared fields are removed, unlike the merge-only upsert");
    assert_eq!(r.tags, tags(&["t"]), "fields outside the patch are untouched");
    save_patch(&db, &s, &MetaPatch { tags: Some(vec![]), ..Default::default() }).await.unwrap();
    assert!(row(&db, &s.path).await.unwrap().tags.is_empty());
    assert_eq!(count(&db).await, 1);
}

#[tokio::test]
async fn a_patch_touches_only_its_own_fields() {
    let db = mem_db().await;
    let s = seed("/pics/a.jpg");
    save_patch(&db, &s, &patch(Some("keep"), Some("keep"), Some("keep"), Some(&["keep"]))).await.unwrap();
    save_patch(&db, &s, &MetaPatch { caption: Some(Some("new".into())), ..Default::default() }).await.unwrap();
    let r = row(&db, &s.path).await.unwrap();
    assert_eq!((r.description.as_deref(), r.caption.as_deref(), r.category.as_deref()), (Some("keep"), Some("new"), Some("keep")));
    assert_eq!(r.tags, tags(&["keep"]));
}

#[tokio::test]
async fn an_empty_patch_still_creates_the_row_and_an_empty_path_is_refused() {
    let db = mem_db().await;
    let out = save_patch(&db, &seed("/pics/b.jpg"), &MetaPatch::default()).await.unwrap();
    assert!(out.created);
    assert_eq!(count(&db).await, 1);
    assert!(save_patch(&db, &seed(""), &MetaPatch::default()).await.is_err());
}

#[tokio::test]
async fn batch_tags_create_missing_rows_and_report_the_stored_tags() {
    let db = mem_db().await;
    let (a, b, c) = (seed("/p/a.jpg"), seed("/p/b.jpg"), seed("/p/c.jpg"));
    save_patch(&db, &a, &MetaPatch { tags: Some(tags(&["Sunset", "beach"])), ..Default::default() }).await.unwrap();
    save_patch(&db, &b, &MetaPatch { tags: Some(tags(&["dog"])), ..Default::default() }).await.unwrap();
    let change = TagChange { add: tags(&["night"]), remove: tags(&["sunset", "missing"]) };
    let report = batch_tags(&db, &[a.clone(), b.clone(), c.clone(), a.clone()], &change).await.unwrap();
    assert_eq!(report.created, 1, "only c.jpg had no row");
    assert_eq!(report.updated, 3);
    assert_eq!(report.tags["/p/a.jpg"], tags(&["beach", "night"]));
    assert_eq!(report.tags["/p/b.jpg"], tags(&["dog", "night"]));
    assert_eq!(report.tags["/p/c.jpg"], tags(&["night"]));
    assert_eq!(row(&db, "/p/a.jpg").await.unwrap().tags, tags(&["beach", "night"]));
    assert_eq!(row(&db, "/p/c.jpg").await.unwrap().tags, tags(&["night"]));
    assert_eq!(count(&db).await, 3);
}

#[tokio::test]
async fn a_batch_that_changes_nothing_writes_nothing() {
    let db = mem_db().await;
    let a = seed("/p/a.jpg");
    save_patch(&db, &a, &MetaPatch { tags: Some(tags(&["x"])), ..Default::default() }).await.unwrap();
    let report = batch_tags(&db, &[a.clone()], &TagChange { add: tags(&["X"]), remove: tags(&["nope"]) }).await.unwrap();
    assert_eq!((report.created, report.updated), (0, 0));
    assert!(report.tags.is_empty());
    assert!(batch_tags(&db, &[], &TagChange::default()).await.is_err(), "an empty batch is an error, never everything");
}

#[tokio::test]
async fn batch_category_sets_and_clears_and_creates_rows() {
    let db = mem_db().await;
    let (a, b) = (seed("/p/a.jpg"), seed("/p/b.jpg"));
    save_patch(&db, &a, &MetaPatch { category: Some(Some("old".into())), ..Default::default() }).await.unwrap();
    let set = batch_category(&db, &[a.clone(), b.clone()], Some("  landscape ")).await.unwrap();
    assert_eq!((set.created, set.updated), (1, 2));
    assert_eq!(row(&db, "/p/a.jpg").await.unwrap().category.as_deref(), Some("landscape"));
    assert_eq!(row(&db, "/p/b.jpg").await.unwrap().category.as_deref(), Some("landscape"));
    let clear = batch_category(&db, &[a.clone()], Some("  ")).await.unwrap();
    assert_eq!(clear.updated, 1);
    assert_eq!(row(&db, "/p/a.jpg").await.unwrap().category, None);
    assert_eq!(row(&db, "/p/b.jpg").await.unwrap().category.as_deref(), Some("landscape"));
}

#[tokio::test]
async fn library_facts_tell_whether_a_file_is_indexed_and_described() {
    let db = mem_db().await;
    let a = seed("/p/a.jpg");
    assert_eq!(library_facts(&db, &a.path).await.unwrap(), LibraryFacts::default());
    save_patch(&db, &a, &patch(Some("text"), None, None, None)).await.unwrap();
    let facts = library_facts(&db, &a.path).await.unwrap();
    assert!(facts.indexed && facts.description);
    save_patch(&db, &a, &patch(None, None, None, None)).await.unwrap();
    assert!(!library_facts(&db, &a.path).await.unwrap().description);
}

#[tokio::test]
async fn rows_are_seeded_from_explorer_rows() {
    let mut t = Thumbnail::new("x.png");
    t.path = "/pics/sub/x.png".into();
    t.file_type = "png".into();
    t.size = 9;
    let s = RowSeed::from_row(&t);
    assert_eq!((s.parent_dir.as_str(), s.filename.as_str(), s.size), ("/pics/sub", "x.png", 9));
    t.parent_dir = "/set/explicitly".into();
    assert_eq!(RowSeed::from_row(&t).parent_dir, "/set/explicitly");
}

#[tokio::test]
async fn suggestion_lists_hold_each_tag_and_category_once_in_name_order() {
    let db = mem_db().await;
    save_patch(&db, &seed("/p/a.jpg"), &MetaPatch { tags: Some(tags(&["beach", "Sunset"])), category: Some(Some("travel".into())), ..Default::default() }).await.unwrap();
    save_patch(&db, &seed("/p/b.jpg"), &MetaPatch { tags: Some(tags(&["sunset", "dog"])), category: Some(Some("travel".into())), ..Default::default() }).await.unwrap();
    save_patch(&db, &seed("/p/c.jpg"), &MetaPatch { category: Some(Some("pets".into())), ..Default::default() }).await.unwrap();
    assert_eq!(all_tags(&db).await.unwrap(), tags(&["beach", "dog", "Sunset"]));
    assert_eq!(all_categories(&db).await.unwrap(), tags(&["pets", "travel"]));
}
