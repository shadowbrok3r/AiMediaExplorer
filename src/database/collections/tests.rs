use std::collections::HashSet;
use std::time::Instant;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::*;
use crate::database::snapshot::no_progress;
use crate::database::surql::{self, VerifyOptions, VerifyTarget};
use crate::database::tests::mem_db;

#[derive(Clone, Debug, SurrealValue)]
pub(crate) struct SeedRow {
    pub id: RecordId,
    pub filename: String,
    pub path: String,
    pub parent_dir: String,
    pub size: i64,
    pub file_type: String,
    pub tags: Vec<String>,
    pub logical_group: RecordId,
}

pub(crate) fn seed_row(i: usize, path: &str) -> SeedRow {
    SeedRow {
        id: RecordId::new("thumbnails", format!("t{i:08}")),
        filename: path.rsplit('/').next().unwrap_or(path).to_string(),
        path: path.to_string(),
        parent_dir: path.rsplit_once('/').map_or("", |(d, _)| d).to_string(),
        size: 100 + i as i64,
        file_type: "jpg".to_string(),
        tags: Vec::new(),
        logical_group: RecordId::new("logical_groups", "Default"),
    }
}

pub(crate) async fn insert_rows(db: &Surreal<Db>, rows: Vec<SeedRow>) {
    for chunk in rows.chunks(500) {
        db.query("INSERT INTO thumbnails $rows").bind(("rows", chunk.to_vec())).await.unwrap().check().unwrap();
    }
}

pub(crate) async fn db() -> Surreal<Db> {
    let db = mem_db().await;
    ensure_schema(&db).await.unwrap();
    db
}

fn paths(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("/{prefix}/dir{}/file{i:06}.jpg", i % 17)).collect()
}

async fn table_count(db: &Surreal<Db>, table: &str) -> u64 {
    surql::count_rows(db, table).await.unwrap()
}

#[test]
fn names_are_cleaned_keyed_in_lowercase_and_the_unfiled_name_is_reserved() {
    assert_eq!(normalize("  Summer   2024 ").unwrap(), ("Summer 2024".to_string(), "summer 2024".to_string()));
    assert_eq!(normalize("ÉCOLE").unwrap().1, "école");
    assert_eq!(normalize("a\tb").unwrap().0, "a b");
    assert_eq!(normalize("").unwrap_err(), CollectionError::Name(NameError::Empty));
    assert_eq!(normalize(" \n ").unwrap_err(), CollectionError::Name(NameError::Empty));
    assert_eq!(normalize("a\u{0}b").unwrap_err(), CollectionError::Name(NameError::Control));
    assert_eq!(normalize(&"x".repeat(65)).unwrap_err(), CollectionError::Name(NameError::TooLong));
    assert!(normalize(&"x".repeat(64)).is_ok());
    for reserved in ["Default", "default", "DEFAULT", "  Default  "] {
        assert_eq!(normalize(reserved).unwrap_err(), CollectionError::Reserved, "{reserved}");
    }
    assert!(normalize("Default 2").is_ok());
    let unusual = ["123", "a/b", "it's \"quoted\"", "`tick`", "日本語", "😀 fun", "a⟩b", "../x"];
    for name in unusual {
        let (display, key) = normalize(name).unwrap();
        assert_eq!(display, name);
        assert_eq!(key, name.to_lowercase());
    }
}

#[test]
fn membership_keys_are_stable_distinct_and_unambiguous() {
    assert_eq!(item_key("a", "/x"), item_key("a", "/x"));
    assert_eq!(item_key("a", "/x").len(), 32);
    assert_ne!(item_key("a", "/x"), item_key("a", "/y"));
    assert_ne!(item_key("a", "/x"), item_key("b", "/x"));
    assert_ne!(item_key("a", "bc"), item_key("ab", "c"));
    assert_eq!(item_id("a", "/x").table.as_str(), ITEMS);
    assert!(is_collection_id(&collection_id("k")));
    assert!(!is_collection_id(&RecordId::new("logical_groups", "k")));
    assert_eq!(key_of(&collection_id("héllo")), "héllo");
}

#[tokio::test]
async fn ensure_schema_is_idempotent_and_leaves_existing_data_alone() {
    let db = mem_db().await;
    insert_rows(&db, vec![seed_row(1, "/a/1.jpg")]).await;
    ensure_schema(&db).await.unwrap();
    let c = create(&db, "Keep").await.unwrap();
    add_paths(&db, &c.id, &["/a/1.jpg".to_string()]).await.unwrap();
    ensure_schema(&db).await.unwrap();
    ensure_schema(&db).await.unwrap();
    assert_eq!(table_count(&db, "thumbnails").await, 1);
    assert_eq!(count_members(&db, &c.id).await.unwrap(), 1);
    let info: surrealdb::types::Value = db.query("INFO FOR TABLE collection_items").await.unwrap().take(0).unwrap();
    let text = serde_json::to_string(&info.into_json_value()).unwrap();
    assert!(text.contains("collection_items_unique_idx") && text.contains("UNIQUE"), "{text}");
    assert!(text.contains("collection_items_path_idx") && text.contains("collection_items_collection_idx"), "{text}");
}

#[tokio::test]
async fn create_get_list_and_duplicate_names_ignoring_case() {
    let db = db().await;
    assert!(list(&db).await.unwrap().is_empty());
    let a = create(&db, "  Holiday   Trip ").await.unwrap();
    assert_eq!(a.name, "Holiday Trip");
    assert_eq!(a.key(), "holiday trip");
    assert!(a.created.is_some() && a.updated.is_some(), "timestamps default to now");
    assert_eq!(a.id, collection_id("holiday trip"));
    for dup in ["holiday trip", "HOLIDAY TRIP", "Holiday  Trip"] {
        let err = create(&db, dup).await.unwrap_err();
        assert!(matches!(err.downcast_ref::<CollectionError>(), Some(CollectionError::Exists(_))), "{dup}: {err:#}");
    }
    assert!(matches!(create(&db, "default").await.unwrap_err().downcast_ref::<CollectionError>(), Some(CollectionError::Reserved)));
    assert!(matches!(create(&db, "").await.unwrap_err().downcast_ref::<CollectionError>(), Some(CollectionError::Name(NameError::Empty))));
    let b = create(&db, "alpha").await.unwrap();
    let c = create(&db, "Zeta").await.unwrap();
    let names: Vec<String> = list(&db).await.unwrap().into_iter().map(|c| c.name).collect();
    assert_eq!(names, vec!["alpha", "Holiday Trip", "Zeta"], "ordered ignoring case");
    assert_eq!(get(&db, &b.id).await.unwrap().unwrap().name, "alpha");
    assert_eq!(get_by_name(&db, "ZETA").await.unwrap().unwrap().id, c.id);
    assert!(get_by_name(&db, "missing").await.unwrap().is_none());
    assert!(get_by_name(&db, "Default").await.unwrap().is_none());
    assert!(get_by_name(&db, "").await.unwrap().is_none());
    assert!(get(&db, &collection_id("nope")).await.unwrap().is_none());
    let (again, created) = get_or_create(&db, "ALPHA").await.unwrap();
    assert!(!created && again.id == b.id);
    let (fresh, created) = get_or_create(&db, "Fresh").await.unwrap();
    assert!(created && fresh.name == "Fresh");
}

#[tokio::test]
async fn unusual_names_round_trip_through_the_record_key() {
    let db = db().await;
    for name in ["123", "a/b", "it's \"quoted\"", "`tick`", "日本語", "😀 fun", "a⟩b", "../x", "x:y", "{brace}"] {
        let c = create(&db, name).await.unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(c.name, name);
        let again = get_by_name(&db, name).await.unwrap().unwrap_or_else(|| panic!("{name} not found"));
        assert_eq!(again.id, c.id);
        add_paths(&db, &c.id, &[format!("/p/{name}")]).await.unwrap();
        assert_eq!(members(&db, &c.id, 0, 10).await.unwrap(), vec![format!("/p/{name}")]);
    }
    assert_eq!(list(&db).await.unwrap().len(), 10);
}

#[tokio::test]
async fn set_meta_updates_color_and_note() {
    let db = db().await;
    let c = create(&db, "Tinted").await.unwrap();
    assert!(c.color.is_none() && c.note.is_none());
    let c2 = set_meta(&db, &c.id, Some("#aabbcc"), Some("a note")).await.unwrap();
    assert_eq!((c2.color.as_deref(), c2.note.as_deref()), (Some("#aabbcc"), Some("a note")));
    let c3 = set_meta(&db, &c.id, None, None).await.unwrap();
    assert!(c3.color.is_none() && c3.note.is_none());
    assert!(set_meta(&db, &collection_id("ghost"), None, None).await.is_err());
}

#[tokio::test]
async fn add_paths_is_idempotent_deduplicating_and_chunked() {
    let db = db().await;
    let c = create(&db, "Big").await.unwrap();
    let n = ADD_CHUNK * 2 + 37;
    let mut input = paths("big", n);
    input.push(input[0].clone());
    input.push(String::new());
    input.push(input[5].clone());
    let first = add_paths(&db, &c.id, &input).await.unwrap();
    assert_eq!(first, AddReport { requested: n + 3, added: n, already: 0, skipped_empty: 1 });
    assert_eq!(count_members(&db, &c.id).await.unwrap(), n as u64);
    let again = add_paths(&db, &c.id, &input).await.unwrap();
    assert_eq!((again.added, again.already), (0, n));
    assert_eq!(count_members(&db, &c.id).await.unwrap(), n as u64);
    let mut more = paths("big", 10);
    more.extend(paths("extra", 5));
    let third = add_paths(&db, &c.id, &more).await.unwrap();
    assert_eq!((third.added, third.already), (5, 10));
    assert_eq!(table_count(&db, ITEMS).await, n as u64 + 5);
    let none = add_paths(&db, &c.id, &[]).await.unwrap();
    assert_eq!(none, AddReport::default());
    let err = add_paths(&db, &collection_id("ghost"), &paths("x", 3)).await.unwrap_err();
    assert!(matches!(err.downcast_ref::<CollectionError>(), Some(CollectionError::NotFound(_))));
    assert_eq!(table_count(&db, ITEMS).await, n as u64 + 5, "nothing is written for a missing collection");
}

#[tokio::test]
async fn a_file_can_be_in_many_collections_and_each_membership_is_separate() {
    let db = db().await;
    let a = create(&db, "A").await.unwrap();
    let b = create(&db, "B").await.unwrap();
    let c = create(&db, "C").await.unwrap();
    let shared = "/pics/shared.jpg".to_string();
    let only_a = "/pics/a.jpg".to_string();
    add_paths(&db, &a.id, &[shared.clone(), only_a.clone()]).await.unwrap();
    add_paths(&db, &b.id, &[shared.clone()]).await.unwrap();
    add_paths(&db, &c.id, &[shared.clone()]).await.unwrap();
    let names: Vec<String> = collections_of(&db, &shared).await.unwrap().into_iter().map(|c| c.name).collect();
    assert_eq!(names, vec!["A", "B", "C"]);
    assert_eq!(collections_of(&db, &only_a).await.unwrap().len(), 1);
    assert!(collections_of(&db, "/pics/none.jpg").await.unwrap().is_empty());
    let counts = counts_for(&db, &[shared.clone(), only_a.clone(), "/pics/none.jpg".to_string()]).await.unwrap();
    assert_eq!(counts.get(&shared), Some(&3));
    assert_eq!(counts.get(&only_a), Some(&1));
    assert!(!counts.contains_key("/pics/none.jpg"));
    assert!(counts_for(&db, &[]).await.unwrap().is_empty());
    assert_eq!(remove_paths(&db, &b.id, &[shared.clone()]).await.unwrap(), 1);
    assert_eq!(counts_for(&db, &[shared.clone()]).await.unwrap().get(&shared), Some(&2));
    assert_eq!(count_members(&db, &a.id).await.unwrap(), 2);
}

#[tokio::test]
async fn remove_paths_counts_only_existing_memberships_and_leaves_others() {
    let db = db().await;
    let a = create(&db, "A").await.unwrap();
    let b = create(&db, "B").await.unwrap();
    let all = paths("r", ADD_CHUNK + 50);
    add_paths(&db, &a.id, &all).await.unwrap();
    add_paths(&db, &b.id, &all[..10]).await.unwrap();
    let mut ask = all[..ADD_CHUNK + 20].to_vec();
    ask.push("/never/added.jpg".to_string());
    ask.push(all[0].clone());
    ask.push(String::new());
    assert_eq!(remove_paths(&db, &a.id, &ask).await.unwrap(), ADD_CHUNK + 20);
    assert_eq!(count_members(&db, &a.id).await.unwrap(), 30);
    assert_eq!(count_members(&db, &b.id).await.unwrap(), 10, "other collections are untouched");
    assert_eq!(remove_paths(&db, &a.id, &ask).await.unwrap(), 0, "removing again is a no-op");
    assert_eq!(remove_paths(&db, &a.id, &[]).await.unwrap(), 0);
    assert_eq!(remove_paths(&db, &collection_id("ghost"), &all[..3]).await.unwrap(), 0);
}

#[tokio::test]
async fn members_page_in_path_order() {
    let db = db().await;
    let c = create(&db, "Paged").await.unwrap();
    let mut all: Vec<String> = (0..25).map(|i| format!("/m/{:02}.jpg", (i * 7) % 25)).collect();
    add_paths(&db, &c.id, &all).await.unwrap();
    all.sort();
    assert_eq!(members(&db, &c.id, 0, 10).await.unwrap(), all[..10]);
    assert_eq!(members(&db, &c.id, 10, 10).await.unwrap(), all[10..20]);
    assert_eq!(members(&db, &c.id, 20, 10).await.unwrap(), all[20..]);
    assert!(members(&db, &c.id, 25, 10).await.unwrap().is_empty());
    assert_eq!(count_members(&db, &c.id).await.unwrap(), 25);
    assert_eq!(count_members(&db, &collection_id("ghost")).await.unwrap(), 0);
}

#[tokio::test]
async fn member_rows_load_through_the_explorer_columns_and_count_unindexed_members() {
    let db = db().await;
    let c = create(&db, "Rows").await.unwrap();
    let all = paths("rows", 30);
    insert_rows(&db, all.iter().take(20).enumerate().map(|(i, p)| seed_row(i, p)).collect()).await;
    add_paths(&db, &c.id, &all).await.unwrap();
    let page = member_rows(&db, &c.id, 0, 100).await.unwrap();
    assert_eq!((page.requested, page.rows.len(), page.missing), (30, 20, 10));
    let sorted: Vec<&str> = page.rows.iter().map(|r| r.path.as_str()).collect();
    let mut expect: Vec<&str> = all.iter().take(20).map(String::as_str).collect();
    expect.sort();
    assert_eq!(sorted, expect, "rows follow member path order");
    assert!(page.rows.iter().all(|r| r.logical_group.table.as_str() == "logical_groups"));
    let everything = all_member_rows(&db, &c.id).await.unwrap();
    assert_eq!((everything.requested, everything.rows.len(), everything.missing), (30, 20, 10));
    assert_eq!(unindexed_count(&db, usize::MAX).await.unwrap(), 10);
    let empty = member_rows(&db, &collection_id("ghost"), 0, 10).await.unwrap();
    assert_eq!((empty.requested, empty.rows.len(), empty.missing), (0, 0, 0));
}

#[tokio::test]
async fn all_member_rows_pages_through_more_members_than_one_page() {
    let db = db().await;
    let c = create(&db, "Wide").await.unwrap();
    let n = PAGE + PAGE / 2;
    let all = paths("wide", n);
    insert_rows(&db, all.iter().enumerate().map(|(i, p)| seed_row(i, p)).collect()).await;
    add_paths(&db, &c.id, &all).await.unwrap();
    let rows = all_member_rows(&db, &c.id).await.unwrap();
    assert_eq!((rows.requested, rows.rows.len(), rows.missing), (n, n, 0));
    let distinct: HashSet<&str> = rows.rows.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(distinct.len(), n);
}

#[tokio::test]
async fn memberships_survive_thumbnail_row_deletion_and_work_for_unindexed_files() {
    let db = db().await;
    let c = create(&db, "Durable").await.unwrap();
    insert_rows(&db, vec![seed_row(1, "/d/one.jpg")]).await;
    add_paths(&db, &c.id, &["/d/one.jpg".to_string(), "/d/not-indexed.jpg".to_string()]).await.unwrap();
    assert_eq!(member_rows(&db, &c.id, 0, 10).await.unwrap().rows.len(), 1);
    db.query("DELETE thumbnails").await.unwrap().check().unwrap();
    assert_eq!(count_members(&db, &c.id).await.unwrap(), 2);
    assert_eq!(member_rows(&db, &c.id, 0, 10).await.unwrap().rows.len(), 0);
    insert_rows(&db, vec![seed_row(2, "/d/one.jpg")]).await;
    let back = member_rows(&db, &c.id, 0, 10).await.unwrap();
    assert_eq!((back.rows.len(), back.missing), (1, 1), "re-adding the row brings the membership back");
}

#[tokio::test]
async fn list_with_counts_uses_one_grouped_query_and_includes_empty_collections() {
    let db = db().await;
    let a = create(&db, "Aa").await.unwrap();
    let b = create(&db, "Bb").await.unwrap();
    let _empty = create(&db, "Cc").await.unwrap();
    add_paths(&db, &a.id, &paths("a", 7)).await.unwrap();
    add_paths(&db, &b.id, &paths("b", 3)).await.unwrap();
    let rows = list_with_counts(&db).await.unwrap();
    let got: Vec<(String, u64)> = rows.iter().map(|c| (c.collection.name.clone(), c.count)).collect();
    assert_eq!(got, vec![("Aa".to_string(), 7), ("Bb".to_string(), 3), ("Cc".to_string(), 0)]);
}

#[tokio::test]
async fn rename_moves_memberships_and_frees_the_old_name() {
    let db = db().await;
    let c = create(&db, "Old Name").await.unwrap();
    let other = create(&db, "Other").await.unwrap();
    let n = MOVE_CHUNK + 123;
    let all = paths("mv", n);
    add_paths(&db, &c.id, &all).await.unwrap();
    add_paths(&db, &other.id, &all[..5]).await.unwrap();
    let rep = rename(&db, &c.id, "New  Name").await.unwrap();
    assert_eq!(rep.moved, n as u64);
    assert_eq!(rep.collection.name, "New Name");
    assert_eq!(rep.collection.id, collection_id("new name"));
    assert!(get(&db, &c.id).await.unwrap().is_none());
    assert_eq!(count_members(&db, &rep.collection.id).await.unwrap(), n as u64);
    assert_eq!(count_members(&db, &other.id).await.unwrap(), 5);
    assert_eq!(table_count(&db, ITEMS).await, n as u64 + 5, "no membership is duplicated or lost");
    assert_eq!(orphan_count(&db).await.unwrap(), 0);
    let again = add_paths(&db, &rep.collection.id, &all[..10]).await.unwrap();
    assert_eq!((again.added, again.already), (0, 10), "moved memberships answer to the new key");
    assert_eq!(remove_paths(&db, &rep.collection.id, &all[..10]).await.unwrap(), 10);
    assert!(create(&db, "old name").await.is_ok(), "the old name is free again");
}

#[tokio::test]
async fn rename_to_a_taken_or_invalid_name_changes_nothing_and_case_changes_keep_the_id() {
    let db = db().await;
    let a = create(&db, "Alpha").await.unwrap();
    let b = create(&db, "Beta").await.unwrap();
    add_paths(&db, &a.id, &paths("a", 4)).await.unwrap();
    for bad in ["beta", "BETA", "", "Default"] {
        assert!(rename(&db, &a.id, bad).await.is_err(), "{bad}");
    }
    assert_eq!(count_members(&db, &a.id).await.unwrap(), 4);
    assert_eq!(count_members(&db, &b.id).await.unwrap(), 0);
    let cased = rename(&db, &a.id, "ALPHA").await.unwrap();
    assert_eq!((cased.moved, cased.collection.id.clone(), cased.collection.name.as_str()), (0, a.id.clone(), "ALPHA"));
    assert_eq!(count_members(&db, &a.id).await.unwrap(), 4);
    assert!(matches!(rename(&db, &collection_id("ghost"), "X").await.unwrap_err().downcast_ref::<CollectionError>(), Some(CollectionError::NotFound(_))));
    let keep = create_with(&db, NewCollection { name: "Meta".into(), color: Some("#fff".into()), note: Some("n".into()), legacy_key: Some("legacy-k".into()), ..Default::default() }).await.unwrap();
    let moved = rename(&db, &keep.id, "Meta Two").await.unwrap().collection;
    assert_eq!((moved.color.as_deref(), moved.note.as_deref(), moved.legacy_key.as_deref()), (Some("#fff"), Some("n"), Some("legacy-k")));
    assert_eq!(moved.created, keep.created, "the creation time is kept");
}

#[tokio::test]
async fn delete_removes_memberships_only() {
    let db = db().await;
    let a = create(&db, "Gone").await.unwrap();
    let b = create(&db, "Stays").await.unwrap();
    let all = paths("del", MOVE_CHUNK + 10);
    insert_rows(&db, all.iter().enumerate().map(|(i, p)| seed_row(i, p)).collect()).await;
    add_paths(&db, &a.id, &all).await.unwrap();
    add_paths(&db, &b.id, &all[..50]).await.unwrap();
    let removed = delete(&db, &a.id).await.unwrap();
    assert_eq!(removed, all.len() as u64);
    assert!(get(&db, &a.id).await.unwrap().is_none());
    assert_eq!(count_members(&db, &b.id).await.unwrap(), 50);
    assert_eq!(table_count(&db, "thumbnails").await, all.len() as u64, "files and thumbnails are untouched");
    assert_eq!(orphan_count(&db).await.unwrap(), 0);
    assert_eq!(delete(&db, &a.id).await.unwrap(), 0, "deleting twice is harmless");
    assert_eq!(counts_for(&db, &all[..1]).await.unwrap().get(&all[0]), Some(&1));
}

#[tokio::test]
async fn orphans_are_counted_and_pruned() {
    let db = db().await;
    let c = create(&db, "Real").await.unwrap();
    add_paths(&db, &c.id, &paths("o", 5)).await.unwrap();
    let ghost = collection_id("ghost");
    db.query("INSERT INTO collection_items $rows")
        .bind(("rows", (0..7).map(|i| ItemIn { id: item_id("ghost", &format!("/g/{i}")), collection: ghost.clone(), path: format!("/g/{i}") }).collect::<Vec<_>>()))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_eq!(orphan_count(&db).await.unwrap(), 7);
    assert_eq!(dangling(&db).await.unwrap().len(), 1);
    assert_eq!(prune_orphans(&db).await.unwrap(), 7);
    assert_eq!(orphan_count(&db).await.unwrap(), 0);
    assert_eq!(count_members(&db, &c.id).await.unwrap(), 5);
}

#[tokio::test]
async fn copy_members_adds_missing_paths_only() {
    let db = db().await;
    let a = create(&db, "Src").await.unwrap();
    let b = create(&db, "Dst").await.unwrap();
    let all = paths("cp", PAGE + 40);
    add_paths(&db, &a.id, &all).await.unwrap();
    add_paths(&db, &b.id, &all[..100]).await.unwrap();
    assert_eq!(copy_members(&db, &a.id, &b.id).await.unwrap(), all.len() - 100);
    assert_eq!(count_members(&db, &b.id).await.unwrap(), all.len() as u64);
    assert_eq!(count_members(&db, &a.id).await.unwrap(), all.len() as u64, "the source keeps its members");
    assert_eq!(copy_members(&db, &a.id, &b.id).await.unwrap(), 0);
    assert!(copy_members(&db, &a.id, &collection_id("ghost")).await.is_err());
}

#[tokio::test]
async fn unfiled_paths_lists_rows_in_no_collection() {
    let db = db().await;
    let c = create(&db, "Some").await.unwrap();
    let all = paths("uf", 30);
    insert_rows(&db, all.iter().enumerate().map(|(i, p)| seed_row(i, p)).collect()).await;
    add_paths(&db, &c.id, &all[..12]).await.unwrap();
    let mut got = unfiled_paths(&db, usize::MAX).await.unwrap();
    got.sort();
    let mut expect = all[12..].to_vec();
    expect.sort();
    assert_eq!(got, expect);
    assert_eq!(unfiled_paths(&db, 5).await.unwrap().len(), 5);
    assert!(unfiled_paths(&db, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn export_and_import_round_trip_carries_collections_memberships_and_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let src = db().await;
    let a = create_with(&src, NewCollection { name: "Trip".into(), color: Some("#123456".into()), note: Some("n".into()), legacy_key: Some("legacy".into()), ..Default::default() }).await.unwrap();
    let b = create(&src, "日本語 `x`").await.unwrap();
    let all = paths("rt", 120);
    add_paths(&src, &a.id, &all).await.unwrap();
    add_paths(&src, &b.id, &all[..30]).await.unwrap();
    let file = tmp.path().join("round.surql");
    let rep = surql::export_atomic(&src, &file, &no_progress()).await.unwrap();
    assert!(rep.bytes > 0);
    let text = std::fs::read_to_string(&file).unwrap();
    for needle in ["DEFINE TABLE collections", "DEFINE TABLE collection_items", "collection_items_unique_idx", "collections_legacy_key_idx"] {
        assert!(text.contains(needle), "export is missing {needle}");
    }
    let verify = surql::verify_export(&file, VerifyTarget::Live(&src), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(verify.ok, "{:?}", verify.problems);
    assert_eq!(verify.imported.tables["collections"].count, 2);
    assert_eq!(verify.imported.tables["collection_items"].count, 150);

    let dst = crate::database::tests::mem_db().await;
    dst.import(&file).await.unwrap();
    assert_eq!(count_members(&dst, &a.id).await.unwrap(), 120);
    assert_eq!(count_members(&dst, &b.id).await.unwrap(), 30);
    let back = get(&dst, &a.id).await.unwrap().unwrap();
    assert_eq!((back.name.as_str(), back.color.as_deref(), back.note.as_deref(), back.legacy_key.as_deref()), ("Trip", Some("#123456"), Some("n"), Some("legacy")));
    assert_eq!(get_by_name(&dst, "日本語 `X`").await.unwrap().unwrap().id, b.id);
    assert_eq!(add_paths(&dst, &a.id, &all[..10]).await.unwrap().added, 0, "ids survive the round trip, so adds stay idempotent");
    let deep = surql::build_manifest(&dst, &no_progress()).await.unwrap();
    assert_eq!(deep.tables["collection_items"].count, 150);
}

#[tokio::test]
async fn a_store_without_collection_tables_exports_and_a_later_ensure_schema_adds_them() {
    let tmp = tempfile::tempdir().unwrap();
    let old = mem_db().await;
    insert_rows(&old, vec![seed_row(1, "/o/1.jpg")]).await;
    let file = tmp.path().join("old.surql");
    surql::export_atomic(&old, &file, &no_progress()).await.unwrap();
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(!text.contains("collection_items"));
    let fresh = mem_db().await;
    fresh.import(&file).await.unwrap();
    ensure_schema(&fresh).await.unwrap();
    let c = create(&fresh, "After").await.unwrap();
    assert_eq!(add_paths(&fresh, &c.id, &["/o/1.jpg".to_string()]).await.unwrap().added, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_adds_of_the_same_paths_never_duplicate() {
    let db = db().await;
    let c = create(&db, "Race").await.unwrap();
    let all = paths("race", 600);
    let mut tasks = Vec::new();
    for _ in 0..4 {
        let (db, id, all) = (db.clone(), c.id.clone(), all.clone());
        tasks.push(tokio::spawn(async move { add_paths(&db, &id, &all).await.unwrap().added }));
    }
    let mut added = 0;
    for t in tasks {
        added += t.await.unwrap();
    }
    assert_eq!(added, 600, "each path is reported as added exactly once");
    assert_eq!(count_members(&db, &c.id).await.unwrap(), 600);
    assert_eq!(table_count(&db, ITEMS).await, 600);
}

#[tokio::test]
async fn probe_runs_every_step_and_leaves_nothing_behind() {
    let db = db().await;
    let steps = probe(&db, 300).await.unwrap();
    let names: Vec<&str> = steps.iter().map(|s| s.step.as_str()).collect();
    assert_eq!(names, ["create", "add_paths", "add_paths again", "list_with_counts", "count_members", "members page", "collections_of", "counts_for", "remove_paths", "delete"]);
    assert!(list(&db).await.unwrap().is_empty());
    assert_eq!(table_count(&db, ITEMS).await, 0);
    let stale = create(&db, &format!("{PROBE_PREFIX}1")).await.unwrap();
    add_paths(&db, &stale.id, &paths("s", 3)).await.unwrap();
    probe(&db, 10).await.unwrap();
    assert!(list(&db).await.unwrap().is_empty(), "a probe left by a crashed run is cleaned up");
}

async fn fifty_thousand(db: &Surreal<Db>, backend: &str) {
    let db = db.clone();
    let a = create(&db, "Fifty K").await.unwrap();
    let b = create(&db, "Other").await.unwrap();
    let n = 50_000;
    let all = paths("fifty", n);
    let mut report = Vec::new();
    macro_rules! timed {
        ($label:expr, $e:expr) => {{
            let t = Instant::now();
            let out = $e;
            report.push(format!("{:<28}{:>8} ms", $label, t.elapsed().as_millis()));
            out
        }};
    }
    let added = timed!("add_paths 50k", add_paths(&db, &a.id, &all).await.unwrap());
    assert_eq!(added.added, n);
    let again = timed!("add_paths 50k again", add_paths(&db, &a.id, &all).await.unwrap());
    assert_eq!((again.added, again.already), (0, n));
    timed!("add_paths 5k to second", add_paths(&db, &b.id, &all[..5_000]).await.unwrap());
    assert_eq!(timed!("count_members", count_members(&db, &a.id).await.unwrap()), n as u64);
    let counts = timed!("list_with_counts", list_with_counts(&db).await.unwrap());
    assert_eq!(counts.iter().map(|c| c.count).collect::<Vec<_>>(), vec![n as u64, 5_000]);
    insert_rows(&db, all[..5_000].iter().enumerate().map(|(i, p)| seed_row(i, p)).collect()).await;
    let loaded = timed!("all_member_rows 5k", all_member_rows(&db, &b.id).await.unwrap());
    assert_eq!((loaded.rows.len(), loaded.missing), (5_000, 0));
    let page = timed!("members page of 500", members(&db, &a.id, 25_000, 500).await.unwrap());
    assert_eq!(page.len(), 500);
    let by_path = timed!("counts_for 5k", counts_for(&db, &all[..5_000]).await.unwrap());
    assert_eq!(by_path.len(), 5_000);
    assert_eq!(timed!("collections_of one path", collections_of(&db, &all[7]).await.unwrap()).len(), 2);
    assert_eq!(timed!("remove_paths 25k", remove_paths(&db, &a.id, &all[..25_000]).await.unwrap()), 25_000);
    let renamed = timed!("rename 25k members", rename(&db, &a.id, "Fifty K renamed").await.unwrap());
    assert_eq!(renamed.moved, 25_000);
    assert_eq!(timed!("copy_members 25k", copy_members(&db, &renamed.collection.id, &b.id).await.unwrap()), 25_000 - 0);
    assert_eq!(timed!("delete 25k", delete(&db, &renamed.collection.id).await.unwrap()), 25_000);
    assert_eq!(table_count(&db, ITEMS).await, 30_000);
    println!("collections 50k timings ({backend}):\n{}", report.join("\n"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fifty_thousand_memberships_scale_and_report_timings() {
    fifty_thousand(&db().await, "kv-mem").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "timing probe on a SurrealKV store; run with --ignored --nocapture, COLLECTIONS_PROBE_DIR picks the folder"]
async fn fifty_thousand_memberships_on_a_disk_store_report_timings() {
    use surrealdb::engine::local::SurrealKv;
    let dir = match std::env::var_os("COLLECTIONS_PROBE_DIR") {
        Some(d) => tempfile::tempdir_in(d).unwrap(),
        None => tempfile::tempdir().unwrap(),
    };
    let store = dir.path().join("store");
    let disk = Surreal::new::<SurrealKv>(store.to_string_lossy().as_ref()).await.unwrap();
    disk.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&disk).await.unwrap();
    ensure_schema(&disk).await.unwrap();
    fifty_thousand(&disk, "surrealkv").await;
}
