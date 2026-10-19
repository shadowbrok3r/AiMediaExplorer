use std::collections::HashMap;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::*;
use crate::database::collections;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed, TagChange};
use crate::database::tests::mem_db;

async fn db() -> Surreal<Db> {
    let db = mem_db().await;
    collections::ensure_schema(&db).await.unwrap();
    db
}

fn seed(path: &str) -> RowSeed {
    RowSeed {
        path: path.to_string(),
        filename: path.rsplit('/').next().unwrap_or(path).to_string(),
        file_type: "jpg".to_string(),
        size: 10,
        parent_dir: path.rsplit_once('/').map_or("", |(d, _)| d).to_string(),
        hash: None,
        modified: None,
    }
}

fn tags(list: &[&str]) -> Vec<String> {
    list.iter().map(|t| t.to_string()).collect()
}

fn caption(text: &str) -> MetaPatch {
    MetaPatch { caption: Some(Some(text.to_string())), ..Default::default() }
}

#[derive(Debug, SurrealValue)]
struct Meta {
    caption: Option<String>,
    category: Option<String>,
    tags: Option<Vec<String>>,
}

async fn meta(db: &Surreal<Db>, path: &str) -> Meta {
    db.query("SELECT caption, category, tags FROM thumbnails WHERE path = $p").bind(("p", path.to_string())).await.unwrap().take::<Vec<Meta>>(0).unwrap().into_iter().next().expect("the row exists")
}

async fn set_raw(db: &Surreal<Db>, path: &str, caption: &str) {
    db.query("UPDATE thumbnails SET caption = $c WHERE path = $p").bind(("c", caption.to_string())).bind(("p", path.to_string())).await.unwrap().check().unwrap();
}

fn entry(seq: i64, batch: i64, path: &str, field: Field, before: Value, after: Value) -> Entry {
    Entry { seq, batch: BatchId(batch), at_ms: 0, source: "test".into(), path: path.into(), field, target: None, before, after, state: EntryState::Applied }
}

fn text(v: &str) -> Value {
    Value::text(Some(v))
}

#[test]
fn a_repeated_change_keeps_the_first_before_and_the_last_after() {
    let batch = begin_batch("collapse");
    record(batch, "/a.jpg", Field::Caption, text("one"), text("two"));
    record(batch, "/a.jpg", Field::Caption, text("two"), text("three"));
    record(batch, "/a.jpg", Field::Tags, Value::tags(&tags(&["x"])), Value::tags(&tags(&["y"])));
    assert_eq!(pending_len(batch), 2);
    record(batch, "/b.jpg", Field::Category, Value::text(None), Value::text(None));
    assert_eq!(pending_len(batch), 2, "a change that ends where it began is dropped");
    record(batch, "/a.jpg", Field::Caption, text("three"), text("one"));
    assert_eq!(pending_len(batch), 1, "a change that returns to the start is dropped");
    abandon(batch);
    assert_eq!(pending_len(batch), 0);
}

#[test]
fn values_survive_the_json_round_trip_and_blank_text_is_none() {
    for (field, value) in [
        (Field::Caption, text("a \"quoted\" line\nbreak")),
        (Field::Caption, Value::Text(None)),
        (Field::Tags, Value::tags(&tags(&["a", "b c"]))),
        (Field::Tags, Value::Tags(Vec::new())),
        (Field::Collection, Value::Member(true)),
        (Field::Collection, Value::Member(false)),
    ] {
        assert_eq!(Value::from_json(field, &value.to_json()), value, "{field:?}");
    }
    assert_eq!(Value::text(Some("   ")), Value::Text(None));
    assert_eq!(Field::parse("tags"), Some(Field::Tags));
    assert_eq!(Field::parse("nope"), None);
    assert!(Field::Caption.is_text() && !Field::Tags.is_text());
}

#[test]
fn the_plan_unwinds_repeated_entries_in_reverse_and_flags_changed_rows() {
    let entries = vec![entry(1, 1, "/a", Field::Caption, text("0"), text("1")), entry(2, 1, "/a", Field::Caption, text("1"), text("2")), entry(3, 1, "/b", Field::Caption, text("x"), text("y"))];
    let mut current: HashMap<Key, Value> = HashMap::new();
    current.insert(("/a".into(), Field::Caption, None), text("2"));
    current.insert(("/b".into(), Field::Caption, None), text("edited since"));
    let undo = plan(&entries, Direction::Undo, &current);
    assert_eq!(undo.writes.iter().map(|w| (w.path.as_str(), w.value.clone())).collect::<Vec<_>>(), vec![("/a", text("1")), ("/a", text("0"))]);
    assert_eq!(undo.skips.len(), 1);
    assert!(undo.skips[0].conflict.reason.contains("edited since"), "{:?}", undo.skips[0]);
    let missing = plan(&entries[..1], Direction::Undo, &HashMap::new());
    assert!(missing.writes.is_empty() && missing.skips[0].conflict.reason.contains("not in the library"));
    let mut redo_from = HashMap::new();
    redo_from.insert(("/a".to_string(), Field::Caption, None), text("0"));
    let redo = plan(&entries[..2], Direction::Redo, &redo_from);
    assert_eq!(redo.writes.iter().map(|w| w.value.clone()).collect::<Vec<_>>(), vec![text("1"), text("2")]);
}

#[test]
fn the_summary_counts_changes_files_and_skips() {
    let mut report = UndoReport { direction: Direction::Undo, batch: Some(BatchId(1)), source: "Add tags".into(), applied: Vec::new(), conflicts: Vec::new() };
    assert_eq!(UndoReport::empty(Direction::Undo).summary(), "Nothing to undo");
    assert_eq!(UndoReport::empty(Direction::Redo).summary(), "Nothing to redo");
    for p in ["/a", "/b", "/b"] {
        report.applied.push(Change { path: p.into(), field: Field::Tags, target: None, value: Value::tags(&tags(&["x"])) });
    }
    assert_eq!(report.summary(), "Undid 3 changes in 2 files (Add tags)");
    report.conflicts.push(Conflict { path: "/c".into(), field: Field::Tags, target: None, reason: "r".into() });
    assert_eq!(report.summary(), "Undid 3 changes in 2 files (Add tags); 1 skipped because the file changed since");
    report.applied.truncate(1);
    report.direction = Direction::Redo;
    assert!(report.summary().starts_with("Redid 1 change ("));
    let patches = report.patches();
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].1.tags, Some(tags(&["x"])));
}

#[tokio::test]
async fn an_inspector_edit_is_journaled_and_undo_and_redo_restore_the_row() {
    let db = db().await;
    let s = seed("/p/a.jpg");
    metadata_edit::save_patch(&db, &s, &MetaPatch { caption: Some(Some("first".into())), tags: Some(tags(&["sun"])), ..Default::default() }).await.unwrap();
    metadata_edit::save_patch(&db, &s, &caption("second")).await.unwrap();
    let entries = recent_with(&db, 10).await.unwrap();
    assert_eq!(entries.len(), 3, "two fields in the first edit, one in the second");
    assert_eq!(entries[0].field, Field::Caption);
    assert_eq!((entries[0].before.clone(), entries[0].after.clone()), (text("first"), text("second")));
    assert_eq!(entries[0].source, "Edit metadata");

    let report = undo_last_with(&db).await.unwrap();
    assert_eq!((report.applied.len(), report.conflicts.len(), report.is_empty()), (1, 0, false));
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("first"));
    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.applied.len(), 2);
    let row = meta(&db, "/p/a.jpg").await;
    assert_eq!((row.caption, row.tags.unwrap_or_default()), (None, Vec::<String>::new()));
    assert!(undo_last_with(&db).await.unwrap().is_empty(), "nothing is left to undo");

    let report = redo_last_with(&db).await.unwrap();
    assert_eq!(report.applied.len(), 2);
    assert_eq!(meta(&db, "/p/a.jpg").await.tags, Some(tags(&["sun"])));
    redo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("second"));
    assert!(redo_last_with(&db).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_new_edit_clears_what_could_be_redone() {
    let db = db().await;
    let s = seed("/p/a.jpg");
    metadata_edit::save_patch(&db, &s, &caption("one")).await.unwrap();
    metadata_edit::save_patch(&db, &s, &caption("two")).await.unwrap();
    undo_last_with(&db).await.unwrap();
    assert_eq!(stats_with(&db).await.unwrap().undone, 1);
    metadata_edit::save_patch(&db, &s, &caption("three")).await.unwrap();
    let stats = stats_with(&db).await.unwrap();
    assert_eq!((stats.undone, stats.applied), (0, 2));
    assert!(redo_last_with(&db).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_row_edited_since_is_skipped_and_never_overwritten() {
    let db = db().await;
    let (a, b) = (seed("/p/a.jpg"), seed("/p/b.jpg"));
    metadata_edit::batch_tags(&db, &[a.clone(), b.clone()], &TagChange { add: tags(&["new"]), remove: Vec::new() }).await.unwrap();
    db.query("UPDATE thumbnails SET tags = ['someone', 'else'] WHERE path = '/p/b.jpg'").await.unwrap().check().unwrap();
    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.applied.len(), 1);
    assert_eq!(report.conflicts.len(), 1);
    assert_eq!(report.conflicts[0].path, "/p/b.jpg");
    assert!(report.conflicts[0].reason.contains("someone, else"), "{}", report.conflicts[0].reason);
    assert_eq!(meta(&db, "/p/b.jpg").await.tags, Some(tags(&["someone", "else"])), "the newer edit stays");
    assert_eq!(meta(&db, "/p/a.jpg").await.tags.unwrap_or_default(), Vec::<String>::new());
    assert!(report.summary().contains("1 skipped"));
    let stats = stats_with(&db).await.unwrap();
    assert_eq!((stats.undone, stats.skipped, stats.applied), (1, 1, 0));
    assert!(undo_last_with(&db).await.unwrap().is_empty(), "a skipped entry does not block the stack");
}

#[tokio::test]
async fn a_redo_checks_the_row_too() {
    let db = db().await;
    let s = seed("/p/a.jpg");
    metadata_edit::save_patch(&db, &s, &caption("one")).await.unwrap();
    undo_last_with(&db).await.unwrap();
    set_raw(&db, "/p/a.jpg", "typed meanwhile").await;
    let report = redo_last_with(&db).await.unwrap();
    assert_eq!((report.applied.len(), report.conflicts.len()), (0, 1));
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("typed meanwhile"));
}

#[tokio::test]
async fn undo_takes_the_newest_batch_first_and_redo_the_one_undone_last() {
    let db = db().await;
    let s = seed("/p/a.jpg");
    for text in ["one", "two", "three"] {
        metadata_edit::save_patch(&db, &s, &caption(text)).await.unwrap();
    }
    undo_last_with(&db).await.unwrap();
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("one"));
    redo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("two"));
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("one"));
    let batches = batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.len(), 3);
    assert!(batches[0].batch > batches[1].batch, "newest first");
}

#[tokio::test]
async fn batch_tags_and_categories_are_one_batch_each_with_an_entry_per_file() {
    let db = db().await;
    let files: Vec<RowSeed> = ["/p/a.jpg", "/p/b.jpg", "/p/c.jpg"].into_iter().map(seed).collect();
    metadata_edit::batch_category(&db, &files, Some("travel")).await.unwrap();
    metadata_edit::batch_tags(&db, &files, &TagChange { add: tags(&["x", "y"]), remove: Vec::new() }).await.unwrap();
    let batches = batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.iter().map(|b| (b.source.as_str(), b.entries, b.files)).collect::<Vec<_>>(), vec![("Add tags", 3, 3), ("Set category", 3, 3)]);
    undo_last_with(&db).await.unwrap();
    for f in &files {
        assert_eq!(meta(&db, &f.path).await.tags.unwrap_or_default(), Vec::<String>::new());
        assert_eq!(meta(&db, &f.path).await.category.as_deref(), Some("travel"));
    }
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.category, None);
    redo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/c.jpg").await.category.as_deref(), Some("travel"));
}

#[tokio::test]
async fn an_edit_that_changes_nothing_leaves_no_entry() {
    let db = db().await;
    let s = seed("/p/a.jpg");
    metadata_edit::save_patch(&db, &s, &caption("same")).await.unwrap();
    metadata_edit::save_patch(&db, &s, &caption("same")).await.unwrap();
    metadata_edit::batch_tags(&db, &[s.clone()], &TagChange::default()).await.unwrap();
    assert_eq!(stats_with(&db).await.unwrap().entries, 1);
}

#[tokio::test]
async fn tag_and_category_rewrites_are_journaled_and_can_be_undone() {
    let db = db().await;
    let files: Vec<RowSeed> = ["/p/a.jpg", "/p/b.jpg"].into_iter().map(seed).collect();
    metadata_edit::batch_tags(&db, &files, &TagChange { add: tags(&["sun", "sea"]), remove: Vec::new() }).await.unwrap();
    metadata_edit::batch_category(&db, &files, Some("beach")).await.unwrap();

    assert_eq!(rename_tag_with(&db, "sun", "sunny").await.unwrap(), 2);
    assert_eq!(meta(&db, "/p/a.jpg").await.tags.map(|t| t.contains(&"sunny".to_string())), Some(true));
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.tags, Some(tags(&["sun", "sea"])));

    assert_eq!(delete_tag_with(&db, "sea").await.unwrap(), 2);
    assert_eq!(meta(&db, "/p/b.jpg").await.tags, Some(tags(&["sun"])));
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/b.jpg").await.tags, Some(tags(&["sun", "sea"])));

    assert_eq!(rename_category_with(&db, "beach", "coast").await.unwrap(), 2);
    assert_eq!(meta(&db, "/p/a.jpg").await.category.as_deref(), Some("coast"));
    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.source, "Rename category \"beach\" to \"coast\"");
    assert_eq!(meta(&db, "/p/a.jpg").await.category.as_deref(), Some("beach"));

    assert_eq!(prune_tags_with(&db, 1).await.unwrap(), 2);
    assert_eq!(meta(&db, "/p/a.jpg").await.tags.map(|t| t.len()), Some(1));
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/a.jpg").await.tags.map(|t| t.len()), Some(2));
}

async fn is_member(db: &Surreal<Db>, collection: &RecordId, path: &str) -> bool {
    collections::members(db, collection, 0, 100).await.unwrap().iter().any(|p| p == path)
}

#[tokio::test]
async fn collection_changes_are_journaled_and_undone() {
    let db = db().await;
    let c = collections::create(&db, "Holidays").await.unwrap();
    let paths = tags(&["/p/a.jpg", "/p/b.jpg"]);
    add_to_collection_with(&db, &c.id, &paths[..1], "Add to \"Holidays\"").await.unwrap();
    let report = add_to_collection_with(&db, &c.id, &paths, "Add to \"Holidays\"").await.unwrap();
    assert_eq!((report.added, report.already), (1, 1));
    let entries = recent_with(&db, 10).await.unwrap();
    assert_eq!(entries.len(), 2, "the file that was already a member has no entry");
    assert!(entries.iter().all(|e| e.field == Field::Collection && e.target.as_deref() == Some(c.key().as_str())));

    undo_last_with(&db).await.unwrap();
    assert!(!is_member(&db, &c.id, "/p/b.jpg").await && is_member(&db, &c.id, "/p/a.jpg").await);
    redo_last_with(&db).await.unwrap();
    assert!(is_member(&db, &c.id, "/p/b.jpg").await);

    assert_eq!(remove_from_collection_with(&db, &c.id, &paths, "Remove").await.unwrap(), 2);
    assert!(!is_member(&db, &c.id, "/p/a.jpg").await);
    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.membership_paths().len(), 2);
    assert!(is_member(&db, &c.id, "/p/a.jpg").await && is_member(&db, &c.id, "/p/b.jpg").await);

    collections::delete(&db, &c.id).await.unwrap();
    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.applied.len() + report.conflicts.len(), 1, "an entry of a deleted collection is applied or reported, never lost");
}

#[tokio::test]
async fn copying_a_collection_is_one_journaled_batch() {
    let db = db().await;
    let (from, to) = (collections::create(&db, "From").await.unwrap(), collections::create(&db, "To").await.unwrap());
    collections::add_paths(&db, &from.id, &tags(&["/a", "/b", "/c"])).await.unwrap();
    collections::add_paths(&db, &to.id, &tags(&["/a"])).await.unwrap();
    assert_eq!(copy_collection_with(&db, &from.id, &to.id, "Copy").await.unwrap(), 2);
    assert_eq!(recent_with(&db, 10).await.unwrap().len(), 2);
    undo_last_with(&db).await.unwrap();
    assert_eq!(collections::count_members(&db, &to.id).await.unwrap(), 1);
}

#[tokio::test]
async fn the_oldest_whole_batches_are_dropped_beyond_the_cap() {
    let db = db().await;
    for i in 0..5 {
        let rec = Recorder::new(&format!("batch {i}"));
        rec.text(&format!("/p/{i}a"), Field::Caption, None, Some("x"));
        rec.text(&format!("/p/{i}b"), Field::Caption, None, Some("x"));
        assert_eq!(rec.commit_with(&db).await, 2);
    }
    assert_eq!(stats_with(&db).await.unwrap().entries, 10);
    assert_eq!(trim_with(&db, 4).await.unwrap(), 6);
    let batches = batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.source.as_str()).collect::<Vec<_>>(), vec!["batch 4", "batch 3"], "whole batches stay or go");
    assert_eq!(trim_with(&db, 4).await.unwrap(), 0);

    let big = Recorder::new("big");
    for i in 0..10 {
        big.text(&format!("/p/big{i}"), Field::Caption, None, Some("y"));
    }
    big.commit_with(&db).await;
    assert_eq!(trim_with(&db, 3).await.unwrap(), 4, "the older batches go");
    let stats = stats_with(&db).await.unwrap();
    assert_eq!((stats.entries, stats.batches), (10, 1), "the newest batch stays even above the cap");
}

#[tokio::test]
async fn a_commit_applies_the_retention_cap() {
    let db = db().await;
    let rec = Recorder::new("many");
    for i in 0..(RETENTION + 20) {
        rec.text(&format!("/p/{i}"), Field::Caption, None, Some("x"));
    }
    assert_eq!(rec.commit_with(&db).await, RETENTION + 20);
    for i in 0..3 {
        let rec = Recorder::new(&format!("later {i}"));
        rec.text(&format!("/q/{i}"), Field::Caption, None, Some("x"));
        rec.commit_with(&db).await;
    }
    let stats = stats_with(&db).await.unwrap();
    assert_eq!(stats.entries, 3, "the oversized old batch was dropped, the three new ones stay");
}

#[tokio::test]
async fn a_batch_is_stored_whole_or_not_at_all() {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns("t").use_db("t").await.unwrap();
    db.query("DEFINE TABLE IF NOT EXISTS metadata_journal TYPE NORMAL SCHEMAFULL; DEFINE FIELD IF NOT EXISTS path ON metadata_journal TYPE string ASSERT $value != '/poison';").await.unwrap().check().unwrap();
    let ok = Recorder::new("fine");
    ok.text("/ok", Field::Caption, None, Some("x"));
    assert_eq!(ok.commit_with(&db).await, 1);
    let bad = Recorder::new("poisoned");
    for p in ["/a", "/poison", "/b"] {
        bad.text(p, Field::Caption, None, Some("x"));
    }
    assert_eq!(bad.commit_with(&db).await, 0, "the commit failed");
    let entries = recent_with(&db, 10).await.unwrap();
    assert_eq!(entries.iter().map(|e| e.path.as_str()).collect::<Vec<_>>(), vec!["/ok"], "none of the poisoned batch was stored");
}

#[tokio::test]
async fn a_recorder_that_is_dropped_leaves_nothing_pending() {
    let batch = {
        let rec = Recorder::new("dropped");
        rec.text("/a", Field::Caption, None, Some("x"));
        rec.batch()
    };
    assert_eq!(pending_len(batch), 0);
    let db = db().await;
    assert_eq!(commit_with(&db, batch).await.unwrap(), 0);
    assert!(recent_with(&db, 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn clear_removes_everything_and_the_journal_is_readable_when_empty() {
    let db = db().await;
    assert!(recent_with(&db, 5).await.unwrap().is_empty());
    assert!(undo_last_with(&db).await.unwrap().is_empty());
    metadata_edit::save_patch(&db, &seed("/p/a.jpg"), &caption("x")).await.unwrap();
    assert_eq!(clear_with(&db).await.unwrap(), 1);
    assert_eq!(stats_with(&db).await.unwrap(), Stats::default());
}

#[tokio::test]
async fn the_revision_grows_with_every_change() {
    let db = db().await;
    let start = revision();
    metadata_edit::save_patch(&db, &seed("/p/a.jpg"), &caption("x")).await.unwrap();
    let after_commit = revision();
    assert!(after_commit > start);
    undo_last_with(&db).await.unwrap();
    assert!(revision() > after_commit);
}

#[tokio::test]
async fn the_last_report_is_kept_for_the_devtools_window() {
    let db = db().await;
    metadata_edit::save_patch(&db, &seed("/p/a.jpg"), &caption("x")).await.unwrap();
    let report = undo_last_with(&db).await.unwrap();
    let kept = last_report().expect("a report is kept");
    assert_eq!(kept.direction, Direction::Undo);
    assert!(kept.summary().len() > 0 && report.batch.is_some());
}

#[tokio::test]
async fn the_inspector_backend_writes_are_journaled_with_readable_labels() {
    use crate::ui::inspector::backend::{Backend, CollectionOp, DbBackend};
    let db = db().await;
    let backend = DbBackend::new(db.clone());
    let (a, b) = (seed("/p/a.jpg"), seed("/p/b.jpg"));
    backend.save(a.clone(), MetaPatch { caption: Some(Some("typed".into())), ..Default::default() }).await.unwrap();
    backend.batch_tags(vec![a.clone(), b.clone()], TagChange { add: tags(&["sun"]), remove: Vec::new() }).await.unwrap();
    backend.batch_tags(vec![a.clone()], TagChange { add: Vec::new(), remove: tags(&["sun"]) }).await.unwrap();
    backend.batch_category(vec![a.clone(), b.clone()], Some("travel".into())).await.unwrap();
    backend.batch_category(vec![b.clone()], None).await.unwrap();
    backend.collection_edit(CollectionOp::Add { name: "Holidays".into(), paths: tags(&["/p/a.jpg", "/p/b.jpg"]) }).await.unwrap();
    backend.collection_edit(CollectionOp::Remove { name: "Holidays".into(), paths: tags(&["/p/a.jpg"]) }).await.unwrap();
    let labels: Vec<String> = batches_with(&db, 20).await.unwrap().into_iter().map(|b| b.source).collect();
    assert_eq!(labels, vec!["Remove from \"Holidays\"", "Add to \"Holidays\"", "Clear category", "Set category", "Remove tags", "Add tags", "Edit metadata"]);

    let report = undo_last_with(&db).await.unwrap();
    assert_eq!(report.membership_paths(), vec!["/p/a.jpg".to_string()]);
    let c = collections::get_by_name(&db, "Holidays").await.unwrap().unwrap();
    assert!(is_member(&db, &c.id, "/p/a.jpg").await, "the removal was undone");
    undo_last_with(&db).await.unwrap();
    assert!(!is_member(&db, &c.id, "/p/b.jpg").await, "the add was undone");
    undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/b.jpg").await.category.as_deref(), Some("travel"), "the cleared category is back");
}

#[tokio::test]
async fn the_explorer_collection_menu_writes_are_journaled() {
    use crate::ui::file_table::menus::logical_groups::{add_to_named, copy_between, remove_from};
    let db = db().await;
    let paths = tags(&["/p/a.jpg", "/p/b.jpg"]);
    add_to_named(&db, "Trips", &paths).await.unwrap();
    let trips = collections::get_by_name(&db, "Trips").await.unwrap().unwrap();
    remove_from(&db, &trips.id, "Trips", &paths[..1]).await.unwrap();
    copy_between(&db, "Trips", "Archive").await.unwrap();
    let labels: Vec<String> = batches_with(&db, 10).await.unwrap().into_iter().map(|b| b.source).collect();
    assert_eq!(labels, vec!["Copy \"Trips\" to \"Archive\"", "Remove from \"Trips\"", "Add to \"Trips\""]);
    let archive = collections::get_by_name(&db, "Archive").await.unwrap().unwrap();
    assert_eq!(collections::count_members(&db, &archive.id).await.unwrap(), 1);
    undo_last_with(&db).await.unwrap();
    assert_eq!(collections::count_members(&db, &archive.id).await.unwrap(), 0, "the copy was undone");
    undo_last_with(&db).await.unwrap();
    assert!(is_member(&db, &trips.id, "/p/a.jpg").await, "the removal was undone");
}

#[tokio::test]
async fn the_devtools_membership_editor_and_the_scan_save_are_journaled() {
    use crate::ui::devtools::collections::apply_paths;
    use crate::ui::file_table::explorer::persist::persist_scanned;
    let db = db().await;
    let c = collections::create(&db, "Manual").await.unwrap();
    apply_paths(&db, &c.id, "/m/a.jpg\n/m/b.jpg", true).await.unwrap();
    apply_paths(&db, &c.id, "/m/a.jpg", false).await.unwrap();
    let labels: Vec<String> = batches_with(&db, 10).await.unwrap().into_iter().map(|b| b.source).collect();
    assert_eq!(labels, vec!["Remove from collection (devtools)", "Add to collection (devtools)"]);

    collections::create(&db, "Scanned").await.unwrap();
    let mut row = crate::Thumbnail::new("s.jpg");
    row.path = "/s/s.jpg".into();
    row.file_type = "jpg".into();
    let report = persist_scanned(&db, vec![row], Some("Scanned")).await.unwrap();
    assert_eq!(report.filed, 1);
    let latest = batches_with(&db, 1).await.unwrap();
    assert_eq!((latest[0].source.as_str(), latest[0].entries), ("Add to \"Scanned\"", 1));
}
