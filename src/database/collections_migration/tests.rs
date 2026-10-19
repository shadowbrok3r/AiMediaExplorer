use std::collections::BTreeMap;
use std::path::PathBuf;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::*;
use crate::database::collections::tests::{SeedRow, db, insert_rows, seed_row};
use crate::database::snapshot::{self, no_progress};
use crate::database::surql::{self, VerifyOptions, VerifyTarget};
use crate::database::tests::make_store;

#[derive(Clone, Debug, SurrealValue)]
struct GroupIn2 {
    id: RecordId,
    name: String,
}

fn bad_name() -> String {
    format!("bad{}name", '\u{7}')
}

/// Legacy groups of the fixture: record key and display name.
fn fixture_groups() -> Vec<(&'static str, String)> {
    vec![
        ("Default", "Default".to_string()),
        ("Holiday", "Holiday".to_string()),
        ("Trip 2023", "Trip 2023".to_string()),
        ("weird", "weird `name` \"q\" 'x' :: [a]".to_string()),
        ("jp", "日本語 😀".to_string()),
        ("num", "123".to_string()),
        ("holiday-lower", "holiday".to_string()),
        ("default-lower", "default".to_string()),
        ("blank", "   ".to_string()),
        ("long", "L".repeat(100)),
        ("ctrl", bad_name()),
    ]
}

/// Number of files linked to each fixture group.
fn group_size(key: &str) -> usize {
    match key {
        "Default" => 9,
        "Holiday" => 7,
        "Trip 2023" => 5,
        "weird" => 3,
        "jp" => 2,
        "num" => 1,
        "holiday-lower" => 4,
        "default-lower" => 2,
        "blank" => 2,
        "long" => 3,
        "ctrl" => 1,
        _ => 0,
    }
}

fn path_of(key: &str, i: usize) -> String {
    format!("/lib/{}/{i:03}.jpg", key.replace(' ', "_"))
}

async fn legacy_db() -> Surreal<Db> {
    let db = db().await;
    fill_legacy(&db).await;
    db
}

async fn fill_legacy(db: &Surreal<Db>) {
    let groups: Vec<GroupIn2> = fixture_groups().into_iter().map(|(k, n)| GroupIn2 { id: RecordId::new("logical_groups", k), name: n }).collect();
    db.query("INSERT INTO logical_groups $rows").bind(("rows", groups)).await.unwrap().check().unwrap();
    let mut rows: Vec<SeedRow> = Vec::new();
    for (key, _) in fixture_groups() {
        for i in 0..group_size(key) {
            let mut r = seed_row(rows.len(), &path_of(key, i));
            r.logical_group = RecordId::new("logical_groups", key);
            rows.push(r);
        }
    }
    let n = rows.len();
    let mut dangling = seed_row(n, "/lib/dangling/0.jpg");
    dangling.logical_group = RecordId::new("logical_groups", "ghost");
    rows.push(dangling);
    insert_rows(db, rows).await;
}

fn options() -> CopyOptions {
    CopyOptions { require_snapshot: false, ..Default::default() }
}

fn finish_options() -> FinishOptions {
    FinishOptions { require_snapshot: false, repair: true }
}

fn migrator(db: &Surreal<Db>) -> Arc<Migrator> {
    Migrator::new(db.clone(), PathBuf::from("/nonexistent/db/ai_search"))
}

async fn links(db: &Surreal<Db>) -> BTreeMap<String, String> {
    #[derive(SurrealValue)]
    struct Row {
        path: String,
        logical_group: RecordId,
    }
    let rows: Vec<Row> = db.query("SELECT path, logical_group FROM thumbnails").await.unwrap().take(0).unwrap();
    rows.into_iter().map(|r| (r.path, collections::key_of(&r.logical_group))).collect()
}

async fn legacy_names(db: &Surreal<Db>) -> BTreeMap<String, String> {
    legacy_groups(db).await.unwrap().into_iter().map(|g| (g.key, g.name)).collect()
}

async fn by_legacy(db: &Surreal<Db>) -> BTreeMap<String, Collection> {
    collections::list(db).await.unwrap().into_iter().filter_map(|c| c.legacy_key.clone().map(|k| (k, c))).collect()
}

fn gin(key: &str, name: &str) -> GroupIn {
    GroupIn { key: key.into(), name: name.into() }
}

#[test]
fn sanitize_drops_controls_collapses_space_and_caps_length() {
    assert_eq!(sanitize_name("  a \t b\n c "), "a b c");
    assert_eq!(sanitize_name(&bad_name()), "badname");
    assert_eq!(sanitize_name("   "), "Untitled");
    assert_eq!(sanitize_name(""), "Untitled");
    assert_eq!(sanitize_name("\u{0}\u{1}"), "Untitled");
    assert_eq!(sanitize_name(&"x".repeat(100)).chars().count(), MAX_NAME_CHARS);
    assert_eq!(sanitize_name(&format!("{} tail", "y".repeat(63))), "y".repeat(63), "a cut that ends in a space is trimmed");
    assert_eq!(sanitize_name("日本語 😀"), "日本語 😀");
    for name in ["`q`", "'x'", "\"q\"", "a:b", "[x]", "../x"] {
        assert_eq!(sanitize_name(name), name);
    }
}

#[test]
fn assignment_gives_every_group_a_distinct_valid_name_and_skips_the_unfiled_group() {
    let groups = vec![
        gin("a", "Default"),
        gin("b", "default"),
        gin("c", "DEFAULT"),
        gin("d", "Holiday"),
        gin("e", "holiday"),
        gin("f", "HOLIDAY"),
        gin("g", "   "),
        gin("h", ""),
        gin("i", &"z".repeat(64)),
        gin("j", &"Z".repeat(64)),
        gin("k", &"z".repeat(90)),
        gin("l", &bad_name()),
        gin("m", "badname"),
        gin("n", "Untitled"),
    ];
    let out = assign_names(&groups, &HashMap::new(), &HashSet::new());
    assert_eq!(out.len(), groups.len() - 1, "only the group named Default is skipped");
    assert!(out.iter().all(|a| a.legacy_key != "a"));
    let mut keys = HashSet::new();
    for a in &out {
        let (display, key) = collections::normalize(&a.name).unwrap_or_else(|e| panic!("{:?} is not a valid name: {e}", a.name));
        assert_eq!(display, a.name, "assigned names are already clean");
        assert!(keys.insert(key), "{:?} collides", a.name);
        assert!(a.name.chars().count() <= MAX_NAME_CHARS, "{}", a.name);
        assert!(!a.reuse);
    }
    let name_of = |k: &str| out.iter().find(|a| a.legacy_key == k).unwrap().name.clone();
    assert_eq!((name_of("f"), name_of("d"), name_of("e")), ("HOLIDAY".to_string(), "Holiday (2)".to_string(), "holiday (3)".to_string()), "case variants are told apart by a suffix, in name order");
    assert!(out.iter().find(|a| a.legacy_key == "b").unwrap().renamed, "default is reserved so it is renamed");
    assert!(name_of("b").starts_with("default ("));
    assert!(out.iter().find(|a| a.legacy_key == "k").unwrap().renamed);
    assert_eq!(name_of("l"), "badname", "control characters are dropped");
    assert_eq!(name_of("m"), "badname (2)", "and the result is told apart from a group that already had that name");
}

#[test]
fn assignment_is_deterministic_regardless_of_input_order() {
    let mut groups: Vec<GroupIn> = (0..30).map(|i| gin(&format!("k{i}"), ["Alpha", "alpha", "ALPHA", "Beta", "", "x y", "X  Y"][i % 7])).collect();
    let a = assign_names(&groups, &HashMap::new(), &HashSet::new());
    groups.reverse();
    let b = assign_names(&groups, &HashMap::new(), &HashSet::new());
    assert_eq!(a, b);
    let names: HashSet<String> = a.iter().map(|x| x.name.to_lowercase()).collect();
    assert_eq!(names.len(), a.len());
}

#[test]
fn assignment_avoids_existing_collections_and_reuses_migrated_groups() {
    let groups = vec![gin("a", "Holiday"), gin("b", "Trip"), gin("c", "Fresh")];
    let taken: HashSet<String> = ["holiday".to_string(), "trip".to_string()].into();
    let existing: HashMap<String, String> = [("b".to_string(), "Trip".to_string())].into();
    let out = assign_names(&groups, &existing, &taken);
    let by = |k: &str| out.iter().find(|a| a.legacy_key == k).unwrap();
    assert!(by("b").reuse && by("b").name == "Trip" && !by("b").renamed);
    assert_eq!(by("a").name, "Holiday (2)", "a name another collection already holds gets a suffix");
    assert!(by("a").renamed && !by("a").reuse);
    assert_eq!(by("c").name, "Fresh");
    let renamed_later: HashMap<String, String> = [("b".to_string(), "Journey".to_string())].into();
    let again = assign_names(&groups, &renamed_later, &taken);
    let b = again.iter().find(|a| a.legacy_key == "b").unwrap();
    assert!(b.reuse && b.name == "Journey" && b.renamed, "a collection renamed after the migration stays linked");
}

#[tokio::test]
async fn the_copy_builds_collections_and_memberships_and_leaves_the_legacy_shape_untouched() {
    let db = legacy_db().await;
    let links_before = links(&db).await;
    let groups_before = legacy_names(&db).await;
    let m = migrator(&db);
    let report = m.run_copy(options()).await.unwrap();
    assert!(!report.cancelled);
    assert_eq!(report.groups_mapped, 10, "every group but Default");
    assert_eq!(report.collections_created, 10);
    let expected: u64 = fixture_groups().iter().filter(|(k, _)| *k != "Default").map(|(k, _)| group_size(k) as u64).sum();
    assert_eq!(report.memberships_added, expected);
    assert_eq!(report.unfiled_rows, group_size("Default") as u64);
    assert_eq!(report.dangling_links, 1);
    assert_eq!(links(&db).await, links_before, "thumbnails.logical_group is not changed");
    assert_eq!(legacy_names(&db).await, groups_before, "logical_groups is not changed");

    let migrated = by_legacy(&db).await;
    assert_eq!(migrated.len(), 10);
    assert!(!migrated.contains_key("Default"), "the unfiled group is not a collection");
    assert!(collections::get_by_name(&db, "Default").await.unwrap().is_none());
    for (key, name) in fixture_groups().into_iter().filter(|(k, _)| *k != "Default") {
        let c = &migrated[key];
        assert_eq!(collections::count_members(&db, &c.id).await.unwrap(), group_size(key) as u64, "{key}");
        let mut paths = collections::members(&db, &c.id, 0, 100).await.unwrap();
        paths.sort();
        let want: Vec<String> = (0..group_size(key)).map(|i| path_of(key, i)).collect();
        assert_eq!(paths, want, "{key}");
        if name == c.name {
            continue;
        }
        assert!(report.renamed.iter().any(|r| r.legacy_key == key && r.assigned == c.name && r.original == name), "{key} is listed as renamed");
    }
    assert_eq!(migrated["Holiday"].name, "Holiday");
    assert_eq!(migrated["Trip 2023"].name, "Trip 2023");
    assert_eq!(migrated["jp"].name, "日本語 😀");
    assert_eq!(migrated["num"].name, "123");
    assert_eq!(migrated["weird"].name, "weird `name` \"q\" 'x' :: [a]");
    assert_eq!(migrated["holiday-lower"].name, "holiday (2)");
    assert_eq!(migrated["default-lower"].name, "default (2)");
    assert_eq!(migrated["blank"].name, "Untitled");
    assert_eq!(migrated["long"].name.chars().count(), MAX_NAME_CHARS);
    assert_eq!(migrated["ctrl"].name, "badname");
    let legacy_created = legacy_groups(&db).await.unwrap().into_iter().find(|g| g.key == "Holiday").unwrap().created;
    assert_eq!(migrated["Holiday"].created, legacy_created, "creation times carry over");

    let state = m.state().await.unwrap();
    assert_eq!(state.phase(), Phase::Copied);
    assert_eq!((state.runs, state.groups_done.len(), state.collections_created), (1, 10, 10));
    assert!(state.started.is_some() && state.copy_done && state.finished.is_none());
}

#[tokio::test]
async fn a_second_copy_changes_nothing() {
    let db = legacy_db().await;
    let m = migrator(&db);
    let first = m.run_copy(options()).await.unwrap();
    let items = collections::count_items(&db).await.unwrap();
    let second = m.run_copy(options()).await.unwrap();
    assert_eq!(second.collections_created, 0);
    assert_eq!(second.memberships_added, 0);
    assert_eq!(second.memberships_existing, first.memberships_added);
    assert_eq!(collections::count_items(&db).await.unwrap(), items);
    assert_eq!(collections::list(&db).await.unwrap().len(), 10);
    assert_eq!(m.state().await.unwrap().runs, 2);
}

#[tokio::test]
async fn an_interrupted_copy_resumes_and_ends_where_an_uninterrupted_one_does() {
    let whole = legacy_db().await;
    migrator(&whole).run_copy(options()).await.unwrap();

    let db = legacy_db().await;
    let m = migrator(&db);
    let partial = m.run_copy(CopyOptions { max_groups: Some(3), ..options() }).await.unwrap();
    assert!(partial.cancelled);
    let mid = m.state().await.unwrap();
    assert_eq!((mid.phase(), mid.groups_done.len(), mid.copy_done), (Phase::Copying, 3, false));
    assert!(collections::count_items(&db).await.unwrap() > 0 && collections::count_items(&db).await.unwrap() < collections::count_items(&whole).await.unwrap());
    let rest = m.run_copy(options()).await.unwrap();
    assert!(!rest.cancelled);
    let end = m.state().await.unwrap();
    assert_eq!((end.phase(), end.groups_done.len(), end.runs), (Phase::Copied, 10, 1), "a resume is not a new pass");
    assert_eq!(rest.memberships_added, whole_added(&whole).await);
    assert_eq!(collections::count_items(&db).await.unwrap(), collections::count_items(&whole).await.unwrap());
    let key_paths = |d: Surreal<Db>| async move {
        let mut out = BTreeMap::new();
        for (k, c) in by_legacy(&d).await {
            out.insert(k, collections::members(&d, &c.id, 0, 1000).await.unwrap());
        }
        out
    };
    assert_eq!(key_paths(db.clone()).await, key_paths(whole.clone()).await);
}

async fn whole_added(db: &Surreal<Db>) -> u64 {
    migrator(db).state().await.unwrap().memberships_added
}

#[tokio::test]
async fn the_progress_counters_follow_the_copy() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.run_copy(options()).await.unwrap();
    let p = m.progress();
    assert_eq!(p.step, Step::Done);
    assert_eq!((p.groups_total, p.groups_done), (10, 10));
    assert_eq!(p.processed, p.memberships_total);
    assert_eq!(p.fraction(), Some(1.0));
    assert!(p.ended.is_some() && p.error.is_none());
    assert!(!m.is_running());
}

#[tokio::test]
async fn a_dry_run_writes_nothing() {
    let db = legacy_db().await;
    let m = migrator(&db);
    let plan = m.plan().await.unwrap();
    assert_eq!(plan.groups.len(), 10);
    assert_eq!(plan.to_create, 10);
    assert_eq!(plan.memberships, fixture_groups().iter().filter(|(k, _)| *k != "Default").map(|(k, _)| group_size(k) as u64).sum::<u64>());
    assert_eq!(plan.unfiled_rows, group_size("Default") as u64);
    assert_eq!(plan.dangling_links, 1);
    assert!(plan.renamed >= 5, "{}", plan.renamed);
    assert!(plan.groups.iter().any(|g| g.assignment.legacy_key == "Holiday" && g.assignment.name == "Holiday" && g.memberships == 7));
    assert_eq!(collections::list(&db).await.unwrap().len(), 0);
    assert_eq!(collections::count_items(&db).await.unwrap(), 0);
    assert_eq!(m.state().await.unwrap(), MigrationState::default());
    let after = m.plan().await.unwrap();
    assert_eq!(after.memberships, plan.memberships);
    m.run_copy(options()).await.unwrap();
    let planned_again = m.plan().await.unwrap();
    assert_eq!(planned_again.to_create, 0, "migrated groups are reused");
    assert!(planned_again.groups.iter().all(|g| g.assignment.reuse));
}

#[tokio::test]
async fn finish_verifies_marks_finished_and_keeps_the_legacy_data() {
    let db = legacy_db().await;
    let m = migrator(&db);
    let err = m.run_finish(finish_options()).await.unwrap_err();
    assert!(format!("{err:#}").contains("copy phase"), "{err:#}");
    assert_eq!(m.progress().step, Step::Failed);
    m.run_copy(options()).await.unwrap();
    let report = m.run_finish(finish_options()).await.unwrap();
    assert!(report.ok, "{}", report.summary);
    assert!(report.legacy_cleanup_pending && !report.repaired);
    assert_eq!(report.verify.groups_checked, 10);
    assert_eq!(report.verify.missing, 0);
    assert_eq!(report.verify.legacy_memberships, report.verify.present);
    let state = m.state().await.unwrap();
    assert_eq!(state.phase(), Phase::Finished);
    assert!(state.finished.is_some() && state.verify.as_ref().is_some_and(|v| v.ok));
    let any: Vec<RecordId> = db.query("SELECT VALUE id FROM logical_groups").await.unwrap().take(0).unwrap();
    assert_eq!(any.len(), 11, "finish does not drop the legacy groups");
    assert_eq!(links(&db).await.len(), 40, "finish does not touch the legacy links");
    assert_eq!(legacy_pending(&db).await.unwrap(), false);
}

#[tokio::test]
async fn finish_repairs_what_went_missing_or_reports_it() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.run_copy(options()).await.unwrap();
    let holiday = by_legacy(&db).await["Holiday"].clone();
    collections::remove_paths(&db, &holiday.id, &[path_of("Holiday", 0), path_of("Holiday", 1)]).await.unwrap();
    let c = by_legacy(&db).await["jp"].clone();
    collections::delete(&db, &c.id).await.unwrap();

    let found = m.verify().await.unwrap();
    assert!(!found.ok);
    assert_eq!(found.missing, 2 + group_size("jp") as u64);
    assert_eq!(found.groups_without_collection, vec!["日本語 😀".to_string()]);
    assert_eq!(found.mismatched.iter().find(|g| g.legacy_key == "Holiday").map(|g| (g.legacy_count, g.present)), Some((7, 5)));
    assert!(m.state().await.unwrap().finished.is_none(), "verify alone changes nothing");

    let strict = m.run_finish(FinishOptions { repair: false, ..finish_options() }).await.unwrap();
    assert!(!strict.ok && strict.summary.starts_with("not finished"), "{}", strict.summary);
    assert_eq!(m.state().await.unwrap().phase(), Phase::Copied);

    let fixed = m.run_finish(finish_options()).await.unwrap();
    assert!(fixed.ok && fixed.repaired, "{}", fixed.summary);
    assert_eq!(m.state().await.unwrap().phase(), Phase::Finished);
    assert_eq!(collections::count_members(&db, &by_legacy(&db).await["Holiday"].id).await.unwrap(), 7);
}

#[tokio::test]
async fn dangling_memberships_make_verification_fail() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.run_copy(options()).await.unwrap();
    db.query("INSERT INTO collection_items { id: collection_items:ghostitem, collection: collections:ghost, path: '/x' }").await.unwrap().check().unwrap();
    let v = m.verify().await.unwrap();
    assert!(!v.ok);
    assert_eq!(v.orphan_memberships, 1);
    assert_eq!(m.status().await.unwrap().orphan_memberships, 1);
}

#[tokio::test]
async fn rollback_removes_the_new_tables_rows_and_nothing_else_and_the_copy_can_run_again() {
    let db = legacy_db().await;
    let links_before = links(&db).await;
    let groups_before = legacy_names(&db).await;
    let m = migrator(&db);
    let err = m.run_rollback(RollbackOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("nothing to roll back"), "{err:#}");
    m.run_copy(options()).await.unwrap();
    m.run_finish(finish_options()).await.unwrap();

    let mine = collections::create(&db, "Made Later").await.unwrap();
    collections::add_paths(&db, &mine.id, &[path_of("Holiday", 0)]).await.unwrap();
    collections::add_paths(&db, &by_legacy(&db).await["Holiday"].id, &["/new/after.jpg".to_string()]).await.unwrap();
    let refused = m.run_rollback(RollbackOptions::default()).await.unwrap();
    assert!(!refused.ok);
    assert_eq!((refused.created_after_migration, refused.memberships_after_migration), (1, 2));
    assert!(refused.summary.starts_with("refused"), "{}", refused.summary);
    assert_eq!(collections::list(&db).await.unwrap().len(), 11, "a refused rollback removes nothing");

    let forced = m.run_rollback(RollbackOptions { force: true, ..Default::default() }).await.unwrap();
    assert!(forced.ok, "{}", forced.summary);
    assert_eq!(forced.collections_removed, 11);
    assert_eq!(collections::list(&db).await.unwrap().len(), 0);
    assert_eq!(collections::count_items(&db).await.unwrap(), 0);
    assert_eq!(links(&db).await, links_before);
    assert_eq!(legacy_names(&db).await, groups_before);
    let state = m.state().await.unwrap();
    assert_eq!(state.phase(), Phase::RolledBack);
    assert!(state.rolled_back.is_some() && state.started.is_none() && state.finished.is_none());
    assert_eq!(legacy_pending(&db).await.unwrap(), true);

    let again = m.run_copy(options()).await.unwrap();
    assert_eq!(again.collections_created, 10);
    assert_eq!(m.state().await.unwrap().phase(), Phase::Copied);
    let clean = m.run_rollback(RollbackOptions::default()).await.unwrap();
    assert!(clean.ok, "a rollback with nothing created afterwards needs no force: {}", clean.summary);
}

#[tokio::test]
async fn removing_a_migrated_membership_alone_does_not_block_rollback() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.run_copy(options()).await.unwrap();
    let c = by_legacy(&db).await["Holiday"].clone();
    collections::remove_paths(&db, &c.id, &[path_of("Holiday", 0)]).await.unwrap();
    assert!(m.run_rollback(RollbackOptions::default()).await.unwrap().ok);
}

#[tokio::test]
async fn a_later_legacy_import_is_picked_up_by_running_the_copy_again() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.run_copy(options()).await.unwrap();
    m.run_finish(finish_options()).await.unwrap();
    let holiday = by_legacy(&db).await["Holiday"].clone();
    let renamed = collections::rename(&db, &holiday.id, "Summer Break").await.unwrap().collection;

    let mut rows: Vec<SeedRow> = Vec::new();
    for i in 0..4 {
        let mut r = seed_row(1000 + i, &format!("/imported/new{i}.jpg"));
        r.logical_group = RecordId::new("logical_groups", "Holiday");
        rows.push(r);
    }
    let mut other = seed_row(2000, "/imported/other.jpg");
    other.logical_group = RecordId::new("logical_groups", "Imported Album");
    rows.push(other);
    insert_rows(&db, rows).await;
    db.query("INSERT INTO logical_groups { id: logical_groups:⟨Imported Album⟩, name: 'Imported Album' }").await.unwrap().check().unwrap();
    assert_eq!(legacy_pending(&db).await.unwrap(), false, "nothing is flagged until the importer says so");
    note_legacy_import(&db).await.unwrap();
    let st = m.state().await.unwrap();
    assert!(st.needs_resync);
    assert_eq!(legacy_pending(&db).await.unwrap(), true);
    assert_eq!(m.status().await.unwrap().state.needs_resync, true);

    let rep = m.run_copy(options()).await.unwrap();
    assert_eq!(rep.collections_created, 1, "only the new album is created; the renamed collection is reused");
    assert_eq!(rep.memberships_added, 5);
    assert_eq!(collections::count_members(&db, &renamed.id).await.unwrap(), 7 + 4);
    assert!(collections::get_by_name(&db, "Holiday").await.unwrap().is_none(), "the old name is not resurrected");
    assert_eq!(collections::list(&db).await.unwrap().len(), 11);
    let st = m.state().await.unwrap();
    assert!(!st.needs_resync && st.finished.is_none() && st.copy_done);
    assert!(m.run_finish(finish_options()).await.unwrap().ok);
    assert_eq!(legacy_pending(&db).await.unwrap(), false);
}

#[tokio::test]
async fn note_legacy_import_is_a_noop_before_any_migration() {
    let db = legacy_db().await;
    note_legacy_import(&db).await.unwrap();
    assert_eq!(load_state(&db).await.unwrap(), MigrationState::default());
}

#[tokio::test]
async fn a_renamed_default_record_is_migrated_with_a_warning() {
    let db = db().await;
    db.query("INSERT INTO logical_groups [{ id: logical_groups:Default, name: 'Everything' }, { id: logical_groups:Other, name: 'Other' }]").await.unwrap().check().unwrap();
    let mut a = seed_row(1, "/w/a.jpg");
    a.logical_group = RecordId::new("logical_groups", "Default");
    let mut b = seed_row(2, "/w/b.jpg");
    b.logical_group = RecordId::new("logical_groups", "Other");
    insert_rows(&db, vec![a, b]).await;
    let m = migrator(&db);
    let rep = m.run_copy(options()).await.unwrap();
    assert_eq!(rep.collections_created, 2);
    assert_eq!(rep.warnings.len(), 1);
    assert!(rep.warnings[0].contains("Everything"), "{}", rep.warnings[0]);
    assert_eq!(collections::get_by_name(&db, "Everything").await.unwrap().map(|c| c.legacy_key), Some(Some("Default".to_string())));
}

#[tokio::test]
async fn a_store_with_only_the_unfiled_group_migrates_nothing_and_needs_no_snapshot() {
    let db = db().await;
    db.query("INSERT INTO logical_groups { id: logical_groups:Default, name: 'Default' }").await.unwrap().check().unwrap();
    insert_rows(&db, vec![seed_row(1, "/u/1.jpg"), seed_row(2, "/u/2.jpg")]).await;
    let m = migrator(&db);
    let rep = m.run_copy(CopyOptions { require_snapshot: true, ..Default::default() }).await.unwrap();
    assert_eq!((rep.groups_mapped, rep.collections_created, rep.memberships_added), (0, 0, 0));
    assert_eq!(rep.snapshot, None, "nothing to protect, so no snapshot of a store that may not even exist");
    assert_eq!(rep.unfiled_rows, 2);
    assert!(m.run_finish(finish_options()).await.unwrap().ok);
    assert_eq!(legacy_pending(&db).await.unwrap(), false);
}

#[tokio::test]
async fn the_copy_takes_a_snapshot_first_and_refuses_to_write_without_one() {
    let db = legacy_db().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "snap_store");
    let m = Migrator::new(db.clone(), store.clone());
    let report = m.run_copy(CopyOptions { require_snapshot: true, ..Default::default() }).await.unwrap();
    let snap = report.snapshot.clone().expect("a snapshot was taken");
    let info = snapshot::list_snapshots_for(&store);
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].label.as_deref(), Some(SNAPSHOT_LABEL));
    assert!(snapshot::read_manifest(&info[0].path).unwrap().consistent);
    assert_eq!(m.state().await.unwrap().snapshot.as_deref(), Some(snap.as_str()));
    let fin = m.run_finish(FinishOptions::default()).await.unwrap();
    assert!(fin.ok);
    assert_eq!(snapshot::list_snapshots_for(&store).len(), 1, "finish reuses the snapshot");
    let rb = m.run_rollback(RollbackOptions { require_snapshot: true, force: false }).await.unwrap();
    assert!(rb.ok);
    assert_eq!(rb.snapshot.as_deref(), Some(snap.as_str()), "a fresh snapshot is reused for the rollback");

    let blocked = legacy_db().await;
    let nb = Migrator::new(blocked.clone(), PathBuf::from("/nonexistent/db/ai_search"));
    let err = nb.run_copy(CopyOptions { require_snapshot: true, ..Default::default() }).await.unwrap_err();
    assert!(!format!("{err:#}").is_empty());
    assert_eq!(collections::list(&blocked).await.unwrap().len(), 0, "no collection is written when the snapshot cannot be taken");
    assert_eq!(load_state(&blocked).await.unwrap().started, None);
    assert_eq!(nb.progress().step, Step::Failed);
    assert!(nb.progress().error.is_some());
}

#[tokio::test]
async fn only_one_step_runs_at_a_time() {
    let db = legacy_db().await;
    let m = migrator(&db);
    m.claim().unwrap();
    assert!(m.is_running());
    assert!(m.run_copy(options()).await.is_err());
    assert!(!m.spawn_copy(options()));
    assert!(!m.spawn_finish(finish_options()));
    assert!(!m.spawn_rollback(RollbackOptions::default()));
    m.shared.running.store(false, Ordering::SeqCst);
    assert!(m.run_copy(options()).await.is_ok());
}

#[tokio::test]
async fn spawned_steps_run_to_completion_and_cancel_is_ignored_while_idle() {
    let db = legacy_db().await;
    let m = migrator(&db);
    assert!(m.spawn_copy(options()));
    for _ in 0..600 {
        if !m.is_running() && m.progress().ended.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(m.state().await.unwrap().phase(), Phase::Copied);
    assert!(m.spawn_finish(finish_options()));
    for _ in 0..600 {
        if !m.is_running() && m.progress().ended.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(m.state().await.unwrap().phase(), Phase::Finished);
    m.cancel();
    assert!(!m.shared.cancel.load(Ordering::SeqCst), "cancel is ignored while nothing runs");
}

#[tokio::test]
async fn the_state_travels_with_an_export_and_the_migration_can_finish_on_the_import() {
    let tmp = tempfile::tempdir().unwrap();
    let src = legacy_db().await;
    let m = migrator(&src);
    m.run_copy(CopyOptions { max_groups: Some(4), ..options() }).await.unwrap();
    let file = tmp.path().join("mid.surql");
    surql::export_atomic(&src, &file, &no_progress()).await.unwrap();
    let verify = surql::verify_export(&file, VerifyTarget::Live(&src), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(verify.ok, "{:?}", verify.problems);
    assert_eq!(verify.imported.tables["collection_migration"].count, 1);

    let dst = crate::database::tests::mem_db().await;
    dst.import(&file).await.unwrap();
    let dm = migrator(&dst);
    let state = dm.state().await.unwrap();
    assert_eq!((state.phase(), state.groups_done.len()), (Phase::Copying, 4));
    assert_eq!(state, m.state().await.unwrap());
    let rest = dm.run_copy(options()).await.unwrap();
    assert!(!rest.cancelled);
    assert!(dm.run_finish(finish_options()).await.unwrap().ok);
    let full = tmp.path().join("full.surql");
    surql::export_atomic(&dst, &full, &no_progress()).await.unwrap();
    let again = surql::verify_export(&full, VerifyTarget::Live(&dst), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(again.ok, "{:?}", again.problems);
    assert_eq!(again.imported.tables["collections"].count, 10);
    assert_eq!(again.imported.tables["logical_groups"].count, 11, "the legacy table is exported unchanged");
}

#[tokio::test]
async fn status_reports_phase_and_live_counts() {
    let db = legacy_db().await;
    let m = migrator(&db);
    let before = m.status().await.unwrap();
    assert_eq!((before.phase, before.legacy_groups, before.collections, before.memberships), (Phase::NotStarted, 10, 0, 0));
    assert_eq!(before.phase_label, "not started");
    m.run_copy(options()).await.unwrap();
    let after = m.status().await.unwrap();
    assert_eq!((after.phase, after.collections), (Phase::Copied, 10));
    assert_eq!(after.memberships, 30);
    assert_eq!(after.orphan_memberships, 0);
    let json = serde_json::to_value(&after).unwrap();
    assert_eq!(json["phase"], "copied");
}

#[tokio::test]
async fn an_unreadable_state_record_starts_from_scratch() {
    let db = legacy_db().await;
    db.query("UPSERT collection_migration:state SET state = { runs: 'not a number', copy_done: 'x' }").await.unwrap().check().unwrap();
    assert_eq!(load_state(&db).await.unwrap(), MigrationState::default());
    let m = migrator(&db);
    assert!(m.run_copy(options()).await.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshots_verify_and_restore_carry_the_collection_tables() {
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("db").join("ai_search");
    let opened = surql::open_store(&store).await.unwrap();
    fill_legacy(&opened.db).await;
    let m = Migrator::new(opened.db.clone(), store.clone());
    m.run_copy(options()).await.unwrap();
    assert!(m.run_finish(finish_options()).await.unwrap().ok);
    drop(m);
    surql::close_store(opened, &store).await;

    let snap = snapshot::create_snapshot(store.clone(), "manual", None, no_progress()).await.unwrap();
    let dir = snap.path.clone();
    let check = tokio::task::spawn_blocking(move || snapshot::verify_snapshot_dir(&dir, &no_progress())).await.unwrap().unwrap();
    assert!(check.ok, "{check:?}");
    let manifest = surql::digest_snapshot_copy(&snap.path, &no_progress()).await.unwrap();
    assert_eq!((manifest.tables["collections"].count, manifest.tables["collection_items"].count, manifest.tables["collection_migration"].count), (10, 30, 1));
    assert!(manifest.tables["collection_items"].indexes.iter().any(|i| i.contains("UNIQUE")));

    let live = surql::open_store(&store).await.unwrap();
    for c in collections::list(&live.db).await.unwrap() {
        collections::delete(&live.db, &c.id).await.unwrap();
    }
    assert_eq!(collections::count_items(&live.db).await.unwrap(), 0);
    surql::close_store(live, &store).await;

    snapshot::restore_snapshot_to(snap.path.clone(), store.clone(), no_progress()).await.unwrap();
    let back = surql::open_store(&store).await.unwrap();
    let status = Migrator::new(back.db.clone(), store.clone()).status().await.unwrap();
    assert_eq!((status.phase, status.collections, status.memberships), (Phase::Finished, 10, 30));
    assert_eq!(legacy_names(&back.db).await.len(), 11);
    surql::close_store(back, &store).await;
}
