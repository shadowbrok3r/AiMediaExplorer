use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, TimeZone};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem, SurrealKv};
use surrealdb::types::{RecordId, SurrealValue};

use super::snapshot::*;
use super::surql::{self, Manifest, VerifyOptions, VerifyTarget};
use super::{DB_NAME, NS, SCHEMA_DDL, define_schema};

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(super) fn now_fixed() -> DateTime<Local> {
    Local.with_ymd_and_hms(2026, 10, 18, 12, 0, 0).unwrap()
}

pub(super) fn plenty(_: &Path) -> io::Result<u64> {
    Ok(u64::MAX / 2)
}

pub(super) fn make_store(root: &Path, name: &str) -> PathBuf {
    let store = root.join("db").join(name);
    for (rel, size) in [("manifest/MANIFEST-1", 120usize), ("sstables/000001.sst", 5000), ("vlog/000001.vlog", 9000), ("wal/00000001.wal", 3000), ("delete_list/deep/nested/file", 10)] {
        let p = store.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, (0..size).map(|i| (i % 251) as u8).collect::<Vec<u8>>()).unwrap();
    }
    fs::create_dir_all(store.join("discard_stats")).unwrap();
    fs::write(store.join(LOCK_FILE), b"lockfile").unwrap();
    store
}

fn tree_listing(root: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            if p.is_dir() {
                out.push((format!("{rel}/"), 0));
                stack.push(p);
            } else {
                out.push((rel, e.metadata().unwrap().len()));
            }
        }
    }
    out.sort();
    out
}

fn versions_for(surrealdb: &str, kv: Option<&str>) -> EngineVersions {
    EngineVersions { surrealdb: surrealdb.into(), surrealkv: kv.map(str::to_string) }
}

fn stamp_with(store: &Path, db: &str, kv: Option<&str>) -> EngineStamp {
    stamp_for(store, &versions_for(db, kv), now_fixed())
}

#[test]
fn lock_parsing_picks_highest_version() {
    let lock = "[[package]]\nname = \"surrealdb\"\nversion = \"3.2.4\"\n\n[[package]]\nname = \"surrealdb-core\"\nversion = \"9.9.9\"\n\n[[package]]\nname = \"surrealdb\"\nversion = \"3.10.0\"\n\n[[package]]\nname = \"surrealdb\"\nversion = \"3.10.0-beta.1\"\n";
    assert_eq!(lock_package_version(lock, "surrealdb").as_deref(), Some("3.10.0"));
    assert_eq!(lock_package_version(lock, "surrealdb-core").as_deref(), Some("9.9.9"));
    assert_eq!(lock_package_version(lock, "absent"), None);
}

#[tokio::test]
async fn compiled_engine_version_matches_running_engine() {
    let v = engine_versions();
    assert!(v.surrealdb.starts_with("3."), "unexpected surrealdb version {}", v.surrealdb);
    assert!(v.surrealkv.is_some());
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    let running = db.version().await.unwrap();
    assert_eq!(running.to_string(), v.surrealdb);
}

#[test]
fn stamp_roundtrip_and_states() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    assert_eq!(read_stamp(&store), StampState::Missing);
    let st = stamp_with(&store, "3.3.4", Some("0.21.4"));
    write_stamp(&store, &st).unwrap();
    assert!(stamp_path(&store).ends_with("db/engine-stamp.json"));
    assert_eq!(read_stamp(&store), StampState::Present(st.clone()));
    fs::write(stamp_path(&store), "{ not json").unwrap();
    assert!(matches!(read_stamp(&store), StampState::Unreadable(_)));
    fs::write(stamp_path(&store), r#"{"surrealdb_version":"3.0.0","app_version":"x","last_opened_rfc3339":"y","store_path":"z"}"#).unwrap();
    assert!(matches!(read_stamp(&store), StampState::Present(s) if s.surrealkv_version.is_none()));
}

#[test]
fn classify_covers_missing_changed_and_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let cur = versions_for("3.3.4", Some("0.21.4"));
    let present = |db: &str, kv: Option<&str>| StampState::Present(stamp_with(&store, db, kv));

    assert_eq!(classify(&store, false, &StampState::Missing, &cur), EngineChange::NoStore);
    assert!(matches!(classify(&store, true, &StampState::Missing, &cur), EngineChange::Changed { from: None, .. }));
    assert!(matches!(classify(&store, true, &StampState::Unreadable("x".into()), &cur), EngineChange::Changed { from: None, .. }));
    assert_eq!(classify(&store, true, &present("3.3.4", Some("0.21.4")), &cur), EngineChange::Unchanged);
    assert_eq!(classify(&store, true, &present("3.3.4", None), &cur), EngineChange::Unchanged);
    match classify(&store, true, &present("3.2.4", Some("0.21.3")), &cur) {
        EngineChange::Changed { from, to, .. } => {
            assert_eq!(from.as_deref(), Some("3.2.4"));
            assert_eq!(to, "3.3.4");
        }
        other => panic!("expected change, got {other:?}"),
    }
    assert!(classify(&store, true, &present("3.3.4", Some("0.21.3")), &cur).is_changed());
    let other_store = tmp.path().join("db").join("other");
    assert!(classify(&other_store, true, &present("3.3.4", Some("0.21.4")), &cur).is_changed());
}

#[test]
fn store_detection_helpers() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(!store_has_data(&tmp.path().join("absent")));
    let only_lock = tmp.path().join("only_lock");
    fs::create_dir_all(&only_lock).unwrap();
    fs::write(only_lock.join(LOCK_FILE), b"x").unwrap();
    assert!(!store_has_data(&only_lock));
    let store = make_store(tmp.path(), "ai_search");
    assert!(store_has_data(&store));
    assert_eq!(dir_size(&store), 120 + 5000 + 9000 + 3000 + 10);
    assert!(!store_in_use(&store));
    let held = fs::File::open(store.join(LOCK_FILE)).unwrap();
    fs2::FileExt::lock_exclusive(&held).unwrap();
    assert!(store_in_use(&store));
}

#[test]
fn snapshot_name_roundtrip() {
    let name = snapshot_file_name("ai_search", "pre", "3.3.4", now_fixed());
    assert_eq!(name, "ai_search-pre-3.3.4-20261018-120000");
    let p = parse_snapshot_name(&name).unwrap();
    assert_eq!((p.store.as_str(), p.label.as_str(), p.engine.as_str()), ("ai_search", "pre", "3.3.4"));
    assert_eq!(p.created, Some(now_fixed()));
    let legacy = parse_snapshot_name("ai_search-pre-3.3.4-20261018").unwrap();
    assert_eq!(legacy.created, Local.with_ymd_and_hms(2026, 10, 18, 0, 0, 0).single());
    let surql = parse_snapshot_name("my-store-post-3.0.0-beta.3-20260102-030405-2.surql").unwrap();
    assert_eq!((surql.store.as_str(), surql.label.as_str(), surql.engine.as_str()), ("my-store", "post", "3.0.0-beta.3"));
    assert!(parse_snapshot_name("random.surql").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_copies_structure_skips_lock_and_verifies() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let info = create_snapshot_blocking(&store, "pre", Some("3.2.4".into()), now_fixed(), &plenty, &no_progress()).unwrap();
    assert_eq!(info.kind, SnapshotKind::Dir);
    assert_eq!(info.name, "ai_search-pre-3.3.4-20261018-120000");
    assert_eq!(info.engine_version.as_deref(), Some("3.3.4"));
    assert!(info.has_manifest);

    let mut src = tree_listing(&store);
    src.retain(|(n, _)| n != LOCK_FILE);
    let mut dst = tree_listing(&info.path);
    dst.retain(|(n, _)| n != MANIFEST_FILE);
    assert_eq!(src, dst);
    assert!(!info.path.join(LOCK_FILE).exists());
    assert!(info.path.join("discard_stats").is_dir(), "empty directories are preserved");
    assert_eq!(fs::read(info.path.join("vlog/000001.vlog")).unwrap(), fs::read(store.join("vlog/000001.vlog")).unwrap());

    let check = verify_snapshot_dir(&info.path, &no_progress()).unwrap();
    assert!(check.ok && check.has_manifest, "{check:?}");
    assert_eq!(check.files_checked, 5);
    assert_eq!(list_snapshots_for(&store).len(), 1);

    fs::write(info.path.join("vlog/000001.vlog"), b"tampered").unwrap();
    let bad = verify_snapshot_dir(&info.path, &no_progress()).unwrap();
    assert!(!bad.ok);
    assert_eq!(bad.mismatched, vec!["vlog/000001.vlog".to_string()]);
    fs::remove_file(info.path.join("wal/00000001.wal")).unwrap();
    fs::write(info.path.join("stray.bin"), b"x").unwrap();
    let worse = verify_snapshot_dir(&info.path, &no_progress()).unwrap();
    assert_eq!(worse.missing, vec!["wal/00000001.wal".to_string()]);
    assert_eq!(worse.extra, vec!["stray.bin".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_second_snapshots_get_unique_names() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let a = create_snapshot_blocking(&store, "manual", None, now_fixed(), &plenty, &no_progress()).unwrap();
    let b = create_snapshot_blocking(&store, "manual", None, now_fixed(), &plenty, &no_progress()).unwrap();
    assert_ne!(a.path, b.path);
    assert_eq!(b.name, "ai_search-manual-3.3.4-20261018-120000-2");
    assert_eq!(list_snapshots_for(&store).len(), 2);
}

#[test]
fn free_space_guard_uses_injected_probe() {
    let dir = Path::new("/nowhere");
    assert!(check_free_space(1000, dir, &|_| Ok(1000)).is_ok());
    match check_free_space(1001, dir, &|_| Ok(1000)) {
        Err(SnapshotError::InsufficientSpace { needed, available, .. }) => assert_eq!((needed, available), (1001, 1000)),
        other => panic!("expected InsufficientSpace, got {other:?}"),
    }
    assert!(matches!(check_free_space(1, dir, &|_| Err(io::Error::other("boom"))), Err(SnapshotError::Io { .. })));
    assert_eq!(required_space(1000), 1100);
    assert!(available_space(Path::new("/definitely/not/here")).unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insufficient_space_blocks_open_without_creating_snapshot() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let size = dir_size(&store);
    let tight = move |_: &Path| -> io::Result<u64> { Ok(required_space(size) - 1) };
    let err = prepare_open_with(&store, MaintenanceSettings::default(), Arc::new(tight), now_fixed(), no_progress()).await.unwrap_err();
    assert!(matches!(err, SnapshotError::InsufficientSpace { .. }), "{err:?}");
    assert!(err.user_message().contains("Open without snapshot"));
    assert!(list_snapshots_for(&store).is_empty());
    assert_eq!(read_stamp(&store), StampState::Missing, "stamp must not be written by the pre-open phase");

    let exact = move |_: &Path| -> io::Result<u64> { Ok(required_space(size)) };
    let pre = prepare_open_with(&store, MaintenanceSettings::default(), Arc::new(exact), now_fixed(), no_progress()).await.unwrap();
    assert!(pre.snapshot.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_open_decisions() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let space: Arc<SpaceFn> = Arc::new(plenty);

    let fresh = tmp.path().join("db").join("fresh");
    let pre = prepare_open_with(&fresh, MaintenanceSettings::default(), space.clone(), now_fixed(), no_progress()).await.unwrap();
    assert_eq!(pre.change, EngineChange::NoStore);
    assert!(pre.snapshot.is_none());

    let store = make_store(tmp.path(), "ai_search");
    let pre = prepare_open_with(&store, MaintenanceSettings::default(), space.clone(), now_fixed(), no_progress()).await.unwrap();
    assert!(matches!(pre.change, EngineChange::Changed { from: None, .. }));
    let snap = pre.snapshot.expect("missing stamp must trigger a snapshot");
    assert!(snap.name.starts_with("ai_search-pre-"));
    assert!(read_stamp(&store) == StampState::Missing);

    finish_open(&store);
    let st = match read_stamp(&store) {
        StampState::Present(s) => s,
        other => panic!("{other:?}"),
    };
    assert_eq!(st.surrealdb_version, engine_versions().surrealdb);
    assert!(st.store_path.ends_with("ai_search"));
    let pre = prepare_open_with(&store, MaintenanceSettings::default(), space.clone(), now_fixed(), no_progress()).await.unwrap();
    assert_eq!(pre.change, EngineChange::Unchanged);
    assert!(pre.snapshot.is_none());

    let older = stamp_with(&store, "3.2.4", Some("0.21.3"));
    write_stamp(&store, &older).unwrap();
    let later = now_fixed() + chrono::Duration::seconds(5);
    let pre = prepare_open_with(&store, MaintenanceSettings::default(), space.clone(), later, no_progress()).await.unwrap();
    assert!(matches!(&pre.change, EngineChange::Changed { from: Some(f), .. } if f == "3.2.4"));
    assert!(pre.snapshot.is_some());

    let off = MaintenanceSettings { snapshot_on_engine_change: false, ..Default::default() };
    let before = list_snapshots_for(&store).len();
    let pre = prepare_open_with(&store, off, space.clone(), later, no_progress()).await.unwrap();
    assert!(pre.change.is_changed() && pre.snapshot.is_none() && pre.note.is_some());
    assert_eq!(list_snapshots_for(&store).len(), before);

    override_snapshot_once();
    assert!(override_pending());
    let pre = prepare_open_with(&store, MaintenanceSettings::default(), space.clone(), later, no_progress()).await.unwrap();
    assert!(pre.snapshot.is_none() && pre.note.is_some());
    assert!(!override_pending(), "override is single use");
    assert_eq!(list_snapshots_for(&store).len(), before);

    let held = fs::File::open(store.join(LOCK_FILE)).unwrap();
    fs2::FileExt::lock_exclusive(&held).unwrap();
    let err = prepare_open_with(&store, MaintenanceSettings::default(), space, later, no_progress()).await.unwrap_err();
    assert!(matches!(err, SnapshotError::InUse(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_export_runs_only_after_an_engine_change_and_within_the_size_threshold() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let db = mem_db().await;
    seed(&db, 12, 6, 200).await;
    let changed = EngineChange::Changed { from: Some("3.2.4".into()), to: "3.3.4".into(), reason: "test".into() };
    let pre = |change: EngineChange, settings: MaintenanceSettings| PreOpen { change, snapshot: None, note: None, settings, restored: None };

    let skipped = auto_export_if_changed(&db, &store, &pre(EngineChange::Unchanged, MaintenanceSettings::default())).await;
    assert!(matches!(skipped, AutoExport::Skipped(_)));
    let disabled = MaintenanceSettings { auto_export_on_engine_change: false, ..Default::default() };
    assert!(matches!(auto_export_if_changed(&db, &store, &pre(changed.clone(), disabled)).await, AutoExport::Skipped(_)));
    let tiny = MaintenanceSettings { export_size_threshold_bytes: 1, ..Default::default() };
    assert!(matches!(auto_export_if_changed(&db, &store, &pre(changed.clone(), tiny)).await, AutoExport::Skipped(m) if m.contains("threshold")));
    assert!(list_snapshots_for(&store).is_empty());

    let done = auto_export_if_changed(&db, &store, &pre(changed, MaintenanceSettings::default())).await;
    let AutoExport::Exported(path, bytes) = done else { panic!("expected an export, got {done:?}") };
    assert!(bytes > 0 && path.is_file());
    let listed = list_snapshots_for(&store);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].label.as_deref(), Some("post"));
    assert_eq!(listed[0].kind, SnapshotKind::Surql);
    let report = surql::verify_export(&path, VerifyTarget::Live(&db), &VerifyOptions::default(), &no_progress()).await.unwrap();
    assert!(report.ok, "{:?}", report.problems);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn maintenance_panel_renders_headless() {
    let ctx = eframe::egui::Context::default();
    for _ in 0..3 {
        let mut out = ctx.run_ui(eframe::egui::RawInput::default(), |ui| super::maintenance::show(ui));
        out.textures_delta.clear();
        assert!(!out.shapes.is_empty());
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    let info = super::maintenance::info_json(Path::new("./db/ai_search_not_present"));
    assert_eq!(info["engine"]["surrealdb"], engine_versions().surrealdb);
    assert_eq!(info["store"]["exists"], false);
    assert_eq!(info["engine_change"]["state"], "no_store");
}

#[test]
fn blocked_state_marks_only_snapshot_failures_as_overridable() {
    set_blocked(&SnapshotError::InsufficientSpace { needed: 10, available: 5, dir: PathBuf::from("/x") });
    assert!(blocked().unwrap().overridable);
    set_blocked(&SnapshotError::InUse(PathBuf::from("/x")));
    let b = blocked().unwrap();
    assert!(!b.overridable && b.message.contains("in use"));
    clear_blocked();
    assert!(blocked().is_none());
}

fn mk_snapshot_dir(snaps: &Path, name: &str) {
    let d = snaps.join(name);
    fs::create_dir_all(d.join("manifest")).unwrap();
    fs::write(d.join("manifest/x"), b"1").unwrap();
}

#[test]
fn retention_keeps_newest_young_and_latest_pre() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snaps = snapshots_dir(&store);
    let names = [
        "ai_search-pre-3.2.4-20260101-100000",
        "ai_search-manual-3.3.4-20260201-100000",
        "ai_search-manual-3.3.4-20260301-100000",
        "ai_search-manual-3.3.4-20260401-100000",
        "ai_search-manual-3.3.4-20260501-100000",
        "ai_search-manual-3.3.4-20261016-100000",
        "ai_search-manual-3.3.4-20261017-100000",
        "ai_search-manual-3.3.4-20261018-110000",
        "other_store-manual-3.3.4-20260101-100000",
    ];
    for n in names {
        mk_snapshot_dir(&snaps, n);
    }
    for n in ["ai_search-post-3.2.4-20260101-100000.surql", "ai_search-export-3.3.4-20260301-100000.surql", "ai_search-export-3.3.4-20260401-100000.surql", "ai_search-export-3.3.4-20261017-100000.surql", "ai_search-export-3.3.4-20261018-100000.surql"] {
        fs::write(snaps.join(n), b"-- x").unwrap();
    }
    let settings = MaintenanceSettings { retention_keep: 2, retention_min_age_days: 7, ..Default::default() };
    let report = prune_snapshots(&store, &settings, now_fixed());

    let left: Vec<String> = list_snapshots_for(&store).into_iter().map(|s| s.name).collect();
    for kept in [
        "ai_search-manual-3.3.4-20261018-110000",
        "ai_search-manual-3.3.4-20261017-100000",
        "ai_search-manual-3.3.4-20261016-100000",
        "ai_search-pre-3.2.4-20260101-100000",
        "other_store-manual-3.3.4-20260101-100000",
        "ai_search-export-3.3.4-20261018-100000.surql",
        "ai_search-export-3.3.4-20261017-100000.surql",
        "ai_search-post-3.2.4-20260101-100000.surql",
    ] {
        assert!(left.iter().any(|n| n == kept), "{kept} should be kept, left: {left:?}");
    }
    for gone in [
        "ai_search-manual-3.3.4-20260201-100000",
        "ai_search-manual-3.3.4-20260301-100000",
        "ai_search-manual-3.3.4-20260401-100000",
        "ai_search-manual-3.3.4-20260501-100000",
        "ai_search-export-3.3.4-20260301-100000.surql",
        "ai_search-export-3.3.4-20260401-100000.surql",
    ] {
        assert!(!left.iter().any(|n| n == gone), "{gone} should be deleted, left: {left:?}");
    }
    assert_eq!(report.deleted.len(), 6);
    assert!(report.protected.iter().any(|p| p.ends_with("ai_search-pre-3.2.4-20260101-100000")));
    assert!(report.protected.iter().any(|p| p.ends_with("ai_search-post-3.2.4-20260101-100000.surql")));
}

#[test]
fn stale_partials_are_swept_and_listing_ignores_them() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snaps = snapshots_dir(&store);
    mk_snapshot_dir(&snaps, "ai_search-manual-3.3.4-20261018-110000.partial");
    fs::write(snaps.join("ai_search-export-3.3.4-20261018-110000.surql.partial"), b"x").unwrap();
    fs::write(snaps.join("notes.txt"), b"x").unwrap();
    assert!(list_snapshots_for(&store).is_empty());
    let report = prune_snapshots(&store, &MaintenanceSettings::default(), now_fixed());
    assert!(report.deleted.is_empty(), "fresh partials are left alone");
}

#[test]
fn delete_snapshot_is_confined_to_snapshots_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snaps = snapshots_dir(&store);
    mk_snapshot_dir(&snaps, "ai_search-manual-3.3.4-20261018-110000");
    assert!(delete_snapshot(&store, &store).is_err());
    assert!(store.exists());
    assert!(delete_snapshot(&store, &snaps.join("ai_search-manual-3.3.4-20261018-110000")).is_ok());
    assert!(list_snapshots_for(&store).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_moves_current_store_aside_and_never_deletes() {
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snap = create_snapshot_blocking(&store, "pre", Some("3.2.4".into()), now_fixed(), &plenty, &no_progress()).unwrap();
    let original = tree_listing(&store);

    fs::write(store.join("vlog/000001.vlog"), b"changed after snapshot").unwrap();
    fs::write(store.join("extra.file"), b"new").unwrap();
    let later = now_fixed() + chrono::Duration::minutes(3);
    let report = restore_snapshot_blocking(&snap.path, &store, later, &plenty, &no_progress()).unwrap();

    let aside = report.replaced_to.clone().expect("old store is kept");
    assert!(aside.ends_with("ai_search.replaced-20261018-120300"));
    assert_eq!(fs::read(aside.join("extra.file")).unwrap(), b"new");
    let mut restored = tree_listing(&store);
    restored.retain(|(n, _)| n != LOCK_FILE && n != MANIFEST_FILE);
    let mut expected = original;
    expected.retain(|(n, _)| n != LOCK_FILE);
    assert_eq!(restored, expected);
    assert!(!store.join(MANIFEST_FILE).exists());
    assert!(matches!(read_stamp(&store), StampState::Present(s) if s.surrealdb_version == "3.2.4"), "pre snapshot restores the engine that wrote it");

    let tampered = snap.path.join("sstables/000001.sst");
    fs::write(&tampered, b"bad").unwrap();
    let err = restore_snapshot_blocking(&snap.path, &store, later + chrono::Duration::minutes(1), &plenty, &no_progress()).unwrap_err();
    assert!(matches!(err, SnapshotError::Corrupt(_)), "{err:?}");

    fs::write(store.join(LOCK_FILE), b"lockfile").unwrap();
    let held = fs::File::open(store.join(LOCK_FILE)).unwrap();
    fs2::FileExt::lock_exclusive(&held).unwrap();
    let err = restore_snapshot_blocking(&snap.path, &store, later, &plenty, &no_progress()).unwrap_err();
    assert!(matches!(err, SnapshotError::InUse(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn staged_restore_is_applied_by_prepare_open() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let store = make_store(tmp.path(), "ai_search");
    let snap = create_snapshot_blocking(&store, "manual", None, now_fixed(), &plenty, &no_progress()).unwrap();
    fs::write(store.join("extra.file"), b"after snapshot").unwrap();
    finish_open(&store);

    stage_restore(&store, &snap.path).unwrap();
    assert!(pending_restore(&store).is_some());
    let pre = prepare_open(&store).await.unwrap();
    assert!(pre.restored.is_some());
    assert!(pending_restore(&store).is_none(), "pending marker is cleared");
    assert!(!store.join("extra.file").exists());
    let aside: Vec<_> = fs::read_dir(tmp.path().join("db")).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains(".replaced-")).collect();
    assert_eq!(aside.len(), 1);
    assert!(aside[0].path().join("extra.file").exists());
}

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue)]
struct SeedThumb {
    id: RecordId,
    filename: String,
    path: String,
    parent_dir: String,
    size: i64,
    thumbnail_b64: Option<String>,
    tags: Option<Vec<String>>,
    logical_group: RecordId,
}

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue)]
struct SeedEmb {
    thumb_ref: Option<RecordId>,
    path: String,
    hash: Option<String>,
    embedding: Vec<f32>,
}

fn unit_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm = v.iter().map(|f| f * f).sum::<f32>().sqrt();
    v.iter_mut().for_each(|f| *f /= norm);
    v
}

fn fake_b64(seed: usize, len: usize) -> String {
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    (0..len).map(|i| ALPHA[(i * 7 + seed * 13 + i / 5) % 64] as char).collect()
}

pub(super) async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    define_schema(&db).await.unwrap();
    db
}

async fn seed(db: &Surreal<Db>, thumbs: usize, embeds: usize, b64_len: usize) {
    db.query("CREATE logical_groups:default SET name = 'Default'").await.unwrap().check().unwrap();
    let group = RecordId::new("logical_groups", "default");
    let rows: Vec<SeedThumb> = (0..thumbs)
        .map(|i| SeedThumb {
            id: RecordId::new("thumbnails", format!("{i:032x}")),
            filename: format!("img{i}.jpg"),
            path: format!("/pics/img{i}.jpg"),
            parent_dir: "/pics".into(),
            size: 1000 + i as i64,
            thumbnail_b64: Some(fake_b64(i, b64_len)),
            tags: Some(vec!["a".into(), format!("t{}", i % 3)]),
            logical_group: group.clone(),
        })
        .collect();
    db.query("INSERT INTO thumbnails $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
    let embs: Vec<SeedEmb> = (0..embeds)
        .map(|i| SeedEmb {
            thumb_ref: Some(RecordId::new("thumbnails", format!("{i:032x}"))),
            path: format!("/pics/img{i}.jpg"),
            hash: Some(format!("h{i}")),
            embedding: unit_vector(i as u64 + 1, 1024),
        })
        .collect();
    db.query("INSERT INTO clip_embeddings $rows").bind(("rows", embs)).await.unwrap().check().unwrap();
    db.query("CREATE assistant_sessions:s1 SET title = 'Chat 1'").await.unwrap().check().unwrap();
    db.query("CREATE user_settings:main SET group_by_category = true, auto_indexing = false, ai_prompt_template = 'x', overwrite_descriptions = false, filter_skip_icons = true, auto_clip_embeddings = false, clip_augment_with_text = false, clip_overwrite_embeddings = false, recent_paths = [], recent_models = [], auto_save_to_database = true")
        .await
        .unwrap()
        .check()
        .unwrap();
}

#[derive(Debug, Deserialize, SurrealValue)]
struct Hit {
    path: String,
    dist: f32,
}

async fn nearest(db: &Surreal<Db>, probe: Vec<f32>, k: i64) -> Vec<Hit> {
    db.query(format!("SELECT path, vector::distance::knn() AS dist FROM clip_embeddings WHERE embedding <| {k}, 64 |> $vec ORDER BY dist"))
        .bind(("vec", probe))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

async fn export_text(db: &Surreal<Db>, dir: &Path, name: &str) -> (PathBuf, String) {
    let path = dir.join(name);
    let rep = surql::export_atomic(db, &path, &no_progress()).await.unwrap();
    assert!(rep.bytes > 0 && !path.with_extension("surql.partial").exists());
    let text = fs::read_to_string(&path).unwrap();
    (path, text)
}

#[test]
fn schema_ddl_is_shared_and_complete() {
    assert!(SCHEMA_DDL.contains("idx_clip_hnsw ON clip_embeddings FIELDS embedding HNSW DIMENSION 1024 TYPE F32 DIST COSINE EFC 120 M 12"));
    assert!(SCHEMA_DDL.trim_start().starts_with("BEGIN;") && SCHEMA_DDL.trim_end().ends_with("COMMIT;"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_import_roundtrip_with_embeddings_and_hnsw() {
    let tmp = tempfile::tempdir().unwrap();
    let src = mem_db().await;
    seed(&src, 60, 40, 3000).await;
    let (path, text) = export_text(&src, tmp.path(), "round.surql").await;

    assert!(text.contains("HNSW DIMENSION 1024"), "export must carry the HNSW index definition");
    assert!(text.contains("INSERT"), "export must carry records");
    assert!(text.contains("embedding: ["), "export must carry embedding vectors");

    let before = surql::build_manifest(&src, &no_progress()).await.unwrap();
    assert_eq!(before.tables["thumbnails"].count, 60);
    assert_eq!(before.tables["clip_embeddings"].count, 40);
    assert_eq!(before.tables["logical_groups"].count, 1);

    let dst = Surreal::new::<Mem>(()).await.unwrap();
    dst.use_ns(NS).use_db(DB_NAME).await.unwrap();
    dst.import(&path).await.unwrap();
    let after = surql::build_manifest(&dst, &no_progress()).await.unwrap();
    assert_eq!(after.counts(), before.counts());
    assert_eq!(after.tables["thumbnails"].id_checksum, before.tables["thumbnails"].id_checksum);
    assert!(surql::compare_manifests(&before, &after).iter().all(|t| t.ok), "{:?}", surql::compare_manifests(&before, &after));
    assert_eq!(surql::count_rows(&dst, "clip_embeddings").await.unwrap(), 40);

    let probe = unit_vector(8, 1024);
    let hits = nearest(&dst, probe.clone(), 3).await;
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].path, "/pics/img7.jpg");
    assert!(hits[0].dist < 1e-3, "distance {}", hits[0].dist);
    let live_hits = nearest(&src, probe, 3).await;
    assert_eq!(live_hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>(), hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verify_export_accepts_good_and_rejects_damaged_files() {
    let tmp = tempfile::tempdir().unwrap();
    let src = mem_db().await;
    seed(&src, 30, 12, 500).await;
    let (path, text) = export_text(&src, tmp.path(), "good.surql").await;
    let opts = VerifyOptions::default();

    let good = surql::verify_export(&path, VerifyTarget::Live(&src), &opts, &no_progress()).await.unwrap();
    assert!(good.ok, "{:?}", good.problems);
    assert_eq!(good.backend, "kv-mem");
    assert!(good.hnsw_defined);
    assert_eq!(good.imported.tables["thumbnails"].count, 30);
    assert!(!good.live_changed_during_check);

    let manifest = surql::build_manifest(&src, &no_progress()).await.unwrap();
    let by_manifest = surql::verify_export(&path, VerifyTarget::Manifest(&manifest), &opts, &no_progress()).await.unwrap();
    assert!(by_manifest.ok);
    let file_only = surql::verify_export(&path, VerifyTarget::FileOnly, &opts, &no_progress()).await.unwrap();
    assert!(file_only.ok && file_only.expected.is_none());

    let mid = tmp.path().join("truncated.surql");
    fs::write(&mid, &text.as_bytes()[..text.len() / 2]).unwrap();
    let r = surql::verify_export(&mid, VerifyTarget::Live(&src), &opts, &no_progress()).await.unwrap();
    assert!(!r.ok, "truncated file must fail");

    let cut_at = text.find("INSERT [ { created").or_else(|| text.rfind("INSERT")).unwrap();
    let clean_cut = tmp.path().join("clean_cut.surql");
    fs::write(&clean_cut, &text[..cut_at]).unwrap();
    let r = surql::verify_export(&clean_cut, VerifyTarget::Live(&src), &opts, &no_progress()).await.unwrap();
    assert!(!r.ok && r.tables.iter().any(|t| !t.ok), "records dropped at a statement boundary must be caught: {:?}", r.problems);

    let marker = "thumbnails:";
    let at = text.find(&format!("id: {marker}")).unwrap() + 4 + marker.len();
    let key_end = at + text[at..].find(|c: char| c == ',' || c == ' ').unwrap();
    let mut altered = text.clone().into_bytes();
    let idx = key_end - 2;
    altered[idx] = b'z';
    let altered_path = tmp.path().join("altered.surql");
    fs::write(&altered_path, altered).unwrap();
    let r = surql::verify_export(&altered_path, VerifyTarget::Live(&src), &opts, &no_progress()).await.unwrap();
    assert!(!r.ok, "altered record id must change the checksum");
    assert!(r.problems.iter().any(|p| p.contains("checksum") || p.contains("count")), "{:?}", r.problems);

    let missing = surql::verify_export(&tmp.path().join("nope.surql"), VerifyTarget::FileOnly, &opts, &no_progress()).await;
    assert!(missing.is_err());

    let tiny_limit = VerifyOptions { mem_limit_bytes: 1, scratch_dir: Some(tmp.path().join("scratch")), ..Default::default() };
    let disk = surql::verify_export(&path, VerifyTarget::Live(&src), &tiny_limit, &no_progress()).await.unwrap();
    assert!(disk.ok, "{:?}", disk.problems);
    assert_eq!(disk.backend, "surrealkv-temp");
}

async fn wait_unlocked(store: &Path) {
    for _ in 0..100 {
        if !store_in_use(store) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("store {} still locked", store.display());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_into_new_store_on_surrealkv_tempdir() {
    let tmp = tempfile::tempdir().unwrap();
    let src = mem_db().await;
    seed(&src, 25, 10, 400).await;
    let (path, _) = export_text(&src, tmp.path(), "seed.surql").await;
    let expected = surql::build_manifest(&src, &no_progress()).await.unwrap();

    let dir = tmp.path().join("newstore");
    let rep = surql::import_into_new_store(&path, &dir, &no_progress()).await.unwrap();
    assert_eq!(rep.manifest.counts(), expected.counts());
    assert_eq!(rep.total_records, expected.total_records());
    assert!(store_has_data(&dir));
    assert!(surql::import_into_new_store(&path, &dir, &no_progress()).await.is_err(), "refuses a populated target");
    assert!(surql::import_into_new_store(&tmp.path().join("missing.surql"), &tmp.path().join("other"), &no_progress()).await.is_err());

    wait_unlocked(&dir).await;
    let reopened = Surreal::new::<SurrealKv>(dir.to_string_lossy().as_ref()).await.unwrap();
    reopened.use_ns(NS).use_db(DB_NAME).await.unwrap();
    let again = surql::build_manifest(&reopened, &no_progress()).await.unwrap();
    assert_eq!(again, expected);
    let hits = nearest(&reopened, unit_vector(4, 1024), 2).await;
    assert_eq!(hits[0].path, "/pics/img3.jpg");
}

#[test]
fn compare_manifests_reports_each_kind_of_difference() {
    let digest = |count: u64, sum: &str| surql::TableDigest { count, id_checksum: sum.into(), fields: vec!["a".into()], indexes: vec!["IDX".into()] };
    let mut expected = Manifest::default();
    expected.tables.insert("t1".into(), digest(3, "aa"));
    expected.tables.insert("t2".into(), digest(1, "bb"));
    expected.tables.insert("only_live".into(), digest(1, "cc"));
    let mut actual = Manifest::default();
    actual.tables.insert("t1".into(), digest(2, "aa"));
    actual.tables.insert("t2".into(), surql::TableDigest { indexes: vec![], ..digest(1, "zz") });
    actual.tables.insert("only_file".into(), digest(1, "dd"));
    let checks = surql::compare_manifests(&expected, &actual);
    let get = |n: &str| checks.iter().find(|c| c.table == n).unwrap();
    assert!(get("t1").problems.iter().any(|p| p.contains("record count")));
    assert!(get("t2").problems.iter().any(|p| p.contains("checksum")) && get("t2").problems.iter().any(|p| p.contains("indexes")));
    assert!(get("only_live").problems[0].contains("missing"));
    assert!(get("only_file").problems[0].contains("only in file"));
    assert!(checks.iter().all(|c| !c.ok));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_end_to_end_info_snapshot_export_verify_import() {
    let _g = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let src = mem_db().await;
    seed(&src, 20, 8, 300).await;
    let (seed_file, _) = export_text(&src, tmp.path(), "seed.surql").await;
    let store = tmp.path().join("db").join("ai_search");
    let s = |v: &str| v.to_string();
    let path_arg = store.to_string_lossy().into_owned();

    let code = tokio::task::spawn_blocking({
        let (seed_file, path_arg) = (seed_file.to_string_lossy().into_owned(), path_arg.clone());
        move || super::cli::run(&[s("--db-import"), seed_file, s("--db-path"), path_arg])
    })
    .await
    .unwrap();
    assert_eq!(code, Some(0));
    wait_unlocked(&store).await;

    let run = |args: Vec<String>| async move { tokio::task::spawn_blocking(move || super::cli::run(&args)).await.unwrap() };
    assert_eq!(run(vec![s("--db-info"), s("--db-path"), path_arg.clone()]).await, Some(0));
    assert_eq!(run(vec![s("--db-snapshot"), s("--db-path"), path_arg.clone()]).await, Some(0));
    assert_eq!(list_snapshots_for(&store).len(), 1);

    let out = tmp.path().join("out.surql");
    let out_arg = out.to_string_lossy().into_owned();
    assert_eq!(run(vec![s("--db-export"), out_arg.clone(), s("--db-path"), path_arg.clone()]).await, Some(0));
    assert!(out.is_file());
    wait_unlocked(&store).await;
    let snaps = list_snapshots_for(&store);
    assert!(snaps.iter().any(|x| x.label.as_deref() == Some("pre")), "unstamped store must be snapshotted on first CLI open: {snaps:?}");
    assert!(matches!(read_stamp(&store), StampState::Present(_)));

    assert_eq!(run(vec![s("--db-verify"), out_arg.clone(), s("--db-path"), path_arg.clone()]).await, Some(0));
    wait_unlocked(&store).await;
    assert_eq!(run(vec![s("--db-verify"), out_arg.clone(), s("--no-compare")]).await, Some(0));
    let cut = tmp.path().join("cut.surql");
    let bytes = fs::read(&out).unwrap();
    fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    assert_eq!(run(vec![s("--db-verify"), cut.to_string_lossy().into_owned(), s("--no-compare")]).await, Some(1));
    assert_eq!(run(vec![s("--db-import"), out_arg.clone(), s("--db-path"), path_arg.clone()]).await, Some(2), "populated store needs --force");

    let snap_dir = list_snapshots_for(&store).into_iter().find(|x| x.kind == SnapshotKind::Dir).unwrap();
    assert_eq!(run(vec![s("--db-verify"), snap_dir.path.to_string_lossy().into_owned()]).await, Some(0));

    let held = fs::File::open(store.join(LOCK_FILE)).unwrap();
    fs2::FileExt::lock_exclusive(&held).unwrap();
    assert_eq!(run(vec![s("--db-export"), tmp.path().join("blocked.surql").to_string_lossy().into_owned(), s("--db-path"), path_arg]).await, Some(3));
}

fn rss_mb() -> (u64, u64) {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| status.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) / 1024;
    (field("VmRSS:"), field("VmHWM:"))
}

fn env_num(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual benchmark driven by BENCH_DIR and BENCH_STAGE"]
async fn bench_stage() {
    let Ok(dir) = std::env::var("BENCH_DIR") else { return };
    let stage = std::env::var("BENCH_STAGE").unwrap_or_else(|_| "gen".into());
    let root = PathBuf::from(dir);
    let store = root.join("db").join("ai_search");
    let started = std::time::Instant::now();
    let progress: ProgressFn = Arc::new(|_, _, _| {});
    match stage.as_str() {
        "gen" => {
            let (thumbs, embeds, b64) = (env_num("BENCH_THUMBS", 50_000), env_num("BENCH_EMBEDS", 20_000), env_num("BENCH_B64", 20_000));
            fs::create_dir_all(&store).unwrap();
            let db = Surreal::new::<SurrealKv>(store.to_string_lossy().as_ref()).await.unwrap();
            db.use_ns(NS).use_db(DB_NAME).await.unwrap();
            define_schema(&db).await.unwrap();
            db.query("CREATE logical_groups:default SET name = 'Default'").await.unwrap().check().unwrap();
            let group = RecordId::new("logical_groups", "default");
            for start in (0..thumbs).step_by(500) {
                let rows: Vec<SeedThumb> = (start..(start + 500).min(thumbs))
                    .map(|i| SeedThumb {
                        id: RecordId::new("thumbnails", format!("{i:032x}")),
                        filename: format!("img{i}.jpg"),
                        path: format!("/pics/dir{}/img{i}.jpg", i % 97),
                        parent_dir: format!("/pics/dir{}", i % 97),
                        size: 1000 + i as i64,
                        thumbnail_b64: Some(fake_b64(i, b64)),
                        tags: Some(vec!["a".into(), format!("t{}", i % 3)]),
                        logical_group: group.clone(),
                    })
                    .collect();
                db.query("INSERT INTO thumbnails $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
            }
            println!("BENCH gen thumbnails done at {:.1}s", started.elapsed().as_secs_f32());
            for start in (0..embeds).step_by(200) {
                let rows: Vec<SeedEmb> = (start..(start + 200).min(embeds))
                    .map(|i| SeedEmb {
                        thumb_ref: Some(RecordId::new("thumbnails", format!("{:032x}", i % thumbs))),
                        path: format!("/pics/dir{}/img{i}.jpg", i % 97),
                        hash: Some(format!("h{i}")),
                        embedding: unit_vector(i as u64 + 1, 1024),
                    })
                    .collect();
                db.query("INSERT INTO clip_embeddings $rows").bind(("rows", rows)).await.unwrap().check().unwrap();
                if start % 4000 == 0 {
                    println!("BENCH gen embeddings {start}/{embeds} at {:.1}s", started.elapsed().as_secs_f32());
                }
            }
            drop(db);
            wait_unlocked(&store).await;
            finish_open(&store);
            println!("BENCH gen thumbs={thumbs} embeds={embeds} b64={b64} secs={:.1} store_bytes={}", started.elapsed().as_secs_f32(), dir_size(&store));
        }
        "snapshot" => {
            let size = dir_size(&store);
            let info = create_snapshot_blocking(&store, "manual", None, Local::now(), &available_space, &progress).unwrap();
            let secs = started.elapsed().as_secs_f32();
            println!("BENCH snapshot bytes={size} secs={secs:.2} mb_per_s={:.0} snapshot={}", size as f32 / 1e6 / secs, info.path.display());
        }
        "verify_snapshot" => {
            let snap = std::env::var("BENCH_SNAPSHOT").unwrap();
            let check = verify_snapshot_dir(Path::new(&snap), &progress).unwrap();
            println!("BENCH verify_snapshot ok={} bytes={} secs={:.2}", check.ok, check.bytes_checked, started.elapsed().as_secs_f32());
        }
        "export" => {
            let opened = surql::open_store(&store).await.unwrap();
            let open_secs = started.elapsed().as_secs_f32();
            let out = root.join("export.surql");
            let rep = surql::export_atomic(&opened.db, &out, &progress).await.unwrap();
            println!("BENCH export open_secs={open_secs:.1} export_secs={:.1} bytes={} snapshot_taken={}", rep.millis as f32 / 1000.0, rep.bytes, opened.pre.snapshot.is_some());
            let t = std::time::Instant::now();
            let m = surql::build_manifest(&opened.db, &progress).await.unwrap();
            println!("BENCH manifest secs={:.1} records={}", t.elapsed().as_secs_f32(), m.total_records());
            surql::close_store(opened, &store).await;
        }
        "open_only" | "manifest" => {
            let opened = surql::open_store(&store).await.unwrap();
            println!("BENCH open secs={:.1} rss_after_open_mb={}", started.elapsed().as_secs_f32(), rss_mb().0);
            if stage == "manifest" {
                let t = std::time::Instant::now();
                let m = surql::build_manifest(&opened.db, &progress).await.unwrap();
                println!("BENCH manifest secs={:.1} records={} rss_mb={}", t.elapsed().as_secs_f32(), m.total_records(), rss_mb().0);
            }
            surql::close_store(opened, &store).await;
        }
        "import" => {
            let target = root.join("newstore");
            let rep = surql::import_into_new_store(&root.join("export.surql"), &target, &progress).await.unwrap();
            println!("BENCH import secs={:.1} records={} store_bytes={}", rep.import_millis as f32 / 1000.0, rep.total_records, dir_size(&target));
        }
        "verify_mem" | "verify_disk" => {
            let opened = surql::open_store(&store).await.unwrap();
            let opts = VerifyOptions { mem_limit_bytes: if stage == "verify_mem" { u64::MAX } else { 0 }, scratch_dir: Some(root.join("scratch")), ..Default::default() };
            let t = std::time::Instant::now();
            let rep = surql::verify_export(&root.join("export.surql"), VerifyTarget::Live(&opened.db), &opts, &progress).await.unwrap();
            println!("BENCH {stage} ok={} backend={} import_secs={:.1} total_secs={:.1} problems={:?}", rep.ok, rep.backend, rep.import_millis as f32 / 1000.0, t.elapsed().as_secs_f32(), rep.problems);
            surql::close_store(opened, &store).await;
        }
        other => panic!("unknown BENCH_STAGE {other}"),
    }
    let (rss, peak) = rss_mb();
    println!("BENCH stage={stage} wall_secs={:.1} rss_mb={rss} peak_rss_mb={peak}", started.elapsed().as_secs_f32());
}

#[test]
fn a_failed_schema_transaction_reports_the_statement_with_its_own_error() {
    let errors = vec![
        (0, "The query was not executed due to a failed transaction".to_string()),
        (109, "Cannot COMMIT: Transaction conflict: Transaction write conflict. This transaction can be retried".to_string()),
        (3, "The query was not executed due to a failed transaction".to_string()),
    ];
    let why = super::schema_failure(errors).unwrap();
    assert!(why.starts_with("schema statement 109: Cannot COMMIT"), "{why}");
    assert_eq!(super::schema_failure(Vec::new()), None);
    let only = super::schema_failure(vec![(2, "The query was not executed due to a failed transaction".to_string())]).unwrap();
    assert!(only.starts_with("schema statement 2:"), "without another cause the first error is reported");
}

#[tokio::test]
async fn schema_setup_is_tried_again_after_a_transaction_conflict_but_not_after_other_errors() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let conflict = || anyhow::anyhow!("schema statement 109: Cannot COMMIT: Transaction conflict: Transaction write conflict. This transaction can be retried");
    assert!(super::is_retriable(&conflict()));
    assert!(!super::is_retriable(&anyhow::anyhow!("schema statement 4: parse error")));

    let calls = AtomicU32::new(0);
    let result = super::retry_conflicts("test", || async { if calls.fetch_add(1, Ordering::SeqCst) < 2 { Err(conflict()) } else { Ok(7) } }).await;
    assert_eq!(result.unwrap(), 7);
    assert_eq!(calls.load(Ordering::SeqCst), 3, "two conflicts, then success");

    let calls = AtomicU32::new(0);
    let failed: anyhow::Result<()> = super::retry_conflicts("test", || async {
        calls.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("schema statement 4: parse error"))
    })
    .await;
    assert!(failed.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1, "other errors are not retried");

    let calls = AtomicU32::new(0);
    let gave_up: anyhow::Result<()> = super::retry_conflicts("test", || async {
        calls.fetch_add(1, Ordering::SeqCst);
        Err(conflict())
    })
    .await;
    assert!(gave_up.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 5, "one try and four repeats");
}
