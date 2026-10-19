use std::time::Duration;

use super::*;
use crate::ui::file_table::menus::db_delete::GLOBAL_DB_TESTS;
use crate::ui::file_table::test_rig::{Rig, db, global_db};

fn folder_scope(path: &std::path::Path) -> Scope {
    Scope::Folder(path.display().to_string())
}

#[test]
fn opening_fills_the_folder_from_the_argument_or_the_tab_once() {
    let mut d = ScanDialog::default();
    d.open_for(Some("/a".into()), &Scope::Library);
    assert!(d.open);
    assert_eq!(d.folder, "/a");
    d.open_for(None, &Scope::Folder("/b".into()));
    assert_eq!(d.folder, "/a", "a folder already chosen is kept");
    let mut fresh = ScanDialog::default();
    fresh.open_for(None, &Scope::Folder("/b".into()));
    assert_eq!(fresh.folder, "/b");
    let mut search = ScanDialog::default();
    search.open_for(Some("   ".into()), &Scope::Tag("x".into()));
    assert!(search.folder.is_empty(), "a tag scope has no folder to offer");
}

#[test]
fn the_dialog_only_accepts_a_folder_that_exists() {
    let dir = tempfile::tempdir().unwrap();
    let mut d = ScanDialog::default();
    assert!(d.scope().unwrap_err().contains("Choose a folder"));
    d.folder = "/definitely/not/here".into();
    assert!(d.scope().unwrap_err().contains("not a folder that exists"));
    d.folder = format!("  {}  ", dir.path().display());
    assert_eq!(d.scope().unwrap(), folder_scope(dir.path()));
}

#[test]
fn the_filters_shown_are_those_of_the_spec_as_chips() {
    let mut spec = FilterSpec::none();
    assert!(filter_labels(&spec).is_empty());
    spec.size.min = Some(10 * 1024);
    spec.skip_icons = true;
    spec.excluded_dirs = vec!["/p/cache".into()];
    let labels = filter_labels(&spec);
    assert_eq!(labels.len(), 3, "{labels:?}");
    assert!(labels.iter().any(|l| l.contains("10")) && labels.iter().any(|l| l.contains("icons")) && labels.iter().any(|l| l.contains("cache")));
    let defaults = filter_labels(&FilterSpec::from_settings(&crate::UiSettings::default()));
    assert!(defaults.iter().any(|l| l.contains("10")), "the default minimum size is a visible chip: {defaults:?}");
}

#[test]
fn a_library_request_is_taken_once_per_scan_of_that_folder() {
    request_library_save("/lib-request/a");
    request_library_save("/lib-request/a");
    assert!(!take_library_save("/lib-request/other"));
    assert!(take_library_save("/lib-request/a"));
    assert!(take_library_save("/lib-request/a"));
    assert!(!take_library_save("/lib-request/a"));
}

#[test]
fn starting_in_a_new_tab_queues_an_open_scope_command_and_the_library_request() {
    let _serial = crate::command::bus::testing::serial();
    let dir = tempfile::tempdir().unwrap();
    let mut rig = Rig::new(1000.0, 600.0);
    rig.ex.toolbar.scan.folder = dir.path().display().to_string();
    rig.ex.toolbar.scan.library = true;
    let text = rig.ex.start_scan().unwrap();
    assert!(text.contains("subfolders") && text.contains("join the library"), "{text}");
    let root = dir.path().display().to_string();
    let sent = crate::command::bus::testing::take_matching(|c| matches!(c, AppCommand::OpenScope { scope: Scope::Folder(p), target: Target::NewTab, recursive: true, .. } if *p == root));
    assert_eq!(sent.len(), 1);
    assert!(take_library_save(&root));
    assert_eq!(rig.ex.toolbar.scan.starts, 1);
}

#[test]
fn starting_in_this_tab_scans_here_without_a_new_tab() {
    let _serial = crate::command::bus::testing::serial();
    let dir = tempfile::tempdir().unwrap();
    let mut rig = Rig::new(1000.0, 600.0);
    rig.ex.toolbar.scan.folder = dir.path().display().to_string();
    rig.ex.toolbar.scan.target = ScanTarget::ThisTab;
    rig.ex.toolbar.scan.recursive = false;
    rig.ex.start_scan().unwrap();
    assert_eq!(rig.ex.scope(), &folder_scope(dir.path()));
    assert!(crate::command::bus::testing::take_matching(|c| matches!(c, AppCommand::OpenScope { .. })).is_empty());
    rig.ex.toolbar.scan.folder = "/no/such/folder".into();
    assert!(rig.ex.start_scan().is_err());
    assert_eq!(rig.ex.scope(), &folder_scope(dir.path()), "a bad folder changes nothing");
}

#[test]
fn the_dialog_is_drawn_with_its_fields_and_the_filters_that_apply() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.edit_filters(|s| s.size.min = Some(20 * 1024));
    rig.ex.toolbar.scan.target = ScanTarget::ThisTab;
    rig.ex.toolbar.scan.open_for(Some("/definitely/not/here".into()), &Scope::Library);
    rig.settle(3);
    for text in ["Scan folder", "Folder", "Include subfolders", "Add the files found to the library", "Show the results in", "What the results list", "Start scan", "Close"] {
        assert!(rig.has_text(text), "{text} is not drawn");
    }
    assert!(rig.has_text("Size >= 20 KiB"), "the tab's own filter is previewed");
    assert!(rig.has_text("/definitely/not/here is not a folder that exists"), "the reason a scan cannot start is shown");
}

#[test]
fn the_start_button_scans_and_closes_the_dialog_and_escape_closes_it_too() {
    let _serial = crate::command::bus::testing::serial();
    let dir = tempfile::tempdir().unwrap();
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.toolbar.scan.target = ScanTarget::ThisTab;
    rig.ex.toolbar.scan.open_for(Some(dir.path().display().to_string()), &Scope::Library);
    rig.settle(3);
    let start = rig.text_pos("Start scan").unwrap();
    rig.click(start + vec2(10.0, 6.0));
    assert!(!rig.ex.toolbar.scan.open);
    assert_eq!(rig.ex.scope(), &folder_scope(dir.path()));
    rig.ex.toolbar.scan.open = true;
    rig.settle(2);
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(!rig.ex.toolbar.scan.open);
}

fn wait_scan_done(rig: &mut Rig) {
    for _ in 0..500 {
        rig.settle(1);
        if rig.ex.scan_done {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the scan did not finish");
}

#[test]
fn a_scan_loads_every_file_and_the_filter_decides_what_is_listed() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("deep").join("deeper");
    std::fs::create_dir_all(&sub).unwrap();
    let small = [dir.path().join("tiny.jpg"), sub.join("icon.png")];
    let big = [dir.path().join("big.jpg"), sub.join("photo.png")];
    for p in &small {
        std::fs::write(p, vec![0u8; 100]).unwrap();
    }
    for p in &big {
        std::fs::write(p, vec![0u8; 40 * 1024]).unwrap();
    }
    std::fs::write(sub.join("clip.mp4"), vec![0u8; 50]).unwrap();
    let mut rig = Rig::new(1000.0, 600.0);
    rig.ex.edit_filters(|s| s.size.min = Some(10 * 1024));
    rig.ex.open_scope_recursive(folder_scope(dir.path()), true).unwrap();
    wait_scan_done(&mut rig);
    let files = rig.ex.table.iter().filter(|r| r.file_type != "<DIR>").count();
    assert_eq!(files, 5, "small files, small videos and big files are all loaded: {:?}", rig.ex.table.iter().map(|r| r.path.clone()).collect::<Vec<_>>());
    rig.settle(2);
    let listed: Vec<String> = rig.ex.view.rows(&rig.ex.table).filter(|r| r.file_type != "<DIR>").map(|r| r.filename.clone()).collect();
    assert_eq!(listed.len(), 2, "only the files of 10 KiB and more are listed: {listed:?}");
    rig.ex.clear_filters();
    rig.settle(2);
    assert_eq!(rig.ex.view.rows(&rig.ex.table).filter(|r| r.file_type != "<DIR>").count(), 5, "removing the chip brings the small files back without a rescan");
}

#[test]
fn a_scan_started_with_the_library_option_saves_what_it_found_as_a_task() {
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<std::path::PathBuf> = (0..3).map(|i| dir.path().join(format!("scansave-{i}.jpg"))).collect();
    for p in &paths {
        std::fs::write(p, vec![0u8; 30 * 1024]).unwrap();
    }
    let mut rig = Rig::new(1000.0, 600.0);
    request_library_save(&dir.path().display().to_string());
    rig.ex.open_scope_recursive(folder_scope(dir.path()), false).unwrap();
    wait_scan_done(&mut rig);
    let title_part = "scanned files to the library";
    let mut state = None;
    for _ in 0..300 {
        rig.settle(1);
        state = crate::tasks::with(|r| r.tasks().iter().find(|t| t.title.contains(title_part) && t.total == 3).map(|t| t.state));
        if state == Some(crate::tasks::TaskState::Completed) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(state, Some(crate::tasks::TaskState::Completed), "the save shows as a finished task");
    let stored: Vec<String> = db(async { crate::database::DB.query("SELECT VALUE path FROM thumbnails WHERE path CONTAINS 'scansave-'").await.unwrap().take(0).unwrap() });
    assert_eq!(stored.len(), 3);
    db(async {
        crate::database::DB.query("DELETE thumbnails WHERE path CONTAINS 'scansave-'").await.unwrap();
    });
}
