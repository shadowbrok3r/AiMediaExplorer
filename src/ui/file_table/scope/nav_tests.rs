//! A tab's navigation as a whole: scopes opened, history walked, batches applied, restores and failures.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use egui::{CentralPanel, Context, Pos2, RawInput, Rect, vec2};

use super::*;
use crate::command::{Scope, SmartView};
use crate::database::Thumbnail;
use crate::ui::file_table::thumbs::service::{Producer, ServiceConfig, ThumbService};
use crate::ui::file_table::thumbs::store::MemStore;
use crate::ui::file_table::thumbs::ThumbPresence;
use crate::ui::file_table::{AIUpdate, FileExplorer, SimilarResult};
use crate::ui::test_support::test_ctx;

fn row(path: &str) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
    t.path = path.to_string();
    t.file_type = "jpg".into();
    t
}

fn batch(paths: &[&str]) -> RowBatch {
    RowBatch { rows: paths.iter().map(|p| row(p)).collect(), ..Default::default() }
}

fn library_paths(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("/lib/img_{i:04}.jpg")).collect()
}

fn rows_of(paths: &[String]) -> RowBatch {
    RowBatch { rows: paths.iter().map(|p| row(p)).collect(), ..Default::default() }
}

/// Sends `body` as if the loader of the current load had.
fn deliver(ex: &FileExplorer, body: LoadBody) {
    let id = ex.nav.active.as_ref().expect("a load is running").id;
    ex.nav.tx.send(LoadMsg { load: id, body }).unwrap();
}

fn pump(ex: &mut FileExplorer) {
    ex.receive_scope(&Context::default());
}

fn outcome(rows: usize) -> Outcome {
    Outcome { rows, batches: 1, elapsed: Duration::from_millis(5), first_batch: Some(Duration::from_millis(2)), ..Default::default() }
}

fn folders() -> (tempfile::TempDir, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let (a, b) = (a.display().to_string(), b.display().to_string());
    (dir, a, b)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_tab_shows_its_folder_with_nothing_to_go_back_to() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a.clone()), false).unwrap();
    assert_eq!(ex.scope(), &Scope::Folder(a.clone()));
    assert_eq!(ex.current_path, a);
    assert!(!ex.can_go_back() && !ex.can_go_forward());
    assert!(ex.can_go_up());
    assert_eq!(ex.nav.history.len(), 1);
    assert_eq!(ex.nav.input, format!("fs:{a}"));
}

#[tokio::test(flavor = "multi_thread")]
async fn opening_scopes_walks_a_history_that_back_and_forward_follow() {
    let (_dir, a, b) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a.clone()), false).unwrap();
    ex.open_scope(Scope::Folder(b.clone())).unwrap();
    ex.open_scope(Scope::Library).unwrap();
    assert_eq!(ex.nav.history.len(), 3);
    assert!(ex.go_back());
    assert_eq!(ex.scope(), &Scope::Folder(b.clone()));
    assert_eq!(ex.current_path, b);
    assert!(ex.go_back());
    assert_eq!(ex.scope(), &Scope::Folder(a.clone()));
    assert!(!ex.go_back(), "the first entry has nothing before it");
    assert!(ex.go_forward() && ex.go_forward());
    assert_eq!(ex.scope(), &Scope::Library);
    assert!(!ex.go_forward());
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_scope_after_going_back_drops_what_was_forward() {
    let (_dir, a, b) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a), false).unwrap();
    ex.open_scope(Scope::Folder(b)).unwrap();
    ex.go_back();
    ex.open_scope(Scope::Tag("x".into())).unwrap();
    assert!(!ex.can_go_forward());
    assert_eq!(ex.nav.history.len(), 2);
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn up_goes_to_the_parent_folder_and_a_query_scope_goes_up_to_the_library() {
    let (dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a), false).unwrap();
    assert!(ex.go_up());
    assert_eq!(ex.scope(), &Scope::Folder(dir.path().display().to_string()));
    ex.open_scope(Scope::Tag("t".into())).unwrap();
    assert!(ex.go_up());
    assert_eq!(ex.scope(), &Scope::Library);
    assert!(!ex.can_go_up() && !ex.go_up(), "the library is the top");
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_folder_that_does_not_exist_cannot_be_opened_and_the_tab_stays_as_it_was() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a.clone()), false).unwrap();
    let err = ex.open_scope(Scope::Folder("/definitely/not/here".into())).unwrap_err();
    assert!(err.contains("not a folder that exists"), "{err}");
    assert_eq!(ex.scope(), &Scope::Folder(a));
    assert_eq!(ex.nav.history.len(), 1);
    assert!(ex.start_at(Scope::Folder("/definitely/not/here".into()), false).is_err());
    assert!(ex.open_scope(Scope::Archive("zip:///definitely/not/here.zip!/".into())).is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn going_back_to_a_folder_that_has_vanished_stays_put_and_says_why() {
    let (_dir, a, b) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a.clone()), false).unwrap();
    ex.open_scope(Scope::Folder(b.clone())).unwrap();
    std::fs::remove_dir_all(&a).unwrap();
    assert!(!ex.go_back());
    assert_eq!(ex.scope(), &Scope::Folder(b));
    assert_eq!(ex.nav.history.position(), 1, "the history did not move");
    assert!(ex.nav.error.as_deref().is_some_and(|e| e.contains("not a folder")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_query_scope_starts_a_load_and_its_batches_fill_the_view() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Tag("sunset".into())).unwrap();
    assert!(ex.is_loading());
    assert_eq!(ex.viewer.mode, crate::ui::file_table::table::ExplorerMode::Database);
    assert!(ex.table.is_empty());
    let paths = library_paths(30);
    deliver(&ex, LoadBody::Rows(rows_of(&paths[..20])));
    pump(&mut ex);
    assert_eq!(ex.table.len(), 20);
    deliver(&ex, LoadBody::Rows(rows_of(&paths[20..])));
    deliver(&ex, LoadBody::Done(outcome(30)));
    pump(&mut ex);
    assert_eq!(ex.table.len(), 30);
    assert!(!ex.is_loading());
    assert_eq!(ex.nav.log.len(), 1);
    let record = ex.nav.log.back().unwrap();
    assert_eq!((record.scope.as_str(), record.kind, record.rows), ("tag:sunset", "tag", 30));
    assert_eq!(ex.nav.active.as_ref().map(|a| (a.rows, a.batches)), Some((30, 2)));
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_from_the_library_are_badged_indexed_and_the_ones_made_from_disk_are_not() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Collection("Trip".into())).unwrap();
    let mut b = batch(&["/lib/a.jpg", "/lib/b.jpg"]);
    b.unindexed = vec!["/lib/b.jpg".into()];
    b.embedded = vec!["/lib/a.jpg".into()];
    deliver(&ex, LoadBody::Rows(b));
    pump(&mut ex);
    assert!(ex.viewer.badges.is_indexed("/lib/a.jpg"));
    assert!(!ex.viewer.badges.is_indexed("/lib/b.jpg"));
    assert_eq!(ex.viewer.badges.badge("/lib/b.jpg"), Badge::NotIndexed);
    assert!(ex.viewer.clip_presence.contains("/lib/a.jpg") && !ex.viewer.clip_presence.contains("/lib/b.jpg"));
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn messages_of_an_older_or_cancelled_load_are_ignored() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Library).unwrap();
    let old = ex.nav.active.as_ref().unwrap().id;
    ex.open_scope(Scope::Recent).unwrap();
    ex.nav.tx.send(LoadMsg { load: old, body: LoadBody::Rows(batch(&["/stale/x.jpg"])) }).unwrap();
    ex.nav.tx.send(LoadMsg { load: old, body: LoadBody::Failed("late".into()) }).unwrap();
    pump(&mut ex);
    assert!(ex.table.is_empty() && ex.nav.error.is_none(), "the older load no longer counts");
    ex.cancel_load();
    deliver(&ex, LoadBody::Rows(batch(&["/cancelled/x.jpg"])));
    pump(&mut ex);
    assert!(ex.table.is_empty(), "rows that arrive after a cancel are dropped");
    assert!(!ex.is_loading());
}

#[tokio::test(flavor = "multi_thread")]
async fn leaving_a_query_scope_for_a_folder_drops_what_the_loader_still_sends() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Library).unwrap();
    let id = ex.nav.active.as_ref().unwrap().id;
    ex.open_scope(Scope::Folder(a)).unwrap();
    ex.nav.tx.send(LoadMsg { load: id, body: LoadBody::Rows(batch(&["/late/x.jpg"])) }).unwrap();
    pump(&mut ex);
    assert!(!ex.table.contains_path("/late/x.jpg"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_search_that_has_more_offers_load_more_and_the_next_page_appends() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Semantic("a cat".into())).unwrap();
    assert!(ex.viewer.showing_similarity, "the similarity column is there from the start");
    let paths = library_paths(10);
    let mut first = rows_of(&paths);
    first.scores = paths.iter().enumerate().map(|(i, p)| (p.clone(), 0.9 - i as f32 * 0.01)).collect();
    deliver(&ex, LoadBody::Rows(first));
    deliver(&ex, LoadBody::Done(Outcome { more: true, next_start: 10, vector: Some(Arc::new(vec![0.5; 4])), ..outcome(10) }));
    pump(&mut ex);
    assert_eq!(ex.viewer.similar_scores.len(), 10);
    assert!(ex.nav.more.is_some());
    assert!(ex.load_more(), "a further page starts");
    assert!(ex.nav.more.is_none(), "until it ends the next page is not offered again");
    assert!(ex.nav.active.as_ref().unwrap().page);
    assert_eq!(ex.table.len(), 10, "the rows already shown stay");
    assert!(!ex.load_more(), "no second page while one is loading");
    let more = library_paths(25);
    deliver(&ex, LoadBody::Rows(rows_of(&more[10..])));
    deliver(&ex, LoadBody::Done(Outcome { more: false, ..outcome(15) }));
    pump(&mut ex);
    assert_eq!(ex.table.len(), 25);
    assert!(ex.nav.more.is_none() && !ex.load_more());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_load_keeps_the_rows_shows_the_error_and_a_reload_clears_it() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Library).unwrap();
    deliver(&ex, LoadBody::Rows(batch(&["/lib/a.jpg"])));
    deliver(&ex, LoadBody::Failed("query timed out".into()));
    pump(&mut ex);
    assert_eq!(ex.table.len(), 1);
    assert_eq!(ex.nav.error.as_deref(), Some("query timed out"));
    assert!(!ex.is_loading());
    let record = ex.nav.log.back().unwrap();
    assert_eq!(record.error.as_deref(), Some("query timed out"));
    let entries = ex.nav.history.len();
    ex.reload_scope();
    assert!(ex.nav.error.is_none() && ex.table.is_empty() && ex.is_loading());
    assert_eq!(ex.nav.history.len(), entries, "a reload is not a new history entry");
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_load_is_logged_as_cancelled_and_adds_no_next_page() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Semantic("q".into())).unwrap();
    ex.cancel_load();
    deliver(&ex, LoadBody::Done(Outcome { cancelled: true, more: true, next_start: 5, ..outcome(5) }));
    pump(&mut ex);
    assert!(ex.nav.log.back().unwrap().cancelled);
    assert!(ex.nav.more.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_address_box_opens_scope_strings_and_paths_and_reports_what_it_cannot_read() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.nav.input = format!("  {a}  ");
    ex.submit_address().unwrap();
    assert_eq!(ex.scope(), &Scope::Folder(a));
    ex.nav.input = "tag:sunset".into();
    ex.submit_address().unwrap();
    assert_eq!(ex.scope(), &Scope::Tag("sunset".into()));
    assert_eq!(ex.nav.input, "tag:sunset");
    ex.nav.input = "wat:1".into();
    assert!(ex.submit_address().unwrap_err().contains("unknown prefix"));
    assert_eq!(ex.scope(), &Scope::Tag("sunset".into()), "bad text changes nothing");
    ex.nav.input = "/definitely/not/here".into();
    assert!(ex.submit_address().is_err());
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_tab_title_comes_from_the_scope_until_one_is_given_and_long_ones_are_cut() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Tag("sunset".into())).unwrap();
    assert_eq!(ex.tab_title(), "Tag: sunset");
    ex.nav.title = Some("Pinned".into());
    assert_eq!(ex.tab_title(), "Pinned");
    ex.open_scope(Scope::Semantic("a very long search phrase that does not fit in a tab".into())).unwrap();
    assert_eq!(ex.tab_title().chars().count(), 36);
    assert!(ex.tab_title().ends_with("..."));
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cached_scan_gives_the_tab_the_title_it_was_saved_with() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Scan(1760000000001)).unwrap();
    assert_eq!(ex.tab_title(), "Scan #1760000000001");
    deliver(&ex, LoadBody::Done(Outcome { title: Some("Scan: /photos".into()), ..outcome(0) }));
    pump(&mut ex);
    assert_eq!(ex.tab_title(), "Scan: /photos");
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_held_in_memory_have_no_scope_to_reload_and_no_history() {
    let mut ex = FileExplorer::new(true);
    ex.show_rows(Some("Selected (2)".into()), vec![row("/s/a.jpg"), row("/s/b.jpg")]);
    assert_eq!(ex.tab_title(), "Selected (2)");
    assert!(ex.nav.rows_only && !ex.can_go_back() && !ex.can_go_up());
    ex.reload_scope();
    assert_eq!(ex.table.len(), 2, "a reload cannot read them again, so it leaves them");
    ex.open_scope(Scope::Tag("x".into())).unwrap();
    assert!(!ex.nav.rows_only && ex.table.is_empty());
    ex.cancel_load();
}

#[tokio::test(flavor = "multi_thread")]
async fn similarity_results_from_other_parts_of_the_app_replace_this_tab_instead_of_opening_one() {
    let mut ex = FileExplorer::new(true);
    ex.viewer.ai_update_tx.send(AIUpdate::SimilarResults {
        origin_path: "/a/origin.jpg".into(),
        results: vec![
            SimilarResult { thumb: row("/a/1.jpg"), created: None, updated: None, similarity_score: Some(0.5), clip_similarity_score: Some(0.8) },
            SimilarResult { thumb: row("/a/2.jpg"), created: None, updated: None, similarity_score: Some(0.4), clip_similarity_score: None },
        ],
    })
    .unwrap();
    ex.receive_ai_update(&Context::default());
    assert_eq!(ex.scope(), &Scope::Similar("/a/origin.jpg".into()));
    assert_eq!(ex.table.len(), 2);
    assert!(ex.viewer.showing_similarity);
    assert_eq!(ex.viewer.similar_scores.get("/a/1.jpg"), Some(&0.8));
    assert_eq!(ex.viewer.similar_scores.get("/a/2.jpg"), Some(&0.4));
    assert!(ex.can_go_back(), "the view it replaced is one step back");
    ex.viewer.ai_update_tx.send(AIUpdate::SimilarResults { origin_path: "query:red car".into(), results: vec![SimilarResult { thumb: row("/a/3.jpg"), created: None, updated: None, similarity_score: Some(0.3), clip_similarity_score: None }] }).unwrap();
    ex.receive_ai_update(&Context::default());
    assert_eq!(ex.scope(), &Scope::Semantic("red car".into()));
    assert_eq!(ex.table.len(), 1);
    ex.viewer.ai_update_tx.send(AIUpdate::SimilarResults { origin_path: "query:red car".into(), results: vec![SimilarResult { thumb: row("/a/4.jpg"), created: None, updated: None, similarity_score: Some(0.2), clip_similarity_score: None }] }).unwrap();
    ex.receive_ai_update(&Context::default());
    assert_eq!(ex.table.len(), 2, "more results of the same search append");
    assert_eq!(ex.nav.history.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_selected_row_comes_back_when_a_scope_is_revisited() {
    let mut ex = FileExplorer::new(true);
    ex.open_scope(Scope::Library).unwrap();
    let paths = library_paths(40);
    deliver(&ex, LoadBody::Rows(rows_of(&paths)));
    deliver(&ex, LoadBody::Done(outcome(40)));
    pump(&mut ex);
    ex.sync_view();
    ex.select_paths([paths[17].as_str()]);
    ex.open_scope(Scope::Tag("t".into())).unwrap();
    assert!(ex.table.is_empty() && ex.selection.is_empty());
    assert!(ex.go_back());
    assert_eq!(ex.nav.restore.as_ref().and_then(|r| r.anchor.clone()), Some(paths[17].clone()));
    deliver(&ex, LoadBody::Rows(rows_of(&paths[..10])));
    pump(&mut ex);
    assert!(ex.nav.restore.is_some(), "the row has not arrived yet, so the restore waits");
    deliver(&ex, LoadBody::Rows(rows_of(&paths[10..])));
    pump(&mut ex);
    assert_eq!(ex.selected_rows().iter().map(|r| r.path.clone()).collect::<Vec<_>>(), vec![paths[17].clone()]);
    assert!(ex.nav.restore.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restore_that_never_finds_its_row_is_dropped_when_the_load_ends() {
    let mut ex = FileExplorer::new(true);
    ex.nav.restore = Some(Restore { scroll_row: Some(500), anchor: Some("/gone/x.jpg".into()) });
    ex.open_scope(Scope::Library).unwrap();
    ex.nav.restore = Some(Restore { scroll_row: Some(500), anchor: Some("/gone/x.jpg".into()) });
    deliver(&ex, LoadBody::Rows(batch(&["/lib/a.jpg"])));
    pump(&mut ex);
    assert!(ex.nav.restore.is_some(), "still loading");
    deliver(&ex, LoadBody::Done(outcome(1)));
    pump(&mut ex);
    assert!(ex.nav.restore.is_none() && ex.selection.is_empty());
}

struct NoProducer;

impl Producer for NoProducer {
    fn request(&self, _path: &Path) -> bool {
        true
    }
}

fn frame(ctx: &Context, ex: &mut FileExplorer, clock: &mut f64) {
    *clock += 0.016;
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1100.0, 600.0))), time: Some(*clock), ..Default::default() };
    let out = ctx.run_ui(input, |ui| {
        CentralPanel::default().show(ui, |ui| ex.details_panel(ui));
    });
    out.drop_without_applying_deltas();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_first_visible_row_is_saved_on_leaving_and_scrolled_back_to_on_return() {
    let svc = ThumbService::new(MemStore::new(), tokio::runtime::Handle::current(), ServiceConfig::default(), Arc::new(NoProducer));
    let mut ex = FileExplorer::new(true);
    ex.viewer.thumbs = svc.clone();
    ex.viewer.thumb_cache = ThumbPresence::new(svc);
    let ctx = test_ctx();
    egui_extras::install_image_loaders(&ctx);
    let mut clock = 0.0;
    ex.open_scope(Scope::Library).unwrap();
    let paths = library_paths(300);
    deliver(&ex, LoadBody::Rows(rows_of(&paths)));
    deliver(&ex, LoadBody::Done(outcome(300)));
    pump(&mut ex);
    for _ in 0..3 {
        frame(&ctx, &mut ex, &mut clock);
    }
    assert_eq!(ex.details.first_visible(), Some(0));
    ex.details.scroll_to_top_row(120);
    for _ in 0..4 {
        frame(&ctx, &mut ex, &mut clock);
    }
    let shown = ex.details.first_visible().expect("rows are drawn");
    assert!((119..=121).contains(&shown), "scrolled to row {shown}");
    ex.open_scope(Scope::Tag("t".into())).unwrap();
    assert_eq!(ex.nav.history.entries()[1].scroll_row, shown, "the position was saved before leaving");
    assert!(ex.go_back());
    assert_eq!(ex.nav.restore.as_ref().and_then(|r| r.scroll_row), Some(shown));
    deliver(&ex, LoadBody::Rows(rows_of(&paths)));
    deliver(&ex, LoadBody::Done(outcome(300)));
    pump(&mut ex);
    assert!(ex.nav.restore.is_none());
    for _ in 0..4 {
        frame(&ctx, &mut ex, &mut clock);
    }
    let back = ex.details.first_visible().expect("rows are drawn");
    assert!(back.abs_diff(shown) <= 1, "returned to row {back}, left at {shown}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_finished_folder_listing_is_recorded_with_its_row_count() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    ex.start_at(Scope::Folder(a), false).unwrap();
    let id = ex.scan_id().expect("the listing has a scan");
    ex.table.push(row("/x/one.jpg"));
    ex.scan_tx.send(crate::ScanEnvelope { scan_id: id, msg: crate::utilities::scan::ScanMsg::Done }).unwrap();
    for _ in 0..200 {
        ex.receive_scan(&Context::default());
        if !ex.nav.log.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let record = ex.nav.log.back().expect("the listing was logged");
    assert_eq!(record.kind, "folder");
    assert!(record.rows >= 1);
    assert!(!ex.is_loading());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_probe_facts_of_a_tab_come_from_its_scan_and_its_password_queue() {
    let (_dir, a, _) = folders();
    let mut ex = FileExplorer::new(true);
    assert_eq!(ex.scan_id(), None);
    assert_eq!(ex.archive_prompt_count(), 0);
    ex.start_at(Scope::Folder(a), false).unwrap();
    assert!(ex.scan_id().is_some());
    ex.scan_tx.send(crate::ScanEnvelope { scan_id: ex.scan_id().unwrap(), msg: crate::utilities::scan::ScanMsg::EncryptedArchives(vec!["/z/a.zip".into(), "/z/b.zip".into(), "/z/a.zip".into()]) }).unwrap();
    for _ in 0..200 {
        ex.receive_scan(&Context::default());
        if ex.archive_prompt_count() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(ex.archive_prompt_count(), 2, "an archive waits once");
    let probe = crate::ui::taskbar::adapter::explorer_probe("Tab", &ex);
    assert_eq!((probe.scan_id, probe.archive_prompts), (ex.scan_id(), 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn smart_scopes_open_like_any_other_scope() {
    let mut ex = FileExplorer::new(true);
    for view in SmartView::ALL {
        ex.open_scope(Scope::Smart(view)).unwrap();
        assert_eq!(ex.scope(), &Scope::Smart(view));
        assert!(ex.nav.input.starts_with("smart:"));
    }
    ex.cancel_load();
    assert_eq!(ex.nav.history.len(), SmartView::ALL.len() + 1);
}
