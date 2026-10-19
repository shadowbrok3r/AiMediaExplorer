//! The Refine tab: field choices, the filtered list, accepts through the journal, the store listing and headless frames.

use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{CentralPanel, Context, Pos2, RawInput, Rect, vec2};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::SurrealValue;

use super::accept;
use super::model::*;
use super::*;
use crate::ai::mcp::clock::FakeClock;
use crate::ai::mcp::proposals::{Changes, NewProposal, ProposalStore};
use crate::ai::refine::Proposal as Generated;
use crate::database::journal;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed, TagChange};
use crate::ui::test_support::{test_ctx, texts};

fn strings(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn generated(path: &str, new_category: Option<&str>, new_tags: &[&str], new_caption: Option<&str>, confidence: Option<f32>, generator: &str) -> Generated {
    Generated {
        path: path.into(),
        current_category: Some("old".into()),
        current_tags: strings(&["keep", "drop"]),
        current_caption: Some("old caption".into()),
        current_description: None,
        new_category: new_category.map(str::to_string),
        new_tags: strings(new_tags),
        new_caption: new_caption.map(str::to_string),
        new_description: None,
        generator: generator.into(),
        confidence,
    }
}

fn local(n: u64, path: &str) -> Proposal {
    Proposal::from_generated(ProposalKey::Local(n), &generated(path, Some("beach"), &["keep", "sea"], Some("A new caption"), Some(0.8), "Heuristic"), 1000 + n).expect("it changes something")
}

fn stored(id: &str, path: &str, changes: Changes, current: &Current) -> Proposal {
    let s = crate::ai::mcp::proposals::Proposal { id: id.into(), path: path.into(), changes, reason: Some("looks like it".into()), source: "tester".into(), status: crate::ai::mcp::proposals::Status::Pending, created_ms: 500, decided_ms: None, note: None };
    Proposal::from_stored(&s, current)
}

#[test]
fn a_generated_proposal_keeps_only_the_fields_that_change() {
    let p = Proposal::from_generated(ProposalKey::Local(1), &generated("/a/x.jpg", Some("old"), &["keep", "drop"], Some("New text"), None, "Jina M0"), 7).unwrap();
    assert_eq!(p.changes.iter().map(|c| c.kind).collect::<Vec<_>>(), vec![FieldKind::Caption], "same category and same tags are not changes");
    assert_eq!(p.source, ProposalSource::Reranker("Jina M0".into()));
    assert!(p.changes.iter().all(|c| c.accepted), "every field starts checked");
    assert!(Proposal::from_generated(ProposalKey::Local(2), &generated("/a/x.jpg", None, &[], None, None, "Heuristic"), 7).is_none());
}

#[test]
fn sources_come_from_the_generator_name() {
    assert_eq!(ProposalSource::from_generator("Heuristic"), ProposalSource::Heuristic);
    assert_eq!(ProposalSource::from_generator("Heuristic (fallback)"), ProposalSource::Heuristic);
    assert_eq!(ProposalSource::from_generator("cloud:openrouter"), ProposalSource::Cloud("openrouter".into()));
    assert_eq!(ProposalSource::from_generator("Jina M0").label(), "Jina M0");
    assert_eq!(ProposalSource::Mcp("claude".into()).label(), "MCP claude");
    assert_eq!(ProposalSource::Mcp(String::new()).label(), "MCP");
    assert_eq!(ProposalSource::Other("Agent".into()).label(), "Agent");
    assert!(ProposalSource::Mcp("x".into()).is_mcp() && !ProposalSource::Heuristic.is_mcp());
}

#[test]
fn a_stored_proposal_shows_what_applying_it_would_leave() {
    let current = Current { category: Some("beach".into()), tags: strings(&["sea", "boat"]), caption: Some("Old".into()), description: None };
    let changes = Changes { caption: Some("Old".into()), category: Some("coast".into()), tags_add: strings(&["SEA", "gold"]), tags_remove: strings(&["boat"]), ..Default::default() };
    let p = stored("p1", "/a/x.jpg", changes, &current);
    assert_eq!(p.key, ProposalKey::Mcp("p1".into()));
    assert_eq!(p.changes.iter().map(|c| c.kind).collect::<Vec<_>>(), vec![FieldKind::Category, FieldKind::Tags], "the equal caption is not a change");
    assert_eq!(p.change(FieldKind::Tags).unwrap().proposed, FieldValue::Tags(strings(&["sea", "gold"])));
    assert_eq!((p.reason.as_deref(), p.confidence, p.created_ms), (Some("looks like it"), None, 500));
    assert!(p.source.is_mcp());
}

#[test]
fn tags_are_marked_kept_added_or_removed_ignoring_case() {
    let (now, next) = tag_diff(&strings(&["Sea", "boat"]), &strings(&["sea", "gold"]));
    assert_eq!(now, vec![("Sea".to_string(), TagMark::Kept), ("boat".to_string(), TagMark::Removed)]);
    assert_eq!(next, vec![("sea".to_string(), TagMark::Kept), ("gold".to_string(), TagMark::Added)]);
    assert_eq!(clipped("short", 10), "short");
    assert_eq!(clipped("a long sentence here", 6), "a long...");
}

#[test]
fn only_checked_fields_are_in_the_mask_and_the_patch() {
    let mut p = local(1, "/a/x.jpg");
    assert_eq!(p.accepted_count(), 3);
    p.changes.iter_mut().find(|c| c.kind == FieldKind::Tags).unwrap().accepted = false;
    let mask = p.mask();
    assert_eq!((mask.category, mask.tags, mask.caption, mask.description), (true, false, true, false));
    let patch = p.patch();
    assert_eq!(patch.category, Some(Some("beach".to_string())));
    assert_eq!(patch.caption, Some(Some("A new caption".to_string())));
    assert!(patch.tags.is_none() && patch.description.is_none(), "an unchecked field is not written");
    p.changes.iter_mut().for_each(|c| c.accepted = false);
    assert!(p.patch().is_empty() && p.mask().is_empty());
}

#[test]
fn a_bulk_accept_covers_only_the_shown_proposals_and_their_checked_fields() {
    let mut all = vec![local(1, "/beach/a.jpg"), local(2, "/beach/b.jpg"), local(3, "/city/c.jpg"), local(4, "/city/d.jpg")];
    all[1].changes.iter_mut().for_each(|c| c.accepted = false);
    all[3].changes.iter_mut().find(|c| c.kind == FieldKind::Caption).unwrap().accepted = false;
    let everything = visible(&all, &ListFilter::default());
    let plan = plan_accept(&all, &everything);
    assert_eq!((plan.files(), plan.fields, plan.unchecked), (3, 8, 1), "the proposal with nothing checked is left out");
    assert_eq!(plan.by_kind, [3, 3, 2, 0], "category, tags, caption, description");
    assert_eq!(plan.summary(), "8 fields in 3 files (category 3, tags 3, caption 2)");

    let city = ListFilter { text: "/city/".into(), ..Default::default() };
    let shown = visible(&all, &city);
    assert_eq!(shown, vec![2, 3]);
    let plan = plan_accept(&all, &shown);
    assert_eq!((plan.files(), plan.fields), (2, 5));
    assert!(plan.items.iter().all(|i| i.path.starts_with("/city/")), "proposals the filter hides are not touched");
    assert_eq!(plan.items[1].mask.caption, false, "the unchecked caption of the second city file stays out");
}

#[test]
fn an_empty_selection_plans_nothing() {
    let mut all = vec![local(1, "/a.jpg"), local(2, "/b.jpg")];
    set_all(&mut all, &[0, 1], false);
    let plan = plan_accept(&all, &[0, 1]);
    assert!(plan.is_empty());
    assert_eq!((plan.fields, plan.unchecked), (0, 2));
    assert!(plan_accept(&all, &[]).is_empty());
    set_all(&mut all, &[1], true);
    assert_eq!(plan_accept(&all, &[0, 1]).files(), 1);
    assert!(plan_accept(&all, &[99]).is_empty(), "an index outside the list is ignored");
}

#[test]
fn the_list_filters_by_text_source_and_confidence_and_sorts() {
    let mut a = local(1, "/z/a.jpg");
    a.confidence = Some(0.9);
    let mut b = local(2, "/y/b.jpg");
    b.confidence = Some(0.2);
    b.created_ms = 9_000;
    let c = stored("p1", "/x/c.jpg", Changes { caption: Some("Gotham".into()), ..Default::default() }, &Current::default());
    let all = vec![a, b, c];
    let by = |f: ListFilter| visible(&all, &f);
    assert_eq!(by(ListFilter::default()), vec![2, 1, 0], "path order");
    assert_eq!(by(ListFilter { sort: SortKey::Confidence, ..Default::default() }), vec![0, 1, 2], "most confident first, unrated last");
    assert_eq!(by(ListFilter { sort: SortKey::Newest, ..Default::default() }), vec![1, 0, 2], "newest first");
    assert_eq!(by(ListFilter { source: Some("MCP tester".into()), ..Default::default() }), vec![2]);
    assert_eq!(by(ListFilter { source: Some("Heuristic".into()), ..Default::default() }), vec![1, 0]);
    assert_eq!(by(ListFilter { min_confidence: 0.5, ..Default::default() }), vec![2, 0], "unrated proposals stay unless excluded");
    assert_eq!(by(ListFilter { min_confidence: 0.5, include_unrated: false, ..Default::default() }), vec![0]);
    assert_eq!(by(ListFilter { text: "GOTHAM".into(), ..Default::default() }), vec![2], "the filter reads proposed text too");
    assert_eq!(by(ListFilter { text: "sea".into(), ..Default::default() }).len(), 2, "and proposed tags");
    assert_eq!(source_labels(&all), vec!["Heuristic".to_string(), "MCP tester".to_string()]);
    assert!(ListFilter { text: "x".into(), ..Default::default() }.is_narrowing() && !ListFilter::default().is_narrowing());
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
}

fn seed(path: &str) -> RowSeed {
    RowSeed::from_path(path)
}

#[derive(Debug, SurrealValue)]
struct Meta {
    caption: Option<String>,
    category: Option<String>,
    tags: Option<Vec<String>>,
}

async fn meta(db: &Surreal<Db>, path: &str) -> Meta {
    db.query("SELECT caption, category, tags FROM thumbnails WHERE path = $p").bind(("p", path.to_string())).await.unwrap().take::<Vec<Meta>>(0).unwrap().into_iter().next().expect("row")
}

async fn library(db: &Surreal<Db>, paths: &[&str]) {
    for p in paths {
        let patch = MetaPatch { caption: Some(Some("old caption".into())), category: Some(Some("old".into())), tags: Some(strings(&["keep", "drop"])), ..Default::default() };
        metadata_edit::save_patch(db, &seed(p), &patch).await.unwrap();
    }
    journal::clear_with(db).await.unwrap();
}

#[tokio::test]
async fn accepting_writes_only_the_checked_fields_as_one_undoable_batch() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg", "/p/b.jpg"]).await;
    let mut all = vec![local(1, "/p/a.jpg"), local(2, "/p/b.jpg")];
    all[0].changes.iter_mut().find(|c| c.kind == FieldKind::Tags).unwrap().accepted = false;
    let plan = plan_accept(&all, &[0, 1]);
    let out = accept::apply(&DbSource::Handle(db.clone()), None, plan).await;
    assert_eq!((out.done.len(), out.failed.len(), out.fields), (2, 0, 5));
    let a = meta(&db, "/p/a.jpg").await;
    assert_eq!((a.category.as_deref(), a.caption.as_deref(), a.tags), (Some("beach"), Some("A new caption"), Some(strings(&["keep", "drop"]))), "the unchecked tags stay");
    assert_eq!(meta(&db, "/p/b.jpg").await.tags, Some(strings(&["keep", "sea"])));
    assert_eq!(out.edits.len(), 2, "the browsers get the stored rows");
    let batches = journal::batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.len(), 1, "one batch for the whole accept");
    assert_eq!((batches[0].source.as_str(), batches[0].entries, batches[0].files), (accept::LOCAL_LABEL, 5, 2));
    let report = journal::undo_last_with(&db).await.unwrap();
    assert_eq!((report.applied.len(), report.conflicts.len()), (5, 0));
    assert_eq!(meta(&db, "/p/b.jpg").await.category.as_deref(), Some("old"));
    assert_eq!(meta(&db, "/p/b.jpg").await.tags, Some(strings(&["keep", "drop"])));
}

#[tokio::test]
async fn an_empty_plan_writes_nothing() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let mut all = vec![local(1, "/p/a.jpg")];
    set_all(&mut all, &[0], false);
    let out = accept::apply(&DbSource::Handle(db.clone()), None, plan_accept(&all, &[0])).await;
    assert!(out.done.is_empty() && out.failed.is_empty());
    assert_eq!(meta(&db, "/p/a.jpg").await.category.as_deref(), Some("old"));
    assert_eq!(journal::stats_with(&db).await.unwrap().entries, 0);
}

fn store_over(db: &Surreal<Db>) -> Arc<ProposalStore> {
    Arc::new(ProposalStore::new(DbSource::Handle(db.clone()), FakeClock::new()))
}

async fn file_proposal(store: &ProposalStore, path: &str, changes: Changes) -> String {
    store.create(NewProposal { path: path.into(), changes: changes.normalized().unwrap(), reason: Some("because".into()), source: "tester".into() }).await.unwrap().0.id
}

#[tokio::test]
async fn the_store_lists_pending_proposals_with_the_values_their_files_have() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let store = store_over(&db);
    let id = file_proposal(&store, "/p/a.jpg", Changes { caption: Some("Fresh".into()), tags_add: strings(&["gold"]), tags_remove: strings(&["drop"]), ..Default::default() }).await;
    let listed = store::list_pending(&store, &DbSource::Handle(db.clone())).await.unwrap();
    assert_eq!(listed.len(), 1);
    let p = &listed[0];
    assert_eq!(p.key, ProposalKey::Mcp(id));
    assert_eq!(p.source.label(), "MCP tester");
    assert_eq!(p.change(FieldKind::Caption).unwrap().current, FieldValue::Text(Some("old caption".into())));
    assert_eq!(p.change(FieldKind::Tags).unwrap().proposed, FieldValue::Tags(strings(&["keep", "gold"])));
    store.reject(&p.key_id()).await.unwrap();
    assert!(store::list_pending(&store, &DbSource::Handle(db)).await.unwrap().is_empty(), "a rejected proposal is no longer pending");
}

trait KeyId {
    fn key_id(&self) -> String;
}

impl KeyId for Proposal {
    fn key_id(&self) -> String {
        match &self.key {
            ProposalKey::Mcp(id) => id.clone(),
            ProposalKey::Local(n) => n.to_string(),
        }
    }
}

#[tokio::test]
async fn stored_proposals_accept_only_the_checked_fields_through_the_store_and_the_journal() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg", "/p/b.jpg"]).await;
    let store = store_over(&db);
    file_proposal(&store, "/p/a.jpg", Changes { caption: Some("Fresh".into()), category: Some("coast".into()), tags_add: strings(&["gold"]), ..Default::default() }).await;
    file_proposal(&store, "/p/b.jpg", Changes { category: Some("coast".into()), ..Default::default() }).await;
    let mut listed = store::list_pending(&store, &DbSource::Handle(db.clone())).await.unwrap();
    listed.sort_by(|a, b| a.path.cmp(&b.path));
    listed[0].changes.iter_mut().find(|c| c.kind == FieldKind::Category).unwrap().accepted = false;
    let plan = plan_accept(&listed, &[0, 1]);
    let out = accept::apply(&DbSource::Handle(db.clone()), Some(&store), plan).await;
    assert_eq!((out.done.len(), out.failed.len(), out.fields), (2, 0, 3));
    let a = meta(&db, "/p/a.jpg").await;
    assert_eq!((a.caption.as_deref(), a.category.as_deref(), a.tags), (Some("Fresh"), Some("old"), Some(strings(&["keep", "drop", "gold"]))));
    assert_eq!(meta(&db, "/p/b.jpg").await.category.as_deref(), Some("coast"));
    let batches = journal::batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.len(), 1, "the store accepts as one batch");
    assert_eq!(batches[0].source, "Accepted MCP proposals");
    assert_eq!(store.counts().await.unwrap().accepted, 2);
    journal::undo_last_with(&db).await.unwrap();
    assert_eq!(meta(&db, "/p/b.jpg").await.category.as_deref(), Some("old"));
}

#[tokio::test]
async fn rejecting_stored_proposals_goes_through_the_store_and_local_ones_need_no_write() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let store = store_over(&db);
    let id = file_proposal(&store, "/p/a.jpg", Changes { caption: Some("Fresh".into()), ..Default::default() }).await;
    let keys = vec![ProposalKey::Mcp(id.clone()), ProposalKey::Local(4), ProposalKey::Mcp("pmissing".into())];
    let out = accept::reject(Some(&store), &keys).await;
    assert_eq!(out.done, vec![ProposalKey::Mcp(id), ProposalKey::Local(4)]);
    assert_eq!(out.failed.len(), 1);
    assert_eq!(meta(&db, "/p/a.jpg").await.caption.as_deref(), Some("old caption"));
    let none = accept::reject(None, &[ProposalKey::Mcp("p1".into())]).await;
    assert!(none.done.is_empty() && none.failed[0].1.contains("not available"));
}

#[tokio::test]
async fn a_missing_file_fails_its_proposal_and_keeps_it_listed() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let store = store_over(&db);
    let id = file_proposal(&store, "/p/a.jpg", Changes { caption: Some("Fresh".into()), ..Default::default() }).await;
    db.query("DELETE thumbnails WHERE path = '/p/a.jpg'").await.unwrap().check().unwrap();
    let key = ProposalKey::Mcp(id);
    let plan = BulkPlan { items: vec![PlanItem { key: key.clone(), path: "/p/a.jpg".into(), mask: crate::ai::mcp::proposals::FieldMask::ALL, patch: None }], fields: 1, by_kind: [0, 0, 1, 0], unchecked: 0 };
    let out = accept::apply(&DbSource::Handle(db), Some(&store), plan).await;
    assert!(out.done.is_empty());
    assert_eq!(out.failed.len(), 1);
    assert!(out.failed[0].1.contains("no longer in the library"), "{:?}", out.failed);
}

fn rig(db: &Surreal<Db>) -> (RefinementsPanel, crossbeam::channel::Receiver<(egui_toast::ToastKind, String)>) {
    let (toast_tx, toast_rx) = crossbeam::channel::unbounded();
    let thumbs = crate::ui::file_table::thumbs::service();
    (RefinementsPanel::with_sources(toast_tx, DbSource::Handle(db.clone()), StoreSource::Fixed(store_over(db)), thumbs), toast_rx)
}

async fn pump_until(panel: &mut RefinementsPanel, done: impl Fn(&RefinementsPanel) -> bool) {
    for _ in 0..200 {
        panel.pump();
        if done(panel) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the panel did not reach the expected state");
}

#[tokio::test]
async fn generated_batches_add_to_the_list_and_replace_a_file_s_earlier_proposal() {
    let db = mem_db().await;
    let (mut panel, toasts) = rig(&db);
    let tx = panel.sender();
    panel.start_run_state_for_tests();
    tx.send(Msg::Progress { done: 1, total: 4 }).unwrap();
    tx.send(Msg::Generated(vec![generated("/p/a.jpg", Some("beach"), &[], None, Some(0.5), "Heuristic"), generated("/p/b.jpg", None, &["x"], None, None, "Jina M0")])).unwrap();
    tx.send(Msg::Generated(vec![generated("/p/a.jpg", Some("coast"), &[], None, Some(0.7), "Heuristic")])).unwrap();
    panel.pump();
    assert_eq!(panel.run_state(), &RunState::Running { done: 1, total: 4 });
    assert_eq!(panel.proposals().len(), 2, "the second proposal for a.jpg replaced the first");
    assert_eq!(panel.proposals().iter().find(|p| p.path == "/p/a.jpg").unwrap().change(FieldKind::Category).unwrap().proposed, FieldValue::Text(Some("coast".into())));
    tx.send(Msg::Finished { cancelled: false, scanned: 4 }).unwrap();
    panel.pump();
    assert_eq!(panel.run_state(), &RunState::Done { scanned: 4, produced: 3, cancelled: false });
    assert!(toasts.try_iter().any(|(_, t)| t.contains("Proposals ready (3)")));
    panel.clear_local();
    assert!(panel.proposals().is_empty());
}

impl RefinementsPanel {
    fn start_run_state_for_tests(&mut self) {
        self.run = RunState::Running { done: 0, total: 0 };
        self.run_produced = 0;
    }
}

#[tokio::test]
async fn a_failed_run_keeps_the_error_and_a_cancelled_one_says_so() {
    let db = mem_db().await;
    let (mut panel, toasts) = rig(&db);
    let tx = panel.sender();
    panel.start_run_state_for_tests();
    tx.send(Msg::Failed("no API key".into())).unwrap();
    panel.pump();
    assert_eq!(panel.run_state(), &RunState::Failed("no API key".into()));
    assert!(toasts.try_iter().any(|(k, t)| k == egui_toast::ToastKind::Error && t.contains("no API key")));
    panel.start_run_state_for_tests();
    tx.send(Msg::Finished { cancelled: true, scanned: 2 }).unwrap();
    panel.pump();
    assert_eq!(panel.run_state(), &RunState::Done { scanned: 2, produced: 0, cancelled: true });
    assert!(toasts.try_iter().next().is_none(), "a stopped run does not toast");
}

#[tokio::test]
async fn accepting_one_proposal_writes_it_and_drops_it_from_the_list() {
    let _serial = crate::command::bus::testing::serial();
    let db = mem_db().await;
    library(&db, &["/p/a.jpg", "/p/b.jpg"]).await;
    let (mut panel, toasts) = rig(&db);
    let tx = panel.sender();
    tx.send(Msg::Generated(vec![generated("/p/a.jpg", Some("beach"), &[], Some("New"), Some(0.9), "Heuristic"), generated("/p/b.jpg", Some("city"), &[], None, Some(0.4), "Heuristic")])).unwrap();
    panel.pump();
    let key = panel.proposals()[0].key.clone();
    panel.accept_one(&key);
    assert!(panel.is_applying(&key));
    pump_until(&mut panel, |p| p.proposals().len() == 1).await;
    assert!(!panel.is_applying(&key));
    assert_eq!(meta(&db, "/p/a.jpg").await.category.as_deref(), Some("beach"));
    assert_eq!(meta(&db, "/p/b.jpg").await.category.as_deref(), Some("old"), "the other file is untouched");
    assert_eq!((panel.tally.applied_fields, panel.tally.applied_files), (2, 1));
    assert!(toasts.try_iter().any(|(k, t)| k == egui_toast::ToastKind::Success && t.contains("Applied 2 fields in 1 file") && t.contains("Ctrl+Z")));
    let edits = crate::ui::journal_ui::take_edits();
    assert!(edits.iter().any(|e| matches!(e, crate::ui::inspector::RowEdit::Patch { path, .. } if path == "/p/a.jpg")));
}

#[tokio::test]
async fn nothing_checked_means_nothing_is_asked_or_written() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let (mut panel, _toasts) = rig(&db);
    panel.sender().send(Msg::Generated(vec![generated("/p/a.jpg", Some("beach"), &[], None, None, "Heuristic")])).unwrap();
    panel.pump();
    set_all(&mut panel.proposals, &[0], false);
    panel.ask_accept_view();
    assert!(panel.confirm.is_none(), "no confirmation for an empty selection");
    assert!(panel.notice.as_ref().is_some_and(|(ok, t)| !ok && t.contains("Nothing is checked")));
    let key = panel.proposals()[0].key.clone();
    panel.accept_one(&key);
    assert!(!panel.is_applying(&key));
    assert_eq!(meta(&db, "/p/a.jpg").await.category.as_deref(), Some("old"));
}

#[tokio::test]
async fn a_bulk_accept_asks_first_and_applies_only_what_the_filter_shows() {
    let db = mem_db().await;
    library(&db, &["/beach/a.jpg", "/city/b.jpg"]).await;
    let (mut panel, _toasts) = rig(&db);
    panel.sender().send(Msg::Generated(vec![generated("/beach/a.jpg", Some("beach"), &[], None, None, "Heuristic"), generated("/city/b.jpg", Some("city"), &[], None, None, "Heuristic")])).unwrap();
    panel.pump();
    panel.filter.text = "/city/".into();
    panel.ask_accept_view();
    let Some(Confirm::Accept(plan)) = panel.confirm.clone() else { panic!("a confirmation is pending") };
    assert_eq!((plan.files(), plan.fields), (1, 1));
    assert_eq!(meta(&db, "/city/b.jpg").await.category.as_deref(), Some("old"), "nothing is written before the answer");
    panel.confirm_pending();
    pump_until(&mut panel, |p| p.proposals().len() == 1).await;
    assert_eq!(meta(&db, "/city/b.jpg").await.category.as_deref(), Some("city"));
    assert_eq!(meta(&db, "/beach/a.jpg").await.category.as_deref(), Some("old"), "the hidden proposal is untouched");
    assert_eq!(panel.proposals()[0].path, "/beach/a.jpg");
    panel.filter.text.clear();
    panel.ask_reject_view();
    assert!(matches!(panel.confirm, Some(Confirm::Reject(ref k)) if k.len() == 1));
    panel.confirm_pending();
    pump_until(&mut panel, |p| p.proposals().is_empty()).await;
    assert_eq!(panel.tally.rejected, 1);
}

#[tokio::test]
async fn stored_proposals_appear_with_their_label_and_keep_the_choices_made_on_them() {
    let db = mem_db().await;
    library(&db, &["/p/a.jpg"]).await;
    let (mut panel, _toasts) = rig(&db);
    let store = store_over(&db);
    panel.store = StoreSource::Fixed(store.clone());
    file_proposal(&store, "/p/a.jpg", Changes { caption: Some("Fresh".into()), category: Some("coast".into()), ..Default::default() }).await;
    panel.refresh_store();
    assert!(panel.store_status.loading);
    pump_until(&mut panel, |p| !p.store_status.loading).await;
    assert_eq!(panel.store_status.pending, 1);
    assert_eq!(panel.proposals()[0].source.label(), "MCP tester");
    panel.proposals[0].changes.iter_mut().find(|c| c.kind == FieldKind::Category).unwrap().accepted = false;
    panel.refresh_store();
    pump_until(&mut panel, |p| !p.store_status.loading).await;
    assert_eq!(panel.proposals().len(), 1);
    assert!(!panel.proposals()[0].change(FieldKind::Category).unwrap().accepted, "the unchecked field stays unchecked");
    panel.clear_local();
    assert_eq!(panel.proposals().len(), 1, "clearing the generated ones leaves the stored ones");
    let key = panel.proposals()[0].key.clone();
    panel.accept_one(&key);
    pump_until(&mut panel, |p| p.proposals().is_empty()).await;
    let row = meta(&db, "/p/a.jpg").await;
    assert_eq!((row.caption.as_deref(), row.category.as_deref()), (Some("Fresh"), Some("old")));
}

#[tokio::test]
async fn a_store_that_cannot_be_read_shows_its_error() {
    let db = mem_db().await;
    let (mut panel, _toasts) = rig(&db);
    panel.sender().send(Msg::Stored(Err("the database is gone".into()))).unwrap();
    panel.pump();
    assert_eq!(panel.store_status.error.as_deref(), Some("the database is gone"));
    let (mut off, _t) = rig(&db);
    off.store = StoreSource::Off;
    off.refresh_store();
    assert!(!off.store_status.loading, "no store, nothing to read");
}

#[test]
fn selection_requests_become_the_scope_of_the_next_run() {
    let (toast_tx, _rx) = crossbeam::channel::unbounded();
    let mut panel = RefinementsPanel::with_sources(toast_tx, DbSource::Global, StoreSource::Off, crate::ui::file_table::thumbs::service());
    panel.absorb_requests(vec![strings(&["/a.jpg", "/b.jpg"]), strings(&["/b.jpg", "/c.jpg"])]);
    assert_eq!(panel.scope_len(), 3);
    panel.take_requests();
    assert_eq!(panel.scope_len(), 3, "a panel outside the application leaves the shared mailbox alone");
    panel.clear_scope();
    assert_eq!(panel.scope_len(), 0);
}

#[tokio::test]
async fn a_run_over_a_selection_that_is_not_in_the_library_finishes_and_is_tracked() {
    let db = mem_db().await;
    let (mut panel, _toasts) = rig(&db);
    panel.absorb_requests(vec![strings(&["/nowhere/at-all-refine-test.jpg"])]);
    panel.start_run();
    assert!(panel.is_running());
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Done { .. } | RunState::Failed(_))).await;
    assert!(matches!(panel.run_state(), RunState::Done { produced: 0, .. }), "{:?}", panel.run_state());
    let tracked = crate::tasks::with(|r| r.tasks().iter().any(|t| t.kind == crate::tasks::TaskKind::Refine && t.title == "Generating refinements for 1 files"));
    assert!(tracked, "the run is in the task list");
    panel.start_run();
    panel.cancel_run();
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Done { .. } | RunState::Failed(_))).await;
}

fn frame_text(ctx: &Context, panel: &mut RefinementsPanel, width: f32) -> Vec<(String, Pos2)> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 900.0))), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| panel.ui(ui));
        });
        out.textures_delta.clear();
        last = texts(&out.shapes);
    }
    last
}

fn filled_panel(db: &Surreal<Db>) -> RefinementsPanel {
    let (mut panel, _toasts) = rig(db);
    panel.proposals = vec![
        Proposal::from_generated(ProposalKey::Local(1), &generated("/photos/trip/beach.jpg", Some("beach"), &["keep", "sea"], Some("A sunny beach"), Some(0.72), "Heuristic"), 1).unwrap(),
        stored("p1", "/photos/trip/forest.jpg", Changes { caption: Some("Pines at dawn".into()), tags_add: strings(&["pine"]), tags_remove: strings(&["drop"]), ..Default::default() }, &Current { category: Some("nature".into()), tags: strings(&["drop", "tree"]), caption: None, description: None }),
    ];
    panel.store_status.pending = 1;
    panel.store_status.at = Some(Instant::now());
    panel
}

#[tokio::test]
async fn the_tab_draws_proposals_with_fields_sources_and_confidence_at_three_widths() {
    let db = mem_db().await;
    let ctx = test_ctx();
    for width in [700.0, 1000.0, 1600.0] {
        let mut panel = filled_panel(&db);
        let drawn = frame_text(&ctx, &mut panel, width);
        let text: Vec<&str> = drawn.iter().map(|(t, _)| t.as_str()).collect();
        for expected in ["AI Refinements", "Generate proposals", "Showing 2 of 2 proposals", "beach.jpg", "forest.jpg", "Category", "Tags", "Caption", "Accept 3 fields", "Reject", "Current", "Proposed", "Check all shown", "Reject shown"] {
            assert!(text.iter().any(|t| t.contains(expected)), "{width}: missing {expected:?} in {text:?}");
        }
        assert!(text.iter().any(|t| t.contains("Heuristic")) && text.iter().any(|t| t.contains("MCP tester")), "{width}: both sources are named");
        assert!(text.iter().any(|t| t.contains("Confidence 72%")), "{width}");
        assert!(text.iter().any(|t| t.contains("+sea")) && text.iter().any(|t| t.contains("-drop")), "{width}: tag changes are marked");
        assert!(text.iter().any(|t| t.contains("Accept 2 fields in 2 files") || t.contains("Accept 5 fields in 2 files")), "{width}: the bulk button counts the checked fields {text:?}");
        assert!(drawn.iter().all(|(_, pos)| pos.x < width), "{width}: nothing starts beyond the right edge");
    }
}

#[tokio::test]
async fn the_tab_shows_a_confirmation_with_exact_counts_before_a_bulk_accept() {
    let db = mem_db().await;
    let ctx = test_ctx();
    let mut panel = filled_panel(&db);
    panel.ask_accept_view();
    let drawn = frame_text(&ctx, &mut panel, 1000.0);
    let text: Vec<&str> = drawn.iter().map(|(t, _)| t.as_str()).collect();
    assert!(text.iter().any(|t| t.starts_with("Apply 5 fields in 2 files")), "{text:?}");
    assert!(text.iter().any(|t| *t == "Confirm") && text.iter().any(|t| *t == "Cancel"));
}

#[tokio::test]
async fn the_tab_explains_an_empty_list_and_a_filter_that_hides_everything() {
    let db = mem_db().await;
    let ctx = test_ctx();
    let (mut panel, _t) = rig(&db);
    let drawn = frame_text(&ctx, &mut panel, 1000.0);
    assert!(drawn.iter().any(|(t, _)| t.contains("No proposals yet")));
    let mut filled = filled_panel(&db);
    filled.filter.text = "zzz-no-match".into();
    let drawn = frame_text(&ctx, &mut filled, 1000.0);
    assert!(drawn.iter().any(|(t, _)| t.contains("No proposal matches the filters")));
    assert!(drawn.iter().any(|(t, _)| t.contains("Showing 0 of 2 proposals")));
}

#[test]
fn the_request_helper_sends_the_refine_command() {
    let _serial = crate::command::bus::testing::serial();
    crate::command::bus::testing::take_matching(|c| matches!(c, crate::command::AppCommand::Refine { .. }));
    request_refine_for_paths(Vec::new());
    request_refine_for_paths(strings(&["/a.jpg"]));
    let sent = crate::command::bus::testing::take_matching(|c| matches!(c, crate::command::AppCommand::Refine { .. }));
    assert_eq!(sent.len(), 1, "an empty request sends nothing");
}

#[test]
fn tag_changes_use_the_batch_tag_semantics_of_the_library() {
    let current = strings(&["keep", "drop"]);
    assert_eq!(metadata_edit::apply_tag_change(&current, &TagChange { add: strings(&["sea"]), remove: strings(&["DROP"]) }), strings(&["keep", "sea"]));
}

#[test]
fn sample_proposals_are_made_up_with_falling_confidence_and_rotating_sources() {
    let rows: Vec<crate::Thumbnail> = (0..5)
        .map(|i| {
            let mut t = crate::Thumbnail::new(&format!("f{i}.jpg"));
            t.path = format!("/s/f{i}.jpg");
            t.tags = strings(&["kept"]);
            t
        })
        .collect();
    let made = sample_proposals(&rows);
    assert_eq!(made.len(), 5);
    assert_eq!(made.iter().map(|p| p.generator.as_str()).collect::<Vec<_>>(), vec!["Heuristic", "Jina M0", "cloud:sample", "Heuristic", "Jina M0"]);
    let confidences: Vec<f32> = made.iter().map(|p| p.confidence.unwrap()).collect();
    assert!(confidences.windows(2).all(|w| w[0] >= w[1]) && confidences.iter().all(|c| *c >= 0.1 && *c <= 0.9));
    assert_eq!(made[2].new_tags, strings(&["kept", "sample-tag-3"]));
    assert_eq!(made[0].new_caption.as_deref(), Some("Sample caption for f0.jpg"));
    let as_proposals: Vec<Proposal> = made.iter().enumerate().filter_map(|(i, g)| Proposal::from_generated(ProposalKey::Local(i as u64), g, 0)).collect();
    assert_eq!(as_proposals.len(), 5, "every sample changes something");
}
