//! The ZeroClaw source of the Refine tab: runs against the loopback mock gateway over a seeded in-memory library, accepts through the journal, the source switch and its reasons at three widths.

use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{CentralPanel, Context, Pos2, RawInput, Rect, vec2};
use serde_json::json;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::SurrealValue;
use zeroclaw_client::mock::MockGateway;

use super::accept;
use super::model::*;
use super::*;
use crate::ai::mcp::clock::FakeClock;
use crate::ai::mcp::proposals::ProposalStore;
use crate::database::journal;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed};
use crate::tasks::TaskKind;
use crate::ui::test_support::{test_ctx, texts};
use crate::zc::jobs::{JobsConfig, ZcJobs};
use crate::zc::runtime_tests::{runtime_for, until};

fn strings(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
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
        metadata_edit::save_patch(db, &RowSeed::from_path(p), &patch).await.unwrap();
    }
    journal::clear_with(db).await.unwrap();
}

async fn rig(db: &Surreal<Db>, name: &str) -> (MockGateway, RefinementsPanel) {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", name);
    until("online", || rt.health().is_online().then_some(())).await;
    let (toast_tx, _toasts) = crossbeam::channel::unbounded();
    let store = Arc::new(ProposalStore::new(DbSource::Handle(db.clone()), FakeClock::new()));
    let mut panel = RefinementsPanel::with_sources(toast_tx, DbSource::Handle(db.clone()), StoreSource::Fixed(store), crate::ui::file_table::thumbs::service());
    panel.set_zc_jobs(ZcJobs::new(rt, JobsConfig::quick()));
    (mock, panel)
}

async fn pump_until(panel: &mut RefinementsPanel, done: impl Fn(&RefinementsPanel) -> bool) {
    for _ in 0..400 {
        panel.pump();
        if done(panel) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the panel did not reach the expected state: {:?}", panel.run_state());
}

fn complete_calls(mock: &MockGateway) -> Vec<serde_json::Value> {
    mock.requests().into_iter().filter(|r| r.path == "/api/complete").map(|r| r.body).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_zeroclaw_run_makes_proposals_that_accept_through_the_journal_and_undo() {
    let _serial = crate::command::bus::testing::serial();
    let db = mem_db().await;
    library(&db, &["/zc/a.jpg", "/zc/b.jpg"]).await;
    let (mock, mut panel) = rig(&db, "zcref-ok").await;
    mock.script_complete_reply(r#"Here you go:
```json
[{"id":1,"new_category":"Beach","new_tags":["keep","Sea"]},{"id":2,"new_caption":"A fresh caption","new_description":"Two sentences. About the file."}]
```"#);
    panel.use_zeroclaw = true;
    panel.absorb_requests(vec![strings(&["/zc/a.jpg", "/zc/b.jpg"])]);
    panel.start_run();
    assert!(panel.is_running());
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Done { .. } | RunState::Failed(_))).await;
    assert_eq!(panel.run_state(), &RunState::Done { scanned: 2, produced: 2, cancelled: false });
    assert_eq!(panel.proposals().len(), 2);
    let a = panel.proposals().iter().find(|p| p.path == "/zc/a.jpg").unwrap();
    assert_eq!(a.source, ProposalSource::ZeroClaw("quick".into()));
    assert_eq!(a.source.label(), "ZeroClaw quick");
    assert_eq!(a.change(FieldKind::Category).unwrap().proposed, FieldValue::Text(Some("beach".into())));
    assert_eq!(a.change(FieldKind::Tags).unwrap().proposed, FieldValue::Tags(strings(&["keep", "sea"])), "tags are lowercase and complete");
    let b = panel.proposals().iter().find(|p| p.path == "/zc/b.jpg").unwrap();
    assert_eq!(b.changes.iter().map(|c| c.kind).collect::<Vec<_>>(), vec![FieldKind::Caption, FieldKind::Description]);
    assert_eq!(b.confidence, None, "no neighbours were found for these files");

    let calls = complete_calls(&mock);
    assert_eq!(calls.len(), 1, "two rows are one request");
    let prompt = calls[0]["prompt"].as_str().unwrap();
    assert!(prompt.contains("\"file\":\"a.jpg\"") && prompt.contains("\"folder\":\"zc\"") && prompt.contains("old caption"), "{prompt}");
    assert!(!prompt.contains("/zc/a.jpg"), "paths stay on this machine");
    let system = calls[0]["system"].as_str().unwrap();
    assert!(system.contains("tags: ") && system.contains("categories: old"), "the library vocabulary guides the model: {system}");
    assert!(crate::tasks::with(|r| r.tasks().iter().any(|t| t.kind == TaskKind::Refine && t.title == "Refine 2 files with ZeroClaw (1 request)" && t.state == crate::tasks::TaskState::Completed)), "the gateway job is the tracked Refine task");

    panel.ask_accept_view();
    panel.confirm_pending();
    pump_until(&mut panel, |p| p.proposals().is_empty()).await;
    let a_row = meta(&db, "/zc/a.jpg").await;
    assert_eq!((a_row.category.as_deref(), a_row.tags), (Some("beach"), Some(strings(&["keep", "sea"]))));
    assert_eq!(meta(&db, "/zc/b.jpg").await.caption.as_deref(), Some("A fresh caption"));
    let batches = journal::batches_with(&db, 10).await.unwrap();
    assert_eq!(batches.len(), 1, "one batch for the whole accept");
    assert_eq!(batches[0].source, accept::LOCAL_LABEL);
    journal::undo_last_with(&db).await.unwrap();
    let restored = meta(&db, "/zc/a.jpg").await;
    assert_eq!((restored.category.as_deref(), restored.tags), (Some("old"), Some(strings(&["keep", "drop"]))));
    assert_eq!(meta(&db, "/zc/b.jpg").await.caption.as_deref(), Some("old caption"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_queued_by_the_zc_command_starts_the_zeroclaw_source_over_its_files() {
    let db = mem_db().await;
    library(&db, &["/zc/e.jpg"]).await;
    let (mock, mut panel) = rig(&db, "zcref-cmd").await;
    mock.script_complete_reply(r#"[{"id":1,"new_category":"harbour"}]"#);
    panel.use_cloud_model = true;
    panel.absorb_zc_runs(vec![ZcRunRequest { paths: strings(&["/zc/e.jpg"]), hint: "quick".into() }]);
    assert!(panel.use_zeroclaw && !panel.use_cloud_model && panel.zc_hint == "quick" && panel.is_running());
    assert_eq!(panel.scope_len(), 1, "the files of the request are the scope");
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Done { .. } | RunState::Failed(_))).await;
    assert_eq!(panel.run_state(), &RunState::Done { scanned: 1, produced: 1, cancelled: false });
    assert_eq!(panel.proposals()[0].source, ProposalSource::ZeroClaw("quick".into()));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hint_typed_in_the_tab_reaches_the_gateway_and_a_bad_reply_fails_the_run() {
    let db = mem_db().await;
    library(&db, &["/zc/c.jpg"]).await;
    let (mock, mut panel) = rig(&db, "zcref-hint").await;
    mock.script_complete_reply("sorry, no JSON today");
    panel.use_zeroclaw = true;
    panel.zc_hint = " quick ".into();
    panel.absorb_requests(vec![strings(&["/zc/c.jpg"])]);
    panel.start_run();
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Failed(_))).await;
    let RunState::Failed(why) = panel.run_state() else { panic!() };
    assert!(why.contains("no JSON"), "{why}");
    assert_eq!(complete_calls(&mock)[0]["hint"], "quick");
    assert!(panel.proposals().is_empty());
    assert_eq!(meta(&db, "/zc/c.jpg").await.category.as_deref(), Some("old"), "nothing was written");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_stops_a_zeroclaw_run_and_keeps_the_task_cancelled() {
    let db = mem_db().await;
    library(&db, &["/zc/d.jpg"]).await;
    let (mock, mut panel) = rig(&db, "zcref-cancel").await;
    mock.script_complete(200, json!({"response": "[]", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    panel.use_zeroclaw = true;
    panel.absorb_requests(vec![strings(&["/zc/d.jpg"])]);
    panel.start_run();
    until("the request", || (!complete_calls(&mock).is_empty()).then_some(())).await;
    panel.cancel_run();
    pump_until(&mut panel, |p| matches!(p.run_state(), RunState::Done { .. } | RunState::Failed(_))).await;
    assert!(matches!(panel.run_state(), RunState::Done { cancelled: true, .. }), "{:?}", panel.run_state());
    assert!(crate::tasks::with(|r| r.tasks().iter().any(|t| t.kind == TaskKind::Refine && t.title == "Refine 1 file with ZeroClaw (1 request)" && t.state == crate::tasks::TaskState::Cancelled)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_needs_the_gateway_online_and_says_why_otherwise() {
    let db = mem_db().await;
    let (mock, mut panel) = rig(&db, "zcref-offline").await;
    drop(mock);
    let rt = panel.zc_jobs().runtime().clone();
    rt.set_profile(crate::zc::profile::ZcProfile::default());
    until("the gateway to read as not set up", || matches!(rt.health(), crate::zc::health::Health::Unconfigured(_)).then_some(())).await;
    panel.use_zeroclaw = true;
    panel.start_run();
    assert!(!panel.is_running());
    let RunState::Failed(why) = panel.run_state().clone() else { panic!("{:?}", panel.run_state()) };
    assert!(why.contains("not set up") || why.contains("Checking"), "{why}");
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

#[tokio::test(flavor = "multi_thread")]
async fn the_source_switch_is_drawn_enabled_with_a_hint_while_online_at_three_widths() {
    let db = mem_db().await;
    let ctx = test_ctx();
    for width in [700.0, 1000.0, 1600.0] {
        let (_mock, mut panel) = rig(&db, "zcref-frame").await;
        let drawn = frame_text(&ctx, &mut panel, width);
        assert!(drawn.iter().any(|(t, _)| t.contains("Use ZeroClaw")), "{width}: {drawn:?}");
        assert!(!drawn.iter().any(|(t, _)| t.contains("ZeroClaw is unavailable")), "{width}");
        panel.use_zeroclaw = true;
        let drawn = frame_text(&ctx, &mut panel, width);
        assert!(drawn.iter().any(|(t, _)| t.contains("Text only: file names")), "{width}: {drawn:?}");
        assert!(drawn.iter().any(|(t, _)| t.contains("Generate proposals")));
        assert!(drawn.iter().all(|(_, pos)| pos.x < width), "{width}: nothing starts beyond the right edge");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_source_switch_is_disabled_with_the_reason_while_the_gateway_is_not_online() {
    let db = mem_db().await;
    let ctx = test_ctx();
    for width in [700.0, 1000.0, 1600.0] {
        let (mock, mut panel) = rig(&db, "zcref-frame-off").await;
        drop(mock);
        let rt = panel.zc_jobs().runtime().clone();
        rt.set_profile(crate::zc::profile::ZcProfile::default());
        until("the gateway to read as not set up", || matches!(rt.health(), crate::zc::health::Health::Unconfigured(_)).then_some(())).await;
        let drawn = frame_text(&ctx, &mut panel, width);
        assert!(drawn.iter().any(|(t, _)| t.contains("Use ZeroClaw")), "{width}");
        assert!(drawn.iter().any(|(t, _)| t.contains("ZeroClaw is unavailable: ")), "{width}: {drawn:?}");
        assert!(drawn.iter().all(|(_, pos)| pos.x < width), "{width}");
    }
}

#[test]
fn a_zeroclaw_proposal_is_named_from_its_generator_and_a_run_has_a_title() {
    assert_eq!(ProposalSource::from_generator("zeroclaw:quick"), ProposalSource::ZeroClaw("quick".into()));
    assert_eq!(ProposalSource::from_generator("zeroclaw:slow").label(), "ZeroClaw slow");
    assert!(!ProposalSource::ZeroClaw("quick".into()).is_mcp());
    let spec = RunSpec { reranker: Default::default(), cloud: None, zeroclaw: None, scope: None };
    assert_eq!(spec.title(), "Generating refinements");
}
