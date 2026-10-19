//! The Ask mode of the query bar against the loopback mock gateway: the question, the preview with its chips, Apply, refusals, cancel and layout at three widths.

use std::time::Duration;

use eframe::egui::*;
use serde_json::json;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use zeroclaw_client::mock::MockGateway;

use super::super::{Plan, plan};
use super::*;
use crate::command::{ErrorKind, QueryMode};
use crate::ui::file_table::test_rig::{ROWS, Rig, text_rects};
use crate::zc::health::Health;
use crate::zc::jobs::JobsConfig;
use crate::zc::profile::ZcProfile;
use crate::zc::runtime_tests::{runtime_for, until};

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    db
}

async fn online(name: &str) -> (MockGateway, Arc<ZcJobs>) {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", name);
    until("online", || rt.health().is_online().then_some(())).await;
    (mock, ZcJobs::new(rt, JobsConfig::quick()))
}

async fn rig_with(width: f32, jobs: Arc<ZcJobs>) -> Rig {
    let mut rig = Rig::new(width, 600.0);
    rig.ex.toolbar.query.ask.use_backend(jobs, DbSource::Handle(mem_db().await));
    rig
}

async fn until_phase(rig: &mut Rig, what: &str, done: impl Fn(&AskState) -> bool) {
    for _ in 0..400 {
        rig.settle(1);
        if done(&rig.ex.toolbar.query.ask) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the Ask mode did not reach {what}: {:?}", rig.ex.toolbar.query.ask.phase);
}

fn ready(ask: &AskState) -> bool {
    matches!(ask.phase, Phase::Ready(_))
}

fn failed(ask: &AskState) -> bool {
    matches!(ask.phase, Phase::Failed { .. })
}

#[test]
fn enter_asks_and_typing_never_runs_anything() {
    assert_eq!(plan(QueryMode::Ask, "all sunsets", None, true), Plan::Ask("all sunsets".into()));
    assert_eq!(plan(QueryMode::Ask, "  all sunsets ", None, true), Plan::Ask("all sunsets".into()));
    assert!(matches!(plan(QueryMode::Ask, "all sunsets", None, false), Plan::Idle(_)), "typing waits for Enter");
    assert!(matches!(plan(QueryMode::Ask, "   ", None, true), Plan::Idle(_)));
    assert!(!QueryMode::Ask.is_live() && QueryMode::ALL.len() == 5 && QueryMode::ALL[4] == QueryMode::Ask);
    assert_eq!(super::super::describe(&Plan::Ask("x".into())), "ask ZeroClaw for a filter: \"x\"");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sentence_becomes_a_preview_with_chips_and_enter_again_applies_it() {
    let (mock, jobs) = online("ask-ok").await;
    mock.script_complete_reply("```json\n{\"tags\":[\"sunset\"],\"description\":\"no\",\"types\":{\"videos\":false}}\n```");
    let mut rig = rig_with(1000.0, jobs).await;
    let before = rig.ex.filter_spec();
    let started = rig.ex.apply_query_command(QueryMode::Ask, "sunset photos without a description", true).unwrap();
    assert!(started.contains("asking ZeroClaw"), "{started}");
    assert!(rig.ex.toolbar.query.ask.is_working());
    until_phase(&mut rig, "a preview", ready).await;
    assert_eq!(rig.ex.filter_spec(), before, "nothing changes before Apply");
    let shown: Vec<String> = crate::ui::test_support::texts(&rig.shapes).into_iter().map(|(t, _)| t.trim().to_string()).collect();
    for expected in ["Filters for \"sunset photos without a description\"", "Tag: sunset", "No description", "Hiding: Videos", "Apply", "Dismiss"] {
        assert!(shown.iter().any(|t| t == expected), "missing {expected:?} in {shown:?}");
    }
    let call = mock.requests().into_iter().find(|r| r.path == "/api/complete").unwrap().body;
    assert_eq!(call["hint"], "quick");
    assert_eq!(call["prompt"], "Sentence: sunset photos without a description\nReply:");
    let system = call["system"].as_str().unwrap();
    assert!(system.contains("\"tags_all\"") && system.contains("Examples:"), "the prompt holds the exact schema and examples");
    assert_eq!(call["temperature"], 0.0);

    let done = rig.ex.run_query().unwrap();
    assert!(done.contains("applied the filters"), "{done}");
    rig.settle(3);
    let spec = rig.ex.filter_spec();
    assert_eq!(spec.tags, ["sunset"]);
    assert_eq!((spec.description, spec.types.videos), (crate::ui::file_table::rows::Tri::No, false));
    assert!(rig.ex.toolbar.query.ask.preview().is_none() && !rig.has_text("Apply"), "the preview is gone");
    assert_eq!(rig.ex.toolbar.query.mode, QueryMode::Ask, "the bar stays in the Ask mode");
    assert_eq!(rig.ex.toolbar.query.text, "sunset photos without a description");
    assert_eq!(rig.ex.table.len(), ROWS + 1, "rows are only hidden, never removed");
    assert!(rig.ex.filter_spec().chips().iter().any(|c| c.label == "Tag: sunset"));
    assert_eq!((rig.ex.toolbar.query.ask.asked, rig.ex.toolbar.query.ask.applied), (1, 1));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_apply_button_applies_and_dismiss_drops_the_preview() {
    let (mock, jobs) = online("ask-button").await;
    mock.script_complete_reply(r#"{"ext_allow":["jpg"]}"#);
    mock.script_complete_reply(r#"{"clip":"yes"}"#);
    let mut rig = rig_with(1000.0, jobs).await;
    rig.ex.apply_query_command(QueryMode::Ask, "only jpg files", true).unwrap();
    until_phase(&mut rig, "a preview", ready).await;
    rig.click(rig.text_pos("Apply").unwrap() + vec2(8.0, 6.0));
    assert_eq!(rig.ex.filter_spec().ext_allow, ["jpg"]);
    assert!(rig.ex.toolbar.query.ask.preview().is_none());

    rig.ex.apply_query_command(QueryMode::Ask, "files with an embedding", true).unwrap();
    until_phase(&mut rig, "a second preview", ready).await;
    assert!(rig.has_text("Replaces the filters of this tab. Rows are only hidden, never removed. Answered by mock-model."), "{:?}", crate::ui::test_support::texts(&rig.shapes));
    rig.click(rig.text_pos("Dismiss").unwrap() + vec2(8.0, 6.0));
    assert!(matches!(rig.ex.toolbar.query.ask.phase, Phase::Idle));
    assert_eq!(rig.ex.filter_spec().ext_allow, ["jpg"], "a dismissed preview changes nothing");
    assert_eq!(rig.ex.filter_spec().clip, crate::ui::file_table::rows::Tri::Any);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_the_parser_refuses_shows_the_error_and_changes_nothing() {
    struct Case {
        name: &'static str,
        reply: &'static str,
        shows: &'static str,
    }
    let cases = [
        Case { name: "unknown field", reply: r#"{"colour":"red"}"#, shows: "unknown field" },
        Case { name: "bad date", reply: r#"{"modified":{"after":"last summer"}}"#, shows: "not a date" },
        Case { name: "no conditions", reply: "{}", shows: "no conditions" },
        Case { name: "prose", reply: "I would filter by sunsets.", shows: "no JSON" },
        Case { name: "wrong type", reply: r#"{"tags":"sunset"}"#, shows: "bad filter" },
    ];
    let (mock, jobs) = online("ask-bad").await;
    let mut rig = rig_with(1000.0, jobs).await;
    rig.ex.edit_filters(|s| s.name = "keepme".into());
    rig.settle(2);
    let before = rig.ex.filter_spec();
    for case in cases {
        mock.script_complete_reply(case.reply);
        rig.ex.apply_query_command(QueryMode::Ask, "something", true).unwrap();
        until_phase(&mut rig, "a failure", failed).await;
        let Phase::Failed { error, .. } = rig.ex.toolbar.query.ask.phase.clone() else { unreachable!() };
        assert!(error.contains(case.shows), "{}: {error}", case.name);
        assert!(rig.has_text("Nothing was changed."), "{}", case.name);
        assert_eq!(rig.ex.filter_spec(), before, "{}", case.name);
        rig.ex.toolbar.query.ask.dismiss();
    }
    assert_eq!(rig.ex.table.len(), ROWS + 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn ask_is_refused_with_the_reason_while_the_gateway_is_not_online() {
    let (mock, jobs) = online("ask-off").await;
    let mut rig = rig_with(1000.0, jobs.clone()).await;
    drop(mock);
    let rt = jobs.runtime().clone();
    rt.set_profile(ZcProfile::default());
    until("not set up", || matches!(rt.health(), Health::Unconfigured(_)).then_some(())).await;
    let reason = rig.ex.toolbar.query.ask.unavailable().expect("Ask is unavailable");
    assert!(reason.contains("not set up"), "{reason}");
    let err = rig.ex.apply_query_command(QueryMode::Ask, "all sunsets", true).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
    assert!(err.message.contains("not set up"), "{}", err.message);
    assert!(matches!(rig.ex.toolbar.query.ask.phase, Phase::Idle));
    rig.settle(3);
    assert!(rig.ex.toolbar.query.history.back().unwrap().outcome.is_err());
    let shown = crate::ui::test_support::texts(&rig.shapes);
    assert!(shown.iter().any(|(t, _)| t.starts_with("Ask is unavailable: ZeroClaw is not set up")), "the reason is shown under the field: {shown:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn typing_in_the_ask_mode_leaves_the_filter_alone_and_a_filter_set_elsewhere_does_not_leave_the_mode() {
    let (_mock, jobs) = online("ask-type").await;
    let mut rig = rig_with(1000.0, jobs).await;
    rig.ex.edit_filters(|s| s.name = "img_03".into());
    rig.settle(3);
    assert_eq!(rig.ex.toolbar.query.text, "img_03");
    rig.ex.toolbar.query.mode = QueryMode::Ask;
    rig.ex.toolbar.query.text = "show me the beach".into();
    rig.ex.toolbar.query.focus = true;
    rig.settle(3);
    rig.type_text(" please");
    assert_eq!(rig.ex.filter_spec().name, "img_03", "the live filter is not cleared by typing a sentence");
    rig.ex.edit_filters(|s| s.query = "sunset".into());
    rig.settle(3);
    assert_eq!((rig.ex.toolbar.query.mode, rig.ex.toolbar.query.text.as_str()), (QueryMode::Ask, "show me the beach please"));
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_stops_the_question_and_the_job() {
    let (mock, jobs) = online("ask-cancel").await;
    mock.script_complete(200, json!({"response": "{}", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    let mut rig = rig_with(1000.0, jobs.clone()).await;
    rig.ex.apply_query_command(QueryMode::Ask, "slow one", true).unwrap();
    until("the request", || (mock.requests().iter().any(|r| r.path == "/api/complete")).then_some(())).await;
    rig.settle(3);
    assert!(rig.has_text("Asking ZeroClaw..."));
    rig.click(rig.text_pos("Cancel").unwrap() + vec2(8.0, 6.0));
    assert!(matches!(rig.ex.toolbar.query.ask.phase, Phase::Idle));
    until("the job to end", || jobs.is_idle().then_some(())).await;
    assert_eq!(jobs.snapshot().totals.cancelled, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_ask_segment_and_its_preview_fit_at_700_1000_and_1600_px() {
    for width in [700.0, 1000.0, 1600.0] {
        let (mock, jobs) = online("ask-width").await;
        mock.script_complete_reply(r#"{"tags":["sunset","beach"],"categories":["landscape"],"modified":{"after":"2024-01-01","before":"2024-12-31"},"description":"no","clip":"yes"}"#);
        let mut rig = rig_with(width, jobs).await;
        rig.ex.apply_query_command(QueryMode::Ask, "my beach and sunset pictures from last year", true).unwrap();
        until_phase(&mut rig, "a preview", ready).await;
        let layout = rig.ex.toolbar.layout.clone().expect("the row was drawn");
        for (name, rect) in layout.parts() {
            assert!(rect.left() >= layout.row.left() - 0.5 && rect.right() <= layout.row.right() + 0.5, "{width}: {name} leaves the row: {rect:?}");
        }
        let rects = text_rects(&rig.shapes);
        for (text, rect) in &rects {
            assert!(rect.right() <= width + 1.0 && rect.left() >= -1.0, "{width}: {text:?} leaves the window: {rect:?}");
        }
        for expected in ["Apply", "Dismiss"] {
            assert!(rects.iter().any(|(t, _)| t == expected), "{width}: {expected} is drawn in {rects:?} with {:?}", rig.ex.toolbar.query.ask.phase);
        }
        let chips: Vec<&str> = rects.iter().map(|(t, _)| t.trim()).filter(|t| t.starts_with("Tag: ") || t.starts_with("Category: ") || t.starts_with("Modified ")).collect();
        assert_eq!(chips.len(), 3, "{width}: {chips:?}");
    }
}
