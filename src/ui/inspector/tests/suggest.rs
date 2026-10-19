//! The Suggest section of the AI tab against the loopback mock gateway: asking, the accept chips, one journal batch per click with Undo, refusals and layout at three widths.

use std::time::Duration;

use serde_json::json;
use zeroclaw_client::mock::MockGateway;

use super::super::prefs::DetailTab;
use super::super::state::{RowEdit, apply_edit};
use super::super::suggest::{self, Env, LABEL, Pick, Store, env_lock};
use super::*;
use crate::ai::mcp::ctx::DbSource;
use crate::command::bus::testing::serial;
use crate::database::journal;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed};
use crate::ui::journal_ui::take_edits;
use crate::zc::health::Health;
use crate::zc::jobs::{JobsConfig, ZcJobs};
use crate::zc::profile::ZcProfile;
use crate::zc::runtime_tests::{runtime_for, until};

fn strings(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

async fn online(name: &str, db: &Surreal<Db>) -> (MockGateway, Env) {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", name);
    until("online", || rt.health().is_online().then_some(())).await;
    let env = Env { jobs: ZcJobs::new(rt, JobsConfig::quick()), db: DbSource::Handle(db.clone()), store: std::sync::Arc::new(Store::default()) };
    (mock, env)
}

async fn seed(rig: &mut Rig, name: &str) {
    let patch = MetaPatch { caption: Some(Some("A red picture".into())), category: Some(Some("misc".into())), tags: Some(strings(&["keep"])), ..Default::default() };
    metadata_edit::save_patch(&rig.db, &RowSeed::from_row(&rig.row(name)), &patch).await.unwrap();
    journal::clear_with(&rig.db).await.unwrap();
    let path = rig.path(name);
    apply_edit(&mut rig.ex, &RowEdit::Patch { path, patch });
}

fn apply_pending(rig: &mut Rig) -> Vec<RowEdit> {
    let edits = take_edits();
    for edit in &edits {
        apply_edit(&mut rig.ex, edit);
    }
    edits
}

#[tokio::test(flavor = "multi_thread")]
async fn accepting_writes_one_journal_batch_that_undo_reverts() {
    let mut rig = Rig::new(1000.0).await;
    seed(&mut rig, "a.png").await;
    let mut stale = rig.row("a.png");
    stale.tags = strings(&["other"]);
    let env_db = DbSource::Handle(rig.db.clone());
    let edit = suggest::accept(&env_db, &stale, &Pick { tags: strings(&["sea", "keep"]), category: Some("beach".into()), caption: Some("Waves.".into()) }).await.unwrap();
    let RowEdit::Patch { path, patch } = edit else { panic!("a patch") };
    assert_eq!(path, rig.path("a.png"));
    assert_eq!(patch.tags, Some(strings(&["keep", "sea"])), "the stored tags are merged, not the stale copy of the row");
    let stored = rig.db_row("a.png").await.unwrap();
    assert_eq!((stored.category.as_deref(), stored.caption.as_deref(), stored.tags), (Some("beach"), Some("Waves."), strings(&["keep", "sea"])));
    let batches = journal::batches_with(&rig.db, 10).await.unwrap();
    assert_eq!(batches.len(), 1, "one click is one batch");
    assert_eq!((batches[0].source.as_str(), batches[0].entries), (LABEL, 3));
    let report = journal::undo_last_with(&rig.db).await.unwrap();
    assert_eq!((report.applied.len(), report.conflicts.len()), (3, 0));
    let back = rig.db_row("a.png").await.unwrap();
    assert_eq!((back.category.as_deref(), back.caption.as_deref(), back.tags), (Some("misc"), Some("A red picture"), strings(&["keep"])));
    assert!(suggest::accept(&env_db, &stale, &Pick::default()).await.is_err(), "an empty pick writes nothing");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_that_is_not_in_the_library_gets_its_row_on_the_first_accept() {
    let rig = Rig::new(1000.0).await;
    let row = rig.row("b.png");
    assert!(rig.db_row("b.png").await.is_none());
    suggest::accept(&DbSource::Handle(rig.db.clone()), &row, &Pick { tags: strings(&["red"]), ..Default::default() }).await.unwrap();
    assert_eq!(rig.db_row("b.png").await.unwrap().tags, strings(&["red"]));
}

#[tokio::test(flavor = "multi_thread")]
async fn suggest_asks_shows_chips_and_every_click_is_one_undoable_write() {
    let _guard = env_lock();
    let _serial = serial();
    take_edits();
    let mut rig = Rig::new(1000.0).await;
    rig.height = 1300.0;
    seed(&mut rig, "a.png").await;
    let (mock, env) = online("sg-flow", &rig.db).await;
    suggest::set_test_env(Some(env.clone()));
    mock.script_complete_reply("```json\n{\"tags\":[\"Red\",\"keep\",\"Square\",\"sunset glow\"],\"category\":\"Abstract\",\"caption\":\"A plain red square.\"}\n```");
    rig.open_tab(DetailTab::Ai);
    rig.select(&["a.png"]);
    rig.settle(5).await;
    assert!(rig.has_text("Suggestions") && rig.has_text("Suggest"), "{:?}", rig.texts());
    assert!(rig.has_text("Tags, a category and a caption from this file's metadata and the files most like it. Only text is sent."));
    rig.click_text("Suggest");
    rig.until("the chips", |r| r.has_text("+ red")).await;
    for chip in ["+ red", "+ square", "+ sunset glow", "+ abstract", "Use caption", "Add all", "Dismiss"] {
        assert!(rig.has_text(chip), "missing {chip}: {:?}", rig.texts());
    }
    assert!(!rig.has_text("+ keep"), "a tag the file has is not suggested");
    assert!(rig.has_text("A plain red square.") && rig.has_text("Answered by mock-model"));
    let call = mock.requests().into_iter().find(|r| r.path == "/api/complete").unwrap().body;
    let prompt = call["prompt"].as_str().unwrap();
    assert!(prompt.contains("\"file\":\"a.png\"") && prompt.contains("A red picture") && !prompt.contains(&rig.path("a.png")), "{prompt}");
    assert!(call["system"].as_str().unwrap().contains("Library tags, most used first: keep"));

    rig.click_text("+ red");
    rig.until("the note", |r| r.has_text("Added tag red. Ctrl+Z undoes it.")).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().tags, strings(&["keep", "red"]));
    let path = rig.path("a.png");
    let edits: Vec<RowEdit> = apply_pending(&mut rig).into_iter().filter(|e| matches!(e, RowEdit::Patch { path: p, .. } if *p == path)).collect();
    assert!(matches!(&edits[..], [RowEdit::Patch { patch, .. }] if patch.tags == Some(strings(&["keep", "red"]))), "{edits:?}");
    assert_eq!(rig.row("a.png").tags, strings(&["keep", "red"]), "the browser row shows it");
    rig.settle(2).await;
    assert!(!rig.has_text("+ red") && rig.has_text("+ square"), "the accepted chip is gone, the others stay");

    rig.click_text("Add all");
    rig.until("the second note", |r| r.has_text("Added tags square, sunset glow and category abstract and the caption. Ctrl+Z undoes it.")).await;
    let stored = rig.db_row("a.png").await.unwrap();
    assert_eq!((stored.category.as_deref(), stored.caption.as_deref(), stored.tags), (Some("abstract"), Some("A plain red square."), strings(&["keep", "red", "square", "sunset glow"])));
    apply_pending(&mut rig);
    let batches = journal::batches_with(&rig.db, 10).await.unwrap();
    assert_eq!(batches.iter().map(|b| (b.source.as_str(), b.entries)).collect::<Vec<_>>(), [(LABEL, 3), (LABEL, 1)], "newest first: Add all is one batch, the chip another");
    rig.settle(2).await;
    assert!(!rig.has_text("Dismiss") && !rig.has_text("Add all"), "nothing is left to suggest");
    journal::undo_last_with(&rig.db).await.unwrap();
    let undone = rig.db_row("a.png").await.unwrap();
    assert_eq!((undone.category.as_deref(), undone.caption.as_deref(), undone.tags), (Some("misc"), Some("A red picture"), strings(&["keep", "red"])));
    suggest::set_test_env(None);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_that_is_not_json_shows_the_error_and_writes_nothing() {
    let _guard = env_lock();
    let mut rig = Rig::new(1000.0).await;
    rig.height = 1100.0;
    seed(&mut rig, "a.png").await;
    let (mock, env) = online("sg-bad", &rig.db).await;
    suggest::set_test_env(Some(env));
    mock.script_complete_reply("Here are some ideas: red, square.");
    rig.open_tab(DetailTab::Ai);
    rig.select(&["a.png"]);
    rig.settle(5).await;
    rig.click_text("Suggest");
    rig.until("the error", |r| r.has_text("Nothing was changed.")).await;
    assert!(rig.texts().iter().any(|t| t.contains("no JSON")), "{:?}", rig.texts());
    assert_eq!(rig.db_row("a.png").await.unwrap().tags, strings(&["keep"]));
    assert!(journal::batches_with(&rig.db, 10).await.unwrap().is_empty());
    suggest::set_test_env(None);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_stops_the_question() {
    let _guard = env_lock();
    let mut rig = Rig::new(1000.0).await;
    rig.height = 1100.0;
    let (mock, env) = online("sg-cancel", &rig.db).await;
    suggest::set_test_env(Some(env.clone()));
    mock.script_complete(200, json!({"response": "{}", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    rig.open_tab(DetailTab::Ai);
    rig.select(&["a.png"]);
    rig.settle(5).await;
    rig.click_text("Suggest");
    rig.until("the wait", |r| r.has_text("Asking ZeroClaw...")).await;
    until("the request", || (mock.requests().iter().any(|r| r.path == "/api/complete")).then_some(())).await;
    rig.click_text("Cancel");
    rig.until("the button back", |r| !r.has_text("Asking ZeroClaw...")).await;
    until("the job to end", || env.jobs.is_idle().then_some(())).await;
    assert_eq!(env.jobs.snapshot().totals.cancelled, 1);
    assert!(rig.has_text("Suggest"));
    suggest::set_test_env(None);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_button_is_disabled_with_the_reason_while_the_gateway_is_not_online() {
    let _guard = env_lock();
    let mut rig = Rig::new(1000.0).await;
    rig.height = 1100.0;
    let (mock, env) = online("sg-off", &rig.db).await;
    drop(mock);
    let rt = env.jobs.runtime().clone();
    rt.set_profile(ZcProfile::default());
    until("not set up", || matches!(rt.health(), Health::Unconfigured(_)).then_some(())).await;
    suggest::set_test_env(Some(env.clone()));
    rig.open_tab(DetailTab::Ai);
    rig.select(&["a.png"]);
    rig.settle(5).await;
    assert!(rig.has_text("Suggest"));
    assert!(rig.texts().iter().any(|t| t.starts_with("ZeroClaw is unavailable: ZeroClaw is not set up")), "{:?}", rig.texts());
    rig.click_text("Suggest");
    rig.settle(3).await;
    assert!(!rig.has_text("Asking ZeroClaw..."), "a disabled button asks nothing");
    assert!(!env.ask(&rig.row("a.png")), "and neither does the API");
    assert!(env.jobs.is_idle());
    suggest::set_test_env(None);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_section_draws_inside_the_rail_at_700_1000_and_1600_px() {
    let _guard = env_lock();
    for width in [700.0, 1000.0, 1600.0] {
        let mut rig = Rig::new(width).await;
        rig.height = 1300.0;
        seed(&mut rig, "a.png").await;
        let (mock, env) = online("sg-width", &rig.db).await;
        suggest::set_test_env(Some(env));
        mock.script_complete_reply(r#"{"tags":["red","square","a-long-tag-name-that-needs-room","another-quite-long-tag"],"category":"abstract","caption":"A plain red square with a caption long enough to need wrapping in a narrow rail."}"#);
        rig.open_tab(DetailTab::Ai);
        rig.select(&["a.png"]);
        rig.settle(5).await;
        rig.click_text("Suggest");
        rig.until("the chips", |r| r.has_text("+ red")).await;
        rig.settle(3).await;
        let rail = rig.out.rect.expect("the rail drew");
        for (text, pos) in crate::ui::test_support::texts(&rig.shapes) {
            if text.starts_with("+ ") || text == "Add all" || text == "Use caption" {
                assert!(pos.x >= rail.left() - 1.0 && pos.x < rail.right(), "{width}: {text:?} at {pos:?} is outside the rail {rail:?}");
            }
        }
        for expected in ["+ red", "+ square", "+ abstract", "Add all", "Dismiss"] {
            assert!(rig.has_text(expected), "{width}: {expected} is drawn");
        }
        suggest::set_test_env(None);
    }
}
