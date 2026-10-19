use std::sync::Mutex;
use std::time::Duration;

use eframe::egui::{Pos2, RawInput, Rect, vec2};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::*;
use crate::ai::mcp::ctx::DbSource;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed};
use crate::ui::refine::StoreSource;
use crate::ui::test_support::{test_ctx, texts};

/// Serializes the tests that use the window state of a shared context.
static LOCK: Mutex<()> = Mutex::new(());

fn drawn(ctx: &Context) -> Vec<String> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 900.0))), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| show(ui.ctx()));
        out.textures_delta.clear();
        last = texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
    }
    last
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
}

fn panels(db: &Surreal<Db>) -> (RefinementsPanel, ImageEditPanel) {
    let (toast_tx, _rx) = crossbeam::channel::unbounded();
    let refine = RefinementsPanel::with_sources(toast_tx, DbSource::Handle(db.clone()), StoreSource::Off, crate::ui::file_table::thumbs::service());
    (refine, ImageEditPanel::default())
}

#[test]
fn pasted_paths_are_trimmed_unquoted_and_deduplicated() {
    assert_eq!(parse_paths(" /a.jpg \n\n\"/b c.jpg\"\n/a.jpg\n  "), vec!["/a.jpg".to_string(), "/b c.jpg".to_string()]);
    assert!(parse_paths("  \n ").is_empty());
}

#[test]
fn the_window_waits_for_a_snapshot_then_lists_both_tabs() {
    let _serial = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = test_ctx();
    assert!(drawn(&ctx).is_empty(), "closed windows draw nothing");
    set_open(&ctx, true);
    assert!(drawn(&ctx).iter().any(|t| t.contains("Waiting for the first frame")));
    publish(&ctx, Snapshot { proposals: 5, shown: 4, by_source: vec![("Heuristic".into(), 3), ("MCP tester".into(), 2)], run: "Idle".into(), confirm: Some("apply 5 fields in 2 files".into()), edit_phase: "Ready".into(), edit_debug_backend: true, ..Default::default() });
    let text = drawn(&ctx);
    for expected in ["Refine", "Image Edit", "5 (4 shown)", "Heuristic: 3", "MCP tester: 2", "Waiting for an answer: apply 5 fields in 2 files", "Add sample proposals", "Ask to accept shown", "Confirm", "Decline", "Test backend", "Prompt"] {
        assert!(text.iter().any(|t| t.contains(expected)), "missing {expected:?} in {text:?}");
    }
    toggle(&ctx);
    assert!(!is_open(&ctx));
}

#[tokio::test]
async fn requests_drive_both_tabs_without_the_mouse() {
    let _serial = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let db = mem_db().await;
    for p in ["/p/a.jpg", "/p/b.jpg", "/p/c.jpg"] {
        metadata_edit::save_patch(&db, &RowSeed::from_path(p), &MetaPatch { tags: Some(vec!["old".into()]), caption: Some(Some("old caption".into())), ..Default::default() }).await.unwrap();
    }
    let (mut refine, mut edit) = panels(&db);
    let ctx = test_ctx();
    set_open(&ctx, true);

    push_request(&ctx, Request::Sample(vec!["/p/a.jpg".into(), "/p/b.jpg".into(), "/p/c.jpg".into(), "/p/not-in-library.jpg".into()]));
    frame(&mut refine, &mut edit, &ctx);
    for _ in 0..200 {
        refine.pump();
        if refine.proposals().len() == 3 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(refine.proposals().len(), 3, "a sample for each file that is in the library");
    let snap = snapshot(&refine, &edit);
    assert_eq!((snap.proposals, snap.shown), (3, 3));
    assert_eq!(snap.by_source.iter().map(|(l, n)| (l.as_str(), *n)).collect::<Vec<_>>(), vec![("Heuristic", 1), ("Jina M0", 1), ("Cloud sample", 1)]);

    push_request(&ctx, Request::CheckAllShown(false));
    push_request(&ctx, Request::AskAcceptShown);
    frame(&mut refine, &mut edit, &ctx);
    assert!(refine.confirm.is_none(), "nothing is checked, nothing is asked");
    push_request(&ctx, Request::CheckAllShown(true));
    push_request(&ctx, Request::AskAcceptShown);
    frame(&mut refine, &mut edit, &ctx);
    assert!(latest(&ctx).unwrap().confirm.unwrap().starts_with("apply "));
    push_request(&ctx, Request::DeclinePending);
    frame(&mut refine, &mut edit, &ctx);
    assert!(refine.confirm.is_none());
    push_request(&ctx, Request::ClearGenerated);
    frame(&mut refine, &mut edit, &ctx);
    assert!(refine.proposals().is_empty());

    push_request(&ctx, Request::UseDebugBackend(true));
    push_request(&ctx, Request::SetPrompt("Make it black and white".into()));
    frame(&mut refine, &mut edit, &ctx);
    assert!(edit.uses_debug_backend());
    assert_eq!(edit.params.prompt, "Make it black and white");
    let snap = latest(&ctx).unwrap();
    assert!(snap.edit_debug_backend && snap.edit_prompt == "Make it black and white" && snap.edit_source.is_none());
    push_request(&ctx, Request::RunEdit);
    frame(&mut refine, &mut edit, &ctx);
    assert!(!edit.is_busy(), "there is no image to edit yet");
    push_request(&ctx, Request::CancelEdit);
    push_request(&ctx, Request::UseDebugBackend(false));
    frame(&mut refine, &mut edit, &ctx);
    assert!(!edit.uses_debug_backend());
}

#[test]
fn nothing_is_published_while_the_window_is_closed() {
    let _serial = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = test_ctx();
    let (toast_tx, _rx) = crossbeam::channel::unbounded();
    let mut refine = RefinementsPanel::with_sources(toast_tx, DbSource::Global, StoreSource::Off, crate::ui::file_table::thumbs::service());
    let mut edit = ImageEditPanel::default();
    frame(&mut refine, &mut edit, &ctx);
    assert!(latest(&ctx).is_none());
    set_open(&ctx, true);
    frame(&mut refine, &mut edit, &ctx);
    assert_eq!(latest(&ctx).unwrap().edit_phase, "Ready");
}
