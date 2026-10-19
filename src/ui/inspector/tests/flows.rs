use std::time::Duration;

use super::super::meta::{Field, SaveState};
use super::super::prefs::DetailTab;
use super::super::state::BatchAction;
use super::*;
use crate::command::bus::testing::{serial, take_matching};
use crate::command::{ActionId, AppCommand, Scope, Section, Selector, ToastLevel, Tool};
use crate::database::metadata_edit::{MetaPatch, RowSeed, save_patch};

fn tags(list: &[&str]) -> Vec<String> {
    list.iter().map(|t| t.to_string()).collect()
}

async fn metadata_rig() -> Rig {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1100.0;
    rig.open_tab(DetailTab::Metadata);
    rig
}

/* ---------------------------------------------- autosave --------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn typing_a_caption_and_leaving_the_field_creates_the_row_and_saves_it() {
    let mut rig = metadata_rig().await;
    rig.select(&["a.png"]);
    rig.settle(6).await;
    assert!(rig.db_row("a.png").await.is_none(), "the file is not in the library yet");
    assert!(rig.has_text("Caption"), "{:?}", rig.texts());
    rig.click_field("Caption", 30.0);
    rig.type_text("A red square");
    assert_eq!(rig.insp.meta.state(), SaveState::Dirty);
    assert_eq!(rig.insp.meta.draft.caption, "A red square");
    assert_eq!(rig.backend.saves.lock().unwrap().len(), 0, "nothing is written while typing");
    rig.click_field("Category", 30.0);
    rig.until("the save to finish", |r| r.insp.meta.state() == SaveState::Saved).await;
    let row = rig.db_row("a.png").await.expect("the first edit created the row");
    assert_eq!(row.caption.as_deref(), Some("A red square"));
    assert_eq!(rig.row("a.png").caption.as_deref(), Some("A red square"), "the browser shows the saved value");
    assert!(rig.notices.iter().any(|(l, t)| *l == ToastLevel::Info && t.contains("Added a.png to the library")), "{:?}", rig.notices);
    assert!(rig.has_text("Saved"));
    let saves = rig.backend.saves.lock().unwrap().clone();
    assert_eq!(saves.len(), 1);
    assert_eq!(saves[0].1.fields(), vec!["caption"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn enter_in_a_single_line_field_saves_at_once() {
    let mut rig = metadata_rig().await;
    rig.select(&["b.png"]);
    rig.settle(6).await;
    rig.click_field("Category", 30.0);
    rig.type_text("pets");
    rig.press(Key::Enter);
    assert!(!rig.insp.lightbox.open, "the Enter that ends a text edit does not open the viewer");
    rig.until("the save to finish", |r| r.insp.meta.state() == SaveState::Saved).await;
    assert_eq!(rig.db_row("b.png").await.unwrap().category.as_deref(), Some("pets"));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idle_edit_is_saved_by_the_debounce_without_leaving_the_field() {
    let mut rig = metadata_rig().await;
    rig.insp.meta.debounce = Duration::from_millis(60);
    rig.select(&["a.png"]);
    rig.settle(6).await;
    rig.click_field("Caption", 30.0);
    rig.type_text("later");
    assert_eq!(rig.insp.meta.state(), SaveState::Dirty);
    rig.until("the debounced save", |r| r.insp.meta.state() == SaveState::Saved).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().caption.as_deref(), Some("later"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_save_shows_in_the_rail_and_as_a_notice_and_retry_works() {
    let mut rig = metadata_rig().await;
    *rig.backend.fail_saves.lock().unwrap() = Some("disk on fire".to_string());
    rig.select(&["a.png"]);
    rig.settle(6).await;
    rig.insp.meta.set_text(Field::Caption, "doomed".into(), std::time::Instant::now());
    rig.insp.flush_now(std::time::Instant::now());
    rig.until("the failure", |r| r.insp.meta.state() == SaveState::Error).await;
    rig.settle(2).await;
    assert!(rig.has_text("disk on fire"), "{:?}", rig.texts());
    assert!(rig.has_text("Retry") && rig.has_text("Not saved"));
    assert!(rig.notices.iter().any(|(l, t)| *l == ToastLevel::Error && t.contains("Could not save the metadata of a.png") && t.contains("disk on fire")), "{:?}", rig.notices);
    assert!(rig.insp.meta.is_dirty(), "the edit is kept for the retry");
    assert!(rig.db_row("a.png").await.is_none());
    *rig.backend.fail_saves.lock().unwrap() = None;
    eprintln!("DBG rect {:?} retry {:?} notsaved {:?} caption {:?}", rig.out.rect, rig.text_pos("Retry"), rig.text_pos("Not saved"), rig.text_pos("Caption"));
    rig.click_text("Retry");
    rig.until("the retry", |r| r.insp.meta.state() == SaveState::Saved).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().caption.as_deref(), Some("doomed"));
    assert_eq!(rig.insp.counters.saves_failed, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn moving_the_focus_saves_the_edits_of_the_file_that_was_left() {
    let mut rig = metadata_rig().await;
    rig.select(&["a.png"]);
    rig.settle(6).await;
    rig.insp.meta.set_text(Field::Caption, "left behind".into(), std::time::Instant::now());
    rig.select(&["b.png"]);
    rig.settle(2).await;
    rig.until("the old file's save", |r| r.insp.counters.saves_ok >= 1).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().caption.as_deref(), Some("left behind"));
    assert!(rig.db_row("b.png").await.is_none(), "the new file was not touched");
    assert_eq!(rig.insp.meta.path(), Some(rig.path("b.png").as_str()));
    assert_eq!(rig.insp.meta.draft.caption, "");
}

#[tokio::test(flavor = "multi_thread")]
async fn tags_are_chips_with_autocomplete_and_removal_saves_at_once() {
    let mut rig = metadata_rig().await;
    let other = RowSeed::from_row(&rig.row("b.png"));
    save_patch(&rig.db, &other, &MetaPatch { tags: Some(tags(&["sunset", "sunrise", "moon"])), ..Default::default() }).await.unwrap();
    rig.select(&["a.png"]);
    rig.until("the suggestion lists", |r| !r.insp.known_tags.is_empty()).await;
    assert_eq!(rig.insp.known_tags, tags(&["moon", "sunrise", "sunset"]));
    rig.click_field("No tags", 24.0);
    rig.type_text("sun");
    rig.settle(2).await;
    assert!(rig.has_text("sunset") && rig.has_text("sunrise") && !rig.has_text("moon"), "{:?}", rig.texts());
    rig.click_text("sunset");
    assert_eq!(rig.insp.meta.draft.tags, tags(&["sunset"]));
    rig.until("the tag save", |r| r.insp.meta.state() == SaveState::Saved).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().tags, tags(&["sunset"]));
    rig.click_field("sunset", 30.0);
    rig.type_text("night,beach");
    rig.press(Key::Enter);
    assert_eq!(rig.insp.meta.draft.tags, tags(&["sunset", "night", "beach"]), "a comma completes a tag and Enter completes the last");
    rig.until("the second save", |r| r.insp.meta.state() == SaveState::Saved).await;
    rig.click_text(crate::icons::p::X);
    assert_eq!(rig.insp.meta.draft.tags, tags(&["night", "beach"]), "the first chip's remove button took it out");
    rig.until("the removal save", |r| r.insp.meta.state() == SaveState::Saved).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().tags, tags(&["night", "beach"]));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unfinished_tag_is_not_carried_to_the_next_file() {
    let mut rig = metadata_rig().await;
    rig.select(&["a.png"]);
    rig.settle(6).await;
    rig.click_field("No tags", 24.0);
    rig.type_text("half");
    assert_eq!(rig.insp.tag_input, "half");
    rig.select(&["b.png"]);
    rig.settle(4).await;
    assert_eq!(rig.insp.tag_input, "", "the draft belongs to the file it was typed for");
    assert!(rig.insp.suggest_field.is_none());
    assert!(rig.insp.meta.draft.tags.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn values_that_arrive_for_the_open_file_replace_untouched_fields_only() {
    let mut rig = metadata_rig().await;
    rig.select(&["a.png"]);
    rig.settle(6).await;
    rig.insp.meta.set_text(Field::Caption, "mine".into(), std::time::Instant::now());
    let path = rig.path("a.png");
    if let Some(mut row) = rig.ex.table.get_mut_by_path(&path) {
        row.description = Some("written by the model".into());
        row.caption = Some("from the model".into());
    }
    rig.settle(3).await;
    assert_eq!(rig.insp.meta.draft.description, "written by the model");
    assert_eq!(rig.insp.meta.draft.caption, "mine", "the field being edited keeps the user's text");
}

/* ----------------------------------------------- batch ------------------------------------------------ */

#[tokio::test(flavor = "multi_thread")]
async fn batch_tags_reach_every_selected_file_the_browser_and_the_library() {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1100.0;
    rig.select(&["a.png", "b.png", "c.png", "sub"]);
    rig.settle(6).await;
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::AddTags(tags(&["trip", "Summer"])), false);
    rig.until("the batch", |r| r.insp.last_batch.is_some()).await;
    for name in ["a.png", "b.png", "c.png"] {
        assert_eq!(rig.row(name).tags, tags(&["trip", "Summer"]), "{name} in the browser");
        assert_eq!(rig.db_row(name).await.unwrap().tags, tags(&["trip", "Summer"]), "{name} in the library");
    }
    assert!(rig.row("notes.txt").tags.is_empty(), "files that were not selected stay as they are");
    let message = rig.insp.last_batch.clone().unwrap();
    assert!(message.contains("2 tags") && message.contains("3 files") && message.contains("3 added to the library"), "{message}");
    rig.settle(3).await;
    assert!(rig.has_text("trip") && rig.has_text("Summer"), "the common tags show as chips: {:?}", rig.texts());

    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::RemoveTags(tags(&["trip"])), false);
    rig.until("the removal", |r| r.row("a.png").tags == tags(&["Summer"])).await;
    assert_eq!(rig.db_row("c.png").await.unwrap().tags, tags(&["Summer"]));
}

#[tokio::test(flavor = "multi_thread")]
async fn batch_category_and_collections_report_counts() {
    let mut rig = Rig::new(1200.0).await;
    rig.select(&["a.png", "b.png"]);
    rig.settle(4).await;
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::SetCategory(Some("landscape".into())), false);
    rig.until("the category", |r| r.row("b.png").category.as_deref() == Some("landscape")).await;
    assert_eq!(rig.db_row("a.png").await.unwrap().category.as_deref(), Some("landscape"));
    assert!(rig.row("c.png").category.is_none());
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::SetCategory(None), false);
    rig.until("the cleared category", |r| r.row("a.png").category.is_none()).await;

    let Rig { insp, ex, .. } = &mut rig;
    insp.run_collection(ex, "Holidays", true);
    rig.until("the collection add", |r| r.insp.last_batch.as_deref().is_some_and(|m| m.contains("Holidays"))).await;
    let message = rig.insp.last_batch.clone().unwrap();
    assert!(message.contains("Added 2 to \"Holidays\"") && message.contains("new collection"), "{message}");
    let listed = crate::database::collections::list_with_counts(&rig.db).await.unwrap();
    assert_eq!((listed.len(), listed[0].count), (1, 2));

    rig.select(&["a.png"]);
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_collection(ex, "holidays", false);
    rig.until("the collection removal", |r| r.insp.last_batch.as_deref().is_some_and(|m| m.starts_with("Removed"))).await;
    assert!(rig.insp.last_batch.clone().unwrap().contains("Removed 1 from \"Holidays\""));
    let listed = crate::database::collections::list_with_counts(&rig.db).await.unwrap();
    assert_eq!(listed[0].count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_action_on_an_empty_or_folder_only_selection_does_nothing_and_says_so() {
    let mut rig = Rig::new(1200.0).await;
    rig.settle(2).await;
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::AddTags(tags(&["x"])), false);
    insp.run_collection(ex, "Any", true);
    assert_eq!(insp.take_notices().iter().filter(|(l, t)| *l == ToastLevel::Warning && t.contains("no files are selected")).count(), 2);
    rig.select(&["sub"]);
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::SetCategory(Some("x".into())), false);
    assert!(insp.take_notices().iter().any(|(l, _)| *l == ToastLevel::Warning));
    assert_eq!(insp.counters.batches, 0);
    rig.settle(2).await;
    assert!(rig.db_row("a.png").await.is_none(), "nothing was written to any file");
}

#[tokio::test(flavor = "multi_thread")]
async fn heavy_ai_work_on_a_large_selection_waits_for_a_confirmation_with_the_count() {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1400.0;
    let extra: Vec<Thumbnail> = (0..40).map(|i| file_row(rig.dir.path(), &format!("x{i:02}.png"), "png")).collect();
    rig.ex.table.bulk_load(extra);
    rig.ex.sync_view();
    let paths: Vec<String> = (0..40).map(|i| rig.path(&format!("x{i:02}.png"))).collect();
    rig.ex.select_paths(paths.iter().map(String::as_str));
    rig.settle(4).await;
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::Describe { regenerate: false }, false);
    let pending = insp.batch.confirm.clone().expect("a confirmation is pending");
    assert_eq!((pending.count, pending.action.clone()), (40, BatchAction::Describe { regenerate: false }));
    assert!(ex.streaming_interim.is_empty(), "nothing started yet");
    rig.settle(3).await;
    assert!(rig.has_text("Describe for 40 files?"), "{:?}", rig.texts());
    rig.click_text("Cancel");
    assert!(rig.insp.batch.confirm.is_none());

    let _serial = serial();
    take_matching(|_| true);
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::GenerateClip, false);
    assert!(insp.batch.confirm.is_some(), "CLIP is heavy too");
    insp.run_batch(ex, BatchAction::GenerateClip, true);
    let sent = take_matching(|c| matches!(c, AppCommand::Run { action: ActionId::GenerateClip, .. }));
    assert!(matches!(&sent[..], [AppCommand::Run { on: Selector::Paths(p), .. }] if p.len() == 40), "{sent:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn small_selections_run_without_asking_and_light_actions_never_ask() {
    let mut rig = Rig::new(1200.0).await;
    rig.select(&["a.png", "b.png", "c.png"]);
    rig.settle(3).await;
    let _serial = serial();
    take_matching(|_| true);
    let Rig { insp, ex, .. } = &mut rig;
    insp.run_batch(ex, BatchAction::GenerateClip, false);
    insp.run_batch(ex, BatchAction::Refine, false);
    insp.run_batch(ex, BatchAction::AttachToAssistant, false);
    assert!(insp.batch.confirm.is_none());
    assert_eq!(take_matching(|c| matches!(c, AppCommand::Run { action: ActionId::GenerateClip, .. })).len(), 1);
    assert_eq!(take_matching(|c| matches!(c, AppCommand::Refine { paths } if paths.len() == 3)).len(), 1);
    assert_eq!(take_matching(|c| matches!(c, AppCommand::AttachToAssistant { paths } if paths.len() == 3)).len(), 1);
}

/* ----------------------------------------------- AI tab ----------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn streaming_text_shows_for_the_focused_file_only_and_follow_is_off_by_default() {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1100.0;
    rig.open_tab(DetailTab::Ai);
    rig.ex.streaming_interim.insert(rig.path("a.png"), "A red square on white".into());
    rig.select(&["a.png"]);
    rig.settle(5).await;
    assert!(rig.has_text("Generating...") && rig.has_text("A red square on white"), "{:?}", rig.texts());
    assert!(!rig.ex.follow_active_vision, "following the model is off by default");
    rig.select(&["b.png"]);
    rig.settle(4).await;
    assert!(!rig.has_text("A red square on white") && !rig.has_text("Generating..."), "the other file's stream is hidden");
    rig.select(&["a.png"]);
    rig.settle(4).await;
    assert!(rig.has_text("A red square on white"), "coming back shows the running stream again");
    rig.click_text("Dismiss");
    rig.settle(2).await;
    assert!(rig.ex.streaming_interim.is_empty() && !rig.has_text("Generating..."));
}

#[tokio::test(flavor = "multi_thread")]
async fn following_the_model_moves_the_rail_only_when_follow_is_on() {
    use crate::ui::file_table::AIUpdate;
    let mut rig = Rig::new(1200.0).await;
    rig.select(&["a.png"]);
    rig.settle(4).await;
    let tx = rig.ex.viewer.ai_update_tx.clone();
    tx.send(AIUpdate::Interim { path: rig.path("b.png"), text: "from the model".into() }).unwrap();
    rig.ex.receive_ai_update(&rig.ctx);
    rig.settle(3).await;
    assert_eq!(rig.insp.last_target, format!("file {}", rig.path("a.png")), "the selection stays with follow off");
    assert!(rig.ex.streaming_interim.contains_key(&rig.path("b.png")), "the stream is still recorded");
    rig.ex.follow_active_vision = true;
    tx.send(AIUpdate::Interim { path: rig.path("c.png"), text: "next".into() }).unwrap();
    rig.ex.receive_ai_update(&rig.ctx);
    rig.settle(3).await;
    assert_eq!(rig.insp.last_target, format!("file {}", rig.path("c.png")), "the rail moves to the streaming file");
    assert_eq!(rig.ex.selection.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_ai_tab_links_to_the_settings_when_no_model_is_set_and_launches_the_other_tools() {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1100.0;
    rig.open_tab(DetailTab::Ai);
    rig.select(&["a.png"]);
    rig.settle(5).await;
    let _serial = serial();
    take_matching(|_| true);
    if crate::ui::inspector::target::current_ai_state() == crate::ui::inspector::target::AiState::NotConfigured {
        assert!(rig.has_text("AI is not configured"), "{:?}", rig.texts());
        rig.click_text("Choose the model folder");
        assert_eq!(take_matching(|c| matches!(c, AppCommand::ShowSettings { section: Some(Section::AiModels) })).len(), 1);
    }
    let a = rig.path("a.png");
    rig.click_text("Generate CLIP");
    assert_eq!(take_matching(|c| matches!(c, AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Paths(p) } if p == &[a.clone()])).len(), 1);
    rig.click_text("Find similar");
    assert_eq!(take_matching(|c| matches!(c, AppCommand::OpenScope { scope: Scope::Similar(p), .. } if *p == a)).len(), 1);
    rig.click_text("Refine this item");
    assert_eq!(take_matching(|c| matches!(c, AppCommand::Refine { paths } if paths == &[a.clone()])).len(), 1);
    rig.open_tab(DetailTab::Edit);
    rig.settle(2).await;
    rig.click_text("Open in Image Edit");
    assert_eq!(take_matching(|c| matches!(c, AppCommand::OpenTool { tool: Tool::ImageEdit, path: Some(p), .. } if *p == a)).len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_header_buttons_send_commands_for_the_focused_file() {
    let mut rig = Rig::new(1200.0).await;
    rig.select(&["a.png"]);
    rig.settle(5).await;
    let _serial = serial();
    take_matching(|_| true);
    let a = rig.path("a.png");
    for (label, action) in [("Open", ActionId::Open), ("Reveal", ActionId::Reveal), ("Copy path", ActionId::CopyPath)] {
        rig.click_text(label);
        let sent = take_matching(|c| matches!(c, AppCommand::Run { on: Selector::Paths(p), .. } if p == &[a.clone()]));
        assert!(matches!(&sent[..], [AppCommand::Run { action: got, .. }] if *got == action), "{label}: {sent:?}");
    }
}
