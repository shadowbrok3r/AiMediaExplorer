//! Wizard tests: the step state machine, what each step writes, the dialog at three window widths and the checklist card.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Context, Event, Pos2, RawInput, Rect, Shape, vec2};

use super::model::{Choice, LocationForm};
use super::*;
use crate::UiSettings;
use crate::command::bus::testing::{serial, take_matching};
use crate::settings::location::Candidate;
use crate::settings::{Backend, Loaded, SettingsStore};
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

struct Nop;

#[async_trait]
impl Backend for Nop {
    async fn load(&self) -> anyhow::Result<Loaded> {
        Ok(Loaded { settings: UiSettings::default(), existed: true })
    }
    async fn merge(&self, _: &[&'static str], _: &UiSettings) -> anyhow::Result<()> {
        Ok(())
    }
}

fn store() -> Arc<SettingsStore> {
    let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
    store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
    store
}

fn fresh() {
    ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
}

fn legacy(root: &Path) -> Vec<Candidate> {
    let store = root.join("db").join("ai_search");
    std::fs::create_dir_all(store.join("manifest")).unwrap();
    crate::settings::location::detect_legacy(Some(root), None)
}

/// A wizard whose location form uses temporary folders.
fn wizard_in(root: &Path, detected: Vec<Candidate>) -> Wizard {
    let mut w = Wizard::default();
    w.location = Some(LocationForm::new(detected, root.join("new").join("db")));
    w.folders = model::FoldersForm::prefilled(&[], &[]);
    w.save_scanned = true;
    w.open = true;
    w
}

fn text_rects(shapes: &[ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(shape: &Shape, clip: Rect, out: &mut Vec<(String, Rect)>) {
        match shape {
            Shape::Text(t) if clip.width() > 0.0 && clip.height() > 0.0 => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()).intersect(clip))),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, clip, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, s.clip_rect, &mut out));
    out
}

struct Rig {
    wizard: Wizard,
    ctx: Context,
    size: (f32, f32),
    library_open: bool,
}

struct Out {
    shapes: Vec<ClippedShape>,
}

impl Out {
    fn text(&self) -> String {
        texts(&self.shapes).into_iter().map(|(t, _)| t).collect::<Vec<_>>().join("\n")
    }

    fn has(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }

    fn pos(&self, text: &str) -> Pos2 {
        text_pos(&self.shapes, text)
    }
}

impl Rig {
    fn new(wizard: Wizard, width: f32) -> Self {
        Self { wizard, ctx: test_ctx(), size: (width, 800.0), library_open: false }
    }

    fn frame(&mut self, events: Vec<Event>) -> Out {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.size.0, self.size.1))), events, ..Default::default() };
        let (wizard, open) = (&mut self.wizard, self.library_open);
        let out = self.ctx.run_ui(input, |_ui| show(&self.ctx, wizard, open));
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        Out { shapes }
    }

    fn settle(&mut self) -> Out {
        (0..3).map(|_| self.frame(Vec::new())).last().unwrap()
    }

    fn click(&mut self, text: &str) -> Out {
        let at = self.settle().pos(text);
        self.frame(press_release(at + vec2(6.0, 6.0)));
        self.settle()
    }
}

/* ------------------------------------------ state machine ------------------------------------------ */

#[test]
fn finishing_a_step_records_it_and_moves_on_and_the_last_step_closes() {
    fresh();
    let mut w = Wizard::default();
    w.open = true;
    for (step, state) in [(Step::Location, StepState::Done), (Step::Folders, StepState::Skipped), (Step::Ai, StepState::Done)] {
        assert_eq!(w.current(), step);
        w.finish_step(state);
        assert_eq!(ui_prefs::get().checklist.state(step), state);
    }
    assert_eq!(w.current(), Step::Scan);
    assert!(w.open);
    w.finish_step(StepState::Skipped);
    assert!(!w.open, "the last step closes the wizard");
    assert!(ui_prefs::get().checklist.is_complete());
    fresh();
}

#[test]
fn back_moves_one_step_and_stops_at_the_first() {
    let mut w = Wizard::default();
    w.step = Step::Ai;
    w.back();
    assert_eq!(w.current(), Step::Folders);
    w.back();
    w.back();
    assert_eq!(w.current(), Step::Location);
}

#[test]
fn opening_at_a_step_or_at_the_first_pending_one() {
    fresh();
    let mut w = Wizard::default();
    w.open_at(Step::Ai, Reason::Asked);
    assert!(w.open && w.current() == Step::Ai && w.reason == Reason::Asked);
    w.close();
    assert!(!w.open);
    ui_prefs::update(|p| {
        p.checklist.location = StepState::Done;
        p.checklist.folders = StepState::Skipped;
    });
    w.open_next_pending();
    assert_eq!(w.current(), Step::Ai, "the first step that is still to do");
    fresh();
}

#[test]
fn a_finished_checklist_opens_the_first_step_again() {
    fresh();
    ui_prefs::update(|p| {
        for s in Step::ALL {
            p.checklist.set(s, StepState::Done);
        }
    });
    let mut w = Wizard::default();
    w.open_next_pending();
    assert_eq!(w.current(), Step::Location);
    fresh();
}

#[test]
fn an_existing_library_marks_the_location_step_done_once() {
    fresh();
    mark_location_done_if_present();
    assert_eq!(ui_prefs::get().checklist.location, StepState::Done);
    ui_prefs::update(|p| p.checklist.location = StepState::Skipped);
    mark_location_done_if_present();
    assert_eq!(ui_prefs::get().checklist.location, StepState::Skipped, "a choice the user made is kept");
    fresh();
}

#[test]
fn the_card_needs_an_empty_loaded_library_listing() {
    assert!(is_empty_library(&Scope::Library, 0, false));
    assert!(!is_empty_library(&Scope::Library, 3, false));
    assert!(!is_empty_library(&Scope::Library, 0, true), "a load in flight is not an empty library");
    assert!(!is_empty_library(&Scope::Recent, 0, false));
    assert!(!is_empty_library(&Scope::Folder("/a".into()), 0, false));
}

/* ------------------------------------------- the steps --------------------------------------------- */

#[test]
fn choosing_a_new_library_creates_the_folder_and_asks_the_database_to_open() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let mut w = wizard_in(root.path(), Vec::new());
    let outcome = w.continue_location().unwrap();
    assert!(matches!(outcome, Outcome::Opening(ref p) if *p == root.path().join("new").join("db")));
    assert!(root.path().join("new").join("db").is_dir());
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ConnectDatabase)).len(), 1);
    assert_eq!(ui_prefs::get().checklist.location, StepState::Done);
    assert_eq!(w.current(), Step::Folders);
    fresh();
}

#[test]
fn an_older_library_is_offered_first_and_opened_without_touching_it() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let detected = legacy(root.path());
    let mut w = wizard_in(root.path(), detected);
    assert_eq!(w.location.as_ref().unwrap().choice, Choice::Legacy(0));
    let before = std::fs::read_dir(root.path().join("db").join("ai_search")).unwrap().count();
    let outcome = w.continue_location().unwrap();
    assert!(matches!(outcome, Outcome::Opening(ref p) if *p == root.path().join("db").join("ai_search")));
    assert_eq!(std::fs::read_dir(root.path().join("db").join("ai_search")).unwrap().count(), before, "the library folder is not changed");
    assert!(!root.path().join("new").exists(), "no second library is created");
    fresh();
}

#[test]
fn a_folder_with_other_files_is_refused_and_nothing_changes() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let busy = root.path().join("busy");
    std::fs::create_dir_all(&busy).unwrap();
    std::fs::write(busy.join("photo.jpg"), "x").unwrap();
    let mut w = wizard_in(root.path(), Vec::new());
    let form = w.location.as_mut().unwrap();
    form.choice = Choice::Custom;
    form.custom = busy.display().to_string();
    assert!(w.continue_location().is_err());
    assert_eq!(w.current(), Step::Location, "the wizard stays on the step");
    assert_eq!(ui_prefs::get().checklist.location, StepState::Pending);
    assert!(take_matching(|c| matches!(c, AppCommand::ConnectDatabase)).is_empty());
    assert!(busy.join("photo.jpg").exists());
    fresh();
}

#[test]
fn skipping_the_location_takes_the_default_folder() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let mut w = wizard_in(root.path(), legacy(root.path()));
    w.use_default_location().unwrap();
    assert!(root.path().join("new").join("db").is_dir(), "the default folder, not the older library");
    fresh();
}

#[test]
fn the_ticked_folders_become_the_watched_folders() {
    fresh();
    let mut w = Wizard::default();
    w.folders = model::FoldersForm::prefilled(&[], &[]);
    w.folders.add("/pics", true);
    w.folders.add("/vids", false);
    w.folders.add("/docs", true);
    w.step = Step::Folders;
    assert_eq!(w.continue_folders(), 2);
    assert_eq!(ui_prefs::get().library.watched, vec!["/pics".to_string(), "/docs".to_string()]);
    assert_eq!(ui_prefs::get().checklist.folders, StepState::Done);
    assert_eq!(w.current(), Step::Ai);

    fresh();
    let mut none = Wizard::default();
    none.step = Step::Folders;
    assert_eq!(none.continue_folders(), 0);
    assert_eq!(ui_prefs::get().checklist.folders, StepState::Skipped, "no folder is a skip");
    fresh();
}

#[test]
fn the_ai_step_records_the_mode_and_a_local_vision_folder() {
    fresh();
    let store = store();
    let mut w = Wizard::default();
    w.step = Step::Ai;
    w.ai.mode = AiMode::Local;
    w.ai.vision_dir = "  /models/joycaption  ".into();
    w.continue_ai_with(&store);
    assert_eq!(store.read(|s| s.joycaption_model_dir.clone()).as_deref(), Some("/models/joycaption"));
    assert_eq!(ui_prefs::get().checklist.ai_mode, AiMode::Local);
    assert_eq!(ui_prefs::get().checklist.ai, StepState::Done);

    fresh();
    let untouched = self::store();
    let mut off = Wizard::default();
    off.step = Step::Ai;
    off.ai.mode = AiMode::Off;
    off.continue_ai_with(&untouched);
    assert_eq!(untouched.read(|s| s.joycaption_model_dir.clone()), None, "off changes no setting");
    assert!(untouched.dirty_fields().is_empty());
    fresh();
}

#[test]
fn the_first_scan_opens_one_recursive_tab_per_watched_folder_and_can_switch_on_saving() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let store = store();
    ui_prefs::update(|p| {
        p.watch("/pics");
        p.watch("/vids");
    });
    let mut w = Wizard::default();
    w.step = Step::Scan;
    w.save_scanned = true;
    assert_eq!(w.start_scan_with(&store), 2);
    let sent = take_matching(|c| matches!(c, AppCommand::OpenScope { recursive: true, .. }));
    assert_eq!(sent.len(), 2);
    assert!(store.read(|s| s.auto_save_to_database));
    assert_eq!(ui_prefs::get().checklist.scan, StepState::Done);
    assert!(!w.open, "the wizard is over");

    fresh();
    let quiet = self::store();
    let mut w = Wizard::default();
    w.step = Step::Scan;
    w.save_scanned = false;
    assert_eq!(w.start_scan_with(&quiet), 0);
    assert!(!quiet.read(|s| s.auto_save_to_database));
    assert_eq!(ui_prefs::get().checklist.scan, StepState::Skipped, "no folders is a skip");
    fresh();
}

#[test]
fn skip_setup_takes_the_default_location_and_skips_the_rest() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let mut w = wizard_in(root.path(), Vec::new());
    w.skip_all().unwrap();
    assert!(!w.open);
    let c = ui_prefs::get().checklist;
    assert_eq!((c.location, c.folders, c.ai, c.scan), (StepState::Done, StepState::Skipped, StepState::Skipped, StepState::Skipped));
    assert!(root.path().join("new").join("db").is_dir());
    fresh();
}

/* ------------------------------------------ the dialog --------------------------------------------- */

#[test]
fn every_step_draws_at_700_1000_and_1600_px_inside_the_window() {
    fresh();
    let root = tempfile::tempdir().unwrap();
    for width in [700.0, 1000.0, 1600.0] {
        for (step, needle) in [(Step::Location, "Create a new library at"), (Step::Folders, "Choose the folders"), (Step::Ai, "AI features describe files"), (Step::Scan, "Save scanned files to the library")] {
            let mut w = wizard_in(root.path(), Vec::new());
            w.step = step;
            let mut rig = Rig::new(w, width);
            let out = rig.settle();
            assert!(out.text().contains(needle), "{step:?} at {width}px:\n{}", out.text());
            assert!(out.text().contains(&format!("Step {} of 4", step.index() + 1)));
            for (text, rect) in text_rects(&out.shapes) {
                assert!(rect.right() <= width + 0.5 && rect.left() >= -0.5, "{step:?} at {width}px: {text:?} leaves the window ({:?})", rect);
            }
        }
    }
}

#[test]
fn the_dialog_offers_the_older_library_the_default_and_a_custom_folder() {
    fresh();
    let root = tempfile::tempdir().unwrap();
    let detected = legacy(root.path());
    let mut rig = Rig::new(wizard_in(root.path(), detected), 1000.0);
    let text = rig.settle().text();
    assert!(text.contains("Use the library found at"), "{text}");
    assert!(text.contains("Create a new library at"));
    assert!(text.contains("Another folder:"));
    assert!(text.contains("is opened as it is, nothing is overwritten"), "{text}");
}

#[test]
fn the_dialog_explains_a_refused_folder_and_disables_continue() {
    fresh();
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "x").unwrap();
    let mut w = wizard_in(root.path(), Vec::new());
    let form = w.location.as_mut().unwrap();
    form.choice = Choice::Custom;
    form.custom = root.path().display().to_string();
    let mut rig = Rig::new(w, 1000.0);
    let out = rig.settle();
    assert!(out.text().contains("not empty"), "{}", out.text());
    rig.click("Continue");
    assert_eq!(rig.wizard.current(), Step::Location, "a disabled Continue does nothing");
}

#[test]
fn the_buttons_move_through_the_steps_and_record_the_checklist() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    let root = tempfile::tempdir().unwrap();
    let mut rig = Rig::new(wizard_in(root.path(), Vec::new()), 1000.0);
    rig.click("Continue");
    assert_eq!(rig.wizard.current(), Step::Folders, "{}", rig.settle().text());
    assert_eq!(ui_prefs::get().checklist.location, StepState::Done);
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ConnectDatabase)).len(), 1);
    rig.click("Skip this step");
    assert_eq!(rig.wizard.current(), Step::Ai);
    assert_eq!(ui_prefs::get().checklist.folders, StepState::Skipped);
    rig.click("Back");
    assert_eq!(rig.wizard.current(), Step::Folders);
    rig.click("Skip this step");
    rig.click("Continue");
    assert_eq!(rig.wizard.current(), Step::Scan);
    assert_eq!(ui_prefs::get().checklist.ai, StepState::Done);
    rig.click("Skip this step");
    assert!(!rig.wizard.open);
    fresh();
}

#[test]
fn the_ai_step_shows_the_zeroclaw_hook_and_a_local_folder_check() {
    fresh();
    let root = tempfile::tempdir().unwrap();
    let mut w = wizard_in(root.path(), Vec::new());
    w.step = Step::Ai;
    w.ai.mode = AiMode::Zeroclaw;
    let mut rig = Rig::new(w, 1000.0);
    let out = rig.settle();
    assert!(out.text().contains("The ZeroClaw connection is not set up in this build."), "the hook is called:\n{}", out.text());
    assert!(rig.wizard.zc_done, "the stub reports the step complete");
    rig.wizard.ai.mode = AiMode::Local;
    rig.wizard.ai.vision_dir = root.path().display().to_string();
    let out = rig.settle();
    assert!(out.text().contains("No model files"), "{}", out.text());
    assert!(out.has("Detect"));
}

#[test]
fn a_closed_wizard_draws_nothing() {
    let mut w = Wizard::default();
    w.open = false;
    let mut rig = Rig::new(w, 1000.0);
    assert!(rig.settle().text().is_empty());
}

/* ------------------------------------------ the card ----------------------------------------------- */

fn card(ctx: &Context, events: Vec<Event>) -> Out {
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 600.0))), events, ..Default::default() };
    let out = ctx.run_ui(input, |ui| {
        eframe::egui::CentralPanel::default().show(ui, |ui| checklist_card(ui));
    });
    let shapes = out.shapes.clone();
    out.drop_without_applying_deltas();
    Out { shapes }
}

#[test]
fn the_card_lists_the_steps_with_their_state_and_opens_the_wizard_at_one() {
    fresh();
    let _serial = serial();
    take_matching(|_| true);
    ui_prefs::update(|p| p.checklist.location = StepState::Done);
    let ctx = test_ctx();
    card(&ctx, Vec::new());
    let out = card(&ctx, Vec::new());
    for step in Step::ALL {
        assert!(out.has(step.label()), "{}", out.text());
    }
    assert!(out.text().contains("1 of 4 steps"));
    assert!(out.has("done") && out.has("to do"));
    let at = out.pos("Open");
    card(&ctx, press_release(at + vec2(4.0, 4.0)));
    let sent = take_matching(|c| matches!(c, AppCommand::Wizard { step: Some(_) }));
    assert_eq!(sent.len(), 1);
    assert!(matches!(sent[0], AppCommand::Wizard { step: Some(Step::Folders) }), "the first step still to do");
    fresh();
}

#[test]
fn hiding_the_card_is_remembered() {
    fresh();
    let ctx = test_ctx();
    card(&ctx, Vec::new());
    let at = card(&ctx, Vec::new()).pos("Hide");
    card(&ctx, press_release(at + vec2(4.0, 4.0)));
    assert!(ui_prefs::get().checklist.dismissed);
    assert!(!ui_prefs::get().checklist.card_visible(true));
    fresh();
}
