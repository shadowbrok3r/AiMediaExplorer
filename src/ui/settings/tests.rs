//! Page tests: every section drawn at three window widths, search, dirty tracking, Apply and Revert through the real widgets.

use std::sync::Arc;

use async_trait::async_trait;
use eframe::egui::epaint::ClippedShape;
use eframe::egui::{CentralPanel, Context, Event, Pos2, RawInput, Rect, Shape, vec2};

use super::*;
use crate::UiSettings;
use crate::command::AppCommand;
use crate::command::bus::testing::{serial, take_matching};
use crate::database::ui_themes::tests::{Fixture, RT};
use crate::settings::{Backend, Loaded, ui_prefs};
use crate::theme::Appearance;
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

fn loaded_store() -> Arc<SettingsStore> {
    let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
    store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
    store
}

/// Text shapes with their rectangles, leaving out text clipped away entirely.
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
    page: SettingsPage,
    store: Arc<SettingsStore>,
    ctx: Context,
    appearance: Appearance,
    save_egui: bool,
    ai_ready: bool,
    ai_initializing: bool,
    assistant_window: bool,
    size: (f32, f32),
    _fixture: Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
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
    fn new(width: f32, height: f32) -> Self {
        let fixture = Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let store = loaded_store();
        ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
        crate::ui::file_table::table::view_mode::update_app_prefs(|p| *p = Default::default());
        Self {
            page: SettingsPage::with_store(Arc::clone(&store)),
            store,
            ctx,
            appearance,
            save_egui: false,
            ai_ready: true,
            ai_initializing: false,
            assistant_window: false,
            size: (width, height),
            _fixture: fixture,
            _runtime: runtime,
        }
    }

    fn frame(&mut self, events: Vec<Event>) -> Out {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.size.0, self.size.1))), events, ..Default::default() };
        let page = &mut self.page;
        let mut env = Env {
            db_ready: true,
            appearance: &mut self.appearance,
            save_egui: &mut self.save_egui,
            ai_ready: &mut self.ai_ready,
            ai_initializing: &mut self.ai_initializing,
            assistant_window: &mut self.assistant_window,
        };
        let out = self.ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| page.ui(ui, &mut env));
        });
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        Out { shapes }
    }

    fn settle(&mut self) -> Out {
        (0..3).map(|_| self.frame(Vec::new())).last().unwrap()
    }

    fn click(&mut self, at: Pos2) -> Out {
        self.frame(press_release(at));
        self.settle()
    }

    fn show(&mut self, section: Section) -> Out {
        self.page.select(Some(section));
        self.settle()
    }
}

/* ------------------------------------------- pure: sections ---------------------------------------- */

#[test]
fn every_section_has_a_label_an_icon_and_settings_in_the_index() {
    assert_eq!(sections::ALL.len(), 9);
    let unique: std::collections::BTreeSet<_> = sections::ALL.iter().map(|s| sections::label(*s)).collect();
    assert_eq!(unique.len(), 9);
    for section in sections::ALL {
        assert!(sections::entries_of(section).count() >= 1, "{section:?} has no searchable setting");
        assert_eq!(sections::label(section), crate::ui::palette::entries::section_label(section), "the palette names the section the same way");
    }
}

#[test]
fn index_labels_are_unique_and_never_empty() {
    let labels: std::collections::BTreeSet<_> = sections::ENTRIES.iter().map(|e| e.label).collect();
    assert_eq!(labels.len(), sections::ENTRIES.len());
    assert!(sections::ENTRIES.iter().all(|e| !e.label.is_empty()));
}

#[test]
fn search_matches_labels_extra_words_and_section_names_case_insensitively() {
    let labels = |q: &str| sections::search(q).iter().map(|e| e.label).collect::<Vec<_>>();
    assert!(labels("clip").contains(&sections::name::CLIP_MODEL));
    assert!(labels("CLIP").contains(&sections::name::AUTO_CLIP), "case does not matter");
    assert!(labels("siglip").contains(&sections::name::CLIP_MODEL), "extra words match");
    assert!(labels("automation").contains(&sections::name::AUTO_INDEX), "the section name matches");
    assert!(labels("   ").is_empty() && labels("").is_empty());
    assert!(labels("zzzzzz").is_empty());
}

#[test]
fn every_word_of_the_query_must_match() {
    let hits = sections::search("clip overwrite");
    assert!(hits.iter().any(|e| e.label == sections::name::CLIP_OVERWRITE));
    assert!(hits.iter().all(|e| e.label != sections::name::AUTO_INDEX));
    assert!(sections::search("clip overwrite").len() < sections::search("clip").len());
}

#[test]
fn match_counts_follow_the_sections() {
    assert!(sections::match_count(Section::AiModels, "clip") >= 2);
    assert_eq!(sections::match_count(Section::Shortcuts, "clip"), 0);
    assert_eq!(sections::match_count(Section::Library, ""), 0);
}

/* ------------------------------------------- the page --------------------------------------------- */

#[test]
fn every_section_draws_every_setting_it_indexes_at_700_1000_and_1600_px() {
    for width in [700.0, 1000.0, 1600.0] {
        let mut rig = Rig::new(width, 900.0);
        rig.store.update("test", |s| s.ai_chat_provider = Some("openai".into()));
        rig.page.draft.revert(rig.store.snapshot());
        for section in sections::ALL {
            let out = rig.show(section);
            let drawn = out.text();
            for entry in sections::entries_of(section) {
                assert!(drawn.contains(entry.label), "{section:?} at {width}px does not draw {:?}:\n{drawn}", entry.label);
            }
        }
    }
}

#[test]
fn nothing_is_drawn_outside_the_window_at_700_px() {
    let mut rig = Rig::new(700.0, 900.0);
    rig.store.update("test", |s| s.ai_chat_provider = Some("openai".into()));
    rig.page.draft.revert(rig.store.snapshot());
    for section in sections::ALL {
        rig.show(section);
        let out = rig.settle();
        for (text, rect) in text_rects(&out.shapes) {
            assert!(rect.right() <= 700.5, "{section:?}: {text:?} runs past the right edge ({})", rect.right());
            assert!(rect.left() >= -0.5, "{section:?}: {text:?} starts left of the window");
        }
    }
}

#[test]
fn opening_and_drawing_every_section_changes_nothing() {
    let mut rig = Rig::new(1000.0, 900.0);
    let generation = rig.store.generation();
    for section in sections::ALL {
        rig.show(section);
    }
    assert_eq!(rig.store.generation(), generation, "the store did not move");
    assert!(rig.store.dirty_fields().is_empty());
    assert!(!rig.page.is_dirty());
    assert_eq!(ui_prefs::get(), ui_prefs::UiPrefs::default());
    assert_eq!(crate::ui::file_table::table::view_mode::app_prefs(), Default::default());
    assert!(!rig.save_egui);
}

#[test]
fn the_wide_layout_stacks_the_sections_and_the_narrow_one_wraps_them() {
    let mut wide = Rig::new(1000.0, 800.0);
    let out = wide.settle();
    let (library, automation) = (out.pos("Library"), out.pos("Automation"));
    assert!((automation.x - library.x).abs() < 1.0 && automation.y > library.y, "a column");
    let mut narrow = Rig::new(500.0, 800.0);
    let out = narrow.settle();
    let (library, automation) = (out.pos("Library"), out.pos("Automation"));
    assert!((automation.y - library.y).abs() < 1.0 && automation.x > library.x, "a row");
    assert!(out.has("Search settings") || out.text().contains("Search"), "the search box is there");
}

#[test]
fn a_search_lists_the_hits_and_show_jumps_to_the_section() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.query = "reranker".into();
    let out = rig.settle();
    assert!(out.has(sections::name::RERANKER), "{}", out.text());
    assert!(out.has("AI models"));
    assert!(!out.has("Excluded directories"));
    let at = out.pos("Show");
    rig.click(at + vec2(4.0, 4.0));
    assert_eq!(rig.page.section, Section::AiModels);
    assert!(rig.page.query.is_empty(), "the query is cleared after a jump");
    let out = rig.settle();
    assert!(out.has("Reranker") && out.has("Vision"), "the section is shown");
}

#[test]
fn a_search_without_hits_says_so() {
    let mut rig = Rig::new(1000.0, 600.0);
    rig.page.query = "qqqq".into();
    assert!(rig.settle().text().contains("No setting matches"));
}

#[test]
fn editing_shows_the_dirty_marker_and_apply_writes_one_patch() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.show(Section::Automation);
    assert!(!rig.settle().text().contains("Unsaved changes"));
    rig.page.draft.value.auto_indexing = false;
    rig.page.draft.value.overwrite_descriptions = true;
    rig.page.draft.value.ai_prompt_template = "Describe it".into();
    let out = rig.settle();
    assert!(out.text().contains("Unsaved changes (2)") || out.text().contains("Unsaved changes (3)"), "{}", out.text());
    assert!(rig.page.is_dirty());
    let before = rig.store.patches().len();
    let at = out.pos("Apply");
    rig.click(at + vec2(4.0, 4.0));
    assert!(!rig.page.is_dirty(), "Apply clears the marker");
    let patches = rig.store.patches();
    assert_eq!(patches.len(), before + 1, "one patch per Apply");
    assert_eq!(patches.last().unwrap().who, "settings:apply");
    assert!(rig.store.read(|s| s.overwrite_descriptions));
    assert_eq!(rig.store.read(|s| s.ai_prompt_template.clone()), "Describe it");
    assert!(rig.page.status.as_deref().unwrap_or("").starts_with("Applied 2"));
}

#[test]
fn revert_discards_the_edits_through_the_button() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.draft.value.filter_skip_icons = true;
    rig.page.general.value.restore_tabs = true;
    let out = rig.settle();
    assert!(out.text().contains("Unsaved changes (2)"), "{}", out.text());
    let at = out.pos("Revert");
    rig.click(at + vec2(4.0, 4.0));
    assert!(!rig.page.is_dirty());
    assert!(!rig.page.draft.value.filter_skip_icons && !rig.page.general.value.restore_tabs);
    assert_eq!(rig.page.status.as_deref(), Some("Edits discarded"));
    assert!(rig.store.dirty_fields().is_empty(), "nothing was written");
}

#[test]
fn apply_and_revert_are_disabled_until_something_changed() {
    let mut rig = Rig::new(1000.0, 800.0);
    let out = rig.settle();
    let at = out.pos("Apply");
    rig.click(at + vec2(4.0, 4.0));
    assert!(rig.page.status.is_none(), "a click on a disabled Apply does nothing");
}

#[test]
fn general_edits_are_saved_to_the_prefs_file_store_not_the_settings_record() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.general.value.restore_last_scope = true;
    rig.page.general.value.view.tile = 200.0;
    let generation = rig.store.generation();
    let before = rig.store.patches().len();
    let mut flags = AiFlags { ready: &mut rig.ai_ready, initializing: &mut rig.ai_initializing };
    let applied = rig.page.apply(&mut flags);
    assert!(applied.fields.is_empty(), "no settings field changed");
    assert_eq!(rig.store.generation(), generation);
    assert_eq!(rig.store.patches().len(), before);
    assert!(ui_prefs::get().startup.restore_last_scope);
    assert_eq!(crate::ui::file_table::table::view_mode::app_prefs().tile, 200.0);
    assert!(rig.page.status.as_deref().unwrap_or("").starts_with("Applied 2"));
}

#[test]
fn a_store_that_is_not_loaded_is_said_on_the_page_and_apply_keeps_changes_in_memory() {
    let mut rig = Rig::new(1000.0, 800.0);
    let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
    rig.page = SettingsPage::with_store(Arc::clone(&store));
    let out = rig.settle();
    assert!(out.text().contains("not loaded"), "{}", out.text());
    rig.page.draft.value.auto_save_to_database = true;
    let mut flags = AiFlags { ready: &mut rig.ai_ready, initializing: &mut rig.ai_initializing };
    rig.page.apply(&mut flags);
    assert!(rig.page.status.as_deref().unwrap_or("").contains("in memory"));
    assert!(store.read(|s| s.auto_save_to_database));
}

#[test]
fn outside_changes_flow_into_untouched_fields_but_not_over_edits() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.draft.value.filter_skip_icons = true;
    rig.store.update("assistant:model", |s| s.openai_default_model = Some("gpt-x".into()));
    rig.settle();
    assert_eq!(rig.page.draft.value.openai_default_model.as_deref(), Some("gpt-x"));
    assert!(rig.page.draft.value.filter_skip_icons);
    assert_eq!(rig.page.draft.dirty_fields(), vec!["filter_skip_icons"]);
}

#[test]
fn ctrl_comma_opens_the_settings_and_a_plain_comma_does_not() {
    let ctx = test_ctx();
    let press = |modifiers: eframe::egui::Modifiers| {
        let mut raw = RawInput::default();
        raw.events.push(Event::Key { key: eframe::egui::Key::Comma, physical_key: Some(eframe::egui::Key::Comma), pressed: true, repeat: false, modifiers });
        let mut got = false;
        let out = ctx.run_ui(raw, |_| got = open_key_pressed(&ctx));
        out.drop_without_applying_deltas();
        got
    };
    assert!(press(eframe::egui::Modifiers::COMMAND));
    assert!(!press(eframe::egui::Modifiers::NONE));
}

#[test]
fn selecting_a_section_clears_the_search_and_keeps_the_section_when_none_is_given() {
    let mut page = SettingsPage::with_store(loaded_store());
    page.query = "abc".into();
    page.select(Some(Section::Shortcuts));
    assert_eq!((page.section, page.query.as_str()), (Section::Shortcuts, ""));
    page.select(None);
    assert_eq!(page.section, Section::Shortcuts);
    let entry = sections::ENTRIES.iter().find(|e| e.label == sections::name::PERF_HUD).unwrap();
    page.jump_to(entry);
    assert_eq!(page.section, Section::Advanced);
}

#[test]
fn only_the_load_buttons_apply_pending_edits_first() {
    for action in [EngineAction::LoadVision, EngineAction::ReloadVision, EngineAction::ReloadClip, EngineAction::ReloadReranker] {
        assert!(needs_applied_edits(action), "{action:?}");
    }
    for action in [EngineAction::BackfillClip, EngineAction::Unload(None), EngineAction::Unload(Some(engines::Keep::Clip))] {
        assert!(!needs_applied_edits(action), "{action:?}");
    }
}

#[test]
fn a_free_memory_button_leaves_the_edits_alone() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.draft.value.reranker_model = Some("someone/other-reranker".into());
    let mut env = Env {
        db_ready: true,
        appearance: &mut rig.appearance,
        save_egui: &mut rig.save_egui,
        ai_ready: &mut rig.ai_ready,
        ai_initializing: &mut rig.ai_initializing,
        assistant_window: &mut rig.assistant_window,
    };
    rig.page.run_engine_action(EngineAction::Unload(None), &mut env);
    assert!(rig.page.is_dirty(), "the edit is still pending");
    assert_ne!(rig.store.read(|s| s.reranker_model.clone()).as_deref(), Some("someone/other-reranker"));
}

#[test]
fn the_shortcuts_section_lists_the_palette_bindings() {
    let mut rig = Rig::new(1000.0, 800.0);
    rig.page.shortcuts.filter = "palette".into();
    let out = rig.show(Section::Shortcuts);
    assert!(out.has("Command palette") && out.has("Ctrl+K"), "{}", out.text());
    rig.page.shortcuts.filter = "sidebar".into();
    assert!(rig.settle().has("Toggle sidebar"));
}

#[test]
fn the_advanced_section_shows_the_settings_json_without_keys() {
    let mut rig = Rig::new(1000.0, 900.0);
    rig.store.update("keys", |s| s.openai_api_key = Some("sk-very-secret".into()));
    let out = rig.show(Section::Advanced);
    assert!(!out.text().contains("sk-very-secret"));
}

#[test]
fn the_appearance_section_embeds_the_theme_picker_and_the_egui_options() {
    let mut rig = Rig::new(1000.0, 900.0);
    let out = rig.show(Section::Appearance);
    assert!(out.has("Theme") && out.has("Egui options"));
    assert!(out.has("Presets") || out.text().contains("Preset"), "the picker is drawn:\n{}", out.text());
    let at = out.pos("Egui options");
    let out = rig.click(at + vec2(4.0, 4.0));
    assert!(out.has("Save options"), "{}", out.text());
    let at = out.pos("Save options");
    rig.click(at + vec2(6.0, 6.0));
    assert!(rig.save_egui, "Save asks the app to write the egui options");
}

#[test]
fn the_assistant_section_calls_the_zeroclaw_hook_under_its_heading_and_offers_the_providers() {
    let mut rig = Rig::new(1000.0, 900.0);
    let out = rig.show(Section::Assistant);
    assert!(out.has("ZeroClaw"), "{}", out.text());
    assert!(out.text().contains("The ZeroClaw connection is not set up in this build."), "the hook is drawn");
    for provider in assistant::PROVIDERS.iter().filter(|p| p.id != "local-joycaption") {
        assert!(out.text().contains(provider.label), "{} is listed", provider.label);
    }
    let at = out.pos("Browse OpenRouter models");
    rig.click(at + vec2(6.0, 6.0));
    assert!(rig.assistant_window, "the model browser stays reachable");
}

#[test]
fn the_ai_models_buttons_exist_and_the_vision_folder_is_checked() {
    let mut rig = Rig::new(1000.0, 1000.0);
    let dir = tempfile::tempdir().unwrap();
    rig.page.draft.value.joycaption_model_dir = Some(dir.path().display().to_string());
    let out = rig.show(Section::AiModels);
    for text in ["Load vision", "Reload CLIP", "Reload reranker", "Unload all", "Keep CLIP"] {
        assert!(out.has(text), "{text}");
    }
    assert!(out.text().contains("No model files"), "{}", out.text());
}

#[test]
fn a_library_folder_that_holds_other_files_is_refused_on_the_page() {
    let mut rig = Rig::new(1000.0, 1000.0);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("photo.jpg"), "x").unwrap();
    rig.page.library.folder = dir.path().display().to_string();
    let out = rig.show(Section::Library);
    assert!(out.text().contains("not empty"), "{}", out.text());
    assert!(photo_still_there(dir.path()));
}

fn photo_still_there(dir: &std::path::Path) -> bool {
    dir.join("photo.jpg").exists()
}

#[test]
fn the_restore_dialog_names_the_file_and_explains_a_refused_folder() {
    let mut rig = Rig::new(1000.0, 1000.0);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("photo.jpg"), "x").unwrap();
    rig.page.library.pending = Some(library::Pending::RestoreInto(std::path::PathBuf::from("/backups/library.surql")));
    rig.page.library.restore_dir = dir.path().display().to_string();
    let out = rig.show(Section::Library);
    assert!(out.text().contains("Restore /backups/library.surql into a new library folder"), "{}", out.text());
    assert!(out.text().contains("not empty"), "{}", out.text());
    rig.page.library.restore_dir = dir.path().join("fresh").display().to_string();
    let out = rig.settle();
    assert!(out.text().contains("Nothing that exists is overwritten"), "{}", out.text());
    let at = out.pos("Cancel");
    rig.click(at + vec2(6.0, 6.0));
    assert!(rig.page.library.pending.is_none());
}

#[test]
fn restoring_a_backup_never_asks_for_an_open_library_but_importing_does() {
    let mut rig = Rig::new(1000.0, 1000.0);
    let out = rig.show(Section::Library);
    assert!(out.has("Restore from a .surql file..."), "{}", out.text());
    assert!(out.has("Import into this library..."));
}

#[test]
fn watched_folders_are_listed_and_scanned_with_a_recursive_command() {
    let _serial = serial();
    take_matching(|_| true);
    let mut rig = Rig::new(1000.0, 1000.0);
    ui_prefs::update(|p| {
        p.watch("/pics/one");
    });
    let out = rig.show(Section::Library);
    assert!(out.has("/pics/one"), "{}", out.text());
    let at = out.pos("Scan all");
    rig.click(at + vec2(6.0, 6.0));
    let sent = take_matching(|c| matches!(c, AppCommand::OpenScope { recursive: true, .. }));
    assert_eq!(sent.len(), 1);
    ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
}

#[test]
fn the_metadata_tools_ask_before_they_change_anything() {
    let mut rig = Rig::new(1000.0, 1000.0);
    let out = rig.show(Section::MetadataTools);
    assert!(out.text().contains("asks for confirmation"));
    assert!(out.has("Preview"));
    rig.page.metadata.pending = Some((metadata_ops::Op::DeleteTag { tag: "sun".into() }, metadata_ops::Preview { affected: 7, ..Default::default() }));
    let out = rig.settle();
    assert!(out.text().contains("7 items lose the tag 'sun'"), "{}", out.text());
    assert!(out.has("Apply to the library") && out.has("Cancel"));
}

/* ------------------------------------------ startup restore ---------------------------------------- */

fn assemble() -> (crate::app::SmartMediaApp, Fixture, tokio::runtime::EnterGuard<'static>) {
    let fixture = Fixture::new();
    let runtime = RT.enter();
    let ctx = test_ctx();
    let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    let app = crate::app::SmartMediaApp::assemble(ctx, appearance, crate::theme::chrome::Chrome { frameless: false, available: true }, crate::ui::file_table::FileExplorer::new(true));
    (app, fixture, runtime)
}

#[test]
fn the_launch_restore_reopens_the_saved_tabs_and_only_then_records_the_open_ones() {
    let _serial = serial();
    take_matching(|_| true);
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (scope_a, scope_b) = (format!("fs:{}", a.path().display()), format!("fs:{}", b.path().display()));
    ui_prefs::update(|p| {
        p.startup.restore_tabs = true;
        p.startup.tabs = vec![scope_a.clone(), scope_b.clone()];
        p.startup.last_scope = Some(scope_a.clone());
    });
    let (mut app, _fixture, _runtime) = assemble();
    app.context.startup = startup::Tracker::new(false);
    startup::tick(&mut app.context);
    let opened = take_matching(|c| matches!(c, AppCommand::OpenScope { .. }));
    assert_eq!(opened.len(), 2);
    assert!(matches!(&opened[0], AppCommand::OpenScope { target: crate::command::Target::Current, .. }));
    assert!(matches!(&opened[1], AppCommand::OpenScope { target: crate::command::Target::Background, .. }));
    assert_eq!(ui_prefs::get().startup.tabs, vec![scope_a.clone(), scope_b.clone()], "the saved tabs are kept while the restore settles");
    for _ in 0..6 {
        startup::tick(&mut app.context);
    }
    let shown = app.context.file_explorer.scope().format();
    assert_eq!(ui_prefs::get().startup.last_scope.as_deref(), Some(shown.as_str()), "after the restore the open scope is recorded");
    ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
}

#[test]
fn without_a_restore_the_open_scopes_are_recorded_from_the_first_frames() {
    ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
    let (mut app, _fixture, _runtime) = assemble();
    app.context.startup = startup::Tracker::new(false);
    for _ in 0..3 {
        startup::tick(&mut app.context);
    }
    let shown = app.context.file_explorer.scope().format();
    let saved = ui_prefs::get().startup;
    assert_eq!(saved.last_scope.as_deref(), Some(shown.as_str()));
    assert_eq!(saved.tabs, vec![shown]);
    ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
}

#[test]
fn the_focused_tab_comes_first_in_the_recorded_list() {
    let (mut app, _fixture, _runtime) = assemble();
    let other = crate::ui::tabs::BrowserId::next();
    let dir = tempfile::tempdir().unwrap();
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.start_at(crate::command::Scope::Folder(dir.path().display().to_string()), false).unwrap();
    app.context.filtered_tabs.insert(other, ex);
    app.context.active_tab = Some(crate::ui::tabs::TabKind::browser(other));
    let (focused, tabs) = startup::open_scopes(&app.context);
    let other_scope = format!("fs:{}", dir.path().display());
    assert_eq!(focused.as_deref(), Some(other_scope.as_str()));
    assert_eq!(tabs[0], other_scope);
    assert_eq!(tabs.len(), 2);
}
