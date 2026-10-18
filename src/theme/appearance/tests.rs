use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::Storage as _;
use egui_theme_kit::saved::theme_to_json;
use egui_theme_kit::{KitConfig, init_context};

use super::*;
use crate::database::ui_themes::tests::{Fixture, preset_theme};
use crate::theme::font_config;

#[derive(Default)]
struct MemStorage(HashMap<String, String>);

impl eframe::Storage for MemStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned()
    }

    fn set_string(&mut self, key: &str, value: String) {
        self.0.insert(key.to_owned(), value);
    }

    fn remove_string(&mut self, key: &str) {
        self.0.remove(key);
    }

    fn flush(&mut self) {}
}

fn started(storage: Option<&dyn eframe::Storage>) -> Context {
    let ctx = Context::default();
    init_context(&ctx, storage, KitConfig::new().parse_cli(false).fonts(font_config()));
    ctx
}

fn apply(ctx: &Context, id: &str) {
    presets::apply_saved_theme(ctx, preset_theme(id));
}

fn is(ctx: &Context, id: &str) -> bool {
    presets::themes_equal(&SavedTheme::current(ctx), &preset_theme(id))
}

fn connect(appearance: &Appearance, fixture: &Fixture) {
    fixture.ready.store(true, Ordering::SeqCst);
    appearance.store.reload();
    appearance.store.flush();
}

fn stored_active(fixture: &Fixture) -> Option<SavedTheme> {
    fixture.loaded().load_active()
}

#[test]
fn the_stored_theme_replaces_a_different_cached_one_when_the_database_loads() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("frost"));
    writer.flush();

    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    assert!(is(&ctx, "graphite"));
    appearance.update(&ctx);
    assert!(is(&ctx, "graphite"), "nothing is applied before the database loads");

    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(is(&ctx, "frost"));
    assert_eq!(persist::source(&ctx), ThemeSource::Stored);
    assert!(!appearance.autosave.is_pending(), "applying the stored theme is not a change to save");
}

#[test]
fn a_matching_cached_theme_is_left_alone() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("noir"));
    writer.flush();

    let ctx = started(None);
    apply(&ctx, "noir");
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    let style = ctx.global_style();
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(std::sync::Arc::ptr_eq(&style, &ctx.global_style()), "no reinstall when the theme already matches");
}

#[test]
fn an_empty_database_is_seeded_with_the_cached_custom_theme() {
    let fixture = Fixture::new();
    let ctx = started(None);
    apply(&ctx, "noir");
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    appearance.store.flush();
    assert!(is(&ctx, "noir"));
    assert!(presets::themes_equal(&stored_active(&fixture).expect("seeded"), &preset_theme("noir")));
}

#[test]
fn an_empty_database_stays_empty_for_the_default_theme() {
    let fixture = Fixture::new();
    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    appearance.store.flush();
    assert!(fixture.raw("SELECT * FROM ui_themes").is_empty());
}

#[test]
fn a_change_made_before_the_database_connected_beats_the_stored_theme() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("noir"));
    writer.flush();
    fixture.ready.store(false, Ordering::SeqCst);

    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    apply(&ctx, "frost");
    appearance.update(&ctx);
    assert!(appearance.autosave.is_pending());
    appearance.autosave.flush(&ctx, &*appearance.store);
    appearance.store.flush();
    assert_eq!(appearance.store.queued_ops(), 1);

    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(is(&ctx, "frost"), "the choice made while offline stays applied");
    assert!(presets::themes_equal(&stored_active(&fixture).expect("written"), &preset_theme("frost")));
}

#[test]
fn a_change_still_waiting_for_its_debounce_is_not_overwritten_by_the_load() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("noir"));
    writer.flush();

    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    apply(&ctx, "frost");
    appearance.update(&ctx);
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(is(&ctx, "frost"));

    std::thread::sleep(PERSIST_DEBOUNCE + Duration::from_millis(100));
    appearance.update(&ctx);
    appearance.store.flush();
    assert!(presets::themes_equal(&stored_active(&fixture).expect("written"), &preset_theme("frost")));
}

#[test]
fn a_pinned_startup_theme_wins_over_the_active_one() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("frost"));
    writer.set_startup(StartupTheme::Preset("noir".to_owned()));
    writer.flush();

    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(is(&ctx, "noir"));
}

#[test]
fn a_theme_file_is_neither_replaced_nor_saved() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("frost"));
    writer.flush();

    let ctx = started(None);
    apply(&ctx, "noir");
    persist::set_source(&ctx, ThemeSource::File("theme.json".into()));
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    connect(&appearance, &fixture);
    appearance.update(&ctx);
    assert!(is(&ctx, "noir"));
    appearance.autosave.flush(&ctx, &*appearance.store);
    appearance.store.flush();
    assert!(presets::themes_equal(&stored_active(&fixture).unwrap(), &preset_theme("frost")));
}

#[test]
fn a_theme_under_the_old_storage_key_is_applied_moved_to_the_database_and_dropped_from_storage() {
    let mut storage = MemStorage::default();
    let legacy = preset_theme("noir-glass");
    storage.set_string(LEGACY_STORAGE_KEY, String::from_utf8(theme_to_json(&legacy).unwrap()).unwrap());

    let fixture = Fixture::new();
    let ctx = started(Some(&storage));
    assert!(is(&ctx, "graphite"), "the kit reads only its own key");
    let mut appearance = Appearance::attach(&ctx, Some(&storage), Arc::new(fixture.store()));
    assert!(is(&ctx, "noir-glass"));
    assert_eq!(persist::source(&ctx), ThemeSource::Stored);

    connect(&appearance, &fixture);
    appearance.update(&ctx);
    appearance.store.flush();
    assert!(presets::themes_equal(&stored_active(&fixture).expect("migrated"), &legacy));

    appearance.save(&ctx, &mut storage);
    assert!(storage.get_string(LEGACY_STORAGE_KEY).is_none());
    assert!(storage.get_string(persist::STORAGE_KEY).is_some());
    let restarted = started(Some(&storage));
    assert!(is(&restarted, "noir-glass"));
}

#[test]
fn the_old_storage_key_never_overrides_the_current_one() {
    let mut storage = MemStorage::default();
    storage.set_string(LEGACY_STORAGE_KEY, String::from_utf8(theme_to_json(&preset_theme("noir")).unwrap()).unwrap());
    storage.set_string(persist::STORAGE_KEY, String::from_utf8(theme_to_json(&preset_theme("frost")).unwrap()).unwrap());
    let ctx = started(Some(&storage));
    let mut appearance = Appearance::attach(&ctx, Some(&storage), Arc::new(Fixture::unready().store()));
    assert!(is(&ctx, "frost"));
    appearance.save(&ctx, &mut storage);
    assert!(storage.get_string(LEGACY_STORAGE_KEY).is_none());
}

#[test]
fn a_theme_file_run_keeps_the_old_storage_key() {
    let mut storage = MemStorage::default();
    storage.set_string(LEGACY_STORAGE_KEY, String::from_utf8(theme_to_json(&preset_theme("noir")).unwrap()).unwrap());
    let ctx = started(Some(&storage));
    persist::set_source(&ctx, ThemeSource::File("theme.json".into()));
    let mut appearance = Appearance::attach(&ctx, Some(&storage), Arc::new(Fixture::unready().store()));
    appearance.save(&ctx, &mut storage);
    assert!(storage.get_string(LEGACY_STORAGE_KEY).is_some());
}

#[test]
fn store_errors_are_reported_once_while_the_window_is_closed() {
    let fixture = Fixture::unready();
    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    appearance.store.reload();
    appearance.store.flush();
    assert!(appearance.store.last_error().is_some());
    appearance.update(&ctx);
    assert!(appearance.store.last_error().is_none());
}

#[test]
fn the_window_applies_presets_and_saves_themes_to_the_database() {
    let fixture = Fixture::new();
    let ctx = started(None);
    let mut appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
    connect(&appearance, &fixture);
    appearance.update(&ctx);

    appearance.open();
    assert!(appearance.is_open());
    for _ in 0..2 {
        ctx.run_ui(Default::default(), |ui| appearance.show(ui.ctx())).drop_without_applying_deltas();
    }

    assert!(appearance.picker.apply_preset(&ctx, "frost"));
    assert!(appearance.picker.save_as(&ctx, &*appearance.store, "Mine", false));
    appearance.picker.set_density(&ctx, egui_theme_kit::Density::Comfortable);
    appearance.update(&ctx);
    appearance.autosave.flush(&ctx, &*appearance.store);
    appearance.store.flush();

    let reopened = fixture.loaded();
    assert_eq!(reopened.user_themes().len(), 1);
    assert_eq!(reopened.user_themes()[0].name, "Mine");
    let active = reopened.load_active().expect("active theme");
    assert_eq!(active.density, egui_theme_kit::Density::Comfortable);

    assert!(appearance.picker.rename(&*appearance.store, "Mine", "Renamed"));
    appearance.picker.delete(&*appearance.store, "Renamed");
    appearance.store.flush();
    assert!(fixture.loaded().user_themes().is_empty());
}

#[test]
fn the_picker_window_draws_with_every_action_glyph() {
    for name in egui_theme_kit::picker::ICON_NAMES {
        let glyph = action_glyph(name).unwrap_or_else(|| panic!("{name} has no glyph"));
        assert_eq!(glyph.chars().count(), 1, "{name}");
    }
    assert_eq!(action_glyph("unknown"), None);
}
