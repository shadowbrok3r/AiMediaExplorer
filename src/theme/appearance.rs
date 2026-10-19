//! Appearance settings: the theme picker window over the database store, startup sync and theme persistence.

use std::sync::Arc;
use std::time::Duration;

use eframe::egui::Context;
use egui_theme_kit::debug_panel;
use egui_theme_kit::persist::{self, ThemeSource};
use egui_theme_kit::picker::{self, PickerOptions, PickerState};
use egui_theme_kit::presets;
use egui_theme_kit::saved::SavedTheme;
use egui_theme_kit::store::{AutoSave, MemoryStore, StartupTheme, ThemeStore};

use crate::database::ui_themes::{LoadEvent, SurrealThemeStore, app_store};

/// Eframe storage key earlier releases wrote the theme under.
pub const LEGACY_STORAGE_KEY: &str = "smart_media.theme.v1";

/// Quiet time after the last theme change before it is written to the store.
const PERSIST_DEBOUNCE: Duration = Duration::from_millis(600);
/// Longest wait for queued theme writes when the app exits.
const EXIT_FLUSH: Duration = Duration::from_millis(1500);

/// Installs the kit on the creation context and returns the Appearance state bound to the app database store.
pub fn setup(cc: &eframe::CreationContext<'_>) -> Appearance {
    let store = Arc::clone(app_store());
    egui_theme_kit::init(cc, super::config::kit_config(store.clone()));
    Appearance::attach(&cc.egui_ctx, cc.storage, store)
}

/// The Appearance window, its store and the sync between the live theme, eframe storage and the database.
pub struct Appearance {
    store: Arc<SurrealThemeStore>,
    picker: PickerState,
    autosave: AutoSave,
    legacy_key_present: bool,
}

impl Appearance {
    /// Binds `store` to an initialised context: carries over a theme saved by an earlier release and registers the store.
    pub fn attach(
        ctx: &Context,
        storage: Option<&dyn eframe::Storage>,
        store: Arc<SurrealThemeStore>,
    ) -> Self {
        let legacy_key_present = storage.is_some_and(|s| s.get_string(LEGACY_STORAGE_KEY).is_some());
        migrate_legacy_theme(ctx, storage);
        store.set_repaint(ctx.clone());
        debug_panel::set_store(ctx, store.clone());
        let mut autosave = AutoSave::new(PERSIST_DEBOUNCE);
        autosave.mark_clean(ctx);
        let options = PickerOptions { icon: Some(action_glyph), ..PickerOptions::default() };
        Self { store, picker: PickerState::new(options), autosave, legacy_key_present }
    }

    /// The database-backed theme store.
    pub fn store(&self) -> &Arc<SurrealThemeStore> {
        &self.store
    }

    pub fn open(&mut self) {
        self.picker.open = true;
    }

    pub fn is_open(&self) -> bool {
        self.picker.open
    }

    /// Per-frame work: applies the stored theme once the database loads, saves theme changes and reports store errors.
    pub fn update(&mut self, ctx: &Context) {
        debug_panel::set_store(ctx, self.store.clone());
        if let Some(event) = self.store.take_load_event() {
            self.on_database_loaded(ctx, event);
        }
        self.autosave.update(ctx, &*self.store);
        if !self.picker.open {
            self.report_store_error();
        }
    }

    /// Draws the Appearance window while it is open.
    pub fn show(&mut self, ctx: &Context) {
        picker::window(ctx, &mut self.picker, &*self.store);
    }

    /// Draws the picker into `ui` instead of its window.
    pub fn show_embedded(&mut self, ui: &mut eframe::egui::Ui) {
        picker::show(ui, &mut self.picker, &*self.store);
    }

    /// Writes a pending theme change and waits briefly for the store to drain; call on exit.
    pub fn flush(&mut self, ctx: &Context) {
        self.picker.flush(ctx, &*self.store);
        self.autosave.flush(ctx, &*self.store);
        if !self.store.flush_within(EXIT_FLUSH) {
            log::warn!("[themes] queued theme writes did not finish before exit");
        }
    }

    /// Writes the live theme to eframe storage, the fast cache read before the database opens.
    pub fn save(&mut self, ctx: &Context, storage: &mut dyn eframe::Storage) {
        egui_theme_kit::save(ctx, storage);
        if self.legacy_key_present && !matches!(persist::source(ctx), ThemeSource::File(_)) {
            storage.remove_string(LEGACY_STORAGE_KEY);
            self.legacy_key_present = false;
        }
    }

    fn on_database_loaded(&mut self, ctx: &Context, event: LoadEvent) {
        if matches!(persist::source(ctx), ThemeSource::File(_)) {
            return;
        }
        if event.active_written_after_start || self.autosave.is_pending() {
            return;
        }
        let stored = MemoryStore::with_snapshot(event.snapshot);
        let (theme, source) = persist::resolve_with_store(&stored, None);
        if source == ThemeSource::Stored {
            if !presets::themes_equal(&SavedTheme::current(ctx), &theme) {
                presets::apply_saved_theme(ctx, theme);
                persist::set_source(ctx, ThemeSource::Stored);
            }
        } else {
            let live = SavedTheme::current(ctx);
            if !persist::is_embedded_default(&live) {
                self.store.save_active(&live);
            }
        }
        self.autosave.mark_clean(ctx);
    }

    fn report_store_error(&self) {
        if let Some(error) = self.store.last_error() {
            self.store.clear_error();
            log::warn!("[themes] {error}");
            crate::ui::dialogs::toast(egui_toast::ToastKind::Warning, format!("Themes: {error}"));
        }
    }
}

/// Applies a theme saved under the old storage key when the current key holds none.
fn migrate_legacy_theme(ctx: &Context, storage: Option<&dyn eframe::Storage>) {
    if matches!(persist::source(ctx), ThemeSource::File(_)) {
        return;
    }
    if persist::load_stored(storage).is_some() {
        return;
    }
    if let Some(theme) = persist::load_stored_from(storage, &[LEGACY_STORAGE_KEY]) {
        presets::apply_saved_theme(ctx, theme);
        persist::set_source(ctx, ThemeSource::Stored);
    }
}

/// Phosphor glyph drawn before each picker action label.
fn action_glyph(action: &str) -> Option<&'static str> {
    use egui_phosphor::regular as p;
    Some(match action {
        "apply" => p::CHECK,
        "revert" => p::ARROW_COUNTER_CLOCKWISE,
        "reset" => p::ARROW_CLOCKWISE,
        "save" => p::FLOPPY_DISK,
        "save_as" => p::FLOPPY_DISK_BACK,
        "rename" => p::PENCIL_SIMPLE,
        "duplicate" => p::COPY,
        "delete" => p::TRASH,
        "startup" => p::PUSH_PIN,
        "import" => p::DOWNLOAD_SIMPLE,
        "export" => p::UPLOAD_SIMPLE,
        "copy" => p::CLIPBOARD,
        "search" => p::MAGNIFYING_GLASS,
        "warning" => p::WARNING,
        "glass" => p::DROP,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
