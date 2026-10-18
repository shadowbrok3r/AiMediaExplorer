//! Startup theme resolution and persistence in eframe storage.

use std::path::{Path, PathBuf};

use eframe::egui::{Context, Id};

use super::{
    presets::styles_visually_equal,
    saved::{SavedTheme, theme_from_json, theme_to_json},
    theme_chrome::mastertech_theme,
};

/// Storage key holding the user's saved theme JSON.
pub const STORAGE_KEY: &str = "smart_media.theme.v1";

/// Where the theme applied at startup came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThemeSource {
    /// The shipped Mastertech file.
    Embedded,
    /// The theme saved by an earlier run.
    Stored,
    /// A file named by `--theme`; never written back to storage.
    File(PathBuf),
}

impl ThemeSource {
    pub fn label(&self) -> String {
        match self {
            Self::Embedded => "embedded default".to_owned(),
            Self::Stored => "saved by an earlier run".to_owned(),
            Self::File(path) => format!("file {}", path.display()),
        }
    }
}

fn source_id() -> Id {
    Id::unique("smart_media.theme.source")
}

/// Records where the startup theme came from.
pub fn set_source(ctx: &Context, source: ThemeSource) {
    ctx.data_mut(|d| d.insert_temp(source_id(), source));
}

/// Where the startup theme came from; embedded when none was recorded.
pub fn source(ctx: &Context) -> ThemeSource {
    ctx.data(|d| d.get_temp::<ThemeSource>(source_id()))
        .unwrap_or(ThemeSource::Embedded)
}

/// Reads a theme file in either the wrapper or bare `egui::Style` shape.
pub fn load_file(path: &Path) -> anyhow::Result<SavedTheme> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("cannot read theme file {}: {e}", path.display()))?;
    theme_from_json(&bytes)
}

/// The theme saved in `storage`, if any; a stored value that cannot be decoded is logged and ignored.
pub fn load_stored(storage: Option<&dyn eframe::Storage>) -> Option<SavedTheme> {
    let json = storage?
        .get_string(STORAGE_KEY)
        .filter(|s| !s.trim().is_empty())?;
    match theme_from_json(json.as_bytes()) {
        Ok(theme) => Some(theme),
        Err(e) => {
            log::warn!("stored theme could not be read ({e}); using the embedded default");
            None
        }
    }
}

/// The theme to apply at startup: the `--theme` file, else the stored theme, else the embedded default.
pub fn resolve(
    storage: Option<&dyn eframe::Storage>,
    file: Option<&Path>,
) -> (SavedTheme, ThemeSource) {
    if let Some(path) = file {
        match load_file(path) {
            Ok(theme) => return (theme, ThemeSource::File(path.to_path_buf())),
            Err(e) => log::warn!(
                "--theme {}: {e}; using the saved or embedded theme",
                path.display()
            ),
        }
    }
    match load_stored(storage) {
        Some(theme) => (theme, ThemeSource::Stored),
        None => (mastertech_theme(), ThemeSource::Embedded),
    }
}

/// Whether `theme` equals the embedded default in every part.
pub fn is_embedded_default(theme: &SavedTheme) -> bool {
    let embedded = mastertech_theme();
    theme.density == embedded.density
        && theme.text_scale == embedded.text_scale
        && theme.glass == embedded.glass
        && styles_visually_equal(&theme.style, &embedded.style)
}

/// Writes the live theme to `storage`, or removes the entry when it equals the embedded default.
pub fn save(ctx: &Context, storage: &mut dyn eframe::Storage) {
    if matches!(source(ctx), ThemeSource::File(_)) {
        return;
    }
    let current = SavedTheme::current(ctx);
    if is_embedded_default(&current) {
        storage.remove_string(STORAGE_KEY);
        return;
    }
    match theme_to_json(&current) {
        Ok(json) => storage.set_string(STORAGE_KEY, String::from_utf8_lossy(&json).into_owned()),
        Err(e) => log::warn!("theme was not saved: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        metrics::Density,
        presets::{PresetStyles, apply_preset, apply_saved_theme, set_density},
    };
    use super::*;
    use eframe::Storage as _;
    use std::collections::HashMap;

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

    #[test]
    fn nothing_stored_resolves_to_the_embedded_default() {
        let storage = MemStorage::default();
        let (theme, source) = resolve(Some(&storage), None);
        assert_eq!(source, ThemeSource::Embedded);
        assert!(is_embedded_default(&theme));
        let (_, source) = resolve(None, None);
        assert_eq!(source, ThemeSource::Embedded);
    }

    #[test]
    fn an_untouched_theme_clears_the_entry_instead_of_pinning_the_default() {
        let ctx = Context::default();
        apply_saved_theme(&ctx, mastertech_theme());
        let mut storage = MemStorage::default();
        storage.set_string(STORAGE_KEY, "stale".to_owned());
        save(&ctx, &mut storage);
        assert_eq!(storage.get_string(STORAGE_KEY), None);
        assert_eq!(resolve(Some(&storage), None).1, ThemeSource::Embedded);
    }

    #[test]
    fn a_changed_theme_round_trips_through_storage() {
        let ctx = Context::default();
        apply_saved_theme(&ctx, mastertech_theme());
        apply_preset(&ctx, PresetStyles::MtechNoirGlass);
        set_density(&ctx, Density::Comfortable);
        let mut storage = MemStorage::default();
        save(&ctx, &mut storage);

        let (theme, source) = resolve(Some(&storage), None);
        assert_eq!(source, ThemeSource::Stored);
        assert_eq!(theme.density, Density::Comfortable);

        let fresh = Context::default();
        apply_saved_theme(&fresh, theme);
        assert_eq!(
            super::super::presets::installed_preset(&fresh),
            PresetStyles::MtechNoirGlass
        );
    }

    #[test]
    fn a_corrupt_entry_falls_back_to_the_embedded_default() {
        let mut storage = MemStorage::default();
        storage.set_string(STORAGE_KEY, "{ definitely not json".to_owned());
        let (theme, source) = resolve(Some(&storage), None);
        assert_eq!(source, ThemeSource::Embedded);
        assert!(is_embedded_default(&theme));
    }

    #[test]
    fn a_theme_file_wins_and_is_never_written_back() {
        let dir = std::env::temp_dir().join(format!("smart_media_theme_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("noir.json");
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechNoir);
        std::fs::write(&path, theme_to_json(&SavedTheme::current(&ctx)).unwrap()).unwrap();

        let (theme, source) = resolve(None, Some(&path));
        assert_eq!(source, ThemeSource::File(path.clone()));
        let fresh = Context::default();
        apply_saved_theme(&fresh, theme);
        set_source(&fresh, source);
        let mut storage = MemStorage::default();
        save(&fresh, &mut storage);
        assert!(storage.get_string(STORAGE_KEY).is_none());

        let missing = dir.join("missing.json");
        assert_eq!(resolve(None, Some(&missing)).1, ThemeSource::Embedded);
        std::fs::remove_dir_all(&dir).ok();
    }
}
