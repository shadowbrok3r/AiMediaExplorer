//! Banner shown while the stored settings cannot be read.

use std::sync::Arc;

use eframe::egui::{Panel, RichText, Sides, Ui};

use super::SettingsStore;
use crate::icons::{self, svg};

/// Draws the banner at the top of `ui` when the last settings load failed.
pub fn show(ui: &mut Ui) {
    show_for(ui, super::store());
}

/// Draws the banner for `store`.
pub fn show_for(ui: &mut Ui, store: &Arc<SettingsStore>) {
    let Some(error) = store.load_failure() else { return };
    if store.banner_dismissed() {
        return;
    }
    Panel::top("settings_load_failure").show(ui, |ui| {
        Sides::new().shrink_left().truncate().show(
            ui,
            |ui| {
                let title = RichText::new("Settings failed to load").color(crate::theme::error(ui)).strong();
                icons::label(ui, svg::WARNING, title);
                ui.label(format!("Running on defaults, changes are not saved. {error}"));
            },
            |ui| {
                if ui.button("Dismiss").clicked() {
                    store.dismiss_banner();
                }
                if ui.button("Details").clicked() {
                    crate::ui::devtools::open_settings_store(ui.ctx());
                }
                if ui.button("Retry").clicked() {
                    let store = Arc::clone(store);
                    tokio::spawn(async move {
                        let _ = store.reload().await;
                    });
                }
            },
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UiSettings;
    use crate::settings::{Backend, Loaded};
    use async_trait::async_trait;
    use eframe::egui::{Context, Pos2, RawInput, Rect, Shape, vec2};

    struct Nop;

    #[async_trait]
    impl Backend for Nop {
        async fn load(&self) -> anyhow::Result<Loaded> {
            Ok(Loaded { settings: UiSettings::default(), existed: false })
        }
        async fn merge(&self, _: &[&'static str], _: &UiSettings) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn banner_text(ctx: &Context, store: &Arc<SettingsStore>) -> String {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 300.0))),
            ..Default::default()
        };
        let mut out = ctx.run_ui(input, |ui| show_for(ui, store));
        out.textures_delta.clear();
        fn walk(shape: &Shape, text: &mut String) {
            match shape {
                Shape::Text(t) => {
                    text.push_str(t.galley.text());
                    text.push('\n');
                }
                Shape::Vec(v) => v.iter().for_each(|s| walk(s, text)),
                _ => {}
            }
        }
        let mut text = String::new();
        out.shapes.iter().for_each(|c| walk(&c.shape, &mut text));
        text
    }

    #[tokio::test]
    async fn the_banner_shows_only_while_loading_failed_and_until_dismissed() {
        let ctx = Context::default();
        ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
        crate::theme::bootstrap_startup_theme(&ctx);
        let store = SettingsStore::new(Arc::new(Nop));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        banner_text(&ctx, &store);
        assert!(!banner_text(&ctx, &store).contains("Settings failed to load"));

        store.set_load_failed("test", "invalid type: null, expected a boolean".into());
        banner_text(&ctx, &store);
        let text = banner_text(&ctx, &store);
        assert!(text.contains("Settings failed to load"), "{text}");
        assert!(text.contains("expected a boolean"), "the error is shown:\n{text}");

        store.dismiss_banner();
        assert!(!banner_text(&ctx, &store).contains("Settings failed to load"));

        store.set_load_failed("test", "again".into());
        assert!(banner_text(&ctx, &store).contains("Settings failed to load"), "a new failure shows the banner again");
    }
}
