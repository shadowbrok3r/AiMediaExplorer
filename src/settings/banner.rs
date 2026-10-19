//! Banners shown while the stored settings cannot be read and while the database cannot be opened.

use std::sync::Arc;

use eframe::egui::{Panel, RichText, Sides, Ui};

use super::SettingsStore;
use crate::command::{AppCommand, Section, Target, Tool};
use crate::icons::{self, svg};
use crate::tasks::{AttentionItem, AttentionKind};

/// Draws the banners at the top of `ui`: the last settings load failed, the database did not open.
pub fn show(ui: &mut Ui) {
    show_for(ui, super::store());
    show_database(ui);
}

/// The failed database start the needs-attention list holds, if any.
pub fn database_failure() -> Option<AttentionItem> {
    if crate::database::is_connected() {
        return None;
    }
    crate::tasks::with(|r| r.attention()).into_iter().find(|i| i.kind == AttentionKind::Database)
}

/// Draws the database banner when the database did not open.
pub fn show_database(ui: &mut Ui) {
    show_database_for(ui, database_failure().as_ref());
}

/// Draws the database banner for `failure`, with Reconnect, Change location and Open logs.
pub fn show_database_for(ui: &mut Ui, failure: Option<&AttentionItem>) {
    let Some(failure) = failure else { return };
    Panel::top("database_connect_failure").show(ui, |ui| {
        Sides::new().shrink_left().truncate().show(
            ui,
            |ui| {
                let title = RichText::new(failure.title.as_str()).color(crate::theme::error(ui)).strong();
                icons::label(ui, svg::WARNING, title);
                ui.label(failure.detail.as_str());
            },
            |ui| {
                if ui.button("Open logs").clicked() {
                    crate::command::send(AppCommand::OpenTool { tool: Tool::Logs, path: None, target: Target::NewTab });
                }
                if ui.button("Change location").clicked() {
                    crate::command::send(AppCommand::ShowSettings { section: Some(Section::Library) });
                }
                if ui.button("Reconnect").clicked() {
                    crate::command::send(AppCommand::ConnectDatabase);
                }
            },
        );
    });
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

    fn database_frame(ctx: &Context, failure: Option<&AttentionItem>, events: Vec<eframe::egui::Event>) -> Vec<(String, Pos2)> {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 300.0))), events, ..Default::default() };
        let out = ctx.run_ui(input, |ui| show_database_for(ui, failure));
        let found = crate::ui::test_support::texts(&out.shapes);
        out.drop_without_applying_deltas();
        found
    }

    fn themed() -> Context {
        let ctx = Context::default();
        ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
        crate::theme::bootstrap_startup_theme(&ctx);
        ctx
    }

    #[test]
    fn the_database_banner_names_the_failure_and_offers_reconnect_change_location_and_logs() {
        let ctx = themed();
        let item = crate::receive::database_attention("/db/lib", "lock held by another process");
        database_frame(&ctx, Some(&item), Vec::new());
        let found = database_frame(&ctx, Some(&item), Vec::new());
        let has = |text: &str| found.iter().any(|(t, _)| t == text);
        assert!(has("Database not connected"), "{found:?}");
        assert!(found.iter().any(|(t, _)| t.contains("/db/lib: lock held")));
        assert!(has("Reconnect") && has("Change location") && has("Open logs"));
    }

    #[test]
    fn without_a_failure_the_database_banner_draws_nothing() {
        let ctx = themed();
        database_frame(&ctx, None, Vec::new());
        assert!(database_frame(&ctx, None, Vec::new()).is_empty());
    }

    #[test]
    fn the_banner_buttons_send_their_commands() {
        use crate::command::bus::testing::{serial, take_matching};
        let _serial = serial();
        take_matching(|_| true);
        let ctx = themed();
        let item = crate::receive::database_attention("/db/lib", "boom");
        database_frame(&ctx, Some(&item), Vec::new());
        let found = database_frame(&ctx, Some(&item), Vec::new());
        for (label, check) in [
            ("Reconnect", (|c: &AppCommand| matches!(c, AppCommand::ConnectDatabase)) as fn(&AppCommand) -> bool),
            ("Change location", |c: &AppCommand| matches!(c, AppCommand::ShowSettings { section: Some(Section::Library) })),
            ("Open logs", |c: &AppCommand| matches!(c, AppCommand::OpenTool { tool: Tool::Logs, .. })),
        ] {
            let at = found.iter().find(|(t, _)| t == label).map(|(_, p)| *p).unwrap();
            database_frame(&ctx, Some(&item), crate::ui::test_support::press_release(at + vec2(6.0, 6.0)));
            assert_eq!(take_matching(check).len(), 1, "{label}");
        }
    }
}
