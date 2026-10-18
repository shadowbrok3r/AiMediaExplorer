//! Devtools window showing the live settings store: JSON with credentials redacted, patch history and controls.

use std::sync::Arc;

use eframe::egui::{CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};
use egui_toast::ToastKind;

use crate::icons::{self, svg};
use crate::settings::{LoadState, SettingsStore};

fn open_id() -> Id {
    Id::new("ame_devtools_settings_store_open")
}

fn json_cache_id() -> Id {
    Id::new("ame_devtools_settings_store_json")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Settings store")
        .open(&mut open)
        .default_width(560.0)
        .default_height(520.0)
        .show(ctx, |ui| body(ui, crate::settings::store()));
    set_open(ctx, open);
}

fn state_text(state: &LoadState) -> String {
    match state {
        LoadState::Pending => "pending (database not read yet)".into(),
        LoadState::Loaded => "loaded".into(),
        LoadState::Failed(e) => format!("FAILED: {e}"),
    }
}

/// Window contents for `store`.
pub fn body(ui: &mut Ui, store: &Arc<SettingsStore>) {
    let state = store.load_state();
    let dirty = store.dirty_fields();
    let save = store.save_status();
    Grid::new("settings_store_info").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label("Generation");
        ui.monospace(store.generation().to_string());
        ui.end_row();
        ui.label("Load state");
        let text = RichText::new(state_text(&state));
        ui.label(if matches!(state, LoadState::Failed(_)) { text.color(crate::theme::error(ui)) } else { text });
        ui.end_row();
        ui.label("Unsaved fields");
        ui.monospace(if dirty.is_empty() { "none".to_string() } else { dirty.join(", ") });
        ui.end_row();
        ui.label("Last save");
        let last = match (&save.last_ok_at, &save.last_error) {
            (_, Some(e)) => format!("failed: {e}"),
            (Some(at), None) => format!("ok at {at} ({} batches)", save.batches_saved),
            (None, None) => "never".into(),
        };
        ui.label(last);
        ui.end_row();
    });

    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        if icons::button(ui, svg::ARROWS_CLOCKWISE, "Reload from DB").on_hover_text("Read the stored record again and replace the live settings").clicked() {
            let store = Arc::clone(store);
            tokio::spawn(async move {
                match store.reload().await {
                    Ok(()) => crate::ui::dialogs::toast(ToastKind::Success, "Settings reloaded from the database"),
                    Err(e) => crate::ui::dialogs::toast(ToastKind::Error, format!("Settings reload failed: {e:#}")),
                }
            });
        }
        if icons::button(ui, svg::WARNING, "Simulate load failure").on_hover_text("Show the load-failure banner without touching the database").clicked() {
            store.set_load_failed("devtools:simulate", "Simulated failure: invalid type: null, expected a boolean".into());
        }
        if icons::button(ui, svg::FLOPPY_DISK, "Force save").on_hover_text("Write every field now as a merge into the stored record").clicked() {
            let store = Arc::clone(store);
            tokio::spawn(async move {
                match store.force_save().await {
                    Ok(n) => crate::ui::dialogs::toast(ToastKind::Success, format!("Saved {n} settings fields")),
                    Err(e) => crate::ui::dialogs::toast(ToastKind::Error, format!("Force save failed: {e:#}")),
                }
            });
        }
    });

    ui.separator();
    CollapsingHeader::new("Recent patches").default_open(true).show(ui, |ui| {
        let patches = store.patches();
        if patches.is_empty() {
            ui.weak("No patches yet.");
            return;
        }
        ScrollArea::vertical().id_salt("settings_store_patches").max_height(180.0).auto_shrink([false, true]).show(ui, |ui| {
            for patch in patches.iter().rev() {
                ui.label(RichText::new(format!("#{} {}  {}", patch.generation, patch.at, patch.who)).strong());
                for change in &patch.changes {
                    ui.monospace(format!("  {}: {} -> {}", change.field, change.old, change.new));
                }
            }
        });
    });

    CollapsingHeader::new("Settings JSON (API keys redacted)").default_open(true).show(ui, |ui| {
        let generation = store.generation();
        let cached = ui.ctx().data(|d| d.get_temp::<(u64, String)>(json_cache_id()));
        let text = match cached {
            Some((g, text)) if g == generation => text,
            _ => {
                let text = serde_json::to_string_pretty(&store.json_redacted()).unwrap_or_default();
                ui.ctx().data_mut(|d| d.insert_temp(json_cache_id(), (generation, text.clone())));
                text
            }
        };
        ScrollArea::vertical().id_salt("settings_store_json").max_height(240.0).auto_shrink([false, true]).show(ui, |ui| {
            let mut view = text.as_str();
            ui.add(TextEdit::multiline(&mut view).code_editor().desired_width(f32::INFINITY));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Backend, Loaded};
    use crate::UiSettings;
    use async_trait::async_trait;
    use eframe::egui::{CentralPanel, RawInput};

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

    fn texts_of(output: &eframe::egui::FullOutput) -> String {
        let mut all = String::new();
        for clipped in &output.shapes {
            collect_text(&clipped.shape, &mut all);
        }
        all
    }

    fn collect_text(shape: &eframe::egui::Shape, out: &mut String) {
        match shape {
            eframe::egui::Shape::Text(t) => {
                out.push_str(t.galley.text());
                out.push('\n');
            }
            eframe::egui::Shape::Vec(v) => v.iter().for_each(|s| collect_text(s, out)),
            _ => {}
        }
    }

    #[tokio::test]
    async fn the_window_renders_patches_and_never_prints_credentials() {
        let ctx = Context::default();
        let store = SettingsStore::new(Arc::new(Nop));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        store.update("keys:test", |s| {
            s.openai_api_key = Some("sk-live-1234567890".into());
            s.filter_skip_icons = true;
        });
        let mut output = None;
        for _ in 0..2 {
            let input = RawInput {
                screen_rect: Some(eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(700.0, 800.0))),
                ..Default::default()
            };
            let mut frame = ctx.run_ui(input, |ui| {
                CentralPanel::default().show(ui, |ui| body(ui, &store));
            });
            frame.textures_delta.clear();
            output = Some(frame);
        }
        let text = texts_of(&output.unwrap());
        assert!(text.contains("keys:test"), "who is listed:\n{text}");
        assert!(text.contains("filter_skip_icons"), "changed field is listed:\n{text}");
        assert!(text.contains("<redacted"), "keys are masked:\n{text}");
        assert!(!text.contains("sk-live-1234567890"), "secret leaked:\n{text}");
    }

    #[tokio::test]
    async fn simulate_failure_state_is_reported_in_the_window() {
        let ctx = Context::default();
        let store = SettingsStore::new(Arc::new(Nop));
        store.set_load_failed("devtools:simulate", "boom".into());
        let input = RawInput {
            screen_rect: Some(eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(700.0, 800.0))),
            ..Default::default()
        };
        let mut output = None;
        for _ in 0..2 {
            let mut frame = ctx.run_ui(input.clone(), |ui| {
                CentralPanel::default().show(ui, |ui| body(ui, &store));
            });
            frame.textures_delta.clear();
            output = Some(frame);
        }
        assert!(texts_of(&output.unwrap()).contains("FAILED: boom"));
    }
}
