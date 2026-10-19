//! Automation section: what runs by itself in the background and the prompt used to describe files.

use eframe::egui::{TextEdit, Ui};

use super::draft::Draft;
use super::sections::name;
use super::widgets::{Focus, caption, group, toggle};

/// The prompt used when none was written.
pub fn default_prompt() -> String {
    crate::UiSettings::default().ai_prompt_template
}

pub fn ui(ui: &mut Ui, focus: &Focus, draft: &mut Draft) {
    let s = &mut draft.value;
    group(ui, "Background work");
    toggle(ui, focus, name::AUTO_INDEX, "New files are described by the vision model in the background", &mut s.auto_indexing);
    toggle(ui, focus, name::AUTO_CLIP, "New files get a CLIP embedding in the background", &mut s.auto_clip_embeddings);
    toggle(ui, focus, name::OVERWRITE_DESC, "Describing a file replaces the description it has", &mut s.overwrite_descriptions);

    group(ui, name::PROMPT);
    ui.horizontal(|ui| {
        caption(ui, focus, name::PROMPT);
        if ui.add_enabled(s.ai_prompt_template != default_prompt(), eframe::egui::Button::new("Reset to default")).clicked() {
            s.ai_prompt_template = default_prompt();
        }
    });
    ui.add(TextEdit::multiline(&mut s.ai_prompt_template).desired_rows(6).desired_width(f32::INFINITY).hint_text("Instructions sent with every file"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_prompt_asks_for_json() {
        assert!(default_prompt().contains("JSON"));
    }
}
