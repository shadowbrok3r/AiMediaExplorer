//! Small widgets shared by the inspector tabs.

use egui::{Button, Color32, CornerRadius, RichText, Ui};

use crate::icons::p;

/// What the user did with a chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChipEvent {
    None,
    Removed,
}

/// Stable colour of a tag.
pub fn tag_tint(tag: &str) -> Color32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tag.to_lowercase().hash(&mut h);
    let v = h.finish();
    Color32::from_rgb((v & 0xFF) as u8 / 2 + 64, ((v >> 8) & 0xFF) as u8 / 2 + 64, ((v >> 16) & 0xFF) as u8 / 2 + 64)
}

/// A tag chip with an optional remove button; `suffix` adds text such as a count.
pub fn chip(ui: &mut Ui, tag: &str, suffix: Option<&str>, removable: bool, dim: bool) -> ChipEvent {
    let mut tint = tag_tint(tag);
    if dim {
        tint = tint.gamma_multiply(0.45);
    }
    let text = match suffix {
        Some(s) => format!("{tag} {s}"),
        None => tag.to_string(),
    };
    ui.spacing_mut().item_spacing.x = 2.0;
    ui.add(Button::new(text).fill(tint).corner_radius(CornerRadius::same(6)).sense(egui::Sense::hover()));
    let mut event = ChipEvent::None;
    if removable && ui.add(Button::new(RichText::new(p::X).size(11.0)).small().frame(false)).on_hover_text(format!("Remove \"{tag}\"")).clicked() {
        event = ChipEvent::Removed;
    }
    ui.spacing_mut().item_spacing.x = 6.0;
    event
}

/// A small coloured label.
pub fn badge(ui: &mut Ui, text: &str, color: Color32) {
    ui.add(Button::new(RichText::new(text).small().color(color)).frame(true).sense(egui::Sense::hover()).small());
}

/// A label and its value on one row of a grid.
pub fn kv(ui: &mut Ui, label: &str, value: impl Into<egui::WidgetText>) {
    ui.label(RichText::new(label).weak());
    ui.add(egui::Label::new(value).wrap());
    ui.end_row();
}

/// "1 file" or "3 files".
pub fn count_noun(n: usize, one: &str, many: &str) -> String {
    if n == 1 { format!("1 {one}") } else { format!("{n} {many}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_colours_ignore_case_and_differ_between_tags() {
        assert_eq!(tag_tint("Cat"), tag_tint("cat"));
        assert_ne!(tag_tint("cat"), tag_tint("dog"));
    }

    #[test]
    fn counts_use_the_singular_for_one() {
        assert_eq!(count_noun(1, "file", "files"), "1 file");
        assert_eq!(count_noun(0, "file", "files"), "0 files");
        assert_eq!(count_noun(12, "file", "files"), "12 files");
    }
}
