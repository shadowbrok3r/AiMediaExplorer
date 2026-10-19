use eframe::egui::*;

use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::filter::{format_size, parse_date, parse_size, split_tokens};
use crate::ui::file_table::rows::{DateRange, Tri};

const EDITOR_WIDTH: f32 = 340.0;

/// Single-line box that returns its text when Enter is pressed or focus leaves it; Escape drops the edit.
fn commit_text(ui: &mut Ui, id: Id, shown: &str, hint: &str, width: f32) -> Option<String> {
    let mut text: String = ui.data(|d| d.get_temp(id)).unwrap_or_else(|| shown.to_string());
    let resp = ui.add(TextEdit::singleline(&mut text).id(id.with("edit")).hint_text(hint).desired_width(width));
    if resp.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text.clone()));
    }
    if !resp.lost_focus() {
        return None;
    }
    ui.data_mut(|d| d.remove_temp::<String>(id));
    (!ui.input(|i| i.key_pressed(Key::Escape))).then_some(text)
}

/// Box that returns its text when Enter is pressed and keeps the focus for the next entry.
fn add_text(ui: &mut Ui, id: Id, hint: &str, width: f32) -> Option<String> {
    let mut text: String = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    let resp = ui.add(TextEdit::singleline(&mut text).id(id.with("edit")).hint_text(hint).desired_width(width));
    let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    if enter {
        ui.data_mut(|d| d.remove_temp::<String>(id));
        resp.request_focus();
        return (!text.trim().is_empty()).then_some(text);
    }
    ui.data_mut(|d| d.insert_temp(id, text));
    None
}

fn heading(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(text).strong().color(crate::theme::primary(ui)));
}

fn tri_row(ui: &mut Ui, label: &str, value: &mut Tri, yes: &str, no: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.selectable_value(value, Tri::No, no);
            ui.selectable_value(value, Tri::Yes, yes);
            ui.selectable_value(value, Tri::Any, "Any");
        });
    });
}

fn date_edit(ui: &mut Ui, id: Id, bound: &mut Option<String>, hint: &str, err: &mut Option<String>) {
    let shown = bound.clone().unwrap_or_default();
    let Some(text) = commit_text(ui, id, &shown, hint, 92.0) else { return };
    let text = text.trim();
    if text.is_empty() {
        *bound = None;
    } else if parse_date(text).is_some() {
        *bound = Some(text.to_string());
    } else {
        *err = Some(format!("'{text}' is not a date like 2024-01-31"));
    }
}

fn date_row(ui: &mut Ui, id: Id, label: &str, range: &mut DateRange, presets: bool, err: &mut Option<String>) {
    ui.horizontal(|ui| {
        ui.label(label);
        date_edit(ui, id.with("after"), &mut range.after, "from YYYY-MM-DD", err);
        ui.label("to");
        date_edit(ui, id.with("before"), &mut range.before, "to YYYY-MM-DD", err);
    });
    if !presets {
        return;
    }
    ui.horizontal(|ui| {
        let today = chrono::Local::now().date_naive();
        for (name, days) in [("Last 7 days", 7), ("Last 30 days", 30), ("Last year", 365)] {
            if ui.small_button(name).clicked() {
                range.after = Some((today - chrono::Duration::days(days)).format("%Y-%m-%d").to_string());
                range.before = None;
            }
        }
    });
}

fn list_add(ui: &mut Ui, id: Id, label: &str, hint: &str, split: bool, list: &mut Vec<String>) {
    ui.horizontal(|ui| {
        ui.label(label);
        let width = ui.available_width();
        if let Some(text) = add_text(ui, id, hint, width) {
            if split {
                list.extend(split_tokens(&text));
            } else {
                list.push(text.trim().to_string());
            }
        }
    });
}

/// Like [`list_add`], with a removable button per entry.
fn entry_list(ui: &mut Ui, id: Id, label: &str, hint: &str, split: bool, list: &mut Vec<String>) {
    list_add(ui, id, label, hint, split, list);
    let mut drop: Option<usize> = None;
    ui.horizontal_wrapped(|ui| {
        for (i, item) in list.iter().enumerate() {
            if icons::small_button(ui, svg::X, item.as_str()).on_hover_text("Remove").clicked() {
                drop = Some(i);
            }
        }
    });
    if let Some(i) = drop {
        list.remove(i);
    }
}

impl FileExplorer {
    /// Controls for every condition of the tab's filter; changes take effect at once and rows are never removed.
    pub fn filter_editor(&mut self, ui: &mut Ui, salt: &str) {
        ui.set_width(EDITOR_WIDTH);
        let id = Id::new(("ame_filter_editor", self.viewer.scope, salt));
        let before = self.filter_spec();
        let mut spec = before.clone();
        let mut err: Option<String> = None;

        let max_height = (ui.ctx().content_rect().height() - 160.0).max(160.0);
        ScrollArea::vertical().max_height(max_height).auto_shrink([false, true]).show(ui, |ui| {
            heading(ui, "Show");
            ui.horizontal_wrapped(|ui| {
                ui.toggle_value(&mut spec.types.images, "Images");
                ui.toggle_value(&mut spec.types.videos, "Videos");
                ui.toggle_value(&mut spec.types.dirs, "Folders");
                ui.toggle_value(&mut spec.types.archives, "Archives");
                ui.toggle_value(&mut spec.types.other, "Other files");
            });
            ui.checkbox(&mut spec.skip_icons, "Skip likely icons").on_hover_text("Hides .ico files and small images that are icon-sized");

            heading(ui, "Text");
            ui.horizontal(|ui| {
                ui.label("Search");
                ui.add(TextEdit::singleline(&mut spec.query).id(id.with("query")).hint_text("name, type, path, tags, category, description").desired_width(ui.available_width()));
            });
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(TextEdit::singleline(&mut spec.name).id(id.with("name")).hint_text("file name contains").desired_width(ui.available_width()));
            });
            list_add(ui, id.with("term"), "Exclude", "text in the path, Enter to add", false, &mut spec.excluded_terms);
            list_add(ui, id.with("dir"), "Folder", "exclude a folder, Enter to add", false, &mut spec.excluded_dirs);

            heading(ui, "Size and dates");
            ui.horizontal(|ui| {
                ui.label("Size");
                for (side, bound) in [("min", &mut spec.size.min), ("max", &mut spec.size.max)] {
                    let shown = bound.map(format_size).unwrap_or_default();
                    if let Some(text) = commit_text(ui, id.with(side), &shown, &format!("{side} e.g. 10 KiB"), 96.0) {
                        match parse_size(&text) {
                            Ok(v) => *bound = v,
                            Err(e) => err = Some(e),
                        }
                    }
                    if side == "min" {
                        ui.label("to");
                    }
                }
            });
            date_row(ui, id.with("modified"), "Modified", &mut spec.modified, true, &mut err);
            date_row(ui, id.with("created"), "Added", &mut spec.created, false, &mut err);

            heading(ui, "Extensions");
            entry_list(ui, id.with("allow"), "Only", "jpg, png, ... Enter to add", true, &mut spec.ext_allow);
            list_add(ui, id.with("deny"), "Not", "gif, tmp, ... Enter to add", true, &mut spec.ext_deny);

            heading(ui, "Library");
            tri_row(ui, "Indexed", &mut spec.indexed, "Yes", "No");
            tri_row(ui, "Description", &mut spec.description, "Has", "None");
            tri_row(ui, "CLIP", &mut spec.clip, "Has", "None");
            tri_row(ui, "Files", &mut spec.offline, "Offline", "Online");
            entry_list(ui, id.with("category"), "Category", "any of, Enter to add", false, &mut spec.categories);
            entry_list(ui, id.with("tag"), "Tag", "Enter to add", false, &mut spec.tags);
            if spec.tags.len() > 1 {
                ui.checkbox(&mut spec.tags_all, "Match all tags");
            }
        });
        if let Some(e) = &err {
            ui.colored_label(crate::theme::error(ui), e);
        }
        if spec != before {
            self.set_filter_spec(spec);
        }

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            let differs = self.filters_differ_from_defaults();
            if ui.add_enabled(differs, Button::new("Make default")).on_hover_text("Use these size, date, icon, extension and folder filters in every new tab and for scans").clicked() {
                self.make_filters_default();
            }
            if ui.button("Reset").on_hover_text("Back to the default filters").clicked() {
                self.reset_filters();
            }
            if ui.button("Clear all").on_hover_text("Remove every filter, the defaults included").clicked() {
                self.clear_filters();
            }
        });
        if !self.last_scan_rows.is_empty() && ui.button("Restore last scan").on_hover_text("Show the results of the last recursive scan again").clicked() {
            self.table.replace_all(self.last_scan_rows.iter().cloned());
            self.viewer.showing_similarity = false;
            self.viewer.similar_scores.clear();
            self.scan_done = true;
            self.file_scan_progress = 1.0;
        }
    }
}
