//! The path area of the toolbar: breadcrumbs that open folders, and a text field with completion that takes a path or any scope string.

use eframe::egui::*;
use egui::text::{CCursor, CCursorRange};
use egui::text_edit::TextEditState;

use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;

/// One path segment: the label shown and the path a click opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crumb {
    pub label: String,
    pub target: String,
}

/// Where the crumbs of the last frame sit, for tests.
#[derive(Clone, Debug)]
pub struct CrumbLayout {
    /// Crumbs moved into the overflow menu.
    pub hidden: usize,
    pub overflow: Option<Rect>,
    pub shown: Vec<(Rect, String)>,
    pub row: Rect,
}

fn layout_id() -> Id {
    Id::new("ame.breadcrumbs.layout")
}

/// The layout recorded by the last [`FileExplorer::path_area`](super::super::FileExplorer::path_area).
pub fn last_layout(ctx: &Context) -> Option<CrumbLayout> {
    ctx.data(|d| d.get_temp::<CrumbLayout>(layout_id()))
}

/// The segments of `path`, each with the cumulative path up to it.
pub fn crumbs(path: &str) -> Vec<Crumb> {
    let mut accum = if path.starts_with('/') { String::from("/") } else { String::new() };
    path.split(['\\', '/'])
        .filter(|s| !s.is_empty())
        .map(|part| {
            if !accum.ends_with(std::path::MAIN_SEPARATOR) && !accum.is_empty() {
                accum.push(std::path::MAIN_SEPARATOR);
            }
            accum.push_str(part);
            Crumb { label: part.to_string(), target: accum.clone() }
        })
        .collect()
}

/// Index of the first crumb shown: the trailing crumbs that fit `avail`, at least the last; the rest go to the overflow menu.
pub fn first_visible(widths: &[f32], sep: f32, overflow_w: f32, avail: f32) -> usize {
    let n = widths.len();
    if n == 0 {
        return 0;
    }
    let tail = |from: usize| widths[from..].iter().sum::<f32>() + sep * (n - from - 1) as f32;
    if tail(0) <= avail {
        return 0;
    }
    (1..n).find(|&from| overflow_w + sep + tail(from) <= avail).unwrap_or(n - 1)
}

/* ---------------------------------------- completion ---------------------------------------- */

/// Scope strings offered while the text is not a path.
const SCOPE_HINTS: [&str; 7] = ["lib:all", "recent:all", "smart:no-clip", "smart:undescribed", "smart:untagged", "smart:duplicates", "smart:failed-thumbs"];
/// Suggestions listed under the field.
pub const SUGGESTIONS: usize = 8;
/// Folders read from disk while completing a path.
const CHILD_LIMIT: usize = 400;

/// True when `text` starts like a scope string: two or more letters and a colon, which no drive letter has.
fn scope_like(text: &str) -> bool {
    text.split_once(':').is_some_and(|(head, _)| head.len() > 1 && head.chars().all(|c| c.is_ascii_alphabetic()))
}

/// True when `text` reads as a place on disk rather than a scope string.
pub fn is_path_like(text: &str) -> bool {
    !scope_like(text) && (text.starts_with('/') || text.starts_with('~') || text.starts_with("\\\\") || text.chars().nth(1) == Some(':') || text.contains(['/', '\\']))
}

/// `text` cut after its last separator: the folder typed so far and the name being typed.
pub fn split_partial(text: &str) -> Option<(&str, &str)> {
    let cut = text.rfind(['/', '\\'])? + 1;
    Some((&text[..cut], &text[cut..]))
}

/// The folders inside `dir` (a path ending in a separator) as full paths, in name order.
pub fn disk_children(dir: &str) -> Vec<String> {
    let Ok(read) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<String> = read.flatten().take(CHILD_LIMIT).filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| format!("{dir}{}", e.file_name().to_string_lossy())).collect();
    out.sort_by_key(|p| p.to_lowercase());
    out
}

/// Up to `limit` places to offer for what is typed: folders under the typed folder, recent paths, then scope strings; nothing typed offers the recent paths.
pub fn completions(input: &str, recents: &[String], children: &dyn Fn(&str) -> Vec<String>, limit: usize) -> Vec<String> {
    let needle = input.trim();
    let lower = needle.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let add = |out: &mut Vec<String>, item: &str| {
        if item != needle && !out.iter().any(|o| o == item) {
            out.push(item.to_string());
        }
    };
    if needle.is_empty() {
        recents.iter().for_each(|r| add(&mut out, r));
        out.truncate(limit);
        return out;
    }
    if is_path_like(needle) {
        if let Some((dir, partial)) = split_partial(needle) {
            let partial = partial.to_lowercase();
            for child in children(dir) {
                let name = child[dir.len()..].to_lowercase();
                if name.starts_with(&partial) {
                    add(&mut out, &child);
                }
            }
        }
    }
    for r in recents.iter().filter(|r| r.to_lowercase().starts_with(&lower)) {
        add(&mut out, r);
    }
    for r in recents.iter().filter(|r| r.to_lowercase().contains(&lower)) {
        add(&mut out, r);
    }
    if !is_path_like(needle) {
        for hint in SCOPE_HINTS.iter().filter(|h| h.starts_with(&lower)) {
            add(&mut out, hint);
        }
    }
    out.truncate(limit);
    out
}

/* ------------------------------------------ state ------------------------------------------- */

/// State of the path area of one tab.
#[derive(Clone, Debug, Default)]
pub struct PathBar {
    pub editing: bool,
    /// The field takes the keyboard focus, with its text selected, on the next frame.
    pub focus: bool,
    /// Suggestion chosen with the arrow keys.
    pub highlight: Option<usize>,
    pub suggestions: Vec<String>,
    computed_for: Option<String>,
    suggest_rect: Option<Rect>,
    /// How often the suggestions were computed.
    pub computed: u64,
}

fn edit_id(ex: &FileExplorer) -> Id {
    Id::new(("ame_path_edit", ex.viewer.scope))
}

fn with_separator(path: &str) -> String {
    if std::path::Path::new(path).is_dir() && !path.ends_with(['/', '\\']) { format!("{path}{}", std::path::MAIN_SEPARATOR) } else { path.to_string() }
}

impl FileExplorer {
    /// Starts editing the address with the current place selected.
    pub fn edit_address(&mut self) {
        self.nav.input = if self.nav.scope.is_filesystem() { self.current_path.clone() } else { self.nav.scope.format() };
        let bar = &mut self.toolbar.path;
        bar.editing = true;
        bar.focus = true;
        bar.highlight = None;
        bar.computed_for = None;
        bar.suggest_rect = None;
    }

    /// Stops editing and drops what was typed.
    pub fn cancel_address(&mut self) {
        self.toolbar.path.editing = false;
        self.toolbar.path.suggestions.clear();
        self.nav.input = self.nav.scope.format();
    }

    /// Opens what the field holds; a bad address keeps the field open with a toast.
    pub fn submit_address_text(&mut self) -> bool {
        match self.submit_address() {
            Ok(()) => {
                self.toolbar.path.editing = false;
                self.toolbar.path.suggestions.clear();
                true
            }
            Err(reason) => {
                crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
                false
            }
        }
    }

    /// Opens the folder a breadcrumb names the way a click does.
    pub fn open_crumb(&mut self, target: String) {
        if let Err(reason) = self.open_scope(super::super::scope::nav::scope_for_path(&target)) {
            crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
        }
    }

    fn refresh_suggestions(&mut self) {
        let input = self.nav.input.clone();
        if self.toolbar.path.computed_for.as_deref() == Some(input.as_str()) {
            return;
        }
        let recents = recent_places(self);
        self.toolbar.path.suggestions = completions(&input, &recents, &disk_children, SUGGESTIONS);
        self.toolbar.path.computed_for = Some(input);
        self.toolbar.path.computed += 1;
        self.toolbar.path.highlight = None;
    }

    /// Draws the path area into `ui` (whose rectangle is the area): crumbs, or the scope's name, or the address field while editing. Returns the rectangle used.
    pub fn path_area(&mut self, ui: &mut Ui) -> Rect {
        let rect = ui.available_rect_before_wrap();
        let tokens = crate::theme::tokens(ui.ctx());
        let radius = CornerRadius::same(crate::theme::metrics(ui.ctx()).radius_sm);
        let stroke = if self.toolbar.path.editing { Stroke::new(1.0, tokens.primary) } else { ui.visuals().widgets.noninteractive.bg_stroke };
        ui.painter().rect(rect, radius, ui.visuals().extreme_bg_color, stroke, StrokeKind::Inside);
        if self.toolbar.path.editing {
            self.address_field(ui, rect);
        } else {
            self.address_view(ui, rect);
        }
        rect
    }

    fn address_view(&mut self, ui: &mut Ui, rect: Rect) {
        let background = ui.interact(rect, Id::new(("ame_path_bg", self.viewer.scope)), Sense::click());
        let pad = ui.spacing().item_spacing.x;
        let spinner = if self.is_loading() { 22.0 } else { 0.0 };
        let inner = Rect::from_min_max(rect.min + vec2(pad, 0.0), pos2((rect.right() - pad - spinner).max(rect.left() + pad), rect.bottom()));
        let mut recorded = CrumbLayout { hidden: 0, overflow: None, shown: Vec::new(), row: rect };
        let mut open = None;
        if self.nav.scope.is_filesystem() {
            let crumbs = crumbs(&self.current_path);
            ui.scope_builder(UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)), |ui| {
                ui.set_clip_rect(inner.intersect(ui.clip_rect()));
                if crumbs.is_empty() {
                    ui.weak("Type a path");
                } else {
                    open = crumb_row(ui, &crumbs, &mut recorded);
                }
            });
        } else {
            let title = self.nav.title.clone().unwrap_or_else(|| self.nav.scope.default_title());
            let note = if self.nav.rows_only { String::new() } else { format!("  {}", self.nav.scope.format()) };
            let count = format!("  {} items", self.table.len());
            ui.scope_builder(UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)), |ui| {
                ui.set_clip_rect(inner.intersect(ui.clip_rect()));
                let text = RichText::new(title).strong();
                ui.add(Label::new(text).truncate().selectable(false));
                ui.add(Label::new(RichText::new(note + &count).weak()).truncate().selectable(false));
            });
        }
        if spinner > 0.0 {
            let at = Rect::from_center_size(pos2(rect.right() - pad - spinner / 2.0, rect.center().y), Vec2::splat(14.0));
            ui.put(at, Spinner::new().size(14.0));
        }
        ui.ctx().data_mut(|d| d.insert_temp(layout_id(), recorded));
        background.clone().on_hover_text("Click to type a path or a scope (Ctrl+L)");
        match open {
            Some((target, true)) => crate::command::send(crate::command::AppCommand::open_in_new_tab(super::super::scope::nav::scope_for_path(&target))),
            Some((target, false)) => self.open_crumb(target),
            None if background.clicked() => self.edit_address(),
            None => {}
        }
    }

    fn address_field(&mut self, ui: &mut Ui, rect: Rect) {
        let id = edit_id(self);
        let first = std::mem::take(&mut self.toolbar.path.focus);
        if first {
            let len = self.nav.input.chars().count();
            let mut state = TextEditState::load(ui.ctx(), id).unwrap_or_default();
            state.cursor.set_char_range(Some(CCursorRange::two(CCursor::new(0), CCursor::new(len))));
            state.store(ui.ctx(), id);
            ui.memory_mut(|m| m.request_focus(id));
        }
        self.refresh_suggestions();
        let count = self.toolbar.path.suggestions.len();
        let (up, down, tab, escape) = ui.input_mut(|i| (i.consume_key(Modifiers::NONE, Key::ArrowUp), i.consume_key(Modifiers::NONE, Key::ArrowDown), i.consume_key(Modifiers::NONE, Key::Tab), i.consume_key(Modifiers::NONE, Key::Escape)));
        let bar = &mut self.toolbar.path;
        if count > 0 {
            if down {
                bar.highlight = Some(bar.highlight.map_or(0, |h| (h + 1) % count));
            }
            if up {
                bar.highlight = Some(bar.highlight.map_or(count - 1, |h| (h + count - 1) % count));
            }
        }
        let mut picked: Option<String> = None;
        if tab && count > 0 {
            let pick = bar.highlight.unwrap_or(0).min(count - 1);
            let item = bar.suggestions[pick].clone();
            self.nav.input = with_separator(&item);
            let len = self.nav.input.chars().count();
            let mut state = TextEditState::load(ui.ctx(), id).unwrap_or_default();
            state.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(len))));
            state.store(ui.ctx(), id);
            ui.memory_mut(|m| m.request_focus(id));
        }
        let inner = rect.shrink2(vec2(2.0, 0.0));
        let response = ui
            .scope_builder(UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)), |ui| {
                TextEdit::singleline(&mut self.nav.input).id(id).frame(Frame::NONE).lock_focus(true).hint_text("Path or scope (lib:all, tag:x, col:Name)").desired_width(inner.width()).show(ui).response
            })
            .inner;
        let pressed_inside = self.toolbar.path.suggest_rect.is_some_and(|r| ui.input(|i| i.pointer.hover_pos().is_some_and(|p| r.contains(p))));
        let enter = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        if enter {
            if let Some(h) = self.toolbar.path.highlight.filter(|h| *h < self.toolbar.path.suggestions.len()) {
                self.nav.input = self.toolbar.path.suggestions[h].clone();
            }
            self.submit_address_text();
            return;
        }
        if escape {
            self.cancel_address();
            return;
        }
        if let Some(list_rect) = self.suggestion_list(ui, rect, &mut picked) {
            self.toolbar.path.suggest_rect = Some(list_rect);
        } else {
            self.toolbar.path.suggest_rect = None;
        }
        if let Some(item) = picked {
            self.nav.input = item;
            self.submit_address_text();
            return;
        }
        if response.lost_focus() && !pressed_inside && !first {
            self.cancel_address();
        }
    }

    /// Draws the suggestions under the field; a clicked one lands in `picked`. Returns the rectangle of the list.
    fn suggestion_list(&mut self, ui: &mut Ui, field: Rect, picked: &mut Option<String>) -> Option<Rect> {
        let list = self.toolbar.path.suggestions.clone();
        if list.is_empty() {
            return None;
        }
        let highlight = self.toolbar.path.highlight;
        let width = field.width().max(280.0);
        let area = Area::new(Id::new(("ame_path_suggest", self.viewer.scope))).order(Order::Foreground).fixed_pos(field.left_bottom() + vec2(0.0, 2.0)).constrain(true).show(ui.ctx(), |ui| {
            Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(width - 12.0);
                for (i, item) in list.iter().enumerate() {
                    let label = ui.add(Button::selectable(highlight == Some(i), item.as_str()).wrap_mode(TextWrapMode::Truncate).min_size(vec2(ui.available_width(), 0.0)));
                    if label.clicked() {
                        *picked = Some(item.clone());
                    }
                }
            });
        });
        Some(area.response.rect)
    }
}

/// Recent places: the tab's own history first, then the saved recent folders.
fn recent_places(ex: &FileExplorer) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |out: &mut Vec<String>, text: String| {
        if !text.is_empty() && !out.contains(&text) {
            out.push(text);
        }
    };
    for entry in ex.nav.history.entries().iter().rev() {
        let text = if let crate::command::Scope::Folder(p) = &entry.scope { p.clone() } else { entry.scope.format() };
        push(&mut out, text);
    }
    for path in crate::settings::store().read(|s| s.recent_paths.clone()) {
        push(&mut out, path);
    }
    out
}

fn crumb_text(label: &str) -> RichText {
    RichText::new(label).underline()
}

fn crumb_width(ui: &Ui, label: &str) -> f32 {
    let galley = WidgetText::from(crumb_text(label)).into_galley(ui, Some(TextWrapMode::Extend), f32::INFINITY, TextStyle::Button);
    galley.size().x + ui.spacing().button_padding.x * 2.0
}

/// Lays the crumbs out left to right inside the remaining width and returns the one clicked.
fn crumb_row(ui: &mut Ui, crumbs: &[Crumb], recorded: &mut CrumbLayout) -> Option<(String, bool)> {
    let item = ui.spacing().item_spacing.x;
    let caret = icons::IconSize::Sm.px(ui);
    let sep = caret + item * 2.0;
    let overflow_w = ui.spacing().interact_size.x.max(crumb_width(ui, "..."));
    let widths: Vec<f32> = crumbs.iter().map(|c| crumb_width(ui, &c.label)).collect();
    let first = first_visible(&widths, sep, overflow_w, ui.available_width());
    recorded.hidden = first;
    let mut open = None;
    if first > 0 {
        let (response, _) = containers::menu::MenuButton::new("...").ui(ui, |ui| {
            for crumb in &crumbs[..first] {
                let item = ui.button(&crumb.label);
                if item.clicked() || item.middle_clicked() {
                    open = Some((crumb.target.clone(), item.middle_clicked() || ui.input(|i| i.modifiers.command)));
                    ui.close();
                }
            }
        });
        recorded.overflow = Some(response.rect);
        svg::CARET_RIGHT.show_tinted(ui, caret, ui.visuals().weak_text_color());
    }
    for (i, crumb) in crumbs.iter().enumerate().skip(first) {
        let response = ui.selectable_label(false, crumb_text(&crumb.label));
        recorded.shown.push((response.rect, crumb.target.clone()));
        if response.clicked() || response.middle_clicked() {
            open = Some((crumb.target.clone(), response.middle_clicked() || ui.input(|i| i.modifiers.command)));
        }
        if i + 1 < crumbs.len() {
            svg::CARET_RIGHT.show_tinted(ui, caret, ui.visuals().weak_text_color());
        }
    }
    open
}

#[cfg(test)]
mod tests;
