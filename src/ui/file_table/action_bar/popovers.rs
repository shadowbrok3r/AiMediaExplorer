//! Popovers of the batch actions: tags, category, collections and confirmations.

use std::collections::BTreeMap;

use eframe::egui::*;

use super::{files_noun, names_text};
use crate::Thumbnail;
use crate::command::ActionId;
use crate::command::actions::spec_of;
use crate::database::metadata_edit::TagChange;
use crate::ui::inspector::tags::{TagTally, split_input, suggest, take_complete};

/// Width of every popover.
const WIDTH: f32 = 360.0;
/// Suggestions listed under a field.
const SUGGESTIONS: usize = 8;
/// Collections listed before the list scrolls.
const LIST_HEIGHT: f32 = 180.0;

/// A confirmation waiting for an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    pub action: ActionId,
    pub count: usize,
    pub names: Vec<String>,
}

/// Which popover is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Popover {
    Tag,
    Category,
    AddToCollection,
    RemoveFromCollection,
    Confirm(Pending),
}

/// What the user decided in a popover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Intent {
    Close,
    Tags(TagChange),
    Category(Option<String>),
    Collection { name: String, add: bool },
    Confirmed(ActionId),
}

#[derive(Clone, Debug, Default)]
pub struct TagForm {
    pub input: String,
    /// Tags completed with a comma and waiting to be added.
    pub add: Vec<String>,
    /// Tags of the selection marked for removal.
    pub remove: Vec<String>,
    pub tally: TagTally,
    pub focus: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CategoryForm {
    pub input: String,
    /// Categories of the selection with the number of files each is on.
    pub current: Vec<(String, usize)>,
    pub uncategorized: usize,
    pub focus: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CollectionForm {
    pub input: String,
    pub focus: bool,
}

/// The state of every popover form.
#[derive(Clone, Debug, Default)]
pub struct Forms {
    pub tag: TagForm,
    pub category: CategoryForm,
    pub collection: CollectionForm,
}

impl Forms {
    /// Fresh forms for `rows`: the tag tally and the categories they hold.
    pub fn for_rows(rows: &[Thumbnail]) -> Self {
        let files: Vec<&Thumbnail> = rows.iter().filter(|r| r.file_type != "<DIR>").collect();
        let tally = TagTally::of(files.iter().map(|r| r.tags.as_slice()));
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut uncategorized = 0;
        for r in &files {
            match r.category.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
                Some(c) => *counts.entry(c.to_string()).or_default() += 1,
                None => uncategorized += 1,
            }
        }
        Self { tag: TagForm { tally, focus: true, ..Default::default() }, category: CategoryForm { current: counts.into_iter().collect(), uncategorized, focus: true, ..Default::default() }, collection: CollectionForm { focus: true, ..Default::default() } }
    }
}

/// What a popover reads while it draws.
pub struct View<'a> {
    pub popover: &'a Popover,
    pub rows: &'a [Thumbnail],
    pub known_tags: &'a [String],
    pub known_categories: &'a [String],
    pub collections: &'a [(String, u64)],
}

impl View<'_> {
    fn files(&self) -> usize {
        self.rows.iter().filter(|r| r.file_type != "<DIR>").count()
    }
}

/// Draws the popover with its bottom edge centred on `anchor`; returns its rectangle and the decision made this frame.
pub fn show(ctx: &Context, forms: &mut Forms, view: &View<'_>, anchor: Pos2, id: Id) -> (Rect, Option<Intent>) {
    let mut intent = None;
    let area = Area::new(id).order(Order::Middle).pivot(Align2::CENTER_BOTTOM).fixed_pos(anchor).constrain(true).show(ctx, |ui| {
        Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(WIDTH);
            intent = match view.popover {
                Popover::Tag => tag_ui(ui, &mut forms.tag, view),
                Popover::Category => category_ui(ui, &mut forms.category, view),
                Popover::AddToCollection => collection_ui(ui, &mut forms.collection, view, true),
                Popover::RemoveFromCollection => collection_ui(ui, &mut forms.collection, view, false),
                Popover::Confirm(p) => confirm_ui(ui, p),
            };
        });
    });
    (area.response.rect, intent)
}

fn header(ui: &mut Ui, text: String) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(text).strong());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            close = crate::icons::icon_button(ui, crate::icons::svg::X, "Close (Esc)").clicked();
        });
    });
    close
}

fn input_row(ui: &mut Ui, text: &mut String, hint: &str, focus: &mut bool, id_salt: &str) -> (Response, bool) {
    let response = TextEdit::singleline(text).id_salt(id_salt).hint_text(hint).desired_width(ui.available_width()).show(ui).response.response;
    if std::mem::take(focus) {
        response.request_focus();
    }
    let entered = response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    (response, entered)
}

fn suggestion_chips(ui: &mut Ui, picks: &[String]) -> Option<String> {
    let mut chosen = None;
    if !picks.is_empty() {
        ui.horizontal_wrapped(|ui| {
            for pick in picks {
                if ui.add(Button::new(RichText::new(pick).small()).small()).clicked() {
                    chosen = Some(pick.clone());
                }
            }
        });
    }
    chosen
}

fn tag_ui(ui: &mut Ui, form: &mut TagForm, view: &View<'_>) -> Option<Intent> {
    let files = view.files();
    let mut intent = None;
    if header(ui, format!("Tag {}", files_noun(files))) {
        return Some(Intent::Close);
    }
    if form.tally.tags.is_empty() {
        ui.label(RichText::new("No file has tags").weak());
    } else {
        ui.label(RichText::new("Tags on the selection (click to remove)").weak().small());
        ui.horizontal_wrapped(|ui| {
            for t in form.tally.tags.clone() {
                let marked = form.remove.iter().any(|r| r.eq_ignore_ascii_case(&t.tag));
                let ratio = if t.count == form.tally.rows { String::new() } else { format!(" {}/{}", t.count, form.tally.rows) };
                let label = if marked { format!("{}{ratio}  (remove)", t.tag) } else { format!("{}{ratio}", t.tag) };
                if ui.add(Button::selectable(marked, label).small()).clicked() {
                    if marked {
                        form.remove.retain(|r| !r.eq_ignore_ascii_case(&t.tag));
                    } else {
                        form.remove.push(t.tag.clone());
                    }
                }
            }
        });
    }
    if !form.add.is_empty() {
        ui.label(RichText::new("To add (click to drop)").weak().small());
        ui.horizontal_wrapped(|ui| {
            for tag in form.add.clone() {
                if ui.add(Button::new(format!("{tag}  x")).small()).clicked() {
                    form.add.retain(|t| *t != tag);
                }
            }
        });
    }
    let (response, entered) = input_row(ui, &mut form.input, "Add tags, comma separated", &mut form.focus, "ame_bar_tag_input");
    if response.changed() && form.input.contains(',') {
        let (done, rest) = take_complete(&form.input);
        for tag in done {
            if !form.add.iter().any(|t| t.eq_ignore_ascii_case(&tag)) {
                form.add.push(tag);
            }
        }
        form.input = rest;
    }
    let present: Vec<String> = form.tally.common().iter().map(|t| t.tag.clone()).chain(form.add.iter().cloned()).collect();
    let picks = suggest(view.known_tags, &form.input, &present, SUGGESTIONS);
    if let Some(pick) = suggestion_chips(ui, &picks) {
        form.add.push(pick);
        form.input.clear();
    }
    let mut add = form.add.clone();
    add.extend(split_input(&form.input));
    let change = TagChange { add, remove: form.remove.clone() };
    let apply_label = match (change.add.is_empty(), change.remove.is_empty()) {
        (_, true) => format!("Add to {}", files_noun(files)),
        (true, false) => format!("Remove from {}", files_noun(files)),
        _ => format!("Apply to {}", files_noun(files)),
    };
    ui.horizontal(|ui| {
        if ui.add_enabled(!change.is_empty() && files > 0, Button::new(apply_label)).clicked() || (entered && !change.is_empty()) {
            intent = Some(Intent::Tags(change.clone()));
        }
        if ui.button("Cancel").clicked() {
            intent = Some(Intent::Close);
        }
    });
    intent
}

fn category_ui(ui: &mut Ui, form: &mut CategoryForm, view: &View<'_>) -> Option<Intent> {
    let files = view.files();
    let mut intent = None;
    if header(ui, format!("Category for {}", files_noun(files))) {
        return Some(Intent::Close);
    }
    let mut parts: Vec<String> = form.current.iter().map(|(c, n)| format!("{c} ({n})")).collect();
    if form.uncategorized > 0 {
        parts.push(format!("none ({})", form.uncategorized));
    }
    ui.label(RichText::new(parts.join(", ")).weak().small());
    let (_, entered) = input_row(ui, &mut form.input, "Category for all selected files", &mut form.focus, "ame_bar_category_input");
    let picks = suggest(view.known_categories, &form.input, &[], SUGGESTIONS);
    if let Some(pick) = suggestion_chips(ui, &picks) {
        form.input = pick;
    }
    let typed = !form.input.trim().is_empty();
    ui.horizontal(|ui| {
        if ui.add_enabled(typed && files > 0, Button::new(format!("Set for {}", files_noun(files)))).clicked() || (entered && typed) {
            intent = Some(Intent::Category(Some(form.input.trim().to_string())));
        }
        if ui.add_enabled(files > 0, Button::new("Clear category")).on_hover_text("Remove the category from every selected file").clicked() {
            intent = Some(Intent::Category(None));
        }
        if ui.button("Cancel").clicked() {
            intent = Some(Intent::Close);
        }
    });
    intent
}

/// Collections whose name contains `filter`, ignoring case.
pub fn matching_collections<'a>(collections: &'a [(String, u64)], filter: &str) -> Vec<&'a (String, u64)> {
    let needle = filter.trim().to_lowercase();
    collections.iter().filter(|(name, _)| needle.is_empty() || name.to_lowercase().contains(&needle)).collect()
}

/// True when the typed name should create a collection: it is not blank and no collection has that name.
pub fn creates_collection(collections: &[(String, u64)], typed: &str) -> bool {
    let typed = typed.trim();
    !typed.is_empty() && !collections.iter().any(|(n, _)| n.eq_ignore_ascii_case(typed))
}

fn collection_ui(ui: &mut Ui, form: &mut CollectionForm, view: &View<'_>, add: bool) -> Option<Intent> {
    let files = view.files();
    let mut intent = None;
    let title = if add { format!("Add {} to a collection", files_noun(files)) } else { format!("Remove {} from a collection", files_noun(files)) };
    if header(ui, title) {
        return Some(Intent::Close);
    }
    let hint = if add { "Filter, or type a new collection name" } else { "Filter the collections" };
    let (_, entered) = input_row(ui, &mut form.input, hint, &mut form.focus, "ame_bar_collection_input");
    let shown = matching_collections(view.collections, &form.input);
    if view.collections.is_empty() {
        ui.label(RichText::new("No collections yet").weak());
    }
    ScrollArea::vertical().max_height(LIST_HEIGHT).auto_shrink([true, true]).show(ui, |ui| {
        for (name, count) in &shown {
            if ui.add(Button::selectable(false, format!("{name} ({count})")).min_size(vec2(ui.available_width(), 0.0))).clicked() {
                intent = Some(Intent::Collection { name: name.clone(), add });
            }
        }
    });
    let typed = form.input.trim().to_string();
    if add && creates_collection(view.collections, &typed) {
        if ui.button(format!("Create \"{typed}\" and add {}", files_noun(files))).clicked() || entered {
            intent = Some(Intent::Collection { name: typed, add: true });
        }
    } else if entered && shown.len() == 1 {
        intent = Some(Intent::Collection { name: shown[0].0.clone(), add });
    }
    if ui.button("Cancel").clicked() {
        intent = Some(Intent::Close);
    }
    intent
}

fn confirm_ui(ui: &mut Ui, pending: &Pending) -> Option<Intent> {
    let (title, body, button) = super::confirm_text(pending.action, pending.count);
    let mut intent = None;
    if header(ui, title) {
        return Some(Intent::Close);
    }
    ui.add(Label::new(body).wrap());
    if !pending.names.is_empty() {
        ui.add_space(crate::theme::metrics(ui.ctx()).gap.y);
        ui.add(Label::new(RichText::new(names_text(&pending.names)).weak()).wrap());
    }
    ui.add_space(crate::theme::metrics(ui.ctx()).gap.y);
    ui.horizontal(|ui| {
        let confirm = if spec_of(pending.action).confirm == crate::command::actions::Confirm::Always && pending.action == ActionId::RemoveFromLibrary { ui.add(crate::theme::danger_button(button)) } else { ui.add(Button::new(button)) };
        if confirm.clicked() {
            intent = Some(Intent::Confirmed(pending.action));
        }
        if ui.button("Cancel").clicked() {
            intent = Some(Intent::Close);
        }
    });
    intent
}
