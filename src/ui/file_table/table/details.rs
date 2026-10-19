//! The details (list) view: a virtualized table over a row store and its view index.

use std::collections::VecDeque;
use std::time::Instant;

use egui::scroll_area::ScrollBarVisibility;
use egui::{Align, CursorIcon, Id, Label, Layout, Rect, RichText, Sense, Stroke, Ui};
use egui_extras::{Column, TableBuilder};

use super::actions::{MenuAction, MenuTargets, menu_entries};
use super::cells::{draw_cell, is_dir};
use super::columns::{Col, DetailsPrefs};
use super::keys::{self, KeyAction};
use super::{FileTableViewer, TabAction};
use crate::Thumbnail;
use crate::icons::{self, IconSize, svg};
use crate::ui::file_table::rows::{PathKey, RowStore, Selection, SortSpec, ViewIndex};

pub const ROW_HEIGHT: f32 = 75.0;
pub const HEADER_HEIGHT: f32 = 26.0;
const FRAME_SAMPLES: usize = 240;

/// Everything drawing reads.
pub struct DetailsInputs<'a> {
    pub store: &'a RowStore<Thumbnail>,
    pub view: &'a ViewIndex,
    pub selection: &'a Selection,
    pub viewer: &'a FileTableViewer,
}

/// Something the user did in the table.
#[derive(Clone, Debug)]
pub enum DetailsEvent {
    /// Primary click on a row with the modifiers held at the time.
    Click { key: PathKey, ctrl: bool, shift: bool },
    /// Secondary click on a row; selects it when it was not selected.
    ContextOn { key: PathKey },
    Open { key: PathKey },
    Sort { column: Col, additive: bool },
    ToggleColumn { column: Col, hidden: bool },
    /// Click on a category or tag in a cell.
    Tab(TabAction),
    Menu(MenuAction),
    Key(KeyAction),
}

pub struct DetailsView {
    pub prefs: DetailsPrefs,
    /// Rows that fit the table body, as of the last frame.
    pub page_rows: usize,
    scroll_to: Option<usize>,
    scroll_top: bool,
    kb_active: bool,
    frame_ms: VecDeque<f32>,
    drawn_rows: usize,
    rects: Vec<(usize, Rect)>,
}

impl Default for DetailsView {
    fn default() -> Self {
        Self { prefs: DetailsPrefs::default(), page_rows: 8, scroll_to: None, scroll_top: false, kb_active: false, frame_ms: VecDeque::new(), drawn_rows: 0, rects: Vec::new() }
    }
}

/// Where the column and sort preferences live between sessions.
pub fn prefs_id() -> Id {
    Id::new("ame_details_prefs")
}

impl DetailsView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the saved preferences.
    pub fn load_prefs(&mut self, ctx: &egui::Context) -> Option<SortSpec> {
        let saved = ctx.data_mut(|d| d.get_persisted::<DetailsPrefs>(prefs_id()))?;
        let sort = saved.sort.clone();
        self.prefs = saved;
        Some(sort)
    }

    /// Saves the preferences with `sort` as the current order.
    pub fn save_prefs(&mut self, ctx: &egui::Context, sort: &SortSpec) {
        self.prefs.sort = sort.clone();
        let prefs = self.prefs.clone();
        ctx.data_mut(|d| d.insert_persisted(prefs_id(), prefs));
    }

    /// Scrolls the view position `pos` into sight on the next frame.
    pub fn scroll_to(&mut self, pos: usize) {
        self.scroll_to = Some(pos);
    }

    /// Jumps to the first row on the next frame.
    pub fn scroll_to_top(&mut self) {
        self.scroll_top = true;
    }

    /// Whether the table currently takes keyboard commands.
    pub fn keyboard_active(&self) -> bool {
        self.kb_active
    }

    pub fn set_keyboard_active(&mut self, active: bool) {
        self.kb_active = active;
    }

    /// Rows laid out in the last frame.
    pub fn drawn_rows(&self) -> usize {
        self.drawn_rows
    }

    /// View position and screen rectangle of each row laid out in the last frame.
    pub fn row_rects(&self) -> &[(usize, Rect)] {
        &self.rects
    }

    /// CPU time of recent frames in milliseconds, oldest first.
    pub fn frame_times(&self) -> &VecDeque<f32> {
        &self.frame_ms
    }

    pub fn clear_frame_times(&mut self) {
        self.frame_ms.clear();
    }

    fn record(&mut self, started: Instant) {
        if self.frame_ms.len() >= FRAME_SAMPLES {
            self.frame_ms.pop_front();
        }
        self.frame_ms.push_back(started.elapsed().as_secs_f32() * 1000.0);
    }

    /// Draws the table into `ui` and returns what the user did.
    pub fn show(&mut self, ui: &mut Ui, input: &DetailsInputs<'_>) -> Vec<DetailsEvent> {
        let started = Instant::now();
        let mut events = Vec::new();
        let area = ui.available_rect_before_wrap();
        let hovered = ui.rect_contains_pointer(area);
        if ui.input(|i| i.pointer.any_pressed()) {
            self.kb_active = hovered;
        }
        let unfocused = ui.memory(|m| m.focused().is_none());
        if !ui.ctx().egui_wants_keyboard_input() && !egui::Popup::is_any_open(ui.ctx()) && (self.kb_active || (hovered && unfocused)) {
            let page = self.page_rows;
            events.extend(ui.input_mut(|i| keys::read(i, page)).into_iter().map(DetailsEvent::Key));
        }

        let viewer = input.viewer;
        let cols = self.prefs.visible(viewer.mode, viewer.showing_similarity);
        let available = Col::available(viewer.mode, viewer.showing_similarity);
        let mods = ui.input(|i| i.modifiers);
        let sort = input.view.sort();
        let total = input.view.len();
        let focus = input.selection.focus();
        let prefs = &self.prefs;

        let spacing_x = ui.spacing().item_spacing.x;
        let widest = cols.iter().map(|c| c.width().initial).sum::<f32>() + spacing_x * (cols.len() as f32 + 1.0) + 16.0;
        let content_w = ui.available_width().max(widest);
        let scroll_to = self.scroll_to.take();
        let scroll_top = std::mem::take(&mut self.scroll_top);
        let mut drawn = 0usize;
        let mut rects = std::mem::take(&mut self.rects);
        rects.clear();
        let body_height = egui::ScrollArea::horizontal()
            .id_salt("ame_details_hscroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.set_width(content_w);
                let mut builder = TableBuilder::new(ui)
                    .id_salt(("ame_details_table", cols.iter().map(|c| c.id()).collect::<Vec<u8>>()))
                    .striped(true)
                    .resizable(true)
                    .sense(Sense::click())
                    .cell_layout(Layout::left_to_right(Align::Center))
                    .auto_shrink([false, false])
                    .min_scrolled_height(0.0)
                    .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible);
                if let Some(pos) = scroll_to {
                    builder = builder.scroll_to_row(pos, None).animate_scrolling(false);
                }
                if scroll_top {
                    builder = builder.vertical_scroll_offset(0.0);
                }
                for (i, col) in cols.iter().enumerate() {
                    let w = col.width();
                    builder = builder.column(if i + 1 == cols.len() {
                        Column::remainder().at_least(w.min).clip(true)
                    } else {
                        Column::initial(w.initial).range(w.min..=w.max).resizable(true).clip(true)
                    });
                }
                let table = builder.header(HEADER_HEIGHT, |mut header| {
                    for col in &cols {
                        let (_, mut resp) = header.col(|ui| header_cell(ui, *col, sort));
                        if col.sortable() {
                            resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                            if resp.clicked() {
                                events.push(DetailsEvent::Sort { column: *col, additive: mods.shift });
                            }
                        }
                        resp.context_menu(|ui| column_menu(ui, prefs, &available, &mut events));
                    }
                });
                let out = table.body(|mut body| {
                    if total == 0 {
                        let hidden = input.store.len();
                        let note = if hidden == 0 { "No items".to_string() } else { format!("{hidden} items hidden by the current filter") };
                        body.ui_mut().vertical_centered(|ui| {
                            ui.add_space(crate::theme::metrics(ui.ctx()).gap.y * 8.0);
                            ui.weak(note);
                        });
                        return;
                    }
                    body.rows(ROW_HEIGHT, total, |mut row| {
                        let Some(idx) = input.view.store_index(row.index()) else { return };
                        let (Some(thumb), Some(key)) = (input.store.get(idx), input.store.key_at(idx)) else { return };
                        drawn += 1;
                        let selected = input.selection.contains(key);
                        let focused = focus == Some(key);
                        row.set_selected(selected);
                        for (n, col) in cols.iter().enumerate() {
                            let last = n + 1 == cols.len();
                            row.col(|ui| {
                                draw_cell(ui, viewer, *col, thumb, &mut events);
                                if focused {
                                    paint_focus(ui, n == 0, last);
                                }
                            });
                        }
                        let resp = row.response();
                        rects.push((row.index(), resp.rect));
                        if resp.clicked() {
                            events.push(DetailsEvent::Click { key: key.clone(), ctrl: mods.command, shift: mods.shift });
                        }
                        if resp.double_clicked() {
                            events.push(DetailsEvent::Open { key: key.clone() });
                        }
                        if resp.secondary_clicked() {
                            events.push(DetailsEvent::ContextOn { key: key.clone() });
                        }
                        resp.context_menu(|ui| row_menu(ui, input, key, thumb, &mut events));
                    });
                });
                out.inner_rect.height()
            })
            .inner;
        let rows_fit = (body_height / ROW_HEIGHT).floor() as usize;
        self.page_rows = rows_fit.max(1);
        self.drawn_rows = drawn;
        self.rects = rects;
        self.record(started);
        events
    }
}

fn header_cell(ui: &mut Ui, col: Col, sort: &SortSpec) {
    ui.horizontal(|ui| {
        ui.add(Label::new(RichText::new(col.header()).strong()).selectable(false).truncate());
        if let Some((priority, descending)) = sort.position_of(col.id()) {
            let icon = if descending { svg::SORT_DESCENDING } else { svg::SORT_ASCENDING };
            ui.add(icons::tinted_image(ui, icon, IconSize::Sm));
            if sort.keys().len() > 1 {
                ui.weak((priority + 1).to_string());
            }
        }
    });
}

fn column_menu(ui: &mut Ui, prefs: &DetailsPrefs, available: &[Col], events: &mut Vec<DetailsEvent>) {
    ui.label(RichText::new("Columns").strong());
    for col in available {
        let mut shown = !prefs.is_hidden(*col);
        let resp = ui.add_enabled(col.hideable(), egui::Checkbox::new(&mut shown, col.menu_name()));
        if resp.changed() {
            events.push(DetailsEvent::ToggleColumn { column: *col, hidden: !shown });
        }
    }
}

/// What a context menu opened on `key` would act on.
fn menu_targets(input: &DetailsInputs<'_>, key: &PathKey, row: &Thumbnail) -> MenuTargets {
    if !input.selection.contains(key) {
        return MenuTargets { rows: 1, dirs: usize::from(is_dir(row)) };
    }
    let dirs = input.selection.keys().iter().filter_map(|k| input.store.get_by_key(k)).filter(|r| is_dir(r)).count();
    MenuTargets { rows: input.selection.len(), dirs }
}

fn row_menu(ui: &mut Ui, input: &DetailsInputs<'_>, key: &PathKey, row: &Thumbnail, events: &mut Vec<DetailsEvent>) {
    let targets = menu_targets(input, key, row);
    ui.label(RichText::new(if targets.rows == 1 { "1 item".to_string() } else { format!("{} items", targets.rows) }).weak());
    ui.separator();
    for (action, enabled) in menu_entries(input.viewer.mode, targets) {
        let button = icons::button_widget(ui, action.icon(), action.label());
        if ui.add_enabled(enabled, button).clicked() {
            events.push(DetailsEvent::Menu(action));
            ui.close();
        }
    }
}

/// Outlines the focused row across its cells.
fn paint_focus(ui: &Ui, first: bool, last: bool) {
    let rect = ui.max_rect().expand2(ui.spacing().item_spacing * 0.5);
    let stroke = Stroke::new(1.5, ui.visuals().selection.stroke.color);
    let painter = ui.painter();
    painter.hline(rect.x_range(), rect.top() + 0.75, stroke);
    painter.hline(rect.x_range(), rect.bottom() - 0.75, stroke);
    if first {
        painter.vline(rect.left() + 0.75, rect.y_range(), stroke);
    }
    if last {
        painter.vline(rect.right() - 0.75, rect.y_range(), stroke);
    }
}
