//! The palette window: query field, ranked rows and keyboard handling.

use eframe::egui::text::{LayoutJob, TextFormat};
use eframe::egui::{Align, Align2, Color32, Context, EventFilter, Id, Key, Modifiers, Modal, Rect, ScrollArea, Sense, TextEdit, TextStyle, Ui, pos2, vec2};

use super::entries::search_entry;
use super::{Effect, Entry, PaletteState, matcher};
use crate::icons::{self, svg};

/// Rows listed at most.
pub const MAX_RESULTS: usize = 60;
const WIDTH: f32 = 580.0;
const LIST_HEIGHT: f32 = 380.0;
const ROW_PAD: f32 = 8.0;

/// An entry with the title positions the query matched.
#[derive(Clone, Debug)]
pub struct Hit<'a> {
    pub entry: &'a Entry,
    pub positions: Vec<usize>,
}

/// The entries `query` selects, best first, followed by the search entry when there is text.
pub fn hits<'a>(query: &str, entries: &'a [Entry], search: Option<&'a Entry>) -> Vec<Hit<'a>> {
    let ranked = matcher::rank(query, entries, |e| (e.title.as_str(), e.keywords.as_str()));
    let mut out: Vec<Hit> = ranked.into_iter().take(MAX_RESULTS).map(|r| Hit { entry: &entries[r.index], positions: r.found.positions }).collect();
    out.extend(search.map(|entry| Hit { entry, positions: Vec::new() }));
    out
}

/// Keys the palette reads in a frame.
#[derive(Default)]
struct Keys {
    up: bool,
    down: bool,
    page_up: bool,
    page_down: bool,
    enter: bool,
    ctrl_enter: bool,
    escape: bool,
}

fn read_keys(ctx: &Context) -> Keys {
    ctx.input_mut(|i| Keys {
        up: i.consume_key(Modifiers::NONE, Key::ArrowUp),
        down: i.consume_key(Modifiers::NONE, Key::ArrowDown),
        page_up: i.consume_key(Modifiers::NONE, Key::PageUp),
        page_down: i.consume_key(Modifiers::NONE, Key::PageDown),
        ctrl_enter: i.consume_key(Modifiers::COMMAND, Key::Enter),
        enter: i.consume_key(Modifiers::NONE, Key::Enter),
        escape: i.consume_key(Modifiers::NONE, Key::Escape),
    })
}

/// The selected index after the keys of a frame, kept inside `0..count`.
pub fn move_selection(selected: usize, count: usize, up: bool, down: bool, page_up: bool, page_down: bool) -> usize {
    if count == 0 {
        return 0;
    }
    let last = count - 1;
    let mut at = selected.min(last);
    if down {
        at = if at == last { 0 } else { at + 1 };
    }
    if up {
        at = if at == 0 { last } else { at - 1 };
    }
    if page_down {
        at = (at + 8).min(last);
    }
    if page_up {
        at = at.saturating_sub(8);
    }
    at
}

/// Draws the palette when open; returns the effect the user chose.
pub fn show(ctx: &Context, state: &mut PaletteState, entries: &[Entry], db_ready: bool) -> Option<Effect> {
    if !state.open {
        return None;
    }
    let keys = read_keys(ctx);
    if keys.escape {
        state.close();
        return None;
    }
    let search = search_entry(&state.query, db_ready);
    let count = hits(&state.query, entries, search.as_ref()).len();
    let moved = keys.up || keys.down || keys.page_up || keys.page_down;
    state.selected = move_selection(state.selected, count, keys.up, keys.down, keys.page_up, keys.page_down);

    let mut chosen: Option<(usize, bool)> = (keys.enter || keys.ctrl_enter).then_some((state.selected, keys.ctrl_enter));
    let width = WIDTH.min(ctx.content_rect().width() - 32.0).max(260.0);
    let id = Id::new("ame_command_palette");
    let area = Modal::default_area(id).anchor(Align2::CENTER_TOP, vec2(0.0, 70.0));
    let before = state.query.clone();
    let response = Modal::new(id).area(area).show(ctx, |ui| {
        ui.set_width(width);
        let keep_arrows = EventFilter { horizontal_arrows: true, vertical_arrows: true, ..Default::default() };
        let edit = ui.add(TextEdit::singleline(&mut state.query).hint_text("Type a command, a place or text to search for").desired_width(f32::INFINITY).event_filter(keep_arrows));
        if !edit.has_focus() {
            edit.request_focus();
        }
        if state.query != before {
            state.selected = 0;
            state.notice = None;
        }
        let found = hits(&state.query, entries, search.as_ref());
        ui.add_space(4.0);
        ScrollArea::vertical().max_height(LIST_HEIGHT).auto_shrink([false, true]).show(ui, |ui| {
            if found.is_empty() {
                ui.weak("Nothing matches. Type text to search the library.");
            }
            for (i, hit) in found.iter().enumerate() {
                let response = row(ui, hit, i == state.selected);
                if i == state.selected && moved {
                    response.scroll_to_me(Some(Align::Center));
                }
                if response.hovered() && ui.input(|inp| inp.pointer.delta() != vec2(0.0, 0.0)) {
                    state.selected = i;
                }
                if response.clicked() {
                    chosen = Some((i, ui.input(|inp| inp.modifiers.command)));
                }
            }
        });
        ui.add_space(2.0);
        footer(ui, state);
        found.len()
    });
    if response.backdrop_response.clicked() {
        state.close();
        return None;
    }
    let (index, new_tab) = chosen?;
    let found = hits(&state.query, entries, search.as_ref());
    let entry = found.get(index)?.entry;
    if let Err(reason) = &entry.enabled {
        state.notice = Some(format!("{}: {reason}", entry.title));
        return None;
    }
    let effect = if new_tab { entry.effect.in_new_tab() } else { entry.effect.clone() };
    state.close();
    Some(effect)
}

fn footer(ui: &mut Ui, state: &PaletteState) {
    match &state.notice {
        Some(notice) => {
            let tone = crate::theme::warn(ui);
            icons::label(ui, svg::WARNING, eframe::egui::RichText::new(notice).color(tone));
        }
        None => {
            ui.label(eframe::egui::RichText::new("Up and Down select. Enter runs. Ctrl+Enter opens in a new tab. Esc closes.").small().weak());
        }
    }
}

fn row(ui: &mut Ui, hit: &Hit, selected: bool) -> eframe::egui::Response {
    let tokens = crate::theme::tokens(ui.ctx());
    let metrics = crate::theme::metrics(ui.ctx());
    let entry = hit.entry;
    let enabled = entry.enabled.is_ok();
    let height = (metrics.row_h + 4.0).max(ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }
    if selected {
        ui.painter().rect_filled(rect, tokens.radius_sm, tokens.primary_soft);
    } else if response.hovered() {
        ui.painter().rect_filled(rect, tokens.radius_sm, tokens.hover);
    }
    let ink = if enabled { tokens.text } else { tokens.text_faint };
    let icon_rect = Rect::from_center_size(pos2(rect.left() + ROW_PAD + metrics.icon_md / 2.0, rect.center().y), vec2(metrics.icon_md, metrics.icon_md));
    crate::ui::titlebar::kit_icon(entry.icon).paint(ui, icon_rect, metrics.icon_md, if selected && enabled { tokens.primary } else { tokens.icon });

    let right_text = match (&entry.shortcut, enabled) {
        (Some(shortcut), true) => format!("{}  {}", shortcut, entry.kind.label()),
        _ => entry.kind.label().to_string(),
    };
    let small = TextStyle::Small.resolve(ui.style());
    let right = ui.painter().layout_no_wrap(right_text, small.clone(), tokens.text_faint);
    let right_x = rect.right() - ROW_PAD - right.size().x;
    ui.painter().galley(pos2(right_x, rect.center().y - right.size().y / 2.0), right, tokens.text_faint);

    let text_left = icon_rect.right() + ROW_PAD;
    let room = (right_x - ROW_PAD - text_left).max(20.0);
    let galley = ui.painter().layout_job(title_job(ui, &entry.title, &hit.positions, ink, tokens.primary, room));
    let title_w = galley.size().x;
    ui.painter().galley(pos2(text_left, rect.center().y - galley.size().y / 2.0), galley, ink);

    let detail = match &entry.enabled {
        Err(reason) => reason.clone(),
        Ok(()) => entry.detail.clone(),
    };
    let detail_left = text_left + title_w + ROW_PAD;
    if !detail.is_empty() && right_x - detail_left > 30.0 {
        let mut job = LayoutJob::simple_singleline(detail, small, tokens.text_faint);
        job.wrap.max_width = right_x - detail_left - ROW_PAD;
        job.wrap.max_rows = 1;
        job.wrap.break_anywhere = true;
        job.wrap.overflow_character = Some('…');
        let g = ui.painter().layout_job(job);
        ui.painter().galley(pos2(detail_left, rect.center().y - g.size().y / 2.0), g, tokens.text_faint);
    }
    response
}

/// The title as one row of text with the matched characters in the accent color.
fn title_job(ui: &Ui, title: &str, positions: &[usize], ink: Color32, accent: Color32, max_width: f32) -> LayoutJob {
    let font = TextStyle::Body.resolve(ui.style());
    let mut job = LayoutJob::default();
    let mut run = String::new();
    let mut run_hot = false;
    let mut flush = |job: &mut LayoutJob, run: &mut String, hot: bool| {
        if !run.is_empty() {
            job.append(run, 0.0, TextFormat { font_id: font.clone(), color: if hot { accent } else { ink }, ..Default::default() });
            run.clear();
        }
    };
    for (i, c) in title.chars().enumerate() {
        let hot = positions.binary_search(&i).is_ok();
        if hot != run_hot {
            flush(&mut job, &mut run, run_hot);
            run_hot = hot;
        }
        run.push(c);
    }
    flush(&mut job, &mut run, run_hot);
    job.wrap.max_width = max_width;
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    job.wrap.overflow_character = Some('…');
    job
}
