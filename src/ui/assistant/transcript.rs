//! Drawing the transcript: message bubbles, collapsed reasoning and tool rows, approval cards, plans, notices and the usage ring.

use std::collections::{HashMap, HashSet};

use base64::Engine as _;
use crossbeam::channel::{Receiver, Sender};
use egui::{Align, Color32, Context, FontId, Frame, Layout, Margin, RichText, ScrollArea, Sense, Stroke, TextEdit, Ui, vec2};
use egui_extras::syntax_highlighting::{CodeTheme, highlight};
use zeroclaw_client::ApprovalDecision;

use super::approvals;
use super::entry::{CallState, Decision, Entry, Kind, Level, NoticeAction, Transcript, tool_name};
use crate::icons::{self, svg};

/// A message longer than this many characters or lines starts collapsed.
const LONG_CHARS: usize = 1200;
const LONG_LINES: usize = 20;
const PREVIEW_LINES: usize = 10;
/// Tool output shown before it is cut.
const RESULT_CHARS: usize = 4000;

/// What a click in the transcript asks the panel to do.
#[derive(Debug, PartialEq)]
pub enum Action {
    Decide { request_id: String, decision: Decision },
    Notice(NoticeAction),
    LoadEarlier,
    Thumb(String),
}

/// Textures of attached pictures, loaded from the library's stored thumbnails.
pub struct Thumbs {
    cache: HashMap<String, egui::TextureHandle>,
    pending: HashSet<String>,
    tx: Sender<(String, Option<(Vec<u8>, u32, u32)>)>,
    rx: Receiver<(String, Option<(Vec<u8>, u32, u32)>)>,
}

impl Default for Thumbs {
    fn default() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self { cache: HashMap::new(), pending: HashSet::new(), tx, rx }
    }
}

impl Thumbs {
    pub fn get(&self, path: &str) -> Option<&egui::TextureHandle> {
        self.cache.get(path)
    }

    /// Starts loading the stored thumbnail of `path` once.
    pub fn request(&mut self, path: &str) {
        if self.cache.contains_key(path) || !self.pending.insert(path.to_string()) {
            return;
        }
        let (tx, path) = (self.tx.clone(), path.to_string());
        tokio::spawn(async move {
            let decoded = match crate::Thumbnail::get_thumbnail_by_path(&path).await {
                Ok(Some(t)) => t.thumbnail_b64.and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok()).and_then(|bytes| image::load_from_memory(&bytes).ok()).map(|img| {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    (rgba.into_raw(), w, h)
                }),
                _ => None,
            };
            let _ = tx.try_send((path, decoded));
        });
    }

    pub fn poll(&mut self, ctx: &Context) {
        while let Ok((path, decoded)) = self.rx.try_recv() {
            self.pending.remove(&path);
            if let Some((rgba, w, h)) = decoded {
                let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                let tex = ctx.load_texture(format!("assistant_attach:{path}"), color, egui::TextureOptions::LINEAR);
                self.cache.insert(path, tex);
                ctx.request_repaint();
            }
        }
    }
}

/// Which collapsed rows the person opened.
#[derive(Default)]
pub struct View {
    pub expanded: HashSet<String>,
    pub thumbs: Thumbs,
}

impl View {
    fn toggle(&mut self, key: &str) {
        if !self.expanded.remove(key) {
            self.expanded.insert(key.to_string());
        }
    }
}

/// A piece of message text: prose or a fenced code block.
#[derive(Debug, PartialEq, Eq)]
pub enum Seg {
    Text(String),
    Code { lang: String, code: String },
}

/// Splits fenced code blocks (```lang ... ```) from the prose around them.
pub fn split_blocks(text: &str) -> Vec<Seg> {
    let mut segs = Vec::new();
    let (mut in_code, mut lang, mut buf) = (false, String::new(), String::new());
    for line in text.lines() {
        let trimmed = line.trim_end_matches('\r');
        if trimmed.starts_with("```") {
            if in_code {
                segs.push(Seg::Code { lang: std::mem::take(&mut lang), code: buf.trim_end().to_string() });
                buf.clear();
                in_code = false;
            } else {
                if !buf.is_empty() {
                    segs.push(Seg::Text(std::mem::take(&mut buf)));
                }
                lang = trimmed.trim_start_matches('`').trim().to_string();
                in_code = true;
            }
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    if !buf.is_empty() {
        segs.push(if in_code { Seg::Code { lang, code: buf.trim_end().to_string() } } else { Seg::Text(buf) });
    }
    segs
}

pub fn is_long(text: &str) -> bool {
    text.len() > LONG_CHARS || text.lines().count() > LONG_LINES
}

/// The first lines of `text` with an ellipsis when more follow.
pub fn preview(text: &str) -> String {
    let mut out: Vec<&str> = text.lines().take(PREVIEW_LINES).collect();
    if text.lines().count() > PREVIEW_LINES {
        out.push("...");
    }
    out.join("\n")
}

fn message_body(ui: &mut Ui, text: &str) {
    for seg in split_blocks(text) {
        match seg {
            Seg::Text(s) => {
                let s = s.trim();
                if !s.is_empty() {
                    ui.label(s);
                }
            }
            Seg::Code { lang, code } => {
                ui.horizontal(|ui| {
                    if icons::button(ui, svg::COPY, "Copy").clicked() {
                        ui.ctx().copy_text(code.clone());
                    }
                    ui.label(RichText::new(if lang.is_empty() { "code".to_string() } else { lang.clone() }).weak());
                });
                Frame::group(ui.style()).show(ui, |ui| {
                    let theme = CodeTheme::from_style(ui.style().as_ref());
                    let style = ui.style().clone();
                    let ctx = ui.ctx().clone();
                    let lang_name = lang.clone();
                    let mut layouter = move |ui: &Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                        let mut job = highlight(&ctx, &style, &theme, &lang_name, text.as_str());
                        job.wrap.max_width = wrap_width;
                        ui.fonts_mut(|f| f.layout_job(job))
                    };
                    let rows = (code.lines().count().max(1) as f32).min(18.0) as usize;
                    let mut shown = code.clone();
                    ui.add(TextEdit::multiline(&mut shown).desired_rows(rows).desired_width(ui.available_width()).font(FontId::monospace(13.0)).interactive(false).layouter(&mut layouter));
                });
            }
        }
    }
}

fn bubble(ui: &mut Ui, user: bool, entry: &Entry, view: &mut View, out: &mut Vec<Action>) {
    let text = entry.text().unwrap_or("");
    let long = is_long(text);
    let collapsed = long && !view.expanded.contains(&entry.key);
    let fill = if user { ui.visuals().widgets.active.bg_fill.gamma_multiply(0.5) } else { ui.visuals().faint_bg_color };
    let radius = 8;
    Frame::new().fill(fill).stroke(ui.visuals().widgets.noninteractive.bg_stroke).corner_radius(radius).inner_margin(Margin::symmetric(8, 6)).show(ui, |ui| {
        ui.set_max_width((ui.available_width()).max(40.0));
        ui.horizontal_wrapped(|ui| {
            ui.strong(if user { "You" } else { "Assistant" });
            if let Some(stamp) = clock(entry.at_ms) {
                ui.label(RichText::new(stamp).weak().small());
            }
            if icons::icon_button(ui, svg::COPY, "Copy").clicked() {
                ui.ctx().copy_text(text.to_string());
            }
            if long {
                let (icon, tip) = if collapsed { (svg::CARET_DOWN, "Expand") } else { (svg::CARET_UP, "Collapse") };
                if icons::icon_button(ui, icon, tip).clicked() {
                    view.toggle(&entry.key);
                }
            }
        });
        if let Kind::User { attachments, chips, .. } = &entry.kind {
            if !chips.is_empty() {
                ui.label(RichText::new(chips.join("  ")).weak().small());
            }
            if !attachments.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for p in attachments {
                        match view.thumbs.get(p) {
                            Some(tex) => {
                                ui.image((tex.id(), vec2(40.0, 40.0))).on_hover_text(p);
                            }
                            None => {
                                let name = std::path::Path::new(p).file_name().and_then(|s| s.to_str()).unwrap_or(p);
                                icons::label(ui, svg::PAPERCLIP, RichText::new(name).small()).on_hover_text(p);
                                out.push(Action::Thumb(p.clone()));
                            }
                        }
                    }
                });
            }
        }
        if let Kind::Assistant { streaming: true, text } = &entry.kind {
            if text.is_empty() {
                ui.label(RichText::new("...").weak());
            }
        }
        if collapsed {
            ui.label(preview(text));
        } else {
            message_body(ui, text);
        }
    });
}

fn clock(at_ms: u64) -> Option<String> {
    if at_ms == 0 {
        return None;
    }
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64).map(|t| t.with_timezone(&chrono::Local).format("%H:%M").to_string())
}

/// A one-line header that opens and closes a body.
fn fold(ui: &mut Ui, view: &mut View, key: &str, icon: icons::Icon, label: String, tint: Option<Color32>, body: impl FnOnce(&mut Ui)) {
    let open = view.expanded.contains(key);
    let caret = if open { svg::CARET_DOWN } else { svg::CARET_RIGHT };
    ui.horizontal(|ui| {
        if icons::icon_button(ui, caret, if open { "Collapse" } else { "Expand" }).clicked() {
            view.toggle(key);
        }
        let text = RichText::new(label).weak();
        approvals::wrap_label(ui, icon, if let Some(c) = tint { text.color(c) } else { text });
    });
    if open {
        Frame::new().inner_margin(Margin { left: 20, ..Default::default() }).show(ui, |ui| body(ui));
    }
}

fn entry_row(ui: &mut Ui, e: &Entry, view: &mut View, now_ms: u64, out: &mut Vec<Action>) {
    match &e.kind {
        Kind::User { .. } => {
            ui.with_layout(Layout::top_down(Align::Max), |ui| bubble(ui, true, e, view, out));
        }
        Kind::Assistant { .. } => bubble(ui, false, e, view, out),
        Kind::Reasoning { text, streaming } => {
            let label = if *streaming { "Thinking...".to_string() } else { format!("Thought ({} characters)", text.chars().count()) };
            fold(ui, view, &e.key, svg::BRAIN, label, None, |ui| {
                ui.label(RichText::new(text).italics().weak());
            });
        }
        Kind::ToolCall { name, args, state, .. } => {
            let status = if *state == CallState::Running { "running" } else { "done" };
            fold(ui, view, &e.key, svg::WRENCH, format!("{} ({status})", tool_name(name)), None, |ui| {
                let pretty = approvals::pretty_args(args, 2000);
                ui.label(RichText::new(if pretty.is_empty() { "no arguments".to_string() } else { pretty }).monospace().small());
            });
        }
        Kind::ToolResult { name, output, .. } => {
            fold(ui, view, &e.key, svg::CHECK_CIRCLE, format!("{} returned {} characters", tool_name(name), output.chars().count()), None, |ui| {
                let shown: String = output.chars().take(RESULT_CHARS).collect();
                ui.label(RichText::new(shown).monospace().small());
                if output.chars().count() > RESULT_CHARS {
                    ui.label(RichText::new("The rest is not shown.").weak());
                }
            });
        }
        Kind::Approval { request_id, tool, summary, args, deadline_ms, decision } => {
            if let Some(answer) = approvals::card(ui, &e.key, tool, summary, args, *deadline_ms, *decision, now_ms) {
                out.push(Action::Decide { request_id: request_id.clone(), decision: answer });
            }
            if *decision == Decision::Pending {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
            }
        }
        Kind::Plan { rows } => {
            Frame::new().stroke(Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)).corner_radius(6).inner_margin(Margin::same(6)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                icons::label(ui, svg::LIST_CHECKS, RichText::new("Plan").strong());
                for r in rows {
                    let done = matches!(r.status.as_deref(), Some("completed" | "done"));
                    let mark = if done { "[x]" } else { "[ ]" };
                    ui.label(RichText::new(format!("{mark} {}", r.content)).weak());
                }
            });
        }
        Kind::Notice { level, text, action } => {
            let color = match level {
                Level::Info => ui.visuals().weak_text_color(),
                Level::Warn => ui.visuals().warn_fg_color,
                Level::Error => ui.visuals().error_fg_color,
            };
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(text).color(color));
                if let Some(a) = action {
                    let (label, icon) = match a {
                        NoticeAction::RePair => ("Re-pair", svg::LINK),
                        NoticeAction::OpenSettings => ("Settings", svg::GEAR),
                        NoticeAction::Retry => ("Retry", svg::ARROW_CLOCKWISE),
                        NoticeAction::ReloadHistory => ("Reload", svg::ARROW_CLOCKWISE),
                    };
                    if icons::small_button(ui, icon, label).clicked() {
                        out.push(Action::Notice(*a));
                    }
                }
            });
        }
    }
}

/// Draws the whole transcript in a scroll area that follows the newest entry.
pub fn show(ui: &mut Ui, t: &Transcript, view: &mut View, now_ms: u64) -> Vec<Action> {
    let mut out = Vec::new();
    view.thumbs.poll(ui.ctx());
    ScrollArea::vertical().id_salt("assistant_transcript").stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
        ui.set_width(ui.available_width());
        if t.entries.is_empty() {
            ui.add_space(16.0);
            ui.label(RichText::new("Ask about your library. The chips below share what you are looking at as text. Type / for quick actions.").weak());
        }
        if t.has_more && ui.small_button("Load earlier messages").clicked() {
            out.push(Action::LoadEarlier);
        }
        for e in &t.entries {
            entry_row(ui, e, view, now_ms, &mut out);
            ui.add_space(6.0);
        }
    });
    out
}

/// Maps a click on an approval card to the wire answer, for tests and callers that only deal in protocol values.
pub fn wire(d: Decision) -> Option<ApprovalDecision> {
    approvals::decision_to_wire(d)
}

/// A ring that fills with the share of the context window in use.
pub fn usage_ring(ui: &mut Ui, fraction: Option<f32>, tip: &str) {
    let edge = ui.text_style_height(&egui::TextStyle::Body);
    let (rect, response) = ui.allocate_exact_size(vec2(edge, edge), Sense::hover());
    let painter = ui.painter();
    let center = rect.center();
    let radius = edge * 0.38;
    painter.circle_stroke(center, radius, Stroke::new(2.0, ui.visuals().widgets.noninteractive.bg_stroke.color));
    if let Some(f) = fraction.filter(|f| *f > 0.0) {
        let color = if f > 0.9 { ui.visuals().error_fg_color } else if f > 0.7 { ui.visuals().warn_fg_color } else { crate::theme::primary(ui) };
        let steps = ((f * 48.0).ceil() as usize).max(2);
        let points: Vec<egui::Pos2> = (0..=steps)
            .map(|i| {
                let angle = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * f * (i as f32 / steps as f32);
                center + radius * vec2(angle.cos(), angle.sin())
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(2.5, color)));
    }
    response.on_hover_text(tip);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_and_fenced_code_are_split() {
        let segs = split_blocks("Intro line\n```rust\nfn main() {}\n```\nafter");
        assert_eq!(
            segs,
            [Seg::Text("Intro line\n".into()), Seg::Code { lang: "rust".into(), code: "fn main() {}".into() }, Seg::Text("after\n".into())]
        );
    }

    #[test]
    fn an_unclosed_fence_keeps_the_rest_as_code() {
        assert_eq!(split_blocks("```\nlet x = 1;\nlet y = 2;"), [Seg::Code { lang: String::new(), code: "let x = 1;\nlet y = 2;".into() }]);
        assert!(split_blocks("").is_empty());
    }

    #[test]
    fn long_messages_are_detected_and_previewed() {
        assert!(!is_long("short"));
        assert!(is_long(&"x".repeat(LONG_CHARS + 1)));
        assert!(is_long(&"a\n".repeat(LONG_LINES + 1)));
        let text = (0..30).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let p = preview(&text);
        assert_eq!(p.lines().count(), PREVIEW_LINES + 1);
        assert!(p.ends_with("..."));
        assert_eq!(preview("one\ntwo"), "one\ntwo");
    }

    #[test]
    fn a_stamp_of_zero_is_not_shown() {
        assert_eq!(clock(0), None);
        assert!(clock(1_700_000_000_000).is_some());
    }

    #[test]
    fn answers_map_to_the_protocol() {
        assert_eq!(wire(Decision::Always), Some(ApprovalDecision::Always));
    }
}
