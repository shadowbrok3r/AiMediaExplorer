//! Drawing of the Refine tab: the generator controls, the filters, the bulk bar and the cards.

use std::time::Duration;

use eframe::egui::*;

use super::model::{FieldChange, FieldValue, Proposal, ProposalKey, SortKey, TagMark, clipped, source_labels, tag_diff};
use super::{Confirm, PAGE, RefinementsPanel, RunState};
use crate::ai::refine::RerankerType;
use crate::icons::{self, svg};
use crate::ui::file_table::thumbs::{ScopeId, ThumbService, ThumbView, Tier};

/// Width below which a card stacks the picture above the details.
const NARROW: f32 = 600.0;
const THUMB: f32 = 96.0;
/// Width of the column with the checkbox and the name of a field.
const LABEL_COL: f32 = 112.0;
const COL_GAP: f32 = 8.0;
const TEXT_CLIP: usize = 360;

/// What the user did on a card.
#[derive(Clone, Debug, PartialEq)]
pub enum CardAction {
    Accept(ProposalKey),
    Reject(ProposalKey),
}

struct CardEnv<'a> {
    thumbs: &'a ThumbService,
    scope: ScopeId,
    narrow: bool,
}

/// "1 file" or "3 files".
pub fn count_noun(n: usize, noun: &str) -> String {
    if n == 1 { format!("1 {noun}") } else { format!("{n} {noun}s") }
}

fn confidence_text(c: f32) -> String {
    format!("{:.0}%", (c * 100.0).clamp(0.0, 100.0))
}

fn confidence_color(ui: &Ui, c: f32) -> Color32 {
    if c >= 0.66 {
        crate::theme::success(ui)
    } else if c >= 0.33 {
        crate::theme::warn(ui)
    } else {
        ui.visuals().weak_text_color()
    }
}

fn tag_line(ui: &mut Ui, tags: &[(String, TagMark)]) {
    if tags.is_empty() {
        ui.weak("(none)");
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        for (tag, mark) in tags {
            let text = match mark {
                TagMark::Kept => RichText::new(tag).weak(),
                TagMark::Added => RichText::new(format!("+{tag}")).color(crate::theme::success(ui)),
                TagMark::Removed => RichText::new(format!("-{tag}")).color(crate::theme::error(ui)).strikethrough(),
            };
            ui.label(text);
        }
    });
}

fn text_cell(ui: &mut Ui, value: &FieldValue, strong: bool) {
    let shown = value.plain();
    let rich = if matches!(value, FieldValue::Text(None)) || shown == "(none)" { RichText::new(shown.clone()).weak() } else if strong { RichText::new(clipped(&shown, TEXT_CLIP)) } else { RichText::new(clipped(&shown, TEXT_CLIP)).weak() };
    let response = ui.add(Label::new(rich).wrap());
    if shown.chars().count() > TEXT_CLIP {
        response.on_hover_text(shown);
    }
}

/// A column of fixed width whose content wraps inside it.
fn cell(ui: &mut Ui, width: f32, add: impl FnOnce(&mut Ui)) {
    ui.allocate_ui_with_layout(vec2(width, 0.0), Layout::top_down(Align::Min), |ui| {
        ui.set_width(width);
        add(ui);
    });
}

/// Widths of the two value columns next to the label column in `total`.
pub fn value_width(total: f32) -> f32 {
    ((total - LABEL_COL - 2.0 * COL_GAP) / 2.0).max(80.0)
}

fn field_row(ui: &mut Ui, change: &mut FieldChange, value_w: f32) {
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = COL_GAP;
        cell(ui, LABEL_COL, |ui| {
            ui.spacing_mut().interact_size.y = ui.text_style_height(&TextStyle::Body);
            ui.checkbox(&mut change.accepted, RichText::new(change.kind.label()).strong());
        });
        match (&change.current, &change.proposed) {
            (FieldValue::Tags(now), FieldValue::Tags(next)) => {
                let (now_marks, next_marks) = tag_diff(now, next);
                cell(ui, value_w, |ui| tag_line(ui, &now_marks));
                cell(ui, value_w, |ui| tag_line(ui, &next_marks));
            }
            (now, next) => {
                cell(ui, value_w, |ui| text_cell(ui, now, false));
                cell(ui, value_w, |ui| text_cell(ui, next, true));
            }
        }
    });
}

fn thumbnail(ui: &mut Ui, env: &CardEnv<'_>, path: &str, name: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(THUMB, THUMB), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(4), ui.visuals().extreme_bg_color);
    let view = match env.thumbs.request(env.scope, path, Tier::Thumb, 0.0) {
        ThumbView::Ready { id, size } => {
            let scale = (THUMB / size.x.max(1.0)).min(THUMB / size.y.max(1.0));
            let fitted = size * scale;
            painter.image(id, Rect::from_center_size(rect.center(), fitted), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            true
        }
        ThumbView::Pending => {
            ui.ctx().request_repaint_after(Duration::from_millis(150));
            false
        }
        ThumbView::Missing => false,
    };
    if !view {
        let icon = icons::file_icon_svg(name, false).image(40.0);
        ui.put(Rect::from_center_size(rect.center(), vec2(40.0, 40.0)), icon);
    }
    painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color), StrokeKind::Inside);
}

fn header_line(ui: &mut Ui, p: &Proposal) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(p.file_name()).strong());
        let folder = p.path.strip_suffix(p.file_name()).unwrap_or("").trim_end_matches(['/', '\\']);
        if !folder.is_empty() {
            ui.label(RichText::new(clipped(folder, 70)).weak().small()).on_hover_text(&p.path);
        }
    });
    ui.horizontal_wrapped(|ui| {
        let label = p.source.label();
        ui.label(RichText::new(format!(" {label} ")).small().background_color(ui.visuals().widgets.inactive.weak_bg_fill));
        match p.confidence {
            Some(c) => {
                ui.label(RichText::new(format!("Confidence {}", confidence_text(c))).small().color(confidence_color(ui, c)));
            }
            None => {
                ui.label(RichText::new("No confidence").small().weak());
            }
        }
        if let Some(reason) = &p.reason {
            ui.label(RichText::new(clipped(reason, 140)).small().italics()).on_hover_text(reason);
        }
    });
}

fn card(ui: &mut Ui, p: &mut Proposal, env: &CardEnv<'_>, applying: bool) -> Option<CardAction> {
    let mut action = None;
    Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let body = |ui: &mut Ui, p: &mut Proposal, action: &mut Option<CardAction>| {
            header_line(ui, p);
            ui.add_space(2.0);
            if p.changes.is_empty() {
                ui.weak("The file already has these values; reject to dismiss the proposal.");
            } else {
                let value_w = value_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = COL_GAP;
                    cell(ui, LABEL_COL, |_| {});
                    cell(ui, value_w, |ui| {
                        ui.label(RichText::new("Current").small().weak());
                    });
                    cell(ui, value_w, |ui| {
                        ui.label(RichText::new("Proposed").small().weak());
                    });
                });
                for change in &mut p.changes {
                    field_row(ui, change, value_w);
                }
            }
            ui.horizontal(|ui| {
                let n = p.accepted_count();
                let label = if n == 1 { "Accept 1 field".to_string() } else { format!("Accept {n} fields") };
                let enabled = n > 0 && !applying;
                if ui.add_enabled(enabled, icons::button_widget(ui, svg::CHECK, label)).on_hover_text(if n == 0 { "Check at least one field" } else { "Write the checked fields to the library" }).clicked() {
                    *action = Some(CardAction::Accept(p.key.clone()));
                }
                if ui.add_enabled(!applying, icons::button_widget(ui, svg::X, "Reject")).on_hover_text("Drop this proposal without writing anything").clicked() {
                    *action = Some(CardAction::Reject(p.key.clone()));
                }
                if applying {
                    ui.spinner();
                }
                if let Some(e) = &p.error {
                    ui.label(RichText::new(clipped(e, 120)).color(crate::theme::error(ui))).on_hover_text(e);
                }
            });
        };
        if env.narrow {
            thumbnail(ui, env, &p.path, p.file_name());
            body(ui, p, &mut action);
        } else {
            ui.horizontal_top(|ui| {
                thumbnail(ui, env, &p.path, p.file_name());
                ui.vertical(|ui| body(ui, p, &mut action));
            });
        }
    });
    action
}

impl RefinementsPanel {
    /// Draws the tab.
    pub fn ui(&mut self, ui: &mut Ui) {
        self.take_requests();
        self.pump();
        self.refresh_store_if_stale();
        self.thumbs.poll(ui.ctx());
        if self.is_running() || self.store_status.loading || !self.applying.is_empty() {
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }
        self.header(ui);
        ui.separator();
        self.filters(ui);
        self.bulk_bar(ui);
        self.confirm_bar(ui);
        ui.separator();
        self.list(ui);
    }

    fn header(&mut self, ui: &mut Ui) {
        icons::heading(ui, svg::AME_REFINE, "AI Refinements");
        ui.label(RichText::new("Review proposed categories, tags, captions and descriptions. Check the fields to keep, then accept; every accept can be undone with Ctrl+Z.").weak());
        if self.scope_len() > 0 {
            ui.horizontal(|ui| {
                ui.colored_label(crate::theme::warn(ui), format!("Selection scope: {} files", self.scope_len()));
                if icons::button(ui, svg::X_CIRCLE, "Clear scope").clicked() {
                    self.clear_scope();
                }
            });
        }
        CollapsingHeader::new("Generator").default_open(true).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ComboBox::from_id_salt("refine-reranker").selected_text(match self.reranker_choice {
                    RerankerType::Heuristic => "Heuristic",
                    RerankerType::JinaM0 => "Jina M0",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.reranker_choice, RerankerType::Heuristic, "Heuristic");
                    ui.selectable_value(&mut self.reranker_choice, RerankerType::JinaM0, "Jina M0");
                });
                if ui.checkbox(&mut self.use_cloud_model, "Use a cloud model").changed() && self.use_cloud_model {
                    self.use_zeroclaw = false;
                }
                if self.use_cloud_model {
                    ComboBox::from_id_salt("refine-cloud-provider").selected_text(self.cloud_provider.clone()).show_ui(ui, |ui| {
                        for name in ["openai", "openrouter", "groq", "gemini", "custom"] {
                            ui.selectable_value(&mut self.cloud_provider, name.to_string(), name);
                        }
                    });
                    ui.add(TextEdit::singleline(&mut self.cloud_model_name).hint_text("Model (blank: the default)").desired_width(180.0));
                }
            });
            self.zeroclaw_source(ui);
            ui.horizontal_wrapped(|ui| {
                if self.is_running() {
                    if icons::button(ui, svg::STOP, "Cancel").on_hover_text("Stop generating; proposals made so far stay").clicked() {
                        self.cancel_run();
                    }
                } else if icons::button(ui, svg::AME_PROPOSAL, "Generate proposals").clicked() {
                    self.start_run();
                }
                if icons::button(ui, svg::BROOM, "Clear generated").on_hover_text("Drop the proposals made in this session; stored proposals stay").clicked() {
                    self.clear_local();
                }
                self.run_status(ui);
            });
            let note = if self.use_zeroclaw { "ZeroClaw runs leave the local models alone and send one request per batch of files." } else { "Generating stops the vision worker (and unloads CLIP for Jina M0) to free memory; cloud runs leave the local models alone." };
            ui.label(RichText::new(note).small().weak());
        });
        ui.horizontal_wrapped(|ui| {
            icons::label(ui, svg::AME_MCP, "Stored proposals");
            let status = &self.store_status;
            match &status.error {
                Some(e) => {
                    ui.label(RichText::new(clipped(e, 100)).color(crate::theme::error(ui))).on_hover_text(e);
                }
                None => {
                    ui.weak(if status.at.is_some() { format!("{} pending from tools", status.pending) } else { "not read yet".to_string() });
                }
            }
            if self.store_status.loading {
                ui.spinner();
            }
            if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh").clicked() {
                self.store_status.at = None;
                self.refresh_store();
            }
        });
    }

    /// The ZeroClaw source: a switch that works only while the gateway is Online, its route hint and what is sent.
    fn zeroclaw_source(&mut self, ui: &mut Ui) {
        let jobs = self.zc_jobs();
        let reason = jobs.availability().reason().map(str::to_string);
        ui.horizontal_wrapped(|ui| {
            let switch = ui.add_enabled(reason.is_none(), Checkbox::new(&mut self.use_zeroclaw, "Use ZeroClaw"));
            match &reason {
                Some(why) => {
                    switch.on_disabled_hover_text(why.as_str());
                    ui.label(RichText::new(format!("ZeroClaw is unavailable: {}", clipped(why, 110))).small().color(crate::theme::warn(ui))).on_hover_text(why.as_str());
                }
                None => {
                    if self.use_zeroclaw {
                        self.use_cloud_model = false;
                        ui.add(TextEdit::singleline(&mut self.zc_hint).hint_text(format!("Hint ({})", jobs.default_hint())).desired_width(110.0));
                        ui.label(RichText::new("Text only: file names, folders and metadata are sent in batches; no pictures.").small().weak());
                    }
                }
            }
        });
    }

    fn run_status(&mut self, ui: &mut Ui) {
        match &self.run {
            RunState::Idle => {}
            RunState::Running { done, total } => {
                if *total > 0 {
                    ui.add(ProgressBar::new((*done as f32 / *total as f32).clamp(0.0, 1.0)).desired_width(180.0).text(format!("{done} of {total}")));
                } else {
                    ui.spinner();
                    ui.weak("Starting...");
                }
            }
            RunState::Done { scanned, produced, cancelled } => {
                let text = if *cancelled { format!("Stopped: {produced} proposals from {scanned} files") } else { format!("{produced} proposals from {scanned} files") };
                ui.weak(text);
            }
            RunState::Failed(e) => {
                ui.label(RichText::new(clipped(e, 140)).color(crate::theme::error(ui))).on_hover_text(e.as_str());
            }
        }
    }

    fn filters(&mut self, ui: &mut Ui) {
        let labels = source_labels(&self.proposals);
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.filter.text).hint_text("Filter by path, tag, category or text").desired_width(240.0));
            let selected = self.filter.source.clone().unwrap_or_else(|| "All sources".to_string());
            ComboBox::from_id_salt("refine-source").selected_text(selected).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.filter.source, None, "All sources");
                for label in &labels {
                    ui.selectable_value(&mut self.filter.source, Some(label.clone()), label);
                }
            });
            ComboBox::from_id_salt("refine-sort").selected_text(format!("Sort: {}", self.filter.sort.label())).show_ui(ui, |ui| {
                for key in SortKey::ALL {
                    ui.selectable_value(&mut self.filter.sort, key, key.label());
                }
            });
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(Slider::new(&mut self.filter.min_confidence, 0.0..=1.0).custom_formatter(|v, _| confidence_text(v as f32)).text("Min confidence"));
            if self.filter.min_confidence > 0.0 {
                ui.checkbox(&mut self.filter.include_unrated, "Keep unrated");
            }
            if self.filter.is_narrowing() && icons::button(ui, svg::X, "Reset filters").clicked() {
                self.filter = super::ListFilter { sort: self.filter.sort, ..Default::default() };
            }
        });
    }

    fn bulk_bar(&mut self, ui: &mut Ui) {
        let shown = self.visible();
        let plan = super::model::plan_accept(&self.proposals, &shown);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("Showing {} of {} proposals", shown.len(), self.proposals.len())).strong());
            if icons::button(ui, svg::LIST_CHECKS, "Check all shown").clicked() {
                super::model::set_all(&mut self.proposals, &shown, true);
            }
            if icons::button(ui, svg::MINUS_CIRCLE, "Uncheck all shown").clicked() {
                super::model::set_all(&mut self.proposals, &shown, false);
            }
            let label = if plan.is_empty() { "Accept checked in view".to_string() } else { format!("Accept {} in {}", count_noun(plan.fields, "field"), count_noun(plan.files(), "file")) };
            let hint = if plan.is_empty() { "Nothing is checked in the shown proposals".to_string() } else { format!("{}. Unchecked fields and unchecked proposals are left alone.", plan.summary()) };
            if ui.add_enabled(!plan.is_empty() && self.confirm.is_none(), icons::button_widget(ui, svg::CHECKS, label)).on_hover_text(hint).clicked() {
                self.ask_accept_view();
            }
            if ui.add_enabled(!shown.is_empty() && self.confirm.is_none(), icons::button_widget(ui, svg::PROHIBIT, "Reject shown")).on_hover_text("Drop every shown proposal").clicked() {
                self.ask_reject_view();
            }
        });
        let tally = self.tally;
        if tally.applied_files + tally.rejected + tally.failed > 0 {
            ui.label(RichText::new(format!("This session: {} in {} applied, {} rejected, {} failed", count_noun(tally.applied_fields, "field"), count_noun(tally.applied_files, "file"), tally.rejected, tally.failed)).small().weak());
        }
        if let Some((ok, text)) = &self.notice {
            let color = if *ok { crate::theme::success(ui) } else { crate::theme::warn(ui) };
            ui.label(RichText::new(text.clone()).color(color));
        }
    }

    fn confirm_bar(&mut self, ui: &mut Ui) {
        let Some(confirm) = self.confirm.clone() else { return };
        let mut answer = None;
        Frame::group(ui.style()).fill(ui.visuals().faint_bg_color).show(ui, |ui| {
            match &confirm {
                Confirm::Accept(plan) => {
                    ui.label(RichText::new(format!("Apply {}?", plan.summary())).strong());
                    ui.label(RichText::new(if plan.unchecked > 0 { format!("{} shown proposals have nothing checked and are left alone. The changes can be undone with Ctrl+Z.", plan.unchecked) } else { "The changes can be undone with Ctrl+Z.".to_string() }).small().weak());
                }
                Confirm::Reject(keys) => {
                    ui.label(RichText::new(format!("Reject {} proposals?", keys.len())).strong());
                    ui.label(RichText::new("Nothing is written to the library; the proposals are dropped.").small().weak());
                }
            }
            ui.horizontal(|ui| {
                if icons::button(ui, svg::CHECK, "Confirm").clicked() {
                    answer = Some(true);
                }
                if icons::button(ui, svg::X, "Cancel").clicked() {
                    answer = Some(false);
                }
            });
        });
        match answer {
            Some(true) => self.confirm_pending(),
            Some(false) => self.confirm = None,
            None => {}
        }
    }

    fn list(&mut self, ui: &mut Ui) {
        let shown = self.visible();
        if self.proposals.is_empty() {
            ui.add_space(6.0);
            let text = match &self.run {
                RunState::Running { .. } => "Generating proposals...",
                _ => "No proposals yet. Generate some, or let a tool file them through the MCP server.",
            };
            ui.weak(text);
            return;
        }
        if shown.is_empty() {
            ui.weak("No proposal matches the filters.");
            return;
        }
        let thumbs = self.thumbs.clone();
        let env = CardEnv { thumbs: &thumbs, scope: self.thumb_scope, narrow: ui.available_width() < NARROW };
        let mut action = None;
        let limit = self.shown.max(PAGE);
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for &i in shown.iter().take(limit) {
                let applying = self.applying.contains(&self.proposals[i].key);
                if let Some(a) = card(ui, &mut self.proposals[i], &env, applying) {
                    action = Some(a);
                }
            }
            if shown.len() > limit && icons::button(ui, svg::ARROW_DOWN, format!("Show {} more", (shown.len() - limit).min(PAGE))).clicked() {
                self.shown = limit + PAGE;
            }
        });
        match action {
            Some(CardAction::Accept(key)) => self.accept_one(&key),
            Some(CardAction::Reject(key)) => self.reject_one(&key),
            None => {}
        }
    }
}
