//! Drawing of the Image Edit tab: source, prompt, model settings, progress, compare view, save actions and history.

use std::time::{Duration, SystemTime};

use eframe::egui::*;

use super::model::{CompareMode, ErrorKind, PRESETS, Phase, file_name};
use super::{EditEntry, ImageEditPanel, save};
use crate::command::{AppCommand, Target, Tool};
use crate::icons::{self, svg};
use crate::ui::file_table::thumbs::ThumbView;

const SOURCE_THUMB: f32 = 72.0;
const HISTORY_THUMB: f32 = 56.0;

/// What the user chose among the result actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultAction {
    SaveNextTo(u64),
    SaveAs(u64),
    Replace(u64),
    AddToLibrary(u64),
    Show(u64),
    Remove(u64),
}

/// The rect inside `rect` that holds an image of `size` at its own aspect ratio.
pub fn contained(rect: Rect, size: Vec2) -> Rect {
    if size.x <= 0.0 || size.y <= 0.0 {
        return rect;
    }
    let scale = (rect.width() / size.x).min(rect.height() / size.y);
    Rect::from_center_size(rect.center(), size * scale)
}

fn paint_texture(painter: &Painter, rect: Rect, id: TextureId, size: Vec2) -> Rect {
    let target = contained(rect, size);
    painter.image(id, target, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    target
}

fn frame_rect(ui: &Ui, painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, CornerRadius::same(4), ui.visuals().extreme_bg_color);
}

fn outline(ui: &Ui, painter: &Painter, rect: Rect) {
    painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color), StrokeKind::Inside);
}

/// A label on a picture with a dark backing so it reads on any image.
fn caption(ui: &Ui, painter: &Painter, rect: Rect, text: &str, right: bool) {
    let galley = painter.layout_no_wrap(text.to_string(), TextStyle::Small.resolve(ui.style()), Color32::WHITE);
    let size = galley.size();
    let pos = if right { rect.right_top() + vec2(-8.0 - size.x, 6.0) } else { rect.left_top() + vec2(8.0, 6.0) };
    painter.rect_filled(Rect::from_min_size(pos - vec2(4.0, 2.0), size + vec2(8.0, 4.0)), CornerRadius::same(3), Color32::from_black_alpha(150));
    painter.galley(pos, galley, Color32::WHITE);
}

/// What stands in for the original while it has no picture.
fn source_hint(view: ThumbView) -> &'static str {
    if view == ThumbView::Missing { "No preview of the original" } else { "Loading preview..." }
}

fn time_label(at: SystemTime) -> String {
    chrono::DateTime::<chrono::Local>::from(at).format("%H:%M:%S").to_string()
}

impl ImageEditPanel {
    /// Draws the tab.
    pub fn ui(&mut self, ui: &mut Ui) {
        self.pump(ui.ctx());
        self.thumbs.poll(ui.ctx());
        if self.phase.is_busy() || self.gguf.fetching || self.model_busy {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            icons::heading(ui, svg::AME_IMAGE_EDIT, "Image Edit");
            self.source_section(ui);
            if self.current_path.is_none() {
                return;
            }
            ui.separator();
            self.prompt_section(ui);
            self.model_section(ui);
            self.run_section(ui);
            self.error_section(ui);
            ui.separator();
            self.compare_section(ui);
            self.history_section(ui);
        });
        self.replace_window(ui.ctx());
    }

    fn source_section(&mut self, ui: &mut Ui) {
        let Some(path) = self.current_path.clone() else {
            ui.add_space(6.0);
            ui.label("No image selected. Choose one, or press Edit in the Inspector with a picture in focus.");
            if icons::button(ui, svg::FILE_IMAGE, "Choose image...").clicked() {
                self.choose_image(ui.ctx());
            }
            return;
        };
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(SOURCE_THUMB, SOURCE_THUMB), Sense::hover());
            let painter = ui.painter_at(rect);
            frame_rect(ui, &painter, rect);
            if let ThumbView::Ready { id, size } = self.source_view() {
                paint_texture(&painter, rect, id, size);
            }
            outline(ui, &painter, rect);
            ui.vertical(|ui| {
                ui.label(RichText::new(file_name(&path)).strong());
                ui.label(RichText::new(&path).weak().small());
                ui.horizontal_wrapped(|ui| {
                    if icons::button(ui, svg::FILE_IMAGE, "Choose image...").clicked() {
                        self.choose_image(ui.ctx());
                    }
                    if icons::button(ui, svg::AME_REVEAL_IN_EXPLORER, "Open folder").clicked() {
                        if let Some(dir) = std::path::Path::new(&path).parent() {
                            let _ = open::that(dir);
                        }
                    }
                    if icons::button(ui, svg::ARROW_SQUARE_OUT, "Open file").clicked() {
                        let _ = open::that(&path);
                    }
                });
            });
        });
    }

    fn choose_image(&self, ctx: &Context) {
        crate::ui::dialogs::pick_file(ctx, "Choose an image to edit", |file| {
            crate::command::send(AppCommand::OpenTool { tool: Tool::ImageEdit, path: Some(file.display().to_string()), target: Target::Current });
        });
    }

    fn prompt_section(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        icons::label(ui, svg::CHAT_TEXT, RichText::new("Prompt").strong());
        ui.horizontal_wrapped(|ui| {
            let append = ui.input(|i| i.modifiers.shift);
            for (label, text) in PRESETS {
                if ui.small_button(label).on_hover_text(format!("{text}\nShift+click adds it to the prompt")).clicked() {
                    self.params.prompt = self.params.with_preset(text, append);
                }
            }
        });
        ui.add(TextEdit::multiline(&mut self.params.prompt).desired_rows(2).desired_width(f32::INFINITY).hint_text("What should change? For example: make the sky a deep sunset orange"));
        ui.horizontal(|ui| {
            icons::label(ui, svg::PROHIBIT, "Avoid");
            ui.add(TextEdit::singleline(&mut self.params.negative_prompt).desired_width(f32::INFINITY).hint_text("Things the edit should not add"));
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(Slider::new(&mut self.params.guidance_scale, 1.0..=12.0).text("Guidance"));
            ui.add(Slider::new(&mut self.params.strength, 0.0..=1.0).text("Strength"));
            ui.add(Slider::new(&mut self.params.steps, 1..=75).text("Steps"));
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.params.prefer_full_precision, "Full precision (BF16/FP32)").on_hover_text("Falls back to a lower precision by itself when the GPU runs out of memory");
            ui.label("Preview every");
            ui.add(DragValue::new(&mut self.params.preview_every_n).range(0..=10).suffix(" steps"));
            ui.weak("(0 = off)");
        });
    }

    fn model_section(&mut self, ui: &mut Ui) {
        CollapsingHeader::new("Model settings").default_open(false).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                icons::label(ui, svg::AME_MODEL, "Model repo");
                ui.add(TextEdit::singleline(&mut self.settings.model_repo).desired_width(260.0).hint_text("e.g. Qwen/Qwen-Image-Edit"));
                let loaded = self.backend.is_loaded();
                let label = if loaded { "Reload" } else { "Load" };
                let enabled = !self.model_busy && !self.phase.is_busy() && !self.settings.model_repo.trim().is_empty();
                if ui.add_enabled(enabled, icons::button_widget(ui, if loaded { svg::ARROWS_CLOCKWISE } else { svg::AME_MODEL }, label)).clicked() {
                    self.load_model();
                }
                if loaded && ui.add_enabled(!self.phase.is_busy(), icons::button_widget(ui, svg::X_CIRCLE, "Unload")).on_hover_text("Free the memory the model uses").clicked() {
                    self.unload_model();
                }
                if self.model_busy {
                    ui.spinner();
                }
                ui.weak(if loaded { "Model loaded" } else { "Model not loaded; the first edit loads it" });
            });
            ui.horizontal_wrapped(|ui| {
                icons::label(ui, svg::CLOUD_ARROW_DOWN, "GGUF");
                ui.add(TextEdit::singleline(&mut self.settings.gguf_repo).desired_width(240.0).hint_text("repo"));
                ui.add(TextEdit::singleline(&mut self.settings.gguf_file).desired_width(200.0).hint_text("file"));
                let can_fetch = !self.gguf.fetching && !self.settings.gguf_repo.trim().is_empty() && !self.settings.gguf_file.trim().is_empty();
                if ui.add_enabled(can_fetch, icons::button_widget(ui, svg::CLOUD_ARROW_DOWN, "Fetch")).clicked() {
                    self.fetch_gguf();
                }
                ui.checkbox(&mut self.settings.gguf_prefer, "Use it when running");
                if self.gguf.fetching {
                    ui.spinner();
                }
                if let Some(status) = &self.gguf.status {
                    ui.label(RichText::new(status).small());
                }
            });
            if let Some(e) = &self.model_error {
                ui.label(RichText::new(&e.message).color(crate::theme::error(ui)));
                ui.label(RichText::new(e.advice()).small().weak());
            }
            if self.debug_backend {
                ui.label(RichText::new("The test backend is on: edits tint or gray the picture in a moment and no model is used.").small().color(crate::theme::warn(ui)));
            }
            ui.label(RichText::new("Running an edit frees the memory of the vision and CLIP models first.").small().weak());
        });
    }

    fn run_section(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if self.phase.is_busy() {
                if ui.add_enabled(self.phase != Phase::Cancelling, icons::button_widget(ui, svg::STOP, "Cancel")).on_hover_text("Stop at the next step; the task list has the same button").clicked() {
                    self.cancel();
                }
            } else {
                let blocked = self.blocked();
                let button = icons::button_widget(ui, svg::AME_IMAGE_EDIT, "Run edit");
                if ui.add_enabled(blocked.is_none(), button).on_disabled_hover_text(blocked.unwrap_or("")).clicked() {
                    if let Err(e) = self.start() {
                        self.note(false, e);
                    }
                }
            }
            match self.phase.fraction() {
                Some(f) => {
                    ui.add(ProgressBar::new(f).desired_width(220.0).text(self.phase.label()));
                }
                None if self.phase.is_busy() => {
                    ui.spinner();
                    ui.label(self.phase.label());
                }
                None => {
                    ui.weak(self.phase.label());
                }
            }
        });
        if let Some((ok, text)) = &self.notice {
            let color = if *ok { crate::theme::success(ui) } else { crate::theme::warn(ui) };
            ui.label(RichText::new(text).color(color));
        }
    }

    fn error_section(&mut self, ui: &mut Ui) {
        let Phase::Failed(error) = self.phase.clone() else { return };
        let mut retry = false;
        let mut retry_lower = false;
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label(RichText::new(format!("The edit failed: {}", error.message)).color(crate::theme::error(ui)));
            ui.label(error.advice());
            ui.horizontal(|ui| {
                if icons::button(ui, svg::ARROW_CLOCKWISE, "Try again").clicked() {
                    retry = true;
                }
                if error.may_retry_lower_precision() && icons::button(ui, svg::SLIDERS, "Retry at lower precision").on_hover_text("Turns off full precision and uses at most 20 steps").clicked() {
                    retry_lower = true;
                }
                if error.kind == ErrorKind::Source && icons::button(ui, svg::FILE_IMAGE, "Choose another image...").clicked() {
                    self.choose_image(ui.ctx());
                }
            });
        });
        if retry_lower {
            if let Err(e) = self.retry_lower_precision() {
                self.note(false, e);
            }
        } else if retry {
            self.phase = Phase::Idle;
            if let Err(e) = self.start() {
                self.note(false, e);
            }
        }
    }

    fn compare_section(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            icons::label(ui, svg::AME_COMPARE, RichText::new("Compare").strong());
            for mode in CompareMode::ALL {
                ui.selectable_value(&mut self.compare, mode, mode.label());
            }
        });
        let source = self.source_view();
        let entry = self.shown_entry().map(|e| (e.id, e.texture.id(), e.texture.size_vec2(), e.full, e.saved.clone(), e.in_library));
        let live = self.live.as_ref().map(|t| (t.id(), t.size_vec2()));
        let width = ui.available_width();
        let height = (width * 0.5).clamp(240.0, 460.0);
        let result = match (&entry, live) {
            (_, Some((id, size))) if self.phase.is_busy() => Some((id, size, "Live preview")),
            (Some((_, id, size, _, _, _)), _) => Some((*id, *size, "Result")),
            _ => None,
        };
        let src = match source {
            ThumbView::Ready { id, size } => Some((id, size)),
            _ => None,
        };
        match self.compare {
            CompareMode::SideBySide => {
                ui.columns(2, |cols| {
                    for (i, col) in cols.iter_mut().enumerate() {
                        let (rect, _) = col.allocate_exact_size(vec2(col.available_width(), height), Sense::hover());
                        let painter = col.painter_at(rect);
                        frame_rect(col, &painter, rect);
                        let (pic, label) = if i == 0 { (src, "Original") } else { (result.map(|(id, size, _)| (id, size)), result.map_or("Result", |r| r.2)) };
                        match pic {
                            Some((id, size)) => {
                                paint_texture(&painter, rect, id, size);
                            }
                            None => {
                                painter.text(rect.center(), Align2::CENTER_CENTER, if i == 0 { source_hint(source) } else if self.phase.is_busy() { "Working..." } else { "No result yet" }, TextStyle::Body.resolve(col.style()), col.visuals().weak_text_color());
                            }
                        }
                        caption(col, &painter, rect, label, false);
                        outline(col, &painter, rect);
                    }
                });
            }
            CompareMode::Swipe | CompareMode::Toggle => {
                let (rect, response) = ui.allocate_exact_size(vec2(width, height), Sense::click_and_drag());
                let painter = ui.painter_at(rect);
                frame_rect(ui, &painter, rect);
                let swipe_mode = self.compare == CompareMode::Swipe;
                if swipe_mode {
                    if let Some(pos) = response.interact_pointer_pos().filter(|_| response.dragged() || response.clicked()) {
                        self.swipe = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                    }
                } else if response.clicked() {
                    self.flipped = !self.flipped;
                }
                let show_source_only = !swipe_mode && !self.flipped && result.is_none();
                match (src, result) {
                    (Some((sid, ssize)), Some((rid, rsize, rlabel))) => {
                        if swipe_mode {
                            paint_texture(&painter, rect, rid, rsize);
                            let cut = rect.left() + rect.width() * self.swipe;
                            let left = painter.with_clip_rect(Rect::from_min_max(rect.min, pos2(cut, rect.max.y)));
                            paint_texture(&left, rect, sid, ssize);
                            painter.line_segment([pos2(cut, rect.top()), pos2(cut, rect.bottom())], Stroke::new(2.0, Color32::WHITE));
                            caption(ui, &painter, rect, "Original", false);
                            caption(ui, &painter, rect, rlabel, true);
                        } else if self.flipped {
                            paint_texture(&painter, rect, sid, ssize);
                            caption(ui, &painter, rect, "Original (click to flip)", false);
                        } else {
                            paint_texture(&painter, rect, rid, rsize);
                            caption(ui, &painter, rect, &format!("{rlabel} (click to flip)"), false);
                        }
                    }
                    (Some((sid, ssize)), None) => {
                        paint_texture(&painter, rect, sid, ssize);
                        caption(ui, &painter, rect, if show_source_only { "Original" } else { "Original; no result yet" }, false);
                    }
                    (None, Some((rid, rsize, rlabel))) => {
                        paint_texture(&painter, rect, rid, rsize);
                        caption(ui, &painter, rect, rlabel, false);
                    }
                    (None, None) => {
                        painter.text(rect.center(), Align2::CENTER_CENTER, source_hint(source), TextStyle::Body.resolve(ui.style()), ui.visuals().weak_text_color());
                    }
                }
                outline(ui, &painter, rect);
            }
        }
        if let Some((id, _, _, full, saved, in_library)) = entry {
            let action = self.result_actions(ui, id, full, saved, in_library);
            self.apply(action, ui.ctx());
        }
    }

    fn result_actions(&mut self, ui: &mut Ui, id: u64, full: [u32; 2], saved: Option<String>, in_library: bool) -> Option<ResultAction> {
        let mut action = None;
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.weak(format!("{} x {} px", full[0], full[1]));
            if icons::button(ui, svg::FLOPPY_DISK, "Save as new file...").on_hover_text("Choose where to write the edited image; the original is not touched").clicked() {
                action = Some(ResultAction::SaveAs(id));
            }
            if icons::button(ui, svg::COPY, "Save next to original").on_hover_text("Writes name-edited.png in the folder of the original").clicked() {
                action = Some(ResultAction::SaveNextTo(id));
            }
            if crate::ui::danger_icon_button(ui, svg::WARNING, "Replace original...").on_hover_text("Overwrite the original file; asks first").clicked() {
                action = Some(ResultAction::Replace(id));
            }
        });
        ui.horizontal_wrapped(|ui| {
            self.load_collections_if_needed();
            icons::label(ui, svg::AME_LOGICAL_GROUP, "Collection");
            let shown = self.collection_pick.clone().unwrap_or_else(|| "None".to_string());
            ComboBox::from_id_salt("image-edit-collection").selected_text(shown).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.collection_pick, None, "None");
                for name in self.collections.clone() {
                    ui.selectable_value(&mut self.collection_pick, Some(name.clone()), name);
                }
            });
            ui.add(TextEdit::singleline(&mut self.new_collection).desired_width(140.0).hint_text("or a new one"));
            let label = if in_library { "Add to library again" } else { "Add to library" };
            if icons::button(ui, svg::AME_DATABASE, label).on_hover_text("Saves the result next to the original when it has no file yet, then adds it to the library").clicked() {
                action = Some(ResultAction::AddToLibrary(id));
            }
        });
        if let Some(path) = saved {
            ui.label(RichText::new(format!("Saved as {path}")).small().color(crate::theme::success(ui)));
        }
        action
    }

    fn apply(&mut self, action: Option<ResultAction>, ctx: &Context) {
        match action {
            Some(ResultAction::SaveNextTo(id)) => self.save_next_to_original(id),
            Some(ResultAction::Replace(id)) => self.ask_replace(id),
            Some(ResultAction::AddToLibrary(id)) => self.add_to_library(id),
            Some(ResultAction::Show(id)) => self.show_entry(id),
            Some(ResultAction::Remove(id)) => self.remove_entry(id),
            Some(ResultAction::SaveAs(id)) => {
                let Some(e) = self.history.iter().find(|e| e.id == id) else { return };
                let name = super::model::unique_output_path(&e.source, "png", |_| false).file_name().and_then(|n| n.to_str()).unwrap_or("edited.png").to_string();
                let (png, tx) = (e.png.clone(), self.tx.clone());
                crate::ui::dialogs::save_file(ctx, "Save the edited image", &name, Some(("Image", &["png", "jpg", "jpeg", "webp", "bmp"])), move |path| {
                    let result = save::write_result(&png, &path);
                    let _ = tx.send(super::job::Msg::Saved { id, result });
                    crate::command::request_repaint();
                });
            }
            None => {}
        }
    }

    fn history_section(&mut self, ui: &mut Ui) {
        if self.history.is_empty() {
            return;
        }
        ui.separator();
        ui.label(RichText::new(format!("History ({})", self.history.len())).strong());
        let shown = self.shown_entry().map(|e| e.id);
        let mut action = None;
        let rows: Vec<(u64, TextureId, Vec2, String, String, SystemTime, bool)> = self.history.iter().rev().map(|e: &EditEntry| (e.id, e.texture.id(), e.texture.size_vec2(), e.prompt.clone(), e.source.clone(), e.created, e.saved.is_some())).collect();
        for (id, tex, size, prompt, source, created, saved) in rows {
            ui.horizontal(|ui| {
                let (rect, response) = ui.allocate_exact_size(vec2(HISTORY_THUMB, HISTORY_THUMB), Sense::click());
                let painter = ui.painter_at(rect);
                frame_rect(ui, &painter, rect);
                paint_texture(&painter, rect, tex, size);
                if shown == Some(id) {
                    painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(2.0, crate::theme::primary(ui)), StrokeKind::Inside);
                }
                if response.clicked() {
                    action = Some(ResultAction::Show(id));
                }
                ui.vertical(|ui| {
                    ui.label(crate::ui::refine::model::clipped(&prompt, 90));
                    ui.label(RichText::new(format!("{} - {} - {}{}", file_name(&source), time_label(created), if saved { "saved" } else { "not saved" }, if shown == Some(id) { " - shown" } else { "" })).small().weak());
                    ui.horizontal(|ui| {
                        if ui.small_button("Show").clicked() {
                            action = Some(ResultAction::Show(id));
                        }
                        if ui.small_button("Remove").clicked() {
                            action = Some(ResultAction::Remove(id));
                        }
                    });
                });
            });
        }
        self.apply(action, ui.ctx());
    }

    fn replace_window(&mut self, ctx: &Context) {
        let Some((entry, source)) = self.replace_target() else { return };
        let (name, full) = (file_name(source).to_string(), source.to_string());
        let _ = entry;
        let mut answer = None;
        Window::new("Replace the original?").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).show(ctx, |ui| {
            ui.label(RichText::new(format!("This overwrites {name} with the edited image.")).strong());
            ui.label(RichText::new(&full).small().weak());
            ui.label("The original cannot be restored afterwards. Use Save as new file to keep it.");
            ui.horizontal(|ui| {
                if crate::ui::danger_icon_button(ui, svg::WARNING, format!("Replace {name}")).clicked() {
                    answer = Some(true);
                }
                if icons::button(ui, svg::X, "Keep the original").clicked() {
                    answer = Some(false);
                }
            });
        });
        match answer {
            Some(true) => self.confirm_replace(),
            Some(false) => self.cancel_replace(),
            None => {}
        }
    }
}
