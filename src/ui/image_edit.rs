use eframe::egui::*;
use crate::ui::status::GlobalStatusIndicator;
use base64::Engine as _;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct ImageEditPanel {
    pub current_path: Option<String>,
    pub prompt: String,
    pub negative_prompt: String,
    pub guidance_scale: f32,
    pub strength: f32,
    pub steps: usize,
    // Local preview cache for this panel
    thumb_cache: Arc<Mutex<HashMap<String, Arc<[u8]>>>>,
    // Model controls/state
    pub model_repo: String,
    pub model_loaded: bool,
    pub loading_model: bool,
    pub model_error: Option<String>,
    pipeline: std::sync::Arc<tokio::sync::Mutex<Option<crate::ai::qwen_image_edit::model::QwenImageEditPipeline>>>,
    // Loader status channel (background -> UI)
    status_tx: crossbeam::channel::Sender<Result<(), String>>,
    status_rx: crossbeam::channel::Receiver<Result<(), String>>,
}

impl Default for ImageEditPanel {
    fn default() -> Self {
        Self {
            current_path: None,
            prompt: String::new(),
            negative_prompt: String::new(),
            guidance_scale: 7.5,
            strength: 0.7,
            steps: 30,
            thumb_cache: Arc::new(Mutex::new(HashMap::new())),
            model_repo: "Qwen/Qwen-Image-Edit".to_string(),
            model_loaded: false,
            loading_model: false,
            model_error: None,
            pipeline: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            // initialize a channel; will be replaced in new() if needed
            status_tx: {
                let (tx, _rx) = crossbeam::channel::unbounded(); tx
            },
            status_rx: {
                let (_tx, rx) = crossbeam::channel::unbounded(); rx
            },
        }
    }
}

impl ImageEditPanel {
    pub fn open_with_path(&mut self, path: &str) {
        self.current_path = Some(path.to_string());
        // Auto-load model as soon as user opens Image Edit with a path.
        if !self.loading_model && !self.model_loaded && !self.model_repo.trim().is_empty() {
            self.loading_model = true;
            self.model_error = None;
            crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Loading model");
            let repo = self.model_repo.clone();
            let slot = self.pipeline.clone();
            // fresh channel for this load
            let (tx, rx) = crossbeam::channel::unbounded();
            self.status_tx = tx.clone();
            self.status_rx = rx;
            tokio::spawn(async move {
                crate::ai::unload_heavy_models_except("").await;
                let prefer = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                let res = crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer)
                    .map_err(|e| e.to_string());
                match res {
                    Ok(p) => {
                        let dev = if matches!(p.device, candle_core::Device::Cuda(_)) { crate::ui::status::DeviceKind::GPU } else { crate::ui::status::DeviceKind::CPU };
                        crate::ui::status::QWEN_EDIT_STATUS.set_device(dev);
                        crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
                        let mut g = slot.lock().await; *g = Some(p); let _ = tx.send(Ok(()));
                    }
                    Err(e) => {
                        log::error!("[ImageEdit] model load failed: {e}");
                        crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed: {e}"));
                        let mut g = slot.lock().await; *g = None; let _ = tx.send(Err(e));
                    }
                }
            });
        }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        if let Some(path) = &self.current_path {
            // If we have a generated result, show it side-by-side or below
            let result_key = format!("imageedit::result::{}", path);
            let has_result = &mut false;
            // Model controls
            ui.horizontal(|ui| {
                ui.label(RichText::new("Source:").underline().strong());
                ui.monospace(path);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let can_load = !self.loading_model && !self.model_repo.trim().is_empty();
                    if self.loading_model {
                        Spinner::new().color(ui.style().visuals.error_fg_color).ui(ui); 
                        ui.label("Loading model…");
                        // Poll loader status channel for completion or error
                        if let Ok(msg) = self.status_rx.try_recv() {
                            self.loading_model = false;
                            match msg {
                                Ok(()) => { self.model_loaded = true; self.model_error = None; }
                                Err(e) => { self.model_loaded = false; self.model_error = Some(e); }
                            }
                        }
                    } else if let Some(err) = &self.model_error { ui.colored_label(Color32::RED, err); }

                    if ui.add_enabled(can_load, Button::new(if self.model_loaded { "Reload Model" } else { "Load Model" })).clicked() {
                        self.loading_model = true;
                        self.model_error = None;
                        crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Loading model");
                        let repo = self.model_repo.clone();
                        let slot = self.pipeline.clone();
                        // create a fresh channel for this load
                        let (tx, rx) = crossbeam::channel::unbounded();
                        self.status_tx = tx.clone();
                        self.status_rx = rx;
                        tokio::spawn(async move {
                            crate::ai::unload_heavy_models_except("").await;
                            let prefer = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                            let res = crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer)
                                .map_err(|e| e.to_string());
                            match res {
                                Ok(p) => {
                                    let dev = if matches!(p.device, candle_core::Device::Cuda(_)) { crate::ui::status::DeviceKind::GPU } else { crate::ui::status::DeviceKind::CPU };
                                    crate::ui::status::QWEN_EDIT_STATUS.set_device(dev);
                                    crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
                                    let mut g = slot.lock().await; *g = Some(p); let _ = tx.send(Ok(()));
                                }
                                Err(e) => {
                                    log::error!("[ImageEdit] model load failed: {e}");
                                    crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed: {e}"));
                                    let mut g = slot.lock().await; *g = None; let _ = tx.send(Err(e));
                                }
                            }
                        });
                    }
                    let hint = if self.model_repo.trim().is_empty() { "e.g. Qwen/Qwen-Image-Edit" } else { "" };
                    TextEdit::singleline(&mut self.model_repo).hint_text(hint).desired_width(250.).ui(ui);
                    ui.label("Model repo:");
                });
            });
            ui.separator();
            ui.add_space(6.0);

            ui.label("Prompt:");
            TextEdit::singleline(&mut self.prompt).desired_width(400.).ui(ui);
            ui.label("Negative:");
            TextEdit::singleline(&mut self.negative_prompt).desired_width(400.).ui(ui);

            Slider::new(&mut self.guidance_scale, 1.0..=12.0).text("Guidance").ui(ui);
            Slider::new(&mut self.strength, 0.0..=1.0).text("Strength").ui(ui);
            Slider::new(&mut self.steps, 1..=75).text("Steps").ui(ui);
            
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let can_run = self.model_loaded && !self.loading_model && self.current_path.is_some();
                if ui.add_enabled(can_run, Button::new("Run Edit")).clicked() {
                    // Run the real edit pipeline and capture output PNG bytes
                    let slot = self.pipeline.clone();
                    let pth = self.current_path.clone().unwrap_or_default();
                    let cache = self.thumb_cache.clone();
                    crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Running, "Editing image");
                    let opts = crate::ai::qwen_image_edit::EditOptions {
                        prompt: self.prompt.clone(),
                        negative_prompt: if self.negative_prompt.trim().is_empty() { None } else { Some(self.negative_prompt.clone()) },
                        guidance_scale: self.guidance_scale,
                        num_inference_steps: self.steps,
                        strength: self.strength,
                        scheduler: Some("flow_match_euler".into()),
                        seed: None,
                        deterministic_vae: true,
                    };
                    tokio::spawn(async move {
                        // Release other heavy models for headroom
                        crate::ai::unload_heavy_models_except("").await;
                        let png = if let Some(pipe) = slot.lock().await.as_ref() {
                            match pipe.run_edit(std::path::Path::new(&pth), &opts) { Ok(b) => b, Err(e) => { log::error!("[ImageEdit] run_edit failed: {e}"); Vec::new() } }
                        } else { Vec::new() };
                        if !png.is_empty() {
                            let key = format!("imageedit::result::{}", &pth);
                            if let Ok(mut guard) = cache.lock() { guard.insert(key, Arc::from(png.into_boxed_slice())); }
                        }
                        crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                    });
                }
                if *has_result && ui.button("Save Result As…").clicked() {
                    // Offer a save dialog and write PNG bytes from cache
                    if let Some(file) = rfd::FileDialog::new()
                        .set_title("Save edited image as PNG")
                        .add_filter("PNG", &["png"]) 
                        .set_file_name("edited.png")
                        .save_file() {
                        if let Ok(cache) = self.thumb_cache.lock() {
                            if let Some(bytes_arc) = cache.get(&result_key) {
                                let bytes: Vec<u8> = bytes_arc.as_ref().to_vec();
                                let _ = std::fs::write(&file, bytes);
                            }
                        }
                    }
                }
                if ui.button("Open Source Folder").clicked() {
                    if let Some(pb) = std::path::Path::new(path).parent() { let _ = open::that(pb); }
                }
                if ui.button("Open Source File").clicked() { let _ = open::that(path); }
            });
            ui.separator();
            ui.columns(2, |ui| {
                ui[0].vertical_centered(|ui| {
                    // Preview the source image (high-res preview)
                    let cache_key = format!("imageedit::preview::{},{}", path, 1600);
                    let mut has_img = false;
                    if let Ok(cache) = self.thumb_cache.lock() {
                        if let Some(bytes_arc) = cache.get(&cache_key) {
                            has_img = true;
                            let img_src = eframe::egui::ImageSource::Bytes {
                                uri: std::borrow::Cow::from(format!("bytes://{}", cache_key)),
                                bytes: eframe::egui::load::Bytes::Shared(bytes_arc.clone()),
                            };
                            eframe::egui::Image::new(img_src)
                                .max_size(ui.available_size()/1.3)
                                .ui(ui);
                        }
                    }
                    if !has_img {
                        let cache = self.thumb_cache.clone();
                        let path_buf = std::path::PathBuf::from(path);
                        let cache_key_task = cache_key.clone();
                        tokio::spawn(async move {
                            let png = crate::utilities::thumbs::generate_image_preview_png(&path_buf, 1600)
                                .or_else(|e| {
                                    log::debug!("[ImageEdit] preview fallback: {}", e);
                                    crate::utilities::thumbs::generate_image_thumb_data(&path_buf)
                                        .and_then(|data_url| {
                                            let (_, b64) = data_url.split_once("data:image/png;base64,").unwrap_or(("", &data_url));
                                            base64::engine::general_purpose::STANDARD
                                                .decode(b64.as_bytes())
                                                .map_err(|e| e.to_string())
                                        })
                                })
                                .unwrap_or_default();
                            if !png.is_empty() {
                                if let Ok(mut guard) = cache.lock() {
                                    guard.insert(cache_key_task, Arc::from(png.into_boxed_slice()));
                                }
                            }
                        });
                        ui.weak("Loading preview…");
                    }

                });
                ui[1].vertical_centered(|ui| {

                    if let Ok(cache) = self.thumb_cache.lock() {
                        if let Some(bytes_arc) = cache.get(&result_key) {
                            *has_result = true;
                            ui.separator();
                            ui.label(RichText::new("Result:").underline().strong());
                            let img_src = eframe::egui::ImageSource::Bytes {
                                uri: std::borrow::Cow::from(format!("bytes://{}", result_key)),
                                bytes: eframe::egui::load::Bytes::Shared(bytes_arc.clone()),
                            };
                            eframe::egui::Image::new(img_src)
                                .max_size(ui.available_size()/1.3)
                                .ui(ui);
                        }
                    }
                });
            });
        } else {
            ui.weak("No image selected.");
        }
    }
}
