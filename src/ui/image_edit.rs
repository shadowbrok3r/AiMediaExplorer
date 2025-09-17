use eframe::egui::*;
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
            model_repo: String::new(),
            model_loaded: false,
            loading_model: false,
            model_error: None,
            pipeline: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
}

impl ImageEditPanel {
    pub fn open_with_path(&mut self, path: &str) {
        self.current_path = Some(path.to_string());
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        ui.heading("Image Edit");
        ui.separator();
        if let Some(path) = &self.current_path {
            // Model controls
            ui.horizontal(|ui| {
                ui.label("Model repo:");
                let hint = if self.model_repo.trim().is_empty() { "e.g. Qwen/your-image-edit-repo" } else { "" };
                ui.add_sized([ui.available_width()*0.7, 20.0], TextEdit::singleline(&mut self.model_repo).hint_text(hint));
                let can_load = !self.loading_model && !self.model_repo.trim().is_empty();
                if ui.add_enabled(can_load, Button::new(if self.model_loaded { "Reload Model" } else { "Load Model" })).clicked() {
                    self.loading_model = true;
                    self.model_error = None;
                    let repo = self.model_repo.clone();
                    let slot = self.pipeline.clone();
                    tokio::spawn(async move {
                        crate::ai::unload_heavy_models_except("").await;
                        let prefer = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                        match crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer) {
                            Ok(p) => { let mut g = slot.lock().await; *g = Some(p); }
                            Err(e) => { log::error!("[ImageEdit] model load failed: {e}"); let mut g = slot.lock().await; *g = None; }
                        }
                    });
                }
            });
            if self.loading_model {
                ui.horizontal(|ui| { ui.label("Loading model…"); Spinner::new().ui(ui); });
                // Poll once per UI frame for loaded status
                if let Ok(g) = self.pipeline.try_lock() {
                    if g.is_some() {
                        self.loading_model = false;
                        self.model_loaded = true;
                    }
                }
            } else if let Some(err) = &self.model_error { ui.colored_label(Color32::RED, err); }

            ui.horizontal(|ui| {
                ui.label(RichText::new("Source:").underline().strong());
                ui.monospace(path);
            });
            ui.add_space(6.0);

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

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Prompt:");
                ui.add_sized([ui.available_width()*0.7, 20.0], TextEdit::singleline(&mut self.prompt));
            });
            ui.horizontal(|ui| {
                ui.label("Negative:");
                ui.add_sized([ui.available_width()*0.7, 20.0], TextEdit::singleline(&mut self.negative_prompt));
            });
            ui.add(Slider::new(&mut self.guidance_scale, 1.0..=12.0).text("Guidance"));
            ui.add(Slider::new(&mut self.strength, 0.0..=1.0).text("Strength"));
            ui.add(Slider::new(&mut self.steps, 1..=75).text("Steps"));

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let can_run = self.model_loaded && !self.loading_model && self.current_path.is_some();
                if ui.add_enabled(can_run, Button::new("Run Edit")).clicked() {
                    // Run the real edit pipeline and capture output PNG bytes
                    let slot = self.pipeline.clone();
                    let pth = self.current_path.clone().unwrap_or_default();
                    let cache = self.thumb_cache.clone();
                    let opts = crate::ai::qwen_image_edit::EditOptions {
                        prompt: self.prompt.clone(),
                        negative_prompt: if self.negative_prompt.trim().is_empty() { None } else { Some(self.negative_prompt.clone()) },
                        guidance_scale: self.guidance_scale,
                        num_inference_steps: self.steps,
                        strength: self.strength,
                        scheduler: Some("flow_match_euler".into()),
                        seed: None,
                    };
                    tokio::spawn(async move {
                        let png = if let Some(pipe) = slot.lock().await.as_ref() {
                            match pipe.run_edit(std::path::Path::new(&pth), &opts) { Ok(b) => b, Err(e) => { log::error!("[ImageEdit] run_edit failed: {e}"); Vec::new() } }
                        } else { Vec::new() };
                        if !png.is_empty() {
                            let key = format!("imageedit::result::{}", &pth);
                            if let Ok(mut guard) = cache.lock() { guard.insert(key, Arc::from(png.into_boxed_slice())); }
                        }
                    });
                }
                if ui.button("Open Source Folder").clicked() {
                    if let Some(pb) = std::path::Path::new(path).parent() { let _ = open::that(pb); }
                }
                if ui.button("Open Source File").clicked() { let _ = open::that(path); }
            });
        } else {
            ui.weak("No image selected.");
        }
    }
}
