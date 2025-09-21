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
    // GGUF controls/state
    gguf_repo: String,
    gguf_file: String,
    gguf_fetching: bool,
    gguf_status: Option<String>,
    gguf_resolved_path: Option<String>,
    gguf_tx: crossbeam::channel::Sender<Result<String, String>>,
    gguf_rx: crossbeam::channel::Receiver<Result<String, String>>,
    gguf_prefer: bool,
    // Precision controls
    prefer_full_precision: bool,
    // Live preview and cancel
    preview_every_n: usize,
    preview_rx: Option<crossbeam::channel::Receiver<Vec<u8>>>,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
            // GGUF defaults: QuantStack/Qwen-Image-Edit-GGUF and default file
            gguf_repo: "QuantStack/Qwen-Image-Edit-GGUF".to_string(),
            // Default to a GGUF build (Qwen_Image_Edit-Q6_K.gguf preferred with CUDA / quantized var builder)
            gguf_file: "Qwen_Image_Edit-Q6_K.gguf".to_string(),
            gguf_fetching: false,
            gguf_status: None,
            gguf_resolved_path: None,
            gguf_tx: {
                let (tx, _rx) = crossbeam::channel::unbounded(); tx
            },
            gguf_rx: {
                let (_tx, rx) = crossbeam::channel::unbounded(); rx
            },
            gguf_prefer: true,
            prefer_full_precision: true,
            preview_every_n: 0,
            preview_rx: None,
            cancel_flag: None,
        }
    }
}

impl ImageEditPanel {
    pub fn open_with_path(&mut self, path: &str) {
        self.current_path = Some(path.to_string());
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
                        let gguf_override = if self.gguf_prefer { self.gguf_resolved_path.clone() } else { None };
                        let prefer_full = self.prefer_full_precision;
                        tokio::spawn(async move {
                            crate::ai::unload_heavy_models_except("").await;
                            // Choose dtype according to user preference: full precision if requested
                            let prefer = if prefer_full {
                                // On CUDA prefer BF16 if available else F32. We’ll let the loader handle incompatibility.
                                if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::BF16 } else { candle_core::DType::F32 }
                            } else {
                                if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 }
                            };
                            let res = if let Some(p) = gguf_override.filter(|s| !s.is_empty()) {
                                crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf_with_overrides(
                                    &repo, prefer, Some(std::path::PathBuf::from(p)), None, None)
                            } else {
                                crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer)
                            }
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
                                    // If full precision requested, attempt a reduced-precision fallback load once
                                    if prefer_full && (e.to_ascii_lowercase().contains("cuda") || e.to_ascii_lowercase().contains("oom") || e.contains("named symbol not found")) {
                                        crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed at full precision: {}. Retrying at reduced precision…", e));
                                        let prefer_low = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                                        let res2 = crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer_low)
                                            .map_err(|e2| e2.to_string());
                                        match res2 {
                                            Ok(p2) => {
                                                let dev = if matches!(p2.device, candle_core::Device::Cuda(_)) { crate::ui::status::DeviceKind::GPU } else { crate::ui::status::DeviceKind::CPU };
                                                crate::ui::status::QWEN_EDIT_STATUS.set_device(dev);
                                                crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
                                                let mut g = slot.lock().await; *g = Some(p2); let _ = tx.send(Ok(()));
                                            }
                                            Err(e3) => {
                                                crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed even after reduced precision: {}", e3));
                                                let mut g = slot.lock().await; *g = None; let _ = tx.send(Err(e3));
                                            }
                                        }
                                    } else {
                                        crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed: {e}"));
                                        let mut g = slot.lock().await; *g = None; let _ = tx.send(Err(e));
                                    }
                                }
                            }
                        });
                    }
                    let hint = if self.model_repo.trim().is_empty() { "e.g. Qwen/Qwen-Image-Edit" } else { "" };
                    TextEdit::singleline(&mut self.model_repo).hint_text(hint).desired_width(250.).ui(ui);
                    ui.label("Model repo:");
                });
            });
            // GGUF controls row
            ui.horizontal(|ui| {
                ui.label(RichText::new("GGUF from repo:").underline());
                TextEdit::singleline(&mut self.gguf_repo)
                    .hint_text("e.g. QuantStack/Qwen-Image-Edit-GGUF")
                    .desired_width(260.)
                    .ui(ui);
                TextEdit::singleline(&mut self.gguf_file)
                    .hint_text("e.g. Qwen_Image_Edit-Q2_K.gguf")
                    .desired_width(220.)
                    .ui(ui);
                let can_fetch = !self.gguf_fetching && !self.gguf_repo.trim().is_empty() && !self.gguf_file.trim().is_empty();
                if ui.add_enabled(can_fetch, Button::new("Fetch GGUF")).clicked() {
                    self.gguf_fetching = true;
                    self.gguf_status = Some("Downloading…".to_string());
                    let repo = self.gguf_repo.clone();
                    let fname = self.gguf_file.clone();
                    let (tx, rx) = crossbeam::channel::unbounded();
                    self.gguf_tx = tx.clone();
                    self.gguf_rx = rx;
                    tokio::spawn(async move {
                        // Use existing HF helper to ensure the file is present locally
                        let res: Result<std::path::PathBuf, String> = (|| {
                            let repo = crate::ai::hf::hf_model(&repo).map_err(|e| e.to_string())?;
                            let p = crate::ai::hf::hf_get_file(&repo, &fname).map_err(|e| e.to_string())?;
                            Ok(p)
                        })();
                        let _ = tx.send(res.map(|pb| pb.to_string_lossy().to_string()));
                    });
                }
                if self.gguf_fetching {
                    if let Ok(msg) = self.gguf_rx.try_recv() {
                        self.gguf_fetching = false;
                        match msg {
                            Ok(path) => {
                                self.gguf_resolved_path = Some(path.clone());
                                self.gguf_status = Some(format!("Ready: {}", path));
                            }
                            Err(e) => {
                                self.gguf_status = Some(format!("Error: {e}"));
                            }
                        }
                    }
                }
                if let Some(msg) = &self.gguf_status { ui.label(msg); }
                ui.separator();
                ui.checkbox(&mut self.gguf_prefer, "Prefer GGUF when running");
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
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.prefer_full_precision, "Full precision (FP32/BF16) when available");
                ui.label(RichText::new("Fallback to lower precision automatically on error").weak());
            });
            ui.horizontal(|ui| {
                ui.label("Preview every N steps (0=off):");
                ui.add(Slider::new(&mut self.preview_every_n, 0..=10));
            });
            
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.loading_model && self.current_path.is_some(), Button::new("Run Edit")).clicked() {
                    // Lazily load the model if not loaded yet, preferring GGUF override if present
                    if !self.model_loaded {
                        self.loading_model = true;
                        self.model_error = None;
                        crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Loading model");
                        let repo = self.model_repo.clone();
                        let slot = self.pipeline.clone();
                        let (tx, rx) = crossbeam::channel::unbounded();
                        self.status_tx = tx.clone();
                        self.status_rx = rx;
                        let gguf_override = self.gguf_resolved_path.clone();
                        // capture intended run params so we continue immediately after load
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
                            deterministic_vae: true,
                            preview_every_n: if self.preview_every_n == 0 { None } else { Some(self.preview_every_n) },
                            preview_tx: {
                                if self.preview_every_n > 0 {
                                    let (tx, rx) = crossbeam::channel::unbounded();
                                    self.preview_rx = Some(rx);
                                    Some(tx)
                                } else { None }
                            },
                            cancel: {
                                let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                self.cancel_flag = Some(flag.clone());
                                Some(flag)
                            },
                        };
                        crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Running, "Editing image");
                        tokio::spawn(async move {
                            crate::ai::unload_heavy_models_except("").await;
                            let prefer = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                            let res = if let Some(p) = gguf_override.filter(|s| !s.is_empty()) {
                                match crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf_with_overrides(
                                    &repo, prefer, Some(std::path::PathBuf::from(p.clone())), None, None) {
                                    Ok(pipe) => Ok(pipe),
                                    Err(e) => {
                                        let es = e.to_string();
                                        let mut hint = String::new();
                                        if es.contains("unknown dtype") || es.contains("unsupported quant") || p.to_ascii_lowercase().contains("-q") {
                                            hint = " Hint: This GGUF looks quantized (e.g., Q2_K/Q4_K), which isn’t supported by the current loader. Try an F16 GGUF (e.g., *-F16.gguf).".to_string();
                                        }
                                        log::warn!("[ImageEdit] GGUF load failed ({}), retrying with safetensors", es);
                                        crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("GGUF load failed: {}. Falling back to safetensors…{}", es, hint));
                                        crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer)
                                    }
                                }
                            } else {
                                crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&repo, prefer)
                            }.map_err(|e| e.to_string());
                            match res {
                                Ok(p) => {
                                    let dev = if matches!(p.device, candle_core::Device::Cuda(_)) { crate::ui::status::DeviceKind::GPU } else { crate::ui::status::DeviceKind::CPU };
                                    crate::ui::status::QWEN_EDIT_STATUS.set_device(dev);
                                    crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
                                    let mut g = slot.lock().await; *g = Some(p); let _ = tx.send(Ok(()));
                                    // Now run the edit
                                    let out = if let Some(pipe) = slot.lock().await.as_ref() {
                                        match pipe.run_edit(std::path::Path::new(&pth), &opts) { Ok(b) => b, Err(e) => { log::error!("[ImageEdit] run_edit failed after load: {e}"); Vec::new() } }
                                    } else { Vec::new() };
                                    if !out.is_empty() {
                                        let key = format!("imageedit::result::{}", &pth);
                                        if let Ok(mut guard) = cache.lock() { guard.insert(key, Arc::from(out.into_boxed_slice())); }
                                    }
                                    crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                                }
                                Err(e) => {
                                    log::error!("[ImageEdit] model load failed: {e}");
                                    crate::ui::status::QWEN_EDIT_STATUS.set_error(format!("Load failed: {e}"));
                                    let mut g = slot.lock().await; *g = None; let _ = tx.send(Err(e));
                                }
                            }
                        });
                    } else {
                        // Run the real edit pipeline and capture output PNG bytes
                        let slot = self.pipeline.clone();
                        let pth = self.current_path.clone().unwrap_or_default();
                        let cache = self.thumb_cache.clone();
                        let repo = self.model_repo.clone();
                        let prefer_full = self.prefer_full_precision;
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
                            preview_every_n: if self.preview_every_n == 0 { None } else { Some(self.preview_every_n) },
                            preview_tx: {
                                if self.preview_every_n > 0 {
                                    let (tx, rx) = crossbeam::channel::unbounded();
                                    self.preview_rx = Some(rx);
                                    Some(tx)
                                } else { None }
                            },
                            cancel: {
                                let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                self.cancel_flag = Some(flag.clone());
                                Some(flag)
                            },
                        };
                        tokio::spawn(async move {
                            // Release other heavy models for headroom
                            crate::ai::unload_heavy_models_except("").await;
                            let mut png = Vec::new();
                            if let Some(pipe) = slot.lock().await.as_ref() {
                                match pipe.run_edit(std::path::Path::new(&pth), &opts) {
                                    Ok(b) => png = b,
                                    Err(e) => {
                                        let es = e.to_string();
                                        log::warn!("[ImageEdit] run_edit error: {es}");
                                        let should_fallback = es.to_ascii_lowercase().contains("cuda")
                                            || es.to_ascii_lowercase().contains("oom")
                                            || es.contains("named symbol not found");
                                        if should_fallback && prefer_full {
                                            // Attempt auto fallback: reload pipeline with reduced precision and retry once
                                            log::warn!("[ImageEdit] retrying with reduced precision (F16/F32)");
                                            let current_repo = repo.clone();
                                            let prefer = if candle_core::Device::new_cuda(0).is_ok() { candle_core::DType::F16 } else { candle_core::DType::F32 };
                                            match crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(&current_repo, prefer) {
                                                Ok(lowp) => {
                                                    let dev = if matches!(lowp.device, candle_core::Device::Cuda(_)) { crate::ui::status::DeviceKind::GPU } else { crate::ui::status::DeviceKind::CPU };
                                                    crate::ui::status::QWEN_EDIT_STATUS.set_device(dev);
                                                    let mut guard = slot.lock().await; *guard = Some(lowp);
                                                    drop(guard);
                                                    // Retry
                                                    if let Some(pipe2) = slot.lock().await.as_ref() {
                                                        match pipe2.run_edit(std::path::Path::new(&pth), &opts) { Ok(b) => png = b, Err(e2) => { log::error!("[ImageEdit] run_edit failed after fallback: {e2}"); } }
                                                    }
                                                }
                                                Err(e2) => {
                                                    log::error!("[ImageEdit] fallback reload failed: {e2}");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if !png.is_empty() {
                                let key = format!("imageedit::result::{}", &pth);
                                if let Ok(mut guard) = cache.lock() { guard.insert(key, Arc::from(png.into_boxed_slice())); }
                            }
                            crate::ui::status::QWEN_EDIT_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                        });
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("Cancel Run").clicked() {
                        if let Some(f) = &self.cancel_flag { f.store(true, std::sync::atomic::Ordering::Relaxed); }
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
                });
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
                    // Live step preview if enabled
                    if let Some(rx) = &self.preview_rx {
                        if let Ok(png) = rx.try_recv() {
                            let key = format!("imageedit::preview_step::{}", path);
                            if let Ok(mut cache) = self.thumb_cache.lock() {
                                cache.insert(key.clone(), Arc::from(png.into_boxed_slice()));
                            }
                        }
                        let key = format!("imageedit::preview_step::{}", path);
                        if let Ok(cache) = self.thumb_cache.lock() {
                            if let Some(bytes_arc) = cache.get(&key) {
                                ui.label(RichText::new("Live Preview:").underline().strong());
                                let img_src = eframe::egui::ImageSource::Bytes {
                                    uri: std::borrow::Cow::from(format!("bytes://{}", key)),
                                    bytes: eframe::egui::load::Bytes::Shared(bytes_arc.clone()),
                                };
                                eframe::egui::Image::new(img_src)
                                    .max_size(ui.available_size()/1.3)
                                    .ui(ui);
                            }
                        }
                    }

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
