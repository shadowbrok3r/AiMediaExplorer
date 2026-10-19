//! The Qwen image edit pipeline as an [`EditBackend`]: one place that loads the model, with the reduced precision and safetensors fallbacks, and runs the edit.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;

use super::backend::{EditBackend, EditRequest, RunCtl};
use super::model::{EditError, ErrorKind, ModelSettings, Params};
use crate::ai::qwen_image_edit::EditOptions;
use crate::ai::qwen_image_edit::model::QwenImageEditPipeline;
use crate::status::{DeviceKind, GlobalStatusIndicator, QWEN_EDIT_STATUS, StatusState};

/// How often the status of the pipeline is copied into the progress of the tab.
const TICK: Duration = Duration::from_millis(150);

type Slot = Arc<tokio::sync::Mutex<Option<QwenImageEditPipeline>>>;

pub struct QwenBackend {
    slot: Slot,
    loaded: std::sync::atomic::AtomicBool,
}

impl Default for QwenBackend {
    fn default() -> Self {
        Self { slot: Arc::new(tokio::sync::Mutex::new(None)), loaded: std::sync::atomic::AtomicBool::new(false) }
    }
}

fn has_cuda() -> bool {
    candle_core::Device::new_cuda(0).is_ok()
}

fn dtype(full_precision: bool) -> candle_core::DType {
    match (has_cuda(), full_precision) {
        (true, true) => candle_core::DType::BF16,
        (true, false) => candle_core::DType::F16,
        (false, _) => candle_core::DType::F32,
    }
}

fn reduced_dtype() -> candle_core::DType {
    if has_cuda() { candle_core::DType::F16 } else { candle_core::DType::F32 }
}

/// Loads the pipeline from the GGUF file when one is given, else from the repo.
fn load_blocking(repo: &str, prefer: candle_core::DType, gguf: Option<PathBuf>) -> Result<QwenImageEditPipeline, String> {
    match gguf {
        Some(path) => QwenImageEditPipeline::load_from_hf_with_overrides(repo, prefer, Some(path), None, None).map_err(|e| e.to_string()),
        None => QwenImageEditPipeline::load_from_hf(repo, prefer).map_err(|e| e.to_string()),
    }
}

fn looks_like_memory_trouble(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("cuda") || m.contains("oom") || m.contains("out of memory") || m.contains("named symbol not found")
}

fn mark_ready(pipeline: &QwenImageEditPipeline) {
    let device = if matches!(pipeline.device, candle_core::Device::Cuda(_)) { DeviceKind::GPU } else { DeviceKind::CPU };
    QWEN_EDIT_STATUS.set_device(device);
    QWEN_EDIT_STATUS.set_state(StatusState::Idle, "Ready");
}

impl QwenBackend {
    async fn load_in(&self, settings: &ModelSettings, params: &Params) -> Result<(), EditError> {
        QWEN_EDIT_STATUS.set_state(StatusState::Initializing, "Loading model");
        crate::ai::unload_heavy_models_except("").await;
        let (repo, prefer_full) = (settings.model_repo.trim().to_string(), params.prefer_full_precision);
        let gguf = settings.gguf_prefer.then(|| settings.gguf_path.clone()).flatten().filter(|p| !p.is_empty()).map(PathBuf::from);
        let first = {
            let (repo, gguf) = (repo.clone(), gguf.clone());
            tokio::task::spawn_blocking(move || load_blocking(&repo, dtype(prefer_full), gguf)).await.map_err(|e| EditError::new(ErrorKind::ModelLoad, e.to_string()))?
        };
        let loaded = match first {
            Ok(p) => Ok(p),
            Err(e) if gguf.is_some() => {
                log::warn!("[ImageEdit] GGUF load failed ({e}), retrying with safetensors");
                QWEN_EDIT_STATUS.set_error(format!("GGUF load failed: {e}. Falling back to safetensors..."));
                let repo = repo.clone();
                tokio::task::spawn_blocking(move || load_blocking(&repo, dtype(prefer_full), None)).await.map_err(|e| EditError::new(ErrorKind::ModelLoad, e.to_string()))?
            }
            Err(e) if prefer_full && looks_like_memory_trouble(&e) => {
                QWEN_EDIT_STATUS.set_error(format!("Load failed at full precision: {e}. Retrying at reduced precision..."));
                let repo = repo.clone();
                tokio::task::spawn_blocking(move || load_blocking(&repo, reduced_dtype(), None)).await.map_err(|e| EditError::new(ErrorKind::ModelLoad, e.to_string()))?
            }
            Err(e) => Err(e),
        };
        match loaded {
            Ok(pipeline) => {
                mark_ready(&pipeline);
                *self.slot.lock().await = Some(pipeline);
                self.loaded.store(true, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                log::error!("[ImageEdit] model load failed: {e}");
                QWEN_EDIT_STATUS.set_error(format!("Load failed: {e}"));
                *self.slot.lock().await = None;
                self.loaded.store(false, Ordering::SeqCst);
                let kind = if looks_like_memory_trouble(&e) { ErrorKind::OutOfMemory } else { ErrorKind::ModelLoad };
                Err(EditError::new(kind, e))
            }
        }
    }

    /// Runs the edit once on a blocking thread with the pipeline in the slot.
    async fn run_once(&self, req: &EditRequest, ctl: &RunCtl) -> Result<Vec<u8>, String> {
        let opts = EditOptions {
            prompt: req.params.prompt.clone(),
            negative_prompt: Some(req.params.negative_prompt.clone()).filter(|n| !n.trim().is_empty()),
            guidance_scale: req.params.guidance_scale,
            num_inference_steps: req.params.steps,
            strength: req.params.strength,
            scheduler: Some("flow_match_euler".into()),
            seed: None,
            deterministic_vae: true,
            preview_every_n: (req.params.preview_every_n > 0).then_some(req.params.preview_every_n),
            preview_tx: ctl.preview.clone().filter(|_| req.params.preview_every_n > 0),
            cancel: Some(ctl.cancel.clone()),
        };
        let (slot, source) = (self.slot.clone(), req.source.clone());
        tokio::task::spawn_blocking(move || {
            let guard = slot.blocking_lock();
            match guard.as_ref() {
                Some(pipeline) => pipeline.run_edit(&source, &opts).map_err(|e| e.to_string()),
                None => Err("the model is not loaded".to_string()),
            }
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

#[async_trait]
impl EditBackend for QwenBackend {
    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load(&self, settings: &ModelSettings, params: &Params) -> Result<(), EditError> {
        self.load_in(settings, params).await
    }

    async fn edit(&self, req: &EditRequest, ctl: &RunCtl) -> Result<Vec<u8>, EditError> {
        crate::ai::unload_heavy_models_except("").await;
        let ticker = {
            let progress = ctl.progress.clone();
            tokio::spawn(async move {
                loop {
                    let status = QWEN_EDIT_STATUS.snapshot();
                    if status.progress_total > 0 {
                        progress(status.progress_current, status.progress_total, &status.detail);
                    }
                    tokio::time::sleep(TICK).await;
                }
            })
        };
        let mut result = self.run_once(req, ctl).await;
        if let Err(e) = &result {
            if req.params.prefer_full_precision && looks_like_memory_trouble(e) && !ctl.cancel.load(Ordering::Relaxed) {
                log::warn!("[ImageEdit] retrying at reduced precision after: {e}");
                let mut lower = req.params.clone();
                lower.prefer_full_precision = false;
                let reloaded = tokio::task::spawn_blocking({
                    let (repo, slot) = (req.settings.model_repo.trim().to_string(), self.slot.clone());
                    move || load_blocking(&repo, reduced_dtype(), None).map(|p| {
                        mark_ready(&p);
                        *slot.blocking_lock() = Some(p);
                    })
                })
                .await;
                if matches!(reloaded, Ok(Ok(()))) {
                    let retry = EditRequest { params: lower, ..req.clone() };
                    result = self.run_once(&retry, ctl).await;
                }
            }
        }
        ticker.abort();
        QWEN_EDIT_STATUS.set_progress(0, 0);
        QWEN_EDIT_STATUS.set_state(StatusState::Idle, "Idle");
        result.map_err(EditError::classified)
    }

    async fn unload(&self) {
        *self.slot.lock().await = None;
        self.loaded.store(false, Ordering::SeqCst);
        QWEN_EDIT_STATUS.set_state(StatusState::Idle, "Unloaded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_trouble_is_told_from_other_failures() {
        for text in ["CUDA_ERROR_OUT_OF_MEMORY", "cuda error", "OOM at step 3", "out of memory", "named symbol not found"] {
            assert!(looks_like_memory_trouble(text), "{text}");
        }
        assert!(!looks_like_memory_trouble("repo not found"));
    }

    #[tokio::test]
    async fn an_unloaded_backend_reports_so_and_unloading_is_harmless() {
        let backend = QwenBackend::default();
        assert!(!backend.is_loaded());
        backend.unload().await;
        assert!(!backend.is_loaded());
        let missing = backend.run_once(&EditRequest { source: PathBuf::from("/no/file.png"), params: Params::default(), settings: ModelSettings::default() }, &RunCtl { cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), progress: Arc::new(|_, _, _| {}), preview: None }).await;
        assert_eq!(missing.unwrap_err(), "the model is not loaded");
    }
}
