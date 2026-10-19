//! A stand-in for the model that edits a picture in a moment: for trying the tab without downloading gigabytes, and for screenshots and tests.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, Rgba};

use super::backend::{EditBackend, EditRequest, RunCtl};
use super::model::{EditError, ErrorKind, ModelSettings, Params, fit_dims};
use crate::status::{GlobalStatusIndicator, QWEN_EDIT_STATUS, StatusState};

/// Environment variable that starts the tab on this backend when set to `test`.
pub const ENV: &str = "SMART_MEDIA_IMAGE_EDIT_BACKEND";
/// A prompt containing this makes the edit fail as if the GPU ran out of memory.
pub const OOM_PROMPT: &str = "simulate oom";
const STEP_DELAY: Duration = Duration::from_millis(120);
const LONGEST_SIDE: u32 = 1024;

/// True when the environment asks for the test backend.
pub fn requested_by_env() -> bool {
    std::env::var(ENV).is_ok_and(|v| v.eq_ignore_ascii_case("test"))
}

#[derive(Default)]
pub struct DebugBackend {
    loaded: AtomicBool,
}

/// The look the prompt asks for: gray for black and white, inverted colors for invert, else a warm tint.
pub fn apply_effect(img: DynamicImage, prompt: &str, strength: f32) -> DynamicImage {
    let p = prompt.to_ascii_lowercase();
    if p.contains("black and white") || p.contains("grayscale") || p.contains("monochrome") {
        return DynamicImage::ImageLuma8(img.to_luma8()).to_rgba8().into();
    }
    if p.contains("invert") || p.contains("negative") {
        let mut out = img;
        out.invert();
        return out;
    }
    let k = strength.clamp(0.0, 1.0);
    let mut rgba = img.to_rgba8();
    for px in rgba.pixels_mut() {
        let Rgba([r, g, b, a]) = *px;
        *px = Rgba([(r as f32 * (1.0 + 0.3 * k)).min(255.0) as u8, g, (b as f32 * (1.0 - 0.3 * k)) as u8, a]);
    }
    rgba.into()
}

fn encode_png(img: &DynamicImage) -> Result<Vec<u8>, EditError> {
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).map_err(|e| EditError::new(ErrorKind::Other, e.to_string()))?;
    Ok(out.into_inner())
}

#[async_trait]
impl EditBackend for DebugBackend {
    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load(&self, _settings: &ModelSettings, _params: &Params) -> Result<(), EditError> {
        tokio::time::sleep(Duration::from_millis(300)).await;
        self.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn edit(&self, req: &EditRequest, ctl: &RunCtl) -> Result<Vec<u8>, EditError> {
        let source = req.source.clone();
        let img = tokio::task::spawn_blocking(move || image::open(&source)).await.map_err(|e| EditError::new(ErrorKind::Other, e.to_string()))?.map_err(|e| EditError::new(ErrorKind::Source, e.to_string()))?;
        let (w, h) = fit_dims(img.width(), img.height(), LONGEST_SIDE);
        let img = if (w, h) == img.dimensions() { img } else { img.resize(w, h, image::imageops::FilterType::Triangle) };
        let steps = (req.params.steps as u64).clamp(1, 12);
        QWEN_EDIT_STATUS.set_state(StatusState::Running, "Denoising");
        for step in 1..=steps {
            if ctl.cancel.load(Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(STEP_DELAY).await;
            QWEN_EDIT_STATUS.set_progress(step, steps);
            (ctl.progress)(step, steps, "Denoising");
            if let Some(tx) = &ctl.preview {
                let blend = apply_effect(img.clone(), &req.params.prompt, req.params.strength * step as f32 / steps as f32);
                if let Ok(png) = encode_png(&blend) {
                    let _ = tx.send(png);
                }
            }
        }
        QWEN_EDIT_STATUS.set_progress(0, 0);
        QWEN_EDIT_STATUS.set_state(StatusState::Idle, "Idle");
        if req.params.prompt.to_ascii_lowercase().contains(OOM_PROMPT) {
            return Err(EditError::new(ErrorKind::OutOfMemory, "CUDA_ERROR_OUT_OF_MEMORY (simulated by the test backend)"));
        }
        encode_png(&apply_effect(img, &req.params.prompt, req.params.strength))
    }

    async fn unload(&self) {
        self.loaded.store(false, Ordering::SeqCst);
    }
}
