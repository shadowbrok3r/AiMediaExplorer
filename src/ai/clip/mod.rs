
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel, TextEmbedding, TextInitOptions, EmbeddingModel};
use crate::ui::status::{CLIP_STATUS, StatusState, GlobalStatusIndicator};
pub mod embed;

pub struct ClipEngine { 
    pub backend: ClipBackend 
}

impl ClipEngine {
    pub fn new_with_model_key(model_key: &str) -> anyhow::Result<Self> {
        // Map simple keys -> underlying backends
        let be = match model_key {
            // FastEmbed
            "unicom-vit-b32" => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::UnicomVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
            "clip-vit-b32" => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
            // SigLIP family via candle
            key if key.starts_with("siglip") => {
                let sig = crate::ai::siglip::SiglipEngine::new(model_key)?;
                ClipBackend::Siglip(sig)
            }
            _ => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::UnicomVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
        };
        Ok(Self { backend: be })
    }
    
    pub fn siglip_logits_image_vs_texts(&mut self, image_path: &str, texts: &[String]) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            ClipBackend::Siglip(sig) => sig.logits_image_vs_texts(image_path, texts),
            _ => anyhow::bail!("siglip_logits_image_vs_texts is only available for the SigLIP backend"),
        }
    }
}

pub enum ClipBackend {
    FastEmbed { image_model: ImageEmbedding, text_model: TextEmbedding },
    Siglip(crate::ai::siglip::SiglipEngine),
}

pub(crate) async fn ensure_clip_engine(engine_slot: &std::sync::Arc<tokio::sync::Mutex<Option<ClipEngine>>>) -> anyhow::Result<()> {
    let mut guard = engine_slot.lock().await;
    if guard.is_none() {
        CLIP_STATUS.set_state(StatusState::Initializing, "Loading model");
        // Prefer cached settings (updated immediately on save) to avoid races with async DB save.
        let cached = crate::database::settings::load_settings();
    let model_key = if let Some(m) = cached.clip_model.clone() {
            m
        } else {
            match crate::database::get_settings().await {
        Ok(s) => s.clip_model.unwrap_or_else(|| "unicom-vit-b32".to_string()),
                Err(e) => {
                    log::warn!("[CLIP] get_settings() failed: {e}. Falling back to default.");
            "unicom-vit-b32".to_string()
                }
            }
        };
        CLIP_STATUS.set_model(&model_key);
        log::info!("[CLIP] Loading model key: {}", model_key);
        match ClipEngine::new_with_model_key(&model_key) {
            Ok(c) => { *guard = Some(c); log::info!("[CLIP] Loaded."); CLIP_STATUS.set_state(StatusState::Idle, "Ready"); },
            Err(e) => { log::error!("[CLIP] Failed to init: {e}"); CLIP_STATUS.set_state(StatusState::Error, "Init failed"); return Err(e); }
        }
    }
    Ok(())
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }
    v
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { dot(a,b) }