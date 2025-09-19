pub mod model;
pub mod config;
pub mod scheduler;

// Minimal VAE placeholder; will implement encode/decode later
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub scale_factor: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self { latent_channels: 4, scale_factor: 0.18215 }
    }
}

#[derive(Debug, Clone)]
pub struct EditOptions {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
    pub strength: f32,
    pub scheduler: Option<String>,
    pub seed: Option<u64>,
    pub deterministic_vae: bool,
}

impl Default for EditOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            guidance_scale: 7.5,
            num_inference_steps: 30,
            strength: 0.7,
            // Match the pipeline's default scheduler from model_index.json
            scheduler: Some("flow_match_euler".into()),
            seed: None,
            deterministic_vae: true,
        }
    }
}

use anyhow::Result;
use candle_core::DType;

impl crate::ai::model::HFAiModel for crate::ai::qwen_image_edit::model::QwenImageEditPipeline {
    fn load_from_hub(model_key: &str) -> Result<Self> {
        // Prefer fp16 on CUDA hardware
        let prefer = if candle_core::Device::new_cuda(0).is_ok() { DType::F16 } else { DType::F32 };
        let pipe = Self::load_from_hf(model_key, prefer)?;
        pipe.info();
        Ok(pipe)
    }
}
