use anyhow::Result;

/// Generic trait for Hugging Face-based AI models that can be loaded from a repo key.
pub trait HFAiModel: Sized {
    fn load_from_hub(model_key: &str) -> Result<Self>;
}

// Re-export a convenient alias for the Qwen image edit pipeline
pub type QwenImageEdit = crate::ai::qwen_image_edit::model::QwenImageEditPipeline;
