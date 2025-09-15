use anyhow::Result;

/// Generic trait for Hugging Face-based AI models that can be loaded from a repo key.
pub trait HFAiModel: Sized {
    fn load_from_hub(model_key: &str) -> Result<Self>;
}
