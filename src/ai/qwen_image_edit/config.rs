use serde::Deserialize;

// Model index for the Qwen image edit pipeline. Some repos specify component
// types as arrays like ["diffusers", "FlowMatchEulerDiscreteScheduler"] rather
// than path refs. Support both forms via an untagged enum.

#[derive(Debug, Clone, Deserialize)]
pub struct ModelIndex {
    #[serde(default)]
    pub _class_name: Option<String>,
    #[serde(default)]
    pub _diffusers_version: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default)]
    pub processor: Option<ComponentSpec>,
    #[serde(default)]
    pub scheduler: Option<ComponentSpec>,
    #[serde(default)]
    pub text_encoder: Option<ComponentSpec>,
    #[serde(default)]
    pub tokenizer: Option<ComponentSpec>,
    #[serde(default)]
    pub transformer: Option<ComponentSpec>,
    #[serde(default)]
    pub vae: Option<ComponentSpec>,

    // Some repos may still use older fields
    #[serde(default)]
    pub unet: Option<ComponentSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComponentRef {
    pub _name_or_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ComponentSpec {
    Ref(ComponentRef),
    // e.g. ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    Pair(Vec<String>),
}

impl ComponentSpec {
    pub fn as_pair(&self) -> Option<(String, String)> {
        match self {
            ComponentSpec::Pair(v) if v.len() >= 2 => Some((v[0].clone(), v[1].clone())),
            _ => None,
        }
    }
}

// Generic scheduler_config.json superset
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SchedulerConfig {
    #[serde(default)]
    pub num_train_timesteps: Option<u32>,
    #[serde(default)]
    pub beta_start: Option<f32>,
    #[serde(default)]
    pub beta_end: Option<f32>,
    #[serde(default)]
    pub beta_schedule: Option<String>,
    #[serde(default)]
    pub trained_betas: Option<Vec<f32>>,
    #[serde(default)]
    pub prediction_type: Option<String>,
    #[serde(default)]
    pub steps_offset: Option<i32>,
    #[serde(default)]
    pub clip_sample: Option<bool>,
    #[serde(default)]
    pub clip_sample_range: Option<f32>,
    #[serde(default)]
    pub rescale_betas_zero_snr: Option<bool>,
    #[serde(default)]
    pub timestep_spacing: Option<String>,
}
