use std::collections::HashMap;

use candle_transformers::models::{
    clip::{text_model::Activation, vision_model::ClipVisionConfig},
    llama::{Config, LlamaEosToks},
};
use serde::{Deserialize, Serialize, Deserializer};

use super::clip_image_processor::CLIPImageProcessor;

// original config from liuhaotian/llava
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLaVAConfig {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    //pub attention_bias: bool,
    //pub attention_dropout: f32,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_bos_token_id")]
    pub bos_token_id: usize,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_eos_token_id")]
    pub eos_token_id: usize,
    //pub freeze_mm_mlp_adapter: bool,
    //pub freeze_mm_vision_resampler: bool,
    //pub hidden_act: String,
    pub hidden_size: usize,
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
    pub image_crop_resolution: usize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_split_resolution: usize,
    //pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mm_hidden_size: usize,
    #[serde(default = "default_mm_patch_merge_type")]
    pub mm_patch_merge_type: String,
    //pub mm_projector_lr: Option<f32>,
    pub mm_projector_type: String,
    //pub mm_resampler_type: Option<String>,
    //pub mm_use_im_patch_token: bool,
    pub mm_use_im_start_end: bool,
    pub mm_vision_select_feature: String,
    pub mm_vision_select_layer: isize,
    pub mm_vision_tower: Option<String>,
    //pub mm_vision_tower_lr: f32,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_pad_token_id")]
    pub pad_token_id: usize,
    //pub pretraining_tp: usize,
    pub rms_norm_eps: f32,
    //pub rope_scaling: Option<f32>,
    pub rope_theta: f32,
    //pub tie_word_embeddings: bool,
    pub tokenizer_model_max_length: Option<usize>,
    //pub tokenizer_padding_side: String,
    pub torch_dtype: String,
    //pub transformers_version: String,
    //pub tune_mm_mlp_adapter: bool
    //pub tune_mm_vision_resampler: bool,
    //pub unfreeze_mm_vision_tower: bool,
    pub use_cache: bool,
    //pub use_mm_proj: bool,
    pub vocab_size: usize,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: isize,
}

fn default_image_token_index() -> isize {
    -200
}

fn default_mm_patch_merge_type() -> String {
    "flat".to_string()
}

fn default_image_aspect_ratio() -> String {
    "square".to_string()
}

// ---- Custom deserialization helpers ---------------------------------------------------------
#[derive(Deserialize)]
#[serde(untagged)]
enum UsizeOrVec { U(usize), V(Vec<usize>) }

fn de_usize_or_seq<'de, D>(deserializer: D) -> Result<usize, D::Error>
where D: Deserializer<'de> {
    let v = UsizeOrVec::deserialize(deserializer)?;
    match v {
        UsizeOrVec::U(u) => Ok(u),
        UsizeOrVec::V(mut vec) => {
            if vec.is_empty() {
                Err(serde::de::Error::custom("empty array for token id"))
            } else {
                // If multiple elements are present, take the first and ignore the rest.
                // This mirrors some HF configs that redundantly list multiple EOS ids.
                if vec.len() > 1 {
                    // We cannot easily log from a serde visitor, but we can still use log if initialized.
                    log::warn!("[joycaption.config] token id array len={} -> taking first element", vec.len());
                }
                Ok(vec.remove(0))
            }
        },
    }
}

impl LLaVAConfig {
    pub fn to_llama_config(&self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps as f64,
            rope_theta: self.rope_theta,
            bos_token_id: Some(self.bos_token_id as u32),
            eos_token_id: Some(LlamaEosToks::Single(self.eos_token_id as u32)),
            use_flash_attn: false,
            rope_scaling: None,
            max_position_embeddings: 0,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HFLLaVATextConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_pad_token_id")]
    pub pad_token_id: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

fn default_num_hidden_layers() -> usize {
    32
}

fn default_use_cache() -> bool {
    true
}

fn default_hidden_size() -> usize {
    4096
}

fn default_intermediate_size() -> usize {
    11008
}

fn default_max_length() -> usize {
    4096
}

fn default_num_attention_heads() -> usize {
    32
}

fn default_num_key_value_heads() -> usize {
    32
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_position_embeddings() -> usize { 4096 }
fn default_model_type() -> String { "llava".to_string() }
fn default_pad_token_id() -> usize { 0 }
fn default_rms_norm_eps() -> f32 { 1e-5 }
fn default_torch_dtype() -> String { "float16".to_string() }
fn default_vocab_size() -> usize { 32000 }

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HFLLaVAVisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_projection_dim")]
    pub projection_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

fn default_vision_hidden_size() -> usize { 1024 }
fn default_vision_image_size() -> usize { 336 }
fn default_vision_intermediate_size() -> usize { 4096 }
fn default_vision_model_type() -> String { "clip_vision_model".to_string() }
fn default_vision_num_attention_heads() -> usize { 16 }
fn default_vision_num_hidden_layers() -> usize { 24 }
fn default_vision_patch_size() -> usize { 14 }
fn default_vision_projection_dim() -> usize { 1024 }

// config from llava-v1.6-vicuna-7b-hf
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HFLLaVAConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default = "default_ignore_index")]
    pub ignore_index: isize,
    #[serde(default)]
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: isize,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: String,
    #[serde(default)]
    pub text_config: HFLLaVATextConfig,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    #[serde(default)]
    pub use_image_newline_parameter: bool,
    #[serde(default)]
    pub vision_config: HFLLaVAVisionConfig,
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: isize,
    #[serde(default = "default_vision_feature_select_strategy")]
    pub vision_feature_select_strategy: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

fn default_ignore_index() -> isize { -1 }
fn default_projector_hidden_act() -> String { "gelu".to_string() }
fn default_vision_feature_layer() -> isize { -2 }
fn default_vision_feature_select_strategy() -> String { "default".to_string() }

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HFGenerationConfig {
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_bos_token_id")]
    pub bos_token_id: usize,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_eos_token_id")]
    pub eos_token_id: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    #[serde(deserialize_with = "de_usize_or_seq", default = "default_pad_token_id")]
    pub pad_token_id: usize,
}

fn default_bos_token_id() -> usize { 1 }
fn default_eos_token_id() -> usize { 2 }

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HFPreProcessorConfig {
    #[serde(default = "default_aspect_ratio_setting")]
    pub aspect_ratio_setting: String,
    #[serde(default = "default_crop_size")]
    pub crop_size: HashMap<String, usize>,
    #[serde(default)]
    pub do_center_crop: bool,
    #[serde(default = "default_true")]
    pub do_convert_rgb: bool,
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    #[serde(default = "default_true")]
    pub do_rescale: bool,
    #[serde(default = "default_true")]
    pub do_resize: bool,
    #[serde(default = "default_image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    pub image_std: Vec<f32>,
    #[serde(default = "default_resample")]
    pub resample: u32,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default = "default_size_map")]
    pub size: HashMap<String, f32>,
}

fn default_aspect_ratio_setting() -> String { "square".to_string() }
fn default_crop_size() -> HashMap<String, usize> { let mut m=HashMap::new(); m.insert("height".to_string(),384); m.insert("width".to_string(),384); m }
fn default_true() -> bool { true }
fn default_image_mean() -> Vec<f32> { vec![0.48145466, 0.4578275, 0.40821073] }
fn default_image_std() -> Vec<f32> { vec![0.26862954, 0.26130258, 0.27577711] }
fn default_resample() -> u32 { 3 }
fn default_rescale_factor() -> f32 { 1.0/255.0 }
fn default_size_map() -> HashMap<String, f32> { let mut m=HashMap::new(); m.insert("shortest_edge".to_string(), 384.0); m }

impl HFPreProcessorConfig {
    pub fn to_clip_image_processor(&self) -> CLIPImageProcessor {
        CLIPImageProcessor {
            size: self.size["shortest_edge"] as u32,
            do_resize: self.do_resize,
            do_center_crop: self.do_center_crop,
            crop_size: self.crop_size["height"] as u32,
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
        }
    }
}

impl HFLLaVAConfig {
    pub fn to_clip_vision_config(&self) -> ClipVisionConfig {
        // Some checkpoints (like working reference) use projection_dim == hidden_size; enforce that if mismatch.
        let projection_dim = if self.vision_config.projection_dim != self.vision_config.hidden_size {
            log::info!("[config.to_clip_vision_config] overriding projection_dim {} -> {} to match hidden_size", self.vision_config.projection_dim, self.vision_config.hidden_size);
            self.vision_config.hidden_size
        } else { self.vision_config.projection_dim };
        ClipVisionConfig {
            embed_dim: self.vision_config.hidden_size,
            activation: Activation::QuickGelu,
            intermediate_size: self.vision_config.intermediate_size,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            projection_dim,
            num_channels: 3,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
        }
    }
    fn map_projector_type(s: &str) -> String {
        if s == "gelu" {
            "mlp2x_gelu".to_string()
        } else {
            s.to_string()
        }
    }

    fn map_select_feature(s: &str) -> String {
        if s == "default" {
            "patch".to_string()
        } else {
            "cls_patch".to_string()
        }
    }

    pub fn to_llava_config(
        &self,
        name: &str,
        generation_config: &HFGenerationConfig,
        preprocessor_config: &HFPreProcessorConfig,
    ) -> LLaVAConfig {
        log::info!(
            "[config.to_llava_config] name={} vision.hidden_size={} vision.projection_dim={} projector_type={} vision_feature_layer={} select_strategy={} final_mm_hidden_size={}",
            name,
            self.vision_config.hidden_size,
            self.vision_config.projection_dim,
            self.projector_hidden_act,
            self.vision_feature_layer,
            self.vision_feature_select_strategy,
            self.vision_config.hidden_size
        );
        LLaVAConfig {
            _name_or_path: name.to_string(),
            architectures: self.architectures.clone(),
            bos_token_id: generation_config.bos_token_id,
            eos_token_id: generation_config.eos_token_id,
            hidden_size: self.text_config.hidden_size,
            image_aspect_ratio: preprocessor_config.aspect_ratio_setting.clone(),
            image_crop_resolution: 224,
            image_grid_pinpoints: self.image_grid_pinpoints.clone(),
            image_split_resolution: 224,
            intermediate_size: self.text_config.intermediate_size,
            max_position_embeddings: self.text_config.max_position_embeddings,
            // Use the vision tower hidden size (projection dim) as mm_hidden_size instead of a hard-coded value.
            mm_hidden_size: self.vision_config.hidden_size,
            mm_patch_merge_type: "spatial_unpad".to_string(),
            mm_projector_type: Self::map_projector_type(&self.projector_hidden_act),
            mm_use_im_start_end: false,
            mm_vision_select_feature: Self::map_select_feature(
                &self.vision_feature_select_strategy,
            ),
            mm_vision_select_layer: self.vision_feature_layer,
            mm_vision_tower: None,
            model_type: self.model_type.clone(),
            num_attention_heads: self.text_config.num_attention_heads,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_key_value_heads: self.text_config.num_key_value_heads,
            pad_token_id: self.text_config.pad_token_id,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            tokenizer_model_max_length: Some(4096),
            torch_dtype: self.torch_dtype.clone(),
            use_cache: self.text_config.use_cache,
            vocab_size: self.vocab_size,
            image_token_index: self.image_token_index,
        }
    }
}
