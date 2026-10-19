use super::*;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct RawVisionConfig {
    hidden_size: Option<usize>,
    #[serde(alias = "in_chans")]
    in_chans: Option<usize>,
    #[serde(alias = "spatial_patch_size")]
    spatial_patch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct RawRopeScaling {
    mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct RawJinaConfig {
    // text-related top-level fields
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_theta: Option<f64>,
    rms_norm_eps: Option<f64>,
    use_sliding_window: Option<bool>,
    sliding_window: Option<usize>,
    max_window_layers: Option<usize>,
    rope_scaling: Option<RawRopeScaling>,
    // token ids
    #[allow(unused)]
    bos_token_id: Option<u32>,
    #[allow(unused)]
    eos_token_id: Option<u32>,
    #[serde(alias = "vision_token_id", alias = "image_token_id")]
    image_token_id: Option<u32>,
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
    // vision
    vision_config: Option<RawVisionConfig>,
}

impl RawJinaConfig {
    pub(super) fn into_qwen_config(self) -> crate::ai::qwen2_5_vl::Config {
        let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
            vocab_size: self.vocab_size.unwrap_or(151665),
            hidden_size: self.hidden_size.unwrap_or(3584),
            intermediate_size: self.intermediate_size.unwrap_or(18944),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(28),
            num_attention_heads: self.num_attention_heads.unwrap_or(28),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(32768),
            rope_theta: self.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            sliding_window: Some(self.sliding_window.unwrap_or(32768)),
            max_window_layers: self.max_window_layers.unwrap_or(28),
            mrope_section: self
                .rope_scaling
                .as_ref()
                .and_then(|r| r.mrope_section.clone()),
            rope_scaling: None,
        };

        let vraw = self.vision_config.unwrap_or(RawVisionConfig {
            hidden_size: None,
            in_chans: None,
            spatial_patch_size: None,
        });
        let vision_cfg = crate::ai::qwen2_5_vl::VisionConfig {
            hidden_size: vraw.hidden_size.unwrap_or(1536),
            intermediate_size: 6144,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 896,
            patch_size: vraw.spatial_patch_size.unwrap_or(14),
            num_channels: vraw.in_chans.unwrap_or(3),
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: None,
            window_size: None,
        };

        crate::ai::qwen2_5_vl::Config {
            vision_config: vision_cfg,
            text_config: text_cfg.clone(),
            hidden_size: text_cfg.hidden_size,
            vision_start_token_id: self.vision_start_token_id.unwrap_or(151652),
            vision_end_token_id: self.vision_end_token_id.unwrap_or(151653),
            image_token_id: self.image_token_id.unwrap_or(151654),
        }
    }
}
