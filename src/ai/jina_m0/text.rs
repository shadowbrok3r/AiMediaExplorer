//! Minimal scaffold for Jina M0 text model forward to be implemented with Candle.
#![allow(dead_code)]

use candle_core::{Tensor, Device, DType};
use anyhow::Result;

#[derive(Debug)]
pub struct TextModelConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

#[derive(Debug)]
pub struct TextModel {
    pub device: Device,
    pub dtype: DType,
    pub config: TextModelConfig,
}

impl TextModel {
    pub fn new(device: Device, dtype: DType, config: TextModelConfig) -> Self {
        Self { device, dtype, config }
    }

    /// Placeholder forward: returns a dummy pooled representation per sequence.
    /// Replace with real transformer forward wired to safetensors weights.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _ = (attention_mask,);
        // [B, L] -> [B, H] dummy: sum token ids, expand to hidden size
        let sums = input_ids.sum(1)?; // [B]
        let b = sums.dims()[0];
        let rep = sums.unsqueeze(1)?.broadcast_as(&[b, self.config.hidden_size])?;
        Ok(rep)
    }
}
