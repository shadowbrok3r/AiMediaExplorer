use super::*;

impl JinaM0Engine {
    pub(super) fn forward_text_hidden(
        &self,
        text: &mut crate::ai::qwen2_5_vl::TextModel,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let position_ids = Tensor::arange(0i64, l as i64, &self.device)?
            .unsqueeze(0)?
            .expand((b, l))?;
        let hidden = text.forward(input_ids, &position_ids, None)?; // [B,L,H]
        Ok(hidden)
    }
    /// Fallback: compute masked-mean directly over token embeddings without decoder layers.
    /// Useful when rotary/attention issues occur but embeddings and norm are valid.
    pub(super) fn embed_only_pooled(
        &self,
        input_ids: &Tensor,
        attn_mask_ids: &Tensor,
        tconf: &crate::ai::qwen2_5_vl::TextConfig,
        text_prefix: &str,
    ) -> Result<Tensor> {
        let pooled = match &self.weights {
            Some(Weights::Single(w)) => unsafe {
                let (emb, norm_w) = crate::ai::hf::with_mmap_varbuilder_single(
                    w,
                    self.dtype,
                    &self.device,
                    |vb| {
                        let vb = if text_prefix.is_empty() {
                            vb
                        } else {
                            vb.pp(text_prefix)
                        };
                        let emb = candle_nn::embedding(
                            tconf.vocab_size,
                            tconf.hidden_size,
                            vb.pp("embed_tokens"),
                        )?;
                        let norm_w = vb.pp("norm").get(tconf.hidden_size, "weight")?;
                        Ok::<(candle_nn::Embedding, Tensor), anyhow::Error>((emb, norm_w))
                    },
                )?;
                let x = emb.forward(input_ids)?; // [B,L,H]
                // RMSNorm inline
                let x_dtype = x.dtype();
                let x32 = x.to_dtype(candle_core::DType::F32)?;
                let variance = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let x32 = x32.broadcast_div(&(variance + tconf.rms_norm_eps)?.sqrt()?)?;
                let x = x32.to_dtype(x_dtype)?;
                let x = x.broadcast_mul(&norm_w)?; // [B,L,H]
                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                let masked = x.broadcast_mul(&mask_e)?; // [B,L,H]
                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B,H]
                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                sum_h.broadcast_div(&lens.unsqueeze(1)?)?
            },
            Some(Weights::Sharded(files)) => unsafe {
                let (emb, norm_w) = crate::ai::hf::with_mmap_varbuilder_multi(
                    files,
                    self.dtype,
                    &self.device,
                    |vb| {
                        let vb = if text_prefix.is_empty() {
                            vb
                        } else {
                            vb.pp(text_prefix)
                        };
                        let emb = candle_nn::embedding(
                            tconf.vocab_size,
                            tconf.hidden_size,
                            vb.pp("embed_tokens"),
                        )?;
                        let norm_w = vb.pp("norm").get(tconf.hidden_size, "weight")?;
                        Ok::<(candle_nn::Embedding, Tensor), anyhow::Error>((emb, norm_w))
                    },
                )?;
                let x = emb.forward(input_ids)?;
                // RMSNorm inline
                let x_dtype = x.dtype();
                let x32 = x.to_dtype(candle_core::DType::F32)?;
                let variance = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let x32 = x32.broadcast_div(&(variance + tconf.rms_norm_eps)?.sqrt()?)?;
                let x = x32.to_dtype(x_dtype)?;
                let x = x.broadcast_mul(&norm_w)?;
                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                let masked = x.broadcast_mul(&mask_e)?; // [B,L,H]
                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B,H]
                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                sum_h.broadcast_div(&lens.unsqueeze(1)?)?
            },
            None => unreachable!(),
        };
        Ok(pooled)
    }
    // Infer text config (vocab_size, hidden_size, intermediate_size, layers, heads) from safetensors shapes
    pub(super) fn discover_text_config_from_file(
        path: &PathBuf,
    ) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        Self::discover_text_config_from_st(&st)
    }

    pub(super) fn discover_text_config_from_multi(
        files: &[PathBuf],
    ) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        // For sharded: read first shard to access metadata of names; if not available, bail
        if let Some(first) = files.first() {
            let bytes = std::fs::read(first)?;
            let st = SafeTensors::deserialize(&bytes)?;
            return Self::discover_text_config_from_st(&st);
        }
        Ok(None)
    }

    pub(super) fn discover_text_config_from_st(
        st: &safetensors::SafeTensors<'_>,
    ) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        // Find embed_tokens weight to get vocab_size, hidden_size
        let mut embed_shape: Option<(usize, usize)> = None;
        for name in st.names() {
            if name.ends_with("embed_tokens.weight") {
                let t = st.tensor(name)?;
                let shape = t.shape();
                if shape.len() == 2 {
                    embed_shape = Some((shape[0] as usize, shape[1] as usize));
                    break;
                }
            }
        }
        let (vocab_size, hidden_size) = match embed_shape {
            Some(s) => s,
            None => return Ok(None),
        };

        // Intermediate size from gate_proj
        let mut intermediate_size: Option<usize> = None;
        for name in st.names() {
            if name.contains("layers.0.mlp.gate_proj.weight") {
                let t = st.tensor(name)?;
                let shape = t.shape();
                if shape.len() == 2 {
                    intermediate_size = Some(shape[0] as usize);
                    break;
                }
            }
        }
        let intermediate_size = intermediate_size.unwrap_or(hidden_size * 4);

        // k_proj out-dim to help infer kv heads
        let mut kv_out: Option<usize> = None;
        for name in st.names() {
            if name.contains("layers.0.self_attn.k_proj.weight") {
                let t = st.tensor(name)?;
                let shape = t.shape();
                if shape.len() == 2 {
                    kv_out = Some(shape[0] as usize);
                    break;
                }
            }
        }
        let kv_out = kv_out.unwrap_or(hidden_size / 6); // heuristic fallback

        // Count layers by finding max index in model.layers.<i>.*
        let mut max_layer = 0usize;
        for name in st.names() {
            if let Some(idx) = name.find("layers.") {
                let rest = &name[idx + "layers.".len()..];
                if let Some(end) = rest.find('.') {
                    if let Ok(i) = rest[..end].parse::<usize>() {
                        if i > max_layer {
                            max_layer = i;
                        }
                    }
                }
            }
        }
        let num_hidden_layers = max_layer + 1;

        // Infer head_dim and kv_heads/attn_heads combinatorially
        let mut best: Option<(usize, usize)> = None; // (num_heads, num_kv_heads)
        let preferred_kv = [4usize, 8, 2, 16, 32];
        for &kv in &preferred_kv {
            if kv_out % kv != 0 {
                continue;
            }
            let head_dim = kv_out / kv; // must divide hidden_size
            if hidden_size % head_dim != 0 {
                continue;
            }
            let num_heads = hidden_size / head_dim;
            if num_heads >= kv && num_heads <= 64 {
                best = Some((num_heads, kv));
                break;
            }
        }
        // fallback: try kv=hidden/64 etc.
        let (num_attention_heads, num_key_value_heads) = best.unwrap_or_else(|| {
            let head_dim = 64.min(hidden_size);
            let nh = hidden_size / head_dim;
            let kvh = (kv_out / head_dim).max(2).min(nh.max(2));
            (nh.max(1), kvh)
        });

        let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            use_sliding_window: false,
            sliding_window: Some(32768),
            max_window_layers: num_hidden_layers,
            mrope_section: None,
            rope_scaling: None,
        };
        Ok(Some(text_cfg))
    }
}
