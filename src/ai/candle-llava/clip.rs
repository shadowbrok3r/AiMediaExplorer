use candle_core::{DType, IndexOp, Result, Shape, Tensor, D};
use candle_nn::{Conv2dConfig, Module};
use candle_transformers::models::clip::{
    text_model::Activation, vision_model::ClipVisionConfig, EncoderConfig,
};

// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/clip modify forward so it can output last 2 layers of hidden_states

#[derive(Clone, Debug)]
struct ClipAttention {
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    q_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: candle_nn::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let embed_dim = c.embed_dim();
        let num_attention_heads = c.num_attention_heads();
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vs.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vs.pp("v_proj"))?;
        let q_proj = candle_nn::linear(embed_dim, embed_dim, vs.pp("q_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let t_total = std::time::Instant::now();
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        log::debug!("[attn] start bsz={bsz} seq_len={seq_len} embed_dim={embed_dim} heads={} head_dim={}", self.num_attention_heads, self.head_dim);

        let t_q = std::time::Instant::now();
        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        log::debug!("[attn] q_proj {:.2}ms", t_q.elapsed().as_secs_f32()*1000.0);
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let t_shape = std::time::Instant::now();
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        log::debug!("[attn] q reshape+cast {:.2}ms", t_shape.elapsed().as_secs_f32()*1000.0);

        let t_k = std::time::Instant::now();
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        log::debug!("[attn] k path {:.2}ms", t_k.elapsed().as_secs_f32()*1000.0);

        let t_v = std::time::Instant::now();
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        log::debug!("[attn] v path {:.2}ms", t_v.elapsed().as_secs_f32()*1000.0);

        let t_scores = std::time::Instant::now();
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        log::debug!("[attn] qk.matmul {:.2}ms", t_scores.elapsed().as_secs_f32()*1000.0);

        let src_len = key_states.dim(1)?;
        let t_mask = std::time::Instant::now();
        let attn_weights = if let Some(causal_attention_mask) = causal_attention_mask {
            attn_weights
                .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
                .broadcast_add(causal_attention_mask)?
                .reshape((bsz * self.num_attention_heads, seq_len, src_len))?
        } else {
            attn_weights
        };
        log::debug!("[attn] mask+broadcast {:.2}ms", t_mask.elapsed().as_secs_f32()*1000.0);

        let t_softmax = std::time::Instant::now();
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        log::debug!("[attn] softmax {:.2}ms", t_softmax.elapsed().as_secs_f32()*1000.0);

        let t_out = std::time::Instant::now();
        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        log::debug!("[attn] weighted value reshape+transpose {:.2}ms", t_out.elapsed().as_secs_f32()*1000.0);
        let r = self.out_proj.forward(&attn_output);
        log::debug!("[attn] done total {:.2}ms", t_total.elapsed().as_secs_f32()*1000.0);
        r
    }
}

#[derive(Clone, Debug)]
struct ClipMlp {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: candle_nn::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let fc1 = candle_nn::linear(c.embed_dim(), c.intermediate_size(), vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(c.intermediate_size(), c.embed_dim(), vs.pp("fc2"))?;

        Ok(ClipMlp {
            fc1,
            fc2,
            activation: c.activation(),
        })
    }
}

impl ClipMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Clone, Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: candle_nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: candle_nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: candle_nn::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.pp("self_attn"), c)?;
        let layer_norm1 = candle_nn::layer_norm(c.embed_dim(), 1e-5, vs.pp("layer_norm1"))?;
        let mlp = ClipMlp::new(vs.pp("mlp"), c)?;
        let layer_norm2 = candle_nn::layer_norm(c.embed_dim(), 1e-5, vs.pp("layer_norm2"))?;

        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let t_total = std::time::Instant::now();
        log::debug!("[layer] enter");
        let residual = xs;
        let t_ln1 = std::time::Instant::now();
        let xs = self.layer_norm1.forward(xs)?;
        log::debug!("[layer] ln1 {:.2}ms", t_ln1.elapsed().as_secs_f32()*1000.0);
        let t_attn = std::time::Instant::now();
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        log::debug!("[layer] attn {:.2}ms", t_attn.elapsed().as_secs_f32()*1000.0);
        let t_res1 = std::time::Instant::now();
        let xs = (xs + residual)?;
        log::debug!("[layer] add1 {:.2}ms", t_res1.elapsed().as_secs_f32()*1000.0);

        let residual = &xs;
        let t_ln2 = std::time::Instant::now();
        let xs = self.layer_norm2.forward(&xs)?;
        log::debug!("[layer] ln2 {:.2}ms", t_ln2.elapsed().as_secs_f32()*1000.0);
        let t_mlp = std::time::Instant::now();
        let xs = self.mlp.forward(&xs)?;
        log::debug!("[layer] mlp {:.2}ms", t_mlp.elapsed().as_secs_f32()*1000.0);
        let t_res2 = std::time::Instant::now();
        let r = xs + residual;
        log::debug!("[layer] add2 {:.2}ms total={:.2}ms", t_res2.elapsed().as_secs_f32()*1000.0, t_total.elapsed().as_secs_f32()*1000.0);
        r
    }
}

#[derive(Clone, Debug)]
pub struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    pub fn new(vs: candle_nn::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers() {
            let layer = ClipEncoderLayer::new(vs.pp(&index.to_string()), c)?;
            layers.push(layer)
        }
        Ok(ClipEncoder { layers })
    }

    pub fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }
    pub fn output_hidden_states(
        &self,
        xs: &Tensor,
        causal_attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        log::debug!("ClipEncoder::output_hidden_states");
        let mut xs = xs.clone();
        let mut hidden_states = Vec::new();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
            hidden_states.push(xs.clone());
        }
        Ok(hidden_states)
    }
}

#[derive(Clone, Debug)]
struct ClipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: candle_nn::Embedding,
    patch_size: usize,
}

impl ClipVisionEmbeddings {
    fn new(vs: candle_nn::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        // originally nn.Parameter
        let class_embedding = if vs.contains_tensor("class_embedding") {
            vs.get(c.embed_dim, "class_embedding")?
        } else {
            Tensor::randn(0f32, 1f32, c.embed_dim, vs.device())?
        };

        let num_patches = (c.image_size / c.patch_size).pow(2);
        log::debug!(
            "[vision.embeddings] image_size={} patch_size={} -> num_patches={}",
            c.image_size, c.patch_size, num_patches
        );
        // First assume there's a separate class position (num_patches + 1). If the checkpoint
        // actually only stored patch positions (common in some variants) we retry without +1.
        let mut num_positions_with_cls = num_patches + 1;
        let mut position_ids = Tensor::arange(0, num_positions_with_cls as i64, vs.device())?;

        let conv2dconfig = Conv2dConfig {
            stride: c.patch_size,
            ..Default::default()
        };
        let position_embedding = match candle_nn::embedding(
            num_positions_with_cls,
            c.embed_dim,
            vs.pp("position_embedding"),
        ) {
            Ok(pe) => {
                log::debug!(
                    "[vision.embeddings] loaded position_embedding with cls row: rows={} dim={}",
                    num_positions_with_cls, c.embed_dim
                );
                pe
            }
            Err(e) => {
                log::debug!(
                    "[vision.embeddings] fallback: could not load position_embedding with cls row ({}). Retrying without cls row.",
                    e
                );
                // Retry without class position.
                num_positions_with_cls = num_patches; // rename semantics: now no cls row.
                position_ids = Tensor::arange(0, num_positions_with_cls as i64, vs.device())?;
                let pe = candle_nn::embedding(
                    num_positions_with_cls,
                    c.embed_dim,
                    vs.pp("position_embedding"),
                )?;
                log::debug!(
                    "[vision.embeddings] loaded position_embedding WITHOUT cls row: rows={} dim={}",
                    num_positions_with_cls, c.embed_dim
                );
                pe
            }
        };
        let patch_embedding = candle_nn::conv2d_no_bias(
            c.num_channels,
            c.embed_dim,
            c.patch_size,
            conv2dconfig,
            vs.pp("patch_embedding"),
        )?;
        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
            patch_size: c.patch_size,
        })
    }
}

impl Module for ClipVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims();
        let t_patch = std::time::Instant::now();
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        log::debug!("[emb] patch_embedding+flatten+transpose {:.2}ms shape={:?}", t_patch.elapsed().as_secs_f32()*1000.0, patch_embeds.shape());
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let t_cls = std::time::Instant::now();
        let class_embeds = self.class_embedding.expand(shape)?;
        let target_dtype = patch_embeds.dtype();
        let class_embeds = if class_embeds.dtype() != target_dtype {
            log::info!(
                "[vision.embeddings.forward] casting class_embeds {:?} -> {:?}",
                class_embeds.dtype(),
                target_dtype
            );
            class_embeds.to_dtype(target_dtype)?
        } else {
            class_embeds
        };
        log::debug!("[emb] class_expand+cast {:.2}ms", t_cls.elapsed().as_secs_f32()*1000.0);
        log::info!(
            "[vision.embeddings.forward] cat class/patch dtypes class={:?} patch={:?}",
            class_embeds.dtype(),
            patch_embeds.dtype()
        );
        let t_cat = std::time::Instant::now();
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        log::debug!("[emb] cat class+patch {:.2}ms", t_cat.elapsed().as_secs_f32()*1000.0);
        let t_pos = std::time::Instant::now();
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        let position_embedding = if position_embedding.dtype() != embeddings.dtype() {
            log::info!(
                "[vision.embeddings.forward] casting position_embedding {:?} -> {:?}",
                position_embedding.dtype(),
                embeddings.dtype()
            );
            position_embedding.to_dtype(embeddings.dtype())?
        } else { position_embedding };
        log::debug!("[emb] pos_embed forward+cast {:.2}ms", t_pos.elapsed().as_secs_f32()*1000.0);
        // Handle off-by-one mismatches between position embeddings and token embeddings.
    let (_emb_bsz, emb_tokens, _) = embeddings.dims3()?; // _emb_bsz kept for potential future per-batch diagnostics
        let (pos_tokens, _) = position_embedding.dims2()?;
        let t_align = std::time::Instant::now();
        let position_embedding = if pos_tokens == emb_tokens {
            position_embedding
        } else if pos_tokens + 1 == emb_tokens {
            // Pad one extra row (e.g. missing class token position).
            let pad_row = position_embedding.i((pos_tokens - 1, ..))?; // reuse last row
            let pad_row = pad_row.unsqueeze(0)?;
            log::info!(
                "[vision.embeddings.forward] padding position_embedding: pos_tokens={} emb_tokens={}",
                pos_tokens, emb_tokens
            );
            Tensor::cat(&[pad_row, position_embedding], 0)?
        } else if pos_tokens == emb_tokens + 1 {
            // Extra row (e.g. class token included twice) -> slice first row.
            log::info!(
                "[vision.embeddings.forward] slicing extra row in position_embedding: pos_tokens={} emb_tokens={}",
                pos_tokens, emb_tokens
            );
            position_embedding.i(1..)?
        } else {
            log::info!(
                "[vision.embeddings.forward] WARNING unmatched position embedding sizes: pos_tokens={} emb_tokens={}",
                pos_tokens, emb_tokens
            );
            if pos_tokens < emb_tokens {
                // Pad by repeating last row.
                let diff = emb_tokens - pos_tokens;
                let last_row = position_embedding.i((pos_tokens - 1, ..))?; // [dim]
                let last_row = last_row.unsqueeze(0)?; // [1, dim]
                // Broadcast last_row to required diff length.
                let pad = last_row.broadcast_as((diff, last_row.shape().dims()[1]))?;
                Tensor::cat(&[position_embedding, pad], 0)?
            } else {
                // Truncate extra rows conservatively keeping first emb_tokens rows.
                position_embedding.i(0..emb_tokens)?
            }
        };
        log::debug!("[emb] pos_align branch {:.2}ms", t_align.elapsed().as_secs_f32()*1000.0);
        let t_add = std::time::Instant::now();
        let r = embeddings.broadcast_add(&position_embedding);
        log::debug!("[emb] broadcast_add {:.2}ms", t_add.elapsed().as_secs_f32()*1000.0);
        r
    }
}

#[derive(Clone, Debug)]
pub struct ClipVisionTransformerWithHiddenStates {
    embeddings: ClipVisionEmbeddings,
    encoder: ClipEncoder,
    pre_layer_norm: Option<candle_nn::LayerNorm>,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ClipVisionTransformerWithHiddenStates {
    pub fn new(vs: candle_nn::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::new(vs.pp("embeddings"), c)?;
        // Some checkpoints use "pre_layrnorm", others "pre_layernorm" – and some have none.
        let pre_layer_norm = match candle_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("pre_layrnorm")) {
            Ok(ln) => {
                log::debug!("[vision.transformer] loaded pre_layer_norm path=pre_layrnorm");
                Some(ln)
            }
            Err(e1) => {
                log::debug!(
                    "[vision.transformer] fallback pre_layer_norm: pre_layrnorm missing ({}) trying pre_layernorm",
                    e1
                );
                match candle_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("pre_layernorm")) {
                    Ok(ln2) => {
                        log::debug!("[vision.transformer] loaded pre_layer_norm path=pre_layernorm");
                        Some(ln2)
                    }
                    Err(e2) => {
                        log::debug!(
                            "[vision.transformer] WARNING no pre-layer norm found (pre_layrnorm / pre_layernorm). Proceeding without it. Errors: {}; {}",
                            e1, e2
                        );
                        None
                    }
                }
            }
        };
        let encoder = ClipEncoder::new(vs.pp("encoder"), &EncoderConfig::Vision(c.clone()))?;
        let final_layer_norm = candle_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            pre_layer_norm,
        })
    }
    
    pub fn output_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        log::debug!("ClipVisionTransformerWithHiddenStates::output_hidden_states");
        //clearly we can optimize memory use if we are sure the select_layer is either -1 or -2. Keep the same behavior as the original python code.
        let hidden_states = pixel_values.apply(&self.embeddings)?;
        log::debug!("pixel_values.apply(&self.embeddings)?");
        let hidden_states = if let Some(ln) = &self.pre_layer_norm {
            log::debug!("Applying layer norm to hidden states");
            hidden_states.apply(ln)?
        } else {
            hidden_states
        };

        log::debug!("self.encoder.output_hidden_states");
        let mut result = self.encoder.output_hidden_states(&hidden_states, None)?;
        let encoder_outputs = result.last().unwrap();
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        result.push(self.final_layer_norm.forward(&pooled_output)?.clone());
        Ok(result)
    }
}

impl Module for ClipVisionTransformerWithHiddenStates {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = pixel_values.apply(&self.embeddings)?;
        let hidden_states = if let Some(ln) = &self.pre_layer_norm {
            hidden_states.apply(ln)?
        } else {
            hidden_states
        };

        let encoder_outputs = self.encoder.forward(&hidden_states, None)?;
        // https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L787
        // pooled_output = encoder_outputs[:, 0, :]
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        self.final_layer_norm.forward(&pooled_output)
    }
}

pub fn clip_vit_large_patch14_336() -> ClipVisionConfig {
    ClipVisionConfig {
        embed_dim: 1024,
        activation: candle_transformers::models::clip::text_model::Activation::QuickGelu,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        projection_dim: 768,
        num_channels: 3,
        image_size: 336,
        patch_size: 14,
    }
}
