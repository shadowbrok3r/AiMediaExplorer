use candle_core::{D, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{Linear, linear_no_bias};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FluxTransformerConfig {
    pub in_channels: usize,         // e.g., 64
    pub out_channels: usize,        // e.g., 16
    pub num_attention_heads: usize, // e.g., 24
    pub attention_head_dim: usize,  // e.g., 128
    pub num_layers: usize,          // e.g., 60
    pub patch_size: usize,          // e.g., 2
    pub joint_attention_dim: usize, // e.g., 3584
    #[serde(default)]
    pub attn_chunk_size: Option<usize>, // Optional: chunk token length to limit peak memory
}

#[derive(Debug, Clone)]
struct Block {
    // Optional: txt_mlp not strictly required if add_* can project from raw text
    to_q: Linear,               // transformer_blocks.{i}.attn.to_q
    to_k: Linear,               // transformer_blocks.{i}.attn.to_k
    to_v: Linear,               // transformer_blocks.{i}.attn.to_v
    to_out0: Linear,            // transformer_blocks.{i}.attn.to_out.0
    add_q_proj: Option<Linear>, // transformer_blocks.{i}.attn.add_q_proj
    add_k_proj: Option<Linear>, // transformer_blocks.{i}.attn.add_k_proj
    add_v_proj: Option<Linear>, // transformer_blocks.{i}.attn.add_v_proj
}

#[derive(Debug, Clone)]
pub struct FluxTransformer2DModel {
    pub config: FluxTransformerConfig,
    pub device: Device,
    // Patch embedding linear: uses 'img_in' with in_dim = (out_channels * ps^2) to model_dim
    img_in: Linear,
    blocks: Vec<Block>,
    // Learned output projection from model_dim to (in_channels * ps^2), taken from a real block's img_mlp.net.2
    out_proj: Option<Linear>,
}

impl FluxTransformer2DModel {
    pub fn new(config: &FluxTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let model_dim = config.num_attention_heads * config.attention_head_dim; // e.g., 3072
        // Patch embedding linear expected as 'img_in': [model_dim, out_channels*ps^2]
        let img_in = linear_no_bias(
            config.out_channels * (config.patch_size * config.patch_size),
            model_dim,
            vb.pp("img_in"),
        )?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let to_q = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("transformer_blocks.{i}.attn.to_q")),
            )?;
            let to_k = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("transformer_blocks.{i}.attn.to_k")),
            )?;
            let to_v = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("transformer_blocks.{i}.attn.to_v")),
            )?;
            let to_out0 = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("transformer_blocks.{i}.attn.to_out.0")),
            )?;
            // add_* projections may be absent in some shards
            let add_q_proj = vb
                .pp(format!("transformer_blocks.{i}.attn.add_q_proj"))
                .get((model_dim, config.joint_attention_dim), "weight")
                .ok()
                .map(|_| {
                    linear_no_bias(
                        config.joint_attention_dim,
                        model_dim,
                        vb.pp(format!("transformer_blocks.{i}.attn.add_q_proj")),
                    )
                })
                .transpose()?;
            let add_k_proj = vb
                .pp(format!("transformer_blocks.{i}.attn.add_k_proj"))
                .get((model_dim, config.joint_attention_dim), "weight")
                .ok()
                .map(|_| {
                    linear_no_bias(
                        config.joint_attention_dim,
                        model_dim,
                        vb.pp(format!("transformer_blocks.{i}.attn.add_k_proj")),
                    )
                })
                .transpose()?;
            let add_v_proj = vb
                .pp(format!("transformer_blocks.{i}.attn.add_v_proj"))
                .get((model_dim, config.joint_attention_dim), "weight")
                .ok()
                .map(|_| {
                    linear_no_bias(
                        config.joint_attention_dim,
                        model_dim,
                        vb.pp(format!("transformer_blocks.{i}.attn.add_v_proj")),
                    )
                })
                .transpose()?;
            blocks.push(Block {
                to_q,
                to_k,
                to_v,
                to_out0,
                add_q_proj,
                add_k_proj,
                add_v_proj,
            });
        }
        // Try to obtain a consistent out projection from the last block's img_mlp.net.2 (fallback to first if missing)
        let mut out_proj = None;
        let last = config.num_layers.saturating_sub(1);
        let try_paths = [
            format!("transformer_blocks.{last}.img_mlp.net.2"),
            "transformer_blocks.0.img_mlp.net.2".to_string(),
        ];
        for p in try_paths.iter() {
            match linear_no_bias(
                model_dim,
                config.in_channels * (config.patch_size * config.patch_size),
                vb.pp(p),
            ) {
                Ok(lp) => {
                    out_proj = Some(lp);
                    break;
                }
                Err(_) => { /* try next */ }
            }
        }
        Ok(Self {
            config: config.clone(),
            device: vb.device().clone(),
            img_in,
            blocks,
            out_proj,
        })
    }

    // x: [B, C_in, H, W] with C_in expected = out_channels or in_channels?
    // We follow config.in_channels as base before patchify; if x has C != in_channels, try to lift it by space-to-depth first when possible.
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let ps = self.config.patch_size;
        let model_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        // Space-to-depth (patchify): [B,C,H,W] -> [B,H/ps*W/ps, C*ps*ps]
        if h % ps != 0 || w % ps != 0 {
            candle_core::bail!("Input spatial size not divisible by patch_size");
        }
        let h2 = h / ps;
        let w2 = w / ps;
        let x_reshaped = x
            .reshape((b, c, h2, ps, w2, ps))?
            .transpose(2, 3)? // (b,c,ps,h2,w2,ps)
            .transpose(4, 5)? // (b,c,ps,h2,ps,w2)
            .reshape((b, c * ps * ps, h2, w2))?; // (b, c*ps*ps, h2, w2)
        let n = h2 * w2;
        let tokens_sd = x_reshaped
            .flatten_from(2)? // (b, c*ps*ps, n)
            .transpose(1, 2)?; // (b, n, c*ps*ps)
        // Embed patches to model_dim using learned 'img_in' linear, expecting in_dim = out_channels*ps^2
        // If input channel product differs, slice to expected
        let needed_in = self.config.out_channels * (ps * ps);
        if tokens_sd.dim(2)? != needed_in {
            if tokens_sd.dim(2)? < needed_in {
                candle_core::bail!(
                    "Not enough channels for patch embed: got {}, need {}",
                    tokens_sd.dim(2)?,
                    needed_in
                );
            }
            // Trim extra channels deterministically to limit memory
            log::warn!(
                "[flux] trimming patch channels from {} to {}",
                tokens_sd.dim(2)?,
                needed_in
            );
        }
        let tokens_in = tokens_sd.i((.., .., 0..needed_in))?; // (b,n,needed_in)
        let mut tokens = self.img_in.forward(&tokens_in)?; // (b,n,D)

        let heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
        let (bt, tlen, _td) = text.dims3()?;
        assert_eq!(bt, b, "batch mismatch");

        // Main blocks
        for blk in &self.blocks {
            // Image tokens already at model_dim
            let img = tokens.clone(); // (b,n,D)
            // Build K,V from text via add_* if available; else fall back to projecting img for K/V (self-attention)
            let (k, v) = if let (Some(kp), Some(vp)) = (&blk.add_k_proj, &blk.add_v_proj) {
                let k = kp
                    .forward(text)? // (b, t, D)
                    .reshape((b, tlen, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,t,h)
                let v = vp
                    .forward(text)?
                    .reshape((b, tlen, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,t,h)
                (k, v)
            } else {
                let k = blk
                    .to_k
                    .forward(&img)?
                    .reshape((b, n, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,n,h)
                let v = blk
                    .to_v
                    .forward(&img)?
                    .reshape((b, n, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,n,h)
                (k, v)
            };
            let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?; // (b,H,h,Lkv)
            let v = v.contiguous()?; // (b,H,Lkv,h)
            let scale = (head_dim as f64).sqrt();

            // Prepare Q input, optionally biased by pooled text through add_q_proj
            let mut q_proj_in = img.clone(); // (b,n,D)
            if let Some(qp) = &blk.add_q_proj {
                // Pool text across sequence to avoid shape mismatch and broadcast to tokens
                let txt_pooled = text.mean(D::Minus2)?; // (b, joint_attention_dim)
                let txt_bias = qp.forward(&txt_pooled)?; // (b, D)
                let txt_bias = txt_bias.unsqueeze(1)?.expand((b, n, model_dim))?; // (b,n,D)
                q_proj_in = (&q_proj_in + &txt_bias)?;
            }

            // Chunk along token dimension to cap peak memory
            let chunk = self.config.attn_chunk_size.unwrap_or(2048).min(n);
            if chunk >= n {
                // Fast path: original behavior
                let q_lin = blk.to_q.forward(&q_proj_in)?; // (b,n,D)
                let q = q_lin
                    .reshape((b, n, heads, head_dim))? // (b,n,H,h)
                    .transpose(1, 2)? // (b,H,n,h)
                    .contiguous()?;
                let attn = (q.matmul(&k_t)? / scale)?; // (b,H,n,Lkv)
                let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
                let ctx = attn
                    .matmul(&v)? // (b,H,n,h)
                    .transpose(1, 2)? // (b,n,H,h)
                    .reshape((b, n, model_dim))?; // (b,n,D)
                let out = blk.to_out0.forward(&ctx)?; // (b,n,D)
                tokens = (img + out)?;
            } else {
                // Chunked path
                let q_lin_full = blk.to_q.forward(&q_proj_in)?; // (b,n,D)
                let mut acc = Tensor::zeros((b, n, model_dim), img.dtype(), img.device())?;
                let mut start = 0;
                while start < n {
                    let end = (start + chunk).min(n);
                    let q_lin_chunk = q_lin_full.i((.., start..end, ..))?; // (b,chunk,D)
                    let q = q_lin_chunk
                        .reshape((b, end - start, heads, head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?; // (b,H,chunk,h)
                    let attn = (q.matmul(&k_t)? / scale)?; // (b,H,chunk,Lkv)
                    let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
                    let ctx = attn
                        .matmul(&v)? // (b,H,chunk,h)
                        .transpose(1, 2)? // (b,chunk,H,h)
                        .reshape((b, end - start, model_dim))?; // (b,chunk,D)
                    acc = acc.slice_assign(&[0..b, start..end, 0..model_dim], &ctx)?;
                    start = end;
                }
                let out = blk.to_out0.forward(&acc)?; // (b,n,D)
                tokens = (img + out)?;
            }
        }

        let y_tokens = if let Some(op) = &self.out_proj {
            op.forward(&tokens)?
        } else {
            tokens.clone()
        };
        let cout_patched = if let Some(_) = &self.out_proj {
            self.config.in_channels * (ps * ps)
        } else {
            model_dim
        };
        let y = y_tokens
            .transpose(1, 2)? // (b, C*ps^2, n)
            .reshape((b, cout_patched, h2, w2))?; // (b, C*ps^2, h2, w2)
        // If out_proj was not available, fallback to taking first C*ps^2 channels from model_dim for depth-to-space
        let needed_out = self.config.in_channels * (ps * ps);
        let y = if cout_patched != needed_out {
            y.i((.., 0..needed_out, .., ..))?
        } else {
            y
        };
        // Depth-to-space via reshape+permute: (b,C*ps^2,h2,w2) -> (b,C,ps,h2,ps,w2) -> (b,C,h2,ps,w2,ps) -> (b,C,h,w)
        let up = y
            .reshape((b, self.config.in_channels, ps, ps, h2, w2))? // (b,C,ps,ps,h2,w2)
            .transpose(2, 4)? // (b,C,h2,ps,ps,w2)
            .transpose(3, 5)? // (b,C,h2,w2,ps,ps)
            .reshape((b, self.config.in_channels, h, w))?; // (b,C,h,w)
        // Reduce in_channels to out_channels if necessary by channel slicing
        let out = up.i((.., 0..self.config.out_channels, .., ..))?;
        Ok(out)
    }
}
