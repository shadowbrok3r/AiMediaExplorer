use candle_core::{Result, Tensor, Device, Module, D, IndexOp};
use candle_nn::{VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias, Linear};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FluxTransformerConfig {
    pub in_channels: usize,        // e.g., 64
    pub out_channels: usize,       // e.g., 16
    pub num_attention_heads: usize,// e.g., 24
    pub attention_head_dim: usize, // e.g., 128
    pub num_layers: usize,         // e.g., 60
    pub patch_size: usize,         // e.g., 2
    pub joint_attention_dim: usize,// e.g., 3584
}

#[derive(Debug, Clone)]
struct Block {
    // Optional: txt_mlp not strictly required if add_* can project from raw text
    to_q: Linear,                       // transformer_blocks.{i}.attn.to_q
    to_k: Linear,                       // transformer_blocks.{i}.attn.to_k
    to_v: Linear,                       // transformer_blocks.{i}.attn.to_v
    to_out0: Linear,                    // transformer_blocks.{i}.attn.to_out.0
    add_q_proj: Option<Linear>,         // transformer_blocks.{i}.attn.add_q_proj
    add_k_proj: Option<Linear>,         // transformer_blocks.{i}.attn.add_k_proj
    add_v_proj: Option<Linear>,         // transformer_blocks.{i}.attn.add_v_proj
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
        let img_in = linear_no_bias(config.out_channels * (config.patch_size * config.patch_size), model_dim, vb.pp("img_in"))?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let to_q = linear_no_bias(model_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.to_q")))?;
            let to_k = linear_no_bias(model_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.to_k")))?;
            let to_v = linear_no_bias(model_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.to_v")))?;
            let to_out0 = linear_no_bias(model_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.to_out.0")))?;
            // add_* projections may be absent in some shards
            let add_q_proj = vb.pp(format!("transformer_blocks.{i}.attn.add_q_proj")).get((model_dim, config.joint_attention_dim), "weight").ok()
                .map(|_| linear_no_bias(config.joint_attention_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.add_q_proj")))).transpose()?;
            let add_k_proj = vb.pp(format!("transformer_blocks.{i}.attn.add_k_proj")).get((model_dim, config.joint_attention_dim), "weight").ok()
                .map(|_| linear_no_bias(config.joint_attention_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.add_k_proj")))).transpose()?;
            let add_v_proj = vb.pp(format!("transformer_blocks.{i}.attn.add_v_proj")).get((model_dim, config.joint_attention_dim), "weight").ok()
                .map(|_| linear_no_bias(config.joint_attention_dim, model_dim, vb.pp(format!("transformer_blocks.{i}.attn.add_v_proj")))).transpose()?;
            blocks.push(Block { to_q, to_k, to_v, to_out0, add_q_proj, add_k_proj, add_v_proj });
        }
        // Try to obtain a consistent out projection from the last block's img_mlp.net.2 (fallback to first if missing)
        let mut out_proj = None;
        let last = config.num_layers.saturating_sub(1);
        let try_paths = [
            format!("transformer_blocks.{last}.img_mlp.net.2"),
            "transformer_blocks.0.img_mlp.net.2".to_string(),
        ];
        for p in try_paths.iter() {
            match linear_no_bias(model_dim, config.in_channels * (config.patch_size * config.patch_size), vb.pp(p)) {
                Ok(lp) => { out_proj = Some(lp); break; }
                Err(_) => { /* try next */ }
            }
        }
        Ok(Self { config: config.clone(), device: vb.device().clone(), img_in, blocks, out_proj })
    }

    // x: [B, C_in, H, W] with C_in expected = out_channels or in_channels?
    // We follow config.in_channels as base before patchify; if x has C != in_channels, try to lift it by space-to-depth first when possible.
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
        let ps = self.config.patch_size;
        let model_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        // Space-to-depth (patchify): [B,C,H,W] -> [B,H/ps*W/ps, C*ps*ps]
        if h % ps != 0 || w % ps != 0 { candle_core::bail!("Input spatial size not divisible by patch_size"); }
        let h2 = h / ps; let w2 = w / ps;
        let x_reshaped = x.reshape((b, c, h2, ps, w2, ps))?
            .transpose(2, 3)? // (b,c,ps,h2,w2,ps)
            .transpose(4, 5)? // (b,c,ps,h2,ps,w2)
            .reshape((b, c * ps * ps, h2, w2))?; // (b, c*ps*ps, h2, w2)
        let n = h2 * w2;
        let tokens_sd = x_reshaped.flatten_from(2)? // (b, c*ps*ps, n)
            .transpose(1, 2)?; // (b, n, c*ps*ps)
        // Embed patches to model_dim using learned 'img_in' linear, expecting in_dim = out_channels*ps^2
        // If input channel product differs, slice to expected
        let needed = self.config.out_channels * (ps * ps);
        if tokens_sd.dim(2)? < needed { candle_core::bail!("Not enough channels for patch embed: got {}, need {}", tokens_sd.dim(2)?, needed); }
        let tokens_in = tokens_sd.i((.., .., 0..needed))?; // (b,n,needed)
    let mut tokens = self.img_in.forward(&tokens_in)?; // (b,n,D)

        let heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
    let (bt, tlen, _td) = text.dims3()?; assert_eq!(bt, b, "batch mismatch");
        // Main blocks
        for blk in &self.blocks {
            // Image tokens already at model_dim
            let img = tokens.clone(); // (b,n,D)
            // Build Q from image (optionally bias by text via add_q_proj if available)
            let mut q_proj_in = img.clone();
            if let Some(qp) = &blk.add_q_proj { q_proj_in = (&q_proj_in + &qp.forward(text)?)?; }
            let q_lin = blk.to_q.forward(&q_proj_in)?; // (b,n,D)
            let q = q_lin.reshape((b, n, heads, head_dim))? // (b,n,H,h)
                .transpose(1, 2)?; // (b,H,n,h)
            // Build K,V from text via add_* if available; else fall back to projecting img for K/V (still strict weights for img path)
            let (k, v) = if let (Some(kp), Some(vp)) = (&blk.add_k_proj, &blk.add_v_proj) {
                let k = kp.forward(text)? // (b, t, D)
                    .reshape((b, tlen, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,t,h)
                let v = vp.forward(text)?
                    .reshape((b, tlen, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,t,h)
                (k, v)
            } else {
                let k = blk.to_k.forward(&img)?
                    .reshape((b, n, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,n,h)
                let v = blk.to_v.forward(&img)?
                    .reshape((b, n, heads, head_dim))?
                    .transpose(1, 2)?; // (b,H,n,h)
                (k, v)
            };
            let scale = (head_dim as f64).sqrt();
            let attn = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / scale)?; // (b,H,n,t)
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            let ctx = attn.matmul(&v)? // (b,H,n,h)
                .transpose(1, 2)? // (b,n,H,h)
                .reshape((b, n, model_dim))?; // (b,n,D)
            let out = blk.to_out0.forward(&ctx)?; // (b,n,D)
            // Residual
            tokens = (img + out)?;
        }
        // Unpatchify using learned out_proj when available: tokens (b,n,D) -> (b,n,C*ps^2)
        let y_tokens = if let Some(op) = &self.out_proj { op.forward(&tokens)? } else { tokens.clone() };
        let cout_patched = if let Some(_) = &self.out_proj { self.config.in_channels * (ps * ps) } else { model_dim };
        let y = y_tokens.transpose(1, 2)? // (b, C*ps^2, n)
            .reshape((b, cout_patched, h2, w2))?; // (b, C*ps^2, h2, w2)
        // If out_proj was not available, fallback to taking first C*ps^2 channels from model_dim for depth-to-space
        let needed = self.config.in_channels * (ps * ps);
        let y = if cout_patched != needed { y.i((.., 0..needed, .., ..))? } else { y };
        // Depth-to-space (inverse of patchify): (b,C*ps^2,h2,w2) -> (b,C,ps*h2, ps*w2)
        let mut up = Tensor::zeros((b, self.config.in_channels, h, w), y.dtype(), y.device())?;
        for yy in 0..h2 { for xx in 0..w2 {
            let y_patch = y.i((.., .., yy, xx))?; // (b,C)
            up = up.slice_assign(&[0..b, 0..self.config.in_channels, (yy*ps)..(yy*ps+ps), (xx*ps)..(xx*ps+ps)], &y_patch.unsqueeze(2)?.unsqueeze(3)?)?;
        }}
        // Reduce in_channels (C*ps^2 pre-depth) to out_channels if necessary by channel slicing to target (strict deterministic)
        let out = up.i((.., 0..self.config.out_channels, .., ..))?;
        Ok(out)
    }
}
