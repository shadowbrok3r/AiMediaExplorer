use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder, conv2d};
use candle_transformers::models::with_tracing::{Linear, linear_no_bias};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QwenImageTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub patch_size: usize,
    pub joint_attention_dim: usize,
}

#[derive(Debug, Clone)]
struct Block {
    ln1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[derive(Debug, Clone)]
pub struct QwenImageTransformer2DModel {
    pub config: QwenImageTransformerConfig,
    pub device: Device,
    // In/out 1x1 convs
    proj_in: Conv2d,
    proj_out: Conv2d,
    // Repeated blocks
    blocks: Vec<Block>,
}

impl QwenImageTransformer2DModel {
    pub fn new(config: &QwenImageTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let model_dim = config.num_attention_heads * config.attention_head_dim;
        let proj_in = conv2d(
            config.in_channels,
            model_dim,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj_in"),
        )?;
        let proj_out = conv2d(
            model_dim,
            config.out_channels,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj_out"),
        )?;
        let mlp_dim = (model_dim as f64 * 4.0) as usize;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let ln1 = candle_nn::layer_norm(model_dim, 1e-6, vb.pp(format!("layers.{i}.ln1")))?;
            let q_proj = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("layers.{i}.attn.q_proj")),
            )?;
            let k_proj = linear_no_bias(
                config.joint_attention_dim,
                model_dim,
                vb.pp(format!("layers.{i}.attn.k_proj")),
            )?;
            let v_proj = linear_no_bias(
                config.joint_attention_dim,
                model_dim,
                vb.pp(format!("layers.{i}.attn.v_proj")),
            )?;
            let o_proj = linear_no_bias(
                model_dim,
                model_dim,
                vb.pp(format!("layers.{i}.attn.o_proj")),
            )?;
            let ln2 = candle_nn::layer_norm(model_dim, 1e-6, vb.pp(format!("layers.{i}.ln2")))?;
            let fc1 = linear_no_bias(model_dim, mlp_dim, vb.pp(format!("layers.{i}.mlp.fc1")))?;
            let fc2 = linear_no_bias(mlp_dim, model_dim, vb.pp(format!("layers.{i}.mlp.fc2")))?;
            blocks.push(Block {
                ln1,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ln2,
                fc1,
                fc2,
            });
        }
        Ok(Self {
            config: config.clone(),
            device: vb.device().clone(),
            proj_in,
            proj_out,
            blocks,
        })
    }

    // Forward over latent tensor with text embeddings conditioning
    // x: [B, C_in, H, W], text: [B, T, D_text], returns [B, C_out, H, W]
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;
        let model_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        let heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;

        // In-proj and flatten spatial to tokens [B, N, D]
        let mut y = self.proj_in.forward(x)?; // [B, D, H, W]
        let n = h * w;
        y = y.flatten_from(2)?; // [B, D, N]
        y = y.transpose(1, 2)?; // [B, N, D]

        // Apply each block
        let (bt, tlen, td) = text.dims3()?;
        assert_eq!(bt, b, "batch mismatch");
        assert_eq!(td, self.config.joint_attention_dim, "text dim mismatch");
        for blk in &self.blocks {
            // Cross-attention to text tokens
            let residual = y.clone();
            let y_ln = blk.ln1.forward(&y)?; // [B,N,D]
            let q = blk
                .q_proj
                .forward(&y_ln)? // [B,N,D]
                .reshape((b, n, heads, head_dim))? // [B,N,H,Hd]
                .transpose(1, 2)?; // [B,H,N,Hd]
            let k = blk
                .k_proj
                .forward(text)? // [B,T,D]
                .reshape((b, tlen, heads, head_dim))?
                .transpose(1, 2)?; // [B,H,T,Hd]
            let v = blk
                .v_proj
                .forward(text)?
                .reshape((b, tlen, heads, head_dim))?
                .transpose(1, 2)?; // [B,H,T,Hd]
            let scale = (head_dim as f64).sqrt();
            // Ensure contiguous tensors before matmul operations
            let q = q.contiguous()?; // [B,H,N,Hd]
            let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?; // [B,H,Hd,T]
            let attn = (q.matmul(&k_t)? / scale)?; // [B,H,N,T]
            let attn = candle_nn::ops::softmax_last_dim(&attn)?; // [B,H,N,T]
            let attn = attn.contiguous()?;
            let v = v.contiguous()?;
            let ctx = attn.matmul(&v)?; // [B,H,N,Hd]
            let ctx = ctx
                .transpose(1, 2)? // [B,N,H,Hd]
                .reshape((b, n, model_dim))?; // [B,N,D]
            let y_new = blk.o_proj.forward(&ctx)?; // [B,N,D]
            let y_res = (y_new + &residual)?; // residual

            // MLP
            let residual2 = y_res.clone();
            let y_ln2 = blk.ln2.forward(&y_res)?;
            let y_m = blk.fc1.forward(&y_ln2)?.gelu()?;
            let y_m = blk.fc2.forward(&y_m)?;
            y = (y_m + residual2)?;
        }

        // Restore spatial and out-proj
        let y = y
            .transpose(1, 2)? // [B,D,N]
            .reshape((b, model_dim, h, w))?; // [B,D,H,W]
        let y = self.proj_out.forward(&y)?; // [B,C_out,H,W]
        Ok(y)
    }
}
