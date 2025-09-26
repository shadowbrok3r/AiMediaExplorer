use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder, conv2d};
use candle_transformers::models::with_tracing::{Linear, linear_no_bias};

// Lightweight ops to support building from GGUF without candle_nn::VarBuilder
#[derive(Debug, Clone)]
struct SimpleLinear {
    // Weight: [out, in]
    w: Tensor,
}

impl SimpleLinear {
    fn new_from_qvb(
        vb: &candle_transformers::quantized_var_builder::VarBuilder,
        prefix: &str,
        path: &str,
        in_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let full = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{prefix}.{path}")
        };
        let qw = vb.pp("").pp(&full).get((out_dim, in_dim), "weight")?;
        let w = qw.dequantize(device)?;
        Ok(Self { w })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., in], w: [out,in] => y: [..., out]
        let wt = self.w.transpose(0, 1)?; // [in,out]
        x.matmul(&wt)
    }
}

#[derive(Debug, Clone)]
struct SimpleLayerNorm {
    weight: Tensor, // [dim]
    bias: Tensor,   // [dim]
    eps: f64,
}

impl SimpleLayerNorm {
    fn new_from_qvb(
        vb: &candle_transformers::quantized_var_builder::VarBuilder,
        prefix: &str,
        path: &str,
        dim: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let full = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{prefix}.{path}")
        };
        let qweight = vb.pp("").pp(&full).get(dim, "weight")?;
        let weight = qweight.dequantize(device)?;
        let qbias = vb.pp("").pp(&full).get(dim, "bias")?;
        let bias = qbias.dequantize(device)?;
        Ok(Self { weight, bias, eps })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Normalize over last dim
        let x_dtype = x.dtype();
        let x32 = x.to_dtype(candle_core::DType::F32)?;
        let mean = x32.mean_keepdim(D::Minus1)?; // [...,1]
        let xv = (&x32 - &mean)?;
        let var = xv.sqr()?.mean_keepdim(D::Minus1)?;
    let eps_t = Tensor::new(self.eps as f32, x.device())?;
    // Add epsilon with explicit broadcasting to handle shapes like [...,1]
    let denom = var.broadcast_add(&eps_t)?.sqrt()?;
    let xhat = xv.broadcast_div(&denom)?; // [...,dim]
        let mut y = xhat.to_dtype(x_dtype)?;
        let mut w = self.weight.clone();
        if w.dtype() != y.dtype() { w = w.to_dtype(y.dtype())?; }
        let mut b = self.bias.clone();
        if b.dtype() != y.dtype() { b = b.to_dtype(y.dtype())?; }
        y = y.broadcast_mul(&w)?;
        y = y.broadcast_add(&b)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
struct SimpleConv1x1 {
    // Weight as [out, in]
    w: Tensor,
}

impl SimpleConv1x1 {
    fn new_from_qvb(
        vb: &candle_transformers::quantized_var_builder::VarBuilder,
        prefix: &str,
        path: &str,
        in_ch: usize,
        out_ch: usize,
        device: &Device,
    ) -> Result<Self> {
        let full = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{prefix}.{path}")
        };
        let qw4 = vb.pp("").pp(&full).get((out_ch, in_ch, 1, 1), "weight")?;
        let w4 = qw4.dequantize(device)?;
        let w2 = w4.reshape((out_ch, in_ch))?;
        Ok(Self { w: w2 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B,Cin,H,W]
        let (b, _cin, h, w) = x.dims4()?;
        let x_flat = x.flatten_from(2)?; // [B,Cin,N]
        let x_bt = x_flat.transpose(1, 2)?; // [B,N,Cin]
        let y = x_bt.matmul(&self.w.transpose(0, 1)?)?; // [B,N,Cout]
        let y = y.transpose(1, 2)?; // [B,Cout,N]
        let y = y.reshape((b, self.w.dim(0)?, h, w))?;
        Ok(y)
    }
}

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
struct SimpleBlock {
    ln1: SimpleLayerNorm,
    q_proj: SimpleLinear,
    k_proj: SimpleLinear,
    v_proj: SimpleLinear,
    o_proj: SimpleLinear,
    ln2: SimpleLayerNorm,
    fc1: SimpleLinear,
    fc2: SimpleLinear,
}

#[derive(Debug, Clone)]
pub struct QwenImageTransformer2DModel {
    pub config: QwenImageTransformerConfig,
    pub device: Device,
    // In/out 1x1 convs
    proj_in: Option<Conv2d>,
    proj_out: Option<Conv2d>,
    proj_in_simple: Option<SimpleConv1x1>,
    proj_out_simple: Option<SimpleConv1x1>,
    // Repeated blocks
    blocks: Vec<Block>,
    blocks_simple: Option<Vec<SimpleBlock>>,
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
            proj_in: Some(proj_in),
            proj_out: Some(proj_out),
            proj_in_simple: None,
            proj_out_simple: None,
            blocks,
            blocks_simple: None,
        })
    }

    // GGUF constructor: builds using dequantized weights from a GGUF file.
    // Note: gated by the optional cfg(feature = "gguf-support"). If the feature is not set, this will not compile.
    pub fn new_from_gguf(
        config: &QwenImageTransformerConfig,
        gguf_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
        use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
        // First attempt: use quantized VarBuilder (fast path)
        let qvb = match QVarBuilder::from_gguf(gguf_path, device) {
            Ok(vb) => {
                log::info!("[qwen-image-edit] GGUF opened via VarBuilder: {}", gguf_path.display());
                vb
            }
            Err(e) => {
                log::warn!("[qwen-image-edit] GGUF VarBuilder failed: {}. Trying direct GGUF reader fallback.", e);
                // Fallback: try a direct GGUF read and manual dequantization for required tensors (tolerates K-quant)
                return Self::new_from_gguf_direct(config, gguf_path, device);
            }
        };
        // Try common prefixes for the transformer scope inside GGUF
        let prefixes = ["", "transformer", "model", "diffusion_model", "net", "module", "transformer_model"];
        let model_dim = config.num_attention_heads * config.attention_head_dim;

        // Build proj_in/proj_out using simple 1x1 convs
        let mut last_err: Option<anyhow::Error> = None;
        for pfx in prefixes.iter() {
            let proj_in = match SimpleConv1x1::new_from_qvb(&qvb, pfx, "proj_in", config.in_channels, model_dim, device) {
                Ok(v) => v,
                Err(e) => { last_err = Some(e.into()); continue; }
            };
            let proj_out = match SimpleConv1x1::new_from_qvb(&qvb, pfx, "proj_out", model_dim, config.out_channels, device) {
                Ok(v) => v,
                Err(e) => { last_err = Some(e.into()); continue; }
            };

            // Build blocks
            let mut sblocks: Vec<SimpleBlock> = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let ln1 = SimpleLayerNorm::new_from_qvb(&qvb, pfx, &format!("layers.{i}.ln1"), model_dim, 1e-6, device)?;
                let q_proj = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.attn.q_proj"), model_dim, model_dim, device)?;
                let k_proj = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.attn.k_proj"), config.joint_attention_dim, model_dim, device)?;
                let v_proj = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.attn.v_proj"), config.joint_attention_dim, model_dim, device)?;
                let o_proj = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.attn.o_proj"), model_dim, model_dim, device)?;
                let ln2 = SimpleLayerNorm::new_from_qvb(&qvb, pfx, &format!("layers.{i}.ln2"), model_dim, 1e-6, device)?;
                let mlp_dim = (model_dim as f64 * 4.0) as usize;
                let fc1 = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.mlp.fc1"), model_dim, mlp_dim, device)?;
                let fc2 = SimpleLinear::new_from_qvb(&qvb, pfx, &format!("layers.{i}.mlp.fc2"), mlp_dim, model_dim, device)?;
                sblocks.push(SimpleBlock { ln1, q_proj, k_proj, v_proj, o_proj, ln2, fc1, fc2 });
            }
            return Ok(Self {
                config: config.clone(),
                device: device.clone(),
                proj_in: None,
                proj_out: None,
                proj_in_simple: Some(proj_in),
                proj_out_simple: Some(proj_out),
                blocks: Vec::new(),
                blocks_simple: Some(sblocks),
            });
        }
        if let Some(e) = last_err { Err(candle_core::Error::Msg(format!("{}", e))) } else { Err(candle_core::Error::Msg("Failed to build Qwen transformer from GGUF: no matching prefixes".to_string())) }
    }

    // Direct GGUF reader fallback tolerant to K-quant: read only the tensors we need and dequantize them.
    fn new_from_gguf_direct(
        config: &QwenImageTransformerConfig,
        gguf_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
    use candle_core::quantized::gguf_file;
        log::info!("[qwen-image-edit] GGUF direct reader: {}", gguf_path.display());
        let mut f = std::fs::File::open(gguf_path)
            .map_err(|e| candle_core::Error::Msg(format!("open gguf failed: {}", e)))?;
        let content = gguf_file::Content::read(&mut f)
            .map_err(|e| candle_core::Error::Msg(format!("read gguf failed: {}", e)))?;
        // Helper to fetch and dequantize a weight tensor by key, reshaping as needed.
        let mut load_q = |name: &str, shape: &[usize]| -> Result<Tensor> {
            // Locate tensor info
            let info = content.tensor_infos.get(name).ok_or_else(|| {
                candle_core::Error::Msg(format!("gguf missing tensor: {}", name))
            })?;
            log::info!("Info: {info:?}");
            // Read and dequantize
            let qt = content
                .tensor(&mut f, name, device)
                .map_err(|e| candle_core::Error::Msg(format!("read tensor {} failed: {}", name, e)))?;
            let t = qt.dequantize(device)?;
            let t = if !shape.is_empty() { t.reshape(shape)? } else { t };
            Ok(t)
        };

        // Try common prefixes and assemble the minimal set of tensors we need.
        let prefixes = [
            "",
            "transformer",
            "model",
            "diffusion_model",
            "net",
            "module",
            "transformer_model",
        ];
        let model_dim = config.num_attention_heads * config.attention_head_dim;
        for pfx in prefixes.iter() {
            let k = |base: &str| if pfx.is_empty() { base.to_string() } else { format!("{}.{}", pfx, base) };
            // proj_in: [model_dim, in_ch, 1,1] in file, reshape to [out,in]
            let w_in = load_q(&k("proj_in.weight"), &[model_dim, config.in_channels, 1, 1]);
            let w_out = load_q(&k("proj_out.weight"), &[config.out_channels, model_dim, 1, 1]);
            if w_in.is_err() || w_out.is_err() { continue; }
            let w_in = w_in?; let w_out = w_out?;
            let proj_in_simple = SimpleConv1x1 { w: w_in.reshape((model_dim, config.in_channels))? };
            let proj_out_simple = SimpleConv1x1 { w: w_out.reshape((config.out_channels, model_dim))? };

            // Blocks
            let mut sblocks: Vec<SimpleBlock> = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let ln1_w = load_q(&k(&format!("layers.{i}.ln1.weight")), &[model_dim])?;
                let ln1_b = load_q(&k(&format!("layers.{i}.ln1.bias")), &[model_dim])?;
                let ln1 = SimpleLayerNorm { weight: ln1_w, bias: ln1_b, eps: 1e-6 };
                let q_proj = SimpleLinear { w: load_q(&k(&format!("layers.{i}.attn.q_proj.weight")), &[model_dim, model_dim])? };
                let k_proj = SimpleLinear { w: load_q(&k(&format!("layers.{i}.attn.k_proj.weight")), &[model_dim, config.joint_attention_dim])? };
                let v_proj = SimpleLinear { w: load_q(&k(&format!("layers.{i}.attn.v_proj.weight")), &[model_dim, config.joint_attention_dim])? };
                let o_proj = SimpleLinear { w: load_q(&k(&format!("layers.{i}.attn.o_proj.weight")), &[model_dim, model_dim])? };
                let ln2_w = load_q(&k(&format!("layers.{i}.ln2.weight")), &[model_dim])?;
                let ln2_b = load_q(&k(&format!("layers.{i}.ln2.bias")), &[model_dim])?;
                let ln2 = SimpleLayerNorm { weight: ln2_w, bias: ln2_b, eps: 1e-6 };
                let mlp_dim = (model_dim as f64 * 4.0) as usize;
                let fc1 = SimpleLinear { w: load_q(&k(&format!("layers.{i}.mlp.fc1.weight")), &[mlp_dim, model_dim])? };
                let fc2 = SimpleLinear { w: load_q(&k(&format!("layers.{i}.mlp.fc2.weight")), &[model_dim, mlp_dim])? };
                sblocks.push(SimpleBlock { ln1, q_proj, k_proj, v_proj, o_proj, ln2, fc1, fc2 });
            }

            log::info!("[qwen-image-edit] GGUF direct reader built transformer using prefix '{}'", pfx);
            return Ok(Self {
                config: config.clone(),
                device: device.clone(),
                proj_in: None,
                proj_out: None,
                proj_in_simple: Some(proj_in_simple),
                proj_out_simple: Some(proj_out_simple),
                blocks: Vec::new(),
                blocks_simple: Some(sblocks),
            });
        }
        Err(candle_core::Error::Msg("Direct GGUF build failed across known prefixes".to_string()))
    }

    // Forward over latent tensor with text embeddings conditioning
    // x: [B, C_in, H, W], text: [B, T, D_text], returns [B, C_out, H, W]
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
        if let (Some(proj_in_s), Some(proj_out_s), Some(sblocks)) = (
            &self.proj_in_simple,
            &self.proj_out_simple,
            &self.blocks_simple,
        ) {
            // GGUF simple path
            let (b, _c, h, w) = x.dims4()?;
            let model_dim = self.config.num_attention_heads * self.config.attention_head_dim;
            let heads = self.config.num_attention_heads;
            let head_dim = self.config.attention_head_dim;

            // In-proj using 1x1 conv equivalent
            let mut y = proj_in_s.forward(x)?; // [B, D, H, W]
            let n = h * w;
            y = y.flatten_from(2)?; // [B, D, N]
            y = y.transpose(1, 2)?; // [B, N, D]

            let (bt, tlen, td) = text.dims3()?;
            assert_eq!(bt, b, "batch mismatch");
            assert_eq!(td, self.config.joint_attention_dim, "text dim mismatch");
            let scale = (head_dim as f64).sqrt();
            for blk in sblocks.iter() {
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
                let q = q.contiguous()?;
                let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?;
                let attn = (q.matmul(&k_t)? / scale)?;
                let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
                let v = v.contiguous()?;
                let ctx = attn.matmul(&v)?; // [B,H,N,Hd]
                let ctx = ctx.transpose(1, 2)?.reshape((b, n, model_dim))?; // [B,N,D]
                let y_new = blk.o_proj.forward(&ctx)?; // [B,N,D]
                let y_res = (y_new + &residual)?;

                let residual2 = y_res.clone();
                let y_ln2 = blk.ln2.forward(&y_res)?;
                let y_m = blk.fc1.forward(&y_ln2)?.gelu()?;
                let y_m = blk.fc2.forward(&y_m)?;
                y = (y_m + residual2)?;
            }

            // Restore spatial and out-proj
            let y = y.transpose(1, 2)?.reshape((b, model_dim, h, w))?;
            let y = proj_out_s.forward(&y)?; // [B,C_out,H,W]
            return Ok(y);
        }
        let (b, _c, h, w) = x.dims4()?;
        let model_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        let heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;

        // In-proj and flatten spatial to tokens [B, N, D]
    let mut y = self.proj_in.as_ref().expect("proj_in").forward(x)?; // [B, D, H, W]
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
        let y = self.proj_out.as_ref().expect("proj_out").forward(&y)?; // [B,C_out,H,W]
        Ok(y)
    }
}
