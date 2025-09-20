use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder, conv_transpose2d,
    conv2d,
};
use safetensors::SafeTensors;
use safetensors::tensor::{TensorView, serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_down_block_types")]
    pub down_block_types: Vec<String>,
    #[serde(default = "default_up_block_types")]
    pub up_block_types: Vec<String>,
    #[serde(default)]
    pub sample_size: Option<usize>,
    #[serde(default)]
    pub scaling_factor: Option<f32>,
}

fn default_block_out_channels() -> Vec<usize> {
    vec![128, 256, 512, 512]
}
fn default_down_block_types() -> Vec<String> {
    vec!["DownEncoderBlock2D".into(); 4]
}
fn default_up_block_types() -> Vec<String> {
    vec!["UpDecoderBlock2D".into(); 4]
}

#[derive(Debug, Clone)]
struct ResBlock2D {
    conv1: Conv2d,
    conv2: Conv2d,
    skip: Option<Conv2d>,
}

impl ResBlock2D {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder, prefix: &str) -> Result<Self> {
        let conv1 = conv2d(
            in_c,
            out_c,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp(format!("{}.conv1", prefix)),
        )?;
        let conv2 = conv2d(
            out_c,
            out_c,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp(format!("{}.conv2", prefix)),
        )?;
        let skip = if in_c != out_c {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Conv2dConfig {
                    stride: 1,
                    padding: 0,
                    ..Default::default()
                },
                vb.pp(format!("{}.skip", prefix)),
            )?)
        } else {
            None
        };
        Ok(Self { conv1, conv2, skip })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(s) = &self.skip {
            s.forward(x)?
        } else {
            x.clone()
        };
        let y = self.conv1.forward(x)?.silu()?;
        let y = self.conv2.forward(&y)?;
        (y + residual)?.silu()
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKL {
    pub config: VaeConfig,
    // Encoder/Decoder
    enc_in: Conv2d,
    down_blocks: Vec<Conv2d>,
    down_res: Vec<ResBlock2D>,
    to_mu: Conv2d,
    to_logvar: Conv2d,
    // Map latent -> feature channels for decoder input
    from_latent: Conv2d,
    up_blocks: Vec<ConvTranspose2d>,
    up_res: Vec<ResBlock2D>,
    dec_out: Conv2d,
    pub device: Device,
}

impl AutoencoderKL {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let enc_in = conv2d(
            config.in_channels,
            config.block_out_channels.get(0).copied().unwrap_or(128),
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp("encoder.conv_in"),
        )?;
        // Downsample stack: stride-2 convs
        let mut down_blocks: Vec<Conv2d> = Vec::new();
        let mut in_c = config.block_out_channels.get(0).copied().unwrap_or(128);
        let mut down_res: Vec<ResBlock2D> = Vec::new();
        for (i, &out_c) in config.block_out_channels.iter().enumerate().skip(1) {
            let conv = conv2d(
                in_c,
                out_c,
                3,
                Conv2dConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(format!("encoder.down.{}", i - 1)),
            )?;
            down_blocks.push(conv);
            let res = ResBlock2D::new(out_c, out_c, vb.pp("encoder"), &format!("res.{}", i - 1))?;
            down_res.push(res);
            in_c = out_c;
        }
        // Heads for mean/logvar
        let to_mu = conv2d(
            in_c,
            config.latent_channels,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("encoder.to_mu"),
        )?;
        let to_logvar = conv2d(
            in_c,
            config.latent_channels,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("encoder.to_logvar"),
        )?;

        // Map latent to decoder start channels
        let start_c = *config.block_out_channels.last().unwrap_or(&in_c);
        let from_latent = conv2d(
            config.latent_channels,
            start_c,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("decoder.from_latent"),
        )?;

        // Upsample stack: stride-2 deconvs reversing channel sizes
        let mut up_blocks: Vec<ConvTranspose2d> = Vec::new();
        let chs: Vec<usize> = config.block_out_channels.clone();
        let mut up_res: Vec<ResBlock2D> = Vec::new();
        if chs.len() >= 2 {
            // reverse transitions: last->prev ...
            for i in (1..chs.len()).rev() {
                let in_ch = chs[i];
                let out_ch = chs[i - 1];
                let deconv = conv_transpose2d(
                    in_ch,
                    out_ch,
                    4,
                    ConvTranspose2dConfig {
                        stride: 2,
                        padding: 1,
                        ..Default::default()
                    },
                    vb.pp(format!("decoder.up.{}", chs.len() - 1 - i)),
                )?;
                up_blocks.push(deconv);
                let res = ResBlock2D::new(
                    out_ch,
                    out_ch,
                    vb.pp("decoder"),
                    &format!("res.{}", chs.len() - 1 - i),
                )?;
                up_res.push(res);
            }
        }

        let dec_out = conv2d(
            config.block_out_channels.get(0).copied().unwrap_or(128),
            config.out_channels,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp("decoder.conv_out"),
        )?;
        Ok(Self {
            config: config.clone(),
            enc_in,
            down_blocks,
            down_res,
            to_mu,
            to_logvar,
            from_latent,
            up_blocks,
            up_res,
            dec_out,
            device: dev,
        })
    }

    // Encode image to latents (stub: no KL yet). x: [B,C,H,W] -> [B,latent_c,H/8,W/8] typically.
    pub fn encode(&self, x: &Tensor, sample: bool) -> Result<Tensor> {
        let _bchw = x.dims4()?;
        let mut y = self.enc_in.forward(x)?; // [B, C0, H, W]
        for (i, conv) in self.down_blocks.iter().enumerate() {
            y = conv.forward(&y)?; // downsample
            if let Some(rb) = self.down_res.get(i) {
                y = rb.forward(&y)?;
            }
        }
        // Project to latent mean/logvar (use mean for deterministic encode)
        let mu = self.to_mu.forward(&y)?; // [B, latent_c, H/8, W/8]
        let logvar = self.to_logvar.forward(&y)?;
        let z = if sample {
            // z = mu + exp(0.5*logvar) * eps, eps ~ N(0,1)
            let half = Tensor::new(0.5f32, &self.device)?.to_dtype(logvar.dtype())?;
            let std = logvar.broadcast_mul(&half)?.exp()?;
            let mut eps = Tensor::randn(0f32, 1f32, mu.dims(), &self.device)?;
            if eps.dtype() != mu.dtype() {
                eps = eps.to_dtype(mu.dtype())?;
            }
            (&mu + std.broadcast_mul(&eps)?)?
        } else {
            mu
        };
        // Apply scaling factor to latents as in SD-style VAEs
        let scale = self.config.scaling_factor.unwrap_or(0.18215);
        let mut scale_t = Tensor::new(scale, &self.device)?;
        if scale_t.dtype() != z.dtype() {
            scale_t = scale_t.to_dtype(z.dtype())?;
        }
        z.broadcast_mul(&scale_t)
    }

    // Convenience deterministic encode that uses mean path
    pub fn encode_deterministic(&self, x: &Tensor) -> Result<Tensor> {
        self.encode(x, false)
    }

    // Levels equals number of stride-2 downsamples
    pub fn levels(&self) -> usize {
        self.down_blocks.len().max(1)
    }

    pub fn downsample_factor(&self) -> usize {
        1usize << self.levels()
    }

    // Pad H and W to multiples of factor, return padded tensor plus sizes (orig_h, orig_w, pad_h, pad_w)
    pub fn pad_to_multiple(
        &self,
        x: &Tensor,
        factor: usize,
    ) -> Result<(Tensor, usize, usize, usize, usize)> {
        let (b, c, h, w) = x.dims4()?;
        let h_pad = ((h + factor - 1) / factor) * factor;
        let w_pad = ((w + factor - 1) / factor) * factor;
        if h_pad == h && w_pad == w {
            return Ok((x.clone(), h, w, h_pad, w_pad));
        }
        let mut y = Tensor::zeros((b, c, h_pad, w_pad), x.dtype(), x.device())?;
        y = y.slice_assign(&[0..b, 0..c, 0..h, 0..w], x)?;
        Ok((y, h, w, h_pad, w_pad))
    }

    // Encode with auto-padding to the required multiple; returns (z, orig_h, orig_w, pad_h, pad_w)
    pub fn encode_with_auto_pad(
        &self,
        x: &Tensor,
        sample: bool,
    ) -> Result<(Tensor, usize, usize, usize, usize)> {
        let factor = self.downsample_factor();
        let (x_pad, h, w, hp, wp) = self.pad_to_multiple(x, factor)?;
        let z = self.encode(&x_pad, sample)?;
        Ok((z, h, w, hp, wp))
    }

    // Decode and crop back to the original (h,w) using the latent spatial multiplier
    pub fn decode_to_original(&self, z: &Tensor, orig_h: usize, orig_w: usize) -> Result<Tensor> {
        // Compute padded size achievable by upsampling levels
        let levels = self.levels();
        let zh = z.dim(2)?;
        let zw = z.dim(3)?;
        let target_h = zh * (1usize << levels);
        let target_w = zw * (1usize << levels);
        let x = self.decode(z, target_h, target_w)?;
        // Center/top-left crop to (orig_h, orig_w)
        let th = orig_h.min(target_h);
        let tw = orig_w.min(target_w);
        x.i((.., .., 0..th, 0..tw))
    }

    // Decode latents to image. z: [B,latent_c,H/8,W/8] -> [B,C,H,W]
    pub fn decode(&self, z: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let b = z.dim(0)?;
        // Rescale back from latent space scaling
        let scale = self.config.scaling_factor.unwrap_or(0.18215);
        let mut inv = Tensor::new(1.0f32 / scale, &self.device)?;
        if inv.dtype() != z.dtype() {
            inv = inv.to_dtype(z.dtype())?;
        }
        let mut y = z.broadcast_mul(&inv)?; // [B, lc, H/8, W/8]
        // Map from latent channels to decoder start channels
        y = self.from_latent.forward(&y)?; // [B, C_last, H/8, W/8]
        for (i, deconv) in self.up_blocks.iter().enumerate() {
            y = deconv.forward(&y)?; // upsample by 2
            if let Some(rb) = self.up_res.get(i) {
                y = rb.forward(&y)?;
            }
        }
        // Trim/pad to exact target size via center crop/pad if needed
        let ysz = (y.dim(2)?, y.dim(3)?);
        if ysz.0 != h || ysz.1 != w {
            // Simple resize by center crop or pad zeros
            let dh = h as isize - ysz.0 as isize;
            let dw = w as isize - ysz.1 as isize;
            if dh != 0 || dw != 0 {
                // For simplicity: if larger needed, pad zeros to bottom/right; if smaller, narrow
                let th = h.min(ysz.0);
                let tw = w.min(ysz.1);
                let y_cropped = y.i((.., .., 0..th, 0..tw))?;
                let mut y_new = Tensor::zeros((b, y.dim(1)?, h, w), y.dtype(), &self.device)?;
                y_new = y_new.slice_assign(&[0..b, 0..y.dim(1)?, 0..th, 0..tw], &y_cropped)?;
                y = y_new;
            }
        }
        self.dec_out.forward(&y)
    }
}

// Public VAE abstraction used by the edit pipeline so different implementations can be swapped.
pub trait VaeLike: Send + Sync {
    fn encode_with_auto_pad(
        &self,
        x: &Tensor,
        sample: bool,
    ) -> Result<(Tensor, usize, usize, usize, usize)>;
    fn decode_to_original(&self, z: &Tensor, orig_h: usize, orig_w: usize) -> Result<Tensor>;
    fn downsample_factor(&self) -> usize;
}

impl VaeLike for AutoencoderKL {
    fn encode_with_auto_pad(
        &self,
        x: &Tensor,
        sample: bool,
    ) -> Result<(Tensor, usize, usize, usize, usize)> {
        Self::encode_with_auto_pad(self, x, sample)
    }
    fn decode_to_original(&self, z: &Tensor, orig_h: usize, orig_w: usize) -> Result<Tensor> {
        Self::decode_to_original(self, z, orig_h, orig_w)
    }
    fn downsample_factor(&self) -> usize {
        Self::downsample_factor(self)
    }
}

pub fn build_vae_from_files(
    files: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    cfg: &VaeConfig,
) -> anyhow::Result<AutoencoderKL> {
    // Strict: require safetensors weights to load successfully.
    unsafe {
        crate::ai::hf::with_mmap_varbuilder_multi(files, dtype, device, |vb| {
            AutoencoderKL::new(cfg, vb).map_err(anyhow::Error::from)
        })
    }
}

fn required_vae_keys(cfg: &VaeConfig) -> Vec<String> {
    let mut keys = Vec::new();
    // encoder input conv
    keys.push("encoder.conv_in.weight".into());
    keys.push("encoder.conv_in.bias".into());
    // downs and res blocks
    let m = cfg.block_out_channels.len();
    if m >= 2 {
        for i in 0..(m - 1) {
            keys.push(format!("encoder.down.{}.weight", i));
            keys.push(format!("encoder.down.{}.bias", i));
            keys.push(format!("encoder.res.{}.conv1.weight", i));
            keys.push(format!("encoder.res.{}.conv1.bias", i));
            keys.push(format!("encoder.res.{}.conv2.weight", i));
            keys.push(format!("encoder.res.{}.conv2.bias", i));
        }
    }
    // latent heads
    keys.push("encoder.to_mu.weight".into());
    keys.push("encoder.to_mu.bias".into());
    keys.push("encoder.to_logvar.weight".into());
    keys.push("encoder.to_logvar.bias".into());
    // decoder from_latent
    keys.push("decoder.from_latent.weight".into());
    keys.push("decoder.from_latent.bias".into());
    // decoder ups and res blocks
    if m >= 2 {
        for i in 0..(m - 1) {
            keys.push(format!("decoder.up.{}.weight", i));
            keys.push(format!("decoder.up.{}.bias", i));
            keys.push(format!("decoder.res.{}.conv1.weight", i));
            keys.push(format!("decoder.res.{}.conv1.bias", i));
            keys.push(format!("decoder.res.{}.conv2.weight", i));
            keys.push(format!("decoder.res.{}.conv2.bias", i));
        }
    }
    keys.push("decoder.conv_out.weight".into());
    keys.push("decoder.conv_out.bias".into());
    keys
}

pub fn build_vae_from_files_with_5d_squeeze(
    files: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    cfg: &VaeConfig,
) -> anyhow::Result<AutoencoderKL> {
    // Gather required weights from shards; if a conv weight is 5D [O,I,T,KH,KW], squeeze T by center slice.
    let needed = required_vae_keys(cfg);
    // Collect fully-owned buffers for each required key
    let mut collected: Vec<(String, safetensors::tensor::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut have: HashSet<String> = HashSet::new();

    for f in files.iter() {
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let st = match SafeTensors::deserialize(&bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for k in needed.iter() {
            if have.contains(k) {
                continue;
            }
            if let Ok(t) = st.tensor(k) {
                let dt = t.dtype();
                let shape: Vec<usize> = t.shape().into();
                let data = t.data();
                let total_elems: usize = shape.iter().product();
                let el = if total_elems == 0 {
                    1
                } else {
                    data.len().saturating_div(total_elems)
                };
                if shape.len() == 5 {
                    // [O,I,T,KH,KW] -> center T slice -> [O,I,KH,KW]
                    let (o, i, tdim, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
                    let tm = tdim / 2;
                    let mut out = vec![0u8; o * i * kh * kw * el];
                    let mut dst_idx = 0usize;
                    for oo in 0..o {
                        for ii in 0..i {
                            for hh in 0..kh {
                                for ww in 0..kw {
                                    let src_index =
                                        ((((oo * i + ii) * tdim + tm) * kh + hh) * kw + ww) * el;
                                    let src_byte_off = src_index;
                                    let dst_byte_off = dst_idx * el;
                                    out[dst_byte_off..dst_byte_off + el]
                                        .copy_from_slice(&data[src_byte_off..src_byte_off + el]);
                                    dst_idx += 1;
                                }
                            }
                        }
                    }
                    collected.push((k.clone(), dt, vec![o, i, kh, kw], out));
                    have.insert(k.clone());
                } else if shape.len() == 4 || shape.len() == 1 {
                    // conv weight or bias
                    let out = data.to_vec();
                    collected.push((k.clone(), dt, shape.clone(), out));
                    have.insert(k.clone());
                } else {
                    // Unexpected rank: copy as-is
                    let out = data.to_vec();
                    collected.push((k.clone(), dt, shape.clone(), out));
                    have.insert(k.clone());
                }
                if have.len() == needed.len() {
                    break;
                }
            }
        }
        if have.len() == needed.len() {
            break;
        }
    }

    // Verify we found all keys; if not, report missing
    if have.len() != needed.len() {
        let missing: Vec<_> = needed
            .iter()
            .filter(|k| !have.contains(*k))
            .cloned()
            .collect();
        anyhow::bail!("VAE adaptation: missing keys in safetensors: {:?}", missing);
    }

    // Serialize adapted minimal weights to a temp file
    // Build lookup map from key to index in collected
    let mut idx: HashMap<&str, usize> = HashMap::new();
    for (i, (k, _, _, _)) in collected.iter().enumerate() {
        idx.insert(k.as_str(), i);
    }
    let ordered_keys: Vec<String> = needed.clone();
    let mut ordered: Vec<(&str, TensorView)> = Vec::with_capacity(ordered_keys.len());
    for k in ordered_keys.iter() {
        if let Some(&i) = idx.get(k.as_str()) {
            let (kstr, dt, shp, buf) = &collected[i];
            let tv =
                TensorView::new(*dt, shp.clone(), buf.as_slice()).map_err(anyhow::Error::msg)?;
            ordered.push((kstr.as_str(), tv));
        }
    }
    let bin = serialize(ordered.into_iter(), None).map_err(anyhow::Error::msg)?;
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), &bin)?;
    let tmp_path = tmp.into_temp_path();
    let tmp_path_buf = std::path::PathBuf::from(tmp_path.as_ref() as &std::path::Path);

    // Strictly build from the adapted file
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&tmp_path_buf), dtype, device)?
    };
    AutoencoderKL::new(cfg, vb).map_err(anyhow::Error::from)
}

// Compatibility: parse Qwen's AutoencoderKLQwenImage config layout and map to our VaeConfig.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfigCompatQwen {
    #[serde(default)]
    _class_name: Option<String>,
    #[serde(default)]
    base_dim: Option<usize>,
    #[serde(default)]
    dim_mult: Option<Vec<usize>>, // e.g., [1,2,4,4]
    #[serde(default)]
    z_dim: Option<usize>,
}

// pub mod qwen_image; // TODO: enable after full mapping is implemented
pub mod qwen_image;
pub use qwen_image::QwenImageVaeSimplified;

pub fn build_qwen_vae_simplified_from_files(
    files: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    scaling: f32,
) -> anyhow::Result<QwenImageVaeSimplified> {
    // Some weights are 5D [O,I,T,KH,KW]. We'll adapt a minimal subset used by the simplified loader.
    use safetensors::SafeTensors;
    use safetensors::tensor::{TensorView, serialize};
    let mut collected: Vec<(String, safetensors::tensor::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Keys we care about for simplified loader
    let wanted_prefixes = [
        "encoder.conv_in",
        "encoder.down_blocks.", // resample.1 and convs
        "encoder.conv_out",
        "decoder.conv_in",
        "decoder.up_blocks.", // upsamplers.0.resample.1 and convs
        "decoder.conv_out",
    ];
    for f in files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            continue;
        };
        for name in st.names() {
            if seen.contains(name) {
                continue;
            }
            if !wanted_prefixes.iter().any(|p| name.starts_with(p)) {
                continue;
            }
            if let Ok(t) = st.tensor(name) {
                let dt = t.dtype();
                let shape: Vec<usize> = t.shape().into();
                let data = t.data();
                // Compute element byte size heuristically
                let elem_size = match dt {
                    safetensors::tensor::Dtype::F16 => 2,
                    safetensors::tensor::Dtype::BF16 => 2,
                    safetensors::tensor::Dtype::F32 => 4,
                    safetensors::tensor::Dtype::F64 => 8,
                    safetensors::tensor::Dtype::I64 => 8,
                    safetensors::tensor::Dtype::I32 => 4,
                    safetensors::tensor::Dtype::I16 => 2,
                    safetensors::tensor::Dtype::I8 => 1,
                    safetensors::tensor::Dtype::U8 => 1,
                    _ => 4,
                };
                if shape.len() == 5 {
                    // [O,I,T,KH,KW] -> take center T slice
                    let (o, i, t, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
                    let tm = t / 2;
                    let mut out = vec![0u8; o * i * kh * kw * elem_size];
                    let mut dst = 0usize;
                    for oo in 0..o {
                        for ii in 0..i {
                            for hh in 0..kh {
                                for ww in 0..kw {
                                    let src_index = ((((oo * i + ii) * t + tm) * kh + hh) * kw
                                        + ww)
                                        * elem_size;
                                    out[dst..dst + elem_size]
                                        .copy_from_slice(&data[src_index..src_index + elem_size]);
                                    dst += elem_size;
                                }
                            }
                        }
                    }
                    collected.push((name.to_string(), dt, vec![o, i, kh, kw], out));
                    seen.insert(name.to_string());
                } else {
                    collected.push((name.to_string(), dt, shape.clone(), data.to_vec()));
                    seen.insert(name.to_string());
                }
            }
        }
    }
    if collected.is_empty() {
        anyhow::bail!("No matching Qwen VAE weights found to adapt.");
    }
    // Build ordered views, and also parse channel hints for specific layers
    let mut enc_in_ch: Option<(usize, usize)> = None;
    let mut enc_out_ch: Option<(usize, usize)> = None;
    let mut dec_in_ch: Option<(usize, usize)> = None;
    let mut dec_out_ch: Option<(usize, usize)> = None;
    let mut enc_down_shapes: Vec<(usize, usize, usize)> = Vec::new();
    let mut dec_up_shapes: Vec<(usize, usize, usize)> = Vec::new();
    let mut ordered: Vec<(&str, TensorView)> = Vec::with_capacity(collected.len());
    // Potential extra conv adapters to bridge channels
    let mut enc_extra: Vec<(String, usize, usize, usize)> = Vec::new();
    let mut dec_extra: Vec<(String, usize, usize, usize)> = Vec::new();
    for (name, dt, shape, buf) in collected.iter() {
        let tv = TensorView::new(*dt, shape.clone(), buf.as_slice()).map_err(anyhow::Error::msg)?;
        ordered.push((name.as_str(), tv));
        // Parse channel info
        if name == "encoder.conv_in.weight" && (shape.len() == 4) {
            enc_in_ch = Some((shape[1], shape[0]));
        } else if name == "encoder.conv_out.weight" && (shape.len() == 4) {
            enc_out_ch = Some((shape[1], shape[0]));
        } else if name == "decoder.conv_in.weight" && (shape.len() == 4) {
            dec_in_ch = Some((shape[1], shape[0]));
        } else if name == "decoder.conv_out.weight" && (shape.len() == 4) {
            dec_out_ch = Some((shape[1], shape[0]));
        } else if name.contains("encoder.down_blocks.")
            && name.ends_with(".resample.1.weight")
            && shape.len() == 4
        {
            // encoder.down_blocks.{i}.resample.1.weight -> [out,in,k,k]
            if let Some(idx_str) = name.split('.').nth(2) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    enc_down_shapes.push((idx, shape[1], shape[0]));
                }
            }
        } else if name.contains("encoder.down_blocks.")
            && (name.contains("conv1.weight")
                || name.contains("conv2.weight")
                || name.contains("conv_shortcut.weight"))
            && shape.len() == 4
        {
            // Adapter convs inside resnets/shortcuts
            let k = shape.get(2).copied().unwrap_or(3);
            let base = if name.ends_with(".weight") {
                name.trim_end_matches(".weight").to_string()
            } else {
                name.clone()
            };
            enc_extra.push((base, shape[1], shape[0], k));
        } else if name.contains("decoder.up_blocks.")
            && name.ends_with(".upsamplers.0.resample.1.weight")
            && shape.len() == 4
        {
            if let Some(idx_str) = name.split('.').nth(2) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    dec_up_shapes.push((idx, shape[1], shape[0]));
                }
            }
        } else if name.contains("decoder.up_blocks.")
            && (name.contains("conv1.weight")
                || name.contains("conv2.weight")
                || name.contains("conv_shortcut.weight")
                || name.contains("time_conv.weight"))
            && (shape.len() == 4 || shape.len() == 5)
        {
            // Decoder adapters; if 5D already squeezed earlier so expect 4D here
            let k = shape
                .get(shape.len().saturating_sub(1))
                .copied()
                .unwrap_or(3);
            // Heuristic: treat 1x1 separately for padding
            let base = if name.ends_with(".weight") {
                name.trim_end_matches(".weight").to_string()
            } else {
                name.clone()
            };
            dec_extra.push((base, shape[1], shape[0], k));
        }
    }
    let bin = serialize(ordered.into_iter(), None).map_err(anyhow::Error::msg)?;
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), &bin)?;
    let tmp_path = tmp.into_temp_path();
    let tmp_path_buf = std::path::PathBuf::from(tmp_path.as_ref() as &std::path::Path);
    // Move extra lists into the closure to satisfy potential 'static bounds
    let enc_extra_owned = enc_extra.clone();
    unsafe {
        crate::ai::hf::with_mmap_varbuilder_multi(
            std::slice::from_ref(&tmp_path_buf),
            dtype,
            device,
            move |vb| {
                // Use hints if available, else fallback to heuristic defaults
                if let (Some(ein), Some(eout), Some(din), Some(dout)) =
                    (enc_in_ch, enc_out_ch, dec_in_ch, dec_out_ch)
                {
                    // Ensure stable ordering by block index then greedily pick a channel-consistent chain.
                    enc_down_shapes.sort_by_key(|x| x.0);
                    dec_up_shapes.sort_by_key(|x| x.0);

                    // Build encoder chain: start from enc_in out-channels and follow shapes with matching in_c.
                    let mut enc_chain: Vec<(usize, usize, usize)> = Vec::new();
                    let mut cur_enc_c = ein.1; // output channels of encoder.conv_in
                    for &triple in enc_down_shapes.iter() {
                        let (_idx, in_c, out_c) = triple;
                        if in_c == cur_enc_c {
                            enc_chain.push(triple);
                            cur_enc_c = out_c;
                        }
                    }
                    // Build decoder chain: start from decoder.conv_in out-channels (after latent lift)
                    let mut dec_chain: Vec<(usize, usize, usize)> = Vec::new();
                    let mut cur_dec_c = din.1; // output channels of decoder.conv_in
                    for &triple in dec_up_shapes.iter() {
                        let (_idx, in_c, out_c) = triple;
                        if in_c == cur_dec_c {
                            dec_chain.push(triple);
                            cur_dec_c = out_c;
                        }
                    }
                    // If chains ended up empty (unexpected), fall back to original lists
                    let enc_use = if enc_chain.is_empty() {
                        &enc_down_shapes[..]
                    } else {
                        &enc_chain[..]
                    };
                    let dec_use = if dec_chain.is_empty() {
                        &dec_up_shapes[..]
                    } else {
                        &dec_chain[..]
                    };

                    QwenImageVaeSimplified::new_with_hints(
                        vb,
                        scaling,
                        ein,
                        enc_use,
                        eout,
                        din,
                        dec_use,
                        dout,
                        &enc_extra_owned,
                    )
                    .map_err(anyhow::Error::from)
                } else {
                    QwenImageVaeSimplified::new(vb, scaling).map_err(anyhow::Error::from)
                }
            },
        )
    }
}

impl VaeConfigCompatQwen {
    pub fn to_internal(&self) -> VaeConfig {
        // Map base_dim and dim_mult to block_out_channels
        let base = self.base_dim.unwrap_or(96);
        let mults = self.dim_mult.clone().unwrap_or_else(|| vec![1, 2, 4, 4]);
        let block_out_channels: Vec<usize> = mults.iter().map(|m| base * *m).collect();
        let latent = self.z_dim.unwrap_or(16);
        VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: latent,
            block_out_channels,
            down_block_types: default_down_block_types(),
            up_block_types: default_up_block_types(),
            sample_size: None,
            scaling_factor: Some(0.18215),
        }
    }
}
