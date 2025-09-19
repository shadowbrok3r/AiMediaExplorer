use candle_core::{Tensor, Result, Device, Module, DType, IndexOp};
use candle_nn::{VarBuilder, conv2d, Conv2d, Conv2dConfig};

// Simplified loader for Qwen AutoencoderKLQwenImage using native key names.
// It uses conv_in, down_blocks.*.resample.1 (3 downs), conv_out (split mu/logvar),
// and decoder conv_in, up_blocks.*.upsamplers.0.resample.1 (3 ups), conv_out.
// 5D conv weights should be converted to 4D beforehand.

#[derive(Debug, Clone)]
pub struct QwenImageVaeSimplified {
    pub device: Device,
    pub dtype: DType,
    pub scaling: f32,
    // encoder
    enc_in: Conv2d,
    enc_downs: Vec<Conv2d>,
    enc_extra: Vec<Conv2d>,
    enc_out: Conv2d, // produces 2*z_dim channels (e.g., 32)
    // decoder
    dec_in: Conv2d,
    dec_ups: Vec<Conv2d>,
    dec_extra: Vec<Conv2d>,
    dec_out: Conv2d,
}

impl QwenImageVaeSimplified {
    pub fn new(vb: VarBuilder, scaling: f32) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let enc_in = conv2d(3, 96, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("encoder.conv_in"))?;
        // Collect encoder downsample convs: encoder.down_blocks.{i}.resample.1
        let mut enc_downs = Vec::new();
        for i in 0..12 { // probe indices
            let key = format!("encoder.down_blocks.{i}.resample.1");
            if vb.pp(&key).contains_tensor("weight") {
                let conv = conv2d(
                    96, 96, 3,
                    Conv2dConfig { stride: 2, padding: 1, ..Default::default() },
                    vb.pp(&key),
                )?;
                enc_downs.push(conv);
            }
        }
        // Fallback to typical three downs if none found by contains
        if enc_downs.is_empty() {
            for i in [2, 5, 8] {
                let key = format!("encoder.down_blocks.{i}.resample.1");
                if vb.pp(&key).contains_tensor("weight") {
                    let conv = conv2d(96, 96, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vb.pp(&key))?;
                    enc_downs.push(conv);
                }
            }
        }
        let enc_out = conv2d(384, 32, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("encoder.conv_out"))?;

        let dec_in = conv2d(16, 384, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("decoder.conv_in"))?;
        let mut dec_ups = Vec::new();
        for i in 0..4 {
            let key = format!("decoder.up_blocks.{i}.upsamplers.0.resample.1");
            if vb.pp(&key).contains_tensor("weight") {
                let conv = conv2d(96, 96, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp(&key))?;
                dec_ups.push(conv);
            }
        }
        if dec_ups.is_empty() {
            for i in [0, 1, 2] {
                let key = format!("decoder.up_blocks.{i}.upsamplers.0.resample.1");
                if vb.pp(&key).contains_tensor("weight") {
                    let conv = conv2d(96, 96, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp(&key))?;
                    dec_ups.push(conv);
                }
            }
        }
        let dec_out = conv2d(96, 3, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("decoder.conv_out"))?;
        Ok(Self { device, dtype, scaling, enc_in, enc_downs, enc_extra: Vec::new(), enc_out, dec_in, dec_ups, dec_extra: Vec::new(), dec_out })
    }

    // Construct with explicit channel hints parsed from safetensors shapes.
    pub fn new_with_hints(
        vb: VarBuilder,
        scaling: f32,
        enc_in_ch: (usize, usize),
        enc_down_shapes: &[(usize, usize, usize)], // (block_index, in_c, out_c)
        enc_out_ch: (usize, usize),
        dec_in_ch: (usize, usize),
        dec_up_shapes: &[(usize, usize, usize)],   // (block_index, in_c, out_c)
        dec_out_ch: (usize, usize),
        enc_extra: &[(String, usize, usize, usize)], // (key, in_c, out_c, k)
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let enc_in = conv2d(enc_in_ch.0, enc_in_ch.1, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("encoder.conv_in"))?;
        let mut enc_downs = Vec::new();
        let mut cur_enc_c = enc_in_ch.1;
        for (i, in_c, out_c) in enc_down_shapes.iter() {
            let key = format!("encoder.down_blocks.{i}.resample.1");
            // Only include if it matches the current channel count
            if *in_c == cur_enc_c {
                let conv = conv2d(*in_c, *out_c, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vb.pp(&key))?;
                enc_downs.push(conv);
                cur_enc_c = *out_c;
            }
        }
        // Extra encoder adapters (e.g., conv1/conv2/conv_shortcut) to reach enc_out input channels
        let mut enc_extra_layers = Vec::new();
        // Greedily add adapters that match current channel count until we reach expected enc_out input
        let target_enc_in = enc_out_ch.0;
        let mut enc_extra_sorted = enc_extra.to_vec();
        enc_extra_sorted.sort_by_key(|(k, _, _, _)| k.clone());
        let mut guard = 0usize;
        while cur_enc_c != target_enc_in && guard < enc_extra_sorted.len() {
            let mut progressed = false;
            for (key, in_c, out_c, k) in enc_extra_sorted.iter() {
                if *in_c == cur_enc_c {
                    let pad = if *k == 1 { 0 } else { 1 };
                    let conv = conv2d(*in_c, *out_c, *k, Conv2dConfig { stride: 1, padding: pad, ..Default::default() }, vb.pp(key))?;
                    enc_extra_layers.push(conv);
                    cur_enc_c = *out_c;
                    progressed = true;
                    break;
                }
            }
            if !progressed { break; }
            guard += 1;
        }
        let enc_out = conv2d(enc_out_ch.0, enc_out_ch.1, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("encoder.conv_out"))?;
        let dec_in = conv2d(dec_in_ch.0, dec_in_ch.1, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("decoder.conv_in"))?;
        let mut dec_ups = Vec::new();
        let mut cur_dec_c = dec_in_ch.1;
        for (i, in_c, out_c) in dec_up_shapes.iter() {
            let key = format!("decoder.up_blocks.{i}.upsamplers.0.resample.1");
            if *in_c == cur_dec_c {
                let conv = conv2d(*in_c, *out_c, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp(&key))?;
                dec_ups.push(conv);
                cur_dec_c = *out_c;
            }
        }
        // Avoid using decoder extra adapters for now to keep channel expectations aligned with upsamplers
        let dec_extra_layers = Vec::new();
        let dec_out = conv2d(dec_out_ch.0, dec_out_ch.1, 3, Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("decoder.conv_out"))?;
        Ok(Self { device, dtype, scaling, enc_in, enc_downs, enc_extra: enc_extra_layers, enc_out, dec_in, dec_ups, dec_extra: dec_extra_layers, dec_out })
    }

    pub fn encode(&self, x: &Tensor, deterministic: bool) -> Result<Tensor> {
        let mut y = self.enc_in.forward(x)?;
        for d in &self.enc_downs { y = d.forward(&y)?; }
        for d in &self.enc_extra { y = d.forward(&y)?; }
        let z_all = self.enc_out.forward(&y)?; // [B,32,H',W']
        let mu = z_all.i((.., 0..16, .., ..))?;
        let logvar = z_all.i((.., 16..32, .., ..))?;
        let z = if deterministic {
            mu
        } else {
            // z = mu + exp(0.5*logvar) * eps
            let std = (&logvar * 0.5)?.exp()?;
            let eps = Tensor::randn(0f32, 1f32, mu.dims(), x.device())?;
            (&mu + std.broadcast_mul(&eps)?)?
        };
        let scale = Tensor::new(self.scaling, x.device())?;
        z.broadcast_mul(&scale)
    }

    fn upsample_nearest(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let mut up = Tensor::zeros((b, c, h * 2, w * 2), x.dtype(), x.device())?;
        for yy in 0..h { for xx in 0..w {
            let patch = x.i((.., .., yy, xx))?; // (b,c)
            up = up.slice_assign(&[0..b, 0..c, (yy*2)..(yy*2+2), (xx*2)..(xx*2+2)], &patch.unsqueeze(2)?.unsqueeze(3)?)?;
        }}
        Ok(up)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let inv = Tensor::new(1.0f32 / self.scaling, z.device())?;
        let mut y = z.broadcast_mul(&inv)?; // [B,16,H',W']
        y = self.dec_in.forward(&y)?; // [B,384,H',W']
        for d in &self.dec_extra { y = d.forward(&y)?; }
        for u in &self.dec_ups {
            y = self.upsample_nearest(&y)?; // x2 spatial
            y = u.forward(&y)?;
        }
        self.dec_out.forward(&y)
    }
}

impl QwenImageVaeSimplified {
    pub fn downsample_factor(&self) -> usize { 1usize << self.enc_downs.len().max(1) }
    pub fn pad_to_multiple(&self, x: &Tensor, factor: usize) -> Result<(Tensor, usize, usize, usize, usize)> {
        let (b, c, h, w) = x.dims4()?;
        let h_pad = ((h + factor - 1) / factor) * factor;
        let w_pad = ((w + factor - 1) / factor) * factor;
        if h_pad == h && w_pad == w { return Ok((x.clone(), h, w, h_pad, w_pad)); }
        let mut y = Tensor::zeros((b, c, h_pad, w_pad), x.dtype(), x.device())?;
        y = y.slice_assign(&[0..b, 0..c, 0..h, 0..w], x)?;
        Ok((y, h, w, h_pad, w_pad))
    }
    pub fn encode_with_auto_pad(&self, x: &Tensor, sample: bool) -> Result<(Tensor, usize, usize, usize, usize)> {
        let factor = self.downsample_factor();
        let (x_pad, h, w, hp, wp) = self.pad_to_multiple(x, factor)?;
        let z = self.encode(&x_pad, !sample)?; // deterministic toggle aligns with caller semantics
        Ok((z, h, w, hp, wp))
    }
    pub fn decode_to_original(&self, z: &Tensor, orig_h: usize, orig_w: usize) -> Result<Tensor> {
        let levels = self.enc_downs.len().max(1);
        let zh = z.dim(2)?;
        let zw = z.dim(3)?;
        let target_h = zh * (1usize << levels);
        let target_w = zw * (1usize << levels);
        let x = self.decode(z)?;
        let th = orig_h.min(target_h);
        let tw = orig_w.min(target_w);
        x.i((.., .., 0..th, 0..tw))
    }
}

impl super::VaeLike for QwenImageVaeSimplified {
    fn encode_with_auto_pad(&self, x: &Tensor, sample: bool) -> Result<(Tensor, usize, usize, usize, usize)> { self.encode_with_auto_pad(x, sample) }
    fn decode_to_original(&self, z: &Tensor, orig_h: usize, orig_w: usize) -> Result<Tensor> { self.decode_to_original(z, orig_h, orig_w) }
    fn downsample_factor(&self) -> usize { self.downsample_factor() }
}
