use super::*;

impl QwenImageEditPipeline {
    // Denoiser scaffold: encode -> noise -> steps -> decode. Returns PNG bytes.
    pub fn denoise_loop(
        &self,
        image_path: &std::path::Path,
        opts: &crate::ai::qwen_image_edit::EditOptions,
    ) -> Result<Vec<u8>> {
        // Validate components
        // Prefer the preloaded VAE; if missing, attempt a last-mile build from repo files so we always use the provided VAE.
        let vae_local;
        let vae: &dyn VaeLike = match self.vae.as_ref() {
            Some(v) => v.as_ref(),
            None => {
                // Try to parse VAE config from captured JSON; if that fails, read from file path.
                let cfg_opt: Option<AEKLConfig> = if let Some(vj) = &self.vae_config_json {
                    serde_json::from_value::<AEKLConfig>(vj.clone())
                        .ok()
                        .or_else(|| {
                            serde_json::from_value::<crate::ai::vae::VaeConfigCompatQwen>(
                                vj.clone(),
                            )
                            .ok()
                            .map(|c| c.to_internal())
                        })
                } else if let Some(p) = &self.paths.vae_config {
                    std::fs::read_to_string(p)
                        .ok()
                        .and_then(|s| serde_json::from_str::<AEKLConfig>(&s).ok())
                        .or_else(|| {
                            std::fs::read_to_string(p)
                                .ok()
                                .and_then(|s| {
                                    serde_json::from_str::<crate::ai::vae::VaeConfigCompatQwen>(&s)
                                        .ok()
                                })
                                .map(|c| c.to_internal())
                        })
                } else {
                    None
                };
                if let Some(mut cfg) = cfg_opt {
                    // Ensure required fields are present even if config is minimal
                    if cfg.in_channels == 0 {
                        cfg.in_channels = 3;
                    }
                    if cfg.out_channels == 0 {
                        cfg.out_channels = 3;
                    }
                    if cfg.latent_channels == 0 {
                        cfg.latent_channels = 16;
                    }
                    // Prefer the correct strict path based on the weight set before attempting a known-failing path.
                    let prefer_qwen_path = crate::ai::vae::vae_has_5d_conv(&self.paths.vae_files)
                        || crate::ai::vae::vae_looks_like_qwen(&self.paths.vae_files);
                    let first_attempt: anyhow::Result<Box<dyn VaeLike>> = if prefer_qwen_path {
                        match crate::ai::vae::build_qwen_vae_from_files_strict(
                            &self.paths.vae_files,
                            self.dtype,
                            &self.device,
                            cfg.scaling_factor.unwrap_or(0.18215),
                        ) {
                            Ok(vq) => Ok(Box::new(vq) as Box<dyn VaeLike>),
                            Err(e) => Err(e),
                        }
                    } else {
                        match build_vae_from_files(
                            &self.paths.vae_files,
                            self.dtype,
                            &self.device,
                            &cfg,
                        ) {
                            Ok(v) => Ok(Box::new(v) as Box<dyn VaeLike>),
                            Err(e) => Err(e),
                        }
                    };
                    match first_attempt {
                        Ok(v) => {
                            vae_local = v;
                            &*vae_local
                        }
                        Err(e) => {
                            log::error!("[qwen-image-edit] VAE strict load failed: {e}");
                            Self::diag_scan_weights(
                                &self.paths.vae_files,
                                "vae",
                                &[
                                    "encoder.conv_in.weight",
                                    "encoder.to_mu.weight",
                                    "encoder.to_logvar.weight",
                                    "decoder.from_latent.weight",
                                    "decoder.conv_out.weight",
                                ],
                            );
                            // Second attempt sequence: if we didn't try qwen first, try it now; otherwise try 5D squeeze 2D VAE
                            if !prefer_qwen_path {
                                match crate::ai::vae::build_qwen_vae_from_files_strict(
                                    &self.paths.vae_files,
                                    self.dtype,
                                    &self.device,
                                    cfg.scaling_factor.unwrap_or(0.18215),
                                ) {
                                    Ok(vq) => {
                                        log::warn!("[qwen-image-edit] VAE strict Qwen adapter used.");
                                        vae_local = Box::new(vq) as Box<dyn VaeLike>;
                                        &*vae_local
                                    }
                                    Err(e2) => {
                                        log::warn!("[qwen-image-edit] VAE strict Qwen adapter failed: {e2}; trying 5D->4D squeeze for internal VAE.");
                                        match build_vae_from_files_with_5d_squeeze(
                                            &self.paths.vae_files,
                                            self.dtype,
                                            &self.device,
                                            &cfg,
                                        ) {
                                            Ok(vs) => {
                                                log::warn!("[qwen-image-edit] VAE 5D->4D squeeze applied.");
                                                vae_local = Box::new(vs) as Box<dyn VaeLike>;
                                                &*vae_local
                                            }
                                            Err(e3) => {
                                                bail!("VAE strict load failed (qwen, squeeze): {}", e3);
                                            }
                                        }
                                    }
                                }
                            } else {
                                // prefer_qwen_path true: try 5D->4D internal VAE as secondary
                                match build_vae_from_files_with_5d_squeeze(
                                    &self.paths.vae_files,
                                    self.dtype,
                                    &self.device,
                                    &cfg,
                                ) {
                                    Ok(v) => {
                                        log::warn!("[qwen-image-edit] VAE 5D->4D squeeze applied (secondary).");
                                        vae_local = Box::new(v) as Box<dyn VaeLike>;
                                        &*vae_local
                                    }
                                    Err(e3) => {
                                        bail!("VAE strict load failed (qwen primary, squeeze secondary): {}", e3);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    bail!("VAE not loaded and failed to parse vae/config.json from repo (strict)");
                }
            }
        };
        // Text encoder is external; we'll build minimal embeddings from weights if available.
        // Selection: if a GGUF transformer was chosen (paths.transformer_gguf is Some), prefer the Qwen transformer
        // over Flux even if a Flux-style mapping is possible from safetensors. Otherwise prefer Flux when available.
        let tr_flux_opt = self.transformer_flux.as_ref();
        let tr_qwen_opt = self.transformer_model.as_ref();
        let prefer_qwen = self.paths.transformer_gguf.is_some();
        if tr_flux_opt.is_none() && tr_qwen_opt.is_none() {
            bail!("Transformer not loaded");
        }
        let sched = self
            .scheduler
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Scheduler missing"))?;

        // Load and to tensor
        let img = image::ImageReader::open(image_path)?.decode()?.to_rgb8();
        let (mut w, mut h) = img.dimensions();
        // Optional: clamp max side to reduce memory (keeps aspect). Fixed conservative cap.
    let max_side: u32 = 768;
        if w.max(h) > max_side {
            let scale = max_side as f32 / w.max(h) as f32;
            w = (w as f32 * scale).round() as u32;
            h = (h as f32 * scale).round() as u32;
        }
        let data: Vec<f32> =
            image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle)
                .pixels()
                .flat_map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ]
                })
                .collect();
        let mut x = candle_core::Tensor::from_vec(
            data,
            (1usize, 3usize, h as usize, w as usize),
            &self.device,
        )?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }

        // Overall progress covers: encode (1), denoise (N steps), decode (1), save (1)
        let total_overall: u64 = (opts.num_inference_steps.max(1) as u64) + 3;
        QWEN_EDIT_STATUS.set_state(StatusState::Running, "Encoding");
        QWEN_EDIT_STATUS.set_progress(0, total_overall);

        // Encode with auto pad (strict)
        let (mut z, oh, ow, _ph, _pw) = vae
            .encode_with_auto_pad(&x, !opts.deterministic_vae)
            .map_err(|e| {
                let xd = x.dims().to_vec();
                anyhow::anyhow!(
                    "VAE encode failed (strict): {} | x: shape={:?} dtype={:?} device={:?}",
                    e,
                    xd,
                    x.dtype(),
                    x.device()
                )
            })?;
        if z.dtype() != self.dtype {
            z = z.to_dtype(self.dtype)?;
        }
        if z.device().is_cuda() != self.device.is_cuda() {
            z = z.to_device(&self.device)?;
        }

        // Mark encode done in overall progress
        QWEN_EDIT_STATUS.set_progress(1, total_overall);

        // Build embeddings for cond and uncond
    let (mut text_cond, mut text_uncond) = if self.tokenizer.is_some() && self.has_text_encoder {
            let tok = self.tokenizer.as_ref().unwrap();
            // Prepare config values
            let (vocab_size, hidden_size, eps) = if let Some(cfg) = &self.text_encoder_config_json {
                let tc = cfg.get("text_config");
                let vs = tc
                    .and_then(|t| t.get("vocab_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(151936) as usize;
                // Determine joint text dim from whichever transformer is present
                let joint_dim = self
                    .transformer_flux
                    .as_ref()
                    .map(|m| m.config.joint_attention_dim)
                    .or_else(|| {
                        self.transformer_model
                            .as_ref()
                            .map(|m| m.config.joint_attention_dim)
                    })
                    .unwrap_or(3584);
                let hs = tc
                    .and_then(|t| t.get("hidden_size").or_else(|| t.get("d_model")))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(joint_dim as u64) as usize;
                // Epsilon: prefer class-specific keys
                let e = if let Some(class) = &self.text_encoder_class {
                    match class {
                        TextEncoderClass::CLIPTextModel => tc
                            .and_then(|t| t.get("layer_norm_eps"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1e-5) as f32,
                        TextEncoderClass::T5EncoderModel => tc
                            .and_then(|t| t.get("rms_norm_eps").or_else(|| t.get("layer_norm_epsilon")))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1e-6) as f32,
                        _ => tc
                            .and_then(|t| t.get("rms_norm_eps"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1e-6) as f32,
                    }
                } else {
                    tc.and_then(|t| t.get("rms_norm_eps")).and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32
                };
                (vs, hs, e)
            } else {
                let joint_dim = self
                    .transformer_flux
                    .as_ref()
                    .map(|m| m.config.joint_attention_dim)
                    .or_else(|| {
                        self.transformer_model
                            .as_ref()
                            .map(|m| m.config.joint_attention_dim)
                    })
                    .unwrap_or(3584);
                (151936usize, joint_dim, 1e-6f32)
            };
            // Try to load embedding + norm for known architectures. Strict: must find weights.
            enum NormKind { RmsNoBias, LayerNormWithBias }
            struct TextEmbPack { emb: Embedding, norm_w: candle_core::Tensor, norm_b: Option<candle_core::Tensor>, kind: NormKind }
            // Force CPU for text embed/norm to avoid GPU kernel use in this stage
            let loaded: anyhow::Result<TextEmbPack> = unsafe {
                with_mmap_varbuilder_multi(
                    &self.paths.text_encoder_files,
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                    |vb: VarBuilder| {
                        // Generic attempt: embed_tokens + norm.weight (RMS)
                        let try_generic = |vb: VarBuilder, pfx: &str| -> anyhow::Result<TextEmbPack> {
                            let vb = if pfx.is_empty() { vb } else { vb.pp(pfx) };
                            let emb = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
                            let norm_w = vb.pp("norm").get(hidden_size, "weight")?;
                            Ok(TextEmbPack { emb, norm_w, norm_b: None, kind: NormKind::RmsNoBias })
                        };
                        // CLIPTextModel: text_model.embeddings.token_embedding, text_model.final_layer_norm.{weight,bias} (LayerNorm)
                        let try_clip = |vb: VarBuilder| -> anyhow::Result<TextEmbPack> {
                            let emb = candle_nn::embedding(vocab_size, hidden_size, vb.pp("text_model").pp("embeddings").pp("token_embedding"))?;
                            let norm_root = vb.pp("text_model").pp("final_layer_norm");
                            let norm_w = norm_root.get(hidden_size, "weight")?;
                            let norm_b = norm_root.get(hidden_size, "bias").ok();
                            Ok(TextEmbPack { emb, norm_w, norm_b, kind: NormKind::LayerNormWithBias })
                        };
                        // T5EncoderModel: shared (weight), encoder.final_layer_norm.weight (RMS)
                        let try_t5 = |vb: VarBuilder| -> anyhow::Result<TextEmbPack> {
                            let emb = candle_nn::embedding(vocab_size, hidden_size, vb.pp("shared"))?;
                            let norm_w = vb.pp("encoder").pp("final_layer_norm").get(hidden_size, "weight")?;
                            Ok(TextEmbPack { emb, norm_w, norm_b: None, kind: NormKind::RmsNoBias })
                        };
                        // Select attempts based on declared class if available, otherwise try all in a reasonable order
                        if let Some(class) = &self.text_encoder_class {
                            match class {
                                TextEncoderClass::CLIPTextModel => {
                                    if let Ok(v) = try_clip(vb.clone()) { return Ok(v); }
                                    // Fallback to generic paths rooted under text_model
                                    for p in ["model.text_model", "text_model", "model", ""].iter() {
                                        if let Ok(v) = try_generic(vb.clone(), p) { return Ok(v); }
                                    }
                                }
                                TextEncoderClass::T5EncoderModel => {
                                    if let Ok(v) = try_t5(vb.clone()) { return Ok(v); }
                                    for p in ["model.text_model", "text_model", "model", ""].iter() {
                                        if let Ok(v) = try_generic(vb.clone(), p) { return Ok(v); }
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Unknown class: try all patterns
                        if let Ok(v) = try_clip(vb.clone()) { return Ok(v); }
                        if let Ok(v) = try_t5(vb.clone()) { return Ok(v); }
                        for p in ["model.text_model", "text_model", "model", ""].iter() {
                            if let Ok(v) = try_generic(vb.clone(), p) { return Ok(v); }
                        }
                        Err(anyhow::anyhow!("text_encoder embed/norm not found"))
                    },
                )
            };
            if let Ok(TextEmbPack { emb: emb_layer, norm_w, norm_b, kind }) = loaded {
                let embed_once = |s: &str| -> anyhow::Result<candle_core::Tensor> {
                    let enc = tok.encode(s, true).map_err(anyhow::Error::msg)?;
                    let ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
                    let input_ids = candle_core::Tensor::new(ids, &candle_core::Device::Cpu)?.unsqueeze(0)?; // [1,L]
                    // Embedding and normalization on CPU for numerical stability
                    let x = emb_layer.forward(&input_ids)?; // [1,L,D] on CPU
                    // Apply normalization according to the detected kind
                    let x_dtype = x.dtype();
                    let mut x = x.to_dtype(candle_core::DType::F32)?;
                    match kind {
                        NormKind::RmsNoBias => {
                            let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                            let eps_t = candle_core::Tensor::new(eps, &candle_core::Device::Cpu)?;
                            x = x.broadcast_div(&(variance.broadcast_add(&eps_t)?.sqrt()?))?;
                            let mut w = norm_w.clone().to_device(&candle_core::Device::Cpu)?;
                            if w.dtype() != x.dtype() { w = w.to_dtype(x.dtype())?; }
                            x = x.broadcast_mul(&w)?;
                        }
                        NormKind::LayerNormWithBias => {
                            let mean = x.mean_keepdim(candle_core::D::Minus1)?;
                            let xc = (&x - &mean)?;
                            let var = xc.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                            let eps_t = candle_core::Tensor::new(eps, &candle_core::Device::Cpu)?;
                            let xnorm = xc.broadcast_div(&(var.broadcast_add(&eps_t)?.sqrt()?))?;
                            let mut w = norm_w.clone().to_device(&candle_core::Device::Cpu)?;
                            if w.dtype() != xnorm.dtype() { w = w.to_dtype(xnorm.dtype())?; }
                            let mut y = xnorm.broadcast_mul(&w)?;
                            if let Some(b) = norm_b.as_ref() {
                                let mut b = b.clone().to_device(&candle_core::Device::Cpu)?;
                                if b.dtype() != y.dtype() { b = b.to_dtype(y.dtype())?; }
                                y = (&y + &b)?;
                            }
                            x = y;
                        }
                    }
                    let mut x = x.to_dtype(x_dtype)?;
                    // Pool along sequence to 1 token to reduce memory (keepdim to remain [1,1,D])
                    x = x.mean_keepdim(candle_core::D::Minus2)?;
                    // Move back to the pipeline device in pipeline dtype
                    let mut x = x.to_device(&self.device)?;
                    if x.dtype() != self.dtype { x = x.to_dtype(self.dtype)?; }
                    Ok(x)
                };
                let mut cond = embed_once(&opts.prompt)?;
                let uncond_text = opts.negative_prompt.as_deref().unwrap_or("");
                let mut uncond = embed_once(uncond_text)?;
                if cond.dtype() != self.dtype {
                    cond = cond.to_dtype(self.dtype)?;
                }
                if uncond.dtype() != self.dtype {
                    uncond = uncond.to_dtype(self.dtype)?;
                }
                (cond, uncond)
            } else {
                bail!("text_encoder embed/norm not found in weights (strict)");
            }
        } else {
            bail!("Tokenizer or text_encoder missing (strict)")
        };
    // Align text tensors to pipeline device; keep dtype consistent with model weights (avoid mixed-precision CUDA issues)
    if text_cond.device().is_cuda() != self.device.is_cuda() { text_cond = text_cond.to_device(&self.device)?; }
    if text_uncond.device().is_cuda() != self.device.is_cuda() { text_uncond = text_uncond.to_device(&self.device)?; }
    if text_cond.dtype() != self.dtype { text_cond = text_cond.to_dtype(self.dtype)?; }
    if text_uncond.dtype() != self.dtype { text_uncond = text_uncond.to_dtype(self.dtype)?; }
        if let Some(tf) = self.transformer_flux.as_ref() {
            let jd = tf.config.joint_attention_dim;
            if text_cond.dim(2)? != jd { anyhow::bail!("Text embedding dim mismatch: got {}, expected {} (joint_attention_dim)", text_cond.dim(2)?, jd); }
            if text_uncond.dim(2)? != jd { anyhow::bail!("Uncond text embedding dim mismatch: got {}, expected {} (joint_attention_dim)", text_uncond.dim(2)?, jd); }
        }

        // If using Flux-style transformer, ensure latent spatial dims are divisible by patch_size
    let mut use_full_attn: bool = false;
    let mut attn_override: Option<usize> = None;
        if let Some(tf) = tr_flux_opt {
            let ps = tf.config.patch_size.max(1);
            let (b, c, zh, zw) = z.dims4()?;
            let zh2 = ((zh + ps - 1) / ps) * ps;
            let zw2 = ((zw + ps - 1) / ps) * ps;
            if zh2 != zh || zw2 != zw {
                let mut z_pad = candle_core::Tensor::zeros((b, c, zh2, zw2), z.dtype(), z.device())?;
                z_pad = z_pad.slice_assign(&[0..b, 0..c, 0..zh, 0..zw], &z)?;
                z = z_pad;
            }
            // Flux expects patch embed input of out_channels * ps^2.
            // Enforce latent channels >= out_channels, and slice down if more.
            let expect = tf.config.out_channels;
            if c < expect {
                anyhow::bail!(
                    "Latent channels from VAE ({}) < transformer's out_channels ({}). Strict mode requires matching channels.",
                    c, expect
                );
            }
            if c > expect { z = z.i((.., 0..expect, .., ..))?; }
            // Log tokens and chunk size for visibility
            let n_tokens = (zh2 / ps) * (zw2 / ps);
            let mut chunk = tf.config.attn_chunk_size.unwrap_or(2048).min(n_tokens);
            // Reduce chunk further for large token counts to limit VRAM (fixed cap)
            if self.device.is_cuda() && n_tokens >= 20000 {
                chunk = chunk.min(1024);
            }
            // Heuristic: if token count is modest and on CUDA, disable chunking to reduce overhead (faster)
            if self.device.is_cuda() && n_tokens <= 4096 {
                use_full_attn = true;
            }
            // Stable override used for all forward calls this run
            attn_override = if use_full_attn { Some(usize::MAX) } else { Some(chunk) };
            log::info!(
                "[qwen-image-edit] Starting denoise: latent=({b},{c},{zh2},{zw2}), ps={ps}, tokens={n_tokens}, chunk={chunk}, heads={}, head_dim={}",
                tf.config.num_attention_heads,
                tf.config.attention_head_dim
            );
        }

        // Noise schedule
        let steps = opts.num_inference_steps.max(1);
        let sigmas = sched.inference_sigmas(steps);
        log::info!(
            "[qwen-image-edit] schedule prepared: steps={} (sigmas.len={})",
            steps,
            sigmas.len()
        );

        // Proper img2img init: mix encoded latents with noise based on strength
        // Compute start step index based on strength (1.0 -> start at highest noise; 0.0 -> start at near zero noise)
        let start_idx = ((steps as f32 - 1.0) * opts.strength.clamp(0.0, 1.0)).floor() as usize;
        let start_idx = start_idx.min(steps.saturating_sub(1));
        log::info!(
            "[qwen-image-edit] schedule: start_idx={} sigma0={:.6} z(before-noise) dims={:?} dtype={:?} device={:?}",
            start_idx,
            sigmas[start_idx].max(1e-6),
            z.dims(),
            z.dtype(),
            z.device()
        );
        let mut z = {
            // Bring z to f32 for noise math then cast back
            let z32 = z.to_dtype(candle_core::DType::F32)?;
            let mut noise = candle_core::Tensor::randn(0f32, 1f32, z32.dims(), &self.device)?;
            if noise.dtype() != z32.dtype() { noise = noise.to_dtype(z32.dtype())?; }
            // Use sigma corresponding to start_idx to scale noise (approximation)
            let sigma0 = sigmas[start_idx].max(1e-6);
            let mut mixed = z32.clone(); // keep encoded content strength implicit via start index
            mixed = (&mixed + noise.broadcast_mul(&candle_core::Tensor::new(sigma0, &self.device)?)?)?;
            mixed.to_dtype(self.dtype)?
        };

        // CFG mix (structure ready). If guidance_scale=1.0, falls back to conditional only.
        let guidance = opts.guidance_scale;
        // Initialize UI for the denoise loop using overall progress
        QWEN_EDIT_STATUS.set_state(StatusState::Running, "Denoising");
        QWEN_EDIT_STATUS.set_detail(format!("Step {}/{}", 0, steps));
        // Reuse guidance scalar tensor across steps to reduce allocations
        let g_scale = if guidance > 1.0 {
            let mut g = candle_core::Tensor::new(guidance as f32, &self.device)?;
            if g.dtype() != z.dtype() { g = g.to_dtype(z.dtype())?; }
            Some(g)
        } else { None };
        // If using Qwen/GGUF transformer, log a similar start line for visibility
        if prefer_qwen && tr_qwen_opt.is_some() {
            let (b, c, zh, zw) = z.dims4()?;
            let model = tr_qwen_opt.unwrap();
            let n_tokens = zh * zw; // tokens == spatial locations
            log::info!(
                "[qwen-image-edit] Starting denoise (Qwen): latent=({b},{c},{zh},{zw}), tokens={n_tokens}, heads={}, head_dim={}",
                model.config.num_attention_heads,
                model.config.attention_head_dim
            );
        }

        // If using Flux-style transformer, precompute text projections once for cond/uncond reuse
        let (pre_text_cond, pre_text_uncond) = if let Some(tf) = tr_flux_opt {
            let t0 = std::time::Instant::now();
            log::info!("[qwen-image-edit] precompute_text start");
            // Ensure text tensors are on the same device/dtype as transformer
            let mut tc = text_cond.clone();
            let mut tu = text_uncond.clone();
            if !tc.device().same_device(&tf.device) { tc = tc.to_device(&tf.device)?; }
            if !tu.device().same_device(&tf.device) { tu = tu.to_device(&tf.device)?; }
            if tc.dtype() != self.dtype { tc = tc.to_dtype(self.dtype)?; }
            if tu.dtype() != self.dtype { tu = tu.to_dtype(self.dtype)?; }
            let outc = tf.precompute_text(&tc).map_err(|e| anyhow::anyhow!(
                "Flux precompute_text(cond) failed: text shape={:?} dtype={:?} device={:?} | {}",
                tc.dims(), tc.dtype(), tc.device(), e
            ))?;
            let outu = tf.precompute_text(&tu).map_err(|e| anyhow::anyhow!(
                "Flux precompute_text(uncond) failed: text shape={:?} dtype={:?} device={:?} | {}",
                tu.dims(), tu.dtype(), tu.device(), e
            ))?;
            log::info!("[qwen-image-edit] precompute_text done in {:?}", t0.elapsed());
            (Some(outc), Some(outu))
        } else { (None, None) };
        for i in start_idx..steps {
            let step_t0 = std::time::Instant::now();
            // Respect cancel flag
            if let Some(flag) = &opts.cancel { if flag.load(std::sync::atomic::Ordering::Relaxed) { log::warn!("[qwen-image-edit] Run canceled at step {}/{}", i, steps); break; } }
            let sigma_from = sigmas[i];
            let sigma_to = sigmas[i + 1];
            log::info!("[qwen-image-edit] step {} begin: sigma_from={:.6} -> sigma_to={:.6}", i + 1, sigma_from, sigma_to);
            // Update overall progress at the start of the iteration
            QWEN_EDIT_STATUS.set_progress(1 + (i as u64) + 1, total_overall);
            QWEN_EDIT_STATUS.set_detail(format!("Step {}/{} (start {})", i + 1, steps, start_idx + 1));
            // Scale model input to sigma space: x_in = x / sqrt(sigma^2 + 1)
            let scale = 1.0f32 / (sigma_from * sigma_from + 1.0).sqrt();
            let z_in = z.broadcast_mul(&candle_core::Tensor::new(scale, &self.device)?.to_dtype(z.dtype())?)?;
            let pred_kind = self
                .scheduler
                .as_ref()
                .map(|s| s.prediction_type.as_str())
                .unwrap_or("epsilon");
            if guidance > 1.0 {
                // Scope temporaries to ensure prompt freeing of GPU buffers each iteration
                let (zu, zc) = {
                    log::info!("[qwen-image-edit] step {}: uncond forward start", i + 1);
                    let zu = if prefer_qwen && tr_qwen_opt.is_some() {
                        tr_qwen_opt.unwrap().forward(&z_in, &text_uncond)?
                    } else if let Some(tf) = tr_flux_opt {
                        // Align inputs to transformer device/dtype
                        let mut zi = z_in.clone();
                        let mut tu = text_uncond.clone();
                        if !zi.device().same_device(&tf.device) { zi = zi.to_device(&tf.device)?; }
                        if zi.dtype() != self.dtype { zi = zi.to_dtype(self.dtype)?; }
                        if !tu.device().same_device(&tf.device) { tu = tu.to_device(&tf.device)?; }
                        if tu.dtype() != self.dtype { tu = tu.to_dtype(self.dtype)?; }
                        match tf.forward_with_precomputed(&zi, &tu, pre_text_uncond.as_ref(), attn_override) {
                            Ok(v) => v,
                            Err(e) => {
                                log::error!(
                                    "[qwen-image-edit] Flux forward(uncond) CUDA path failed: z_in={:?} text={:?} dz={:?} dt={:?} devz={:?} devt={:?} | {}",
                                    z_in.dims(), text_uncond.dims(), z_in.dtype(), text_uncond.dtype(), z_in.device(), text_uncond.device(), e
                                );
                                // Fallback: try on CPU once for diagnostics
                                let z_cpu = z_in.to_device(&candle_core::Device::Cpu)?;
                                let t_cpu = text_uncond.to_device(&candle_core::Device::Cpu)?;
                                // Build a CPU Flux model to avoid device mismatch
                                let v = {
                                    let cfg_flux = tf.config.clone();
                                    let vb_cpu = unsafe {
                                        candle_nn::VarBuilder::from_mmaped_safetensors(
                                            &self.paths.transformer_files,
                                            candle_core::DType::F32,
                                            &candle_core::Device::Cpu,
                                        )?
                                    };
                                    let tf_cpu = crate::ai::flux_transformer::FluxTransformer2DModel::new(&cfg_flux, vb_cpu)?;
                                    tf_cpu.forward_with_precomputed(&z_cpu, &t_cpu, None, attn_override)
                                }
                                    .map_err(|e2| anyhow::anyhow!("Flux forward(uncond) failed on CPU too: {}", e2))?;
                                v.to_device(&self.device)?
                            }
                        }
                    } else {
                        tr_qwen_opt.unwrap().forward(&z_in, &text_uncond)?
                    };
                    log::info!("[qwen-image-edit] step {}: uncond forward done (+{:?})", i + 1, step_t0.elapsed());
                    log::info!("[qwen-image-edit] step {}: cond forward start", i + 1);
                    let zc = if prefer_qwen && tr_qwen_opt.is_some() {
                        tr_qwen_opt.unwrap().forward(&z_in, &text_cond)?
                    } else if let Some(tf) = tr_flux_opt {
                        let mut zi = z_in.clone();
                        let mut tc = text_cond.clone();
                        if !zi.device().same_device(&tf.device) { zi = zi.to_device(&tf.device)?; }
                        if zi.dtype() != self.dtype { zi = zi.to_dtype(self.dtype)?; }
                        if !tc.device().same_device(&tf.device) { tc = tc.to_device(&tf.device)?; }
                        if tc.dtype() != self.dtype { tc = tc.to_dtype(self.dtype)?; }
                        match tf.forward_with_precomputed(&zi, &tc, pre_text_cond.as_ref(), attn_override) {
                            Ok(v) => v,
                            Err(e) => {
                                log::error!(
                                    "[qwen-image-edit] Flux forward(cond) CUDA path failed: z_in={:?} text={:?} dz={:?} dt={:?} devz={:?} devt={:?} | {}",
                                    z_in.dims(), text_cond.dims(), z_in.dtype(), text_cond.dtype(), z_in.device(), text_cond.device(), e
                                );
                                let z_cpu = z_in.to_device(&candle_core::Device::Cpu)?;
                                let t_cpu = text_cond.to_device(&candle_core::Device::Cpu)?;
                                let v = {
                                    let cfg_flux = tf.config.clone();
                                    let vb_cpu = unsafe {
                                        candle_nn::VarBuilder::from_mmaped_safetensors(
                                            &self.paths.transformer_files,
                                            candle_core::DType::F32,
                                            &candle_core::Device::Cpu,
                                        )?
                                    };
                                    let tf_cpu = crate::ai::flux_transformer::FluxTransformer2DModel::new(&cfg_flux, vb_cpu)?;
                                    tf_cpu.forward_with_precomputed(&z_cpu, &t_cpu, None, attn_override)
                                }
                                    .map_err(|e2| anyhow::anyhow!("Flux forward(cond) failed on CPU too: {}", e2))?;
                                v.to_device(&self.device)?
                            }
                        }
                    } else {
                        tr_qwen_opt.unwrap().forward(&z_in, &text_cond)?
                    };
                    log::info!("[qwen-image-edit] step {}: cond forward done (+{:?})", i + 1, step_t0.elapsed());
                    (zu, zc)
                };
                let diff = (zc - &zu)?;
                let guided = (&zu + diff.broadcast_mul(g_scale.as_ref().unwrap())?)?; // guided model output
                // Derive epsilon depending on prediction_type
                let eps = if pred_kind == "sample" {
                    // model output is x0 => eps = (x - x0)/sigma
                    let num = (&z - &guided)?;
                    let inv = if sigma_from.abs() < 1e-6 { 0.0 } else { 1.0 / sigma_from };
                    num.broadcast_mul(&candle_core::Tensor::new(inv, &self.device)?.to_dtype(z.dtype())?)?
                } else {
                    // epsilon
                    guided.clone()
                };
                // Euler step: x_{t-1} = x_t + (sigma_to - sigma_from) * eps
                let ds = sigma_to - sigma_from;
                z = (&z + eps.broadcast_mul(&candle_core::Tensor::new(ds, &self.device)?.to_dtype(z.dtype())?)?)?;
                // Live preview from predicted x0
                if let (Some(every), Some(tx)) = (opts.preview_every_n, opts.preview_tx.as_ref()) {
                    if every > 0 && (i + 1) % every == 0 {
                        let x0 = if pred_kind == "sample" {
                            guided
                        } else {
                            (&z - eps.broadcast_mul(&candle_core::Tensor::new(sigma_from, &self.device)?.to_dtype(z.dtype())?)?)?
                        };
                        if let Ok(mut img_t) = vae.decode_to_original(&x0, oh, ow) {
                            if img_t.dtype() != self.dtype { img_t = img_t.to_dtype(self.dtype)?; }
                            let x32 = img_t.to_dtype(candle_core::DType::F32)?;
                            let one_vec = candle_core::Tensor::ones(x32.dims(), candle_core::DType::F32, &x32.device())?;
                            let half = candle_core::Tensor::new(0.5f32, &x32.device())?;
                            let x32 = (&x32 + &one_vec)?;
                            let x32 = x32.broadcast_mul(&half)?;
                            let x32 = x32.clamp(0f32, 1f32)?;
                            let scale = candle_core::Tensor::new(255.0f32, &x32.device())?;
                            let x_u8 = x32.broadcast_mul(&scale)?.to_dtype(candle_core::DType::U8)?;
                            let chw = x_u8.i(0)?; // [3,H,W]
                            let (hh, ww) = (chw.dim(1)?, chw.dim(2)?);
                            let hwc = chw.permute((1, 2, 0))?; // [H,W,3]
                            let hwc_cpu = hwc.to_device(&candle_core::Device::Cpu)?;
                            let flat = hwc_cpu.reshape((hh * ww * 3,))?;
                            if let Ok(data) = flat.to_vec1::<u8>() {
                                if let Some(img) = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_vec(ww as u32, hh as u32, data) {
                                    let mut out_bytes = Vec::new();
                                    let mut cur = std::io::Cursor::new(&mut out_bytes);
                                    if image::DynamicImage::ImageRgb8(img).write_to(&mut cur, image::ImageFormat::Png).is_ok() {
                                        let _ = tx.send(out_bytes);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                let pred = if prefer_qwen && tr_qwen_opt.is_some() {
                    tr_qwen_opt.unwrap().forward(&z_in, &text_cond)?
                } else if let Some(tf) = tr_flux_opt {
                    // Align inputs to transformer device/dtype
                    let mut zi = z_in.clone();
                    let mut tc = text_cond.clone();
                    if !zi.device().same_device(&tf.device) { zi = zi.to_device(&tf.device)?; }
                    if zi.dtype() != self.dtype { zi = zi.to_dtype(self.dtype)?; }
                    if !tc.device().same_device(&tf.device) { tc = tc.to_device(&tf.device)?; }
                    if tc.dtype() != self.dtype { tc = tc.to_dtype(self.dtype)?; }
                    tf.forward_with_precomputed(&zi, &tc, pre_text_cond.as_ref(), attn_override)
                        .map_err(|e| anyhow::anyhow!("Flux forward(single) failed: z_in={:?} text={:?} dtype(z)={:?} dtype(t)={:?} device(z)={:?} device(t)={:?} | {}", z_in.dims(), text_cond.dims(), z_in.dtype(), text_cond.dtype(), z_in.device(), text_cond.device(), e))?
                } else {
                    tr_qwen_opt.unwrap().forward(&z_in, &text_cond)?
                };
                // Derive epsilon depending on prediction_type
                let eps = if pred_kind == "sample" {
                    let num = (&z - &pred)?;
                    let inv = if sigma_from.abs() < 1e-6 { 0.0 } else { 1.0 / sigma_from };
                    num.broadcast_mul(&candle_core::Tensor::new(inv, &self.device)?.to_dtype(z.dtype())?)?
                } else {
                    pred.clone()
                };
                let ds = sigma_to - sigma_from;
                z = (&z + eps.broadcast_mul(&candle_core::Tensor::new(ds, &self.device)?.to_dtype(z.dtype())?)?)?;
                // Live preview from predicted x0
                if let (Some(every), Some(tx)) = (opts.preview_every_n, opts.preview_tx.as_ref()) {
                    if every > 0 && (i + 1) % every == 0 {
                        let x0 = if pred_kind == "sample" { pred } else { (&z - eps.broadcast_mul(&candle_core::Tensor::new(sigma_from, &self.device)?.to_dtype(z.dtype())?)?)? };
                        if let Ok(mut img_t) = vae.decode_to_original(&x0, oh, ow) {
                            if img_t.dtype() != self.dtype { img_t = img_t.to_dtype(self.dtype)?; }
                            let x32 = img_t.to_dtype(candle_core::DType::F32)?;
                            let one_vec = candle_core::Tensor::ones(x32.dims(), candle_core::DType::F32, &x32.device())?;
                            let half = candle_core::Tensor::new(0.5f32, &x32.device())?;
                            let x32 = (&x32 + &one_vec)?;
                            let x32 = x32.broadcast_mul(&half)?;
                            let x32 = x32.clamp(0f32, 1f32)?;
                            let scale = candle_core::Tensor::new(255.0f32, &x32.device())?;
                            let x_u8 = x32.broadcast_mul(&scale)?.to_dtype(candle_core::DType::U8)?;
                            let chw = x_u8.i(0)?; // [3,H,W]
                            let (hh, ww) = (chw.dim(1)?, chw.dim(2)?);
                            let hwc = chw.permute((1, 2, 0))?; // [H,W,3]
                            let hwc_cpu = hwc.to_device(&candle_core::Device::Cpu)?;
                            let flat = hwc_cpu.reshape((hh * ww * 3,))?;
                            if let Ok(data) = flat.to_vec1::<u8>() {
                                if let Some(img) = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_vec(ww as u32, hh as u32, data) {
                                    let mut out_bytes = Vec::new();
                                    let mut cur = std::io::Cursor::new(&mut out_bytes);
                                    if image::DynamicImage::ImageRgb8(img).write_to(&mut cur, image::ImageFormat::Png).is_ok() {
                                        let _ = tx.send(out_bytes);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Decode back to original size
        QWEN_EDIT_STATUS.set_detail("Decoding");
        QWEN_EDIT_STATUS.set_progress(1 + (steps as u64), total_overall);
        // Watchdog to surface long-running decode: update detail with elapsed time every second
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        use std::time::{Duration, Instant};
        let decoding_flag = Arc::new(AtomicBool::new(true));
        let decoding_flag_w = decoding_flag.clone();
        let z_info = format!(
            "z: dtype={:?} device={:?} shape={:?}; vae on {:?}",
            z.dtype(), z.device(), z.dims(), self.device
        );
        log::info!("[qwen-image-edit] Starting VAE decode_to_original: {} -> target=({}, {})", z_info, oh, ow);
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut secs = 0u64;
            while decoding_flag_w.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(1));
                secs += 1;
                // Avoid spamming logs; update UI detail only
                QWEN_EDIT_STATUS.set_detail(format!("Decoding… {}s", secs));
            }
            let total = start.elapsed().as_secs();
            if total > 0 { log::info!("[qwen-image-edit] VAE decode finished (watchdog): {}s", total); }
        });
        // Decode with CPU fallback if we used a CPU VAE for encode, or CUDA decode fails similarly
        let mut x_out = match vae.decode_to_original(&z, oh, ow) { Ok(img) => img, Err(e) => return Err(anyhow::Error::msg(format!("VAE decode failed (strict): {}", e))) };
        // Stop watchdog thread
        decoding_flag.store(false, Ordering::Relaxed);
        if x_out.dtype() != self.dtype {
            x_out = x_out.to_dtype(self.dtype)?;
        }
        // To PNG efficiently: move to CPU, HWC layout, flatten, and encode
        QWEN_EDIT_STATUS.set_detail("Saving");
        QWEN_EDIT_STATUS.set_progress(1 + (steps as u64) + 1, total_overall);
    let x32 = x_out.to_dtype(candle_core::DType::F32)?;
    // Most VAEs output in [-1, 1]; remap to [0,1] before clamping
    let one_vec = candle_core::Tensor::ones(x32.dims(), candle_core::DType::F32, &x32.device())?;
    let half = candle_core::Tensor::new(0.5f32, &x32.device())?;
    let x32 = (&x32 + &one_vec)?;
    let x32 = x32.broadcast_mul(&half)?;
    let x32 = x32.clamp(0f32, 1f32)?;
    let scale = candle_core::Tensor::new(255.0f32, &x32.device())?;
        let x_u8 = x32.broadcast_mul(&scale)?.to_dtype(candle_core::DType::U8)?;
        let chw = x_u8.i(0)?; // [3,H,W]
        let (hh, ww) = (chw.dim(1)?, chw.dim(2)?);
        let hwc = chw.permute((1, 2, 0))?; // [H,W,3]
        let hwc_cpu = hwc.to_device(&candle_core::Device::Cpu)?;
        let flat = hwc_cpu.reshape((hh * ww * 3,))?;
        let data: Vec<u8> = flat.to_vec1::<u8>()?;
        if let Some(img) = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_vec(
            ww as u32,
            hh as u32,
            data,
        ) {
            let mut out_bytes = Vec::new();
            let mut cur = std::io::Cursor::new(&mut out_bytes);
            image::DynamicImage::ImageRgb8(img).write_to(&mut cur, image::ImageFormat::Png)?;
            // Mark complete then reset to idle
            QWEN_EDIT_STATUS.set_progress(total_overall, total_overall);
            QWEN_EDIT_STATUS.set_state(StatusState::Idle, "Done");
            // Optionally clear progress bar after completion
            QWEN_EDIT_STATUS.set_progress(0, 0);
            Ok(out_bytes)
        } else {
            anyhow::bail!("Failed to construct image buffer: invalid dimensions")
        }
    }
}
