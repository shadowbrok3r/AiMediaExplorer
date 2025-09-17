use anyhow::{Result, bail};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;

use crate::ai::hf::{hf_model, hf_get_file, with_mmap_varbuilder_multi, pick_device_cuda0_or_cpu};
use super::config::{ModelIndex, ComponentSpec, SchedulerConfig};
use super::scheduler::FlowMatchEulerDiscreteScheduler;
use image::{GenericImageView, ImageBuffer, Rgb, imageops::FilterType};

#[derive(Debug, Clone)]
pub struct QwenImageEditPaths {
    pub repo_id: String,
    pub model_index_json: std::path::PathBuf,
    pub tokenizer_json: std::path::PathBuf,
    pub text_encoder_files: Vec<std::path::PathBuf>,
    pub transformer_files: Vec<std::path::PathBuf>,
    pub vae_files: Vec<std::path::PathBuf>,
    pub scheduler_json: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct QwenImageEditPipeline {
    pub device: Device,
    pub dtype: DType,
    pub paths: QwenImageEditPaths,
    pub model_index: Option<ModelIndex>,
    pub scheduler: Option<FlowMatchEulerDiscreteScheduler>,
    // TODO: hold tokenizer, text encoder, unet/transformer, vae, scheduler instances
}

impl QwenImageEditPipeline {
    pub fn load_from_hf(repo_id: &str, prefer_dtype: DType) -> Result<Self> {
        // Discover and download files described in the prompt (model_index.json, safetensors for transformer/vae, tokenizer.json, scheduler config)
        let repo = hf_model(repo_id)?;
        let model_index_json = hf_get_file(&repo, "model_index.json")?;
        // Parse model_index.json
        let model_index: Option<ModelIndex> = std::fs::read_to_string(&model_index_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        // Tokenizer
        let tokenizer_json = repo.get("tokenizer/tokenizer.json").or_else(|_| repo.get("tokenizer.json")).unwrap_or(repo.get("tokenizer.json")?);
        // Text encoder weights (could be multiple shards)
        let mut text_encoder_files = Vec::new();
        for name in ["text_encoder/model.safetensors", "text_encoder/diffusion_pytorch_model.safetensors"].iter() {
            if let Ok(p) = repo.get(name) { text_encoder_files.push(p); }
        }
        // Transformer/UNet weights (can be sharded)
        let mut transformer_files = Vec::new();
        for name in ["transformer/model.safetensors", "transformer/diffusion_pytorch_model.safetensors"].iter() {
            if let Ok(p) = repo.get(name) { transformer_files.push(p); }
        }
        if transformer_files.is_empty() { bail!("No transformer safetensors found in repo {repo_id}"); }
        // VAE
        let mut vae_files = Vec::new();
        for name in ["vae/diffusion_pytorch_model.safetensors", "vae/model.safetensors"].iter() {
            if let Ok(p) = repo.get(name) { vae_files.push(p); }
        }
        if vae_files.is_empty() { bail!("No VAE safetensors found in repo {repo_id}"); }
        // Scheduler config
        let scheduler_json = repo.get("scheduler/scheduler_config.json").or_else(|_| repo.get("scheduler_config.json"))?;

        let device = pick_device_cuda0_or_cpu();
        let dtype = prefer_dtype;

        // Load scheduler config (best-effort)
        let scheduler_cfg: Option<SchedulerConfig> = std::fs::read_to_string(&scheduler_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        let scheduler = scheduler_cfg.as_ref().map(FlowMatchEulerDiscreteScheduler::from_config);

        Ok(Self {
            device,
            dtype,
            paths: QwenImageEditPaths {
                repo_id: repo_id.to_string(),
                model_index_json,
                tokenizer_json,
                text_encoder_files,
                transformer_files,
                vae_files,
                scheduler_json,
            },
            model_index,
            scheduler,
        })
    }

    pub fn info(&self) {
        log::info!("[qwen-image-edit] repo={} dtype={:?} device={:?}", self.paths.repo_id, self.dtype, self.device);
        log::info!(" transformer: {} shards", self.paths.transformer_files.len());
        log::info!(" vae: {} shards", self.paths.vae_files.len());
        log::info!(" text-encoder: {} shards", self.paths.text_encoder_files.len());
        if let Some(mi) = &self.model_index {
            log::info!(" model_index: class={:?} version={:?} type={:?}", mi._class_name, mi._diffusers_version, mi.model_type);
            let pr = |label: &str, c: &Option<ComponentSpec>| {
                if let Some(c) = c {
                    match c {
                        ComponentSpec::Ref(r) => log::info!("  {}: ref {:?}", label, r._name_or_path),
                        ComponentSpec::Pair(p) => log::info!("  {}: pair {:?}", label, p),
                    }
                }
            };
            pr("processor", &mi.processor);
            pr("scheduler", &mi.scheduler);
            pr("text_encoder", &mi.text_encoder);
            pr("tokenizer", &mi.tokenizer);
            pr("transformer", &mi.transformer);
            pr("vae", &mi.vae);
            pr("unet", &mi.unet);
        }
        if let Some(s) = &self.scheduler {
            log::info!(" scheduler: FlowMatchEulerDiscrete num_train_timesteps={} spacing={} pred={}", s.num_train_timesteps, s.timestep_spacing, s.prediction_type);
        }
    }

    pub fn dummy_run(&self) -> Result<()> {
        // Placeholder to exercise safetensors load paths via VarBuilder
        unsafe {
            let _ = with_mmap_varbuilder_multi(&self.paths.transformer_files, self.dtype, &self.device, |_vb: VarBuilder| {
                Ok(())
            })?;
            let _ = with_mmap_varbuilder_multi(&self.paths.vae_files, self.dtype, &self.device, |_vb: VarBuilder| {
                Ok(())
            })?;
        }
        if let Some(s) = &self.scheduler {
            let _ts = s.timesteps(10);
        }
        Ok(())
    }

    // First working edit pass: deterministic latent smoothing guided by scheduler timesteps.
    // This is a placeholder for the full diffusion pipeline; it performs real image processing
    // so users can iterate in the UI while we wire the actual transformer/vae.
    pub fn run_edit(&self, image_path: &std::path::Path, opts: &crate::ai::qwen_image_edit::EditOptions) -> Result<Vec<u8>> {
        // Load and normalize source image
        let dyn_img = image::ImageReader::open(image_path)?.decode()?;
        let (w, h) = dyn_img.dimensions();
        let src_rgb = dyn_img.to_rgb8();
        // Choose latent size as image/8 (typical for VAE), keep >= 32
        let lw = (w.max(64) / 8).max(16);
        let lh = (h.max(64) / 8).max(16);
        let low_rgb = image::imageops::resize(&src_rgb, lw, lh, FilterType::Triangle);

        // Map RGB -> 4-channel latent (learned-free projection)
        let mut latents: Vec<f32> = vec![0.0; (lw * lh * 4) as usize];
        for y in 0..lh {
            for x in 0..lw {
                let p = low_rgb.get_pixel(x, y);
                let r = p[0] as f32 / 255.0;
                let g = p[1] as f32 / 255.0;
                let b = p[2] as f32 / 255.0;
                let i = ((y * lw + x) * 4) as usize;
                latents[i + 0] = r;
                latents[i + 1] = g;
                latents[i + 2] = b;
                latents[i + 3] = (r + g + b) / 3.0;
            }
        }
        // Add prompt-strength controlled noise
        let seed = opts.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        });
        // Simple xorshift-based PRNG to avoid extra deps
        let mut state = seed ^ 0x9E3779B97F4A7C15u64;
        let mut next_f32 = || {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545F4914F6CDD1Du64);
            // map to [0,1)
            (v as f64 / (u64::MAX as f64)) as f32
        };
        let noise_scale = opts.strength.clamp(0.0, 1.0);
        for v in latents.iter_mut() {
            let r = next_f32() * 2.0 - 1.0; // [-1,1]
            *v = (*v * (1.0 - noise_scale) + r * noise_scale).clamp(-1.0, 1.0);
        }

        // Smoothing kernel for denoise step (separable 1,2,1)
        let steps = opts.num_inference_steps.max(1) as usize;
        let ch = 4usize;
        if let Some(sched) = &self.scheduler {
            let sigmas = sched.inference_sigmas(steps);
            // Working buffers
            let mut latents_in = vec![0.0f32; latents.len()];
            let mut blurred = vec![0.0f32; latents.len()];
            let mut tmp = vec![0.0f32; latents.len()];
            // 1-2-1 separable kernel
            let k = [1.0f32, 2.0, 1.0];
            let ksum = 4.0f32;
            let blur = |src: &[f32], dst: &mut [f32], tmp: &mut [f32]| {
                // Horizontal pass into tmp
                for y in 0..(lh as usize) {
                    for x in 0..(lw as usize) {
                        for c in 0..ch {
                            let xm1 = x.saturating_sub(1);
                            let xp1 = (x + 1).min((lw as usize) - 1);
                            let idx = (y * lw as usize + x) * ch + c;
                            let i0 = (y * lw as usize + xm1) * ch + c;
                            let i1 = idx;
                            let i2 = (y * lw as usize + xp1) * ch + c;
                            tmp[idx] = (src[i0] * k[0] + src[i1] * k[1] + src[i2] * k[2]) / ksum;
                        }
                    }
                }
                // Vertical pass into dst
                for y in 0..(lh as usize) {
                    for x in 0..(lw as usize) {
                        for c in 0..ch {
                            let ym1 = y.saturating_sub(1);
                            let yp1 = (y + 1).min((lh as usize) - 1);
                            let idx = (y * lw as usize + x) * ch + c;
                            let i0 = (ym1 * lw as usize + x) * ch + c;
                            let i1 = idx;
                            let i2 = (yp1 * lw as usize + x) * ch + c;
                            dst[idx] = (tmp[i0] * k[0] + tmp[i1] * k[1] + tmp[i2] * k[2]) / ksum;
                        }
                    }
                }
            };

            for i in 0..steps {
                let sigma_from = sigmas[i];
                let sigma_to = sigmas[i + 1];
                // Prepare model input
                latents_in.copy_from_slice(&latents);
                sched.scale_model_input(&mut latents_in, sigma_from);
                // Pseudo-model prediction: high-frequency component as epsilon approx
                blur(&latents_in, &mut blurred, &mut tmp);
                let mut model_eps = vec![0.0f32; latents.len()];
                for j in 0..latents.len() {
                    model_eps[j] = latents_in[j] - blurred[j];
                }
                // Euler step in sigma space
                sched.step_euler(&mut latents, &model_eps, sigma_from, sigma_to);
            }
        }

        // Decode latent -> RGB (project first 3 channels) and upscale back to original size
        let mut low_out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(lw, lh);
        for y in 0..lh {
            for x in 0..lw {
                let i = ((y * lw + x) * 4) as usize;
                let r = (latents[i + 0].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (latents[i + 1].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (latents[i + 2].clamp(0.0, 1.0) * 255.0) as u8;
                low_out.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        let out_img = image::imageops::resize(&low_out, w, h, FilterType::Lanczos3);
        let mut buf = Vec::new();
        {
            let mut cur = std::io::Cursor::new(&mut buf);
            out_img.write_to(&mut cur, image::ImageFormat::Png)?;
        }
        Ok(buf)
    }
}
