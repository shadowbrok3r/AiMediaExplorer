use anyhow::{Result, bail};
use candle_core::IndexOp;
use candle_core::Module;
use candle_core::{DType, Device};
use candle_nn::{Embedding, VarBuilder};
// use image::{GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use super::config::{ComponentSpec, ModelIndex, SchedulerConfig};
use super::scheduler::FlowMatchEulerDiscreteScheduler;
use crate::ai::hf::{hf_get_file, hf_model, pick_device_cuda0_or_cpu, with_mmap_varbuilder_multi};
use crate::ai::flux_transformer::{FluxTransformer2DModel, FluxTransformerConfig};
use crate::ai::qwen_image_transformer::{QwenImageTransformer2DModel, QwenImageTransformerConfig};
use regex::Regex;
use crate::ai::vae::{
    VaeConfig as AEKLConfig, VaeLike, build_vae_from_files,
    build_vae_from_files_with_5d_squeeze, build_qwen_vae_from_files_strict,
};
use safetensors::SafeTensors;
use std::path::{Path, PathBuf};
use std::collections::HashSet;
use tokenizers::Tokenizer;
use crate::status::{QWEN_EDIT_STATUS, GlobalStatusIndicator, StatusState};
#[derive(Debug, Clone)]
pub enum TransformerClass {
    UNet2DConditionModel,
    FluxTransformer2DModel,
    DiT,
    QwenImageTransformer2DModel,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub enum TextEncoderClass {
    CLIPTextModel,
    T5EncoderModel,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct QwenImageEditPaths {
    pub repo_id: String,
    pub model_index_json: std::path::PathBuf,
    pub tokenizer_json: std::path::PathBuf,
    pub text_encoder_files: Vec<std::path::PathBuf>,
    pub transformer_files: Vec<std::path::PathBuf>,
    pub vae_files: Vec<std::path::PathBuf>,
    pub scheduler_json: std::path::PathBuf,
    // Optional GGUF alternatives
    pub transformer_gguf: Option<std::path::PathBuf>,
    pub text_encoder_gguf: Option<std::path::PathBuf>,
    pub mmproj_gguf: Option<std::path::PathBuf>,
    // Optional configs
    pub processor_preprocessor_config: Option<std::path::PathBuf>,
    pub processor_tokenizer_config: Option<std::path::PathBuf>,
    pub processor_tokenizer_json: Option<std::path::PathBuf>,
    pub tokenizer_config: Option<std::path::PathBuf>,
    pub text_encoder_config: Option<std::path::PathBuf>,
    pub transformer_config: Option<std::path::PathBuf>,
    pub vae_config: Option<std::path::PathBuf>,
}

pub struct QwenImageEditPipeline {
    pub device: Device,
    pub dtype: DType,
    pub paths: QwenImageEditPaths,
    pub model_index: Option<ModelIndex>,
    pub scheduler: Option<FlowMatchEulerDiscreteScheduler>,
    pub tokenizer: Option<Tokenizer>,
    pub text_encoder_config_json: Option<serde_json::Value>,
    pub transformer_config_json: Option<serde_json::Value>,
    pub vae_config_json: Option<serde_json::Value>,
    pub text_encoder_class: Option<TextEncoderClass>,
    pub transformer_class: Option<TransformerClass>,
    // We intentionally avoid binding to a local Qwen2.5-VL implementation here.
    // The text encoder lives under the repo's text_encoder/ with its own safetensors.
    // Track availability only; actual embedding generation will be wired natively later.
    pub has_text_encoder: bool,
    pub vae: Option<Box<dyn VaeLike>>, // Boxed VAE abstraction
    pub transformer_model: Option<QwenImageTransformer2DModel>,
    pub transformer_flux: Option<FluxTransformer2DModel>,
    // TODO: hold tokenizer, text encoder, unet/transformer, vae, scheduler instances
}

mod load;

mod denoise;

impl QwenImageEditPipeline {
    // Strictly load from a local HF snapshot folder (no GGUF, no simplified fallbacks)
    pub fn load_from_local(root: &Path, prefer_dtype: DType) -> Result<Self> {
        let join = |s: &str| root.join(s);
        let model_index_json = join("model_index.json");
        let model_index: Option<ModelIndex> = std::fs::read_to_string(&model_index_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());

        // Tokenizer discovery
        let tokenizer_json = [
            join("tokenizer/tokenizer.json"),
            join("processor/tokenizer.json"),
            join("tokenizer.json"),
            join("processor/tokenizer_config.json"),
            join("tokenizer/tokenizer_config.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("No tokenizer json found under {}", root.display()))?;

        // Optional processor configs
        let processor_preprocessor_config = Some(join("processor/preprocessor_config.json")).filter(|p| p.exists());
        let processor_tokenizer_config = Some(join("processor/tokenizer_config.json")).filter(|p| p.exists());
        let processor_tokenizer_json = Some(join("processor/tokenizer.json")).filter(|p| p.exists());
        let tokenizer_config = Some(join("tokenizer/tokenizer_config.json")).filter(|p| p.exists());

        // Text encoder
        let text_encoder_config = Some(join("text_encoder/config.json")).filter(|p| p.exists());
        let mut text_encoder_files: Vec<PathBuf> = Vec::new();
        let te_index = join("text_encoder/model.safetensors.index.json");
        if te_index.exists() {
            if let Ok(s) = std::fs::read_to_string(&te_index) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                        let mut uniq: HashSet<String> = HashSet::new();
                        for f in map.values().filter_map(|v| v.as_str()) { uniq.insert(f.to_string()); }
                        let mut list: Vec<_> = uniq.into_iter().collect();
                        list.sort();
                        for f in list { text_encoder_files.push(join(&format!("text_encoder/{}", f))); }
                    }
                }
            }
        }
        if text_encoder_files.is_empty() {
            for name in [
                "text_encoder/model.safetensors",
                "text_encoder/diffusion_pytorch_model.safetensors",
            ] {
                let p = join(name);
                if p.exists() { text_encoder_files.push(p); }
            }
        }

        // Transformer
        let transformer_config = Some(join("transformer/config.json")).filter(|p| p.exists());
        let mut transformer_files: Vec<PathBuf> = Vec::new();
        let tr_index = join("transformer/diffusion_pytorch_model.safetensors.index.json");
        if tr_index.exists() {
            if let Ok(s) = std::fs::read_to_string(&tr_index) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                        let mut uniq: HashSet<String> = HashSet::new();
                        for f in map.values().filter_map(|v| v.as_str()) { uniq.insert(f.to_string()); }
                        let mut list: Vec<_> = uniq.into_iter().collect();
                        list.sort();
                        for f in list { transformer_files.push(join(&format!("transformer/{}", f))); }
                    }
                }
            }
        }
        if transformer_files.is_empty() {
            for name in [
                "transformer/model.safetensors",
                "transformer/diffusion_pytorch_model.safetensors",
            ] {
                let p = join(name);
                if p.exists() { transformer_files.push(p); }
            }
        }
        if transformer_files.is_empty() { bail!("No transformer safetensors under {}", root.display()); }

        // VAE
        let vae_config = Some(join("vae/config.json")).filter(|p| p.exists());
        let mut vae_files: Vec<PathBuf> = Vec::new();
        for name in ["vae/diffusion_pytorch_model.safetensors", "vae/model.safetensors"] {
            let p = join(name);
            if p.exists() { vae_files.push(p); }
        }
        if vae_files.is_empty() { bail!("No VAE safetensors under {}", root.display()); }

        // Scheduler
        let scheduler_json = [join("scheduler/scheduler_config.json"), join("scheduler_config.json")]
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("No scheduler_config.json under {}", root.display()))?;

        let device = pick_device_cuda0_or_cpu();
        let dtype = prefer_dtype;

        // Load scheduler config
        let scheduler_cfg: SchedulerConfig = std::fs::read_to_string(&scheduler_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let scheduler = Some(FlowMatchEulerDiscreteScheduler::from_config(&scheduler_cfg));

        // Parse configs
        let tokenizer = Tokenizer::from_file(&tokenizer_json).ok();
        let text_encoder_config_json: Option<serde_json::Value> = text_encoder_config
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let transformer_config_json: Option<serde_json::Value> = transformer_config
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let vae_config_json: Option<serde_json::Value> = vae_config
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

        // Class tags
        let text_encoder_class = text_encoder_config_json
            .as_ref()
            .and_then(|v: &serde_json::Value| v.get("_class_name"))
            .and_then(|v| v.as_str())
            .map(|s| match s { "CLIPTextModel" | "CLIPTextModelWithProjection" => TextEncoderClass::CLIPTextModel, "T5EncoderModel" => TextEncoderClass::T5EncoderModel, other => TextEncoderClass::Unknown(other.to_string()) });
        let transformer_class = transformer_config_json
            .as_ref()
            .and_then(|v: &serde_json::Value| v.get("_class_name"))
            .and_then(|v| v.as_str())
            .map(|s| match s { "UNet2DConditionModel" => TransformerClass::UNet2DConditionModel, "FluxTransformer2DModel" | "Transformer2DModel" | "DiT" => TransformerClass::DiT, "QwenImageTransformer2DModel" => TransformerClass::QwenImageTransformer2DModel, other => TransformerClass::Unknown(other.to_string()) });

        // VAE strictly
    let vae_cfg: Option<AEKLConfig> = vae_config_json.as_ref().and_then(|v| serde_json::from_value::<AEKLConfig>(v.clone()).ok());
        let vae = if let Some(cfg) = &vae_cfg {
            if let Ok(v) = build_vae_from_files(&vae_files, dtype, &device, cfg) {
                Some(Box::new(v) as Box<dyn VaeLike>)
            } else if let Ok(v) = build_vae_from_files_with_5d_squeeze(&vae_files, dtype, &device, cfg) {
                log::warn!("[qwen-image-edit] VAE 5D->4D squeeze applied (strict).");
                Some(Box::new(v) as Box<dyn VaeLike>)
            } else if let Ok(vq) = build_qwen_vae_from_files_strict(&vae_files, dtype, &device, cfg.scaling_factor.unwrap_or(0.18215)) {
                log::warn!("[qwen-image-edit] VAE strict Qwen adapter used.");
                Some(Box::new(vq) as Box<dyn VaeLike>)
            } else { None }
        } else { None };
        if vae.is_none() { bail!("VAE load failed strictly from local snapshot"); }

        // Transformer strictly
        let mut transformer_model: Option<QwenImageTransformer2DModel> = None;
        if let Some(tr_json) = &transformer_config_json {
            if let Ok(cfg) = serde_json::from_value::<QwenImageTransformerConfig>(tr_json.clone()) {
                let vb_root = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&transformer_files, dtype, &device) }?;
                let roots = ["", "transformer", "model", "diffusion_model", "module", "net", "transformer_model"];
                for r in roots.iter() {
                    let vb_try = if r.is_empty() { vb_root.clone() } else { vb_root.pp(*r) };
                    if let Ok(m) = QwenImageTransformer2DModel::new(&cfg, vb_try) { transformer_model = Some(m); break; }
                }
            }
        }
        let transformer_flux = if let Some(tr_json) = &transformer_config_json {
            if let Ok(cfg_flux) = serde_json::from_value::<FluxTransformerConfig>(tr_json.clone()) {
                let looks_flux = if let Some(first) = transformer_files.get(0) {
                    if let Ok(bytes) = std::fs::read(first) {
                        match SafeTensors::deserialize(&bytes) {
                            Ok(st) => st
                                .names()
                                .iter()
                                .any(|k| k.starts_with("transformer_blocks.")),
                            Err(_) => false,
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if looks_flux {
                    FluxTransformer2DModel::new(&cfg_flux, unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&transformer_files, dtype, &device) }?).ok()
                } else { None }
            } else { None }
        } else { None };
        if transformer_model.is_none() && transformer_flux.is_none() { bail!("Transformer strict build failed from local snapshot"); }

        let has_text_encoder = text_encoder_config_json.is_some() && !text_encoder_files.is_empty();

        Ok(Self {
            device,
            dtype,
            paths: QwenImageEditPaths {
                repo_id: root.display().to_string(),
                model_index_json,
                tokenizer_json,
                text_encoder_files,
                transformer_files,
                vae_files,
                scheduler_json,
                transformer_gguf: None,
                text_encoder_gguf: None,
                mmproj_gguf: None,
                processor_preprocessor_config,
                processor_tokenizer_config,
                processor_tokenizer_json,
                tokenizer_config,
                text_encoder_config,
                transformer_config,
                vae_config,
            },
            model_index,
            scheduler,
            tokenizer,
            text_encoder_config_json,
            transformer_config_json,
            vae_config_json,
            text_encoder_class,
            transformer_class,
            has_text_encoder,
            vae,
            transformer_model,
            transformer_flux,
        })
    }
    // Lightweight helper: scan a few safetensors files and log whether expected keys exist, with shapes.
    fn diag_scan_weights(files: &[std::path::PathBuf], label: &str, expected_keys: &[&str]) {
        for (i, f) in files.iter().take(3).enumerate() {
            match std::fs::read(f) {
                Ok(bytes) => match SafeTensors::deserialize(&bytes) {
                    Ok(st) => {
                        let keys = st.names();
                        log::info!(
                            "[qwen-image-edit] {label} diag: file#{i} {} keys",
                            keys.len()
                        );
                        // For transformer, print a small sample of keys to reveal top-level prefixes
                        if label.contains("transformer") {
                            for (idx, k) in keys.iter().take(30).enumerate() {
                                log::info!("  key[{}]: {}", idx, k);
                            }
                            // Also print keys that hint at embeddings/projections
                            for k in keys.iter().filter(|n| {
                                n.contains("patch") || n.contains("embed") || n.contains("proj")
                            }) {
                                log::info!("  hint: {}", k);
                            }
                            // Try to print shapes for representative projection weights if present
                            let probe =
                                |pat: &str,
                                 st: &safetensors::SafeTensors|
                                 -> Option<(String, Vec<usize>)> {
                                    let name = keys.iter().find(|n| n.contains(pat))?;
                                    if let Ok(t) = st.tensor(*name) {
                                        let shape: Vec<_> = t.shape().into();
                                        return Some(((*name).to_string(), shape));
                                    }
                                    None
                                };
                            for pat in [
                                "img_mlp.net.0.proj.weight",
                                "img_mlp.net.2.weight",
                                "txt_mlp.net.0.proj.weight",
                                "attn.to_q.weight",
                                "attn.to_k.weight",
                                "attn.to_v.weight",
                                "attn.to_out.0.weight",
                                "to_add_out.weight",
                            ] {
                                if let Some((name, shape)) = probe(pat, &st) {
                                    log::info!("  shape {} => {:?}", name, shape);
                                }
                            }
                        }
                        for k in expected_keys {
                            // exact match or prefix variants
                            let found = keys.iter().any(|n| *n == *k)
                                || keys.iter().any(|n| n.ends_with(k))
                                || keys.iter().any(|n| n.contains(k));
                            if found {
                                // print first shape we find
                                if let Some(name) = keys
                                    .iter()
                                    .find(|n| **n == *k)
                                    .or_else(|| keys.iter().find(|n| n.ends_with(*k)))
                                    .or_else(|| keys.iter().find(|n| n.contains(*k)))
                                {
                                    if let Ok(t) = st.tensor(*name) {
                                        let shape: Vec<_> = t.shape().into();
                                        log::info!("  key '{}' -> shape {:?}", name, shape);
                                    } else {
                                        log::info!("  key '{}' present (failed to read)", name);
                                    }
                                }
                            } else {
                                log::warn!("  missing expected key pattern: {}", k);
                            }
                        }
                        // For VAE, list some common prefixes to help spot 5D kernels
                        if label.contains("vae") {
                            let mut count = 0usize;
                            for name in keys
                                .iter()
                                .filter(|n| n.starts_with("encoder") || n.starts_with("decoder"))
                            {
                                if let Ok(t) = st.tensor(*name) {
                                    let shape: Vec<_> = t.shape().into();
                                    log::info!("   {} -> {:?}", name, shape);
                                    count += 1;
                                    if count > 120 {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => log::warn!(
                        "[qwen-image-edit] {label} diag: cannot parse safetensors {}: {e}",
                        f.display()
                    ),
                },
                Err(e) => log::warn!(
                    "[qwen-image-edit] {label} diag: cannot read {}: {e}",
                    f.display()
                ),
            }
        }
    }

    pub fn load_from_hf(repo_id: &str, prefer_dtype: DType) -> Result<Self> {
        Self::load_from_hf_inner(repo_id, prefer_dtype, (None, None, None))
    }

    pub fn load_from_hf_with_overrides(
        repo_id: &str,
        prefer_dtype: DType,
        transformer_gguf: Option<PathBuf>,
        text_encoder_gguf: Option<PathBuf>,
        mmproj_gguf: Option<PathBuf>,
    ) -> Result<Self> {
        Self::load_from_hf_inner(
            repo_id,
            prefer_dtype,
            (transformer_gguf, text_encoder_gguf, mmproj_gguf),
        )
    }

    pub fn info(&self) {
        log::info!(
            "[qwen-image-edit] repo={} dtype={:?} device={:?}",
            self.paths.repo_id,
            self.dtype,
            self.device
        );
        log::info!(
            " transformer: {} shards",
            self.paths.transformer_files.len()
        );
        log::info!(" vae: {} shards", self.paths.vae_files.len());
        log::info!(
            " text-encoder: {} shards",
            self.paths.text_encoder_files.len()
        );
        let opt = |label: &str, p: &Option<std::path::PathBuf>| {
            if let Some(p) = p {
                log::info!("  {}: {}", label, p.display());
            }
        };
        // GGUF optional components
        opt("gguf.transformer", &self.paths.transformer_gguf);
        opt("gguf.text_encoder", &self.paths.text_encoder_gguf);
        opt("gguf.mmproj", &self.paths.mmproj_gguf);
        opt(
            "processor.preprocessor_config",
            &self.paths.processor_preprocessor_config,
        );
        opt(
            "processor.tokenizer_config",
            &self.paths.processor_tokenizer_config,
        );
        opt(
            "processor.tokenizer.json",
            &self.paths.processor_tokenizer_json,
        );
        opt("tokenizer.tokenizer_config", &self.paths.tokenizer_config);
        opt("text_encoder.config", &self.paths.text_encoder_config);
        opt("transformer.config", &self.paths.transformer_config);
        opt("vae.config", &self.paths.vae_config);
        if let Some(mi) = &self.model_index {
            log::info!(
                " model_index: class={:?} version={:?} type={:?}",
                mi._class_name,
                mi._diffusers_version,
                mi.model_type
            );
            let pr = |label: &str, c: &Option<ComponentSpec>| {
                if let Some(c) = c {
                    match c {
                        ComponentSpec::Ref(r) => {
                            log::info!("  {}: ref {:?}", label, r._name_or_path)
                        }
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
            log::info!(
                " scheduler: FlowMatchEulerDiscrete num_train_timesteps={} spacing={} pred={}",
                s.num_train_timesteps,
                s.timestep_spacing,
                s.prediction_type
            );
        }
        if let Some(te) = &self.text_encoder_config_json {
            if let Some(class) = te.get("_class_name").and_then(|v| v.as_str()) {
                log::info!(" text_encoder._class_name={}", class);
            }
        }
        if let Some(tr) = &self.transformer_config_json {
            if let Some(class) = tr.get("_class_name").and_then(|v| v.as_str()) {
                log::info!(" transformer._class_name={}", class);
            }
        }
        if let Some(va) = &self.vae_config_json {
            if let Some(class) = va.get("_class_name").and_then(|v| v.as_str()) {
                log::info!(" vae._class_name={}", class);
            }
        }
        if let Some(k) = &self.text_encoder_class {
            log::info!(" text_encoder.kind={:?}", k);
        }
        if let Some(k) = &self.transformer_class {
            log::info!(" transformer.kind={:?}", k);
        }
    }

    pub fn run_edit(
        &self,
        image_path: &std::path::Path,
        opts: &crate::ai::qwen_image_edit::EditOptions,
    ) -> Result<Vec<u8>> {
        self.denoise_loop(image_path, opts)
    }
}
