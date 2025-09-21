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
use crate::ui::status::{QWEN_EDIT_STATUS, GlobalStatusIndicator, StatusState};
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
    fn load_from_hf_inner(
        repo_id: &str,
        prefer_dtype: DType,
        gguf_overrides: (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>),
    ) -> Result<Self> {
        let (transformer_gguf, text_encoder_gguf, mut mmproj_gguf) = gguf_overrides;
        log::info!(
            "[qwen-image-edit] load_from_hf: repo={} prefer_dtype={:?} overrides: transformer_gguf={:?} text_encoder_gguf={:?} mmproj_gguf={:?}",
            repo_id, prefer_dtype, transformer_gguf.as_ref().map(|p| p.display().to_string()), text_encoder_gguf.as_ref().map(|p| p.display().to_string()), mmproj_gguf.as_ref().map(|p| p.display().to_string())
        );
        // Discover and download files described in the prompt (model_index.json, safetensors for transformer/vae, tokenizer.json, scheduler config)
        let repo = hf_model(repo_id)?;
        let model_index_json = hf_get_file(&repo, "model_index.json")?;
        // Parse model_index.json
        let model_index: Option<ModelIndex> = std::fs::read_to_string(&model_index_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        // Processor/tokenizer related configs (optional but useful)
        let processor_preprocessor_config = repo.get("processor/preprocessor_config.json").ok();
        let processor_tokenizer_config = repo.get("processor/tokenizer_config.json").ok();
        let processor_tokenizer_json = repo.get("processor/tokenizer.json").ok();
        let tokenizer_config = repo.get("tokenizer/tokenizer_config.json").ok();

        // Tokenizer: prefer tokenizer.json (tokenizers format) under tokenizer/ or processor/, fall back to root or tokenizer_config.json
        let tokenizer_json = {
            let candidates = [
                "tokenizer/tokenizer.json",
                "processor/tokenizer.json",
                "tokenizer.json",
                "processor/tokenizer_config.json",
                "tokenizer/tokenizer_config.json",
            ];
            let mut found = None;
            for c in candidates.iter() {
                if let Ok(p) = repo.get(c) {
                    found = Some(p);
                    break;
                }
            }
            found.ok_or_else(|| anyhow::anyhow!("No tokenizer json found in repo {repo_id}"))?
        };
        // Text encoder weights (could be multiple shards)
        let mut text_encoder_files: Vec<std::path::PathBuf> = Vec::new();
        let text_encoder_config = repo.get("text_encoder/config.json").ok();
        // Prefer sharded index if present
        if let Ok(idx_path) = repo.get("text_encoder/model.safetensors.index.json") {
            if let Ok(s) = std::fs::read_to_string(&idx_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    let mut uniq: HashSet<String> = HashSet::new();
                    if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                        for fname in map.values().filter_map(|v| v.as_str()) {
                            uniq.insert(fname.to_string());
                        }
                    }
                    let mut list: Vec<String> = uniq.into_iter().collect();
                    list.sort();
                    for f in list.into_iter() {
                        let full = format!("text_encoder/{}", f);
                        if let Ok(p) = repo.get(&full) {
                            text_encoder_files.push(p);
                        }
                    }
                }
            }
        }
        if text_encoder_files.is_empty() {
            for name in [
                "text_encoder/model.safetensors",
                "text_encoder/diffusion_pytorch_model.safetensors",
            ]
            .iter()
            {
                if let Ok(p) = repo.get(name) {
                    text_encoder_files.push(p);
                }
            }
        }
        // Transformer/UNet weights (can be sharded)
    let mut transformer_files: Vec<std::path::PathBuf> = Vec::new();
        let transformer_config = repo.get("transformer/config.json").ok();
        if let Ok(idx_path) = repo.get("transformer/diffusion_pytorch_model.safetensors.index.json")
        {
            if let Ok(s) = std::fs::read_to_string(&idx_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    let mut uniq: HashSet<String> = HashSet::new();
                    if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                        for fname in map.values().filter_map(|v| v.as_str()) {
                            uniq.insert(fname.to_string());
                        }
                    }
                    let mut list: Vec<String> = uniq.into_iter().collect();
                    list.sort();
                    for f in list.into_iter() {
                        let full = format!("transformer/{}", f);
                        if let Ok(p) = repo.get(&full) {
                            transformer_files.push(p);
                        }
                    }
                }
            }
        }
        if transformer_files.is_empty() {
            for name in [
                "transformer/model.safetensors",
                "transformer/diffusion_pytorch_model.safetensors",
            ]
            .iter()
            {
                if let Ok(p) = repo.get(name) {
                    transformer_files.push(p);
                }
            }
        }
        if transformer_files.is_empty() {
            bail!("No transformer safetensors found in repo {repo_id}");
        }
        // VAE
    let mut vae_files = Vec::new();
    let mut vae_config = repo.get("vae/config.json").ok();
        for name in [
            "vae/diffusion_pytorch_model.safetensors",
            "vae/model.safetensors",
        ]
        .iter()
        {
            if let Ok(p) = repo.get(name) {
                vae_files.push(p);
            }
        }
        if vae_files.is_empty() {
            bail!("No VAE safetensors found in repo {repo_id}");
        }
        // Scheduler config
        let scheduler_json = repo
            .get("scheduler/scheduler_config.json")
            .or_else(|_| repo.get("scheduler_config.json"))?;

    let device = pick_device_cuda0_or_cpu();
        let dtype = prefer_dtype;

        // Load scheduler config; fall back to defaults if parsing fails so we always have a scheduler
        let scheduler_cfg: SchedulerConfig = std::fs::read_to_string(&scheduler_json)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let scheduler = Some(FlowMatchEulerDiscreteScheduler::from_config(&scheduler_cfg));

        // (GGUF attempt happens after configs are parsed; see below.)

        // Parse component configs
        let text_encoder_config_json = if let Some(p) = &text_encoder_config {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        } else {
            None
        };
        let transformer_config_json = if let Some(p) = &transformer_config {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        } else {
            None
        };
        let mut vae_config_json = if let Some(p) = &vae_config {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        } else {
            None
        };

        // Load tokenizer if file is tokenizer.json, otherwise try to map config->file
        let tokenizer = if tokenizer_json
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("tokenizer.json"))
            .unwrap_or(false)
        {
            Tokenizer::from_file(&tokenizer_json).ok()
        } else {
            None
        };

        // Detect component classes from configs
        let text_encoder_class = text_encoder_config_json
            .as_ref()
            .and_then(|v: &serde_json::Value| v.get("_class_name"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "CLIPTextModel" | "CLIPTextModelWithProjection" => TextEncoderClass::CLIPTextModel,
                "T5EncoderModel" => TextEncoderClass::T5EncoderModel,
                other => TextEncoderClass::Unknown(other.to_string()),
            });
        let transformer_class = transformer_config_json
            .as_ref()
            .and_then(|v: &serde_json::Value| v.get("_class_name"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "UNet2DConditionModel" => TransformerClass::UNet2DConditionModel,
                "FluxTransformer2DModel" | "Transformer2DModel" | "DiT" => TransformerClass::DiT,
                "QwenImageTransformer2DModel" => TransformerClass::QwenImageTransformer2DModel,
                other => TransformerClass::Unknown(other.to_string()),
            });

        // If a GGUF transformer path is provided, auto-discover companion assets from the same folder tree.
        // Typical layout (QuantStack/Qwen-Image-Edit-GGUF):
        //   <root>/Qwen_Image_Edit-*.gguf
        //   <root>/vae/{model.safetensors,config.json}
        //   <root>/mmproj/{*.gguf|*.safetensors}
        if let Some(gguf_path) = &transformer_gguf {
            let mut candidate_roots: Vec<PathBuf> = Vec::new();
            if let Some(d) = gguf_path.parent() { candidate_roots.push(d.to_path_buf()); }
            if let Some(d2) = gguf_path.parent().and_then(|d| d.parent()) { candidate_roots.push(d2.to_path_buf()); }
            candidate_roots.sort(); candidate_roots.dedup();

            // Prefer VAE found next to the GGUF
            let mut picked_vae: Option<PathBuf> = None;
            let mut picked_vae_cfg: Option<PathBuf> = None;
            for root in &candidate_roots {
                let vae_dir = root.join("vae");
                if vae_dir.exists() {
                    // Only accept the exact filename supplied by the GGUF repo
                    let f = vae_dir.join("Qwen_Image-VAE.safetensors");
                    if f.exists() { picked_vae = Some(f); }
                    let cfg_path = vae_dir.join("config.json");
                    if cfg_path.exists() { picked_vae_cfg = Some(cfg_path); }
                    if picked_vae.is_some() { break; }
                }
            }
            if let Some(vf) = picked_vae {
                log::info!("[qwen-image-edit] Using VAE from GGUF folder: {}", vf.display());
                vae_files.clear();
                vae_files.push(vf);
                if let Some(cfgp) = picked_vae_cfg {
                    vae_config = Some(cfgp.clone());
                    if let Ok(s) = std::fs::read_to_string(&cfgp) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) { vae_config_json = Some(v); }
                    }
                }
            }

            // Prefer mmproj discovered next to GGUF (exact file name)
            if mmproj_gguf.is_none() {
                let mut found_mmproj: Option<PathBuf> = None;
                for root in &candidate_roots {
                    let mm_dir = root.join("mmproj");
                    if mm_dir.exists() {
                        let p = mm_dir.join("Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf");
                        if p.exists() { found_mmproj = Some(p); }
                    }
                    if found_mmproj.is_some() { break; }
                }
                if let Some(mp) = found_mmproj {
                    log::info!("[qwen-image-edit] Found mmproj near GGUF: {}", mp.display());
                    mmproj_gguf = Some(mp);
                }
            }

            // HF fallback: if VAE or its config or mmproj were not found locally next to GGUF, try fetching from the GGUF repo
            let need_vae = vae_files.is_empty();
            let need_vae_cfg = vae_config_json.is_none();
            let need_mmproj = mmproj_gguf.is_none();
            if need_vae || need_vae_cfg || need_mmproj {
                // Known repo carrying GGUF + companions
                let gg_repo = match hf_model("QuantStack/Qwen-Image-Edit-GGUF") {
                    Ok(r) => r,
                    Err(e) => { log::warn!("[qwen-image-edit] GGUF HF repo unavailable: {}", e); hf_model(repo_id)? }
                };
                // Attempt VAE fetch
                if need_vae {
                    let name = "vae/Qwen_Image-VAE.safetensors";
                    if let Ok(p) = gg_repo.get(name) {
                        log::info!("[qwen-image-edit] Downloaded VAE from GGUF repo: {}", p.display());
                        vae_files.clear();
                        vae_files.push(p);
                    }
                }
                if need_vae_cfg {
                    if let Ok(cfgp) = gg_repo.get("vae/config.json") {
                        if let Ok(s) = std::fs::read_to_string(&cfgp) {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) { vae_config_json = Some(v); vae_config = Some(cfgp); }
                        }
                    }
                }
                // Attempt mmproj fetch (prefer BF16 GGUF name the user cited)
                if need_mmproj {
                    let name = "mmproj/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf";
                    if let Ok(p) = gg_repo.get(name) {
                        log::info!("[qwen-image-edit] Downloaded mmproj from GGUF repo: {}", p.display());
                        mmproj_gguf = Some(p);
                    }
                }
            }
        }

        // Do not instantiate a local Qwen2.5-VL module; only note availability of external weights/config.
        let has_text_encoder = text_encoder_config_json.is_some() && !text_encoder_files.is_empty();

        // Try to build VAE
        let vae_cfg = if let Some(v) = &vae_config_json {
            serde_json::from_value::<AEKLConfig>(v.clone()).ok()
        } else {
            None
        };
        let vae = if let Some(cfg) = &vae_cfg {
            build_vae_from_files(&vae_files, dtype, &device, cfg)
                .ok()
                .map(|v| Box::new(v) as Box<dyn VaeLike>)
        } else {
            None
        };

        // Build transformer strictly from weights; prefer Flux-style when class matches and keys indicate transformer_blocks.*.
        // Try GGUF transformer first if requested; fall back to safetensors
        let mut transformer_model: Option<QwenImageTransformer2DModel> = None;
        if transformer_model.is_none() {
            if let Some(p) = &transformer_gguf {
                if let Some(tr_json) = &transformer_config_json {
                    if let Ok(cfg) = serde_json::from_value::<QwenImageTransformerConfig>(tr_json.clone()) {
                        log::info!(
                            "[qwen-image-edit] Attempting GGUF transformer: path={} (in_ch={}, out_ch={}, heads={}, head_dim={}, layers={}, joint_dim={}) on {:?}",
                            p.display(), cfg.in_channels, cfg.out_channels, cfg.num_attention_heads, cfg.attention_head_dim, cfg.num_layers, cfg.joint_attention_dim, device
                        );
                        match QwenImageTransformer2DModel::new_from_gguf(&cfg, p.as_path(), &device) {
                            Ok(m) => {
                                transformer_model = Some(m);
                                log::info!("[qwen-image-edit] GGUF transformer loaded successfully from {}", p.display());
                            }
                            Err(e) => {
                                // Try alternative GGUF quant files in the same directory.
                                // Preference order (avoid K-quant first due to Candle 0.9 parser limits):
                                //   Q5_0 -> Q4_0 -> Q8_0 -> Q5_K_M -> Q6_K
                                // Also: never retry the same file we already attempted.
                                let mut tried_alt = false;
                                if let Some(dir) = p.parent() {
                                    if let Ok(read) = std::fs::read_dir(dir) {
                                        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
                                        let mut q5_0: Option<std::path::PathBuf> = None;
                                        let mut q4_0: Option<std::path::PathBuf> = None;
                                        let mut q8_0: Option<std::path::PathBuf> = None;
                                        let mut q5_km: Option<std::path::PathBuf> = None;
                                        let mut q6_k: Option<std::path::PathBuf> = None;
                                        for ent in read.flatten() {
                                            let path = ent.path();
                                            if !path.is_file() { continue; }
                                            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                                                if !ext.eq_ignore_ascii_case("gguf") { continue; }
                                            } else { continue; }
                                            if path == *p { continue; }
                                            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_uppercase();
                                            if name.contains("Q5_0") { q5_0 = Some(path.clone()); }
                                            if name.contains("Q4_0") { q4_0 = Some(path.clone()); }
                                            if name.contains("Q8_0") { q8_0 = Some(path.clone()); }
                                            if name.contains("Q5_K_M") { q5_km = Some(path.clone()); }
                                            if name.contains("Q6_K") { q6_k = Some(path.clone()); }
                                        }
                                        // Push in preference order
                                        for opt in [q5_0, q4_0, q8_0, q5_km, q6_k] {
                                            if let Some(pth) = opt { candidates.push(pth); }
                                        }
                                        for altp in candidates {
                                            log::warn!(
                                                "[qwen-image-edit] GGUF build failed for {}. Trying sibling quant: {}",
                                                p.display(),
                                                altp.display()
                                            );
                                            match QwenImageTransformer2DModel::new_from_gguf(&cfg, altp.as_path(), &device) {
                                                Ok(m2) => {
                                                    transformer_model = Some(m2);
                                                    log::info!("[qwen-image-edit] GGUF transformer loaded successfully from sibling quant {}", altp.display());
                                                    tried_alt = true;
                                                    break;
                                                }
                                                Err(err_alt) => {
                                                    log::warn!("[qwen-image-edit] Sibling quant {} also failed: {}", altp.display(), err_alt);
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                                if !tried_alt {
                                    // Try fetching alternative quants from the GGUF repo.
                                    // Preference: Q5_0 -> Q4_0 -> Q8_0 -> Q5_K_M -> Q6_K
                                    let mut fetched_alt: Option<std::path::PathBuf> = None;
                                    if let Ok(gg_repo) = hf_model("QuantStack/Qwen-Image-Edit-GGUF") {
                                        let preferred = [
                                            "Qwen_Image_Edit-Q5_0.gguf",
                                            "Qwen_Image_Edit-Q4_0.gguf",
                                            "Qwen_Image_Edit-Q8_0.gguf",
                                            "Qwen_Image_Edit-Q5_K_M.gguf",
                                            "Qwen_Image_Edit-Q6_K.gguf",
                                        ];
                                        for fname in preferred {
                                            // Avoid fetching the same variant as the one that already failed
                                            if let Some(orig) = p.file_name().and_then(|s| s.to_str()) {
                                                if orig.eq_ignore_ascii_case(fname) { continue; }
                                            }
                                            if let Ok(pp) = gg_repo.get(fname) {
                                                fetched_alt = Some(pp);
                                                log::warn!(
                                                    "[qwen-image-edit] Downloaded alternative GGUF quant {} from QuantStack/Qwen-Image-Edit-GGUF",
                                                    fname
                                                );
                                                break;
                                            }
                                        }
                                    }
                                    if let Some(altp) = fetched_alt {
                                        match QwenImageTransformer2DModel::new_from_gguf(&cfg, altp.as_path(), &device) {
                                            Ok(m3) => {
                                                transformer_model = Some(m3);
                                                log::info!("[qwen-image-edit] GGUF transformer loaded successfully from fetched alternative {}", altp.display());
                                                tried_alt = true;
                                            }
                                            Err(err_alt) => {
                                                log::warn!("[qwen-image-edit] Fetched alternative {} also failed: {}", altp.display(), err_alt);
                                            }
                                        }
                                    }
                                }
                                if !tried_alt {
                                // Enrich the error for unknown dtype cases (common with K-quant, e.g., Q6_K)
                                let err_s = format!("{}", e);
                                let file_name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                                let mut extra = String::new();
                                // Try to extract a numeric dtype code from the error message
                                if let Ok(re) = Regex::new(r"unknown dtype for tensor\s+(\d+)") {
                                    if let Some(cap) = re.captures(&err_s) {
                                        if let Some(m) = cap.get(1) {
                                            let code_str = m.as_str();
                                            // Map a few well-known ggml/gguf codes to user-friendly names when possible.
                                            // Note: exact mappings can differ by build; provide best-effort hints.
                                            let hint = match code_str {
                                                "30" => "This looks like a K-quant GGUF (often Q6_K).",
                                                "28" => "This may be a K-quant GGUF (often Q5_K_M).",
                                                "26" => "This may be a K-quant GGUF (often Q4_K_M).",
                                                _ => "This GGUF uses a quantization type not recognized by the current GGUF reader.",
                                            };
                                            extra.push_str(&format!(
                                                "\n  • DType code: {} → {}",
                                                code_str, hint
                                            ));
                                        }
                                    }
                                }
                                // Add filename-derived quant hint
                                if file_name.to_ascii_uppercase().contains("Q6_K") {
                                    extra.push_str("\n  • File hint: name contains Q6_K (K-quant). Consider using Q8_0 or Q5_K_M if this build lacks Q6_K.");
                                } else if file_name.to_ascii_uppercase().contains("Q5_K") {
                                    extra.push_str("\n  • File hint: name contains Q5_K (K-quant). If unsupported, try Q8_0 or a different K variant.");
                                }
                                // Include file size if available
                                if let Ok(meta) = std::fs::metadata(p) {
                                    extra.push_str(&format!("\n  • File size: {} bytes", meta.len()));
                                }
                                // Device/dtype context
                                extra.push_str(&format!(
                                    "\n  • Target device: {:?}, preferred dtype: {:?}",
                                    device, dtype
                                ));
                                // Suggestions
                                extra.push_str(
                                    "\n  • Suggestions: try a non-K GGUF quant (Q5_0 or Q4_0 preferred) or update Candle crates.\n    Note: Candle 0.9 fails parsing GGUF if the file contains any unknown quant dtypes anywhere, even if unused by us.\n    Falling back to safetensors next."
                                );
                                log::error!(
                                    "[qwen-image-edit] GGUF transformer load failed for {}:\n  {}{}",
                                    p.display(),
                                    err_s,
                                    extra
                                );
                                }
                            }
                        }
                    }
                }
            }
        }
        if transformer_model.is_none() {
            // If snapshot clearly looks like Flux (transformer_blocks.* present), skip Qwen transformer build entirely.
            let mut looks_flux = false;
            if let Some(first) = transformer_files.get(0) {
                if let Ok(bytes) = std::fs::read(first) {
                    if let Ok(st) = safetensors::SafeTensors::deserialize(&bytes) {
                        looks_flux = st
                            .names()
                            .iter()
                            .any(|k| k.starts_with("transformer_blocks."));
                    }
                }
            }
            transformer_model = if looks_flux {
                log::info!("[qwen-image-edit] Skipping Qwen transformer build: Flux-style keys detected in snapshot");
                None
            } else if let Some(tr_json) = &transformer_config_json {
            match serde_json::from_value::<QwenImageTransformerConfig>(tr_json.clone()) {
                Ok(cfg) => {
                    // Try safetensors first
                    let vb_root = match unsafe {
                        candle_nn::VarBuilder::from_mmaped_safetensors(
                            &transformer_files,
                            dtype,
                            &device,
                        )
                    } {
                        Ok(vb) => vb,
                        Err(e) => {
                            log::error!("[qwen-image-edit] Failed to map transformer weights: {e}");
                            Self::diag_scan_weights(
                                &transformer_files,
                                "transformer",
                                &[
                                    "proj_in.weight",
                                    "proj_out.weight",
                                    "layers.0.ln1.weight",
                                    "layers.0.attn.q_proj.weight",
                                    "layers.0.attn.k_proj.weight",
                                    "layers.0.attn.v_proj.weight",
                                    "layers.0.attn.o_proj.weight",
                                ],
                            );
                            anyhow::bail!("Failed to map transformer safetensors; see logs for missing keys and consider using GGUF");
                        }
                    };
                    // Try common root prefixes used by HF repos for the previous QwenImageTransformer2DModel implementation
                    let roots = [
                        "",
                        "transformer",
                        "model",
                        "diffusion_model",
                        "module",
                        "net",
                        "transformer_model",
                    ];
                    let mut built = None;
                    for r in roots.iter() {
                        log::info!("[qwen-image-edit] Trying transformer safetensors with prefix '{}'", r);
                        let vb_try = if r.is_empty() {
                            vb_root.clone()
                        } else {
                            vb_root.pp(*r)
                        };
                        match QwenImageTransformer2DModel::new(&cfg, vb_try) {
                            Ok(m) => {
                                built = Some(m);
                                log::info!("[qwen-image-edit] Transformer built from safetensors using prefix '{}'", r);
                                break;
                            }
                            Err(e) => {
                                log::debug!(
                                    "[qwen-image-edit] transformer build failed for root='{}': {}",
                                    r,
                                    e
                                );
                                continue;
                            }
                        }
                    }
                    if built.is_none() {
                        log::error!(
                            "[qwen-image-edit] Failed to build transformer from weights across known prefixes."
                        );
                        Self::diag_scan_weights(
                            &transformer_files,
                            "transformer",
                            &[
                                "proj_in.weight",
                                "proj_out.weight",
                                "layers.0.ln1.weight",
                                "layers.0.attn.q_proj.weight",
                                "layers.0.attn.k_proj.weight",
                                "layers.0.attn.v_proj.weight",
                                "layers.0.attn.o_proj.weight",
                            ],
                        );
                    }
                    built
                }
                Err(e) => {
                    log::error!("[qwen-image-edit] transformer config parse failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        }

        // Flux-style transformer path using transformer_blocks.* keys when class is QwenImageTransformer2DModel
        let transformer_flux = if let Some(tr_json) = &transformer_config_json {
            match serde_json::from_value::<FluxTransformerConfig>(tr_json.clone()) {
                Ok(cfg_flux) => {
                    // We determine Flux suitability by sampling keys for 'transformer_blocks.' prefix in the first shard
                    let mut looks_flux = false;
                    if let Some(first) = transformer_files.get(0) {
                        if let Ok(bytes) = std::fs::read(first) {
                            if let Ok(st) = safetensors::SafeTensors::deserialize(&bytes) {
                                looks_flux = st
                                    .names()
                                    .iter()
                                    .any(|k| k.starts_with("transformer_blocks."));
                            }
                        }
                    }
                    if looks_flux {
                        // Try common prefixes for the Flux layout similar to the Qwen builder path
                        let roots = [
                            "",
                            "transformer",
                            "model",
                            "diffusion_model",
                            "module",
                            "net",
                            "transformer_model",
                        ];
                        let vb_root = unsafe {
                            candle_nn::VarBuilder::from_mmaped_safetensors(
                                &transformer_files,
                                dtype,
                                &device,
                            )?
                        };
                        let mut built: Option<FluxTransformer2DModel> = None;
                        for r in roots.iter() {
                            let vb_try = if r.is_empty() { vb_root.clone() } else { vb_root.pp(*r) };
                            log::info!("[qwen-image-edit] Trying Flux transformer with prefix '{}'", r);
                            match FluxTransformer2DModel::new(&cfg_flux, vb_try) {
                                Ok(m) => { built = Some(m); log::info!("[qwen-image-edit] Flux transformer built using prefix '{}'", r); break; },
                                Err(e) => { log::debug!("[qwen-image-edit] Flux build failed for root='{}': {}", r, e); }
                            }
                        }
                        if built.is_none() {
                            log::error!("[qwen-image-edit] Flux transformer build failed across known prefixes. Snapshot may lack 'img_in' or expected keys.");
                        }
                        built
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // If no transformer was constructed (neither GGUF nor safetensors/Flux), surface an error early.
        if transformer_model.is_none() && transformer_flux.is_none() {
            anyhow::bail!("No compatible transformer found (GGUF or safetensors). Check logs above for key name mismatches.");
        }

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
                transformer_gguf,
                text_encoder_gguf,
                mmproj_gguf,
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

    pub fn run_edit(
        &self,
        image_path: &std::path::Path,
        opts: &crate::ai::qwen_image_edit::EditOptions,
    ) -> Result<Vec<u8>> {
        self.denoise_loop(image_path, opts)
    }
}
