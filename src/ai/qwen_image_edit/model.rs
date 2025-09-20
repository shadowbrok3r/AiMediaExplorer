use anyhow::{Result, bail};
use candle_core::IndexOp;
use candle_core::Module;
use candle_core::{DType, Device};
use candle_nn::{Embedding, VarBuilder};

use super::config::{ComponentSpec, ModelIndex, SchedulerConfig};
use super::scheduler::FlowMatchEulerDiscreteScheduler;
use crate::ai::hf::{hf_get_file, hf_model, pick_device_cuda0_or_cpu, with_mmap_varbuilder_multi};
// use image::{GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use crate::ai::flux_transformer::{FluxTransformer2DModel, FluxTransformerConfig};
use crate::ai::qwen_image_transformer::{QwenImageTransformer2DModel, QwenImageTransformerConfig};
use crate::ai::vae::{
    VaeConfig as AEKLConfig, VaeLike, build_qwen_vae_simplified_from_files, build_vae_from_files,
    build_vae_from_files_with_5d_squeeze,
};
use safetensors::SafeTensors;
use std::path::PathBuf;
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
        let (transformer_gguf, text_encoder_gguf, mmproj_gguf) = gguf_overrides;
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
        let vae_config = repo.get("vae/config.json").ok();
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
        let mut dtype = prefer_dtype;
        // Prefer F16 on CUDA; BF16 is unsupported on older GPUs (e.g., Turing/2070S) and can cause missing CUDA symbols.
        if matches!(device, candle_core::Device::Cuda(_)) {
            if dtype != candle_core::DType::F16 {
                log::warn!(
                    "[qwen-image-edit] Using F16 on CUDA for compatibility (was {:?}).",
                    dtype
                );
            }
            dtype = candle_core::DType::F16;
        }

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
        let vae_config_json = if let Some(p) = &vae_config {
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
                                log::warn!("[qwen-image-edit] GGUF transformer load failed: {}. Will try safetensors next.", e);
                            }
                        }
                    }
                }
            }
        }
        if transformer_model.is_none() {
            transformer_model = if let Some(tr_json) = &transformer_config_json {
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
                        match FluxTransformer2DModel::new(&cfg_flux, unsafe {
                            candle_nn::VarBuilder::from_mmaped_safetensors(
                                &transformer_files,
                                dtype,
                                &device,
                            )?
                        }) {
                            Ok(m) => Some(m),
                            Err(e) => {
                                log::error!(
                                    "[qwen-image-edit] Flux transformer build failed: {}",
                                    e
                                );
                                None
                            }
                        }
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
                    match build_vae_from_files(
                        &self.paths.vae_files,
                        self.dtype,
                        &self.device,
                        &cfg,
                    ) {
                        Ok(v) => {
                            vae_local = Box::new(v) as Box<dyn VaeLike>;
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
                            // Attempt 5D->4D squeeze adaptation using real weights
                            match build_vae_from_files_with_5d_squeeze(
                                &self.paths.vae_files,
                                self.dtype,
                                &self.device,
                                &cfg,
                            ) {
                                Ok(v) => {
                                    log::warn!("[qwen-image-edit] VAE 5D->4D squeeze applied.");
                                    vae_local = Box::new(v) as Box<dyn VaeLike>;
                                    &*vae_local
                                }
                                Err(e2) => {
                                    log::warn!(
                                        "[qwen-image-edit] VAE squeeze failed: {e2}; trying Qwen-native simplified VAE mapping."
                                    );
                                    // Try Qwen-native simplified VAE as a strict mapping using native names; scaling factor from config or default
                                    let scale = cfg.scaling_factor.unwrap_or(0.18215);
                                    match build_qwen_vae_simplified_from_files(
                                        &self.paths.vae_files,
                                        self.dtype,
                                        &self.device,
                                        scale,
                                    ) {
                                        Ok(q) => {
                                            vae_local = Box::new(q) as Box<dyn VaeLike>;
                                            &*vae_local
                                        }
                                        Err(e3) => {
                                            bail!(
                                                "Failed to build any VAE (internal, 5D squeeze, qwen-native): {e3}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    bail!(
                        "VAE not loaded and failed to parse vae/config.json from repo (both internal and compatibility)."
                    );
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
        // Optional: clamp max side to reduce memory (keeps aspect). Default very high.
    // Conservative cap to limit VRAM; consider making this a user option if needed.
    // Lowered default to 1024 for broader GPU compatibility (e.g., 2070S).
    let max_side: u32 = 1024;
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

        // Encode with auto pad, with runtime CPU fallback if CUDA named symbol is missing
        let mut vae_cpu_fallback: Option<Box<dyn crate::ai::vae::VaeLike>> = None;
        let (mut z, oh, ow, _ph, _pw) = match vae.encode_with_auto_pad(&x, !opts.deterministic_vae) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("{}", e);
                if msg.contains("CUDA_ERROR_NOT_FOUND") || msg.contains("named symbol not found") {
                    log::warn!(
                        "[qwen-image-edit] VAE encode failed on CUDA due to named symbol; retrying encode on CPU/F32 with simplified Qwen VAE."
                    );
                    let cpu = candle_core::Device::Cpu;
                    let scale = 0.18215f32;
                    // Build a temporary CPU simplified VAE and use it only for encode/decode
                    match crate::ai::vae::build_qwen_vae_simplified_from_files(
                        &self.paths.vae_files,
                        candle_core::DType::F32,
                        &cpu,
                        scale,
                    ) {
                        Ok(vs) => {
                            let vbox: Box<dyn crate::ai::vae::VaeLike> = Box::new(vs);
                            // Move input to CPU for CPU VAE
                            let mut x_cpu = x.to_device(&cpu)?;
                            if x_cpu.dtype() != candle_core::DType::F32 {
                                x_cpu = x_cpu.to_dtype(candle_core::DType::F32)?;
                            }
                            let out = vbox.encode_with_auto_pad(&x_cpu, !opts.deterministic_vae)?;
                            vae_cpu_fallback = Some(vbox);
                            out
                        }
                        Err(e2) => {
                            // If fallback build also fails, return original error
                            return Err(anyhow::anyhow!(
                                "VAE encode failed on CUDA and CPU fallback build failed: {} | {}",
                                msg, e2
                            ));
                        }
                    }
                } else {
                    return Err(anyhow::Error::msg(msg));
                }
            }
        };
        if z.dtype() != self.dtype {
            z = z.to_dtype(self.dtype)?;
        }
        if z.device().is_cuda() != self.device.is_cuda() {
            z = z.to_device(&self.device)?;
        }

        // Mark encode done in overall progress
        QWEN_EDIT_STATUS.set_progress(1, total_overall);

        // Build embeddings for cond and uncond
        let (text_cond, text_uncond) = if self.tokenizer.is_some() && self.has_text_encoder {
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
                    .and_then(|t| t.get("hidden_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(joint_dim as u64) as usize;
                let e = tc
                    .and_then(|t| t.get("rms_norm_eps"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-6) as f32;
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
            // Try to load embedding + norm; if not found, fall back to zeros.
            let loaded: anyhow::Result<(Embedding, candle_core::Tensor)> = unsafe {
                with_mmap_varbuilder_multi(
                    &self.paths.text_encoder_files,
                    self.dtype,
                    &self.device,
                    |vb: VarBuilder| {
                        let try_build = |vb: VarBuilder, pfx: &str| -> anyhow::Result<(Embedding, candle_core::Tensor)> {
                        let vb = if pfx.is_empty() { vb } else { vb.pp(pfx) };
                        let emb = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
                        let norm_w = vb.pp("norm").get(hidden_size, "weight")?;
                        Ok((emb, norm_w))
                    };
                        let attempts = ["model.text_model", "text_model", ""]; // order of likelihood
                        for p in attempts.iter() {
                            if let Ok(v) = try_build(vb.clone(), p) {
                                return Ok(v);
                            }
                        }
                        Err(anyhow::anyhow!("text_encoder embed/norm not found"))
                    },
                )
            };
            if let Ok((emb_layer, mut norm_w)) = loaded {
                let mut embed_once = |s: &str| -> anyhow::Result<candle_core::Tensor> {
                    let enc = tok.encode(s, true).map_err(anyhow::Error::msg)?;
                    let ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
                    let input_ids = candle_core::Tensor::new(ids, &self.device)?.unsqueeze(0)?; // [1,L]
                    let x = emb_layer.forward(&input_ids)?; // [1,L,D]
                    // RMSNorm inline
                    let x_dtype = x.dtype();
                    let x32 = x.to_dtype(candle_core::DType::F32)?;
                    let variance = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                    let eps_t = candle_core::Tensor::new(eps, &self.device)?;
                    let x32 = x32.broadcast_div(&((variance + &eps_t)?.sqrt()?))?;
                    let mut x = x32.to_dtype(x_dtype)?;
                    if norm_w.dtype() != x.dtype() {
                        norm_w = norm_w.to_dtype(x.dtype())?;
                    }
                    x = x.broadcast_mul(&norm_w)?;
                    // Pool along sequence to 1 token to reduce memory (keepdim to remain [1,1,D])
                    x = x.mean_keepdim(candle_core::D::Minus2)?;
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
                log::warn!(
                    "[qwen-image-edit] text_encoder embed/norm not found in weights; falling back to zero embeddings."
                );
                let d = self
                    .transformer_flux
                    .as_ref()
                    .map(|m| m.config.joint_attention_dim)
                    .or_else(|| {
                        self.transformer_model
                            .as_ref()
                            .map(|m| m.config.joint_attention_dim)
                    })
                    .unwrap_or(3584);
                let zero =
                    candle_core::Tensor::zeros((1usize, 1usize, d), self.dtype, &self.device)?;
                log::warn!("Created zeroed tensor");
                (zero.clone(), zero)
            }
        } else {
            let d = self
                .transformer_flux
                .as_ref()
                .map(|m| m.config.joint_attention_dim)
                .or_else(|| {
                    self.transformer_model
                        .as_ref()
                        .map(|m| m.config.joint_attention_dim)
                })
                .unwrap_or(3584);
            let zero = candle_core::Tensor::zeros((1usize, 1usize, d), self.dtype, &self.device)?;
            (zero.clone(), zero)
        };

        // If using Flux-style transformer, ensure latent spatial dims are divisible by patch_size
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
            // Optionally trim channels if encoder produced extra channels beyond expected latent channels
            let expect = tf.config.in_channels;
            if c > expect { z = z.i((.., 0..expect, .., ..))?; }
            // Log tokens and chunk size for visibility
            let n_tokens = (zh2 / ps) * (zw2 / ps);
            let chunk = tf.config.attn_chunk_size.unwrap_or(2048).min(n_tokens);
            log::info!(
                "[qwen-image-edit] Starting denoise: latent=({b},{c},{zh2},{zw2}), ps={ps}, tokens={n_tokens}, chunk={chunk}, heads={}, head_dim={}",
                tf.config.num_attention_heads,
                tf.config.attention_head_dim
            );
        }

        // Noise schedule
        let steps = opts.num_inference_steps.max(1);
        let sigmas = sched.inference_sigmas(steps);

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

        for i in 0..steps {
            let _sigma_from = sigmas[i];
            let _sigma_to = sigmas[i + 1];
            // Update overall progress at the start of the iteration
            QWEN_EDIT_STATUS.set_progress(1 + (i as u64) + 1, total_overall);
            QWEN_EDIT_STATUS.set_detail(format!("Step {}/{}", i + 1, steps));
            if guidance > 1.0 {
                // Scope temporaries to ensure prompt freeing of GPU buffers each iteration
                let z_new = {
                    let zu = if prefer_qwen && tr_qwen_opt.is_some() {
                        tr_qwen_opt.unwrap().forward(&z, &text_uncond)?
                    } else if let Some(tf) = tr_flux_opt {
                        tf.forward(&z, &text_uncond)?
                    } else {
                        tr_qwen_opt.unwrap().forward(&z, &text_uncond)?
                    };
                    let zc = if prefer_qwen && tr_qwen_opt.is_some() {
                        tr_qwen_opt.unwrap().forward(&z, &text_cond)?
                    } else if let Some(tf) = tr_flux_opt {
                        tf.forward(&z, &text_cond)?
                    } else {
                        tr_qwen_opt.unwrap().forward(&z, &text_cond)?
                    };
                    let diff = (zc - &zu)?;
                    // Multiply once with prebuilt guidance tensor
                    (&zu + diff.broadcast_mul(g_scale.as_ref().unwrap())?)?
                };
                z = z_new;
            } else {
                z = if prefer_qwen && tr_qwen_opt.is_some() {
                    tr_qwen_opt.unwrap().forward(&z, &text_cond)?
                } else if let Some(tf) = tr_flux_opt {
                    tf.forward(&z, &text_cond)?
                } else {
                    tr_qwen_opt.unwrap().forward(&z, &text_cond)?
                };
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
        let mut x_out = if let Some(ref vcpu) = vae_cpu_fallback {
            let mut z_cpu = z.to_device(&candle_core::Device::Cpu)?;
            if z_cpu.dtype() != candle_core::DType::F32 {
                z_cpu = z_cpu.to_dtype(candle_core::DType::F32)?;
            }
            vcpu.decode_to_original(&z_cpu, oh, ow)?
        } else {
            match vae.decode_to_original(&z, oh, ow) {
                Ok(img) => img,
                Err(e) => {
                    let msg = format!("{}", e);
                    if msg.contains("CUDA_ERROR_NOT_FOUND") || msg.contains("named symbol not found") {
                        log::warn!(
                            "[qwen-image-edit] VAE decode failed on CUDA due to named symbol; retrying decode on CPU/F32 with simplified Qwen VAE."
                        );
                        let cpu = candle_core::Device::Cpu;
                        let scale = 0.18215f32;
                        let vb = crate::ai::vae::build_qwen_vae_simplified_from_files(
                            &self.paths.vae_files,
                            candle_core::DType::F32,
                            &cpu,
                            scale,
                        );
                        match vb {
                            Ok(vs) => {
                                let vbox: Box<dyn crate::ai::vae::VaeLike> = Box::new(vs);
                                let mut z_cpu = z.to_device(&cpu)?;
                                if z_cpu.dtype() != candle_core::DType::F32 {
                                    z_cpu = z_cpu.to_dtype(candle_core::DType::F32)?;
                                }
                                vbox.decode_to_original(&z_cpu, oh, ow)?
                            }
                            Err(e2) => {
                                return Err(anyhow::anyhow!(
                                    "VAE decode failed on CUDA and CPU fallback build failed: {} | {}",
                                    msg, e2
                                ));
                            }
                        }
                    } else {
                        return Err(anyhow::Error::msg(msg));
                    }
                }
            }
        };
        // Stop watchdog thread
        decoding_flag.store(false, Ordering::Relaxed);
        if x_out.dtype() != self.dtype {
            x_out = x_out.to_dtype(self.dtype)?;
        }
        // To PNG efficiently: move to CPU, HWC layout, flatten, and encode
        QWEN_EDIT_STATUS.set_detail("Saving");
        QWEN_EDIT_STATUS.set_progress(1 + (steps as u64) + 1, total_overall);
        let x32 = x_out.to_dtype(candle_core::DType::F32)?;
        let x32 = x32.clamp(0f32, 1f32)?;
        let scale = candle_core::Tensor::new(255.0f32, &self.device)?;
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
