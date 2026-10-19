use super::*;

impl QwenImageEditPipeline {
    pub(super) fn load_from_hf_inner(
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
}
