use anyhow::{Result, anyhow, ensure};
use candle_core::{Device, DType, Tensor, Module};
use tokenizers::{Tokenizer, EncodeInput};
use std::path::PathBuf;
use serde::Deserialize;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};
use safetensors::SafeTensors;

#[derive(Debug, Clone)]
enum Weights {
    Single(PathBuf),
    Sharded(Vec<PathBuf>),
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LegacyPadIds { pub pad_token_id: Option<i64>, pub eos_token_id: Option<i64>, pub bos_token_id: Option<i64> }

pub struct JinaM0Engine {
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
    weights: Option<Weights>,
    pub config: Option<crate::ai::qwen2_5_vl::Config>,
    pub legacy_ids: Option<LegacyPadIds>,
}

impl JinaM0Engine {
    /// Discover the exact text model prefix from the weights by locating embed_tokens weight under text model path.
    fn discover_text_prefix_from_weights(&self) -> Result<Option<String>> {
        let scan_st = |st: &SafeTensors| -> Option<String> {
            // Look for X.embed_tokens.weight; X could be "model.text_model" or "text_model" or just "" (root)
            for name in st.names() {
                if !name.ends_with("embed_tokens.weight") { continue; }
                // We want the parent path before embed_tokens.weight
                if let Some(idx) = name.find("embed_tokens.weight") {
                    let path = &name[..idx];
                    // Trim trailing dot
                    let path = path.trim_end_matches('.');
                    // Common cases:
                    // - model.text_model.embed_tokens.weight -> prefix "model.text_model"
                    // - text_model.embed_tokens.weight -> prefix "text_model"
                    // - embed_tokens.weight -> prefix ""
                    return Some(path.to_string());
                }
            }
            None
        };
        match &self.weights {
            Some(Weights::Single(w)) => {
                let bytes = std::fs::read(w)?; let st = SafeTensors::deserialize(&bytes)?; Ok(scan_st(&st))
            }
            Some(Weights::Sharded(files)) => {
                if let Some(first) = files.first() {
                    let bytes = std::fs::read(first)?; let st = SafeTensors::deserialize(&bytes)?; Ok(scan_st(&st))
                } else { Ok(None) }
            }
            None => Ok(None)
        }
    }

    /// Discover the MLP head base by looking for both "<base>.0.weight" and "<base>.2.weight" pairs.
    fn discover_head_mlp_base_from_weights(&self) -> Result<Option<String>> {
        let scan_st = |st: &SafeTensors| -> Option<String> {
            use std::collections::HashSet;
            let mut bases_0: HashSet<String> = HashSet::new();
            let mut bases_2: HashSet<String> = HashSet::new();
            for name in st.names() {
                if name.ends_with(".0.weight") {
                    bases_0.insert(name.trim_end_matches(".0.weight").to_string());
                } else if name.ends_with(".2.weight") {
                    bases_2.insert(name.trim_end_matches(".2.weight").to_string());
                }
            }
            // Intersection preference by heuristic keywords
            let mut candidates: Vec<String> = bases_0.intersection(&bases_2).cloned().collect();
            candidates.sort_by_key(|b| {
                let lb = b.to_ascii_lowercase();
                let mut score = 0i32;
                if lb.contains("score") { score -= 10; }
                if lb.contains("rank") { score -= 8; }
                if lb.contains("head") { score -= 6; }
                score + (lb.matches('.').count() as i32) // fewer dots earlier
            });
            candidates.into_iter().next()
        };
        match &self.weights {
            Some(Weights::Single(w)) => {
                let bytes = std::fs::read(w)?; let st = SafeTensors::deserialize(&bytes)?; Ok(scan_st(&st))
            }
            Some(Weights::Sharded(files)) => {
                // Scan all shards to accumulate bases across files
                let mut best: Option<String> = None;
                for f in files {
                    let bytes = std::fs::read(f)?; let st = SafeTensors::deserialize(&bytes)?;
                    if let Some(b) = scan_st(&st) { best.get_or_insert(b); }
                }
                Ok(best)
            }
            None => Ok(None)
        }
    }
    pub fn load_from_repo(repo: &str) -> Result<Self> {
        log::info!("Loading {repo}");
        let device = crate::ai::hf::pick_device_cuda0_or_cpu();
        let tokenizer = crate::ai::hf::load_tokenizer(repo, None)?;
        let mdl = crate::ai::hf::hf_model(repo)?;
        let dtype = DType::F32;
        // Read config.json if present and convert top-level Jina fields into our nested config
        let (config, legacy_ids) = match mdl.get("config.json") {
            Ok(p) => {
                log::info!("[Rerank/Jina] Found config.json at {:?}", p);
                let bytes = std::fs::read(p)?;
                let legacy = serde_json::from_slice::<LegacyPadIds>(&bytes).ok();
                // Try multiple parse strategies to accommodate different HF config layouts
                let full = Self::parse_qwen_config_from_bytes(&bytes).map_err(|e| {
                    log::error!("[Rerank/Jina] Failed to parse config.json: {e}");
                    e
                }).ok();
                (full, legacy)
            }
            Err(e) => {
                log::warn!("[Rerank/Jina] config.json not found in repo: {e}");
                (None, None)
            }
        };
        // Capture weights location for later model construction
        let mut weights: Option<Weights> = None;
        if let Ok(w) = mdl.get("model.safetensors") {
            // Validate mmap is possible now, but keep the path for future opens
            unsafe { crate::ai::hf::with_mmap_varbuilder_single(&w, dtype, &device, |_vb| Ok(())) }?;
            weights = Some(Weights::Single(w));
        } else if let Ok(idx) = mdl.get("model.safetensors.index.json") {
            let files = candle_examples::hub_load_local_safetensors(idx.parent().unwrap(), "model.safetensors.index.json")?;
            unsafe { crate::ai::hf::with_mmap_varbuilder_multi(&files, dtype, &device, |_vb| Ok(())) }?;
            weights = Some(Weights::Sharded(files));
        }
        if let Some(cfg) = &config {
            log::info!(
                "[Rerank/Jina] Repo config loaded: hidden={} layers={} heads={} kv_heads={} vocab={}",
                cfg.text_config.hidden_size,
                cfg.text_config.num_hidden_layers,
                cfg.text_config.num_attention_heads,
                cfg.text_config.num_key_value_heads,
                cfg.text_config.vocab_size
            );
        } else {
            log::warn!("[Rerank/Jina] No config.json found or failed to parse; will try to infer from weights");
        }
        Ok(Self { tokenizer, device, dtype, weights, config, legacy_ids })
    }

    fn parse_qwen_config_from_bytes(bytes: &[u8]) -> Result<crate::ai::qwen2_5_vl::Config> {
        // 1) Try our RawJinaConfig (top-level fields)
        if let Ok(raw) = serde_json::from_slice::<RawJinaConfig>(bytes) {
            return Ok(raw.into_qwen_config());
        }
        // 2) Try nested text_config/vision_config (common HF layout)
        #[derive(Debug, Clone, Deserialize)]
        struct HFTextConfig {
            vocab_size: Option<usize>,
            hidden_size: Option<usize>,
            intermediate_size: Option<usize>,
            num_hidden_layers: Option<usize>,
            num_attention_heads: Option<usize>,
            num_key_value_heads: Option<usize>,
            max_position_embeddings: Option<usize>,
            rope_theta: Option<f64>,
            rms_norm_eps: Option<f64>,
            use_sliding_window: Option<bool>,
            sliding_window: Option<usize>,
            max_window_layers: Option<usize>,
        }
        #[derive(Debug, Clone, Deserialize)]
        struct HFVisionConfig {
            hidden_size: Option<usize>,
            in_chans: Option<usize>,
            spatial_patch_size: Option<usize>,
        }
        #[derive(Debug, Clone, Deserialize)]
        struct HFConfigRoot {
            text_config: Option<HFTextConfig>,
            vision_config: Option<HFVisionConfig>,
            vision_start_token_id: Option<u32>,
            vision_end_token_id: Option<u32>,
            #[serde(alias = "vision_token_id", alias = "image_token_id")]
            image_token_id: Option<u32>,
        }
        if let Ok(root) = serde_json::from_slice::<HFConfigRoot>(bytes) {
            if let Some(t) = root.text_config {
                let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
                    vocab_size: t.vocab_size.unwrap_or(151665),
                    hidden_size: t.hidden_size.unwrap_or(3584),
                    intermediate_size: t.intermediate_size.unwrap_or(18944),
                    num_hidden_layers: t.num_hidden_layers.unwrap_or(28),
                    num_attention_heads: t.num_attention_heads.unwrap_or(28),
                    num_key_value_heads: t.num_key_value_heads.unwrap_or(4),
                    max_position_embeddings: t.max_position_embeddings.unwrap_or(32768),
                    rope_theta: t.rope_theta.unwrap_or(1_000_000.0),
                    rms_norm_eps: t.rms_norm_eps.unwrap_or(1e-5),
                    use_sliding_window: t.use_sliding_window.unwrap_or(false),
                    sliding_window: t.sliding_window,
                    max_window_layers: t.max_window_layers.unwrap_or_else(|| t.num_hidden_layers.unwrap_or(28)),
                    mrope_section: None,
                    rope_scaling: None,
                };
                let vraw = root.vision_config.unwrap_or(HFVisionConfig { hidden_size: None, in_chans: None, spatial_patch_size: None });
                let vision_cfg = crate::ai::qwen2_5_vl::VisionConfig {
                    hidden_size: vraw.hidden_size.unwrap_or(1536),
                    intermediate_size: 6144,
                    num_hidden_layers: 32,
                    num_attention_heads: 16,
                    image_size: 896,
                    patch_size: vraw.spatial_patch_size.unwrap_or(14),
                    num_channels: vraw.in_chans.unwrap_or(3),
                    spatial_merge_size: 2,
                    temporal_patch_size: 2,
                    out_hidden_size: None,
                    window_size: None,
                };
                return Ok(crate::ai::qwen2_5_vl::Config {
                    vision_config: vision_cfg,
                    hidden_size: text_cfg.hidden_size,
                    text_config: text_cfg,
                    vision_start_token_id: root.vision_start_token_id.unwrap_or(151652),
                    vision_end_token_id: root.vision_end_token_id.unwrap_or(151653),
                    image_token_id: root.image_token_id.unwrap_or(151654),
                });
            }
        }
        // 3) Generic resilient parse: read as Value and map fields manually
        let v: serde_json::Value = serde_json::from_slice(bytes)?;
        let get_usize = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|u| u as usize);
        let get_f64 = |k: &str| v.get(k).and_then(|x| x.as_f64());
        let get_bool = |k: &str| v.get(k).and_then(|x| x.as_bool());
        let vocab_size = get_usize("vocab_size").unwrap_or(151936);
        let hidden_size = get_usize("hidden_size").unwrap_or(1536);
        let intermediate_size = get_usize("intermediate_size").unwrap_or(8960);
        let num_hidden_layers = get_usize("num_hidden_layers").unwrap_or(28);
        let num_attention_heads = get_usize("num_attention_heads").unwrap_or(12);
        let num_key_value_heads = get_usize("num_key_value_heads").unwrap_or(2);
        let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(32768);
        let rope_theta = get_f64("rope_theta").unwrap_or(1_000_000.0);
        let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
        let use_sliding_window = get_bool("use_sliding_window").unwrap_or(false);
        let sliding_window = get_usize("sliding_window");
        let max_window_layers = get_usize("max_window_layers").unwrap_or(num_hidden_layers);
        let rope_scaling = v.get("rope_scaling").and_then(|rs| rs.get("mrope_section")).and_then(|m| m.as_array()).map(|arr| {
            arr.iter().filter_map(|x| x.as_u64()).map(|u| u as usize).collect::<Vec<_>>()
        });
        let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            mrope_section: rope_scaling,
            rope_scaling: None,
        };
        let vc = v.get("vision_config").cloned().unwrap_or(serde_json::json!({}));
        let v_hidden = vc.get("hidden_size").and_then(|x| x.as_u64()).map(|u| u as usize).unwrap_or(1536);
        let v_in = vc.get("in_chans").and_then(|x| x.as_u64()).map(|u| u as usize).unwrap_or(3);
        let v_patch = vc.get("spatial_patch_size").and_then(|x| x.as_u64()).map(|u| u as usize).unwrap_or(14);
        let vision_cfg = crate::ai::qwen2_5_vl::VisionConfig {
            hidden_size: v_hidden,
            intermediate_size: 6144,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 896,
            patch_size: v_patch,
            num_channels: v_in,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: None,
            window_size: None,
        };
        let vision_start_token_id = v.get("vision_start_token_id").and_then(|x| x.as_u64()).map(|u| u as u32).unwrap_or(151652);
        let vision_end_token_id = v.get("vision_end_token_id").and_then(|x| x.as_u64()).map(|u| u as u32).unwrap_or(151653);
        let image_token_id = v.get("image_token_id").or_else(|| v.get("vision_token_id")).and_then(|x| x.as_u64()).map(|u| u as u32).unwrap_or(151654);
        Ok(crate::ai::qwen2_5_vl::Config {
            vision_config: vision_cfg,
            text_config: text_cfg.clone(),
            hidden_size: text_cfg.hidden_size,
            vision_start_token_id,
            vision_end_token_id,
            image_token_id,
        })
    }

    /// Text-only scoring using the language backbone under the "model" prefix and Jina's score head.
    pub fn compute_score_text(&self, pairs: &[(String, String)], max_length: usize) -> Result<Vec<f32>> {
        // 1) Build token ids and attention masks [B, L]
    let (input_ids, attn_mask_ids) = self.prepare_inputs(pairs, max_length)?;
    let (b, l) = (input_ids.dim(0)?, input_ids.dim(1)?);
    log::info!("[Rerank/Jina] Scoring {} pairs (seq_len={} max_length={})", b, l, max_length);
        let bsz = input_ids.dim(0)?;

        // 2) Deterministic text-only backbone: discover the exact prefix from safetensors and build once
        let tconf = if let Some(cfg) = &self.config { cfg.text_config.clone() } else {
            match &self.weights {
                Some(Weights::Single(w)) => Self::discover_text_config_from_file(w)?.ok_or_else(|| anyhow!("Failed to infer text config from single weights"))?,
                Some(Weights::Sharded(files)) => Self::discover_text_config_from_multi(files)?.ok_or_else(|| anyhow!("Failed to infer text config from sharded weights"))?,
                None => return Err(anyhow!("Missing model weights (safetensors)")),
            }
        };
        let text_prefix = self.discover_text_prefix_from_weights()?.unwrap_or_default();
        log::info!("[Rerank/Jina] Using text prefix: '{}'", if text_prefix.is_empty() { "<root>" } else { &text_prefix });

        // 2b) Try to build and run the text backbone; if it fails, fall back to embed-only pooling
        let pooled = match &self.weights {
            Some(Weights::Single(w)) => {
                let full = unsafe {
                    crate::ai::hf::with_mmap_varbuilder_single(w, self.dtype, &self.device, |vb| {
                        let vb = if text_prefix.is_empty() { vb } else { vb.pp(&text_prefix) };
                        crate::ai::qwen2_5_vl::TextModel::new(&tconf, vb).map_err(anyhow::Error::from)
                    })
                };
                match full {
                    Ok(mut model) => {
                        match self.forward_text_hidden(&mut model, &input_ids) {
                            Ok(text_hidden) => {
                                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                                let masked = text_hidden.broadcast_mul(&mask_e)?; // [B, L, H]
                                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B, H]
                                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                                sum_h.broadcast_div(&lens.unsqueeze(1)?)? // [B,H]
                            }
                            Err(e) => {
                                log::warn!("[Rerank/Jina] Text forward failed: {e:?}. Falling back to embed-only pooling.");
                                self.embed_only_pooled(&input_ids, &attn_mask_ids, &tconf, &text_prefix)?
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("[Rerank/Jina] TextModel build failed: {e:?}. Falling back to embed-only pooling.");
                        self.embed_only_pooled(&input_ids, &attn_mask_ids, &tconf, &text_prefix)?
                    }
                }
            }
            Some(Weights::Sharded(files)) => {
                let full = unsafe {
                    crate::ai::hf::with_mmap_varbuilder_multi(files, self.dtype, &self.device, |vb| {
                        let vb = if text_prefix.is_empty() { vb } else { vb.pp(&text_prefix) };
                        crate::ai::qwen2_5_vl::TextModel::new(&tconf, vb).map_err(anyhow::Error::from)
                    })
                };
                match full {
                    Ok(mut model) => {
                        match self.forward_text_hidden(&mut model, &input_ids) {
                            Ok(text_hidden) => {
                                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                                let masked = text_hidden.broadcast_mul(&mask_e)?; // [B, L, H]
                                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B, H]
                                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                                sum_h.broadcast_div(&lens.unsqueeze(1)?)? // [B,H]
                            }
                            Err(e) => {
                                log::warn!("[Rerank/Jina] Text forward failed: {e:?}. Falling back to embed-only pooling.");
                                self.embed_only_pooled(&input_ids, &attn_mask_ids, &tconf, &text_prefix)?
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("[Rerank/Jina] TextModel build failed: {e:?}. Falling back to embed-only pooling.");
                        self.embed_only_pooled(&input_ids, &attn_mask_ids, &tconf, &text_prefix)?
                    }
                }
            }
            None => unreachable!(),
        };
        log::info!("[Rerank/Jina] Pooled hidden dim: {}", pooled.dim(1).unwrap_or(0));

        // 4) Apply Jina's score head. Prefer the two-layer MLP head (score.0 -> GELU -> score.2),
        // fall back to a single linear head discovery if needed.
    let hidden_size = pooled.dim(1)?;
        let logits = {
            // Prefer an automatically discovered two-layer MLP head base
            if let Some(base) = self.discover_head_mlp_base_from_weights()? {
                log::info!("[Rerank/Jina] Using two-layer MLP head base: {}", base);
                match &self.weights {
                    Some(Weights::Single(w)) => unsafe {
                        let head = crate::ai::hf::with_mmap_varbuilder_single(w, self.dtype, &self.device, |vb| {
                            let h = linear(hidden_size, hidden_size, vb.pp(&format!("{}.0", base)))?;
                            let out = linear(hidden_size, 1, vb.pp(&format!("{}.2", base)))?;
                            Ok::<(Linear, Linear), anyhow::Error>((h, out))
                        })?;
                        self.apply_two_layer_head(&pooled, head)?
                    },
                    Some(Weights::Sharded(files)) => unsafe {
                        let head = crate::ai::hf::with_mmap_varbuilder_multi(files, self.dtype, &self.device, |vb| {
                            let h = linear(hidden_size, hidden_size, vb.pp(&format!("{}.0", base)))?;
                            let out = linear(hidden_size, 1, vb.pp(&format!("{}.2", base)))?;
                            Ok::<(Linear, Linear), anyhow::Error>((h, out))
                        })?;
                        self.apply_two_layer_head(&pooled, head)?
                    },
                    None => unreachable!(),
                }
            } else {
                // Linear fallback
                match &self.weights {
                    Some(Weights::Single(_)) => {
                        if let Some(head) = self.load_scoring_head_linear_single(match &self.weights { Some(Weights::Single(w)) => w, _ => unreachable!() }, hidden_size)? {
                            log::info!("[Rerank/Jina] Using linear head (single)");
                            head.forward(&pooled)?
                        } else {
                            log::warn!("[Rerank/Jina] Ranking head weights not found (linear, single). Dumping head-like keys…");
                            self.debug_log_head_candidates(12);
                            return Err(anyhow!("Ranking head weights not found (linear, single)."));
                        }
                    }
                    Some(Weights::Sharded(files)) => {
                        if let Some(head) = self.load_scoring_head_linear_sharded(files, hidden_size)? {
                            log::info!("[Rerank/Jina] Using linear head (sharded)");
                            head.forward(&pooled)?
                        } else {
                            log::warn!("[Rerank/Jina] Ranking head weights not found (linear, sharded). Dumping head-like keys…");
                            self.debug_log_head_candidates(12);
                            return Err(anyhow!("Ranking head weights not found (linear, sharded)."));
                        }
                    }
                    None => unreachable!(),
                }
            }
        };

        // 5) Sigmoid to [0,1]
        let mut scores = logits.flatten_all()?.to_vec1::<f32>()?;
        log::info!("[Rerank/Jina] Raw logits len: {} (batch {})", scores.len(), bsz);
        for s in &mut scores { *s = 1.0 / (1.0 + (-*s).exp()); }
        if scores.len() != bsz {
            log::error!("[Rerank/Jina] Score length mismatch: got {} expected {}", scores.len(), bsz);
            ensure!(scores.len() == bsz, "score length mismatch");
        }
        Ok(scores)
    }

    /// Tokenize text pairs and build [B, L] input_ids and attention_mask tensors.
    fn prepare_inputs(&self, pairs: &[(String, String)], max_length: usize) -> Result<(Tensor, Tensor)> {
        // Determine pad token id
        let pad_id: i64 = self
            .legacy_ids
            .as_ref()
            .and_then(|c| c.pad_token_id.or(c.eos_token_id).or(c.bos_token_id))
            .unwrap_or(0);
        let mut tokens: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut masks: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        for (q, d) in pairs.iter() {
            let enc = self
                .tokenizer
                .encode(EncodeInput::Dual(q.as_str().into(), d.as_str().into()), true)
                .map_err(anyhow::Error::msg)?;
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            if ids.len() > max_length { ids.truncate(max_length); }
            let mut mask: Vec<i64> = vec![1; ids.len()];
            if ids.len() < max_length {
                let pad_len = max_length - ids.len();
                ids.extend(std::iter::repeat(pad_id).take(pad_len));
                mask.extend(std::iter::repeat(0).take(pad_len));
            }
            tokens.push(ids);
            masks.push(mask);
        }
        let input_ids = Tensor::new(tokens, &self.device)?;   // [B, L]
        let attention = Tensor::new(masks, &self.device)?;    // [B, L]
        Ok((input_ids, attention))
    }

    /// Try to load a single-linear scoring head from safetensors using common prefixes.
    /// Returns Ok(None) if none of the expected prefixes are present.
    fn load_scoring_head_linear_single(&self, _w: &PathBuf, hidden_size: usize) -> Result<Option<Linear>> { self.load_scoring_head_linear_inner(hidden_size, false) }

    fn load_scoring_head_linear_sharded(&self, _files: &[PathBuf], hidden_size: usize) -> Result<Option<Linear>> { self.load_scoring_head_linear_inner(hidden_size, true) }

    fn load_scoring_head_linear_inner(&self, hidden_size: usize, sharded: bool) -> Result<Option<Linear>> {
        static BASE_PREFIXES: &[&str] = &[
            "score_head",
            "ranking_head",
            "rank_head",
            "reranker_head",
            "classification_head",
            "cls_head",
            "head",
            "score.2",
        ];

        // Build candidate prefixes including discovered text prefix and its root
        let mut candidates: Vec<String> = Vec::new();
        for p in BASE_PREFIXES { candidates.push((*p).to_string()); }
        if let Ok(Some(tp)) = self.discover_text_prefix_from_weights() {
            if !tp.is_empty() {
                for p in BASE_PREFIXES { candidates.push(format!("{tp}.{p}")); }
                // Also try the root segment (e.g., "model" from "model.text_model")
                if let Some(root) = tp.split('.').next() {
                    for p in BASE_PREFIXES { candidates.push(format!("{root}.{p}")); }
                }
            }
        }

        let try_build = |vb: candle_nn::VarBuilder, cands: &[String]| -> Result<Option<Linear>> {
            for p in cands {
                // Try no-bias first
                if let Ok(l) = linear_no_bias(hidden_size, 1, vb.pp(p)) {
                    return Ok(Some(l));
                }
                // Then with bias
                if let Ok(l) = linear(hidden_size, 1, vb.pp(p)) {
                    return Ok(Some(l));
                }
            }
            Ok(None)
        };

        // Helper to build with a discovered prefix
        let try_build_with_prefix = |vb: candle_nn::VarBuilder, prefix: &str| -> Result<Option<Linear>> {
            if let Ok(l) = linear_no_bias(hidden_size, 1, vb.pp(prefix)) { return Ok(Some(l)); }
            if let Ok(l) = linear(hidden_size, 1, vb.pp(prefix)) { return Ok(Some(l)); }
            Ok(None)
        };

        match (&self.weights, sharded) {
            (Some(Weights::Single(w)), false) => unsafe {
                if let Some(lin) = crate::ai::hf::with_mmap_varbuilder_single(w, self.dtype, &self.device, |vb| try_build(vb, &candidates))? {
                    return Ok(Some(lin));
                }
                if let Some(prefix) = Self::discover_head_prefix_from_file(w)? {
                    return crate::ai::hf::with_mmap_varbuilder_single(w, self.dtype, &self.device, |vb| try_build_with_prefix(vb, &prefix));
                }
                Ok(None)
            },
            (Some(Weights::Sharded(files)), true) => unsafe {
                crate::ai::hf::with_mmap_varbuilder_multi(files, self.dtype, &self.device, |vb| try_build(vb, &candidates))
            },
            _ => Ok(None),
        }
    }

    fn apply_two_layer_head(&self, x: &Tensor, head: (Linear, Linear)) -> Result<Tensor> {
        let (h, out) = head;
        let x = h.forward(x)?.gelu()?;
        Ok(out.forward(&x)?)
    }



    /// Inspect a single-file safetensors to locate a likely ranking head prefix by name.
    fn discover_head_prefix_from_file(path: &PathBuf) -> Result<Option<String>> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        // Score candidates by presence of keywords and having a ".weight" tensor
        let mut best: Option<(i64, String)> = None; // (score, prefix)
        for name in st.names() {
            if !name.ends_with(".weight") { continue; }
            // Extract prefix before trailing .weight
            let prefix = &name[..name.len() - ".weight".len()];
            let mut score: i64 = 0;
            let lname = prefix.to_ascii_lowercase();
            if lname.contains("rank") { score += 5; }
            if lname.contains("rerank") { score += 5; }
            if lname.contains("score") { score += 4; }
            if lname.contains("head") { score += 3; }
            if lname.contains("cls") { score += 1; }
            // Prefer shapes compatible with [hidden_size, 1] or [1, hidden_size]
            if let Ok(t) = st.tensor(name) {
                let shape = t.shape();
                // Add weight for likely shapes
                if shape.len() == 2 {
                    let r0 = shape[0] as i64;
                    let r1 = shape[1] as i64;
                    // Hx1 or 1xH get extra boost
                    if r0 == 1 || r1 == 1 { score += 6; }
                }
            }
            // Prefer shorter prefixes (shallower) slightly
            score -= prefix.matches('.').count() as i64; // fewer dots => higher score
            if score > 0 {
                match &best {
                    None => best = Some((score, prefix.to_string())),
                    Some((bs, _)) if score > *bs => best = Some((score, prefix.to_string())),
                    _ => {}
                }
            }
        }
        Ok(best.map(|(_, p)| p))
    }

    /// Debug helper: log a few head-like candidate keys to help map custom repos.
    fn debug_log_head_candidates(&self, limit: usize) {
        let mut logged = 0usize;
        match &self.weights {
            Some(Weights::Single(w)) => {
                if let Ok(bytes) = std::fs::read(w) {
                    if let Ok(st) = SafeTensors::deserialize(&bytes) {
                        for name in st.names() {
                            let ln = name.to_ascii_lowercase();
                            if ln.ends_with(".weight") && (ln.contains("rank") || ln.contains("score") || ln.contains("head")) {
                                if logged < limit { log::info!("[Rerank/Jina] head-candidate: {name}"); logged += 1; }
                                if logged >= limit { break; }
                            }
                        }
                    }
                }
            }
            Some(Weights::Sharded(files)) => {
                for f in files {
                    if let Ok(bytes) = std::fs::read(f) {
                        if let Ok(st) = SafeTensors::deserialize(&bytes) {
                            for name in st.names() {
                                let ln = name.to_ascii_lowercase();
                                if ln.ends_with(".weight") && (ln.contains("rank") || ln.contains("score") || ln.contains("head")) {
                                    if logged < limit { log::info!("[Rerank/Jina] head-candidate: {name}"); logged += 1; }
                                    if logged >= limit { break; }
                                }
                            }
                        }
                    }
                    if logged >= limit { break; }
                }
            }
            None => {}
        }
        if logged == 0 { log::info!("[Rerank/Jina] No head-like keys found in weights (scanned up to {limit} entries per shard)"); }
    }
}


impl JinaM0Engine {
    fn forward_text_hidden(
        &self,
        text: &mut crate::ai::qwen2_5_vl::TextModel,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let position_ids = Tensor::arange(0i64, l as i64, &self.device)?
            .unsqueeze(0)?.expand((b, l))?;
        let hidden = text.forward(input_ids, &position_ids, None)?; // [B,L,H]
        Ok(hidden)
    }
    /// Fallback: compute masked-mean directly over token embeddings without decoder layers.
    /// Useful when rotary/attention issues occur but embeddings and norm are valid.
    fn embed_only_pooled(
        &self,
        input_ids: &Tensor,
        attn_mask_ids: &Tensor,
        tconf: &crate::ai::qwen2_5_vl::TextConfig,
        text_prefix: &str,
    ) -> Result<Tensor> {
        let pooled = match &self.weights {
            Some(Weights::Single(w)) => unsafe {
                let (emb, norm_w) = crate::ai::hf::with_mmap_varbuilder_single(w, self.dtype, &self.device, |vb| {
                    let vb = if text_prefix.is_empty() { vb } else { vb.pp(text_prefix) };
                    let emb = candle_nn::embedding(tconf.vocab_size, tconf.hidden_size, vb.pp("embed_tokens"))?;
                    let norm_w = vb.pp("norm").get(tconf.hidden_size, "weight")?;
                    Ok::<(candle_nn::Embedding, Tensor), anyhow::Error>((emb, norm_w))
                })?;
                let x = emb.forward(input_ids)?; // [B,L,H]
                // RMSNorm inline
                let x_dtype = x.dtype();
                let x32 = x.to_dtype(candle_core::DType::F32)?;
                let variance = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let x32 = x32.broadcast_div(&(variance + tconf.rms_norm_eps)?.sqrt()?)?;
                let x = x32.to_dtype(x_dtype)?;
                let x = x.broadcast_mul(&norm_w)?; // [B,L,H]
                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                let masked = x.broadcast_mul(&mask_e)?; // [B,L,H]
                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B,H]
                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                sum_h.broadcast_div(&lens.unsqueeze(1)?)?
            },
            Some(Weights::Sharded(files)) => unsafe {
                let (emb, norm_w) = crate::ai::hf::with_mmap_varbuilder_multi(files, self.dtype, &self.device, |vb| {
                    let vb = if text_prefix.is_empty() { vb } else { vb.pp(text_prefix) };
                    let emb = candle_nn::embedding(tconf.vocab_size, tconf.hidden_size, vb.pp("embed_tokens"))?;
                    let norm_w = vb.pp("norm").get(tconf.hidden_size, "weight")?;
                    Ok::<(candle_nn::Embedding, Tensor), anyhow::Error>((emb, norm_w))
                })?;
                let x = emb.forward(input_ids)?;
                // RMSNorm inline
                let x_dtype = x.dtype();
                let x32 = x.to_dtype(candle_core::DType::F32)?;
                let variance = x32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let x32 = x32.broadcast_div(&(variance + tconf.rms_norm_eps)?.sqrt()?)?;
                let x = x32.to_dtype(x_dtype)?;
                let x = x.broadcast_mul(&norm_w)?;
                let mask_f = attn_mask_ids.to_dtype(candle_core::DType::F32)?; // [B, L]
                let mask_e = mask_f.unsqueeze(2)?; // [B, L, 1]
                let masked = x.broadcast_mul(&mask_e)?; // [B,L,H]
                let sum_h = masked.sum(candle_core::D::Minus2)?; // [B,H]
                let lens = mask_f.sum(candle_core::D::Minus1)?; // [B]
                let lens = (lens + Tensor::new(1e-6f32, &self.device)?)?; // avoid div by zero
                sum_h.broadcast_div(&lens.unsqueeze(1)?)?
            },
            None => unreachable!(),
        };
        Ok(pooled)
    }
    // Infer text config (vocab_size, hidden_size, intermediate_size, layers, heads) from safetensors shapes
    fn discover_text_config_from_file(path: &PathBuf) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        Self::discover_text_config_from_st(&st)
    }

    fn discover_text_config_from_multi(files: &[PathBuf]) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        // For sharded: read first shard to access metadata of names; if not available, bail
        if let Some(first) = files.first() {
            let bytes = std::fs::read(first)?;
            let st = SafeTensors::deserialize(&bytes)?;
            return Self::discover_text_config_from_st(&st);
        }
        Ok(None)
    }

    fn discover_text_config_from_st(st: &safetensors::SafeTensors<'_>) -> Result<Option<crate::ai::qwen2_5_vl::TextConfig>> {
        // Find embed_tokens weight to get vocab_size, hidden_size
        let mut embed_shape: Option<(usize, usize)> = None;
        for name in st.names() {
            if name.ends_with("embed_tokens.weight") {
                let t = st.tensor(name)?;
                let shape = t.shape();
                if shape.len() == 2 {
                    embed_shape = Some((shape[0] as usize, shape[1] as usize));
                    break;
                }
            }
        }
        let (vocab_size, hidden_size) = match embed_shape { Some(s) => s, None => return Ok(None) };

        // Intermediate size from gate_proj
        let mut intermediate_size: Option<usize> = None;
        for name in st.names() {
            if name.contains("layers.0.mlp.gate_proj.weight") {
                let t = st.tensor(name)?;
                let shape = t.shape();
                if shape.len() == 2 { intermediate_size = Some(shape[0] as usize); break; }
            }
        }
        let intermediate_size = intermediate_size.unwrap_or(hidden_size * 4);

        // k_proj out-dim to help infer kv heads
        let mut kv_out: Option<usize> = None;
        for name in st.names() {
            if name.contains("layers.0.self_attn.k_proj.weight") {
                let t = st.tensor(name)?; let shape = t.shape();
                if shape.len() == 2 { kv_out = Some(shape[0] as usize); break; }
            }
        }
        let kv_out = kv_out.unwrap_or(hidden_size / 6); // heuristic fallback

        // Count layers by finding max index in model.layers.<i>.*
        let mut max_layer = 0usize;
        for name in st.names() {
            if let Some(idx) = name.find("layers.") {
                let rest = &name[idx + "layers.".len()..];
                if let Some(end) = rest.find('.') {
                    if let Ok(i) = rest[..end].parse::<usize>() { if i > max_layer { max_layer = i; } }
                }
            }
        }
        let num_hidden_layers = max_layer + 1;

        // Infer head_dim and kv_heads/attn_heads combinatorially
        let mut best: Option<(usize, usize)> = None; // (num_heads, num_kv_heads)
        let preferred_kv = [4usize, 8, 2, 16, 32];
        for &kv in &preferred_kv {
            if kv_out % kv != 0 { continue; }
            let head_dim = kv_out / kv; // must divide hidden_size
            if hidden_size % head_dim != 0 { continue; }
            let num_heads = hidden_size / head_dim;
            if num_heads >= kv && num_heads <= 64 {
                best = Some((num_heads, kv));
                break;
            }
        }
        // fallback: try kv=hidden/64 etc.
        let (num_attention_heads, num_key_value_heads) = best.unwrap_or_else(|| {
            let head_dim = 64.min(hidden_size);
            let nh = hidden_size / head_dim; let kvh = (kv_out / head_dim).max(2).min(nh.max(2));
            (nh.max(1), kvh)
        });

        let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            use_sliding_window: false,
            sliding_window: Some(32768),
            max_window_layers: num_hidden_layers,
            mrope_section: None,
            rope_scaling: None,
        };
        Ok(Some(text_cfg))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawVisionConfig {
    hidden_size: Option<usize>,
    #[serde(alias = "in_chans")] 
    in_chans: Option<usize>,
    #[serde(alias = "spatial_patch_size")] 
    spatial_patch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRopeScaling { mrope_section: Option<Vec<usize>> }

#[derive(Debug, Clone, Deserialize)]
struct RawJinaConfig {
    // text-related top-level fields
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_theta: Option<f64>,
    rms_norm_eps: Option<f64>,
    use_sliding_window: Option<bool>,
    sliding_window: Option<usize>,
    max_window_layers: Option<usize>,
    rope_scaling: Option<RawRopeScaling>,
    // token ids
    #[allow(unused)]    
    bos_token_id: Option<u32>,
    #[allow(unused)]    
    eos_token_id: Option<u32>,
    #[serde(alias = "vision_token_id", alias = "image_token_id")]
    image_token_id: Option<u32>,
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
    // vision
    vision_config: Option<RawVisionConfig>,
}

impl RawJinaConfig {
    fn into_qwen_config(self) -> crate::ai::qwen2_5_vl::Config {
        let text_cfg = crate::ai::qwen2_5_vl::TextConfig {
            vocab_size: self.vocab_size.unwrap_or(151665),
            hidden_size: self.hidden_size.unwrap_or(3584),
            intermediate_size: self.intermediate_size.unwrap_or(18944),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(28),
            num_attention_heads: self.num_attention_heads.unwrap_or(28),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(32768),
            rope_theta: self.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            sliding_window: Some(self.sliding_window.unwrap_or(32768)),
            max_window_layers: self.max_window_layers.unwrap_or(28),
            mrope_section: self.rope_scaling.as_ref().and_then(|r| r.mrope_section.clone()),
            rope_scaling: None,
        };

        let vraw = self.vision_config.unwrap_or(RawVisionConfig { hidden_size: None, in_chans: None, spatial_patch_size: None });
        let vision_cfg = crate::ai::qwen2_5_vl::VisionConfig {
            hidden_size: vraw.hidden_size.unwrap_or(1536),
            intermediate_size: 6144,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 896,
            patch_size: vraw.spatial_patch_size.unwrap_or(14),
            num_channels: vraw.in_chans.unwrap_or(3),
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: None,
            window_size: None,
        };

        crate::ai::qwen2_5_vl::Config {
            vision_config: vision_cfg,
            text_config: text_cfg.clone(),
            hidden_size: text_cfg.hidden_size,
            vision_start_token_id: self.vision_start_token_id.unwrap_or(151652),
            vision_end_token_id: self.vision_end_token_id.unwrap_or(151653),
            image_token_id: self.image_token_id.unwrap_or(151654),
        }
    }
}
