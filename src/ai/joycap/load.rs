use crate::app::{MAX_NEW_TOKENS, TEMPERATURE};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use candle_core::DType;
use std::path::Path;
use crate::ai::candle_llava::{
    config::{HFLLaVAConfig, 
        HFGenerationConfig, 
        HFPreProcessorConfig
    },
    model::LLaVA,
    llama::Cache,
};

impl super::JoyCaptionModel {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        use crate::ui::status::{VISION_STATUS, StatusState, GlobalStatusIndicator};
        VISION_STATUS.set_state(StatusState::Initializing, "Reading config");
        log::info!("[joycaption] loading model from dir: {}", dir.display());
        let config_path = dir.join("config.json");
        let gen_cfg_path = dir.join("generation_config.json");
        let pre_proc_path = dir.join("preprocessor_config.json");
        let processor_path = dir.join("processor_config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        if !config_path.exists() { anyhow::bail!("Missing config.json in {:?}", dir); }
        if !tokenizer_path.exists() { anyhow::bail!("Missing tokenizer.json in {:?}", dir); }

        // --------- START new phased normalization (mirroring reference loader) ---------
        let mut cfg_val: Value = serde_json::from_slice(&std::fs::read(&config_path)?)
            .with_context(|| format!("parse config.json failed: {}", config_path.display()))?;
        log::debug!("[joycaption] cfg_val root keys: {:?}", cfg_val.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()));
        let gen_val_opt: Option<Value> = std::fs::read(&gen_cfg_path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        log::debug!("[joycaption] gen_val_opt present: {}", gen_val_opt.is_some());
        let mut pre_val_raw: Option<Value> = std::fs::read(&pre_proc_path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        if pre_val_raw.is_none() { pre_val_raw = std::fs::read(&processor_path).ok().and_then(|b| serde_json::from_slice(&b).ok()); }
        let mut pre_val: Value = pre_val_raw.unwrap_or_else(|| json!({}));

        // Phase 0: helper to collapse single/array token id to scalar
        fn scalarize_id(opt: Option<&Value>, default_: i64) -> (Value, i64) {
            match opt {
                Some(Value::Number(n)) => { let v = n.as_i64().unwrap_or(default_); (json!(v), v) }
                Some(Value::Array(a)) => { let v = a.get(0).and_then(|x| x.as_i64()).unwrap_or(default_); (json!(v), v) }
                Some(other) => (other.clone(), default_),
                None => (json!(default_), default_),
            }
        }

        // Phase 1: Determine BOS/EOS/PAD sources (prefer config root, then generation config) BEFORE mutation
        let bos_from_gen = gen_val_opt.as_ref().and_then(|v| v.get("bos_token_id")).cloned();
        let eos_from_gen = gen_val_opt.as_ref().and_then(|v| v.get("eos_token_id")).cloned();
        let pad_from_gen = gen_val_opt.as_ref().and_then(|v| v.get("pad_token_id")).cloned();

        let bos = cfg_val.get("bos_token_id").cloned().or(bos_from_gen).unwrap_or(json!(1));
        let eos = cfg_val.get("eos_token_id").cloned().or(eos_from_gen).unwrap_or(json!(2));
        let pad = cfg_val.get("pad_token_id").cloned().or(pad_from_gen).unwrap_or_else(|| eos.clone());

        // Inject into cfg before deserializing to HFLLaVAConfig
        let obj = cfg_val.as_object_mut().expect("config.json must be an object");
        obj.insert("bos_token_id".to_string(), bos);
        obj.insert("eos_token_id".to_string(), eos);
        obj.insert("pad_token_id".to_string(), pad);

        // ---- PHASE 1: read from immutable cfg ----
        let tc_ro = cfg_val.get("text_config"); // immutable

        let bos_src = tc_ro.and_then(|tc| tc.get("bos_token_id"))
            .or(cfg_val.get("bos_token_id"))
            .or(gen_val_opt.as_ref().and_then(|v| v.get("bos_token_id")));
        let eos_src = tc_ro.and_then(|tc| tc.get("eos_token_id"))
            .or(cfg_val.get("eos_token_id"))
            .or(gen_val_opt.as_ref().and_then(|v| v.get("eos_token_id")));
        let pad_src = tc_ro.and_then(|tc| tc.get("pad_token_id"))
            .or(cfg_val.get("pad_token_id"))
            .or(gen_val_opt.as_ref().and_then(|v| v.get("pad_token_id")));
        let vocab_src = tc_ro
            .and_then(|tc| tc.get("vocab_size"))
            .or(cfg_val.get("vocab_size"));

        let (bos_val, _bos_num) = scalarize_id(bos_src, 128000); // Llama-3 BOS
        let (eos_val, eos_num) = scalarize_id(eos_src, 128001);  // Llama-3 EOS
        let (pad_val, _pad_num) = scalarize_id(pad_src, eos_num); // default PAD = EOS
        let vocab_val = vocab_src.cloned().unwrap_or(serde_json::json!(128256)); // Llama-3 default
        let vocab_num = vocab_val.as_u64().unwrap_or(128256) as usize;

        // ---- PHASE 2: write into mutable cfg (root + text_config) ----
        let root = cfg_val.as_object_mut().expect("config.json must be an object");
        let tc_val = root.entry("text_config".to_string()).or_insert(json!({}));
        let tc = tc_val.as_object_mut().expect("text_config must be an object");
        
        tc.insert("vocab_size".to_string(), vocab_val.clone());
        tc.insert("bos_token_id".to_string(), bos_val.clone());
        tc.insert("eos_token_id".to_string(), eos_val.clone());
        tc.insert("pad_token_id".to_string(), pad_val.clone());
        
        root.insert("vocab_size".to_string(), vocab_val);
        root.insert("bos_token_id".to_string(), bos_val);
        root.insert("eos_token_id".to_string(), eos_val);
        root.insert("pad_token_id".to_string(), pad_val);

        // ensure a dtype so later code doesn't panic (optional)
        root.entry("torch_dtype".to_string()).or_insert(json!("bfloat16"));
        root.entry("ignore_index".to_string())
            .or_insert(serde_json::json!(-100));  // PyTorch CrossEntropyLoss default
        root.entry("image_grid_pinpoints".to_string())
            .or_insert(serde_json::json!([]));
        root.entry("use_image_newline_parameter".to_string())
            .or_insert(serde_json::json!(false));
        // ---- ensure vision_config.projection_dim exists ----
        let vis_cfg_ro = cfg_val.get("vision_config");
        let hidden_sz = vis_cfg_ro
            .and_then(|v| v.get("hidden_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024); // conservative fallback if missing

        let hf_llava_config: HFLLaVAConfig = serde_json::from_value(cfg_val.clone())?;
        let has_proj = vis_cfg_ro
            .and_then(|v| v.get("projection_dim"))
            .is_some();

        {
            // mutate root
            let root_obj = cfg_val.as_object_mut().expect("config.json must be an object");
            let vc_entry = root_obj
                .entry("vision_config".to_string())
                .or_insert(serde_json::json!({}));
            let vc = vc_entry.as_object_mut().expect("vision_config must be an object");

            if !has_proj {
                vc.insert("projection_dim".to_string(), serde_json::json!(hidden_sz));
            }

            // <<< add this: also ensure vocab_size lives in vision_config >>>
            vc.entry("vocab_size".to_string())
                .or_insert(serde_json::json!(vocab_num));
        }

        // (Already parsed hf_llava_config above)
        // ----- normalize generation_config BEFORE deserializing -----
        let gen_norm_val = if let Some(mut gv) = gen_val_opt {
            if let Some(obj) = gv.as_object_mut() {
                // destructure the tuple to get the Value and (optionally) the numeric id
                let (bos_v, _bos_id) = scalarize_id(obj.get("bos_token_id"), 128000);
                let (eos_v, _eos_id) = scalarize_id(obj.get("eos_token_id"), 128001);

                let pad_v: Value = obj
                    .get("pad_token_id")
                    .cloned()
                    .unwrap_or_else(|| eos_v.clone()); // default pad = eos

                obj.insert("bos_token_id".to_string(), bos_v);
                obj.insert("eos_token_id".to_string(), eos_v.clone());
                obj.entry("pad_token_id".to_string()).or_insert(pad_v);
            }
            gv
        } else {
            serde_json::json!({
                "bos_token_id": 128000,
                "eos_token_id": 128001,
                "pad_token_id": 128001
            })
        };

        // Now safe to parse
        let generation_config: HFGenerationConfig = serde_json::from_value(gen_norm_val.clone())?;

        // ensure object
        if let Some(obj) = pre_val.as_object_mut() {
            // rename image_aspect_ratio → aspect_ratio_setting
            if let Some(aspect) = obj.remove("image_aspect_ratio") {
                obj.insert("aspect_ratio_setting".to_string(), aspect);
            } else {
                obj.insert("aspect_ratio_setting".to_string(), serde_json::json!("square"));
            }

            // add crop_size if missing
            if !obj.contains_key("crop_size") {
                obj.insert(
                    "crop_size".to_string(),
                    serde_json::json!({ "height": 384, "width": 384 }),
                );
            }

            // add do_center_crop if missing
            obj.entry("do_center_crop".to_string())
                .or_insert(serde_json::json!(false));

            // normalize "size": add shortest_edge if only height/width are present
            if let Some(size_val) = obj.get_mut("size") {
                if let Some(size_map) = size_val.as_object_mut() {
                    if !size_map.contains_key("shortest_edge") {
                        if let Some(h) = size_map.get("height").and_then(|v| v.as_i64()) {
                            size_map.insert("shortest_edge".to_string(), serde_json::json!(h));
                        } else {
                            size_map.insert("shortest_edge".to_string(), serde_json::json!(384));
                        }
                    }
                }
            } else {
                obj.insert(
                    "size".to_string(),
                    serde_json::json!({ "shortest_edge": 384 }),
                );
            }
        }


        
        let preprocessor_config: HFPreProcessorConfig = serde_json::from_value(pre_val)?;
        log::info!("preprocessor_config");
        let mut llava_config = hf_llava_config.to_llava_config("fancyfeast/llama-joycaption-beta-one-hf-llava", &generation_config, &preprocessor_config);
        let requested_dtype_str = llava_config.torch_dtype.clone();
        let mut effective_dtype = match requested_dtype_str.as_str() {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => DType::F32,
        };
        let device = candle_examples::device(if cfg!(feature="cpu") { true } else { false })?;
        log::error!("DEVICE: {device:?}");
        let is_cpu = device.is_cpu();
        if is_cpu {
            if matches!(effective_dtype, DType::BF16 | DType::F16) {
                log::info!("[dtype] CPU detected: falling back from {:?} to F32 for compatibility", effective_dtype);
                effective_dtype = DType::F32;
                llava_config.torch_dtype = "float32".to_string();
            }
        }
        let dtype = effective_dtype;
        let llama_config = llava_config.to_llama_config();
        log::info!("llava_config");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("Err: {e:?}"))?;
        log::info!("tokenizer");
        let clip_vision_config = hf_llava_config.to_clip_vision_config();
        log::info!("clip_vision_config");
        
        let mut temperature: f32 = TEMPERATURE;
        let mut top_p: f32 = 0.9;
        let mut top_k: Option<usize> = None;
        let mut repetition_penalty: Option<f32> = None;
        let device = candle_examples::device(is_cpu)?;

        let cache = Cache::new(true, dtype, &llama_config, &device)?;
        if let Some(obj) = gen_norm_val.as_object() {
            if let Some(t) = obj.get("temperature").and_then(|v| v.as_f64()) { temperature = t as f32; }
            if let Some(tp) = obj.get("top_p").and_then(|v| v.as_f64()) { top_p = tp as f32; }
            if let Some(tk) = obj.get("top_k").and_then(|v| v.as_u64()) { if tk > 0 { top_k = Some(tk as usize); } }
            if let Some(rp) = obj.get("repetition_penalty").and_then(|v| v.as_f64()) { let rp_f = rp as f32; if rp_f > 0.0 && (rp_f - 1.0).abs() > f32::EPSILON { repetition_penalty = Some(rp_f); } }
        }
        let weight_filenames = candle_examples::hub_load_local_safetensors(dir, "model.safetensors.index.json")?;
        let processor = preprocessor_config.to_clip_image_processor();
        VISION_STATUS.set_state(StatusState::Initializing, "Mapping weights");
        let mut vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, dtype, &device)? };
        // Global upcast on CPU if original requested bf16/f16 to avoid kernel unsupported ops.
        if matches!(device, candle_core::Device::Cpu) && matches!(dtype, DType::BF16 | DType::F16) {
            log::info!("[dtype] Upcasting all parameters to F32 for CPU execution");
            // Re-create VarBuilder with F32 target (re-mapping underlying storage lazily)
            vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, DType::F32, &device)? };
        }
        VISION_STATUS.set_state(StatusState::Initializing, "Building model graph");
        let llava: LLaVA = LLaVA::load(vb, &llava_config, Some(clip_vision_config))?;
        if temperature < 0.0 { temperature = 0.5; }
        log::info!("[joycaption] sampling defaults: temperature={:.3} top_p={:.3} top_k={:?} repetition_penalty={:?}", temperature, top_p, top_k, repetition_penalty);
        let eos_id_usize = llava_config.eos_token_id;
        VISION_STATUS.set_state(StatusState::Idle, "Ready");
        Ok(Self { llava, tokenizer, processor, llava_config, cache, eos_token_id: eos_id_usize, max_new_tokens: MAX_NEW_TOKENS, temperature, _top_p: top_p, _top_k: top_k, _repetition_penalty: repetition_penalty, device })
    }
}