#![cfg(feature = "joycaption")]
#![allow(unused)]
use candle_core::{DType, Tensor, Device, IndexOp};
use crate::ai::generate::VisionDescription;
use crate::ai::candle_llava::load_image;
use crate::app::{DEFAULT_JOYCAPTION_PATH, MAX_NEW_TOKENS, TEMPERATURE};
use tokio::sync::{mpsc, oneshot};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use once_cell::sync::OnceCell;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::thread;
use super::candle_llava::{
    config::{HFLLaVAConfig, HFGenerationConfig, HFPreProcessorConfig, LLaVAConfig},
    model::LLaVA,
    conversation::Conversation,
    utils::{tokenizer_image_token},
    clip_image_processor::CLIPImageProcessor,
    llama::Cache,
};

// Worker handle & message definitions
struct WorkerHandle { tx: mpsc::UnboundedSender<WorkMsg> }

enum WorkMsg {
    Describe { path: PathBuf, reply: oneshot::Sender<anyhow::Result<VisionDescription>> },
    // Unified streaming variant: caller must supply instruction text.
    StreamDescribeBytes { bytes: Vec<u8>, instruction: String, token_tx: mpsc::UnboundedSender<String>, done: oneshot::Sender<anyhow::Result<String>> },
}

static WORKER: OnceCell<WorkerHandle> = OnceCell::new();

async fn ensure_worker_started() -> Result<&'static WorkerHandle> {
    if let Some(h) = WORKER.get() { return Ok(h); }
    WORKER.get_or_try_init(|| {
        let dir_env = std::env::var("JOYCAPTION_MODEL_DIR").ok();
        log::info!("JOYCAPTION_MODEL_DIR: {dir_env:?}");
        let candidate = dir_env.as_deref().unwrap_or(DEFAULT_JOYCAPTION_PATH);
        log::info!("candidate: {candidate}");
        let model_dir = PathBuf::from(candidate);
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkMsg>();
        thread::spawn(move || {
            let model = match JoyCaptionModel::load_from_dir(&model_dir) { Ok(m)=>m, Err(e)=>{ log::error!("[joycaption] failed to load model: {e}"); return; } };
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    WorkMsg::Describe { path, reply } => {
                        let res = model.describe_image(&path);
                        let _ = reply.send(res.map_err(|e| e.into()));
                    }
                    WorkMsg::StreamDescribeBytes { bytes, instruction, token_tx, done } => {
                        let (prompt, _query) = model.build_prompt(&instruction);
                        match model.stream_generate_bytes(&prompt, &bytes, |tok| { let _ = token_tx.send(tok.to_string()); }) {
                            Ok(full) => { let _ = done.send(Ok(full)); },
                            Err(e) => { let _ = done.send(Err(e)); },
                        }
                    }
                }
            }
        });
        Ok(WorkerHandle { tx })
    })
}

pub fn is_enabled() -> bool {
    let env_path_ok = std::env::var("JOYCAPTION_MODEL_DIR").ok().map(|p| Path::new(&p).exists()).unwrap_or(false);
    env_path_ok || Path::new(DEFAULT_JOYCAPTION_PATH).exists()
}

pub async fn ensure_loaded() -> Result<()> { let _ = ensure_worker_started().await?; Ok(()) }

pub async fn describe_image(image_path: &Path) -> Result<VisionDescription> {
    let worker = ensure_worker_started().await?;
    let (reply_tx, reply_rx) = oneshot::channel();
    worker.tx.send(WorkMsg::Describe { path: image_path.to_path_buf(), reply: reply_tx }).map_err(|e| anyhow::anyhow!("worker send failed: {e}"))?;
    reply_rx.await.map_err(|e| anyhow::anyhow!("worker dropped: {e}"))?
}

/// Stream describe raw image bytes with an instruction using the background worker.
/// Returns the full generated string (caller may parse JSON from it). Falls back with
/// an error if the worker/model isn't loaded.
pub async fn stream_describe_bytes(bytes: Vec<u8>, instruction: &str) -> Result<String> {
    let worker = ensure_worker_started().await?;
    use tokio::sync::mpsc as tmpsc;
    let (token_tx, mut token_rx) = tmpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = oneshot::channel();
    worker.tx.send(WorkMsg::StreamDescribeBytes { bytes, instruction: instruction.to_string(), token_tx, done: done_tx })
        .map_err(|e| anyhow::anyhow!("worker send failed: {e}"))?;
    // Collect tokens concurrently while awaiting final result.
    let mut collected = String::new();
    while let Ok(Some(tok)) = tokio::time::timeout(std::time::Duration::from_millis(5), token_rx.recv()).await {
        collected.push_str(&tok);
    }
    match done_rx.await {
        Ok(Ok(full)) => Ok(full),
        Ok(Err(e)) => {
            // If streaming path failed but we collected partial tokens, return them for diagnostic.
            if !collected.is_empty() { Ok(collected) } else { Err(e) }
        }
        Err(e) => Err(anyhow::anyhow!("worker dropped: {e}")),
    }
}

/// Streaming variant that surfaces each newly generated token (or fragment) via the provided callback.
/// The callback receives each incremental fragment exactly as produced (not the full accumulated text).
/// Returns the final full generated string (either the model's final decode or the accumulated text if
/// an error occurs after partial output).
pub async fn stream_describe_bytes_with_callback<F>(bytes: Vec<u8>, instruction: &str, mut on_token: F) -> Result<String>
where
    F: FnMut(&str),
{
    let worker = ensure_worker_started().await?;
    use tokio::sync::mpsc as tmpsc;
    let (token_tx, mut token_rx) = tmpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .tx
        .send(WorkMsg::StreamDescribeBytes { bytes, instruction: instruction.to_string(), token_tx, done: done_tx })
        .map_err(|e| anyhow::anyhow!("worker send failed: {e}"))?;
    let mut collected = String::new();
    // Race token stream vs completion; once done resolves, break and drain residual tokens.
    let mut done_rx = done_rx; // mutable so we can poll by reference
    let mut final_result: Option<anyhow::Result<String>> = None;
    // Main loop until done fires
    while final_result.is_none() {
        tokio::select! {
            maybe_tok = token_rx.recv() => {
                if let Some(tok) = maybe_tok {
                    collected.push_str(&tok);
                    on_token(&tok);
                } else {
                    // token channel closed; keep waiting for done (or break if done already captured)
                }
            }
            done_res = &mut done_rx => {
                final_result = Some(done_res.map_err(|e| anyhow::anyhow!("worker dropped: {e}")).and_then(|inner| inner));
            }
        }
    }
    // Drain any tokens that slipped in after done firing but before channel closure
    while let Ok(tok) = token_rx.try_recv() {
        collected.push_str(&tok);
        on_token(&tok);
    }
    match final_result.unwrap() {
        Ok(full) => Ok(full),
        Err(e) => if collected.is_empty() { Err(e) } else { Ok(collected) }
    }
}

struct JoyCaptionModel {
    llava: LLaVA,
    tokenizer: Tokenizer,
    processor: CLIPImageProcessor,
    llava_config: LLaVAConfig,
    cache: Cache,
    eos_token_id: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    device: Device,
}

impl JoyCaptionModel {
    fn load_from_dir(dir: &Path) -> Result<Self> {
    use crate::ui::status::{JOY_STATUS, StatusState, GlobalStatusIndicator};
    JOY_STATUS.set_state(StatusState::Initializing, "Reading config");
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
        let device = candle_examples::device(false)?;

        let cache = Cache::new(true, dtype, &llama_config, &device)?;
        if let Some(obj) = gen_norm_val.as_object() {
            if let Some(t) = obj.get("temperature").and_then(|v| v.as_f64()) { temperature = t as f32; }
            if let Some(tp) = obj.get("top_p").and_then(|v| v.as_f64()) { top_p = tp as f32; }
            if let Some(tk) = obj.get("top_k").and_then(|v| v.as_u64()) { if tk > 0 { top_k = Some(tk as usize); } }
            if let Some(rp) = obj.get("repetition_penalty").and_then(|v| v.as_f64()) { let rp_f = rp as f32; if rp_f > 0.0 && (rp_f - 1.0).abs() > f32::EPSILON { repetition_penalty = Some(rp_f); } }
        }
        let weight_filenames = candle_examples::hub_load_local_safetensors(dir, "model.safetensors.index.json")?;
        let processor = preprocessor_config.to_clip_image_processor();
    JOY_STATUS.set_state(StatusState::Initializing, "Mapping weights");
        let mut vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, dtype, &device)? };
        // Global upcast on CPU if original requested bf16/f16 to avoid kernel unsupported ops.
        if matches!(device, candle_core::Device::Cpu) && matches!(dtype, DType::BF16 | DType::F16) {
            log::info!("[dtype] Upcasting all parameters to F32 for CPU execution");
            // Re-create VarBuilder with F32 target (re-mapping underlying storage lazily)
            vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, DType::F32, &device)? };
        }
    JOY_STATUS.set_state(StatusState::Initializing, "Building model graph");
        let llava: LLaVA = LLaVA::load(vb, &llava_config, Some(clip_vision_config))?;
        if temperature < 0.0 { temperature = 0.5; }
        log::info!("[joycaption] sampling defaults: temperature={:.3} top_p={:.3} top_k={:?} repetition_penalty={:?}", temperature, top_p, top_k, repetition_penalty);
        let eos_id_usize = llava_config.eos_token_id;
    JOY_STATUS.set_state(StatusState::Idle, "Ready");
        Ok(Self { llava, tokenizer, processor, llava_config, cache, eos_token_id: eos_id_usize, max_new_tokens: MAX_NEW_TOKENS, temperature, top_p, top_k, repetition_penalty, device })
    }

    fn build_prompt(&self, user_prompt: &str) -> (String, String) {
        let mut conv = Conversation::conv_llava_v0();
        let query = format!("<image>\n{}", user_prompt);
        conv.append_user_message(Some(&query));
        conv.append_assistant_message(None);
        let prompt = conv.get_prompt();
        (prompt, query)
    }

    fn run_generation(&self, prompt: &str, image_path: &Path) -> Result<String> {
        let img = image::ImageReader::open(image_path)?.decode()?;
        self.run_generation_from_image(prompt, img)
    }

    fn run_generation_bytes(&self, prompt: &str, bytes: &[u8]) -> Result<String> {
        let img = image::load_from_memory(bytes)?;
        self.run_generation_from_image(prompt, img)
    }

    fn run_generation_from_image(&self, prompt: &str, img: image::DynamicImage) -> Result<String> {
        let requested_dtype_str = self.llava_config.torch_dtype.clone();
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
            }
        }
        let dtype = effective_dtype;
        let ((w, h), image_tensor) = load_image(&img, &self.processor, &self.llava_config, dtype)?;
        log::info!("[joycaption.gen] start size={}x{} prompt_len={}", w, h, prompt.len());
        let img_tensor = image_tensor.to_device(&self.device)?;
        log::error!("img_tensor: {:?}", img_tensor.dtype());
        let tokens = tokenizer_image_token(
            prompt,
            &self.tokenizer,
            self.llava_config.image_token_index as i64,
            &self.llava_config,
        )?;
        log::info!("[joycaption.gen] temperature={:.3}", self.temperature);
        let input_embeds = self.llava.prepare_inputs_labels_for_multimodal(&tokens, &[img_tensor], &[(w,h)])?;
        log::info!("[joycaption.gen] prepare_inputs_labels_for_multimodal");
        use candle_transformers::generation::{Sampling, LogitsProcessor};
        // Currently candle's LogitsProcessor::from_sampling supports temperature / argmax. top_p/top_k not wired yet here.
        let temperature = f64::from(self.temperature);
        let sampling = if temperature <= 0.0 { Sampling::ArgMax } else { Sampling::All { temperature } };
        
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);
        let mut cache = self.cache.clone();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut idx_pos = 0usize;
        let mut embeds = input_embeds.clone();
        const LOG_INTERVAL: usize = 8;
        log::info!("[joycaption.gen] running steps: {}", self.max_new_tokens);
        for step in 0..self.max_new_tokens {
            let (_, total_len, _) = embeds.dims3()?;
            let (context_size, context_index) = if cache.use_kv_cache && step > 0 { (1, idx_pos) } else { (total_len, 0) };
            let input = embeds.i((.., total_len - context_size.., ..))?;
            let logits = self.llava.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let (_, input_len, _) = input.dims3()?; idx_pos += input_len;
            let next_token = logits_processor.sample(&logits)?;
            if next_token as usize == self.eos_token_id { log::info!("[joycaption.gen] eos step={} total_tokens={}", step, token_ids.len()); break; }
            let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &self.device)?;
            let next_embeds = self.llava.llama.embed(&next_token_tensor)?.unsqueeze(0)?;
            embeds = Tensor::cat(&[embeds, next_embeds], 1)?;
            token_ids.push(next_token);
            if step % LOG_INTERVAL == 0 { log::info!("[joycaption.gen] step={} generated_tokens={} last_token={}", step, token_ids.len(), next_token); }
        }
        let text = if token_ids.is_empty() { String::new() } else { self.tokenizer.decode(&token_ids, true).unwrap_or_default() };
        log::info!("[joycaption.gen] done chars={} tokens={}", text.len(), token_ids.len());
        Ok(text)
    }

    fn stream_generate_bytes(&self, prompt: &str, bytes: &[u8], on_token: impl FnMut(&str)) -> Result<String> {
        let img = image::load_from_memory(bytes)?;
        self.stream_generate_from_image(prompt, img, on_token)
    }

    fn stream_generate_from_image(&self, prompt: &str, img: image::DynamicImage, mut on_token: impl FnMut(&str)) -> Result<String> {
    use crate::ui::status::{JOY_STATUS, StatusState, GlobalStatusIndicator, VISION_TOKENS, VISION_MAX_TOKENS};
        let requested_dtype_str = self.llava_config.torch_dtype.clone();
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
            }
        }
        let dtype = effective_dtype;
        let ((w, h), image_tensor) = load_image(&img, &self.processor, &self.llava_config, dtype)?;
        log::info!("[joycaption.gen] start size={}x{} prompt_len={}", w, h, prompt.len());
        let img_tensor = image_tensor.to_device(&self.device)?;
        log::error!("img_tensor: {:?}", img_tensor.dtype());
        let tokens = tokenizer_image_token(
            prompt,
            &self.tokenizer,
            self.llava_config.image_token_index as i64,
            &self.llava_config,
        )?;
        log::info!("[joycaption.gen] temperature={:.3}", self.temperature);
    JOY_STATUS.set_state(StatusState::Running, format!("Generating ({w}x{h})"));
    JOY_STATUS.set_progress(0, self.max_new_tokens as u64);
    VISION_TOKENS.store(0, std::sync::atomic::Ordering::Relaxed);
    VISION_MAX_TOKENS.store(self.max_new_tokens, std::sync::atomic::Ordering::Relaxed);
        let input_embeds = self.llava.prepare_inputs_labels_for_multimodal(&tokens, &[img_tensor], &[(w,h)])?;
        use candle_transformers::generation::{Sampling, LogitsProcessor};
        let temperature = f64::from(self.temperature);
        let sampling = if temperature <= 0.0 { Sampling::ArgMax } else { Sampling::All { temperature } };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);
        let mut cache = self.cache.clone();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut idx_pos = 0usize;
        let mut embeds = input_embeds.clone();
        let mut last_decoded_len = 0usize;
        const STREAM_LOG_INTERVAL: usize = 8;
        for step in 0..self.max_new_tokens {
            let (_, total_len, _) = embeds.dims3()?;
            let (context_size, context_index) = if cache.use_kv_cache && step > 0 { (1, idx_pos) } else { (total_len, 0) };
            let input = embeds.i((.., total_len - context_size.., ..))?;
            let logits = self.llava.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let (_, input_len, _) = input.dims3()?; idx_pos += input_len;
            let next_token = logits_processor.sample(&logits)?;
            if next_token as usize == self.eos_token_id { log::debug!("[joycaption.stream] eos step={} total_tokens={}", step, token_ids.len()); break; }
            let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &self.device)?;
            let next_embeds = self.llava.llama.embed(&next_token_tensor)?.unsqueeze(0)?;
            embeds = Tensor::cat(&[embeds, next_embeds], 1)?;
            token_ids.push(next_token);
            // Progress update (token based)
            let produced = token_ids.len() as u64;
            VISION_TOKENS.store(produced as usize, std::sync::atomic::Ordering::Relaxed);
            JOY_STATUS.set_progress(produced, self.max_new_tokens as u64);
            // Attempt ultra-incremental decode: decode just the last token id alone to guess its text.
            // Some tokenizers may require full context to merge bytes properly; fallback to full decode diff if needed.
            let mut emitted_this_step = false;
            if let Ok(last_piece) = self.tokenizer.decode(&[next_token], true) {
                if !last_piece.is_empty() {
                    on_token(&last_piece);
                    emitted_this_step = true;
                }
            }
            if !emitted_this_step {
                // Fallback: full decode diff (previous behavior)
                if let Ok(full_so_far) = self.tokenizer.decode(&token_ids, true) {
                    if full_so_far.len() > last_decoded_len {
                        let new_part = &full_so_far[last_decoded_len..];
                        if !new_part.is_empty() { on_token(new_part); }
                        last_decoded_len = full_so_far.len();
                    }
                }
            } else {
                // Maintain last_decoded_len by full length occasionally to keep diff logic consistent
                if step % STREAM_LOG_INTERVAL == 0 {
                    if let Ok(full_so_far) = self.tokenizer.decode(&token_ids, true) { last_decoded_len = full_so_far.len(); }
                }
            }
            if step % STREAM_LOG_INTERVAL == 0 { log::info!("[joycaption.stream] step={} generated_tokens={} last_token={}", step, token_ids.len(), next_token); }
        }
        let text = if token_ids.is_empty() { String::new() } else { self.tokenizer.decode(&token_ids, true).unwrap_or_default() };
        log::info!("[joycaption.stream] done chars={} tokens={}\nText: {}", text.len(), token_ids.len(), text);
        JOY_STATUS.set_state(StatusState::Idle, format!("Generated {} tokens", token_ids.len()));
        Ok(text)
    }

    fn describe_image(&self, image_path: &Path) -> Result<VisionDescription> {
        let instruction = "Analyze the image and produce concise JSON with keys: description (detailed multi-sentence), caption (short), tags (array of lowercase single-word nouns), category (single general category). Return ONLY JSON.";
        let (prompt, _query) = self.build_prompt(instruction);
        let raw = self.run_generation(&prompt, image_path)?;
        let vd = extract_json_vision(&raw)
            .and_then(|v| serde_json::from_value::<VisionDescription>(v).ok())
            .unwrap_or_else(|| VisionDescription { description: raw.trim().to_string(), caption: raw.split('.').next().unwrap_or(&raw).trim().to_string(), tags: Vec::new(), category: "general".into() });
        Ok(vd)
    }
}

pub fn extract_json_vision(raw: &str) -> Option<Value> {
    let bytes = raw.as_bytes();
    let start = bytes.iter().position(|b| *b == b'{')?;
    let end = bytes.iter().rposition(|b| *b == b'}')?;
    if end <= start { return None; }
    let slice = &raw[start..=end];
    serde_json::from_str::<Value>(slice).ok()
}

