use crate::ai::candle_llava::{
    clip_image_processor::CLIPImageProcessor, config::LLaVAConfig, conversation::Conversation,
    llama::Cache, model::LLaVA,
};
use crate::ui::status::GlobalStatusIndicator;
use candle_core::Device;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use once_cell::sync::OnceCell;

pub mod generate;
pub mod load;
pub mod stream;

// Optional moondream fallbacks when LLaVA fails (e.g., CUDA OOM)
struct MoondreamState {
    model: candle_transformers::models::quantized_moondream::Model,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
}

static MOONDREAM_Q_FALLBACK: OnceCell<std::sync::Mutex<MoondreamState>> = OnceCell::new();

fn moondream_quantized_fallback(bytes: &[u8], instruction: &str) -> anyhow::Result<String> {
    use candle_core::{DType, Tensor};
    use candle_transformers::{
        generation::LogitsProcessor,
        models::{moondream, quantized_moondream},
    };
    use tokenizers::Tokenizer;
    use std::path::PathBuf;
    // Resolve local files via env or download from HF hub if missing
    let gguf_path: PathBuf = if let Some(p) = std::env::var("MOONDREAM_GGUF").ok()
        .or_else(|| std::env::var("MOONDREAM_GGUF_PATH").ok())
    {
        PathBuf::from(p)
    } else {
        // Fallback to HF: santiagomed/candle-moondream model-q4_0.gguf
        let repo = crate::ai::hf::hf_model("santiagomed/candle-moondream")?;
        crate::ai::hf::hf_get_file(&repo, "model-q4_0.gguf")?
    };
    let tok_path: PathBuf = if let Some(p) = std::env::var("MOONDREAM_TOKENIZER").ok()
        .or_else(|| std::env::var("MOONDREAM_TOKENIZER_PATH").ok())
    {
        PathBuf::from(p)
    } else {
        let repo = crate::ai::hf::hf_model("santiagomed/candle-moondream")?;
        crate::ai::hf::hf_get_file(&repo, "tokenizer.json")?
    };
    if !gguf_path.exists() {
        anyhow::bail!("Moondream gguf not found at {}", gguf_path.display());
    }
    if !tok_path.exists() {
        anyhow::bail!("Moondream tokenizer.json not found at {}", tok_path.display());
    }
    // Acquire or initialize cached model
    let lock = MOONDREAM_Q_FALLBACK.get_or_try_init(|| {
        // Initialize
        let device = candle_examples::device(if cfg!(feature="cpu") { true } else { false })?;
        let config = moondream::Config::v2();
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&gguf_path, &device)?;
        let model = quantized_moondream::Model::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
        Ok::<_, anyhow::Error>(std::sync::Mutex::new(MoondreamState { model, tokenizer, device }))
    })?;
    let mut guard = lock.lock().map_err(|_| anyhow::anyhow!("Moondream fallback poisoned"))?;
    let device = guard.device.clone();
    // Image preproc (378x378 normalized as in examples)
    let img = image::load_from_memory(bytes)?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), &device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], &device)?.reshape((3, 1, 1))?;
    let image = (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?
        .to_device(&device)?;
    let image_embeds = image.unsqueeze(0)?;
    let image_embeds = image_embeds.apply(guard.model.vision_encoder())?;
    // Stronger JSON-only prompt for reliability
    let prompt = format!(
        concat!(
            "You are a vision assistant. Return ONLY a valid JSON object with these keys: ",
            "description (string, multi-sentence), caption (string, short), ",
            "tags (array of lowercase single-word nouns), category (string).\n",
            "Do not include any text outside the JSON.\n",
            "Question: {}\n\nAnswer:"
        ),
        instruction
    );
    // Generation loop
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.5), Some(0.95));
    let repeat_penalty: f32 = 1.1;
    let repeat_last_n: usize = 96;
    // Tokenization
    let mut tokens = guard
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    if tokens.is_empty() { anyhow::bail!("Empty prompt for Moondream fallback"); }
    let special_token = guard.tokenizer.get_vocab(true).get("<|endoftext|>").copied()
        .ok_or_else(|| anyhow::anyhow!("Moondream tokenizer missing <|endoftext|>"))?;
    let (bos_token, eos_token) = (special_token, special_token);
    let sample_len = 512usize;
    let mut generated = String::new();
    let mut started_json = false;
    let mut brace_depth: i32 = 0;
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
    let ctxt: Vec<u32> = tokens[tokens.len().saturating_sub(context_size)..].to_vec();
    let input = Tensor::new(ctxt.as_slice(), &device)?.unsqueeze(0)?;
        let logits = if index > 0 {
            guard.model.text_model.forward(&input)?
        } else {
            let bos = Tensor::new(&[bos_token], &device)?.unsqueeze(0)?;
            guard.model.text_model.forward_with_img(&bos, &input, &image_embeds)?
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        // Apply repeat penalty on last N tokens
        let logits = if repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if next_token == eos_token || tokens.ends_with(&[27, 10619, 29]) { break; }
        let tok = guard.tokenizer.decode(&[next_token], true).map_err(anyhow::Error::msg)?;
        generated.push_str(&tok);
        // Early stop when a single top-level JSON object is complete
        for ch in tok.chars() {
            if ch == '{' { started_json = true; brace_depth += 1; }
            if ch == '}' { brace_depth -= 1; }
        }
        if started_json && brace_depth <= 0 && generated.contains('{') && generated.trim_end().ends_with('}') {
            break;
        }
    }
    Ok(generated)
}

// Unquantized fallback (higher quality). Uses moondream1 safetensors.
struct MoondreamFullState {
    model: candle_transformers::models::moondream::Model,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

static MOONDREAM_F32_FALLBACK: OnceCell<std::sync::Mutex<MoondreamFullState>> = OnceCell::new();

fn moondream_unquantized_fallback(bytes: &[u8], instruction: &str) -> anyhow::Result<String> {
    use candle_core::{DType, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::{generation::LogitsProcessor, models::moondream};
    use tokenizers::Tokenizer;
    use std::path::PathBuf;
    // Resolve files via HF cache (no env overrides)
    let (model_path, tok_path): (PathBuf, PathBuf) = {
        let repo = crate::ai::hf::hf_model("vikhyatk/moondream1")?;
        (crate::ai::hf::hf_get_file(&repo, "model.safetensors")?, crate::ai::hf::hf_get_file(&repo, "tokenizer.json")?)
    };
    if !model_path.exists() { anyhow::bail!("moondream model.safetensors not found at {}", model_path.display()); }
    if !tok_path.exists() { anyhow::bail!("moondream tokenizer.json not found at {}", tok_path.display()); }

    // Acquire or initialize cached model
    let lock = MOONDREAM_F32_FALLBACK.get_or_try_init(|| {
        let device = candle_examples::device(if cfg!(feature="cpu") { true } else { false })?;
        // dtype: GPU F16 else CPU F32
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let config = moondream::Config::v2();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&model_path), dtype, &device)? };
        let model = moondream::Model::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
        Ok::<_, anyhow::Error>(std::sync::Mutex::new(MoondreamFullState { model, tokenizer, device, dtype }))
    })?;
    let mut guard = lock.lock().map_err(|_| anyhow::anyhow!("Moondream fallback poisoned"))?;
    // Image preprocessing
    let img = image::load_from_memory(bytes)?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), &candle_core::Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
    let image = (data.to_dtype(DType::F32)? / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?.to_device(&guard.device)?.to_dtype(guard.dtype)?;
    let image_embeds = image.unsqueeze(0)?;
    let image_embeds = image_embeds.apply(guard.model.vision_encoder())?;
    // Strong prompt & sampling
    let prompt = format!(
        concat!(
            "You are a vision assistant. Return ONLY a valid JSON object with these keys: ",
            "description (string, multi-sentence), caption (string, short), ",
            "tags (array of lowercase single-word nouns), category (string).\n",
            "Do not include any text outside the JSON.\n",
            "Question: {}\n\nAnswer:"
        ),
        instruction
    );
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.5), Some(0.95));
    let repeat_penalty: f32 = 1.1;
    let repeat_last_n: usize = 96;
    // Tokenization
    let mut tokens = guard.tokenizer.encode(prompt.as_str(), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    if tokens.is_empty() { anyhow::bail!("Empty prompt for Moondream fallback"); }
    let special_token = guard.tokenizer.get_vocab(true).get("<|endoftext|>").copied().ok_or_else(|| anyhow::anyhow!("Moondream tokenizer missing <|endoftext|>"))?;
    let (bos_token, eos_token) = (special_token, special_token);
    let sample_len = 640usize;
    let mut generated = String::new();
    let mut started_json = false;
    let mut brace_depth: i32 = 0;
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt: Vec<u32> = tokens[tokens.len().saturating_sub(context_size)..].to_vec();
        let input = Tensor::new(ctxt.as_slice(), &guard.device)?.unsqueeze(0)?;
        let logits = if index > 0 {
            guard.model.text_model.forward(&input)?
        } else {
            let bos = Tensor::new(&[bos_token], &guard.device)?.unsqueeze(0)?;
            guard.model.text_model.forward_with_img(&bos, &input, &image_embeds)?
        };
        let logits = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        let logits = if repeat_penalty == 1.0 { logits } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(&logits, repeat_penalty, &tokens[start_at..])?
        };
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if next_token == eos_token || tokens.ends_with(&[27, 10619, 29]) { break; }
        let tok = guard.tokenizer.decode(&[next_token], true).map_err(anyhow::Error::msg)?;
        generated.push_str(&tok);
        for ch in tok.chars() { if ch == '{' { started_json = true; brace_depth += 1; } if ch == '}' { brace_depth -= 1; } }
        if started_json && brace_depth <= 0 && generated.contains('{') && generated.trim_end().ends_with('}') { break; }
    }
    Ok(generated)
}

fn moondream_fallback(bytes: &[u8], instruction: &str) -> anyhow::Result<String> {
    // Prefer higher quality unquantized; fall back to quantized if it fails to load or run.
    match moondream_unquantized_fallback(bytes, instruction) {
        Ok(s) => Ok(s),
        Err(e1) => {
            log::warn!("Unquantized Moondream failed, falling back to quantized: {e1}");
            moondream_quantized_fallback(bytes, instruction)
        }
    }
}

pub struct JoyCaptionModel {
    llava: LLaVA,
    tokenizer: Tokenizer,
    processor: CLIPImageProcessor,
    llava_config: LLaVAConfig,
    cache: Cache,
    eos_token_id: usize,
    max_new_tokens: usize,
    temperature: f32,
    _top_p: f32,
    _top_k: Option<usize>,
    _repetition_penalty: Option<f32>,
    device: Device,
}

impl JoyCaptionModel {
    fn build_prompt(&self, user_prompt: &str) -> (String, String) {
        let mut conv = Conversation::conv_llava_v0();
        let query = format!("<image>\n{}", user_prompt);
        conv.append_user_message(Some(&query));
        conv.append_assistant_message(None);
        let prompt = conv.get_prompt();
        (prompt, query)
    }
}

// Worker handle & message definitions
pub struct WorkerHandle {
    tx: mpsc::UnboundedSender<WorkMsg>,
}

enum WorkMsg {
    Describe {
        path: PathBuf,
        reply: oneshot::Sender<anyhow::Result<crate::ai::VisionDescription>>,
    },
    // Unified streaming variant: caller must supply instruction text.
    StreamDescribeBytes {
        bytes: Vec<u8>,
        instruction: String,
        token_tx: mpsc::UnboundedSender<String>,
        done: oneshot::Sender<anyhow::Result<String>>,
    },
    Shutdown,
}

static WORKER: OnceCell<WorkerHandle> = OnceCell::new();

pub async fn ensure_worker_started() -> anyhow::Result<&'static WorkerHandle> {
    if let Some(h) = WORKER.get() {
        return Ok(h);
    }
    WORKER.get_or_try_init(|| {
        // Prefer user setting if available; fall back to default constant
        let candidate = crate::database::settings::load_settings()
            .and_then(|s| s.joycaption_model_dir)
            .unwrap_or_else(|| crate::app::DEFAULT_JOYCAPTION_PATH.to_string());
        log::info!("[joycap] model dir: {candidate}");
        // Surface the selected model path in the status hover UI
        crate::ui::status::VISION_STATUS.set_model(&candidate);
        let model_dir = PathBuf::from(candidate);
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkMsg>();
        std::thread::spawn(move || {
            let model = match JoyCaptionModel::load_from_dir(&model_dir) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("[joycaption] failed to load model: {e}");
                    return;
                }
            };
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    WorkMsg::Describe { path, reply } => {
                        let res = model.describe_image(&path);
                        let _ = reply.send(res.map_err(|e| e.into()));
                    }
                    WorkMsg::StreamDescribeBytes {
                        bytes,
                        instruction,
                        token_tx,
                        done,
                    } => {
                        let (prompt, _query) = model.build_prompt(&instruction);
                        match model.stream_generate_bytes(&prompt, &bytes, |tok| {
                            let _ = token_tx.send(tok.to_string());
                        }) {
                            Ok(full) => {
                                let _ = done.send(Ok(full));
                            }
                            Err(e) => {
                                let _ = done.send(Err(e));
                            }
                        }
                    }
                    WorkMsg::Shutdown => {
                        log::info!("[joycaption] worker received Shutdown; exiting thread and dropping model");
                        break;
                    }
                }
            }
        });
        Ok(WorkerHandle { tx })
    })
}

/// Attempt to unload/stop the JoyCaption worker. This replaces the worker handle with None,
/// causing ensure_worker_started() to reconstruct it on next use. Existing background thread
/// will exit when channel drops.
pub async fn stop_worker() {
    if let Some(h) = WORKER.get() {
        let _ = h.tx.send(WorkMsg::Shutdown);
        crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Idle, "Unloaded");
    }
}

/// Stream describe raw image bytes with an instruction using the background worker.
/// Returns the full generated string (caller may parse JSON from it). Falls back with
/// an error if the worker/model isn't loaded.
pub async fn stream_describe_bytes(bytes: Vec<u8>, instruction: &str) -> anyhow::Result<String> {
    let worker = ensure_worker_started().await?;
    use tokio::sync::mpsc as tmpsc;
    let (token_tx, mut token_rx) = tmpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = oneshot::channel();
    let bytes_for_worker = bytes.clone();
    worker
        .tx
        .send(WorkMsg::StreamDescribeBytes {
            bytes: bytes_for_worker,
            instruction: instruction.to_string(),
            token_tx,
            done: done_tx,
        })
        .map_err(|e| anyhow::anyhow!("worker send failed: {e}"))?;
    // Collect tokens concurrently while awaiting final anyhow::Result.
    let mut collected = String::new();
    while let Ok(Some(tok)) =
        tokio::time::timeout(std::time::Duration::from_millis(5), token_rx.recv()).await
    {
        collected.push_str(&tok);
    }
    match done_rx.await {
        Ok(Ok(full)) => Ok(full),
        Ok(Err(e)) => {
            // If streaming path failed but we collected partial tokens, return them for diagnostic.
            if !collected.is_empty() {
                Ok(collected)
            } else {
                // Try fallback Moondream if available
                crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Fallback: Moondream");
                match moondream_fallback(&bytes, instruction) {
                    Ok(text) => {
                        crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Idle, "Fallback done");
                        Ok(text)
                    },
                    Err(_fe) => Err(e),
                }
            }
        }
        Err(e) => {
            // Worker channel closed: attempt fallback
            crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Fallback: Moondream");
            match moondream_fallback(&bytes, instruction) {
                Ok(text) => {
                    crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Idle, "Fallback done");
                    Ok(text)
                },
                Err(_fe) => Err(anyhow::anyhow!("worker dropped: {e}")),
            }
        }
    }
}

/// Streaming variant that surfaces each newly generated token (or fragment) via the provided callback.
/// The callback receives each incremental fragment exactly as produced (not the full accumulated text).
/// Returns the final full generated string (either the model's final decode or the accumulated text if
/// an error occurs after partial output).
pub async fn stream_describe_bytes_with_callback<F>(
    bytes: Vec<u8>,
    instruction: &str,
    mut on_token: F,
) -> anyhow::Result<String>
where
    F: FnMut(&str),
{
    let worker = ensure_worker_started().await?;
    use tokio::sync::mpsc as tmpsc;
    let (token_tx, mut token_rx) = tmpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = oneshot::channel();
    let bytes_for_worker = bytes.clone();
    worker
        .tx
        .send(WorkMsg::StreamDescribeBytes {
            bytes: bytes_for_worker,
            instruction: instruction.to_string(),
            token_tx,
            done: done_tx,
        })
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
        Err(e) => {
            if collected.is_empty() {
                // Attempt fallback and emit as a single chunk
                crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Fallback: Moondream");
                match moondream_fallback(&bytes, instruction) {
                    Ok(text) => { crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Idle, "Fallback done"); on_token(&text); Ok(text) }
                    Err(_fe) => Err(e),
                }
            } else {
                Ok(collected)
            }
        }
    }
}

pub fn extract_json_vision(raw: &str) -> Option<serde_json::Value> {
    let bytes = raw.as_bytes();
    let start = bytes.iter().position(|b| *b == b'{')?;
    let end = bytes.iter().rposition(|b| *b == b'}')?;
    if end <= start {
        return None;
    }
    let slice = &raw[start..=end];
    serde_json::from_str::<serde_json::Value>(slice).ok()
}

pub async fn describe_image(
    image_path: &std::path::Path,
) -> anyhow::Result<crate::ai::VisionDescription> {
    let worker = ensure_worker_started().await?;
    let (reply_tx, reply_rx) = oneshot::channel();
    worker
        .tx
        .send(WorkMsg::Describe {
            path: image_path.to_path_buf(),
            reply: reply_tx,
        })
        .map_err(|e| anyhow::anyhow!("worker send failed: {e}"))?;
    reply_rx
        .await
        .map_err(|e| anyhow::anyhow!("worker dropped: {e}"))?
}
