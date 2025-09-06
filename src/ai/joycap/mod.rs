use crate::ai::candle_llava::{
    clip_image_processor::CLIPImageProcessor, config::LLaVAConfig, conversation::Conversation,
    llama::Cache, model::LLaVA,
};
use crate::app::DEFAULT_JOYCAPTION_PATH;
use crate::ui::status::GlobalStatusIndicator;
use candle_core::Device;
use once_cell::sync::OnceCell;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

pub mod generate;
pub mod load;
pub mod stream;

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
}

static WORKER: OnceCell<WorkerHandle> = OnceCell::new();

pub async fn ensure_worker_started() -> anyhow::Result<&'static WorkerHandle> {
    if let Some(h) = WORKER.get() {
        return Ok(h);
    }
    WORKER.get_or_try_init(|| {
        let dir_env = std::env::var("JOYCAPTION_MODEL_DIR").ok();
        log::info!("JOYCAPTION_MODEL_DIR: {dir_env:?}");
        let candidate = dir_env.as_deref().unwrap_or(DEFAULT_JOYCAPTION_PATH);
        log::info!("candidate: {candidate}");
        // Surface the selected model path in the status hover UI
        crate::ui::status::JOY_STATUS.set_model(candidate);
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
                }
            }
        });
        Ok(WorkerHandle { tx })
    })
}

/// Stream describe raw image bytes with an instruction using the background worker.
/// Returns the full generated string (caller may parse JSON from it). Falls back with
/// an error if the worker/model isn't loaded.
pub async fn stream_describe_bytes(bytes: Vec<u8>, instruction: &str) -> anyhow::Result<String> {
    let worker = ensure_worker_started().await?;
    use tokio::sync::mpsc as tmpsc;
    let (token_tx, mut token_rx) = tmpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .tx
        .send(WorkMsg::StreamDescribeBytes {
            bytes,
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
                Err(e)
            }
        }
        Err(e) => Err(anyhow::anyhow!("worker dropped: {e}")),
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
    worker
        .tx
        .send(WorkMsg::StreamDescribeBytes {
            bytes,
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
                Err(e)
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
