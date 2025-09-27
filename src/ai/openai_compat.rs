use anyhow::Result;
use futures_util::StreamExt;
use base64::Engine as _;
use async_openai::{config::OpenAIConfig, Client};
use async_openai::types::{
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestMessageContentPartTextArgs,
    ChatCompletionRequestMessageContentPartImageArgs,
    CreateChatCompletionRequestArgs,
    ImageUrlArgs,
};
use crate::ai::openrouter_types::{
    OpenRouterChatCompletionResponseStream,
    OpenRouterChatCompletionStreamResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: String,              // openai|grok|gemini|groq|openrouter|custom
    pub api_key: Option<String>,
    pub base_url: Option<String>,      // for custom; for others we choose defaults
    pub model: String,
    pub organization: Option<String>,  // OpenAI optional
    pub temperature: Option<f32>,      // optional sampling temperature for chat completions
    pub zdr: bool
}

fn default_base(provider: &str) -> String {
    match provider {
        "openai" => "https://api.openai.com/v1".into(),
        "grok" => "https://api.x.ai/v1".into(),
        "gemini" => "https://generativelanguage.googleapis.com/v1beta".into(), // OpenAI-compatible proxy assumed
        "groq" => "https://api.groq.com/openai/v1".into(),
        "openrouter" => "https://openrouter.ai/api/v1".into(),
        _ => "http://localhost:11434/v1".into(),
    }
}

fn image_bytes_to_data_url(bytes: &[u8]) -> String {
    let mime = infer::get(bytes).map(|t| t.mime_type()).unwrap_or("image/png");
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{};base64,{}", mime, b64)
}

/// Build an async-openai client configured for the given provider/base/key.
fn build_async_openai_client(
    provider: &str,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
) -> Result<Client<OpenAIConfig>> {
    let api_base = base_url.unwrap_or_else(|| default_base(provider));
    let mut cfg = OpenAIConfig::new().with_api_base(api_base);
    if let Some(key) = api_key { cfg = cfg.with_api_key(key); }
    if let Some(org) = organization { cfg = cfg.with_org_id(org); }
    Ok(Client::with_config(cfg))
}

/// List available models via async-openai for a given provider. Primarily targets OpenRouter.
pub async fn list_models_via_async_openai(provider: &str, api_key: Option<String>, base_url: Option<String>) -> Result<Vec<String>> {
    let client = build_async_openai_client(provider, api_key, base_url, None)?;
    let resp = client.models().list().await?;
    let mut out: Vec<String> = resp.data.into_iter().map(|m| m.id).collect();
    out.sort();
    Ok(out)
}

// ---------------- OpenRouter typed model schema ----------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterArchitecture {
    #[serde(default)] pub input_modalities: Vec<String>,
    #[serde(default)] pub output_modalities: Vec<String>,
    #[serde(default)] pub tokenizer: Option<String>,
    #[serde(default)] pub instruct_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterTopProvider {
    #[serde(default)] pub is_moderated: Option<bool>,
    #[serde(default)] pub context_length: Option<u64>,
    #[serde(default)] pub max_completion_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterPricing {
    #[serde(default)] pub prompt: Option<String>,
    #[serde(default)] pub completion: Option<String>,
    #[serde(default)] pub image: Option<String>,
    #[serde(default)] pub request: Option<String>,
    #[serde(default)] pub web_search: Option<String>,
    #[serde(default)] pub internal_reasoning: Option<String>,
    #[serde(default)] pub input_cache_read: Option<String>,
    #[serde(default)] pub input_cache_write: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterModel {
    pub id: String,
    #[serde(default)] pub name: Option<String>,
    #[serde(default)] pub created: Option<f64>,
    #[serde(default)] pub description: Option<String>,
    #[serde(default)] pub architecture: Option<OpenRouterArchitecture>,
    #[serde(default)] pub top_provider: Option<OpenRouterTopProvider>,
    #[serde(default)] pub pricing: Option<OpenRouterPricing>,
    #[serde(default)] pub canonical_slug: Option<String>,
    #[serde(default)] pub context_length: Option<u64>,
    #[serde(default)] pub hugging_face_id: Option<String>,
    #[serde(default)] pub per_request_limits: Option<Value>,
    #[serde(default)] pub supported_parameters: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModelsEnvelope { data: Vec<OpenRouterModel> }

pub async fn fetch_openrouter_models_json(api_key: Option<String>, base_url: Option<String>) -> Result<Vec<OpenRouterModel>> {
    let client = build_async_openai_client("openrouter", api_key, base_url, None)?;
    let resp: OpenRouterModelsEnvelope = client.models().list_byot().await?;
    let mut models: Vec<OpenRouterModel> = resp.data;
    models.sort_by(|a,b| a.id.cmp(&b.id));
    Ok(models)
}


pub async fn stream_multimodal_reply(
    cfg: ProviderConfig,
    prompt: &str,
    images: &[Vec<u8>],
    on_token: impl FnMut(&str),
) -> Result<String> {
    // All providers go through the OpenAI-compatible path now
    log::error!("Cfg: {cfg:?}");
    stream_openai_compatible(cfg, prompt, images, on_token).await
}

async fn stream_openai_compatible(
    cfg: ProviderConfig,
    prompt: &str,
    images: &[Vec<u8>],
    mut on_token: impl FnMut(&str),
) -> Result<String> {
    // Build async-openai client with provider-specific base URL, API key, and optional org header
    let client = build_async_openai_client(
        &cfg.provider,
        cfg.api_key.clone(),
        cfg.base_url.clone(),
        cfg.organization.clone(),
    )?;

    // Build messages using typed builders, supporting multimodal (text + multiple image_url parts)
    let system_instruction = "You are a helpful assistant. Produce helpful, concise text suitable for a chat UI.";
    let system_msg = ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_instruction)
            .build()?
    );

    log::error!("system_msg: {system_msg:?}");

    let user_content = if images.is_empty() {
        ChatCompletionRequestUserMessageContent::Text(prompt.to_string())
    } else {
        let mut parts = Vec::with_capacity(1 + images.len());
        parts.push(
            ChatCompletionRequestMessageContentPartTextArgs::default()
                .text(prompt.to_string())
                .build()?
                .into()
        );
        for img in images {
            let data_url = image_bytes_to_data_url(img);
            let image_part = ChatCompletionRequestMessageContentPartImageArgs::default()
                .image_url(
                    ImageUrlArgs::default()
                        .url(data_url)
                        .build()?
                )
                .build()?;
            parts.push(image_part.into());
        }
        ChatCompletionRequestUserMessageContent::Array(parts)
    };

    log::error!("user_content: {user_content:?}");

    let user_msg = ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_content)
            .build()?
    );

    log::error!("user_msg: {user_msg:?}");

    let req = CreateChatCompletionRequestArgs::default()
        .model(cfg.model.clone())
        .messages(vec![system_msg, user_msg])
        .temperature(cfg.temperature.unwrap_or(0.5))
        .stream(true)
        .build()?;

    if cfg.provider == "openrouter" {
        log::error!("OpenRouter Request: {req:?}");
        // BYOT streaming with our custom OpenRouter stream type
        let mut stream: OpenRouterChatCompletionResponseStream = client
            .chat()
            .create_stream_byot(req)
            .await
            .map_err(|e| anyhow::anyhow!("openrouter stream init failed: {e}"))?;

        // Box the concrete stream into our alias
        let mut acc = String::new();
        while let Some(chunk) = stream.next().await {
            let resp: OpenRouterChatCompletionStreamResponse = chunk?;
            for choice in resp.choices {
                if let Some(delta) = choice.delta.content {
                    if !delta.is_empty() {
                        on_token(&delta);
                        acc.push_str(&delta);
                    }
                }
            }
        }
        Ok(acc)
    } else {
        // Standard OpenAI-compatible streaming
        let mut stream = client.chat().create_stream(req).await?;
        let mut acc = String::new();
        while let Some(event) = stream.next().await {
            let resp = event?;
            for choice in resp.choices {
                if let Some(delta) = choice.delta.content {
                    if !delta.is_empty() {
                        on_token(&delta);
                        acc.push_str(&delta);
                    }
                }
            }
        }
        Ok(acc)
    }
}
