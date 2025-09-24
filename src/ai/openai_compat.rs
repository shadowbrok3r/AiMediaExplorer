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
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT};

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: String,              // openai|grok|gemini|groq|openrouter|custom
    pub api_key: Option<String>,
    pub base_url: Option<String>,      // for custom; for others we choose defaults
    pub model: String,
    pub organization: Option<String>,  // OpenAI optional
    pub temperature: Option<f32>,      // optional sampling temperature for chat completions
}

fn default_base_and_header(provider: &str) -> (String, String) {
    match provider {
        "openai" => ("https://api.openai.com/v1".into(), "Authorization".into()),
        "grok" => ("https://api.x.ai/v1".into(), "Authorization".into()),
        "gemini" => ("https://generativelanguage.googleapis.com/v1beta".into(), "x-goog-api-key".into()),
        "groq" => ("https://api.groq.com/openai/v1".into(), "Authorization".into()),
        "openrouter" => ("https://openrouter.ai/api/v1".into(), "Authorization".into()),
        _ => ("http://localhost:11434/v1".into(), "Authorization".into()),
    }
}

// (compat note) Additional provider families can be added here in future.

fn image_bytes_to_data_url(bytes: &[u8]) -> String {
    let mime = infer::get(bytes).map(|t| t.mime_type()).unwrap_or("image/png");
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{};base64,{}", mime, b64)
}

/// Build an async-openai client configured for the given provider/base/key.
/// For OpenRouter we add recommended headers (HTTP-Referer, X-Title) if provided via env.
fn build_async_openai_client(
    provider: &str,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
) -> Result<Client<OpenAIConfig>> {
    // Determine base URL defaults per provider
    let (default_base, _auth_header_name) = default_base_and_header(provider);
    let api_base = base_url.unwrap_or(default_base);

    let mut cfg = OpenAIConfig::new().with_api_base(api_base);
    if let Some(key) = api_key { cfg = cfg.with_api_key(key); }

    // Build reqwest client with optional OpenRouter headers
    let mut headers = HeaderMap::new();
    if let Some(org) = organization.as_deref() {
        // OpenAI optional organization header
        if let Ok(val) = HeaderValue::from_str(org) {
            headers.insert(HeaderName::from_static("openai-organization"), val);
        }
    }
    if provider == "openrouter" {
        if let Ok(site) = std::env::var("OPENROUTER_SITE") {
            if let Ok(val) = HeaderValue::from_str(&site) { headers.insert(HeaderName::from_static("referer"), val); }
        }
        if let Ok(title) = std::env::var("OPENROUTER_TITLE") {
            if let Ok(val) = HeaderValue::from_str(&title) { headers.insert(HeaderName::from_static("x-title"), val); }
        }
        // Some proxies benefit from explicit Accept
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    }
    let http_client = reqwest::Client::builder()
        .default_headers(headers)
        .build()?;

    Ok(Client::with_config(cfg).with_http_client(http_client))
}

/// List available models via async-openai for a given provider. Primarily targets OpenRouter.
pub async fn list_models_via_async_openai(provider: &str, api_key: Option<String>, base_url: Option<String>) -> Result<Vec<String>> {
    let client = build_async_openai_client(provider, api_key, base_url, None)?;
    let resp = client.models().list().await?;
    let mut out = Vec::new();
    for m in resp.data {
        out.push(m.id);
    }
    out.sort();
    Ok(out)
}

pub async fn stream_multimodal_reply(
    cfg: ProviderConfig,
    prompt: &str,
    images: &[Vec<u8>],
    on_token: impl FnMut(&str),
) -> Result<String> {
    if cfg.provider == "gemini" {
        return stream_gemini(cfg, prompt, images, on_token).await;
    }
    // OpenAI-compatible (Chat Completions)
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

    let user_msg = ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_content)
            .build()?
    );

    let req = CreateChatCompletionRequestArgs::default()
        .model(cfg.model.clone())
        .messages(vec![system_msg, user_msg])
        .temperature(cfg.temperature.unwrap_or(0.2))
        .build()?;

    let mut stream = client.chat().create_stream(req).await?;
    let mut acc = String::new();
    while let Some(event) = stream.next().await {
        let resp = event?; // Map OpenAIError via ? into anyhow
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

async fn stream_gemini(
    cfg: ProviderConfig,
    prompt: &str,
    images: &[Vec<u8>],
    mut on_token: impl FnMut(&str),
) -> Result<String> {
    // Gemini streaming uses a different endpoint and schema.
    // We'll use: POST /models/{model}:streamGenerateContent?alt=sse&key=API_KEY
    let base = cfg.base_url.clone().unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".into());
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse&key={}",
        base.trim_end_matches('/'),
        cfg.model,
        cfg.api_key.clone().unwrap_or_default()
    );
    let client = reqwest::Client::builder().build()?;
        // Build Gemini payload with multiple inline_data or text-only; request helpful concise text
        let instruction = "You are a helpful assistant. Produce helpful, concise text suitable for a chat UI.";
    let contents = if images.is_empty() {
        serde_json::json!([{ "role":"user","parts":[{"text": format!("{}\n\n{}", instruction, prompt)}]}])
    } else {
        let mut parts: Vec<serde_json::Value> = vec![serde_json::json!({"text": format!("{}\n\n{}", instruction, prompt)})];
        for img in images {
            let mime = infer::get(img).map(|t| t.mime_type()).unwrap_or("image/png");
            let b64 = base64::engine::general_purpose::STANDARD.encode(img);
            parts.push(serde_json::json!({"inline_data": {"mime_type": mime, "data": b64}}));
        }
        serde_json::json!([{ "role":"user","parts": parts }])
    };
    let body = serde_json::json!({"contents": contents});
    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("gemini error {}: {}", status, text);
    }
    let mut stream = resp.bytes_stream();
    let mut acc = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let s = String::from_utf8_lossy(&chunk);
        for line in s.split('\n') {
            let line = line.trim();
            if !line.starts_with("data:") { continue; }
            let json = line.trim_start_matches("data:").trim();
            if json.is_empty() || json == "[DONE]" { continue; }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
                // Look for candidates[0].content.parts[].text
                if let Some(parts) = v.pointer("/candidates/0/content/parts").and_then(|x| x.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            on_token(t);
                            acc.push_str(t);
                        }
                    }
                }
            }
        }
    }
    Ok(acc)
}
