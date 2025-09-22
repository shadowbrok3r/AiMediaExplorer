use anyhow::Result;
use futures_util::StreamExt;
use base64::Engine as _;

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: String,              // openai|grok|gemini|groq|openrouter|custom
    pub api_key: Option<String>,
    pub base_url: Option<String>,      // for custom; for others we choose defaults
    pub model: String,
    pub organization: Option<String>,  // OpenAI optional
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

fn is_openai_chat_compatible(provider: &str) -> bool {
    match provider {
        // These use the OpenAI Chat Completions API shape
        "openai" | "grok" | "groq" | "openrouter" | "custom" => true,
        // Gemini differs; we will call a separate path
        _ => false,
    }
}

fn image_bytes_to_data_url(bytes: &[u8]) -> String {
    let mime = infer::get(bytes).map(|t| t.mime_type()).unwrap_or("image/jpeg");
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{};base64,{}", mime, b64)
}

pub async fn stream_multimodal_reply(
    cfg: ProviderConfig,
    prompt: &str,
    image_bytes: Option<&[u8]>,
    on_token: impl FnMut(&str),
) -> Result<String> {
    if cfg.provider == "gemini" {
        return stream_gemini(cfg, prompt, image_bytes, on_token).await;
    }
    // OpenAI-compatible (Chat Completions)
    stream_openai_compatible(cfg, prompt, image_bytes, on_token).await
}

async fn stream_openai_compatible(
    cfg: ProviderConfig,
    prompt: &str,
    image_bytes: Option<&[u8]>,
    mut on_token: impl FnMut(&str),
) -> Result<String> {
    let (default_base, auth_header_name) = default_base_and_header(&cfg.provider);
    let base = cfg.base_url.clone().unwrap_or(default_base);
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let client = reqwest::Client::builder().build()?;
    let mut req = client.post(&url);

    if let Some(key) = cfg.api_key.as_deref() {
        // Bearer token for most providers
        if auth_header_name.eq_ignore_ascii_case("authorization") {
            req = req.header("Authorization", format!("Bearer {}", key));
        } else {
            req = req.header(auth_header_name.clone(), key);
        }
    }
    if let Some(org) = cfg.organization.as_deref() {
        req = req.header("OpenAI-Organization", org);
    }

    // Build messages with optional image data URL. Add a system instruction to return strict JSON only.
    let system_instruction = "You are a vision assistant. Reply with ONLY a single JSON object with keys: description (string), caption (string), tags (array of strings), category (string). Do not include code fences or extra prose.";
    let messages = if let Some(img) = image_bytes {
        let data_url = image_bytes_to_data_url(img);
        serde_json::json!([
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": [
                {"type":"text", "text": prompt},
                {"type":"image_url", "image_url": {"url": data_url}}
            ]}
        ])
    } else {
        serde_json::json!([
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ])
    };

    let body = serde_json::json!({
        "model": cfg.model,
        "stream": true,
        "messages": messages,
        // sensible defaults
        "temperature": 0.2
    });

    let resp = req.json(&body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("provider error {}: {}", status, text);
    }
    let mut stream = resp.bytes_stream();
    let mut acc = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        // Expect OpenAI SSE-like: lines starting with "data: " and possibly [DONE]
        let s = String::from_utf8_lossy(&chunk);
        for line in s.split('\n') {
            let line = line.trim();
            if line.is_empty() { continue; }
            // Some servers may deliver full JSON not SSE. Try flexible parse.
            if let Some(stripped) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("Data: ")) {
                if stripped == "[DONE]" { return Ok(acc); }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(stripped) {
                    if let Some(delta) = v.pointer("/choices/0/delta/content").and_then(|x| x.as_str()) {
                        on_token(delta);
                        acc.push_str(delta);
                    } else if let Some(content) = v.pointer("/choices/0/message/content").and_then(|x| x.as_str()) {
                        // Some servers send full content objects at once
                        on_token(content);
                        acc.push_str(content);
                    }
                }
            } else {
                // Try parse line as a JSON object
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(delta) = v.pointer("/choices/0/delta/content").and_then(|x| x.as_str()) {
                        on_token(delta);
                        acc.push_str(delta);
                    } else if let Some(content) = v.pointer("/choices/0/message/content").and_then(|x| x.as_str()) {
                        on_token(content);
                        acc.push_str(content);
                    }
                }
            }
        }
    }
    Ok(acc)
}

async fn stream_gemini(
    cfg: ProviderConfig,
    prompt: &str,
    image_bytes: Option<&[u8]>,
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
    // Build Gemini payload with inline data or text-only, prefixed with an instruction to respond with strict JSON only.
    let instruction = "You are a vision assistant. Reply with ONLY a single JSON object with keys: description (string), caption (string), tags (array of strings), category (string). Do not include code fences or extra prose.";
    let contents = if let Some(img) = image_bytes {
        let mime = infer::get(img).map(|t| t.mime_type()).unwrap_or("image/jpeg");
        let b64 = base64::engine::general_purpose::STANDARD.encode(img);
        serde_json::json!([{
            "role":"user",
            "parts":[
                {"text": format!("{}\n\n{}", instruction, prompt)},
                {"inline_data": {"mime_type": mime, "data": b64}}
            ]
        }])
    } else {
        serde_json::json!([{"role":"user","parts":[{"text": format!("{}\n\n{}", instruction, prompt)}]}])
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
