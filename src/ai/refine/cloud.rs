//! Refinement by a cloud model: the provider choice, the prompt and the parser of the reply.

use super::{Proposal, lower_set, same_text};
use crate::Thumbnail;
use crate::ai::openai_compat::ProviderConfig;

/// Fallback model when neither the panel nor the settings name one.
const DEFAULT_MODEL: &str = "gpt-5-mini";

/// The fields a model proposed; empty ones mean no change.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CloudReply {
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub caption: Option<String>,
    pub description: Option<String>,
}

/// The request for the metadata of `row`.
pub fn cloud_prompt(row: &Thumbnail) -> String {
    format!(
        "You are an assistant that refines existing media metadata.\nReturn ONLY a compact JSON object with exactly these keys: new_category (string or null), new_tags (array of strings), new_caption (string or null), new_description (string or null).\nIf you cannot confidently improve a field, set it to null (or empty array for tags). Do not include explanatory text.\nCurrent data:\nPath: {path}\nCategory: {cat}\nTags: {tags}\nCaption: {cap}\nDescription: {desc}\nJSON:",
        path = row.path,
        cat = row.category.clone().unwrap_or_default(),
        tags = row.tags.join(", "),
        cap = row.caption.clone().unwrap_or_default(),
        desc = row.description.clone().unwrap_or_default()
    )
}

/// A JSON object from a reply that may be fenced or wrapped in text.
fn parse_jsonish(text: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        return Some(v);
    }
    let t = text.trim();
    if let Some(start) = t.find("```") {
        let rest = &t[start + 3..];
        let rest = if rest.to_ascii_lowercase().starts_with("json") { &rest[4..] } else { rest };
        if let Some(end) = rest.find("```") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&rest[..end]) {
                return Some(v);
            }
        }
    }
    if let (Some(i), Some(j)) = (t.find('{'), t.rfind('}')) {
        if i < j {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t[i..=j]) {
                return Some(v);
            }
        }
    }
    None
}

/// The fields of a model reply, or none when it holds no JSON object.
pub fn parse_cloud_reply(text: &str) -> Option<CloudReply> {
    let value = parse_jsonish(text)?;
    let object = value.as_object()?;
    let text_of = |key: &str| object.get(key).and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    Some(CloudReply {
        category: text_of("new_category"),
        tags: object.get("new_tags").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default(),
        caption: text_of("new_caption"),
        description: text_of("new_description"),
    })
}

/// Connection details for `provider`, with the model of the panel or the settings default.
pub fn provider_config(provider: &str, model: &str) -> ProviderConfig {
    let settings = crate::settings::snapshot();
    let model = if model.trim().is_empty() { settings.openai_default_model.clone().unwrap_or_else(|| DEFAULT_MODEL.to_string()) } else { model.trim().to_string() };
    let env_or = |opt: &Option<String>, key: &str| opt.clone().or_else(|| std::env::var(key).ok());
    let (api_key, base_url, organization) = match provider {
        "openrouter" => (env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"), Some("https://openrouter.ai/api/v1".to_string()), None),
        "groq" => (env_or(&settings.groq_api_key, "GROQ_API_KEY"), settings.openai_base_url.clone(), None),
        "gemini" => (env_or(&settings.gemini_api_key, "GEMINI_API_KEY"), settings.openai_base_url.clone(), None),
        _ => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
    };
    ProviderConfig { provider: provider.to_string(), api_key, base_url, model, organization, temperature: Some(0.2), zdr: true }
}

/// Asks the model for the metadata of `row`; none when the reply proposes nothing the row does not have.
pub async fn refine_with_cloud(cfg: &ProviderConfig, row: &Thumbnail) -> Result<Option<Proposal>, String> {
    let text = crate::ai::openai_compat::simple_text_completion(cfg.clone(), &cloud_prompt(row)).await.map_err(|e| e.to_string())?;
    let reply = parse_cloud_reply(&text).ok_or_else(|| format!("the reply of {} held no JSON object", cfg.provider))?;
    Ok(reply_to_proposal(row, reply, &format!("cloud:{}", cfg.provider)))
}

/// The proposal a reply makes for `row`, leaving out values the row already has.
pub fn reply_to_proposal(row: &Thumbnail, reply: CloudReply, generator: &str) -> Option<Proposal> {
    let mut p = Proposal::for_row(row, generator);
    p.new_category = reply.category.filter(|c| !same_text(&row.category, &Some(c.clone())));
    p.new_caption = reply.caption.filter(|c| !same_text(&row.caption, &Some(c.clone())));
    p.new_description = reply.description.filter(|d| !same_text(&row.description, &Some(d.clone())));
    if !reply.tags.is_empty() && lower_set(&reply.tags) != lower_set(&row.tags) {
        p.new_tags = reply.tags;
    }
    (!p.is_empty()).then_some(p)
}
