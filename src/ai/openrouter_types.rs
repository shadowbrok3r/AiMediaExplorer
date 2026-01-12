use std::pin::Pin;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use async_openai::types::chat::{ChatChoiceStream, CompletionUsage};

pub type OpenRouterChatCompletionResponseStream = Pin<
    Box<dyn futures_util::Stream<Item = Result<OpenRouterChatCompletionStreamResponse, async_openai::error::OpenAIError>> + Send>,
>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OpenRouterChatMessage {
    pub role: String,
    #[serde(default)] pub content: Option<String>,
    #[serde(default)] pub refusal: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OpenRouterChatChoice {
    pub message: OpenRouterChatMessage,
    #[serde(default)] pub logprobs: Option<Value>,
    #[serde(default)] pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OpenRouterChatCompletionResponse {
    pub id: String,
    pub choices: Vec<OpenRouterChatChoice>,
    #[serde(default)] pub provider: Option<String>,
    pub model: String,
    pub object: String,
    pub created: u64,
    #[serde(default)] pub system_fingerprint: Option<Value>,
    #[serde(default)] pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OpenRouterChatCompletionStreamResponse {
    pub choices: Vec<ChatChoiceStream>,
    #[serde(default)] pub created: Option<u32>,
    #[serde(default)] pub model: Option<String>,
    #[serde(default)] pub object: Option<String>,
    #[serde(default)] pub usage: Option<CompletionUsage>,
}



// Convenience to extract final assembled content from a full completion response
impl OpenRouterChatCompletionResponse {
    pub fn joined_content(&self) -> String {
        let mut out = String::new();
        for c in &self.choices {
            if let Some(text) = &c.message.content { out.push_str(text); }
        }
        out
    }
}
