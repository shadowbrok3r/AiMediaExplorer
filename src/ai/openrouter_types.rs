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
    #[serde(default)] pub choices: Vec<ChatChoiceStream>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_usage_only_chunk_without_choices_parses() {
        let chunk: OpenRouterChatCompletionStreamResponse =
            serde_json::from_str(r#"{"model":"m","object":"chat.completion.chunk","usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}"#).unwrap();
        assert!(chunk.choices.is_empty());
        assert_eq!(chunk.usage.map(|u| u.total_tokens), Some(7));
    }

    #[test]
    fn a_content_chunk_parses_and_joined_content_concatenates() {
        let chunk: OpenRouterChatCompletionStreamResponse =
            serde_json::from_str(r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
        let full: OpenRouterChatCompletionResponse = serde_json::from_str(
            r#"{"id":"1","model":"m","object":"chat.completion","created":1,"choices":[{"index":0,"message":{"role":"assistant","content":"a"}},{"index":1,"message":{"role":"assistant","content":"b"}}]}"#,
        )
        .unwrap();
        assert_eq!(full.joined_content(), "ab");
    }
}
