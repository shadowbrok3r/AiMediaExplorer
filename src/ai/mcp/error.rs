//! Result and error types of tool calls.

use serde_json::Value;

use super::gpu::GpuError;
use super::policy::PolicyError;

/// An image returned next to the text of a tool result.
#[derive(Clone, Debug, PartialEq)]
pub struct ImagePart {
    pub mime: String,
    pub data_base64: String,
}

/// What a tool returns: a short text for the model, JSON data and optional images.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutput {
    pub text: String,
    pub data: Value,
    pub images: Vec<ImagePart>,
}

impl ToolOutput {
    pub fn new(text: impl Into<String>, data: Value) -> Self {
        Self { text: text.into(), data, images: Vec::new() }
    }

    pub fn with_image(mut self, mime: impl Into<String>, data_base64: String) -> Self {
        self.images.push(ImagePart { mime: mime.into(), data_base64 });
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolError {
    /// The arguments are wrong; the caller should fix the call.
    Invalid(String),
    NotFound(String),
    /// The path policy or a tool rule forbids the call.
    Denied(String),
    /// Too many calls or a full queue; retry later.
    Busy(String),
    /// A backend the tool needs is not available.
    Unavailable(String),
    Timeout(String),
    Failed(String),
    Cancelled,
}

impl ToolError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid_arguments",
            Self::NotFound(_) => "not_found",
            Self::Denied(_) => "denied",
            Self::Busy(_) => "busy",
            Self::Unavailable(_) => "unavailable",
            Self::Timeout(_) => "timeout",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::Invalid(m) | Self::NotFound(m) | Self::Denied(m) | Self::Busy(m) | Self::Unavailable(m) | Self::Timeout(m) | Self::Failed(m) => m.clone(),
            Self::Cancelled => "the call was cancelled".to_string(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub fn failed(message: impl std::fmt::Display) -> Self {
        Self::Failed(message.to_string())
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind(), self.message())
    }
}

impl std::error::Error for ToolError {}

impl From<anyhow::Error> for ToolError {
    fn from(e: anyhow::Error) -> Self {
        Self::Failed(format!("{e:#}"))
    }
}

impl From<surrealdb::Error> for ToolError {
    fn from(e: surrealdb::Error) -> Self {
        let text = e.to_string();
        if text.contains("Specify a namespace") || text.contains("Specify a database") {
            return Self::Unavailable("the library database is still opening; try again in a moment".to_string());
        }
        Self::Failed(format!("database error: {text}"))
    }
}

impl From<PolicyError> for ToolError {
    fn from(e: PolicyError) -> Self {
        match e {
            PolicyError::Outside => Self::Denied(e.to_string()),
            PolicyError::Missing => Self::NotFound(e.to_string()),
            other => Self::Invalid(other.to_string()),
        }
    }
}

impl From<GpuError> for ToolError {
    fn from(e: GpuError) -> Self {
        match e {
            GpuError::QueueFull { .. } => Self::Busy(e.to_string()),
            GpuError::QueueTimeout(_) | GpuError::JobTimeout(_) => Self::Timeout(e.to_string()),
            GpuError::Cancelled => Self::Cancelled,
            GpuError::Job(m) => Self::Failed(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_and_gpu_errors_map_to_distinct_kinds() {
        assert_eq!(ToolError::from(PolicyError::Outside).kind(), "denied");
        assert_eq!(ToolError::from(PolicyError::Missing).kind(), "not_found");
        assert_eq!(ToolError::from(PolicyError::Empty).kind(), "invalid_arguments");
        assert_eq!(ToolError::from(GpuError::QueueFull { depth: 3, max: 2 }).kind(), "busy");
        assert_eq!(ToolError::from(GpuError::JobTimeout(std::time::Duration::from_secs(1))).kind(), "timeout");
        assert_eq!(ToolError::from(GpuError::Cancelled), ToolError::Cancelled);
        assert_eq!(ToolError::from(GpuError::Job("x".into())), ToolError::Failed("x".into()));
        assert_eq!(ToolError::invalid("bad").to_string(), "invalid_arguments: bad");
        assert_eq!(ToolError::Cancelled.message(), "the call was cancelled");
    }

    #[tokio::test]
    async fn a_store_that_has_not_picked_its_namespace_is_unavailable() {
        let db = surrealdb::Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
        let err = match db.query("SELECT * FROM thumbnails").await {
            Err(e) => e,
            Ok(mut r) => r.take::<Vec<i64>>(0).unwrap_err(),
        };
        assert_eq!(ToolError::from(err).kind(), "unavailable");
    }

    #[test]
    fn an_output_collects_images() {
        let out = ToolOutput::new("t", serde_json::json!({"a": 1})).with_image("image/png", "AAAA".into());
        assert_eq!(out.images, vec![ImagePart { mime: "image/png".into(), data_base64: "AAAA".into() }]);
    }
}
