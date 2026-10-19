//! The resources of the server: the library guide.

use rmcp::ErrorData as McpError;
use rmcp::model::{ReadResourceResult, Resource, ResourceContents};

use super::ctx::ToolCtx;
use super::guide::GUIDE_URI;

/// The resources a client can read.
pub fn list(ctx: &ToolCtx) -> Vec<Resource> {
    let text = ctx.agent.guide.text();
    vec![Resource::new(GUIDE_URI, "library_guide").with_title("Library guide").with_description("The owner's conventions for tags, categories, captions and sensitive pictures; read it before proposing changes.").with_mime_type("text/markdown").with_size(text.len() as u64)]
}

/// The contents of the resource `uri`.
pub fn read(ctx: &ToolCtx, uri: &str) -> Result<ReadResourceResult, McpError> {
    if uri != GUIDE_URI {
        return Err(McpError::resource_not_found(format!("no resource {uri}; the server offers {GUIDE_URI}"), None));
    }
    Ok(ReadResourceResult::new(vec![ResourceContents::text(ctx.agent.guide.text(), GUIDE_URI).with_mime_type("text/markdown")]))
}
