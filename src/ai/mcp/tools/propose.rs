//! Tools that suggest metadata changes; the user decides in the application.

use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::{page_json, page_limit, row_by_path, safe_text};
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::proposals::{Changes, NewProposal, Proposal, Status};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};

const TIMEOUT: Duration = Duration::from_secs(30);

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("metadata_propose", ToolClass::Propose, TIMEOUT, "Propose metadata changes", "Suggest a new caption, description, category or tag changes for one library item. Nothing is written: the suggestion waits as a pending proposal until the user accepts or rejects it in the application.", metadata_propose),
        entry("metadata_list_proposals", ToolClass::Read, TIMEOUT, "List metadata proposals", "Proposals that were filed, newest first, with their status: pending, accepted, rejected or failed.", metadata_list_proposals),
    ]
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct ProposeArgs {
    /// Path of a library item.
    pub path: String,
    /// New caption, up to 300 characters.
    pub caption: Option<String>,
    /// New description, up to 4000 characters.
    pub description: Option<String>,
    /// New category, up to 80 characters.
    pub category: Option<String>,
    /// Tags to add, at most 30, each up to 40 characters.
    #[serde(default)]
    pub tags_add: Vec<String>,
    /// Tags to remove.
    #[serde(default)]
    pub tags_remove: Vec<String>,
    /// Why, up to 500 characters; shown to the user.
    pub reason: Option<String>,
}

pub fn proposal_json(p: &Proposal) -> serde_json::Value {
    json!({
        "id": p.id,
        "path": p.path,
        "status": p.status.as_str(),
        "changes": p.changes,
        "reason": p.reason,
        "source": p.source,
        "created_ms": p.created_ms,
        "decided_ms": p.decided_ms,
        "note": p.note,
    })
}

async fn metadata_propose(ctx: Arc<ToolCtx>, call: CallInfo, args: ProposeArgs) -> Result<ToolOutput, ToolError> {
    let path = args.path.trim().to_string();
    ctx.policy.refresh(&ctx.db).await;
    ctx.policy.check_lexical(&path)?;
    let changes = Changes { caption: args.caption, description: args.description, category: args.category, tags_add: args.tags_add, tags_remove: args.tags_remove }.normalized()?;
    let db = ctx.db.get()?;
    if row_by_path(db, &path).await?.is_none() {
        return Err(ToolError::NotFound(format!("{} is not in the library", safe_text(&path, 100))));
    }
    let reason = args.reason.as_deref().map(|r| safe_text(r, crate::ai::mcp::proposals::MAX_REASON)).filter(|r| !r.is_empty());
    let (proposal, created) = ctx.proposals.create(NewProposal { path, changes, reason, source: call.client.clone() }).await?;
    let text = if created { format!("proposal {} is waiting for the user's review", proposal.id) } else { format!("the same proposal {} is already waiting for review", proposal.id) };
    let mut data = proposal_json(&proposal);
    data["created"] = json!(created);
    data["written"] = json!(false);
    Ok(ToolOutput::new(text, data))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct ListProposalsArgs {
    /// Only proposals in this status: pending, accepted, rejected or failed.
    pub status: Option<String>,
    /// Only proposals for this path.
    pub path: Option<String>,
    /// Proposals per page, 1 to 100; default 25.
    pub limit: Option<usize>,
    /// Proposals to skip.
    pub offset: Option<usize>,
}

async fn metadata_list_proposals(ctx: Arc<ToolCtx>, _call: CallInfo, args: ListProposalsArgs) -> Result<ToolOutput, ToolError> {
    let status = match args.status.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => Some(Status::parse(&s.to_lowercase()).ok_or_else(|| ToolError::invalid("status must be pending, accepted, rejected or failed"))?),
        None => None,
    };
    let limit = page_limit(args.limit, 25, 100);
    let offset = args.offset.unwrap_or(0);
    let (rows, total) = ctx.proposals.list(status, args.path.as_deref().map(str::trim).filter(|p| !p.is_empty()), offset, limit).await?;
    let counts = ctx.proposals.counts().await?;
    let mut data = page_json(rows.iter().map(proposal_json).collect(), total, offset, limit);
    data["counts"] = json!(counts);
    Ok(ToolOutput::new(format!("{} of {total} proposals", rows.len()), data))
}
