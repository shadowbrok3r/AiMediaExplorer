//! Tools that suggest tag and category renames and merges; the user decides in the application.

use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::{page_json, page_limit, safe_text};
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::proposals::{MAX_REASON, Status};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};
use crate::ai::mcp::taxonomy::{Kind, NewTaxonomy, Rename, TaxProposal};

const TIMEOUT: Duration = Duration::from_secs(30);

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("taxonomy_propose_rename", ToolClass::Propose, TIMEOUT, "Propose a tag or category rename", "Suggest renaming a tag or category everywhere in the library, or merging several tags or categories into one name. Nothing is changed: the suggestion waits as a pending proposal, with the number of items it would touch, until the user accepts or rejects it in the application. Use list_tags or list_categories first to see the names that exist.", taxonomy_propose_rename),
        entry("taxonomy_list_proposals", ToolClass::Read, TIMEOUT, "List taxonomy proposals", "Tag and category rename proposals that were filed, newest first, with their status: pending, accepted, rejected or failed.", taxonomy_list_proposals),
    ]
}

/// What a proposal renames.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum KindArg {
    Tag,
    Category,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct ProposeRenameArgs {
    /// Whether the names are tags or categories.
    pub kind: KindArg,
    /// The existing names to rename or merge, 1 to 10. One name is a rename, several are a merge. Names match exactly, including case.
    pub from: Vec<String>,
    /// The new name: a tag up to 40 characters, a category up to 80. A name that already exists merges the items into it.
    pub to: String,
    /// Why, up to 500 characters; shown to the user.
    pub reason: Option<String>,
}

pub fn proposal_json(p: &TaxProposal) -> serde_json::Value {
    json!({
        "id": p.id,
        "kind": p.kind.as_str(),
        "from": p.old_names,
        "to": p.new_name,
        "status": p.status.as_str(),
        "reason": p.reason,
        "source": p.source,
        "created_ms": p.created_ms,
        "decided_ms": p.decided_ms,
        "note": p.note,
        "items_at_filing": p.items,
        "applied": p.applied,
    })
}

async fn taxonomy_propose_rename(ctx: Arc<ToolCtx>, call: CallInfo, args: ProposeRenameArgs) -> Result<ToolOutput, ToolError> {
    let kind = match args.kind {
        KindArg::Tag => Kind::Tag,
        KindArg::Category => Kind::Category,
    };
    let rename = Rename::new(kind, &args.from, &args.to)?;
    let store = &ctx.agent.taxonomy;
    let preview = store.preview(&rename).await?;
    if preview.items == 0 {
        let names = rename.old_names.iter().map(|n| format!("'{}'", safe_text(n, 60))).collect::<Vec<_>>().join(", ");
        return Err(ToolError::NotFound(format!("no item in the library has the {} {names}; check the exact names with list_{}", kind.as_str(), if kind == Kind::Tag { "tags" } else { "categories" })));
    }
    let reason = args.reason.as_deref().map(|r| safe_text(r, MAX_REASON)).filter(|r| !r.is_empty());
    let (proposal, created) = store.create(NewTaxonomy { rename, reason, source: call.client.clone() }, preview.items).await?;
    let text = if created { format!("proposal {} is waiting for the user's review: {}", proposal.id, preview.summary()) } else { format!("the same proposal {} is already waiting for review", proposal.id) };
    let mut data = proposal_json(&proposal);
    data["created"] = json!(created);
    data["written"] = json!(false);
    data["preview"] = serde_json::to_value(&preview).map_err(ToolError::failed)?;
    data["preview"]["summary"] = json!(preview.summary());
    Ok(ToolOutput::new(text, data))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct ListArgs {
    /// Only proposals in this status: pending, accepted, rejected or failed.
    pub status: Option<String>,
    /// Proposals per page, 1 to 100; default 25.
    pub limit: Option<usize>,
    /// Proposals to skip.
    pub offset: Option<usize>,
}

async fn taxonomy_list_proposals(ctx: Arc<ToolCtx>, _call: CallInfo, args: ListArgs) -> Result<ToolOutput, ToolError> {
    let status = match args.status.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => Some(Status::parse(&s.to_lowercase()).ok_or_else(|| ToolError::invalid("status must be pending, accepted, rejected or failed"))?),
        None => None,
    };
    let limit = page_limit(args.limit, 25, 100);
    let offset = args.offset.unwrap_or(0);
    let store = &ctx.agent.taxonomy;
    let (rows, total) = store.list(status, offset, limit).await?;
    let mut data = page_json(rows.iter().map(proposal_json).collect(), total, offset, limit);
    data["counts"] = json!(store.counts().await?);
    Ok(ToolOutput::new(format!("{} of {total} taxonomy proposals", rows.len()), data))
}
