//! Proposals that tools filed in the MCP store, listed next to the ones this session generated.

use std::sync::Arc;

use crate::ai::mcp::ctx::DbSource;
use crate::ai::mcp::proposals::{ProposalStore, Status};
use crate::database::journal;

use super::model::{Current, Proposal};

/// Most proposals read at a time.
pub const LIST_LIMIT: usize = 200;

/// Where the store comes from.
#[derive(Clone)]
pub enum StoreSource {
    /// The store of the server in this process, looked up on first use.
    Hub,
    /// A store given by a test.
    Fixed(Arc<ProposalStore>),
    /// No store: no stored proposals are listed.
    Off,
}

impl StoreSource {
    pub fn get(&self) -> Option<Arc<ProposalStore>> {
        match self {
            Self::Hub => Some(crate::ai::mcp::server::hub().ctx.proposals.clone()),
            Self::Fixed(store) => Some(store.clone()),
            Self::Off => None,
        }
    }
}

/// The pending proposals of the store with the values their files have now.
pub async fn list_pending(store: &ProposalStore, db: &DbSource) -> Result<Vec<Proposal>, String> {
    let (stored, _total) = store.list(Some(Status::Pending), None, 0, LIST_LIMIT).await.map_err(|e| e.message())?;
    let handle = db.get().map_err(|e| e.message())?;
    let paths: Vec<String> = stored.iter().map(|p| p.path.clone()).collect();
    let rows = journal::snapshots_with(handle, &paths).await.map_err(|e| format!("{e:#}"))?;
    Ok(stored
        .iter()
        .map(|s| {
            let current = rows.get(&s.path).map(|r| Current { category: r.category.clone(), tags: r.tags.clone().unwrap_or_default(), caption: r.caption.clone(), description: r.description.clone() }).unwrap_or_default();
            Proposal::from_stored(s, &current)
        })
        .collect())
}
