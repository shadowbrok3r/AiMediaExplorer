//! Applying and rejecting proposals: writes go through the metadata journal and the MCP proposal store.

use crate::ai::mcp::ctx::DbSource;
use crate::ai::mcp::proposals::{FieldMask, ProposalStore};
use crate::database::journal;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed};
use crate::ui::inspector::RowEdit;

use super::model::{BulkPlan, PlanItem, ProposalKey};

/// Journal label of accepted proposals of this session.
pub const LOCAL_LABEL: &str = "Accepted refinements";

/// What an accept or reject did, per proposal.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub done: Vec<ProposalKey>,
    pub failed: Vec<(ProposalKey, String)>,
    /// Fields written, over the proposals that were applied.
    pub fields: usize,
    /// The rows as they are stored now, for the browsers.
    pub edits: Vec<RowEdit>,
}

impl Outcome {
    pub fn files(&self) -> usize {
        self.done.len()
    }
}

fn mask_count(mask: FieldMask) -> usize {
    [mask.caption, mask.description, mask.category, mask.tags].into_iter().filter(|on| *on).count()
}

fn all_failed(items: &[PlanItem], why: &str) -> Outcome {
    Outcome { failed: items.iter().map(|i| (i.key.clone(), why.to_string())).collect(), ..Default::default() }
}

/// Applies the checked fields of the plan: proposals of this session as one journal batch, proposals of the store through the store.
pub async fn apply(db: &DbSource, store: Option<&ProposalStore>, plan: BulkPlan) -> Outcome {
    let handle = match db.get() {
        Ok(handle) => handle,
        Err(e) => return all_failed(&plan.items, &e.message()),
    };
    let mut out = Outcome::default();
    let (local, stored): (Vec<&PlanItem>, Vec<&PlanItem>) = plan.items.iter().partition(|i| matches!(i.key, ProposalKey::Local(_)));

    if !local.is_empty() {
        let writes: Vec<(RowSeed, MetaPatch)> = local.iter().map(|i| (RowSeed::from_path(&i.path), i.patch.clone().unwrap_or_default())).collect();
        let results = metadata_edit::save_patches_as(handle, &writes, LOCAL_LABEL).await;
        for (item, result) in local.iter().zip(results) {
            match result {
                Ok(_) => {
                    out.fields += mask_count(item.mask);
                    out.done.push(item.key.clone());
                }
                Err(e) => out.failed.push((item.key.clone(), format!("{e:#}"))),
            }
        }
    }

    if !stored.is_empty() {
        match store {
            None => out.failed.extend(stored.iter().map(|i| (i.key.clone(), "the MCP proposal store is not available".to_string()))),
            Some(store) => {
                let requests: Vec<(String, FieldMask)> = stored.iter().filter_map(|i| if let ProposalKey::Mcp(id) = &i.key { Some((id.clone(), i.mask)) } else { None }).collect();
                for ((id, result), item) in store.accept_many(&requests).await.into_iter().zip(stored.iter()) {
                    let key = ProposalKey::Mcp(id);
                    match result {
                        Ok(_) => {
                            out.fields += mask_count(item.mask);
                            out.done.push(key);
                        }
                        Err(e) => out.failed.push((key, e.message())),
                    }
                }
            }
        }
    }

    let paths: Vec<String> = plan.items.iter().filter(|i| out.done.contains(&i.key)).map(|i| i.path.clone()).collect();
    out.edits = stored_edits(handle, &paths).await;
    out
}

/// The row values now stored for `paths`, as edits for the rows the browsers show.
pub async fn stored_edits(db: &surrealdb::Surreal<surrealdb::engine::local::Db>, paths: &[String]) -> Vec<RowEdit> {
    let Ok(rows) = journal::snapshots_with(db, paths).await else { return Vec::new() };
    paths
        .iter()
        .filter_map(|path| rows.get(path).map(|s| (path, s)))
        .map(|(path, s)| RowEdit::Patch {
            path: path.clone(),
            patch: MetaPatch { description: Some(s.description.clone()), caption: Some(s.caption.clone()), category: Some(s.category.clone()), tags: Some(s.tags.clone().unwrap_or_default()) },
        })
        .collect()
}

/// Rejects the proposals of the store among `keys`; proposals of this session need no write.
pub async fn reject(store: Option<&ProposalStore>, keys: &[ProposalKey]) -> Outcome {
    let mut out = Outcome::default();
    for key in keys {
        match (key, store) {
            (ProposalKey::Local(_), _) => out.done.push(key.clone()),
            (ProposalKey::Mcp(id), Some(store)) => match store.reject(id).await {
                Ok(_) => out.done.push(key.clone()),
                Err(e) => out.failed.push((key.clone(), e.message())),
            },
            (ProposalKey::Mcp(_), None) => out.failed.push((key.clone(), "the MCP proposal store is not available".to_string())),
        }
    }
    out
}
