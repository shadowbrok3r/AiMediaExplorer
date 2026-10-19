//! Metadata changes proposed by tools, kept in the `mcp_proposals` table until the user accepts or rejects them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use surrealdb::types::{RecordId, SurrealValue};

use super::clock::Clock;
use super::ctx::DbSource;
use super::error::ToolError;

pub const TABLE: &str = "mcp_proposals";
const DDL: &str = "DEFINE TABLE IF NOT EXISTS mcp_proposals TYPE NORMAL SCHEMALESS PERMISSIONS FULL;";

pub const MAX_CAPTION: usize = 300;
pub const MAX_DESCRIPTION: usize = 4000;
pub const MAX_CATEGORY: usize = 80;
pub const MAX_TAG: usize = 40;
pub const MAX_TAGS: usize = 30;
pub const MAX_REASON: usize = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pending,
    Accepted,
    Rejected,
    Failed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Failed => "failed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [Self::Pending, Self::Accepted, Self::Rejected, Self::Failed].into_iter().find(|v| v.as_str() == s)
    }
}

/// The edits a proposal asks for; empty fields mean no change.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Changes {
    pub caption: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub tags_add: Vec<String>,
    pub tags_remove: Vec<String>,
}

fn without_controls(text: &str) -> String {
    text.chars().filter(|c| !c.is_control() || matches!(c, '\n' | '\t')).collect()
}

/// Text on one line with runs of white space collapsed, cut to `max` characters.
fn one_line(text: &str, max: usize) -> Option<String> {
    let line = without_controls(text).split_whitespace().collect::<Vec<_>>().join(" ");
    (!line.is_empty()).then(|| line.chars().take(max).collect())
}

/// Text with its line breaks kept, trimmed and cut to `max` characters.
fn multi_line(text: &str, max: usize) -> Option<String> {
    let cleaned = without_controls(text);
    let cleaned = cleaned.trim();
    (!cleaned.is_empty()).then(|| cleaned.chars().take(max).collect())
}

fn clean_tags(tags: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tag in tags {
        let Some(t) = one_line(tag, MAX_TAG) else { continue };
        if !out.iter().any(|o| o.to_lowercase() == t.to_lowercase()) {
            out.push(t);
        }
        if out.len() >= MAX_TAGS {
            break;
        }
    }
    out
}

impl Changes {
    /// Trimmed, length-limited changes, or an error when nothing is left to change.
    pub fn normalized(self) -> Result<Self, ToolError> {
        let caption = self.caption.as_deref().and_then(|t| one_line(t, MAX_CAPTION));
        let description = self.description.as_deref().and_then(|t| multi_line(t, MAX_DESCRIPTION));
        let category = self.category.as_deref().and_then(|t| one_line(t, MAX_CATEGORY));
        let tags_add = clean_tags(&self.tags_add);
        let tags_remove = clean_tags(&self.tags_remove);
        let changes = Self { caption, description, category, tags_add, tags_remove };
        if changes.is_empty() {
            return Err(ToolError::invalid("a proposal needs at least one of caption, description, category, tags_add or tags_remove"));
        }
        Ok(changes)
    }

    pub fn is_empty(&self) -> bool {
        self.caption.is_none() && self.description.is_none() && self.category.is_none() && self.tags_add.is_empty() && self.tags_remove.is_empty()
    }
}

/// The current metadata of a library row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Existing {
    pub caption: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
}

/// Fields that change when a proposal is applied; `None` leaves a field alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub caption: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub tags: Option<Vec<String>>,
}

impl Applied {
    pub fn is_empty(&self) -> bool {
        self.caption.is_none() && self.description.is_none() && self.category.is_none() && self.tags.is_none()
    }
}

/// What applying `changes` to `existing` would write.
pub fn apply_changes(existing: &Existing, changes: &Changes) -> Applied {
    let differs = |new: &Option<String>, old: &Option<String>| new.as_ref().filter(|n| old.as_ref() != Some(*n)).cloned();
    let mut tags: Vec<String> = existing.tags.iter().filter(|t| !changes.tags_remove.iter().any(|r| r.to_lowercase() == t.to_lowercase())).cloned().collect();
    for add in &changes.tags_add {
        if !tags.iter().any(|t| t.to_lowercase() == add.to_lowercase()) {
            tags.push(add.clone());
        }
    }
    Applied {
        caption: differs(&changes.caption, &existing.caption),
        description: differs(&changes.description, &existing.description),
        category: differs(&changes.category, &existing.category),
        tags: (tags != existing.tags).then_some(tags),
    }
}

/// A proposal as stored.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Proposal {
    pub id: String,
    pub path: String,
    pub changes: Changes,
    pub reason: Option<String>,
    pub source: String,
    pub status: Status,
    pub created_ms: u64,
    pub decided_ms: Option<u64>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, SurrealValue)]
struct ProposalRow {
    id: RecordId,
    path: String,
    caption: Option<String>,
    description: Option<String>,
    category: Option<String>,
    tags_add: Vec<String>,
    tags_remove: Vec<String>,
    reason: Option<String>,
    source: String,
    status: String,
    created_ms: i64,
    decided_ms: Option<i64>,
    note: Option<String>,
}

#[derive(Clone, Debug, SurrealValue)]
struct ItemRow {
    caption: Option<String>,
    description: Option<String>,
    category: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, SurrealValue)]
struct CountRow {
    status: String,
    n: i64,
}

#[derive(Debug, SurrealValue)]
struct NRow {
    n: i64,
}

fn key_of(id: &RecordId) -> String {
    crate::database::collections::key_of(id)
}

impl From<ProposalRow> for Proposal {
    fn from(r: ProposalRow) -> Self {
        Self {
            id: key_of(&r.id),
            path: r.path,
            changes: Changes { caption: r.caption, description: r.description, category: r.category, tags_add: r.tags_add, tags_remove: r.tags_remove },
            reason: r.reason,
            source: r.source,
            status: Status::parse(&r.status).unwrap_or(Status::Failed),
            created_ms: r.created_ms.max(0) as u64,
            decided_ms: r.decided_ms.map(|v| v.max(0) as u64),
            note: r.note,
        }
    }
}

/// Counts of proposals by status.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Counts {
    pub pending: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub failed: u64,
}

/// What a caller gives to create a proposal.
#[derive(Clone, Debug)]
pub struct NewProposal {
    pub path: String,
    pub changes: Changes,
    pub reason: Option<String>,
    pub source: String,
}

pub struct ProposalStore {
    db: DbSource,
    clock: Arc<dyn Clock>,
    schema_ready: AtomicBool,
}

impl ProposalStore {
    pub fn new(db: DbSource, clock: Arc<dyn Clock>) -> Self {
        Self { db, clock, schema_ready: AtomicBool::new(false) }
    }

    async fn handle(&self) -> Result<&surrealdb::Surreal<surrealdb::engine::local::Db>, ToolError> {
        let db = self.db.get()?;
        if !self.schema_ready.load(Ordering::Acquire) {
            db.query(DDL).await?.check()?;
            self.schema_ready.store(true, Ordering::Release);
        }
        Ok(db)
    }

    fn new_id() -> Result<String, ToolError> {
        let mut bytes = [0u8; 4];
        getrandom::fill(&mut bytes).map_err(ToolError::failed)?;
        Ok(format!("p{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()))
    }

    /// Stores a pending proposal; an identical pending one is returned instead (the flag is false then).
    pub async fn create(&self, new: NewProposal) -> Result<(Proposal, bool), ToolError> {
        let db = self.handle().await?;
        let pending: Vec<ProposalRow> = db.query("SELECT * FROM mcp_proposals WHERE path = $p AND status = 'pending'").bind(("p", new.path.clone())).await?.take(0)?;
        if let Some(same) = pending.into_iter().map(Proposal::from).find(|p| p.changes == new.changes && p.source == new.source) {
            return Ok((same, false));
        }
        let id = Self::new_id()?;
        let row = ProposalRow {
            id: RecordId::new(TABLE, id.as_str()),
            path: new.path,
            caption: new.changes.caption,
            description: new.changes.description,
            category: new.changes.category,
            tags_add: new.changes.tags_add,
            tags_remove: new.changes.tags_remove,
            reason: new.reason,
            source: new.source,
            status: Status::Pending.as_str().to_string(),
            created_ms: self.clock.unix_ms() as i64,
            decided_ms: None,
            note: None,
        };
        let created: Option<ProposalRow> = db.query("CREATE ONLY $id CONTENT $row RETURN AFTER").bind(("id", RecordId::new(TABLE, id.as_str()))).bind(("row", row)).await?.take(0)?;
        let created = created.ok_or_else(|| ToolError::failed("the proposal was not stored"))?;
        Ok((created.into(), true))
    }

    pub async fn get(&self, id: &str) -> Result<Option<Proposal>, ToolError> {
        let db = self.handle().await?;
        let row: Option<ProposalRow> = db.query("SELECT * FROM ONLY $id").bind(("id", RecordId::new(TABLE, id))).await?.take(0)?;
        Ok(row.map(Into::into))
    }

    /// A page of proposals, newest first, and the number that match.
    pub async fn list(&self, status: Option<Status>, path: Option<&str>, offset: usize, limit: usize) -> Result<(Vec<Proposal>, u64), ToolError> {
        let db = self.handle().await?;
        let mut clauses = Vec::new();
        if status.is_some() {
            clauses.push("status = $status");
        }
        if path.is_some() {
            clauses.push("path = $path");
        }
        let filter = if clauses.is_empty() { String::new() } else { format!("WHERE {}", clauses.join(" AND ")) };
        let mut page = db.query(format!("SELECT * FROM mcp_proposals {filter} ORDER BY created_ms DESC, id DESC LIMIT $l START $o")).bind(("l", limit as i64)).bind(("o", offset as i64));
        let mut count = db.query(format!("SELECT count() AS n FROM mcp_proposals {filter} GROUP ALL"));
        if let Some(s) = status {
            page = page.bind(("status", s.as_str().to_string()));
            count = count.bind(("status", s.as_str().to_string()));
        }
        if let Some(p) = path {
            page = page.bind(("path", p.to_string()));
            count = count.bind(("path", p.to_string()));
        }
        let rows: Vec<ProposalRow> = page.await?.take(0)?;
        let total: Vec<NRow> = count.await?.take(0)?;
        Ok((rows.into_iter().map(Into::into).collect(), total.first().map_or(0, |r| r.n.max(0) as u64)))
    }

    pub async fn counts(&self) -> Result<Counts, ToolError> {
        let db = self.handle().await?;
        let rows: Vec<CountRow> = db.query("SELECT status, count() AS n FROM mcp_proposals GROUP BY status").await?.take(0)?;
        let mut counts = Counts::default();
        for r in rows {
            let n = r.n.max(0) as u64;
            match Status::parse(&r.status) {
                Some(Status::Pending) => counts.pending = n,
                Some(Status::Accepted) => counts.accepted = n,
                Some(Status::Rejected) => counts.rejected = n,
                _ => counts.failed += n,
            }
        }
        Ok(counts)
    }

    async fn decide(&self, id: &str, status: Status, note: Option<String>) -> Result<Proposal, ToolError> {
        let db = self.handle().await?;
        let row: Option<ProposalRow> = db
            .query("UPDATE ONLY $id SET status = $status, decided_ms = $at, note = $note RETURN AFTER")
            .bind(("id", RecordId::new(TABLE, id)))
            .bind(("status", status.as_str().to_string()))
            .bind(("at", self.clock.unix_ms() as i64))
            .bind(("note", note))
            .await?
            .take(0)?;
        row.map(Into::into).ok_or_else(|| ToolError::NotFound(format!("no proposal {id}")))
    }

    async fn pending(&self, id: &str) -> Result<Proposal, ToolError> {
        let proposal = self.get(id).await?.ok_or_else(|| ToolError::NotFound(format!("no proposal {id}")))?;
        if proposal.status != Status::Pending {
            return Err(ToolError::invalid(format!("proposal {id} is already {}", proposal.status.as_str())));
        }
        Ok(proposal)
    }

    pub async fn reject(&self, id: &str) -> Result<Proposal, ToolError> {
        self.pending(id).await?;
        self.decide(id, Status::Rejected, None).await
    }

    /// Writes the proposed changes to the library row and marks the proposal accepted.
    pub async fn accept(&self, id: &str) -> Result<Proposal, ToolError> {
        let proposal = self.pending(id).await?;
        let db = self.handle().await?;
        let row: Option<ItemRow> = db.query("SELECT caption, description, category, tags FROM thumbnails WHERE path = $p").bind(("p", proposal.path.clone())).await?.take(0)?;
        let Some(row) = row else {
            let note = "the file is no longer in the library".to_string();
            self.decide(id, Status::Failed, Some(note.clone())).await?;
            return Err(ToolError::NotFound(note));
        };
        let existing = Existing { caption: row.caption, description: row.description, category: row.category, tags: row.tags.unwrap_or_default() };
        let applied = apply_changes(&existing, &proposal.changes);
        let nothing_to_write = applied.is_empty();
        if !nothing_to_write {
            let mut sets = Vec::new();
            if applied.caption.is_some() {
                sets.push("caption = $caption");
            }
            if applied.description.is_some() {
                sets.push("description = $description");
            }
            if applied.category.is_some() {
                sets.push("category = $category");
            }
            if applied.tags.is_some() {
                sets.push("tags = $tags");
            }
            db.query(format!("UPDATE thumbnails SET {} WHERE path = $path RETURN NONE", sets.join(", ")))
                .bind(("path", proposal.path.clone()))
                .bind(("caption", applied.caption))
                .bind(("description", applied.description))
                .bind(("category", applied.category))
                .bind(("tags", applied.tags))
                .await?
                .check()?;
        }
        let note = nothing_to_write.then(|| "the row already had these values".to_string());
        self.decide(id, Status::Accepted, note).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn normalizing_trims_dedupes_and_limits() {
        let changes = Changes {
            caption: Some("  A sunny \n beach  ".into()),
            description: Some("x".repeat(MAX_DESCRIPTION + 50)),
            category: Some("  ".into()),
            tags_add: strings(&["Beach", "beach", " sun ", "", "t\u{0}ab"]),
            tags_remove: strings(&["old"]),
        }
        .normalized()
        .unwrap();
        assert_eq!(changes.caption.as_deref(), Some("A sunny beach"));
        assert_eq!(changes.description.as_ref().map(|d| d.chars().count()), Some(MAX_DESCRIPTION));
        assert_eq!(changes.category, None);
        assert_eq!(changes.tags_add, strings(&["Beach", "sun", "tab"]));
        assert_eq!(changes.tags_remove, strings(&["old"]));
    }

    #[test]
    fn a_proposal_without_changes_is_invalid() {
        let err = Changes { caption: Some("  ".into()), ..Default::default() }.normalized().unwrap_err();
        assert_eq!(err.kind(), "invalid_arguments");
        assert!(Changes::default().is_empty());
    }

    #[test]
    fn tag_lists_are_capped() {
        let many: Vec<String> = (0..100).map(|i| format!("tag{i}")).collect();
        let changes = Changes { tags_add: many, ..Default::default() }.normalized().unwrap();
        assert_eq!(changes.tags_add.len(), MAX_TAGS);
        let long = Changes { tags_add: vec!["y".repeat(100)], ..Default::default() }.normalized().unwrap();
        assert_eq!(long.tags_add[0].chars().count(), MAX_TAG);
    }

    #[test]
    fn applying_changes_adds_removes_and_skips_equal_values() {
        let existing = Existing { caption: Some("old".into()), description: None, category: Some("beach".into()), tags: strings(&["Sea", "sand", "boat"]) };
        let changes = Changes { caption: Some("old".into()), description: Some("d".into()), category: Some("coast".into()), tags_add: strings(&["SAND", "sun"]), tags_remove: strings(&["BOAT"]) };
        let applied = apply_changes(&existing, &changes);
        assert_eq!(applied.caption, None, "same caption is not a change");
        assert_eq!(applied.description.as_deref(), Some("d"));
        assert_eq!(applied.category.as_deref(), Some("coast"));
        assert_eq!(applied.tags, Some(strings(&["Sea", "sand", "sun"])));
        let noop = apply_changes(&existing, &Changes { tags_add: strings(&["sea"]), ..Default::default() });
        assert!(noop.is_empty());
    }

    #[test]
    fn statuses_round_trip_through_text() {
        for s in [Status::Pending, Status::Accepted, Status::Rejected, Status::Failed] {
            assert_eq!(Status::parse(s.as_str()), Some(s));
        }
        assert_eq!(Status::parse("nope"), None);
    }
}
