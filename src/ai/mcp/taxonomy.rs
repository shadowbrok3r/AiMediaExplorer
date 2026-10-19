//! Tag and category renames and merges proposed by tools, kept in the `mcp_taxonomy_proposals` table until the user accepts or rejects them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use surrealdb::types::{RecordId, SurrealValue};

use super::clock::Clock;
use super::ctx::DbSource;
use super::error::ToolError;
use super::proposals::{Counts, MAX_CATEGORY, MAX_REASON, MAX_TAG, Status};

pub const TABLE: &str = "mcp_taxonomy_proposals";
const DDL: &str = "DEFINE TABLE IF NOT EXISTS mcp_taxonomy_proposals TYPE NORMAL SCHEMALESS PERMISSIONS FULL;";

/// Most names a proposal may merge into one.
pub const MAX_SOURCES: usize = 10;
/// Pending proposals allowed before new ones are refused.
pub const MAX_PENDING: u64 = 100;
/// Longest old name accepted; library rows can hold names longer than the limits for new ones.
const MAX_OLD_NAME: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Tag,
    Category,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::Category => "category",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        [Self::Tag, Self::Category].into_iter().find(|k| k.as_str() == text.trim().to_ascii_lowercase())
    }

    /// Longest new name.
    pub fn max_new_len(self) -> usize {
        match self {
            Self::Tag => MAX_TAG,
            Self::Category => MAX_CATEGORY,
        }
    }
}

/// Text on one line without control characters, cut to `max` characters; `None` when nothing is left.
fn name(text: &str, max: usize) -> Option<String> {
    let line = text.chars().filter(|c| !c.is_control() || c.is_whitespace()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
    (!line.is_empty()).then(|| line.chars().take(max).collect())
}

/// A rename or merge with checked names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Rename {
    pub kind: Kind,
    pub old_names: Vec<String>,
    pub new_name: String,
}

impl Rename {
    /// Cleans the names and checks that the rename means something.
    pub fn new(kind: Kind, old_names: &[String], new_name: &str) -> Result<Self, ToolError> {
        let mut olds: Vec<String> = Vec::new();
        for raw in old_names {
            if let Some(n) = name(raw, MAX_OLD_NAME) {
                if !olds.contains(&n) {
                    olds.push(n);
                }
            }
        }
        if olds.is_empty() {
            return Err(ToolError::invalid(format!("from must name at least one {}", kind.as_str())));
        }
        if olds.len() > MAX_SOURCES {
            return Err(ToolError::invalid(format!("from may name at most {MAX_SOURCES} names; split the merge into several proposals")));
        }
        let new = name(new_name, kind.max_new_len()).ok_or_else(|| ToolError::invalid(format!("to must be a {} name of up to {} characters", kind.as_str(), kind.max_new_len())))?;
        if olds.contains(&new) {
            return Err(ToolError::invalid(format!("'{new}' is both an old and the new name")));
        }
        Ok(Self { kind, old_names: olds, new_name: new })
    }

    /// Words for lists and notices.
    pub fn title(&self) -> String {
        let olds = self.old_names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
        let verb = if self.old_names.len() == 1 { "Rename" } else { "Merge" };
        let prep = if self.old_names.len() == 1 { "to" } else { "into" };
        format!("{verb} {} {olds} {prep} '{}'", self.kind.as_str(), self.new_name)
    }
}

/// What a rename would touch in the library.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Preview {
    /// Items that carry at least one old name.
    pub items: u64,
    /// Of those, items that already carry the new name and merge.
    pub merged: u64,
    /// Items that carry the new name now.
    pub target_items: u64,
    /// Items per old name.
    pub per_name: Vec<NameCount>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NameCount {
    pub name: String,
    pub items: u64,
}

impl Preview {
    /// True when applying the rename would change something.
    pub fn changes_anything(&self) -> bool {
        self.items > 0
    }

    /// The sentence the review shows.
    pub fn summary(&self) -> String {
        let items = |n: u64| if n == 1 { "1 item".to_string() } else { format!("{n} items") };
        if self.items == 0 {
            return "Nothing to change: no item carries the old name".to_string();
        }
        let mut text = format!("{} change", items(self.items));
        if self.merged > 0 {
            text.push_str(&format!("; {} already carry the new name and merge", self.merged));
        }
        text
    }
}

/// A proposal as stored.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaxProposal {
    pub id: String,
    pub kind: Kind,
    pub old_names: Vec<String>,
    pub new_name: String,
    pub reason: Option<String>,
    pub source: String,
    pub status: Status,
    pub created_ms: u64,
    pub decided_ms: Option<u64>,
    pub note: Option<String>,
    /// Items the rename touched when it was filed.
    pub items: u64,
    /// Changes written by an accept.
    pub applied: Option<u64>,
}

impl TaxProposal {
    pub fn rename(&self) -> Rename {
        Rename { kind: self.kind, old_names: self.old_names.clone(), new_name: self.new_name.clone() }
    }
}

#[derive(Clone, Debug, SurrealValue)]
struct TaxRow {
    id: RecordId,
    kind: String,
    old_names: Vec<String>,
    new_name: String,
    reason: Option<String>,
    source: String,
    status: String,
    created_ms: i64,
    decided_ms: Option<i64>,
    note: Option<String>,
    items: i64,
    applied: Option<i64>,
}

#[derive(Debug, SurrealValue)]
struct CountRow {
    n: i64,
}

#[derive(Debug, SurrealValue)]
struct StatusCount {
    status: String,
    n: i64,
}

fn key_of(id: &RecordId) -> String {
    crate::database::collections::key_of(id)
}

impl From<TaxRow> for TaxProposal {
    fn from(r: TaxRow) -> Self {
        Self {
            id: key_of(&r.id),
            kind: Kind::parse(&r.kind).unwrap_or(Kind::Tag),
            old_names: r.old_names,
            new_name: r.new_name,
            reason: r.reason,
            source: r.source,
            status: Status::parse(&r.status).unwrap_or(Status::Failed),
            created_ms: r.created_ms.max(0) as u64,
            decided_ms: r.decided_ms.map(|v| v.max(0) as u64),
            note: r.note,
            items: r.items.max(0) as u64,
            applied: r.applied.map(|v| v.max(0) as u64),
        }
    }
}

/// What a caller gives to file a proposal.
#[derive(Clone, Debug)]
pub struct NewTaxonomy {
    pub rename: Rename,
    pub reason: Option<String>,
    pub source: String,
}

pub struct TaxonomyStore {
    db: DbSource,
    clock: Arc<dyn Clock>,
    schema_ready: AtomicBool,
}

impl TaxonomyStore {
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
        Ok(format!("x{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()))
    }

    /// Runs a count query with the names bound as `$olds`, `$new` and `$old`.
    async fn count(&self, sql: &str, olds: &[String], new: &str, old: &str) -> Result<u64, ToolError> {
        let db = self.db.get()?;
        let rows: Vec<CountRow> = db.query(sql).bind(("olds", olds.to_vec())).bind(("new", new.to_string())).bind(("old", old.to_string())).await?.take(0)?;
        Ok(rows.first().map_or(0, |r| r.n.max(0) as u64))
    }

    /// Counts what `rename` would touch; nothing is written.
    pub async fn preview(&self, rename: &Rename) -> Result<Preview, ToolError> {
        let (olds, new) = (&rename.old_names, rename.new_name.as_str());
        let (items, merged, target_items) = match rename.kind {
            Kind::Tag => (
                self.count("SELECT count() AS n FROM thumbnails WHERE tags CONTAINSANY $olds GROUP ALL", olds, new, "").await?,
                self.count("SELECT count() AS n FROM thumbnails WHERE tags CONTAINSANY $olds AND tags CONTAINS $new GROUP ALL", olds, new, "").await?,
                self.count("SELECT count() AS n FROM thumbnails WHERE tags CONTAINS $new GROUP ALL", olds, new, "").await?,
            ),
            Kind::Category => (
                self.count("SELECT count() AS n FROM thumbnails WHERE category IN $olds GROUP ALL", olds, new, "").await?,
                0,
                self.count("SELECT count() AS n FROM thumbnails WHERE category = $new GROUP ALL", olds, new, "").await?,
            ),
        };
        let mut per_name = Vec::with_capacity(olds.len());
        for old in olds {
            let sql = match rename.kind {
                Kind::Tag => "SELECT count() AS n FROM thumbnails WHERE tags CONTAINS $old GROUP ALL",
                Kind::Category => "SELECT count() AS n FROM thumbnails WHERE category = $old GROUP ALL",
            };
            per_name.push(NameCount { name: old.clone(), items: self.count(sql, olds, new, old).await? });
        }
        Ok(Preview { items, merged, target_items, per_name })
    }

    /// Stores a pending proposal; an identical pending one is returned instead (the flag is false then).
    pub async fn create(&self, new: NewTaxonomy, items: u64) -> Result<(TaxProposal, bool), ToolError> {
        let db = self.handle().await?;
        let pending: Vec<TaxRow> = db.query("SELECT * FROM mcp_taxonomy_proposals WHERE status = 'pending'").await?.take(0)?;
        let mut sorted = new.rename.old_names.clone();
        sorted.sort();
        let same = pending.iter().cloned().map(TaxProposal::from).find(|p| {
            let mut olds = p.old_names.clone();
            olds.sort();
            p.kind == new.rename.kind && olds == sorted && p.new_name == new.rename.new_name && p.source == new.source
        });
        if let Some(same) = same {
            return Ok((same, false));
        }
        if pending.len() as u64 >= MAX_PENDING {
            return Err(ToolError::Busy(format!("{MAX_PENDING} taxonomy proposals already wait for review; ask the user to review them before filing more")));
        }
        let id = Self::new_id()?;
        let row = TaxRow {
            id: RecordId::new(TABLE, id.as_str()),
            kind: new.rename.kind.as_str().to_string(),
            old_names: new.rename.old_names,
            new_name: new.rename.new_name,
            reason: new.reason,
            source: new.source,
            status: Status::Pending.as_str().to_string(),
            created_ms: self.clock.unix_ms() as i64,
            decided_ms: None,
            note: None,
            items: items as i64,
            applied: None,
        };
        let created: Option<TaxRow> = db.query("CREATE ONLY $id CONTENT $row RETURN AFTER").bind(("id", RecordId::new(TABLE, id.as_str()))).bind(("row", row)).await?.take(0)?;
        let created = created.ok_or_else(|| ToolError::failed("the proposal was not stored"))?;
        Ok((created.into(), true))
    }

    pub async fn get(&self, id: &str) -> Result<Option<TaxProposal>, ToolError> {
        let db = self.handle().await?;
        let row: Option<TaxRow> = db.query("SELECT * FROM ONLY $id").bind(("id", RecordId::new(TABLE, id))).await?.take(0)?;
        Ok(row.map(Into::into))
    }

    /// A page of proposals, newest first, and the number that match.
    pub async fn list(&self, status: Option<Status>, offset: usize, limit: usize) -> Result<(Vec<TaxProposal>, u64), ToolError> {
        let db = self.handle().await?;
        let filter = if status.is_some() { "WHERE status = $status" } else { "" };
        let mut page = db.query(format!("SELECT * FROM mcp_taxonomy_proposals {filter} ORDER BY created_ms DESC, id DESC LIMIT $l START $o")).bind(("l", limit as i64)).bind(("o", offset as i64));
        let mut count = db.query(format!("SELECT count() AS n FROM mcp_taxonomy_proposals {filter} GROUP ALL"));
        if let Some(s) = status {
            page = page.bind(("status", s.as_str().to_string()));
            count = count.bind(("status", s.as_str().to_string()));
        }
        let rows: Vec<TaxRow> = page.await?.take(0)?;
        let total: Vec<CountRow> = count.await?.take(0)?;
        Ok((rows.into_iter().map(Into::into).collect(), total.first().map_or(0, |r| r.n.max(0) as u64)))
    }

    pub async fn counts(&self) -> Result<Counts, ToolError> {
        let db = self.handle().await?;
        let rows: Vec<StatusCount> = db.query("SELECT status, count() AS n FROM mcp_taxonomy_proposals GROUP BY status").await?.take(0)?;
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

    async fn decide(&self, id: &str, status: Status, note: Option<String>, applied: Option<u64>) -> Result<TaxProposal, ToolError> {
        let db = self.handle().await?;
        let row: Option<TaxRow> = db
            .query("UPDATE ONLY $id SET status = $status, decided_ms = $at, note = $note, applied = $applied RETURN AFTER")
            .bind(("id", RecordId::new(TABLE, id)))
            .bind(("status", status.as_str().to_string()))
            .bind(("at", self.clock.unix_ms() as i64))
            .bind(("note", note))
            .bind(("applied", applied.map(|n| n as i64)))
            .await?
            .take(0)?;
        row.map(Into::into).ok_or_else(|| ToolError::NotFound(format!("no proposal {id}")))
    }

    async fn pending(&self, id: &str) -> Result<TaxProposal, ToolError> {
        let proposal = self.get(id).await?.ok_or_else(|| ToolError::NotFound(format!("no proposal {id}")))?;
        if proposal.status != Status::Pending {
            return Err(ToolError::invalid(format!("proposal {id} is already {}", proposal.status.as_str())));
        }
        Ok(proposal)
    }

    pub async fn reject(&self, id: &str) -> Result<TaxProposal, ToolError> {
        self.pending(id).await?;
        self.decide(id, Status::Rejected, None, None).await
    }

    /// Applies the rename through the journaled tag or category functions, one undo step per old name, and marks the proposal accepted.
    pub async fn accept(&self, id: &str) -> Result<TaxProposal, ToolError> {
        let proposal = self.pending(id).await?;
        let db = self.handle().await?;
        let mut changed = 0u64;
        for old in &proposal.old_names {
            let done = match proposal.kind {
                Kind::Tag => crate::database::journal::rename_tag_with(db, old, &proposal.new_name).await,
                Kind::Category => crate::database::journal::rename_category_with(db, old, &proposal.new_name).await,
            };
            match done {
                Ok(n) => changed += n,
                Err(e) => {
                    let note = format!("stopped at '{old}' after {changed} changes: {e:#}");
                    self.decide(id, Status::Failed, Some(note.clone()), Some(changed)).await?;
                    return Err(ToolError::Failed(note));
                }
            }
        }
        let note = (changed == 0).then(|| "no item carried the old names any more".to_string());
        self.decide(id, Status::Accepted, note, Some(changed)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn names_are_cleaned_deduplicated_and_limited() {
        let r = Rename::new(Kind::Tag, &strings(&["  sun\tset ", "sun set", "", "dusk\u{0}"]), " sunset ").unwrap();
        assert_eq!(r.old_names, strings(&["sun set", "dusk"]));
        assert_eq!(r.new_name, "sunset");
        let long = Rename::new(Kind::Tag, &strings(&["a"]), &"x".repeat(100)).unwrap();
        assert_eq!(long.new_name.chars().count(), MAX_TAG);
        let cat = Rename::new(Kind::Category, &strings(&["a"]), &"y".repeat(200)).unwrap();
        assert_eq!(cat.new_name.chars().count(), MAX_CATEGORY);
    }

    #[test]
    fn a_rename_needs_names_and_a_new_one() {
        for (from, to) in [(vec![], "b"), (strings(&["  "]), "b"), (strings(&["a"]), "  "), (strings(&["a", "b"]), "b"), (strings(&["a"]), "a")] {
            let err = Rename::new(Kind::Tag, &from, to).unwrap_err();
            assert_eq!(err.kind(), "invalid_arguments", "{from:?} {to:?}");
        }
        let many: Vec<String> = (0..MAX_SOURCES + 1).map(|i| format!("t{i}")).collect();
        assert!(Rename::new(Kind::Tag, &many, "x").unwrap_err().message().contains("at most"));
        assert!(Rename::new(Kind::Tag, &many[..MAX_SOURCES], "x").is_ok());
        assert!(Rename::new(Kind::Tag, &strings(&["Beach"]), "beach").is_ok(), "a change of case is a rename");
    }

    #[test]
    fn titles_say_rename_or_merge() {
        let one = Rename::new(Kind::Tag, &strings(&["a"]), "b").unwrap();
        assert_eq!(one.title(), "Rename tag 'a' to 'b'");
        let many = Rename::new(Kind::Category, &strings(&["a", "b"]), "c").unwrap();
        assert_eq!(many.title(), "Merge category 'a', 'b' into 'c'");
    }

    #[test]
    fn the_summary_counts_items_and_merges() {
        assert!(Preview::default().summary().starts_with("Nothing to change"));
        let p = Preview { items: 4, merged: 1, target_items: 2, per_name: vec![] };
        assert_eq!(p.summary(), "4 items change; 1 already carry the new name and merge");
        assert_eq!(Preview { items: 1, ..Default::default() }.summary(), "1 item change");
    }

    #[test]
    fn kinds_parse_from_text() {
        assert_eq!(Kind::parse(" Tag "), Some(Kind::Tag));
        assert_eq!(Kind::parse("category"), Some(Kind::Category));
        assert_eq!(Kind::parse("folder"), None);
    }
}
