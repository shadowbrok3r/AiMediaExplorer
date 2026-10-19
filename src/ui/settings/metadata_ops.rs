//! Bulk tag and category edits: what each would touch, counted before anything is written, and the runs themselves.

use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::SurrealValue;

use crate::tasks::{TaskKind, TaskSpec};

/// One bulk edit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    RenameCategory { old: String, new: String },
    RenameTag { old: String, new: String },
    DeleteTag { tag: String },
    PruneTags { limit: u32 },
}

/// Largest tag count per item the prune accepts.
pub const MAX_PRUNE_LIMIT: u32 = 64;

impl Op {
    /// Why the edit cannot run, when it cannot.
    pub fn validate(&self) -> Result<(), String> {
        let blank = |s: &str| s.trim().is_empty();
        match self {
            Self::RenameCategory { old, new } | Self::RenameTag { old, new } => {
                let what = if matches!(self, Self::RenameCategory { .. }) { "category" } else { "tag" };
                if blank(old) || blank(new) {
                    return Err(format!("Enter the old and the new {what}"));
                }
                if old.trim() == new.trim() {
                    return Err(format!("The old and the new {what} are the same"));
                }
                Ok(())
            }
            Self::DeleteTag { tag } if blank(tag) => Err("Enter the tag to delete".to_string()),
            Self::DeleteTag { .. } => Ok(()),
            Self::PruneTags { limit } if *limit > MAX_PRUNE_LIMIT => Err(format!("The limit is at most {MAX_PRUNE_LIMIT}")),
            Self::PruneTags { .. } => Ok(()),
        }
    }

    /// The edit with its names trimmed.
    pub fn trimmed(&self) -> Self {
        match self {
            Self::RenameCategory { old, new } => Self::RenameCategory { old: old.trim().to_string(), new: new.trim().to_string() },
            Self::RenameTag { old, new } => Self::RenameTag { old: old.trim().to_string(), new: new.trim().to_string() },
            Self::DeleteTag { tag } => Self::DeleteTag { tag: tag.trim().to_string() },
            Self::PruneTags { limit } => Self::PruneTags { limit: *limit },
        }
    }

    /// Words for the task and the confirmation.
    pub fn title(&self) -> String {
        match self {
            Self::RenameCategory { old, new } => format!("Rename category '{old}' to '{new}'"),
            Self::RenameTag { old, new } => format!("Rename tag '{old}' to '{new}'"),
            Self::DeleteTag { tag } => format!("Delete tag '{tag}'"),
            Self::PruneTags { limit } => format!("Keep at most {limit} tags per item"),
        }
    }
}

/// What an edit would touch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Preview {
    /// Items the edit changes.
    pub affected: u64,
    /// Items that already carry the new name; they merge.
    pub merged: u64,
    /// Tags a prune drops in total.
    pub removed: u64,
}

impl Preview {
    /// The sentence the confirmation shows.
    pub fn summary(&self, op: &Op) -> String {
        let items = |n: u64| if n == 1 { "1 item".to_string() } else { format!("{n} items") };
        match op {
            Op::RenameCategory { new, .. } | Op::RenameTag { new, .. } if self.affected == 0 => format!("Nothing to change: no item matches. {}", self.merged_note(new)).trim().to_string(),
            Op::RenameCategory { new, .. } => format!("{} change category. {}", items(self.affected), self.merged_note(new)).trim().to_string(),
            Op::RenameTag { new, .. } => format!("{} change tag. {}", items(self.affected), self.merged_note(new)).trim().to_string(),
            Op::DeleteTag { .. } if self.affected == 0 => "Nothing to change: no item has this tag".to_string(),
            Op::DeleteTag { tag } => format!("{} lose the tag '{tag}'", items(self.affected)),
            Op::PruneTags { .. } if self.affected == 0 => "Nothing to change: no item has more tags than that".to_string(),
            Op::PruneTags { .. } => format!("{} lose tags, {} tags are dropped in total", items(self.affected), self.removed),
        }
    }

    fn merged_note(&self, new: &str) -> String {
        match self.merged {
            0 => String::new(),
            n => format!("{} already {} '{new}' and merge.", if n == 1 { "1 item".to_string() } else { format!("{n} items") }, if n == 1 { "has" } else { "have" }),
        }
    }

    /// True when running the edit would change something.
    pub fn changes_anything(&self) -> bool {
        self.affected > 0
    }
}

#[derive(Debug, serde::Deserialize, SurrealValue)]
struct CountRow {
    n: i64,
}

async fn count(db: &Surreal<Db>, sql: &str, binds: Vec<(&'static str, String)>) -> anyhow::Result<u64> {
    let mut query = db.query(sql);
    for (name, value) in binds {
        query = query.bind((name, value));
    }
    let rows: Vec<CountRow> = query.await?.take(0)?;
    Ok(rows.first().map_or(0, |r| r.n.max(0) as u64))
}

/// Counts what `op` would touch in `db`; nothing is written.
pub async fn preview(db: &Surreal<Db>, op: &Op) -> anyhow::Result<Preview> {
    op.validate().map_err(anyhow::Error::msg)?;
    let op = op.trimmed();
    let pair = |old: &str, new: &str| vec![("old", old.to_string()), ("new", new.to_string())];
    Ok(match &op {
        Op::RenameTag { old, new } => Preview {
            affected: count(db, "SELECT count() AS n FROM thumbnails WHERE tags CONTAINS $old GROUP ALL", pair(old, new)).await?,
            merged: count(db, "SELECT count() AS n FROM thumbnails WHERE tags CONTAINS $old AND tags CONTAINS $new GROUP ALL", pair(old, new)).await?,
            removed: 0,
        },
        Op::DeleteTag { tag } => Preview {
            affected: count(db, "SELECT count() AS n FROM thumbnails WHERE tags CONTAINS $tag GROUP ALL", vec![("tag", tag.clone())]).await?,
            ..Default::default()
        },
        Op::RenameCategory { old, new } => Preview {
            affected: count(db, "SELECT count() AS n FROM thumbnails WHERE category = $old GROUP ALL", pair(old, new)).await?,
            merged: count(db, "SELECT count() AS n FROM thumbnails WHERE category = $new GROUP ALL", pair(old, new)).await?,
            removed: 0,
        },
        Op::PruneTags { limit } => {
            let lengths: Vec<i64> = db
                .query("SELECT VALUE array::len(tags) FROM thumbnails WHERE array::len(tags) > $limit")
                .bind(("limit", i64::from(*limit)))
                .await?
                .take(0)?;
            Preview { affected: lengths.len() as u64, merged: 0, removed: lengths.iter().map(|n| (*n - i64::from(*limit)).max(0) as u64).sum() }
        }
    })
}

/// Runs `op` on the library as a task; returns the number of items it changed.
pub async fn run(op: Op) -> anyhow::Result<u64> {
    op.validate().map_err(anyhow::Error::msg)?;
    let op = op.trimmed();
    let spec = TaskSpec::new(TaskKind::Database, op.title()).user_initiated();
    crate::tasks::track(spec, |_| async move {
        match &op {
            Op::RenameCategory { old, new } => crate::Thumbnail::rename_category(old, new).await,
            Op::RenameTag { old, new } => crate::Thumbnail::rename_tag(old, new).await,
            Op::DeleteTag { tag } => crate::Thumbnail::delete_tag(tag).await,
            Op::PruneTags { limit } => crate::Thumbnail::prune_tags(i64::from(*limit)).await,
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::collections::tests::db;
    use crate::database::scope_queries::tests::{insert, seed};
    use crate::database::thumbnails::{DELETE_TAG_SQL, RENAME_TAG_SQL};

    async fn library() -> Surreal<Db> {
        let db = db().await;
        let mut rows: Vec<_> = (0..6).map(|i| seed(i, &format!("/lib/p{i}.jpg"))).collect();
        rows[0].tags = vec!["sun".into(), "sea".into(), "sand".into()];
        rows[1].tags = vec!["sun".into()];
        rows[2].tags = vec!["sea".into(), "sun".into()];
        rows[3].tags = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
        rows[0].category = Some("beach".into());
        rows[1].category = Some("beach".into());
        rows[2].category = Some("coast".into());
        insert(&db, rows).await;
        db
    }

    fn rename(old: &str, new: &str) -> Op {
        Op::RenameTag { old: old.into(), new: new.into() }
    }

    #[test]
    fn edits_without_names_or_with_equal_names_are_refused() {
        assert!(rename("", "b").validate().is_err());
        assert!(rename("a", "  ").validate().is_err());
        assert!(rename("a", " a ").validate().is_err());
        assert!(Op::RenameCategory { old: "x".into(), new: "x".into() }.validate().is_err());
        assert!(Op::DeleteTag { tag: " ".into() }.validate().is_err());
        assert!(Op::PruneTags { limit: MAX_PRUNE_LIMIT + 1 }.validate().is_err());
        assert!(rename("a", "b").validate().is_ok());
        assert!(Op::PruneTags { limit: 0 }.validate().is_ok());
    }

    #[test]
    fn names_are_trimmed_before_they_run() {
        assert_eq!(rename(" a ", " b").trimmed(), rename("a", "b"));
        assert_eq!(Op::DeleteTag { tag: " x ".into() }.trimmed(), Op::DeleteTag { tag: "x".into() });
    }

    #[test]
    fn the_summary_names_the_counts_and_the_merge() {
        let op = rename("sun", "sea");
        assert_eq!(Preview { affected: 3, merged: 2, removed: 0 }.summary(&op), "3 items change tag. 2 items already have 'sea' and merge.");
        assert_eq!(Preview { affected: 1, merged: 1, removed: 0 }.summary(&op), "1 item change tag. 1 item already has 'sea' and merge.");
        assert!(Preview::default().summary(&op).starts_with("Nothing to change"));
        assert!(!Preview::default().changes_anything());
        let prune = Op::PruneTags { limit: 2 };
        assert_eq!(Preview { affected: 4, merged: 0, removed: 9 }.summary(&prune), "4 items lose tags, 9 tags are dropped in total");
        assert_eq!(Preview { affected: 2, ..Default::default() }.summary(&Op::DeleteTag { tag: "x".into() }), "2 items lose the tag 'x'");
    }

    #[tokio::test]
    async fn a_tag_rename_preview_matches_what_the_rename_changes() {
        let db = library().await;
        let p = preview(&db, &rename("sun", "sea")).await.unwrap();
        assert_eq!((p.affected, p.merged), (3, 2));
        let renamed: Vec<crate::Thumbnail> = db.query(RENAME_TAG_SQL).bind(("old", "sun".to_string())).bind(("new", "sea".to_string())).await.unwrap().take(0).unwrap();
        assert_eq!(renamed.len() as u64, p.affected, "the preview counts exactly the rows the update returns");
        assert_eq!(preview(&db, &rename("sun", "sea")).await.unwrap().affected, 0, "after the rename nothing carries the old tag");
    }

    #[tokio::test]
    async fn a_delete_preview_counts_the_items_with_the_tag() {
        let db = library().await;
        let p = preview(&db, &Op::DeleteTag { tag: " sea ".into() }).await.unwrap();
        assert_eq!(p, Preview { affected: 2, merged: 0, removed: 0 });
        let deleted: Vec<crate::Thumbnail> = db.query(DELETE_TAG_SQL).bind(("tag", "sea".to_string())).await.unwrap().take(0).unwrap();
        assert_eq!(deleted.len(), 2);
        assert_eq!(preview(&db, &Op::DeleteTag { tag: "nothing".into() }).await.unwrap(), Preview::default());
    }

    #[tokio::test]
    async fn a_category_preview_counts_the_items_and_the_ones_already_there() {
        let db = library().await;
        let p = preview(&db, &Op::RenameCategory { old: "beach".into(), new: "coast".into() }).await.unwrap();
        assert_eq!((p.affected, p.merged), (2, 1));
        let p = preview(&db, &Op::RenameCategory { old: "nope".into(), new: "coast".into() }).await.unwrap();
        assert_eq!(p.affected, 0);
    }

    #[tokio::test]
    async fn a_prune_preview_counts_items_and_dropped_tags() {
        let db = library().await;
        let p = preview(&db, &Op::PruneTags { limit: 1 }).await.unwrap();
        assert_eq!(p.affected, 3, "three items have more than one tag");
        assert_eq!(p.removed, 2 + 1 + 4);
        let p = preview(&db, &Op::PruneTags { limit: 3 }).await.unwrap();
        assert_eq!((p.affected, p.removed), (1, 2));
        assert_eq!(preview(&db, &Op::PruneTags { limit: 10 }).await.unwrap(), Preview::default());
    }

    #[tokio::test]
    async fn an_invalid_edit_never_reaches_the_database() {
        let db = library().await;
        assert!(preview(&db, &rename("a", "a")).await.is_err());
        assert!(run(Op::DeleteTag { tag: "".into() }).await.is_err());
    }
}
