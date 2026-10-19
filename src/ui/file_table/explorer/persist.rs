//! Saving rows found by a scan or chosen in the view: one batched write, and one collection lookup and membership write per batch.

use std::collections::HashSet;

use surrealdb::types::{Datetime, SurrealValue};
use surrealdb::{Connection, Surreal};

use crate::database::{THUMBNAILS, Thumbnail, collections};

/// Rows looked up or written per query.
const CHUNK: usize = 400;

/// What a save did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Persisted {
    /// Rows written that were not in the library.
    pub inserted: usize,
    /// Rows that were already there and only had their size and date refreshed.
    pub existing: usize,
    /// Memberships added to the collection.
    pub filed: usize,
    /// The collection was named but does not exist.
    pub collection_missing: bool,
}

#[derive(Clone, Debug, SurrealValue)]
struct Stat {
    path: String,
    size: u64,
    modified: Option<Datetime>,
}

async fn existing_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<HashSet<String>> {
    let mut found = HashSet::new();
    for chunk in paths.chunks(CHUNK) {
        let rows: Vec<String> = db.query("SELECT VALUE path FROM thumbnails WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.take(0)?;
        found.extend(rows);
    }
    Ok(found)
}

/// Saves `rows` to the library, inserting new ones together and refreshing size and date of known ones, and files them in `collection` when named.
pub async fn persist_scanned<C: Connection>(db: &Surreal<C>, rows: Vec<Thumbnail>, collection: Option<&str>) -> anyhow::Result<Persisted> {
    let mut seen = HashSet::new();
    let rows: Vec<Thumbnail> = rows.into_iter().filter(|r| !r.path.is_empty() && r.file_type != "<DIR>" && seen.insert(r.path.clone())).collect();
    let mut report = Persisted::default();
    if rows.is_empty() {
        return Ok(report);
    }
    let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
    let existing = existing_paths(db, &paths).await?;
    let (old, new): (Vec<Thumbnail>, Vec<Thumbnail>) = rows.into_iter().partition(|r| existing.contains(&r.path));
    let (with_pixels, plain): (Vec<Thumbnail>, Vec<Thumbnail>) = new.into_iter().partition(|r| r.thumbnail_b64.as_deref().is_some_and(|s| !s.trim().is_empty()));
    for chunk in plain.chunks(CHUNK) {
        let _: Vec<Thumbnail> = db.insert(THUMBNAILS).content::<Vec<Thumbnail>>(chunk.to_vec()).await?;
        report.inserted += chunk.len();
    }
    if !with_pixels.is_empty() {
        report.inserted += with_pixels.len();
        crate::database::upsert_rows_and_get_ids(with_pixels).await?;
    }
    for chunk in old.chunks(CHUNK) {
        let stats: Vec<Stat> = chunk.iter().map(|r| Stat { path: r.path.clone(), size: r.size, modified: r.modified.clone() }).collect();
        db.query("FOR $r IN $stats { UPDATE thumbnails SET size = $r.size, modified = $r.modified ?? modified WHERE path = $r.path; }")
            .bind(("stats", stats))
            .await?
            .check()?;
        report.existing += chunk.len();
    }
    if let Some(name) = collection {
        match collections::get_by_name(db, name).await? {
            Some(found) => report.filed = collections::add_paths(db, &found.id, &paths).await?.added,
            None => report.collection_missing = true,
        }
    }
    Ok(report)
}

/// One line for the toast after a save.
pub fn saved_summary(done: &Persisted, collection: Option<&str>) -> String {
    let mut text = format!("Saved {} new files to the library; {} were already there", done.inserted, done.existing);
    match collection {
        Some(name) if done.collection_missing => text.push_str(&format!(". The collection \"{name}\" does not exist, so nothing was filed")),
        Some(name) => text.push_str(&format!(". Added {} to \"{name}\"", done.filed)),
        None => {}
    }
    text
}

/// Rows collected from finished thumbnails before they are saved together.
const PENDING_FLUSH: usize = 64;

impl crate::ui::file_table::FileExplorer {
    /// Saves rows in the background to the library and the active collection; the collection is read once for the batch.
    pub(crate) fn queue_persist(&self, rows: Vec<Thumbnail>) {
        if rows.is_empty() {
            return;
        }
        self.viewer.badges.mark_indexed(rows.iter().map(|r| r.path.as_str()));
        let collection = crate::command::context::active_collection();
        tokio::spawn(async move {
            match persist_scanned(&crate::database::DB, rows, collection.as_deref()).await {
                Ok(done) if done.collection_missing => log::warn!("[scan] the active collection {collection:?} no longer exists; rows were saved without it"),
                Ok(done) => log::debug!("[scan] saved {} new rows, refreshed {}, filed {}", done.inserted, done.existing, done.filed),
                Err(e) => log::error!("[scan] saving rows failed: {e:#}"),
            }
        });
    }

    /// Saves `rows` to the library and the active collection and toasts what was saved.
    pub fn save_rows_to_library(&self, rows: Vec<Thumbnail>) {
        if rows.is_empty() {
            crate::ui::dialogs::toast(egui_toast::ToastKind::Info, "There are no files to save".to_string());
            return;
        }
        self.viewer.badges.mark_indexed(rows.iter().map(|r| r.path.as_str()));
        let collection = crate::command::context::active_collection();
        tokio::spawn(async move {
            let (kind, text) = match persist_scanned(&crate::database::DB, rows, collection.as_deref()).await {
                Ok(done) => (egui_toast::ToastKind::Success, saved_summary(&done, collection.as_deref())),
                Err(e) => (egui_toast::ToastKind::Error, format!("Saving to the library failed: {e:#}")),
            };
            crate::ui::dialogs::toast(kind, text);
        });
    }

    /// Saves the rows whose thumbnails finished once enough have gathered, or all of them with `force`; nothing is saved while auto save is off.
    pub(crate) fn flush_pending_rows(&mut self, force: bool) {
        if self.pending_thumb_rows.is_empty() || (!force && self.pending_thumb_rows.len() < PENDING_FLUSH) {
            return;
        }
        let batch = std::mem::take(&mut self.pending_thumb_rows);
        if self.viewer.ui_settings.auto_save_to_database {
            self.queue_persist(batch);
        } else {
            log::debug!("Skipping batch DB save: auto_save_to_database disabled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::collections::tests::db;

    fn row(path: &str, size: u64) -> Thumbnail {
        let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
        t.path = path.to_string();
        t.file_type = "jpg".into();
        t.size = size;
        t.parent_dir = path.rsplit_once('/').map_or("", |(d, _)| d).to_string();
        t
    }

    async fn stored(db: &Surreal<surrealdb::engine::local::Db>) -> Vec<(String, u64)> {
        #[derive(serde::Deserialize, SurrealValue)]
        struct P {
            path: String,
            size: u64,
        }
        let rows: Vec<P> = db.query("SELECT path, size FROM thumbnails ORDER BY path").await.unwrap().take(0).unwrap();
        rows.into_iter().map(|p| (p.path, p.size)).collect()
    }

    #[tokio::test]
    async fn new_rows_are_inserted_together_and_existing_ones_are_refreshed_not_duplicated() {
        let db = db().await;
        let first = persist_scanned(&db, vec![row("/p/a.jpg", 1), row("/p/b.jpg", 2)], None).await.unwrap();
        assert_eq!((first.inserted, first.existing), (2, 0));
        let again = persist_scanned(&db, vec![row("/p/a.jpg", 10), row("/p/b.jpg", 2), row("/p/c.jpg", 3)], None).await.unwrap();
        assert_eq!((again.inserted, again.existing), (1, 2));
        assert_eq!(stored(&db).await, vec![("/p/a.jpg".into(), 10), ("/p/b.jpg".into(), 2), ("/p/c.jpg".into(), 3)]);
    }

    #[tokio::test]
    async fn a_refresh_keeps_the_metadata_of_existing_rows() {
        let db = db().await;
        let mut known = row("/p/a.jpg", 1);
        known.description = Some("a cat".into());
        known.tags = vec!["pet".into()];
        persist_scanned(&db, vec![known], None).await.unwrap();
        persist_scanned(&db, vec![row("/p/a.jpg", 99)], None).await.unwrap();
        let rows: Vec<Thumbnail> = db.query(format!("SELECT {} FROM thumbnails", crate::database::thumbnails::row_cols())).await.unwrap().take(0).unwrap();
        assert_eq!((rows[0].size, rows[0].description.as_deref(), rows[0].tags.clone()), (99, Some("a cat"), vec!["pet".to_string()]));
    }

    #[tokio::test]
    async fn duplicates_empty_paths_and_folders_are_left_out() {
        let db = db().await;
        let mut dir = row("/p/sub", 0);
        dir.file_type = "<DIR>".into();
        let report = persist_scanned(&db, vec![row("/p/a.jpg", 1), row("/p/a.jpg", 1), row("", 5), dir], None).await.unwrap();
        assert_eq!(report.inserted, 1);
        assert_eq!(stored(&db).await.len(), 1);
        assert_eq!(persist_scanned(&db, Vec::new(), Some("X")).await.unwrap(), Persisted::default());
    }

    #[tokio::test]
    async fn a_batch_is_filed_in_the_collection_with_one_lookup_and_reports_the_count() {
        let db = db().await;
        let c = collections::create(&db, "Trip").await.unwrap();
        let rows: Vec<Thumbnail> = (0..CHUNK + 50).map(|i| row(&format!("/t/{i:05}.jpg"), i as u64 + 1)).collect();
        let report = persist_scanned(&db, rows.clone(), Some("trip")).await.unwrap();
        assert_eq!((report.inserted, report.filed, report.collection_missing), (CHUNK + 50, CHUNK + 50, false));
        assert_eq!(collections::count_members(&db, &c.id).await.unwrap(), (CHUNK + 50) as u64);
        let again = persist_scanned(&db, rows, Some("Trip")).await.unwrap();
        assert_eq!((again.inserted, again.existing, again.filed), (0, CHUNK + 50, 0), "memberships are not repeated");
    }

    #[tokio::test]
    async fn an_unknown_collection_is_reported_and_the_rows_are_still_saved() {
        let db = db().await;
        let report = persist_scanned(&db, vec![row("/p/a.jpg", 1)], Some("Gone")).await.unwrap();
        assert!(report.collection_missing);
        assert_eq!((report.inserted, report.filed), (1, 0));
        assert_eq!(stored(&db).await.len(), 1);
    }
}
