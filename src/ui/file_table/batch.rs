//! Library edits of the action bar: batch tags, category, collections and removal, as async functions that report one line of counts.

use std::collections::HashMap;

use surrealdb::engine::local::Db;
use surrealdb::{Connection, Surreal};

use super::menus::logical_groups::{self, GroupAction};
use crate::Thumbnail;
use crate::database::collections;
use crate::database::metadata_edit::{self, RowSeed, TagChange};

/// The file rows among `rows`: no folders, no rows without a path.
pub fn files_of(rows: &[&Thumbnail]) -> Vec<Thumbnail> {
    rows.iter().filter(|r| r.file_type != "<DIR>" && !r.path.is_empty()).map(|r| (*r).clone()).collect()
}

pub fn seeds(files: &[Thumbnail]) -> Vec<RowSeed> {
    files.iter().map(RowSeed::from_row).collect()
}

/// "the tag "x"" for one tag, "N tags" for several.
pub fn describe_tags(tags: &[String]) -> String {
    match tags {
        [one] => format!("the tag \"{one}\""),
        many => format!("{} tags", many.len()),
    }
}

/// "N file" or "N files".
pub fn files_noun(n: usize) -> String {
    noun(n, "file")
}

fn noun(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

fn created_note(created: usize) -> String {
    if created == 0 { String::new() } else { format!("; {} added to the library", files_noun(created)) }
}

/// The tags each file would have after `change`, from the rows as shown.
pub fn local_tags(files: &[Thumbnail], change: &TagChange) -> HashMap<String, Vec<String>> {
    files.iter().map(|r| (r.path.clone(), metadata_edit::apply_tag_change(&r.tags, change))).collect()
}

/// What a tag batch did.
#[derive(Clone, Debug, PartialEq)]
pub struct TagResult {
    pub message: String,
    /// Tags now stored per path, for rows whose tags changed.
    pub tags: HashMap<String, Vec<String>>,
}

/// Adds and removes tags on `files`, creating their library rows when missing.
pub async fn tag_files<C: Connection>(db: &Surreal<C>, files: &[Thumbnail], change: &TagChange) -> anyhow::Result<TagResult> {
    anyhow::ensure!(!files.is_empty(), "no files to tag");
    anyhow::ensure!(!change.is_empty(), "no tags to add or remove");
    let report = metadata_edit::batch_tags(db, &seeds(files), change).await?;
    let verb = match (change.add.is_empty(), change.remove.is_empty()) {
        (false, true) => format!("Added {} to {}", describe_tags(&change.add), files_noun(report.updated)),
        (true, false) => format!("Removed {} from {}", describe_tags(&change.remove), files_noun(report.updated)),
        _ => format!("Changed the tags of {}", files_noun(report.updated)),
    };
    let unchanged = files.len().saturating_sub(report.updated);
    let rest = if unchanged > 0 { format!("; {} had no change", files_noun(unchanged)) } else { String::new() };
    let tags = if report.tags.is_empty() { local_tags(files, change) } else { report.tags };
    Ok(TagResult { message: format!("{verb}{rest}{}", created_note(report.created)), tags })
}

/// Sets, or with `None` clears, the category of `files`.
pub async fn categorize_files<C: Connection>(db: &Surreal<C>, files: &[Thumbnail], category: Option<&str>) -> anyhow::Result<String> {
    anyhow::ensure!(!files.is_empty(), "no files to change");
    let report = metadata_edit::batch_category(db, &seeds(files), category).await?;
    let verb = match category.map(str::trim).filter(|c| !c.is_empty()) {
        Some(c) => format!("Set the category \"{c}\" on {}", files_noun(report.updated)),
        None => format!("Cleared the category of {}", files_noun(report.updated)),
    };
    Ok(format!("{verb}{}", created_note(report.created)))
}

/// Adds `files` to the collection `name` (created when absent) or removes them from it.
pub async fn collect_files(db: &Surreal<Db>, files: Vec<Thumbnail>, name: &str, add: bool) -> anyhow::Result<String> {
    anyhow::ensure!(!files.is_empty(), "no files to change");
    let (clean, _) = collections::normalize(name)?;
    logical_groups::apply_group_action(db, if add { GroupAction::Add } else { GroupAction::Remove }, &clean, files).await
}

/// Removes `files` and their CLIP embeddings from the library; files on disk and collection memberships stay.
pub async fn remove_files(files: &[Thumbnail]) -> anyhow::Result<String> {
    anyhow::ensure!(!files.is_empty(), "no files to remove");
    let ids = files.iter().map(|r| r.id.clone()).collect();
    let paths: Vec<String> = files.iter().map(|r| r.path.clone()).collect();
    let (embeddings, rows) = super::menus::db_delete::delete_rows(ids, paths).await?;
    let missing = files.len().saturating_sub(rows);
    let rest = if missing > 0 { format!("; {missing} not in it") } else { String::new() };
    Ok(format!("Removed {} from the library ({} deleted){rest}", files_noun(rows), noun(embeddings, "CLIP embedding")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::collections::tests::{db, insert_rows, seed_row};

    fn file(path: &str, tags: &[&str]) -> Thumbnail {
        let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
        t.path = path.to_string();
        t.file_type = "jpg".into();
        t.tags = tags.iter().map(|s| s.to_string()).collect();
        t.parent_dir = path.rsplit_once('/').map_or("", |(d, _)| d).to_string();
        t
    }

    fn change(add: &[&str], remove: &[&str]) -> TagChange {
        TagChange { add: add.iter().map(|s| s.to_string()).collect(), remove: remove.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn folders_and_empty_paths_are_not_files() {
        let mut dir = file("/p/sub", &[]);
        dir.file_type = "<DIR>".into();
        let blank = file("", &[]);
        let a = file("/p/a.jpg", &[]);
        let rows = [&dir, &blank, &a];
        assert_eq!(files_of(&rows).len(), 1);
        assert!(files_of(&[]).is_empty());
    }

    #[test]
    fn the_counts_read_in_words() {
        assert_eq!(files_noun(1), "1 file");
        assert_eq!(files_noun(0), "0 files");
        assert_eq!(describe_tags(&["a".into()]), "the tag \"a\"");
        assert_eq!(describe_tags(&["a".into(), "b".into()]), "2 tags");
    }

    #[test]
    fn local_tags_follow_the_change_without_touching_the_database() {
        let files = [file("/p/a.jpg", &["old", "keep"]), file("/p/b.jpg", &[])];
        let tags = local_tags(&files, &change(&["new"], &["old"]));
        assert_eq!(tags["/p/a.jpg"], ["keep", "new"]);
        assert_eq!(tags["/p/b.jpg"], ["new"]);
    }

    #[tokio::test]
    async fn adding_tags_reports_counts_and_creates_missing_library_rows() {
        let db = db().await;
        insert_rows(&db, vec![seed_row(1, "/t/a.jpg")]).await;
        let files = [file("/t/a.jpg", &[]), file("/t/b.jpg", &[])];
        let result = tag_files(&db, &files, &change(&["sun", "sea"], &[])).await.unwrap();
        assert_eq!(result.message, "Added 2 tags to 2 files; 1 file added to the library");
        assert_eq!(result.tags["/t/a.jpg"], ["sun", "sea"]);
        let again = tag_files(&db, &files, &change(&["sun"], &[])).await.unwrap();
        assert_eq!(again.message, "Added the tag \"sun\" to 0 files; 2 files had no change");
    }

    #[tokio::test]
    async fn a_batch_tag_change_is_one_undoable_journal_batch() {
        let db = db().await;
        insert_rows(&db, vec![seed_row(1, "/u/a.jpg"), seed_row(2, "/u/b.jpg")]).await;
        let files = [file("/u/a.jpg", &[]), file("/u/b.jpg", &[])];
        tag_files(&db, &files, &change(&["sun"], &[])).await.unwrap();
        let tags_of = |path: &'static str| {
            let db = db.clone();
            async move {
                let rows: Vec<Vec<String>> = db.query("SELECT VALUE tags FROM thumbnails WHERE path = $p").bind(("p", path)).await.unwrap().take(0).unwrap();
                rows.into_iter().next().unwrap_or_default()
            }
        };
        assert_eq!(tags_of("/u/a.jpg").await, ["sun"]);
        let report = crate::database::journal::undo_last_with(&db).await.unwrap();
        assert_eq!(report.applied.len(), 2, "both files revert in one batch");
        assert!(tags_of("/u/a.jpg").await.is_empty() && tags_of("/u/b.jpg").await.is_empty());
        crate::database::journal::redo_last_with(&db).await.unwrap();
        assert_eq!(tags_of("/u/b.jpg").await, ["sun"]);
    }

    #[tokio::test]
    async fn removing_a_tag_from_some_files_counts_only_those_that_had_it() {
        let db = db().await;
        let files = [file("/t/a.jpg", &[]), file("/t/b.jpg", &[])];
        tag_files(&db, &files[..1], &change(&["x"], &[])).await.unwrap();
        let result = tag_files(&db, &files, &change(&[], &["x"])).await.unwrap();
        assert_eq!(result.message, "Removed the tag \"x\" from 1 file; 1 file had no change; 1 file added to the library");
    }

    #[tokio::test]
    async fn an_empty_batch_is_an_error_not_a_change() {
        let db = db().await;
        assert!(tag_files(&db, &[], &change(&["x"], &[])).await.is_err());
        assert!(tag_files(&db, &[file("/t/a.jpg", &[])], &change(&[], &[])).await.is_err());
        assert!(categorize_files(&db, &[], Some("x")).await.is_err());
        assert!(remove_files(&[]).await.is_err());
        assert!(collect_files(&db, Vec::new(), "X", true).await.is_err());
    }

    #[tokio::test]
    async fn the_category_is_set_and_cleared_with_counts() {
        let db = db().await;
        insert_rows(&db, vec![seed_row(1, "/t/a.jpg"), seed_row(2, "/t/b.jpg")]).await;
        let files = [file("/t/a.jpg", &[]), file("/t/b.jpg", &[])];
        assert_eq!(categorize_files(&db, &files, Some(" Beach ")).await.unwrap(), "Set the category \"Beach\" on 2 files");
        let stored: Vec<Option<String>> = db.query("SELECT VALUE category FROM thumbnails ORDER BY path").await.unwrap().take(0).unwrap();
        assert_eq!(stored, vec![Some("Beach".to_string()); 2]);
        assert_eq!(categorize_files(&db, &files, None).await.unwrap(), "Cleared the category of 2 files");
    }

    #[tokio::test]
    async fn collections_take_and_release_files_and_are_created_on_demand() {
        let db = db().await;
        let files = vec![file("/c/a.jpg", &[]), file("/c/b.jpg", &[])];
        let added = collect_files(&db, files.clone(), "  Trip ", true).await.unwrap();
        assert!(added.contains("Added 2 files to \"Trip\"") && added.contains("new collection"), "{added}");
        let again = collect_files(&db, files.clone(), "trip", true).await.unwrap();
        assert!(again.contains("Added 0 files") && again.contains("2 were already there"), "{again}");
        let removed = collect_files(&db, files[..1].to_vec(), "Trip", false).await.unwrap();
        assert!(removed.contains("Removed 1 files"), "{removed}");
        assert!(collect_files(&db, files, "   ", true).await.is_err(), "a blank name is refused");
    }

    #[tokio::test]
    async fn removal_deletes_the_rows_of_the_given_paths_only() {
        let _serial = super::super::menus::db_delete::GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        crate::settings::testing::mem_db();
        let db = &*crate::database::DB;
        db.query(
            r#"
            CREATE thumbnails:r1 SET path = '/rm/1.jpg', filename = '1.jpg';
            CREATE thumbnails:r2 SET path = '/rm/2.jpg', filename = '2.jpg';
            CREATE clip_embeddings:re1 SET path = '/rm/1.jpg', thumb_ref = thumbnails:r1, embedding = [0.1];
            "#,
        )
        .await
        .unwrap()
        .check()
        .unwrap();
        let mut gone = file("/rm/1.jpg", &[]);
        gone.id = surrealdb::types::RecordId::new("thumbnails", "other-id");
        let unknown = file("/rm/none.jpg", &[]);
        let message = remove_files(&[gone, unknown]).await.unwrap();
        assert_eq!(message, "Removed 1 file from the library (1 CLIP embedding deleted); 1 not in it");
        let left: Vec<String> = db.query("SELECT VALUE path FROM thumbnails WHERE path CONTAINS '/rm/'").await.unwrap().take(0).unwrap();
        assert_eq!(left, vec!["/rm/2.jpg".to_string()], "a row is found by its path even when the id on screen differs");
        db.query("DELETE thumbnails WHERE path CONTAINS '/rm/'; DELETE clip_embeddings WHERE path CONTAINS '/rm/'").await.unwrap().check().unwrap();
    }
}
