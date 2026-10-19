//! Many-to-many collections: named sets of file paths in `collections` (keyed by lowercase name) and `collection_items` (keyed by a hash of collection key and path).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

use anyhow::Context;
use chrono::{DateTime, Utc};
use egui_theme_kit::store::{NameError, clean_name};
use serde::Serialize;
use surrealdb::types::{Datetime, RecordId, RecordIdKey, SurrealValue, ToSql};
use surrealdb::{Connection, Surreal};

use super::files::Thumbnail;
use super::thumb_blobs::retry_conflicts;

pub const COLLECTIONS: &str = "collections";
pub const ITEMS: &str = "collection_items";
/// Name of the legacy group that means "not filed anywhere"; it is never a collection.
pub const UNFILED_NAME: &str = "Default";
/// Paths written per statement.
pub const ADD_CHUNK: usize = 1000;
const LOOKUP_CHUNK: usize = 500;
const MOVE_CHUNK: usize = 2000;
const PAGE: usize = 2000;
const PROBE_PREFIX: &str = "zz-probe-";

const DDL: &str = r#"
    DEFINE TABLE IF NOT EXISTS collections TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS name ON collections TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS created ON collections TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS updated ON collections TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS color ON collections TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS note ON collections TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS legacy_key ON collections TYPE option<string> PERMISSIONS FULL;
    DEFINE INDEX IF NOT EXISTS collections_legacy_key_idx ON collections FIELDS legacy_key;

    DEFINE TABLE IF NOT EXISTS collection_items TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS collection ON collection_items TYPE record<collections> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS path ON collection_items TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS added ON collection_items TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
    DEFINE INDEX IF NOT EXISTS collection_items_unique_idx ON collection_items FIELDS collection, path UNIQUE;
    DEFINE INDEX IF NOT EXISTS collection_items_path_idx ON collection_items FIELDS path;
    DEFINE INDEX IF NOT EXISTS collection_items_collection_idx ON collection_items FIELDS collection;

    DEFINE TABLE IF NOT EXISTS collection_migration TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS state ON collection_migration TYPE object FLEXIBLE PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS updated ON collection_migration TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
"#;

/// Defines the collection tables on a handle already pointed at NS/DB; safe to run on every start.
pub async fn ensure_schema<C: Connection>(db: &Surreal<C>) -> anyhow::Result<()> {
    db.query(DDL).await?.check().context("defining the collection tables failed")?;
    Ok(())
}

/// Why a collection operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollectionError {
    Name(NameError),
    Reserved,
    Exists(String),
    NotFound(String),
}

impl fmt::Display for CollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(e) => write!(f, "{e}"),
            Self::Reserved => write!(f, "\"{UNFILED_NAME}\" is reserved for files that are in no collection"),
            Self::Exists(n) => write!(f, "a collection named \"{n}\" already exists"),
            Self::NotFound(n) => write!(f, "there is no collection \"{n}\""),
        }
    }
}

impl std::error::Error for CollectionError {}

/// The display name and record key for `name`, or why it is not usable.
pub fn normalize(name: &str) -> Result<(String, String), CollectionError> {
    let display = clean_name(name).map_err(CollectionError::Name)?;
    let key = display.to_lowercase();
    if key == UNFILED_NAME.to_lowercase() {
        return Err(CollectionError::Reserved);
    }
    Ok((display, key))
}

pub fn collection_id(key: &str) -> RecordId {
    RecordId::new(COLLECTIONS, key)
}

pub fn is_collection_id(id: &RecordId) -> bool {
    id.table.as_str() == COLLECTIONS
}

/// The record key of `id` as text.
pub fn key_of(id: &RecordId) -> String {
    match &id.key {
        RecordIdKey::String(s) => s.clone(),
        RecordIdKey::Number(n) => n.to_string(),
        RecordIdKey::Uuid(u) => u.to_string(),
        other => other.to_sql(),
    }
}

/// Record key of the membership of `path` in the collection with key `collection_key`.
pub fn item_key(collection_key: &str, path: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(collection_key.as_bytes());
    hasher.update(&[0]);
    hasher.update(path.as_bytes());
    hasher.finalize().as_bytes()[..16].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn item_id(collection_key: &str, path: &str) -> RecordId {
    RecordId::new(ITEMS, item_key(collection_key, path))
}

#[derive(Clone, Debug, SurrealValue)]
struct CollectionRow {
    id: RecordId,
    name: String,
    created: Option<Datetime>,
    updated: Option<Datetime>,
    color: Option<String>,
    note: Option<String>,
    legacy_key: Option<String>,
}

#[derive(Clone, Debug, SurrealValue)]
struct CollectionIn {
    name: String,
    created: Option<Datetime>,
    updated: Option<Datetime>,
    color: Option<String>,
    note: Option<String>,
    legacy_key: Option<String>,
}

#[derive(Clone, Debug, SurrealValue)]
struct ItemIn {
    id: RecordId,
    collection: RecordId,
    path: String,
}

#[derive(Debug, SurrealValue)]
struct IdOnly {
    id: RecordId,
}

#[derive(Debug, SurrealValue)]
struct ItemRef {
    id: RecordId,
    path: String,
}

#[derive(Debug, SurrealValue)]
struct CountRow {
    n: i64,
}

#[derive(Debug, SurrealValue)]
struct CollectionCountRow {
    collection: RecordId,
    n: i64,
}

#[derive(Debug, SurrealValue)]
struct PathCountRow {
    path: String,
    n: i64,
}

/// One collection.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Collection {
    pub id: RecordId,
    pub name: String,
    pub created: Option<DateTime<Utc>>,
    pub updated: Option<DateTime<Utc>>,
    pub color: Option<String>,
    pub note: Option<String>,
    /// Record key of the legacy logical group this collection was migrated from.
    pub legacy_key: Option<String>,
}

impl Collection {
    pub fn key(&self) -> String {
        key_of(&self.id)
    }
}

impl From<CollectionRow> for Collection {
    fn from(r: CollectionRow) -> Self {
        Self {
            id: r.id,
            name: r.name,
            created: r.created.map(Into::into),
            updated: r.updated.map(Into::into),
            color: r.color,
            note: r.note,
            legacy_key: r.legacy_key,
        }
    }
}

/// A collection with its membership count.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CollectionCount {
    pub collection: Collection,
    pub count: u64,
}

/// Fields of a collection to create.
#[derive(Clone, Debug, Default)]
pub struct NewCollection {
    pub name: String,
    pub created: Option<DateTime<Utc>>,
    pub updated: Option<DateTime<Utc>>,
    pub color: Option<String>,
    pub note: Option<String>,
    pub legacy_key: Option<String>,
}

impl NewCollection {
    pub fn named(name: &str) -> Self {
        Self { name: name.to_string(), ..Default::default() }
    }
}

fn already_exists(e: &anyhow::Error) -> bool {
    e.to_string().contains("already exists")
}

/// All collections ordered by name ignoring case.
pub async fn list<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<Collection>> {
    let rows: Vec<CollectionRow> = db.query("SELECT * FROM collections").await?.take(0)?;
    let mut out: Vec<Collection> = rows.into_iter().map(Into::into).collect();
    out.sort_by_key(|c| (c.name.to_lowercase(), c.name.clone()));
    Ok(out)
}

/// All collections with their membership counts, from one grouped query.
pub async fn list_with_counts<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<CollectionCount>> {
    let mut resp = db.query("SELECT * FROM collections; SELECT collection, count() AS n FROM collection_items GROUP BY collection;").await?;
    let rows: Vec<CollectionRow> = resp.take(0)?;
    let counts: Vec<CollectionCountRow> = resp.take(1)?;
    let counts: HashMap<String, u64> = counts.into_iter().map(|c| (key_of(&c.collection), c.n.max(0) as u64)).collect();
    let mut out: Vec<CollectionCount> = rows
        .into_iter()
        .map(|r| {
            let count = counts.get(&key_of(&r.id)).copied().unwrap_or(0);
            CollectionCount { collection: r.into(), count }
        })
        .collect();
    out.sort_by_key(|c| (c.collection.name.to_lowercase(), c.collection.name.clone()));
    Ok(out)
}

pub async fn get<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<Option<Collection>> {
    let row: Option<CollectionRow> = db.query("SELECT * FROM ONLY $id").bind(("id", id.clone())).await?.take(0)?;
    Ok(row.map(Into::into))
}

/// The collection called `name` (ignoring case), if any.
pub async fn get_by_name<C: Connection>(db: &Surreal<C>, name: &str) -> anyhow::Result<Option<Collection>> {
    match normalize(name) {
        Ok((_, key)) => get(db, &collection_id(&key)).await,
        Err(_) => Ok(None),
    }
}

async fn require<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<Collection> {
    get(db, id).await?.ok_or_else(|| CollectionError::NotFound(key_of(id)).into())
}

/// Creates a collection; fails when the name is invalid, reserved or taken ignoring case.
pub async fn create<C: Connection>(db: &Surreal<C>, name: &str) -> anyhow::Result<Collection> {
    create_with(db, NewCollection::named(name)).await
}

/// [`create`] with explicit timestamps, colour, note and legacy key.
pub async fn create_with<C: Connection>(db: &Surreal<C>, new: NewCollection) -> anyhow::Result<Collection> {
    let (display, key) = normalize(&new.name)?;
    let id = collection_id(&key);
    if get(db, &id).await?.is_some() {
        return Err(CollectionError::Exists(display).into());
    }
    let row = CollectionIn {
        name: display.clone(),
        created: new.created.map(Into::into),
        updated: new.updated.map(Into::into),
        color: new.color,
        note: new.note,
        legacy_key: new.legacy_key,
    };
    let created: Option<CollectionRow> = db
        .query("CREATE ONLY $id CONTENT $row RETURN AFTER")
        .bind(("id", id))
        .bind(("row", row))
        .await
        .map_err(anyhow::Error::from)
        .and_then(|mut r| r.take(0).map_err(Into::into))
        .map_err(|e| if already_exists(&e) { CollectionError::Exists(display.clone()).into() } else { e })?;
    created.map(Into::into).context("the collection was not created")
}

/// The collection called `name`, created when absent; the flag tells whether it was created.
pub async fn get_or_create<C: Connection>(db: &Surreal<C>, name: &str) -> anyhow::Result<(Collection, bool)> {
    if let Some(c) = get_by_name(db, name).await? {
        return Ok((c, false));
    }
    match create(db, name).await {
        Ok(c) => Ok((c, true)),
        Err(e) if matches!(e.downcast_ref::<CollectionError>(), Some(CollectionError::Exists(_))) => {
            let existing = get_by_name(db, name).await?.context("the collection vanished while it was being created")?;
            Ok((existing, false))
        }
        Err(e) => Err(e),
    }
}

/// Sets the colour and note of a collection.
pub async fn set_meta<C: Connection>(db: &Surreal<C>, id: &RecordId, color: Option<&str>, note: Option<&str>) -> anyhow::Result<Collection> {
    require(db, id).await?;
    db.query("UPDATE $id SET color = $color, note = $note, updated = time::now()")
        .bind(("id", id.clone()))
        .bind(("color", color.map(str::to_string)))
        .bind(("note", note.map(str::to_string)))
        .await?
        .check()?;
    require(db, id).await
}

/// Outcome of [`rename`].
#[derive(Clone, Debug, Serialize)]
pub struct RenameReport {
    pub collection: Collection,
    pub moved: u64,
}

/// Renames a collection; its memberships move to the collection with the new key.
pub async fn rename<C: Connection>(db: &Surreal<C>, id: &RecordId, new_name: &str) -> anyhow::Result<RenameReport> {
    let old = require(db, id).await?;
    let (display, new_key) = normalize(new_name)?;
    let old_key = old.key();
    if new_key == old_key {
        db.query("UPDATE $id SET name = $name, updated = time::now()").bind(("id", id.clone())).bind(("name", display)).await?.check()?;
        return Ok(RenameReport { collection: require(db, id).await?, moved: 0 });
    }
    let new_id = collection_id(&new_key);
    if get(db, &new_id).await?.is_some() {
        return Err(CollectionError::Exists(display).into());
    }
    let created = create_with(
        db,
        NewCollection { name: display, created: old.created, updated: None, color: old.color.clone(), note: old.note.clone(), legacy_key: old.legacy_key.clone() },
    )
    .await?;
    let mut moved = 0u64;
    loop {
        let rows: Vec<ItemRef> = db
            .query("SELECT id, path FROM collection_items WHERE collection = $c LIMIT $n")
            .bind(("c", id.clone()))
            .bind(("n", MOVE_CHUNK as i64))
            .await?
            .take(0)?;
        if rows.is_empty() {
            break;
        }
        let new_rows: Vec<ItemIn> = rows.iter().map(|r| ItemIn { id: item_id(&new_key, &r.path), collection: new_id.clone(), path: r.path.clone() }).collect();
        let old_ids: Vec<RecordId> = rows.into_iter().map(|r| r.id).collect();
        let n = old_ids.len() as u64;
        retry_conflicts(|| async {
            db.query("BEGIN; INSERT IGNORE INTO collection_items $rows RETURN NONE; DELETE $old RETURN NONE; COMMIT;")
                .bind(("rows", new_rows.clone()))
                .bind(("old", old_ids.clone()))
                .await?
                .check()
                .context("moving memberships failed")?;
            Ok(())
        })
        .await?;
        moved += n;
    }
    db.query("DELETE $id RETURN NONE").bind(("id", id.clone())).await?.check()?;
    Ok(RenameReport { collection: created, moved })
}

/// Deletes a collection and its memberships (never files or thumbnails); returns the memberships removed.
pub async fn delete<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<u64> {
    let removed = clear_members(db, id).await?;
    db.query("DELETE $id RETURN NONE").bind(("id", id.clone())).await?.check()?;
    Ok(removed)
}

/// Removes every membership of the collection `id`, keeping the collection.
pub async fn clear_members<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<u64> {
    let mut removed = 0u64;
    loop {
        let ids: Vec<RecordId> = db
            .query("SELECT VALUE id FROM collection_items WHERE collection = $c LIMIT $n")
            .bind(("c", id.clone()))
            .bind(("n", MOVE_CHUNK as i64))
            .await?
            .take(0)?;
        if ids.is_empty() {
            return Ok(removed);
        }
        removed += ids.len() as u64;
        retry_conflicts(|| async {
            db.query("DELETE $ids RETURN NONE").bind(("ids", ids.clone())).await?.check()?;
            Ok(())
        })
        .await?;
    }
}

fn distinct(paths: &[String]) -> (Vec<&str>, usize) {
    let mut seen: HashSet<&str> = HashSet::with_capacity(paths.len());
    let mut out = Vec::with_capacity(paths.len());
    let mut empty = 0;
    for p in paths {
        if p.is_empty() {
            empty += 1;
        } else if seen.insert(p.as_str()) {
            out.push(p.as_str());
        }
    }
    (out, empty)
}

/// Outcome of [`add_paths`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AddReport {
    pub requested: usize,
    pub added: usize,
    pub already: usize,
    pub skipped_empty: usize,
}

/// Inserts memberships without checking that the collection exists.
pub(super) async fn insert_paths<C: Connection>(db: &Surreal<C>, id: &RecordId, key: &str, paths: &[&str]) -> anyhow::Result<usize> {
    let mut added = 0;
    for chunk in paths.chunks(ADD_CHUNK) {
        let rows: Vec<ItemIn> = chunk.iter().map(|p| ItemIn { id: item_id(key, p), collection: id.clone(), path: (*p).to_string() }).collect();
        added += retry_conflicts(|| async {
            let ids: Vec<RecordId> = db
                .query("INSERT IGNORE INTO collection_items $rows RETURN VALUE id")
                .bind(("rows", rows.clone()))
                .await?
                .check()
                .context("adding to the collection failed")?
                .take(0)?;
            Ok(ids.len())
        })
        .await?;
    }
    Ok(added)
}

/// Adds paths to a collection; paths already in it, repeats and empty strings are skipped.
pub async fn add_paths<C: Connection>(db: &Surreal<C>, id: &RecordId, paths: &[String]) -> anyhow::Result<AddReport> {
    require(db, id).await?;
    let (unique, skipped_empty) = distinct(paths);
    let added = insert_paths(db, id, &key_of(id), &unique).await?;
    if added > 0 {
        let touched = retry_conflicts(|| async {
            db.query("UPDATE $id SET updated = time::now() RETURN NONE").bind(("id", id.clone())).await?.check()?;
            Ok(())
        })
        .await;
        if let Err(e) = touched {
            log::debug!("[collections] could not update the modified time: {e:#}");
        }
    }
    Ok(AddReport { requested: paths.len(), added, already: unique.len() - added, skipped_empty })
}

/// Removes paths from a collection; returns how many memberships existed.
pub async fn remove_paths<C: Connection>(db: &Surreal<C>, id: &RecordId, paths: &[String]) -> anyhow::Result<usize> {
    let key = key_of(id);
    let (unique, _) = distinct(paths);
    let mut removed = 0;
    for chunk in unique.chunks(ADD_CHUNK) {
        let ids: Vec<RecordId> = chunk.iter().map(|p| item_id(&key, p)).collect();
        removed += retry_conflicts(|| async {
            let gone: Vec<IdOnly> = db.query("DELETE $ids RETURN BEFORE").bind(("ids", ids.clone())).await?.check()?.take(0)?;
            Ok(gone.len())
        })
        .await?;
    }
    Ok(removed)
}

pub async fn count_members<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<u64> {
    let rows: Vec<CountRow> = db.query("SELECT count() AS n FROM collection_items WHERE collection = $c GROUP ALL").bind(("c", id.clone())).await?.take(0)?;
    Ok(rows.first().map_or(0, |r| r.n.max(0) as u64))
}

/// A page of member paths in path order.
pub async fn members<C: Connection>(db: &Surreal<C>, id: &RecordId, offset: usize, limit: usize) -> anyhow::Result<Vec<String>> {
    let paths: Vec<String> = db
        .query("SELECT VALUE path FROM collection_items WHERE collection = $c ORDER BY path LIMIT $l START $o")
        .bind(("c", id.clone()))
        .bind(("l", limit as i64))
        .bind(("o", offset as i64))
        .await?
        .take(0)?;
    Ok(paths)
}

/// Thumbnail rows for `paths`, in the order of `paths`; paths without a row are left out.
pub async fn rows_for_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<Vec<Thumbnail>> {
    let mut by_path: HashMap<String, Thumbnail> = HashMap::with_capacity(paths.len());
    for chunk in paths.chunks(LOOKUP_CHUNK) {
        let rows: Vec<Thumbnail> = db
            .query(format!("SELECT {} FROM thumbnails WHERE path IN $paths", super::thumbnails::row_cols()))
            .bind(("paths", chunk.to_vec()))
            .await?
            .take(0)?;
        by_path.extend(rows.into_iter().map(|r| (r.path.clone(), r)));
    }
    Ok(paths.iter().filter_map(|p| by_path.remove(p)).collect())
}

/// A page of members as thumbnail rows.
#[derive(Clone, Debug, Default)]
pub struct MemberRows {
    pub rows: Vec<Thumbnail>,
    /// Member paths in the page.
    pub requested: usize,
    /// Members with no thumbnail row (files that are not indexed).
    pub missing: usize,
}

/// A page of members loaded as thumbnail rows through the explorer's row columns.
pub async fn member_rows<C: Connection>(db: &Surreal<C>, id: &RecordId, offset: usize, limit: usize) -> anyhow::Result<MemberRows> {
    let paths = members(db, id, offset, limit).await?;
    let rows = rows_for_paths(db, &paths).await?;
    Ok(MemberRows { requested: paths.len(), missing: paths.len().saturating_sub(rows.len()), rows })
}

/// Every member as a thumbnail row, loaded page by page.
pub async fn all_member_rows<C: Connection>(db: &Surreal<C>, id: &RecordId) -> anyhow::Result<MemberRows> {
    let mut all = MemberRows::default();
    let mut offset = 0;
    loop {
        let page = member_rows(db, id, offset, PAGE).await?;
        if page.requested == 0 {
            return Ok(all);
        }
        offset += page.requested;
        all.requested += page.requested;
        all.missing += page.missing;
        all.rows.extend(page.rows);
    }
}

/// Collections that contain `path`.
pub async fn collections_of<C: Connection>(db: &Surreal<C>, path: &str) -> anyhow::Result<Vec<Collection>> {
    let ids: Vec<RecordId> = db.query("SELECT VALUE collection FROM collection_items WHERE path = $p").bind(("p", path.to_string())).await?.take(0)?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<CollectionRow> = db.query("SELECT * FROM $ids").bind(("ids", ids)).await?.take(0)?;
    let mut out: Vec<Collection> = rows.into_iter().map(Into::into).collect();
    out.sort_by_key(|c| (c.name.to_lowercase(), c.name.clone()));
    Ok(out)
}

/// How many collections each of `paths` is in; paths in none are absent.
pub async fn counts_for<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<HashMap<String, u32>> {
    let mut out = HashMap::new();
    for chunk in paths.chunks(LOOKUP_CHUNK) {
        let rows: Vec<PathCountRow> = db
            .query("SELECT path, count() AS n FROM collection_items WHERE path IN $paths GROUP BY path")
            .bind(("paths", chunk.to_vec()))
            .await?
            .take(0)?;
        out.extend(rows.into_iter().map(|r| (r.path, r.n.max(0) as u32)));
    }
    Ok(out)
}

/// Copies every member of `from` into `to`; returns the memberships added.
pub async fn copy_members<C: Connection>(db: &Surreal<C>, from: &RecordId, to: &RecordId) -> anyhow::Result<usize> {
    require(db, from).await?;
    require(db, to).await?;
    let to_key = key_of(to);
    let mut added = 0;
    let mut offset = 0;
    loop {
        let page = members(db, from, offset, PAGE).await?;
        if page.is_empty() {
            break;
        }
        offset += page.len();
        let refs: Vec<&str> = page.iter().map(String::as_str).collect();
        added += insert_paths(db, to, &to_key, &refs).await?;
    }
    Ok(added)
}

/// Paths of thumbnail rows that are in no collection, at most `limit`.
pub async fn unfiled_paths<C: Connection>(db: &Surreal<C>, limit: usize) -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    let mut after: Option<String> = None;
    while out.len() < limit {
        let mut q = db.query(format!("SELECT VALUE path FROM thumbnails {} ORDER BY path LIMIT $n", if after.is_some() { "WHERE path > $after" } else { "" })).bind(("n", PAGE as i64));
        if let Some(a) = &after {
            q = q.bind(("after", a.clone()));
        }
        let page: Vec<String> = q.await?.take(0)?;
        let Some(last) = page.last() else { break };
        after = Some(last.clone());
        let filed = counts_for(db, &page).await?;
        out.extend(page.into_iter().filter(|p| !filed.contains_key(p)).take(limit - out.len()));
    }
    Ok(out)
}

/// Ids of collections that memberships point to but that do not exist, with the memberships each has.
pub async fn dangling<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<(RecordId, u64)>> {
    let mut resp = db.query("SELECT VALUE id FROM collections; SELECT collection, count() AS n FROM collection_items GROUP BY collection;").await?;
    let existing: Vec<RecordId> = resp.take(0)?;
    let counts: Vec<CollectionCountRow> = resp.take(1)?;
    let existing: HashSet<String> = existing.iter().map(key_of).collect();
    Ok(counts.into_iter().filter(|c| !existing.contains(&key_of(&c.collection))).map(|c| (c.collection, c.n.max(0) as u64)).collect())
}

/// Memberships whose collection no longer exists.
pub async fn orphan_count<C: Connection>(db: &Surreal<C>) -> anyhow::Result<u64> {
    Ok(dangling(db).await?.iter().map(|(_, n)| n).sum())
}

/// Deletes memberships whose collection no longer exists; returns how many.
pub async fn prune_orphans<C: Connection>(db: &Surreal<C>) -> anyhow::Result<u64> {
    let mut removed = 0;
    for (id, _) in dangling(db).await? {
        removed += clear_members(db, &id).await?;
    }
    Ok(removed)
}

/// Total memberships.
pub async fn count_items<C: Connection>(db: &Surreal<C>) -> anyhow::Result<u64> {
    let rows: Vec<CountRow> = db.query("SELECT count() AS n FROM collection_items GROUP ALL").await?.take(0)?;
    Ok(rows.first().map_or(0, |r| r.n.max(0) as u64))
}

/// Memberships whose path has no thumbnail row, scanning at most `limit` memberships.
pub async fn unindexed_count<C: Connection>(db: &Surreal<C>, limit: usize) -> anyhow::Result<u64> {
    let mut missing = 0u64;
    let mut seen = 0usize;
    let mut after: Option<String> = None;
    while seen < limit {
        let mut q = db
            .query(format!("SELECT VALUE path FROM collection_items {} ORDER BY path LIMIT $n", if after.is_some() { "WHERE path > $after" } else { "" }))
            .bind(("n", PAGE as i64));
        if let Some(a) = &after {
            q = q.bind(("after", a.clone()));
        }
        let page: Vec<String> = q.await?.take(0)?;
        let Some(last) = page.last() else { break };
        after = Some(last.clone());
        seen += page.len();
        let unique: Vec<String> = page.into_iter().collect::<HashSet<_>>().into_iter().collect();
        for chunk in unique.chunks(LOOKUP_CHUNK) {
            let found: Vec<String> = db.query("SELECT VALUE path FROM thumbnails WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.take(0)?;
            missing += (chunk.len() - found.len()) as u64;
        }
    }
    Ok(missing)
}

/// One timed step of [`probe`].
#[derive(Clone, Debug, Serialize)]
pub struct ProbeStep {
    pub step: String,
    pub millis: u128,
    pub detail: String,
}

/// Timings of the API on a temporary collection of `n` synthetic paths; the collection is deleted afterwards.
pub async fn probe<C: Connection>(db: &Surreal<C>, n: usize) -> anyhow::Result<Vec<ProbeStep>> {
    for stale in list(db).await?.into_iter().filter(|c| c.name.starts_with(PROBE_PREFIX)) {
        delete(db, &stale.id).await?;
    }
    let name = format!("{PROBE_PREFIX}{}", Utc::now().timestamp());
    let mut steps = Vec::new();
    macro_rules! timed {
        ($label:expr, $detail:expr, $fut:expr) => {{
            let t = Instant::now();
            let out = $fut.await?;
            steps.push(ProbeStep { step: $label.to_string(), millis: t.elapsed().as_millis(), detail: $detail(&out) });
            out
        }};
    }
    let c = timed!("create", |c: &Collection| c.name.clone(), create(db, &name));
    let paths: Vec<String> = (0..n).map(|i| format!("/__collections_probe__/{:03}/f{i}.jpg", i % 500)).collect();
    let result = async {
        timed!("add_paths", |r: &AddReport| format!("{} added", r.added), add_paths(db, &c.id, &paths));
        timed!("add_paths again", |r: &AddReport| format!("{} already", r.already), add_paths(db, &c.id, &paths));
        timed!("list_with_counts", |l: &Vec<CollectionCount>| format!("{} collections", l.len()), list_with_counts(db));
        timed!("count_members", |n: &u64| format!("{n}"), count_members(db, &c.id));
        timed!("members page", |p: &Vec<String>| format!("{} paths", p.len()), members(db, &c.id, n / 2, 500));
        timed!("collections_of", |l: &Vec<Collection>| format!("{} collections", l.len()), collections_of(db, &paths[0]));
        timed!("counts_for", |m: &HashMap<String, u32>| format!("{} paths", m.len()), counts_for(db, &paths[..paths.len().min(5000)]));
        timed!("remove_paths", |n: &usize| format!("{n} removed"), remove_paths(db, &c.id, &paths[..paths.len() / 2]));
        anyhow::Ok(())
    }
    .await;
    let t = Instant::now();
    let removed = delete(db, &c.id).await?;
    steps.push(ProbeStep { step: "delete".into(), millis: t.elapsed().as_millis(), detail: format!("{removed} memberships") });
    result?;
    Ok(steps)
}

#[cfg(test)]
pub(crate) mod tests;
