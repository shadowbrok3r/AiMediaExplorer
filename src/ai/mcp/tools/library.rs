//! Tools that read the library: search, details, statistics, collections, tags and categories.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::common::{CountRow, ITEM_DESCRIPTION_CHARS, LIST_DESCRIPTION_CHARS, ItemRow, item_json, page_json, page_limit, row_by_path, rows_by_paths, safe_text};
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};
use crate::database::ClipEmbeddingRow;
use crate::database::collections;

const MAX_TERMS: usize = 6;
const TERM_NAMES: [&str; MAX_TERMS] = ["t0", "t1", "t2", "t3", "t4", "t5"];
const READ_TIMEOUT: Duration = Duration::from_secs(30);

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("library_search", ToolClass::Read, READ_TIMEOUT, "Search the library", "Find library items by words in file name, caption, description, tags and category, with filters for tags, category, file type, folder and collection. Returns one page of items with metadata.", library_search),
        entry("library_semantic_search", ToolClass::Gpu, Duration::from_secs(300), "Search the library by meaning", "Find items whose image content matches a text description, using the local CLIP model and its vector index. Returns the best matches with a similarity from -1 to 1 (higher is closer).", library_semantic_search),
        entry("library_similar_to", ToolClass::Read, Duration::from_secs(300), "Find similar items", "Find items that look like a library item, by comparing CLIP embeddings. The item needs a stored embedding, or a readable file inside the allowed folders.", library_similar_to),
        entry("library_get_item", ToolClass::Read, READ_TIMEOUT, "Get one library item", "Full metadata of one library item by path: caption, description, tags, category, collections and embedding status. Optionally includes its stored thumbnail as an image.", library_get_item),
        entry("library_stats", ToolClass::Read, READ_TIMEOUT, "Library statistics", "Counts of items, described, tagged and categorized items, CLIP embeddings and collections, the largest categories and tags, and the folders the library covers.", library_stats),
        entry("list_collections", ToolClass::Read, READ_TIMEOUT, "List collections", "Collections with their item counts, optionally filtered by a name fragment.", list_collections),
        entry("collection_items", ToolClass::Read, READ_TIMEOUT, "List items of a collection", "One page of the items in a collection, in path order, with metadata.", collection_items),
        entry("list_tags", ToolClass::Read, READ_TIMEOUT, "List tags", "Tags in use with the number of items that carry each, optionally filtered by a text fragment.", list_tags),
        entry("list_categories", ToolClass::Read, READ_TIMEOUT, "List categories", "Categories in use with the number of items in each, optionally filtered by a text fragment.", list_categories),
    ]
}

fn db_of(ctx: &ToolCtx) -> Result<&Surreal<Db>, ToolError> {
    ctx.db.get()
}

fn clean_path(path: &str) -> Result<String, ToolError> {
    let p = path.trim();
    if p.is_empty() {
        return Err(ToolError::invalid("path is empty"));
    }
    Ok(p.to_string())
}

// ---------------------------------------------------------------- library_search

#[derive(Clone, Copy, Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum SortKey {
    #[default]
    Path,
    Filename,
    Modified,
    Size,
}

impl SortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Path => "path",
            Self::Filename => "filename",
            Self::Modified => "modified",
            Self::Size => "size",
        }
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct SearchArgs {
    /// Words to find in file name, caption, description, tags or category; every word must match somewhere. Case is ignored.
    pub query: Option<String>,
    /// Tags the item must carry, all of them, spelled as list_tags shows them.
    pub tags: Vec<String>,
    /// Category the item must be in (case ignored).
    pub category: Option<String>,
    /// File extension without the dot, for example jpg.
    pub file_type: Option<String>,
    /// Only items below this folder path.
    pub folder: Option<String>,
    /// Only items in the collection with this name.
    pub collection: Option<String>,
    /// True to list only items without a description.
    pub undescribed: Option<bool>,
    /// True to list only items without tags.
    pub untagged: Option<bool>,
    /// Sort column; default path.
    pub sort: Option<SortKey>,
    /// Sort from high to low.
    pub descending: Option<bool>,
    /// Items per page, 1 to 100; default 25.
    pub limit: Option<usize>,
    /// Items to skip; default 0.
    pub offset: Option<usize>,
}

/// The WHERE clause of a search and the lower-cased terms it binds.
#[derive(Debug, PartialEq, Eq)]
pub struct Filter {
    pub clause: String,
    pub terms: Vec<String>,
}

fn term_clause(i: usize) -> String {
    let t = TERM_NAMES[i];
    format!(
        "(string::lowercase(filename ?? '') CONTAINS ${t} OR string::lowercase(caption ?? '') CONTAINS ${t} OR string::lowercase(description ?? '') CONTAINS ${t} OR string::lowercase(category ?? '') CONTAINS ${t} OR string::lowercase(array::join(tags ?? [], ' ')) CONTAINS ${t})"
    )
}

/// Builds the WHERE clause for `args`; `has_collection` says whether a collection filter resolved.
pub fn build_filter(args: &SearchArgs, has_collection: bool) -> Result<Filter, ToolError> {
    let terms: Vec<String> = args.query.as_deref().unwrap_or("").split_whitespace().map(|t| t.to_lowercase()).collect();
    if terms.len() > MAX_TERMS {
        return Err(ToolError::invalid(format!("query has {} words; use at most {MAX_TERMS}", terms.len())));
    }
    let mut parts: Vec<String> = (0..terms.len()).map(term_clause).collect();
    if !args.tags.is_empty() {
        parts.push("tags CONTAINSALL $tags".to_string());
    }
    if args.category.as_deref().is_some_and(|c| !c.trim().is_empty()) {
        parts.push("string::lowercase(category ?? '') = $cat".to_string());
    }
    if args.file_type.as_deref().is_some_and(|c| !c.trim().is_empty()) {
        parts.push("string::lowercase(file_type ?? '') = $ft".to_string());
    }
    if args.folder.as_deref().is_some_and(|c| !c.trim().is_empty()) {
        parts.push("string::starts_with(path, $prefix)".to_string());
    }
    if has_collection {
        parts.push("path IN (SELECT VALUE path FROM collection_items WHERE collection = $coll)".to_string());
    }
    if args.undescribed == Some(true) {
        parts.push("string::len(description ?? '') = 0".to_string());
    }
    if args.untagged == Some(true) {
        parts.push("array::len(tags ?? []) = 0".to_string());
    }
    let clause = if parts.is_empty() { String::new() } else { format!("WHERE {}", parts.join(" AND ")) };
    Ok(Filter { clause, terms })
}

/// A folder path with a trailing separator.
fn folder_prefix(folder: &str) -> String {
    let f = folder.trim();
    if f.ends_with('/') || f.ends_with('\\') { f.to_string() } else { format!("{f}{}", if f.contains('\\') && !f.contains('/') { '\\' } else { '/' }) }
}

async fn library_search(ctx: Arc<ToolCtx>, _call: CallInfo, args: SearchArgs) -> Result<ToolOutput, ToolError> {
    let db = db_of(&ctx)?;
    let limit = page_limit(args.limit, 25, 100);
    let offset = args.offset.unwrap_or(0).min(1_000_000);
    let collection: Option<RecordId> = match args.collection.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
        Some(name) => Some(collections::get_by_name(db, name).await?.ok_or_else(|| ToolError::NotFound(format!("no collection named '{name}'; call list_collections")))?.id),
        None => None,
    };
    let filter = build_filter(&args, collection.is_some())?;
    let order = format!("ORDER BY {} {}", args.sort.unwrap_or_default().column(), if args.descending == Some(true) { "DESC" } else { "ASC" });
    let sql = format!("SELECT {cols} FROM thumbnails {w} {order} LIMIT $limit START $start; SELECT count() AS n FROM thumbnails {w} GROUP ALL;", cols = super::common::ITEM_COLS, w = filter.clause);
    let mut q = db.query(sql).bind(("limit", limit as i64)).bind(("start", offset as i64));
    for (i, term) in filter.terms.iter().enumerate() {
        q = q.bind((TERM_NAMES[i], term.clone()));
    }
    if !args.tags.is_empty() {
        q = q.bind(("tags", args.tags.clone()));
    }
    if let Some(c) = args.category.as_deref().filter(|c| !c.trim().is_empty()) {
        q = q.bind(("cat", c.trim().to_lowercase()));
    }
    if let Some(t) = args.file_type.as_deref().filter(|c| !c.trim().is_empty()) {
        q = q.bind(("ft", t.trim().trim_start_matches('.').to_lowercase()));
    }
    if let Some(f) = args.folder.as_deref().filter(|c| !c.trim().is_empty()) {
        q = q.bind(("prefix", folder_prefix(f)));
    }
    if let Some(id) = collection {
        q = q.bind(("coll", id));
    }
    let mut resp = q.await?;
    let rows: Vec<ItemRow> = resp.take(0)?;
    let total: Vec<CountRow> = resp.take(1)?;
    let total = total.first().map_or(0, |r| r.n.max(0) as u64);
    let items: Vec<Value> = rows.iter().map(|r| item_json(r, LIST_DESCRIPTION_CHARS)).collect();
    let text = format!("{} of {} matching items (offset {offset})", items.len(), total);
    Ok(ToolOutput::new(text, page_json(items, total, offset, limit)))
}

// ---------------------------------------------------------------- semantic search

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct SemanticArgs {
    /// What the images should show, in plain words, up to 512 characters.
    pub query: String,
    /// Results to return, 1 to 200; default 24.
    pub top_k: Option<usize>,
    /// Results to skip; offset plus top_k may not exceed 1000.
    pub offset: Option<usize>,
    /// Drop results below this cosine similarity (-1 to 1).
    pub min_similarity: Option<f32>,
}

fn knn_ef(wanted: usize) -> usize {
    (wanted * 2).clamp(64, crate::database::clip_embeddings::KNN_MAX_K)
}

/// Validated page bounds of a KNN request.
pub fn knn_bounds(top_k: Option<usize>, offset: Option<usize>) -> Result<(usize, usize), ToolError> {
    let top_k = page_limit(top_k, 24, 200);
    let offset = offset.unwrap_or(0);
    if offset + top_k > crate::database::clip_embeddings::KNN_MAX_K {
        return Err(ToolError::invalid(format!("offset plus top_k may not exceed {}", crate::database::clip_embeddings::KNN_MAX_K)));
    }
    Ok((top_k, offset))
}

async fn hits_json(db: &Surreal<Db>, hits: Vec<crate::database::clip_embeddings::SimilarPath>, offset: usize, min_similarity: Option<f32>) -> Result<Vec<Value>, ToolError> {
    let paths: Vec<String> = hits.iter().map(|h| h.path.clone()).collect();
    let rows = rows_by_paths(db, &paths).await?;
    let mut out = Vec::with_capacity(hits.len());
    for (i, hit) in hits.iter().enumerate() {
        let similarity = hit.similarity();
        if min_similarity.is_some_and(|m| similarity < m) {
            continue;
        }
        let mut item = match rows.get(&hit.path) {
            Some(row) => item_json(row, LIST_DESCRIPTION_CHARS),
            None => json!({"path": hit.path}),
        };
        item["rank"] = json!(offset + i + 1);
        item["similarity"] = json!((similarity * 10_000.0).round() / 10_000.0);
        item["distance"] = json!((hit.dist * 10_000.0).round() / 10_000.0);
        out.push(item);
    }
    Ok(out)
}

async fn library_semantic_search(ctx: Arc<ToolCtx>, call: CallInfo, args: SemanticArgs) -> Result<ToolOutput, ToolError> {
    let query = args.query.trim();
    if query.is_empty() || query.chars().count() > 512 {
        return Err(ToolError::invalid("query must have 1 to 512 characters"));
    }
    let (top_k, offset) = knn_bounds(args.top_k, args.offset)?;
    let db = db_of(&ctx)?;
    let label = format!("embed text '{}'", safe_text(query, 40));
    let (vector, gpu) = super::gpu::gpu_job(&ctx, &call, "clip_text", crate::tasks::TaskKind::Clip, label, ctx.clip.embed_text(query)).await?;
    let hits = ClipEmbeddingRow::find_similar_paths_in(db, &vector, top_k, knn_ef(offset + top_k), offset).await?;
    let items = hits_json(db, hits, offset, args.min_similarity).await?;
    let text = format!("{} items closest to '{}'", items.len(), safe_text(query, 60));
    Ok(ToolOutput::new(text, json!({"query": query, "items": items, "returned": items.len(), "offset": offset, "top_k": top_k, "gpu": gpu})))
}

// ---------------------------------------------------------------- similar items

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct SimilarArgs {
    /// Path of a library item.
    pub path: String,
    /// Results to return, 1 to 200; default 24.
    pub top_k: Option<usize>,
    /// Results to skip; offset plus top_k may not exceed 1000.
    pub offset: Option<usize>,
    /// Drop results below this cosine similarity (-1 to 1).
    pub min_similarity: Option<f32>,
}

#[derive(Debug, SurrealValue)]
struct EmbeddingOnly {
    embedding: Vec<f32>,
}

async fn stored_embedding(db: &Surreal<Db>, path: &str) -> Result<Option<Vec<f32>>, ToolError> {
    let rows: Vec<EmbeddingOnly> = db.query("SELECT embedding FROM clip_embeddings WHERE path = $p LIMIT 1").bind(("p", path.to_string())).await?.take(0)?;
    Ok(rows.into_iter().next().map(|r| r.embedding).filter(|e| !e.is_empty()))
}

async fn library_similar_to(ctx: Arc<ToolCtx>, call: CallInfo, args: SimilarArgs) -> Result<ToolOutput, ToolError> {
    let path = clean_path(&args.path)?;
    let (top_k, offset) = knn_bounds(args.top_k, args.offset)?;
    let db = db_of(&ctx)?;
    ctx.policy.refresh(&ctx.db).await;
    ctx.policy.check_lexical(&path)?;
    let vector = match stored_embedding(db, &path).await? {
        Some(v) => v,
        None => {
            let real = ctx.policy.check_file(&path).map_err(|_| ToolError::NotFound(format!("{path} has no stored embedding; call clip_embed with dry_run false first")))?;
            let label = format!("embed {}", real.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
            super::gpu::gpu_job(&ctx, &call, "clip_image", crate::tasks::TaskKind::Clip, label, ctx.clip.embed_image(&real)).await?.0
        }
    };
    let hits = ClipEmbeddingRow::find_similar_paths_in(db, &vector, top_k + 1, knn_ef(offset + top_k + 1), offset).await?;
    let hits: Vec<_> = hits.into_iter().filter(|h| h.path != path).take(top_k).collect();
    let items = hits_json(db, hits, offset, args.min_similarity).await?;
    Ok(ToolOutput::new(format!("{} items similar to {}", items.len(), safe_text(&path, 80)), json!({"path": path, "items": items, "returned": items.len(), "offset": offset, "top_k": top_k})))
}

// ---------------------------------------------------------------- one item

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct GetItemArgs {
    /// Path of a library item.
    pub path: String,
    /// True to attach the stored thumbnail as an image.
    pub include_thumbnail: Option<bool>,
}

async fn library_get_item(ctx: Arc<ToolCtx>, _call: CallInfo, args: GetItemArgs) -> Result<ToolOutput, ToolError> {
    let path = clean_path(&args.path)?;
    let db = db_of(&ctx)?;
    let row = row_by_path(db, &path).await?.ok_or_else(|| ToolError::NotFound(format!("{path} is not in the library")))?;
    let names: Vec<String> = collections::collections_of(db, &path).await?.into_iter().map(|c| c.name).collect();
    let embedding = stored_embedding(db, &path).await?;
    let mut data = item_json(&row, ITEM_DESCRIPTION_CHARS);
    data["collections"] = json!(names);
    data["has_description"] = json!(row.description.as_deref().is_some_and(|d| !d.trim().is_empty()));
    data["clip_embedding"] = json!({"stored": embedding.is_some(), "dimensions": embedding.as_ref().map(Vec::len)});
    let mut out = ToolOutput::new(format!("{} ({} tags, {} collections)", safe_text(&path, 100), row.tags.as_ref().map_or(0, Vec::len), names.len()), data);
    if args.include_thumbnail == Some(true) {
        let key = crate::database::thumb_blobs::key_for_path(&path);
        let mut blobs = crate::database::thumb_blobs::get_many_with(db, std::slice::from_ref(&key)).await?;
        match blobs.remove(&key) {
            Some(blob) => {
                use base64::Engine as _;
                let mime = if blob.format == "png" { "image/png" } else { "image/jpeg" };
                out = out.with_image(mime, base64::engine::general_purpose::STANDARD.encode(&blob.bytes));
            }
            None => out.data["thumbnail"] = json!("none stored"),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- statistics

pub async fn tag_counts(db: &Surreal<Db>) -> Result<Vec<(String, u64)>, ToolError> {
    let lists: Vec<Vec<String>> = db.query("SELECT VALUE tags FROM thumbnails WHERE tags != NONE").await?.take(0)?;
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for tag in lists.into_iter().flatten().filter(|t| !t.trim().is_empty()) {
        *counts.entry(tag).or_default() += 1;
    }
    Ok(counts.into_iter().collect())
}

#[derive(Debug, SurrealValue)]
struct CategoryCount {
    category: Option<String>,
    n: i64,
}

pub async fn category_counts(db: &Surreal<Db>) -> Result<Vec<(String, u64)>, ToolError> {
    let rows: Vec<CategoryCount> = db.query("SELECT category, count() AS n FROM thumbnails GROUP BY category").await?.take(0)?;
    Ok(rows.into_iter().filter_map(|r| r.category.filter(|c| !c.trim().is_empty()).map(|c| (c, r.n.max(0) as u64))).collect())
}

async fn count_of(db: &Surreal<Db>, sql: &str) -> Result<u64, ToolError> {
    let rows: Vec<CountRow> = db.query(sql).await?.take(0)?;
    Ok(rows.first().map_or(0, |r| r.n.max(0) as u64))
}

fn top(mut counts: Vec<(String, u64)>, n: usize) -> Vec<Value> {
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts.into_iter().take(n).map(|(name, count)| json!({"name": safe_text(&name, 80), "count": count})).collect()
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct StatsArgs {}

async fn library_stats(ctx: Arc<ToolCtx>, _call: CallInfo, _args: StatsArgs) -> Result<ToolOutput, ToolError> {
    let db = db_of(&ctx)?;
    ctx.policy.refresh(&ctx.db).await;
    let items = count_of(db, "SELECT count() AS n FROM thumbnails GROUP ALL").await?;
    let described = count_of(db, "SELECT count() AS n FROM thumbnails WHERE string::len(description ?? '') > 0 GROUP ALL").await?;
    let tagged = count_of(db, "SELECT count() AS n FROM thumbnails WHERE array::len(tags ?? []) > 0 GROUP ALL").await?;
    let categorized = count_of(db, "SELECT count() AS n FROM thumbnails WHERE string::len(category ?? '') > 0 GROUP ALL").await?;
    let embeddings = count_of(db, "SELECT count() AS n FROM clip_embeddings GROUP ALL").await?;
    let collection_count = collections::list(db).await?.len();
    let categories = category_counts(db).await?;
    let tags = tag_counts(db).await?;
    let proposals = ctx.proposals.counts().await.ok();
    let data = json!({
        "items": items,
        "described": described,
        "tagged": tagged,
        "categorized": categorized,
        "with_clip_embedding": embeddings,
        "without_clip_embedding": items.saturating_sub(embeddings),
        "collections": collection_count,
        "distinct_categories": categories.len(),
        "distinct_tags": tags.len(),
        "top_categories": top(categories, 10),
        "top_tags": top(tags, 10),
        "folders": ctx.policy.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
        "pending_proposals": proposals.map(|p| p.pending),
    });
    Ok(ToolOutput::new(format!("{items} items, {described} described, {embeddings} embedded, {collection_count} collections"), data))
}

// ---------------------------------------------------------------- collections

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct ListArgs {
    /// Only names containing this text (case ignored).
    pub contains: Option<String>,
    /// Order by count (high to low) or by name; default name for collections and count for tags and categories.
    pub sort: Option<ListSort>,
    /// Entries per page, 1 to 500; default 100.
    pub limit: Option<usize>,
    /// Entries to skip.
    pub offset: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum ListSort {
    Name,
    Count,
}

fn paged_counts(mut counts: Vec<(String, u64)>, args: &ListArgs, default_sort: ListSort) -> (Vec<Value>, u64, usize, usize) {
    if let Some(f) = args.contains.as_deref().map(|f| f.trim().to_lowercase()).filter(|f| !f.is_empty()) {
        counts.retain(|(name, _)| name.to_lowercase().contains(&f));
    }
    match args.sort.unwrap_or(default_sort) {
        ListSort::Count => counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase()))),
        ListSort::Name => counts.sort_by_key(|(name, _)| name.to_lowercase()),
    }
    let total = counts.len() as u64;
    let limit = page_limit(args.limit, 100, 500);
    let offset = args.offset.unwrap_or(0);
    let items = counts.into_iter().skip(offset).take(limit).map(|(name, count)| json!({"name": safe_text(&name, 120), "count": count})).collect();
    (items, total, offset, limit)
}

async fn list_collections(ctx: Arc<ToolCtx>, _call: CallInfo, args: ListArgs) -> Result<ToolOutput, ToolError> {
    let db = db_of(&ctx)?;
    let all = collections::list_with_counts(db).await?;
    let notes: BTreeMap<String, Option<String>> = all.iter().map(|c| (c.collection.name.clone(), c.collection.note.clone())).collect();
    let counts: Vec<(String, u64)> = all.into_iter().map(|c| (c.collection.name, c.count)).collect();
    let (mut items, total, offset, limit) = paged_counts(counts, &args, ListSort::Name);
    for item in &mut items {
        let note = item["name"].as_str().and_then(|n| notes.get(n)).cloned().flatten();
        item["note"] = json!(note.map(|n| safe_text(&n, 200)));
    }
    Ok(ToolOutput::new(format!("{} of {total} collections", items.len()), page_json(items, total, offset, limit)))
}

async fn list_tags(ctx: Arc<ToolCtx>, _call: CallInfo, args: ListArgs) -> Result<ToolOutput, ToolError> {
    let counts = tag_counts(db_of(&ctx)?).await?;
    let (items, total, offset, limit) = paged_counts(counts, &args, ListSort::Count);
    Ok(ToolOutput::new(format!("{} of {total} tags", items.len()), page_json(items, total, offset, limit)))
}

async fn list_categories(ctx: Arc<ToolCtx>, _call: CallInfo, args: ListArgs) -> Result<ToolOutput, ToolError> {
    let counts = category_counts(db_of(&ctx)?).await?;
    let (items, total, offset, limit) = paged_counts(counts, &args, ListSort::Count);
    Ok(ToolOutput::new(format!("{} of {total} categories", items.len()), page_json(items, total, offset, limit)))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct CollectionItemsArgs {
    /// Name of the collection.
    pub name: String,
    /// Items per page, 1 to 100; default 50.
    pub limit: Option<usize>,
    /// Items to skip.
    pub offset: Option<usize>,
}

async fn collection_items(ctx: Arc<ToolCtx>, _call: CallInfo, args: CollectionItemsArgs) -> Result<ToolOutput, ToolError> {
    let db = db_of(&ctx)?;
    let collection = collections::get_by_name(db, &args.name).await?.ok_or_else(|| ToolError::NotFound(format!("no collection named '{}'; call list_collections", safe_text(&args.name, 80))))?;
    let limit = page_limit(args.limit, 50, 100);
    let offset = args.offset.unwrap_or(0);
    let total = collections::count_members(db, &collection.id).await?;
    let paths = collections::members(db, &collection.id, offset, limit).await?;
    let rows = rows_by_paths(db, &paths).await?;
    let items: Vec<Value> = paths.iter().map(|p| rows.get(p).map_or_else(|| json!({"path": p}), |r| item_json(r, LIST_DESCRIPTION_CHARS))).collect();
    let mut data = page_json(items, total, offset, limit);
    data["collection"] = json!(collection.name);
    Ok(ToolOutput::new(format!("{} of {total} items in '{}'", data["returned"], safe_text(&collection.name, 60)), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(query: &str) -> SearchArgs {
        SearchArgs { query: Some(query.into()), ..Default::default() }
    }

    #[test]
    fn every_word_becomes_a_clause_and_is_lower_cased() {
        let f = build_filter(&args("Red  CAR"), false).unwrap();
        assert_eq!(f.terms, vec!["red", "car"]);
        assert_eq!(f.clause.matches("CONTAINS $t0").count(), 5);
        assert_eq!(f.clause.matches("CONTAINS $t1").count(), 5);
        assert!(f.clause.starts_with("WHERE ("));
        assert_eq!(f.clause.matches(" AND ").count(), 1);
    }

    #[test]
    fn no_filters_means_no_where_clause() {
        assert_eq!(build_filter(&SearchArgs::default(), false).unwrap(), Filter { clause: String::new(), terms: vec![] });
        assert_eq!(build_filter(&args("   "), false).unwrap().clause, "");
    }

    #[test]
    fn each_option_adds_one_clause() {
        let a = SearchArgs {
            tags: vec!["x".into()],
            category: Some("c".into()),
            file_type: Some(".JPG".into()),
            folder: Some("/p".into()),
            undescribed: Some(true),
            untagged: Some(true),
            ..Default::default()
        };
        let f = build_filter(&a, true).unwrap();
        for part in ["tags CONTAINSALL $tags", "= $cat", "= $ft", "starts_with(path, $prefix)", "collection_items", "string::len(description", "array::len(tags"] {
            assert!(f.clause.contains(part), "{part} missing from {}", f.clause);
        }
        assert_eq!(f.clause.matches(" AND ").count(), 6);
    }

    #[test]
    fn false_flags_and_blank_options_add_nothing() {
        let a = SearchArgs { undescribed: Some(false), untagged: Some(false), category: Some("  ".into()), folder: Some("".into()), ..Default::default() };
        assert_eq!(build_filter(&a, false).unwrap().clause, "");
    }

    #[test]
    fn too_many_words_are_refused() {
        let err = build_filter(&args("a b c d e f g"), false).unwrap_err();
        assert_eq!(err.kind(), "invalid_arguments");
    }

    #[test]
    fn folder_prefixes_end_with_a_separator() {
        assert_eq!(folder_prefix("/a/b"), "/a/b/");
        assert_eq!(folder_prefix("/a/b/"), "/a/b/");
        assert_eq!(folder_prefix("C:\\pics"), "C:\\pics\\");
    }

    #[test]
    fn knn_bounds_clamp_the_page_and_refuse_deep_offsets() {
        assert_eq!(knn_bounds(None, None).unwrap(), (24, 0));
        assert_eq!(knn_bounds(Some(5000), Some(10)).unwrap(), (200, 10));
        assert!(knn_bounds(Some(200), Some(801)).is_err());
        assert!(knn_bounds(Some(200), Some(800)).is_ok());
    }

    #[test]
    fn counts_filter_sort_and_page() {
        let counts = vec![("Beach".to_string(), 3), ("beach house".to_string(), 9), ("City".to_string(), 3), ("sea".to_string(), 1)];
        let (items, total, offset, limit) = paged_counts(counts.clone(), &ListArgs { contains: Some("BEACH".into()), ..Default::default() }, ListSort::Count);
        assert_eq!((total, offset, limit), (2, 0, 100));
        assert_eq!(items[0]["name"], "beach house");
        let (by_name, _, _, _) = paged_counts(counts.clone(), &ListArgs::default(), ListSort::Name);
        assert_eq!(by_name.iter().map(|i| i["name"].as_str().unwrap()).collect::<Vec<_>>(), vec!["Beach", "beach house", "City", "sea"]);
        let (tail, total, _, _) = paged_counts(counts, &ListArgs { limit: Some(2), offset: Some(3), sort: Some(ListSort::Count), ..Default::default() }, ListSort::Name);
        assert_eq!((tail.len(), total), (1, 4));
    }
}
