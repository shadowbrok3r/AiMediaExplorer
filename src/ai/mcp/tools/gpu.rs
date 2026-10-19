//! Tools that run local models through the GPU queue, and the queue's status.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::common::{row_by_path, safe_text};
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::gpu::{JobSpec, JobStats};
use crate::ai::mcp::proposals::{Changes, NewProposal};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};
use crate::tasks::{TaskKind, TaskSpec};

/// Whole-call limit for a GPU tool: the wait in the queue plus the run.
const GPU_TOOL_TIMEOUT: Duration = Duration::from_secs(480);
/// Vector size the HNSW index was defined with.
pub const EMBEDDING_DIMENSIONS: usize = 1024;

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("describe_image", ToolClass::Gpu, GPU_TOOL_TIMEOUT, "Describe an image", "Run the local vision model on an image file inside the allowed folders and return its description, caption, tags and category. With dry_run true (the default) nothing is stored; with dry_run false the result is filed as a proposal for the user to review, never written to the library directly.", describe_image),
        entry("clip_embed", ToolClass::Gpu, GPU_TOOL_TIMEOUT, "Compute a CLIP embedding", "Run the local CLIP model on an image file inside the allowed folders. With dry_run true (the default) only the vector size and length are returned; with dry_run false the embedding of a library item is stored in the search index.", clip_embed),
        entry("gpu_queue_status", ToolClass::Read, Duration::from_secs(10), "GPU queue status", "The job running on the GPU, the jobs waiting, the queue limits and the most recent jobs.", gpu_queue_status),
    ]
}

/// Polls until the task center asks the task to stop.
pub async fn task_cancelled(reporter: crate::tasks::Reporter) {
    loop {
        tokio::time::sleep(Duration::from_millis(250)).await;
        if reporter.cancel_requested() {
            return;
        }
    }
}

/// Runs `work` as the next job of the GPU queue and shows it as a task while it runs.
pub async fn gpu_job<T, Fut>(ctx: &ToolCtx, call: &CallInfo, kind: &'static str, task_kind: TaskKind, label: String, work: Fut) -> Result<(T, JobStats), ToolError>
where
    Fut: Future<Output = anyhow::Result<T>>,
{
    let spec = JobSpec::new(kind, label.clone(), call.client.clone());
    let done = ctx
        .gpu
        .run(spec, &call.cancel, |job| async move {
            let handle = crate::tasks::start(TaskSpec::new(task_kind, format!("MCP: {label}")).visible_after(Duration::from_secs(2)));
            let result = tokio::select! {
                r = work => r.map_err(|e| format!("{e:#}")),
                _ = job.cancel.cancelled() => Err("the job was cancelled".to_string()),
                _ = task_cancelled(handle.reporter()) => Err("the job was cancelled from the task list".to_string()),
            };
            match &result {
                Ok(_) => handle.complete(),
                Err(e) => handle.fail(e.clone()),
            }
            result
        })
        .await?;
    Ok(done)
}

fn is_image_path(path: &std::path::Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_image(&e.to_ascii_lowercase()))
}

pub(super) async fn checked_image(ctx: &ToolCtx, path: &str) -> Result<std::path::PathBuf, ToolError> {
    ctx.policy.refresh(&ctx.db).await;
    let real = ctx.policy.check_file(path.trim())?;
    if !is_image_path(&real) {
        return Err(ToolError::invalid("the file is not a supported image type"));
    }
    Ok(real)
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct DescribeArgs {
    /// Absolute path of an image inside the library, a watched folder or the Pictures folder.
    pub path: String,
    /// True (default) returns the description only; false also files it as a proposal for review.
    pub dry_run: Option<bool>,
}

async fn describe_image(ctx: Arc<ToolCtx>, call: CallInfo, args: DescribeArgs) -> Result<ToolOutput, ToolError> {
    let real = checked_image(&ctx, &args.path).await?;
    let dry_run = args.dry_run.unwrap_or(true);
    let name = real.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let (vd, gpu) = gpu_job(&ctx, &call, "describe", TaskKind::Describe, format!("describe {}", safe_text(&name, 60)), ctx.vision.describe(&real)).await?;
    let mut data = json!({
        "path": args.path.trim(),
        "description": safe_text(&vd.description, 4000),
        "caption": safe_text(&vd.caption, 300),
        "tags": vd.tags.iter().take(30).map(|t| safe_text(t, 40)).collect::<Vec<_>>(),
        "category": safe_text(&vd.category, 80),
        "dry_run": dry_run,
        "stored": false,
        "gpu": gpu,
    });
    if !dry_run {
        let db = ctx.db.get()?;
        let path = args.path.trim().to_string();
        if row_by_path(db, &path).await?.is_none() {
            data["note"] = json!("the file is not in the library, so no proposal was filed");
        } else {
            let changes = Changes { caption: Some(vd.caption.clone()), description: Some(vd.description.clone()), category: Some(vd.category.clone()), tags_add: vd.tags.clone(), tags_remove: Vec::new() }.normalized()?;
            let (proposal, created) = ctx.proposals.create(NewProposal { path, changes, reason: Some("generated by describe_image".to_string()), source: format!("describe_image:{}", call.client) }).await?;
            data["proposal_id"] = json!(proposal.id);
            data["stored"] = json!(created);
            data["note"] = json!("filed as a pending proposal; nothing was written to the library");
        }
    }
    Ok(ToolOutput::new(format!("described {} in {} ms (queue wait {} ms)", safe_text(&name, 60), gpu.ran_ms, gpu.waited_ms), data))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct EmbedArgs {
    /// Absolute path of an image inside the library, a watched folder or the Pictures folder.
    pub path: String,
    /// True (default) only reports the vector; false stores the embedding of a library item.
    pub dry_run: Option<bool>,
}

#[derive(Debug, SurrealValue)]
struct IdOnly {
    id: RecordId,
}

/// Stores `vector` as the embedding of the library item `path`; false when the item is not in the library.
pub async fn store_embedding(db: &Surreal<Db>, path: &str, vector: &[f32]) -> Result<bool, ToolError> {
    let thumb: Vec<IdOnly> = db.query("SELECT id FROM thumbnails WHERE path = $p LIMIT 1").bind(("p", path.to_string())).await?.take(0)?;
    let Some(thumb) = thumb.into_iter().next() else { return Ok(false) };
    let existing: Vec<IdOnly> = db.query("SELECT id FROM clip_embeddings WHERE path = $p LIMIT 1").bind(("p", path.to_string())).await?.take(0)?;
    match existing.into_iter().next() {
        Some(row) => {
            db.query("UPDATE $id SET embedding = $e, updated = time::now() RETURN NONE").bind(("id", row.id)).bind(("e", vector.to_vec())).await?.check()?;
        }
        None => {
            db.query("CREATE clip_embeddings SET path = $p, thumb_ref = $t, embedding = $e RETURN NONE").bind(("p", path.to_string())).bind(("t", thumb.id)).bind(("e", vector.to_vec())).await?.check()?;
        }
    }
    Ok(true)
}

async fn clip_embed(ctx: Arc<ToolCtx>, call: CallInfo, args: EmbedArgs) -> Result<ToolOutput, ToolError> {
    let real = checked_image(&ctx, &args.path).await?;
    let dry_run = args.dry_run.unwrap_or(true);
    let name = real.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let (vector, gpu) = gpu_job(&ctx, &call, "clip_image", TaskKind::Clip, format!("embed {}", safe_text(&name, 60)), ctx.clip.embed_image(&real)).await?;
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mut data = json!({
        "path": args.path.trim(),
        "dimensions": vector.len(),
        "norm": (norm * 10_000.0).round() / 10_000.0,
        "dry_run": dry_run,
        "stored": false,
        "gpu": gpu,
    });
    if !dry_run {
        if vector.len() != EMBEDDING_DIMENSIONS {
            return Err(ToolError::failed(format!("the model produced {} values but the search index holds {EMBEDDING_DIMENSIONS}; switch the CLIP model in the settings", vector.len())));
        }
        let stored = store_embedding(ctx.db.get()?, args.path.trim(), &vector).await?;
        data["stored"] = json!(stored);
        if !stored {
            data["note"] = json!("the file is not in the library, so nothing was stored");
        }
    }
    Ok(ToolOutput::new(format!("embedded {} ({} values, {} ms)", safe_text(&name, 60), vector.len(), gpu.ran_ms), data))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct QueueArgs {}

async fn gpu_queue_status(ctx: Arc<ToolCtx>, _call: CallInfo, _args: QueueArgs) -> Result<ToolOutput, ToolError> {
    let status = ctx.gpu.status();
    let text = format!("{} running, {} waiting (room for {})", usize::from(status.running.is_some()), status.queued.len(), status.max_queued);
    Ok(ToolOutput::new(text, serde_json::to_value(&status).map_err(ToolError::failed)?))
}
