//! Tools that edit a picture with the local image edit model as a background job whose result the user reviews.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::safe_text;
use super::gpu::checked_image;
use super::preview::encode_jpeg;
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::edits::{EditJob, EditState};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};
use crate::ui::image_edit::Params;

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PROMPT: usize = 1000;
const MAX_NEGATIVE: usize = 500;
const PREVIEW_PX: u32 = 768;
const MAX_LISTED: usize = 20;

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("image_edit_start", ToolClass::Gpu, TIMEOUT, "Start an image edit", "Edit a picture with the local image edit model. The call returns a job id at once; the job waits its turn in the GPU queue and can take minutes. The original is never changed: the result is written to a scratch folder of the application and offered to the user, who saves it as a new file or rejects it. Poll image_edit_status for progress.", image_edit_start),
        entry("image_edit_status", ToolClass::Read, Duration::from_secs(20), "Image edit status", "Progress of one image edit job (state, steps done, queue position, error, result size and whether the user saved it), or the most recent jobs when no job_id is given. With include_preview the finished result is attached as a JPEG.", image_edit_status),
        entry("image_edit_cancel", ToolClass::Gpu, Duration::from_secs(10), "Cancel an image edit", "Stop a queued or running image edit job. A job that is already done cannot be cancelled.", image_edit_cancel),
    ]
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct StartArgs {
    /// Absolute path of an image inside the library, a watched folder or the Pictures folder. It is only read.
    pub path: String,
    /// What to change in the picture, up to 1000 characters.
    pub prompt: String,
    /// What the result should avoid, up to 500 characters.
    pub negative_prompt: Option<String>,
    /// Denoising steps, 1 to 50; default 30.
    pub steps: Option<usize>,
    /// How closely to follow the prompt, 1 to 20; default 7.5.
    pub guidance_scale: Option<f32>,
    /// How far to move away from the original, 0.1 to 1; default 0.7.
    pub strength: Option<f32>,
}

/// Edit settings from the arguments, or what is wrong with them.
fn params_of(args: &StartArgs) -> Result<Params, ToolError> {
    let clean = |text: &str, max: usize| text.chars().filter(|c| !c.is_control() || c.is_whitespace()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ").chars().take(max).collect::<String>();
    let prompt = clean(&args.prompt, MAX_PROMPT);
    if prompt.is_empty() {
        return Err(ToolError::invalid("prompt must say what to change"));
    }
    let mut params = Params { prompt, negative_prompt: args.negative_prompt.as_deref().map(|n| clean(n, MAX_NEGATIVE)).unwrap_or_default(), ..Params::default() };
    if let Some(steps) = args.steps {
        if !(1..=50).contains(&steps) {
            return Err(ToolError::invalid("steps must be between 1 and 50"));
        }
        params.steps = steps;
    }
    if let Some(g) = args.guidance_scale {
        if !(1.0..=20.0).contains(&g) {
            return Err(ToolError::invalid("guidance_scale must be between 1 and 20"));
        }
        params.guidance_scale = g;
    }
    if let Some(s) = args.strength {
        if !(0.1..=1.0).contains(&s) {
            return Err(ToolError::invalid("strength must be between 0.1 and 1"));
        }
        params.strength = s;
    }
    Ok(params)
}

fn position(ctx: &ToolCtx, job: &EditJob) -> Option<usize> {
    if job.state != EditState::Queued {
        return None;
    }
    let status = ctx.gpu.status();
    let marker = format!("({})", job.id);
    let ahead = status.queued.iter().position(|q| q.label.ends_with(&marker))?;
    Some(ahead + usize::from(status.running.is_some()))
}

async fn image_edit_start(ctx: Arc<ToolCtx>, call: CallInfo, args: StartArgs) -> Result<ToolOutput, ToolError> {
    let params = params_of(&args)?;
    let real = checked_image(&ctx, &args.path).await?;
    let job = ctx.agent.edits.start(ctx.gpu.clone(), &call.client, real, args.path.trim(), params)?;
    let mut data = job.to_json(ctx.clock.unix_ms(), position(&ctx, &job));
    data["poll_with"] = json!("image_edit_status");
    Ok(ToolOutput::new(format!("image edit job {} is queued; poll image_edit_status for progress. The original stays as it is.", job.id), data))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct StatusArgs {
    /// The job to report; without it the most recent jobs are listed.
    pub job_id: Option<String>,
    /// Attach the finished result as a JPEG up to 768 pixels; only with job_id.
    pub include_preview: Option<bool>,
    /// Jobs to list when no job_id is given, 1 to 20; default 10.
    pub limit: Option<usize>,
}

async fn image_edit_status(ctx: Arc<ToolCtx>, _call: CallInfo, args: StatusArgs) -> Result<ToolOutput, ToolError> {
    let edits = &ctx.agent.edits;
    let now = ctx.clock.unix_ms();
    let Some(id) = args.job_id.as_deref().map(str::trim).filter(|i| !i.is_empty()) else {
        let limit = args.limit.unwrap_or(10).clamp(1, MAX_LISTED);
        let jobs = edits.list();
        let (active, unreviewed) = edits.counts();
        let items: Vec<_> = jobs.iter().take(limit).map(|j| j.to_json(now, position(&ctx, j))).collect();
        let data = json!({"items": items, "returned": items.len(), "total": jobs.len(), "active": active, "waiting_for_review": unreviewed});
        return Ok(ToolOutput::new(format!("{} of {} image edit jobs; {active} active", items.len(), jobs.len()), data));
    };
    let job = edits.get(id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {}", safe_text(id, 40))))?;
    let data = job.to_json(now, position(&ctx, &job));
    let mut text = match job.state {
        EditState::Running | EditState::LoadingModel => format!("job {} is {}, step {} of {}", job.id, job.state.as_str(), job.done, job.total),
        state => format!("job {} is {}", job.id, state.as_str()),
    };
    let mut out = ToolOutput::new(String::new(), data);
    if args.include_preview.unwrap_or(false) && job.state == EditState::Done {
        match edits.result_file(&job.id) {
            Some(file) => {
                let bytes = tokio::task::spawn_blocking(move || {
                    let img = image::open(&file).map_err(|e| ToolError::failed(format!("the result could not be read: {e}")))?;
                    encode_jpeg(&img, PREVIEW_PX)
                })
                .await
                .map_err(ToolError::failed)??;
                out = out.with_image("image/jpeg", base64::engine::general_purpose::STANDARD.encode(&bytes.0));
                text.push_str("; the result is attached");
            }
            None => text.push_str("; the result file is no longer available"),
        }
    }
    out.text = text;
    Ok(out)
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct CancelArgs {
    /// The job to stop, as returned by image_edit_start.
    pub job_id: String,
}

async fn image_edit_cancel(ctx: Arc<ToolCtx>, _call: CallInfo, args: CancelArgs) -> Result<ToolOutput, ToolError> {
    let job = ctx.agent.edits.cancel(args.job_id.trim())?;
    Ok(ToolOutput::new(format!("job {} was cancelled", job.id), job.to_json(ctx.clock.unix_ms(), None)))
}
