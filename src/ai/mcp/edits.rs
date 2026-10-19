//! Image edits started by tools: run on the Image Edit backend through the GPU queue, written to a scratch folder and offered to the person as a result to save.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::clock::Clock;
use super::ctx::DbSource;
use super::error::ToolError;
use super::gpu::{GpuError, GpuQueue, JobSpec};
use super::notify::Notifier;
use super::tools::common::safe_text;
use super::tools::gpu::task_cancelled;
use crate::command::ToastLevel;
use crate::tasks::{TaskKind, TaskSpec};
use crate::ui::image_edit::model::{EditError, unique_output_path};
use crate::ui::image_edit::{EditBackend, EditRequest, ModelSettings, Params, RunCtl};

/// Jobs allowed to wait or run at once.
pub const MAX_ACTIVE: usize = 3;
/// Finished results allowed to wait for the person before new jobs are refused.
pub const MAX_UNREVIEWED: usize = 20;
/// Jobs kept in the list; the oldest reviewed or failed ones go first.
const KEEP: usize = 40;
/// Longest wait for the GPU and longest run of an edit, including loading the model.
const WAIT: Duration = Duration::from_secs(15 * 60);
const RUN: Duration = Duration::from_secs(45 * 60);
/// Folder name of the results inside the application data folder.
pub const SCRATCH_DIR: &str = "agent-edits";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditState {
    Queued,
    LoadingModel,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl EditState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::LoadingModel => "loading_model",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// True while the job waits or runs.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::LoadingModel | Self::Running)
    }
}

/// What the person decided about a finished result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Review {
    /// Nothing to review: the job is not done.
    NotReady,
    Pending,
    /// Saved as a new file next to the original.
    Saved,
    /// Saved next to the original and added to the library.
    Added,
    Rejected,
}

impl Review {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotReady => "not_ready",
            Self::Pending => "pending",
            Self::Saved => "saved",
            Self::Added => "added",
            Self::Rejected => "rejected",
        }
    }
}

/// The file a finished edit left in the scratch folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditResult {
    pub file: PathBuf,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
}

/// One edit job as people and tools see it.
#[derive(Clone, Debug)]
pub struct EditJob {
    pub id: String,
    pub client: String,
    /// The file the edit started from.
    pub source: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f32,
    pub strength: f32,
    pub state: EditState,
    pub done: u64,
    pub total: u64,
    pub detail: String,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub error: Option<String>,
    pub result: Option<EditResult>,
    pub review: Review,
    /// Where a saved result went.
    pub saved_to: Option<String>,
}

impl EditJob {
    pub fn file_name(&self) -> String {
        Path::new(&self.source).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| self.source.clone())
    }

    /// The JSON the tools return; the scratch path stays inside the application.
    pub fn to_json(&self, now_ms: u64, queue_position: Option<usize>) -> Value {
        let end = self.finished_ms.unwrap_or(now_ms);
        json!({
            "job_id": self.id,
            "state": self.state.as_str(),
            "path": self.source,
            "prompt": safe_text(&self.prompt, 300),
            "progress": {"done": self.done, "total": self.total, "detail": safe_text(&self.detail, 80)},
            "queue_position": queue_position,
            "created_ms": self.created_ms,
            "elapsed_ms": self.started_ms.map(|s| end.saturating_sub(s)),
            "error": self.error.as_deref().map(|e| safe_text(e, 300)),
            "result": self.result.as_ref().map(|r| json!({"width": r.width, "height": r.height, "bytes": r.bytes})),
            "review": self.review.as_str(),
            "saved_to": self.saved_to,
            "note": match (self.state, self.review) {
                (EditState::Done, Review::Pending) => "the result waits for the user to save or reject it in the application; the original is untouched",
                (EditState::Done, Review::Saved | Review::Added) => "the user saved the result as a new file",
                (EditState::Done, Review::Rejected) => "the user rejected the result",
                _ => "",
            },
        })
    }
}

struct Slot {
    job: EditJob,
    cancel: CancellationToken,
}

/// Why a running edit stopped before it had a result.
enum Stop {
    Cancelled,
    Failed(String),
}

pub struct EditJobs {
    backend: Arc<dyn EditBackend>,
    scratch: PathBuf,
    clock: Arc<dyn Clock>,
    notifier: Arc<dyn Notifier>,
    slots: Mutex<Vec<Slot>>,
}

/// Sets the cancel flag of an edit when dropped before the edit ended.
struct StopOnDrop {
    flag: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(true, Ordering::SeqCst);
        }
    }
}

fn stop_of(e: EditError) -> Stop {
    Stop::Failed(e.message)
}

/// The folder results are written to when none is given: inside the application data folder.
pub fn default_scratch() -> PathBuf {
    directories::ProjectDirs::from("", "", "smart-media").map(|d| d.data_local_dir().join(SCRATCH_DIR)).unwrap_or_else(|| std::env::temp_dir().join("smart-media").join(SCRATCH_DIR))
}

impl EditJobs {
    pub fn new(backend: Arc<dyn EditBackend>, scratch: PathBuf, clock: Arc<dyn Clock>, notifier: Arc<dyn Notifier>) -> Arc<Self> {
        Arc::new(Self { backend, scratch, clock, notifier, slots: Mutex::new(Vec::new()) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Slot>> {
        self.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn scratch_dir(&self) -> &Path {
        &self.scratch
    }

    /// Jobs, newest first.
    pub fn list(&self) -> Vec<EditJob> {
        self.lock().iter().rev().map(|s| s.job.clone()).collect()
    }

    pub fn get(&self, id: &str) -> Option<EditJob> {
        self.lock().iter().find(|s| s.job.id == id).map(|s| s.job.clone())
    }

    /// Jobs that wait or run, and finished results nobody has reviewed.
    pub fn counts(&self) -> (usize, usize) {
        let slots = self.lock();
        (slots.iter().filter(|s| s.job.state.is_active()).count(), slots.iter().filter(|s| s.job.review == Review::Pending).count())
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut EditJob)) {
        if let Some(slot) = self.lock().iter_mut().find(|s| s.job.id == id) {
            f(&mut slot.job);
        }
    }

    fn new_id() -> Result<String, ToolError> {
        let mut bytes = [0u8; 4];
        getrandom::fill(&mut bytes).map_err(ToolError::failed)?;
        Ok(format!("e{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()))
    }

    /// Drops the oldest jobs that need nothing more from anyone once the list is long.
    fn trim(&self) {
        let mut slots = self.lock();
        while slots.len() > KEEP {
            let Some(at) = slots.iter().position(|s| !s.job.state.is_active() && s.job.review != Review::Pending) else { break };
            let gone = slots.remove(at);
            if let Some(r) = gone.job.result {
                let _ = std::fs::remove_file(r.file);
            }
        }
    }

    /// Registers an edit of `source` and runs it in the background; the call returns at once.
    pub fn start(self: &Arc<Self>, gpu: Arc<GpuQueue>, client: &str, source: PathBuf, shown_source: &str, params: Params) -> Result<EditJob, ToolError> {
        let (active, unreviewed) = self.counts();
        if active >= MAX_ACTIVE {
            return Err(ToolError::Busy(format!("{active} image edits are already queued or running; wait for one to finish or cancel one")));
        }
        if unreviewed >= MAX_UNREVIEWED {
            return Err(ToolError::Busy(format!("{unreviewed} finished edits wait for the user's review; ask the user to save or reject them first")));
        }
        let status = gpu.status();
        if status.queued.len() >= status.max_queued {
            return Err(ToolError::Busy(format!("the GPU queue is full ({} of {} places taken); try again later", status.queued.len(), status.max_queued)));
        }
        let id = Self::new_id()?;
        let cancel = CancellationToken::new();
        let job = EditJob {
            id: id.clone(),
            client: client.to_string(),
            source: shown_source.to_string(),
            prompt: params.prompt.clone(),
            negative_prompt: Some(params.negative_prompt.clone()).filter(|n| !n.is_empty()),
            steps: params.steps,
            guidance_scale: params.guidance_scale,
            strength: params.strength,
            state: EditState::Queued,
            done: 0,
            total: params.steps as u64,
            detail: String::new(),
            created_ms: self.clock.unix_ms(),
            started_ms: None,
            finished_ms: None,
            error: None,
            result: None,
            review: Review::NotReady,
            saved_to: None,
        };
        self.lock().push(Slot { job: job.clone(), cancel: cancel.clone() });
        self.trim();
        let req = EditRequest { source, params, settings: ModelSettings::default() };
        tokio::spawn(self.clone().run(gpu, id, client.to_string(), req, cancel));
        Ok(job)
    }

    /// Stops a queued or running job.
    pub fn cancel(&self, id: &str) -> Result<EditJob, ToolError> {
        let token = {
            let mut slots = self.lock();
            let slot = slots.iter_mut().find(|s| s.job.id == id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {id}")))?;
            if !slot.job.state.is_active() {
                return Err(ToolError::invalid(format!("job {id} is already {}", slot.job.state.as_str())));
            }
            slot.job.state = EditState::Cancelled;
            slot.job.finished_ms = Some(self.clock.unix_ms());
            slot.cancel.clone()
        };
        token.cancel();
        self.get(id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {id}")))
    }

    async fn run(self: Arc<Self>, gpu: Arc<GpuQueue>, id: String, client: String, req: EditRequest, cancel: CancellationToken) {
        let name = req.source.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let spec = JobSpec::new("image_edit", format!("edit {} ({id})", safe_text(&name, 40)), client).with_limits(WAIT, RUN);
        let this = self.clone();
        let (job_id, task_name) = (id.clone(), name.clone());
        let outcome = gpu
            .run(spec, &cancel, move |_| async move {
                this.update(&job_id, |j| j.started_ms = Some(this.clock.unix_ms()));
                let handle = crate::tasks::start(TaskSpec::new(TaskKind::ImageEdit, format!("MCP: edit {}", safe_text(&task_name, 60))).visible_after(Duration::from_secs(2)));
                let reporter = handle.reporter();
                let flag = Arc::new(AtomicBool::new(false));
                let mut guard = StopOnDrop { flag: flag.clone(), armed: true };
                let work = this.run_edit(&job_id, req, flag.clone(), reporter.clone());
                let result = tokio::select! {
                    r = work => r,
                    _ = task_cancelled(reporter) => {
                        flag.store(true, Ordering::SeqCst);
                        Err(Stop::Cancelled)
                    }
                };
                guard.armed = false;
                match &result {
                    Ok(_) => handle.complete(),
                    Err(Stop::Cancelled) => handle.cancelled(),
                    Err(Stop::Failed(e)) => handle.fail(e.clone()),
                }
                Ok::<_, String>(result)
            })
            .await;
        match outcome {
            Ok((Ok(result), _)) => {
                self.update(&id, |j| {
                    j.state = EditState::Done;
                    j.review = Review::Pending;
                    j.finished_ms = Some(self.clock.unix_ms());
                    j.result = Some(result);
                });
                self.notifier.notify(ToastLevel::Success, &format!("Image edit of {} is ready. Review it in Settings > Metadata tools.", safe_text(&name, 60)));
            }
            Ok((Err(Stop::Cancelled), _)) | Err(GpuError::Cancelled) => self.finish_cancelled(&id),
            Ok((Err(Stop::Failed(why)), _)) => self.fail(&id, &name, why),
            Err(e) => self.fail(&id, &name, e.to_string()),
        }
    }

    fn finish_cancelled(&self, id: &str) {
        self.update(id, |j| {
            if j.state.is_active() {
                j.state = EditState::Cancelled;
                j.finished_ms = Some(self.clock.unix_ms());
            }
        });
    }

    fn fail(&self, id: &str, name: &str, why: String) {
        let already = self.get(id).is_some_and(|j| j.state == EditState::Cancelled);
        self.update(id, |j| {
            if !already {
                j.state = EditState::Failed;
                j.finished_ms = Some(self.clock.unix_ms());
                j.error = Some(why.clone());
            }
        });
        if !already {
            self.notifier.notify(ToastLevel::Error, &format!("Image edit of {} failed: {}", safe_text(name, 60), safe_text(&why, 160)));
        }
    }

    async fn run_edit(self: &Arc<Self>, id: &str, req: EditRequest, flag: Arc<AtomicBool>, task: crate::tasks::Reporter) -> Result<EditResult, Stop> {
        if !self.backend.is_loaded() {
            self.update(id, |j| j.state = EditState::LoadingModel);
            self.backend.load(&req.settings, &req.params).await.map_err(stop_of)?;
        }
        if flag.load(Ordering::SeqCst) {
            return Err(Stop::Cancelled);
        }
        let total = req.params.steps as u64;
        self.update(id, |j| {
            j.state = EditState::Running;
            j.total = total;
        });
        let progress: Arc<dyn Fn(u64, u64, &str) + Send + Sync> = {
            let (this, id) = (self.clone(), id.to_string());
            Arc::new(move |done, total, detail| {
                this.update(&id, |j| {
                    if j.state == EditState::Running {
                        j.done = done;
                        j.total = total;
                        j.detail = detail.to_string();
                    }
                });
                task.progress(done, total);
            })
        };
        let ctl = RunCtl { cancel: flag.clone(), progress, preview: None };
        let png = self.backend.edit(&req, &ctl).await.map_err(stop_of)?;
        if flag.load(Ordering::SeqCst) {
            return Err(Stop::Cancelled);
        }
        let file = self.scratch.join(format!("{id}.png"));
        let written = file.clone();
        let scratch = self.scratch.clone();
        tokio::task::spawn_blocking(move || write_scratch(&scratch, &written, &png)).await.map_err(|e| Stop::Failed(e.to_string()))?.map_err(Stop::Failed)
            .map(|(bytes, width, height)| EditResult { file, bytes, width, height })
    }

    /// Writes the result next to the original under a name that is not taken; the original is never written.
    pub fn save_new(&self, id: &str) -> Result<(EditJob, PathBuf), ToolError> {
        let job = self.reviewable(id)?;
        let result = job.result.clone().ok_or_else(|| ToolError::failed("the job has no result"))?;
        let target = unique_output_path(&job.source, "png", |p| p.exists());
        let bytes = std::fs::read(&result.file).map_err(|e| ToolError::failed(format!("the result could not be read: {e}")))?;
        write_new(&target, &bytes).map_err(ToolError::Failed)?;
        let _ = std::fs::remove_file(&result.file);
        let shown = target.display().to_string();
        self.update(id, |j| {
            j.review = Review::Saved;
            j.saved_to = Some(shown.clone());
        });
        Ok((self.get(id).unwrap_or(job), target))
    }

    /// [`save_new`](Self::save_new), then adds the new file to the library.
    pub async fn add_to_library(&self, id: &str, db: &DbSource) -> Result<(EditJob, PathBuf), ToolError> {
        let (_, target) = self.save_new(id)?;
        let handle = db.get()?;
        let seed = crate::database::metadata_edit::RowSeed::from_path(&target.display().to_string());
        crate::database::metadata_edit::ensure_rows(handle, &[seed]).await.map_err(|e| ToolError::failed(format!("{e:#}")))?;
        self.update(id, |j| j.review = Review::Added);
        Ok((self.get(id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {id}")))?, target))
    }

    /// Discards the result of a finished job.
    pub fn reject(&self, id: &str) -> Result<EditJob, ToolError> {
        let job = self.reviewable(id)?;
        if let Some(r) = &job.result {
            let _ = std::fs::remove_file(&r.file);
        }
        self.update(id, |j| j.review = Review::Rejected);
        self.get(id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {id}")))
    }

    /// The job `id` when it is finished and still waits for a decision.
    fn reviewable(&self, id: &str) -> Result<EditJob, ToolError> {
        let job = self.get(id).ok_or_else(|| ToolError::NotFound(format!("no image edit job {id}")))?;
        match (job.state, job.review) {
            (EditState::Done, Review::Pending) => Ok(job),
            (EditState::Done, review) => Err(ToolError::invalid(format!("the result of job {id} is already {}", review.as_str()))),
            (state, _) => Err(ToolError::invalid(format!("job {id} is {} and has no result to review", state.as_str()))),
        }
    }

    /// The scratch file of a finished job, for the preview.
    pub fn result_file(&self, id: &str) -> Option<PathBuf> {
        self.get(id).and_then(|j| j.result).map(|r| r.file).filter(|f| f.is_file())
    }
}

/// Writes `png` to `file` through a temporary name; returns its size and dimensions.
fn write_scratch(scratch: &Path, file: &Path, png: &[u8]) -> Result<(u64, u32, u32), String> {
    std::fs::create_dir_all(scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    let part = file.with_extension("png.part");
    std::fs::write(&part, png).map_err(|e| format!("{}: {e}", part.display()))?;
    std::fs::rename(&part, file).map_err(|e| format!("{}: {e}", file.display()))?;
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(png)).with_guessed_format().map_err(|e| e.to_string())?.into_dimensions().map_err(|e| format!("the result is not a picture: {e}"))?;
    Ok((png.len() as u64, w, h))
}

/// Writes `bytes` to `path`, failing when the file exists.
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn states_name_themselves_and_only_waiting_and_running_are_active() {
        for (s, name, active) in [(EditState::Queued, "queued", true), (EditState::LoadingModel, "loading_model", true), (EditState::Running, "running", true), (EditState::Done, "done", false), (EditState::Failed, "failed", false), (EditState::Cancelled, "cancelled", false)] {
            assert_eq!((s.as_str(), s.is_active()), (name, active));
        }
    }

    #[test]
    fn a_result_is_written_whole_and_never_over_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let img = image::RgbImage::from_pixel(6, 4, image::Rgb([1, 2, 3]));
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let file = dir.path().join("scratch").join("e1.png");
        let (bytes, w, h) = write_scratch(&dir.path().join("scratch"), &file, png.get_ref()).unwrap();
        assert_eq!((w, h, bytes), (6, 4, png.get_ref().len() as u64));
        assert!(file.is_file() && !file.with_extension("png.part").exists());
        let target = dir.path().join("a-edited.png");
        write_new(&target, b"one").unwrap();
        assert!(write_new(&target, b"two").is_err(), "an existing file is never replaced");
        assert_eq!(std::fs::read(&target).unwrap(), b"one");
        assert!(write_scratch(dir.path(), &dir.path().join("bad.png"), b"not a picture").is_err());
    }

    #[test]
    fn the_scratch_folder_is_inside_the_data_folder() {
        let dir = default_scratch();
        assert!(dir.ends_with(SCRATCH_DIR));
    }
}
