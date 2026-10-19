//! Everything a tool needs: the database, the models, the UI, the path policy and the GPU queue.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio_util::sync::CancellationToken;

use super::clock::Clock;
use super::error::ToolError;
use super::gpu::GpuQueue;
use super::policy::PathPolicy;
use super::proposals::ProposalStore;
use super::snapshot::SnapshotCell;
use crate::command::AppCommand;

/// The database a tool reads: the application's or a handle given by a test.
#[derive(Clone)]
pub enum DbSource {
    Global,
    Handle(Surreal<Db>),
}

impl DbSource {
    pub fn get(&self) -> Result<&Surreal<Db>, ToolError> {
        match self {
            Self::Handle(db) => Ok(db),
            Self::Global if crate::database::is_connected() => Ok(&crate::database::DB),
            Self::Global => Err(ToolError::Unavailable("the library database is not connected yet".to_string())),
        }
    }
}

/// Text and image embedding by the local CLIP model.
#[async_trait]
pub trait ClipBackend: Send + Sync {
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>>;
    async fn embed_image(&self, path: &Path) -> anyhow::Result<Vec<f32>>;
}

/// Description of an image by the local vision model.
#[async_trait]
pub trait VisionBackend: Send + Sync {
    async fn describe(&self, path: &Path) -> anyhow::Result<crate::ai::VisionDescription>;
}

/// Where commands for the UI go.
pub trait UiBridge: Send + Sync {
    fn send(&self, cmd: AppCommand);
}

/// The application's command bus.
pub struct BusUi;

impl UiBridge for BusUi {
    fn send(&self, cmd: AppCommand) {
        crate::command::send(cmd);
    }
}

/// The local models of the application.
pub struct EngineBackends;

#[async_trait]
impl ClipBackend for EngineBackends {
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let engine = &crate::ai::GLOBAL_AI_ENGINE;
        engine.ensure_clip_engine().await?;
        let mut guard = engine.clip_engine.lock().await;
        let clip = guard.as_mut().ok_or_else(|| anyhow::anyhow!("the CLIP model is not loaded"))?;
        clip.embed_text(text)
    }

    async fn embed_image(&self, path: &Path) -> anyhow::Result<Vec<f32>> {
        let engine = &crate::ai::GLOBAL_AI_ENGINE;
        engine.ensure_clip_engine().await?;
        let mut guard = engine.clip_engine.lock().await;
        let clip = guard.as_mut().ok_or_else(|| anyhow::anyhow!("the CLIP model is not loaded"))?;
        clip.embed_image_path(&path.to_string_lossy())
    }
}

#[async_trait]
impl VisionBackend for EngineBackends {
    async fn describe(&self, path: &Path) -> anyhow::Result<crate::ai::VisionDescription> {
        crate::ai::GLOBAL_AI_ENGINE
            .generate_vision_description(&path.to_path_buf())
            .await
            .ok_or_else(|| anyhow::anyhow!("the vision model produced no description"))
    }
}

/// Shared state behind every tool call.
pub struct ToolCtx {
    pub db: DbSource,
    pub clock: Arc<dyn Clock>,
    pub clip: Arc<dyn ClipBackend>,
    pub vision: Arc<dyn VisionBackend>,
    pub policy: Arc<PathPolicy>,
    pub gpu: Arc<GpuQueue>,
    pub ui: Arc<dyn UiBridge>,
    pub proposals: Arc<ProposalStore>,
    pub snapshot: SnapshotCell,
}

/// Facts about one call.
#[derive(Clone)]
pub struct CallInfo {
    /// Short label of the caller, for the activity log.
    pub client: String,
    /// Cancelled when the caller abandons the call.
    pub cancel: CancellationToken,
}

impl CallInfo {
    pub fn in_process(label: &str) -> Self {
        Self { client: label.to_string(), cancel: CancellationToken::new() }
    }
}
