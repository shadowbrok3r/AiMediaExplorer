//! Test fixtures: an in-memory library, fake models, a recording UI and a fake clock.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem};
use surrealdb::types::{RecordId, SurrealValue};
use tokio_util::sync::CancellationToken;

use super::activity::{ActivityLog, ServerStats};
use super::agent::AgentCtx;
use super::clock::FakeClock;
use super::ctx::{CallInfo, ClipBackend, DbSource, ToolCtx, UiBridge, VisionBackend};
use super::edits::EditJobs;
use super::error::{ToolError, ToolOutput};
use super::gpu::{GpuLimits, GpuQueue};
use super::guide::FixedGuide;
use super::notify::RecordingNotifier;
use super::policy::PathPolicy;
use super::taxonomy::TaxonomyStore;
use super::proposals::ProposalStore;
use super::registry::ToolRegistry;
use super::snapshot::AppSnapshot;
use crate::command::AppCommand;

pub const DIMS: usize = 1024;

pub async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
}

/// A unit vector near the axis `topic`, tilted by `tilt` towards a private side axis.
pub fn topic_vector(topic: usize, tilt: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    v[topic] = 1.0;
    v[100 + topic] = tilt;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn topic_of(text: &str) -> usize {
    let t = text.to_lowercase();
    ["beach", "forest", "city"].iter().position(|k| t.contains(k)).unwrap_or(3)
}

/// CLIP stand-in: the topic comes from a keyword in the text or file name.
pub struct FakeClip {
    pub text_calls: AtomicUsize,
    pub image_calls: AtomicUsize,
    active: AtomicUsize,
    pub max_active: AtomicUsize,
    pub delay: Mutex<Duration>,
    pub fail: AtomicBool,
}

impl FakeClip {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { text_calls: 0.into(), image_calls: 0.into(), active: 0.into(), max_active: 0.into(), delay: Mutex::new(Duration::ZERO), fail: false.into() })
    }

    async fn work(&self) -> anyhow::Result<()> {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        let delay = *self.delay.lock().unwrap();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        anyhow::ensure!(!self.fail.load(Ordering::SeqCst), "the CLIP model is broken");
        Ok(())
    }
}

#[async_trait]
impl ClipBackend for FakeClip {
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.text_calls.fetch_add(1, Ordering::SeqCst);
        self.work().await?;
        Ok(topic_vector(topic_of(text), 0.0))
    }

    async fn embed_image(&self, path: &Path) -> anyhow::Result<Vec<f32>> {
        self.image_calls.fetch_add(1, Ordering::SeqCst);
        self.work().await?;
        Ok(topic_vector(topic_of(&path.to_string_lossy()), 0.05))
    }
}

/// Vision model stand-in that counts overlapping calls.
pub struct FakeVision {
    pub calls: AtomicUsize,
    active: AtomicUsize,
    pub max_active: AtomicUsize,
    pub delay: Mutex<Duration>,
}

impl FakeVision {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { calls: 0.into(), active: 0.into(), max_active: 0.into(), delay: Mutex::new(Duration::ZERO) })
    }
}

#[async_trait]
impl VisionBackend for FakeVision {
    async fn describe(&self, path: &Path) -> anyhow::Result<crate::ai::VisionDescription> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        let delay = *self.delay.lock().unwrap();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        Ok(crate::ai::VisionDescription { description: format!("A picture named {name}"), caption: format!("{name} caption"), tags: vec![name.clone(), "generated".to_string()], category: "test".to_string() })
    }
}

#[derive(Default)]
pub struct RecordingUi {
    pub sent: Mutex<Vec<AppCommand>>,
}

impl RecordingUi {
    pub fn commands(&self) -> Vec<AppCommand> {
        self.sent.lock().unwrap().clone()
    }
}

impl UiBridge for RecordingUi {
    fn send(&self, cmd: AppCommand) {
        self.sent.lock().unwrap().push(cmd);
    }
}

#[derive(Clone, Debug, SurrealValue)]
struct SeedThumb {
    id: RecordId,
    filename: String,
    path: String,
    parent_dir: String,
    size: i64,
    file_type: String,
    tags: Vec<String>,
    caption: Option<String>,
    description: Option<String>,
    category: Option<String>,
    logical_group: RecordId,
}

#[derive(Clone, Debug, SurrealValue)]
struct SeedEmbedding {
    path: String,
    thumb_ref: RecordId,
    embedding: Vec<f32>,
}

/// One seeded library file.
pub struct Seed {
    pub rel: &'static str,
    pub caption: Option<&'static str>,
    pub description: Option<&'static str>,
    pub category: Option<&'static str>,
    pub tags: &'static [&'static str],
    /// Topic and tilt of the stored embedding; none for files without one.
    pub embedding: Option<(usize, f32)>,
}

pub const SEEDS: &[Seed] = &[
    Seed { rel: "photos/beach1.png", caption: Some("Sunny beach"), description: Some("A wide sandy beach with blue sea"), category: Some("landscape"), tags: &["beach", "sea", "summer"], embedding: Some((0, 0.1)) },
    Seed { rel: "photos/beach2.png", caption: Some("Beach at dusk"), description: None, category: Some("landscape"), tags: &["beach", "dusk"], embedding: Some((0, 0.2)) },
    Seed { rel: "photos/sub/beach3.png", caption: None, description: None, category: None, tags: &[], embedding: None },
    Seed { rel: "photos/forest1.png", caption: Some("Pine forest"), description: Some("Tall pines in fog"), category: Some("nature"), tags: &["forest", "fog"], embedding: Some((1, 0.1)) },
    Seed { rel: "photos/city1.png", caption: Some("City night"), description: Some("Neon street"), category: Some("urban"), tags: &["city", "night"], embedding: Some((2, 0.1)) },
];

pub struct Fixture {
    pub ctx: Arc<ToolCtx>,
    pub registry: ToolRegistry,
    pub db: Surreal<Db>,
    pub ui: Arc<RecordingUi>,
    pub clock: Arc<FakeClock>,
    pub clip: Arc<FakeClip>,
    pub vision: Arc<FakeVision>,
    pub activity: Arc<ActivityLog>,
    pub stats: Arc<ServerStats>,
    pub notifier: Arc<RecordingNotifier>,
    pub guide: Arc<FixedGuide>,
    pub root: tempfile::TempDir,
}

impl Fixture {
    /// Absolute path of a seeded or created file.
    pub fn path(&self, rel: &str) -> String {
        self.root.path().join(rel).to_string_lossy().to_string()
    }

    pub fn dir(&self) -> PathBuf {
        self.root.path().to_path_buf()
    }

    pub async fn call(&self, tool: &str, args: Value) -> Result<ToolOutput, ToolError> {
        self.call_as(tool, args, "tester", CancellationToken::new()).await
    }

    pub async fn call_as(&self, tool: &str, args: Value, client: &str, cancel: CancellationToken) -> Result<ToolOutput, ToolError> {
        self.registry.call(self.ctx.clone(), CallInfo { client: client.to_string(), cancel }, tool, args).await
    }

    /// The ids of the items in a page result, in order.
    pub fn paths(out: &ToolOutput) -> Vec<String> {
        out.data["items"].as_array().unwrap().iter().map(|i| i["path"].as_str().unwrap().to_string()).collect()
    }
}

#[derive(Clone, Debug, SurrealValue)]
struct CountRow {
    n: i64,
}

/// Number of rows in `table`.
pub async fn row_count(db: &Surreal<Db>, table: &str) -> i64 {
    let rows: Vec<CountRow> = db.query(format!("SELECT count() AS n FROM {table} GROUP ALL")).await.unwrap().take(0).unwrap();
    rows.first().map_or(0, |r| r.n)
}

fn write_png(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([200, 120, 40])).save(path).unwrap();
}

/// A library of five files in a temporary folder, with an extra file outside it.
pub async fn fixture() -> Fixture {
    fixture_with(GpuLimits { max_queued: 8, queue_timeout: Duration::from_secs(30), job_timeout: Duration::from_secs(30) }).await
}

pub async fn fixture_with(limits: GpuLimits) -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let photos = root.path().join("photos");
    let db = mem_db().await;
    let group = RecordId::new("logical_groups", "Default");
    let mut thumbs = Vec::new();
    let mut embeddings = Vec::new();
    for (i, seed) in SEEDS.iter().enumerate() {
        let path = root.path().join(seed.rel);
        write_png(&path);
        let id = RecordId::new("thumbnails", format!("t{i:03}"));
        let path_text = path.to_string_lossy().to_string();
        thumbs.push(SeedThumb {
            id: id.clone(),
            filename: path.file_name().unwrap().to_string_lossy().to_string(),
            path: path_text.clone(),
            parent_dir: path.parent().unwrap().to_string_lossy().to_string(),
            size: 100 + i as i64,
            file_type: "png".to_string(),
            tags: seed.tags.iter().map(|t| t.to_string()).collect(),
            caption: seed.caption.map(str::to_string),
            description: seed.description.map(str::to_string),
            category: seed.category.map(str::to_string),
            logical_group: group.clone(),
        });
        if let Some((topic, tilt)) = seed.embedding {
            embeddings.push(SeedEmbedding { path: path_text, thumb_ref: id, embedding: topic_vector(topic, tilt) });
        }
    }
    db.query("INSERT INTO thumbnails $rows").bind(("rows", thumbs)).await.unwrap().check().unwrap();
    db.query("INSERT INTO clip_embeddings $rows").bind(("rows", embeddings)).await.unwrap().check().unwrap();
    write_png(&root.path().join("outside_library/secret.png"));
    std::fs::write(photos.join("notes.txt"), "not an image").unwrap();

    let holidays = crate::database::collections::create(&db, "Holidays").await.unwrap();
    crate::database::collections::create(&db, "Empty").await.unwrap();
    let members: Vec<String> = ["photos/beach1.png", "photos/beach2.png"].iter().map(|r| root.path().join(r).to_string_lossy().to_string()).collect();
    crate::database::collections::add_paths(&db, &holidays.id, &members).await.unwrap();

    let clock = FakeClock::new();
    let clip = FakeClip::new();
    let vision = FakeVision::new();
    let ui = Arc::new(RecordingUi::default());
    let activity = Arc::new(ActivityLog::default());
    let stats = Arc::new(ServerStats::default());
    let notifier = Arc::new(RecordingNotifier::default());
    let guide = Arc::new(FixedGuide::new());
    let agent = Arc::new(AgentCtx {
        taxonomy: Arc::new(TaxonomyStore::new(DbSource::Handle(db.clone()), clock.clone())),
        edits: EditJobs::new(Arc::new(crate::ui::image_edit::DebugBackend::default()), root.path().join("scratch"), clock.clone(), notifier.clone()),
        notifier: notifier.clone(),
        guide: guide.clone(),
    });
    let ctx = Arc::new(ToolCtx {
        db: DbSource::Handle(db.clone()),
        clock: clock.clone(),
        clip: clip.clone(),
        vision: vision.clone(),
        policy: Arc::new(PathPolicy::with_roots(vec![photos])),
        gpu: Arc::new(GpuQueue::new(limits, clock.clone())),
        ui: ui.clone(),
        proposals: Arc::new(ProposalStore::new(DbSource::Handle(db.clone()), clock.clone())),
        snapshot: Arc::new(std::sync::RwLock::new(AppSnapshot::default())),
        agent,
    });
    let registry = super::tools::build_registry(activity.clone(), stats.clone());
    Fixture { ctx, registry, db, ui, clock, clip, vision, activity, stats, notifier, guide, root }
}
