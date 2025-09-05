pub mod ai_search;
pub mod index;
pub mod data_extraction;
pub mod generate;
pub mod cache;
pub mod clip; // CLIP embeddings & similarity
#[cfg(feature = "joycaption")]
pub mod joycaption_adapter;
#[cfg(feature = "joycaption")]
#[path = "candle-llava/mod.rs"]
#[cfg(feature = "joycaption")]
pub mod candle_llava;
pub mod bulk;

pub use ai_search::*;

use crate::database::FileMetadata;
use once_cell::sync::Lazy;

// Global lazy AI engine accessor. Initialized on first use. Heavy components (vision model)
// are loaded explicitly via `init_global_ai_engine_async()` to avoid blocking UI thread.
pub static GLOBAL_AI_ENGINE: Lazy<AISearchEngine> = Lazy::new(|| AISearchEngine::new());

/// Kick off async initialization of the global AI engine (vision model + cached rows).

/// Log current process memory (RSS / Working Set) for diagnostics.
/// Enabled by callers (e.g., bulk) when SM_LOG_MEM env var is set.
#[allow(unused)]
pub fn log_memory_usage(note: Option<&str>) {
    #[cfg(windows)]
    {
        use windows::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows::Win32::System::Threading::GetCurrentProcess;
        unsafe {
            let handle = GetCurrentProcess();
            let mut counters = PROCESS_MEMORY_COUNTERS::default();
            if K32GetProcessMemoryInfo(handle, &mut counters, std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32).as_bool() {
                let rss = counters.WorkingSetSize as f64 / (1024.0 * 1024.0);
                if let Some(n) = note { log::info!("[mem] {:.2} MiB - {}", rss, n); } else { log::info!("[mem] {:.2} MiB", rss); }
            }
        }
    }
    #[cfg(not(windows))]
    {
        // Fallback: use sysinfo only if added; otherwise noop
        if let Some(n) = note { log::info!("[mem] note={}", n); }
    }
}
/// Safe to call multiple times; subsequent calls become no-ops.
pub async fn init_global_ai_engine_async() {
    // Ensure background index worker & model; load cached metadata.
    // We purposefully do these sequentially to minimize concurrent model load attempts.
    GLOBAL_AI_ENGINE.ensure_index_worker().await;
    if let Err(e) = GLOBAL_AI_ENGINE.ensure_vision_model().await {
        log::warn!("[AI] vision model init failed: {e}");
    }
    let loaded = GLOBAL_AI_ENGINE.load_cached().await;
    log::info!("[AI] global engine initialized (cached {loaded} rows)");
}

// AI Search Engine core (semantic/kalosm integration removed)
#[derive(Clone)]
pub struct AISearchEngine {
    pub files: std::sync::Arc<tokio::sync::Mutex<Vec<FileMetadata>>>,
    pub path_to_id: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, String>>>,
    pub indexing_in_progress: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, usize>>>, // path -> reentry count
    

    // Control flags for manual vs automatic behaviors
    pub auto_descriptions_enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub auto_clip_enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,

    // Async indexing queue (fire-and-forget). UI enqueues metadata; background worker performs heavy work on Tokio runtime.
    pub index_tx: std::sync::Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<FileMetadata>>>>,
    // Progress metrics (atomics for cheap cross-thread reads)
    pub index_queue_len: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub index_active: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub index_completed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    // CLIP engine (fastembed) for image + text embeddings
    pub clip_engine: std::sync::Arc<tokio::sync::Mutex<Option<crate::ai::clip::ClipEngine>>>,
    // Currently active path for vision description generation (UI can auto-follow)
    pub active_vision_path: std::sync::Arc<tokio::sync::Mutex<Option<String>>>,
}

impl AISearchEngine {
    /// Start background indexing worker if not already started.
    pub async fn ensure_index_worker(&self) {
        let mut guard = self.index_tx.lock().await;
        if guard.is_some() { return; }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<FileMetadata>();
        *guard = Some(tx);
        let engine = self.clone();
        tokio::spawn(async move {
            while let Some(meta) = rx.recv().await {
                let path = meta.path.clone();
                engine.index_active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(e) = engine.index_file(meta).await {
                    log::warn!("[AI] queue index failed for {}: {}", path, e);
                }
                engine.index_active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                engine.index_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Decrement queue len (saturating)
                engine.index_queue_len.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |v| Some(v.saturating_sub(1))).ok();
            }
            log::info!("[AI] indexing worker channel closed");
        });
    }

    /// Enqueue a file metadata record for background indexing. Returns false if queue not ready yet.
    pub async fn enqueue_index(&self, meta: FileMetadata) -> bool {
        if self.index_tx.lock().await.is_none() { self.ensure_index_worker().await; }
        let sent = if let Some(tx) = self.index_tx.lock().await.as_ref() { tx.send(meta).is_ok() } else { false };
        if sent { self.index_queue_len.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        sent
    }
}
