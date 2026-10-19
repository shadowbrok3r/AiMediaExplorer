//! UI-neutral engine status registry shared by the backends, the task registry and the status widgets.

use once_cell::sync::Lazy;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize};

/// High-level lifecycle state for a backend component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusState {
    Idle,
    Initializing,
    Running,
    Waiting,
    Error,
}

/// Shared metadata describing a component status.
#[derive(Debug, Clone)]
pub struct StatusMeta {
    pub name: &'static str,
    pub model: Option<&'static str>,
    pub detail: String,
    pub state: StatusState,
    // Device where the model is running.
    pub device: Option<DeviceKind>,
    pub started_at_ms: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub extra: Vec<(String, String)>,
    pub error: Option<String>,
}

impl Default for StatusMeta {
    fn default() -> Self {
        Self {
            name: "",
            model: None,
            detail: String::new(),
            state: StatusState::Idle,
            device: None,
            started_at_ms: 0,
            progress_current: 0,
            progress_total: 0,
            extra: vec![],
            error: None,
        }
    }
}

/// Trait for anything that can render itself as a compact indicator + optional hover card.
pub trait GlobalStatusIndicator {
    /// Name / short key.
    fn key(&self) -> &'static str;
    /// Retrieve current meta snapshot.
    fn snapshot(&self) -> StatusMeta;
    /// Update convenience helpers.
    fn set_state(&self, state: StatusState, detail: impl Into<String>);
    fn set_progress(&self, current: u64, total: u64);
    /// Set or update the model string shown in the hover card.
    /// Note: stored as a leaked &'static str to match StatusMeta's lifetime.
    fn set_model(&self, model: &str);
    /// Set device where this model is running.
    fn set_device(&self, device: DeviceKind);
    /// Set an error message and mark state as Error (displayed in hover card).
    fn set_error(&self, err: impl Into<String>);
    /// Clear any existing error message (does not change state).
    fn clear_error(&self);
    /// Update just the detail string without changing the state.
    fn set_detail(&self, detail: impl Into<String>);
}

// Internal representation stored globally.
#[derive(Debug)]
struct GlobalStatusInner {
    meta: StatusMeta,
}

static STATUSES: Lazy<RwLock<std::collections::HashMap<&'static str, GlobalStatusInner>>> =
    Lazy::new(|| RwLock::new(Default::default()));

fn now_ms() -> u64 {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()) as u64
}

/// Handle for a registered global status.
#[derive(Clone)]
pub struct RegisteredStatus {
    key: &'static str,
}

impl RegisteredStatus {
    pub fn register(name: &'static str, model: Option<&'static str>) -> Self {
        let mut w = STATUSES.write().unwrap();
        w.entry(name).or_insert_with(|| GlobalStatusInner {
            meta: StatusMeta {
                name,
                model,
                device: None,
                ..Default::default()
            },
        });
        Self { key: name }
    }
}

impl GlobalStatusIndicator for RegisteredStatus {
    fn key(&self) -> &'static str {
        self.key
    }
    fn snapshot(&self) -> StatusMeta {
        STATUSES
            .read()
            .unwrap()
            .get(self.key)
            .map(|i| i.meta.clone())
            .unwrap_or_default()
    }
    fn set_state(&self, state: StatusState, detail: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.state = state;
            inner.meta.detail = detail.into();
            if matches!(state, StatusState::Running | StatusState::Initializing) {
                inner.meta.started_at_ms = now_ms();
            }
        }
    }
    fn set_progress(&self, current: u64, total: u64) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.progress_current = current;
            inner.meta.progress_total = total;
        }
    }
    fn set_model(&self, model: &str) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            // Leak to obtain 'static lifetime acceptable for process lifetime.
            let leaked: &'static str = Box::leak(model.to_string().into_boxed_str());
            inner.meta.model = Some(leaked);
        }
    }
    fn set_device(&self, device: DeviceKind) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.device = Some(device);
        }
    }
    fn set_error(&self, err: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.error = Some(err.into());
            inner.meta.state = StatusState::Error;
        }
    }
    fn clear_error(&self) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.error = None;
        }
    }
    fn set_detail(&self, detail: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.detail = detail.into();
        }
    }
}

/// Access snapshot for arbitrary key (used by UI aggregation).
pub fn snapshot(key: &'static str) -> Option<StatusMeta> {
    STATUSES.read().ok()?.get(key).map(|i| i.meta.clone())
}
pub fn all_snapshots() -> Vec<StatusMeta> {
    STATUSES
        .read()
        .map(|m| m.values().map(|i| i.meta.clone()).collect())
        .unwrap_or_default()
}

/// Coarse device classification for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind { CPU, GPU }

pub static DB_STATUS: Lazy<RegisteredStatus> = Lazy::new(|| RegisteredStatus::register("DB", None));
pub static VISION_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("VISION", None));
pub static CLIP_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("CLIP", None));
pub static RERANK_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("RERANK", None));
pub static QWEN_EDIT_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("QWEN-EDIT", None));
pub static ASSIST_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("ASSIST", None));

/// Token-progress tracker for generation (vision description) from anywhere.
pub static VISION_TOKENS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
pub static VISION_MAX_TOKENS: Lazy<AtomicUsize> =
    Lazy::new(|| AtomicUsize::new(crate::app::MAX_NEW_TOKENS));
pub static LAST_ERROR_TS_MS: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Indicator order: the core engines first, then any others by name.
pub const INDICATOR_ORDER: [&str; 7] = ["DB", "VISION", "CLIP", "RERANK", "QWEN-EDIT", "ASSIST", "ZEROCLAW"];

/// Sort key placing the core engines in [`INDICATOR_ORDER`] and every other name after them.
pub fn order_key(name: &str) -> (usize, String) {
    let rank = INDICATOR_ORDER.iter().position(|n| *n == name).unwrap_or(INDICATOR_ORDER.len());
    (rank, name.to_string())
}

/// Snapshots of every registered engine in indicator order.
pub fn engine_snapshots() -> Vec<StatusMeta> {
    let mut all = all_snapshots();
    all.sort_by_key(|m| order_key(m.name));
    all
}

/// Fraction done and whether the engine reports a total at all.
pub fn progress_of(meta: &StatusMeta) -> (f32, bool) {
    if meta.progress_total > 0 {
        ((meta.progress_current as f32 / meta.progress_total as f32).clamp(0.0, 1.0), true)
    } else {
        (0.0, false)
    }
}

/// True while the engine is loading or working.
pub fn is_active(meta: &StatusMeta) -> bool {
    matches!(meta.state, StatusState::Initializing | StatusState::Running)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicators_keep_a_fixed_engine_order() {
        let mut names = vec!["ASSIST", "ZZZ", "DB", "CLIP", "AAA", "VISION"];
        names.sort_by_key(|n| order_key(n));
        assert_eq!(names, vec!["DB", "VISION", "CLIP", "ASSIST", "AAA", "ZZZ"]);
    }

    #[test]
    fn progress_needs_a_total_and_clamps_to_one() {
        let mut meta = StatusMeta::default();
        assert_eq!(progress_of(&meta), (0.0, false));
        meta.progress_total = 10;
        meta.progress_current = 5;
        assert_eq!(progress_of(&meta), (0.5, true));
        meta.progress_current = 50;
        assert_eq!(progress_of(&meta), (1.0, true));
    }

    #[test]
    fn only_initializing_and_running_engines_are_active() {
        let state = |state| is_active(&StatusMeta { state, ..Default::default() });
        assert!(state(StatusState::Initializing));
        assert!(state(StatusState::Running));
        assert!(!state(StatusState::Idle));
        assert!(!state(StatusState::Waiting));
        assert!(!state(StatusState::Error));
    }
}
