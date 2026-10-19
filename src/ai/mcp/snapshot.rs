//! What the UI shows right now, published for tools that read the application context.

use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use serde::Serialize;

/// Shortest time between two publishes from the frame loop.
const PUBLISH_EVERY: Duration = Duration::from_millis(250);

/// Most selected paths a snapshot carries.
const SELECTION_SAMPLE: usize = 20;

/// The focused view as the UI last reported it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AppSnapshot {
    pub db_connected: bool,
    /// Title of the focused tab.
    pub tab: Option<String>,
    /// Scope string of the focused tab.
    pub scope: Option<String>,
    pub row_count: Option<usize>,
    pub selection_count: usize,
    /// Up to 20 selected paths when the UI provides them.
    pub selection_sample: Vec<String>,
    pub focused_path: Option<String>,
    /// Text of each active filter.
    pub filters: Vec<String>,
    pub collection: Option<String>,
    /// Unix milliseconds of the last publish; 0 before the UI published.
    pub updated_ms: u64,
}

impl AppSnapshot {
    /// The snapshot without its timestamp, for change detection.
    fn content(&self) -> Self {
        Self { updated_ms: 0, ..self.clone() }
    }
}

pub type SnapshotCell = Arc<RwLock<AppSnapshot>>;

static GLOBAL: Lazy<SnapshotCell> = Lazy::new(|| Arc::new(RwLock::new(AppSnapshot::default())));
static LAST_PUBLISH: Mutex<Option<Instant>> = Mutex::new(None);

/// The snapshot the UI publishes into.
pub fn global() -> SnapshotCell {
    GLOBAL.clone()
}

/// Stores `snapshot` in `cell` when its content changed; returns whether it did.
pub fn store(cell: &SnapshotCell, mut snapshot: AppSnapshot, now_ms: u64) -> bool {
    let mut current = cell.write().unwrap_or_else(|e| e.into_inner());
    if current.content() == snapshot.content() && current.updated_ms != 0 {
        return false;
    }
    snapshot.updated_ms = now_ms;
    *current = snapshot;
    true
}

pub fn current(cell: &SnapshotCell) -> AppSnapshot {
    cell.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Reads the focused explorer through the status bar's adapter.
pub fn from_app(app: &mut crate::app::SmartMediaApp) -> AppSnapshot {
    let ex = app.context.active_explorer();
    let tab = Some(ex.tab_title());
    let stats = crate::ui::taskbar::adapter::compute_view_stats(ex);
    let focused = ex.current_thumb.path.clone();
    AppSnapshot {
        db_connected: crate::database::is_connected(),
        tab,
        scope: Some(ex.scope().format()),
        row_count: Some(stats.items),
        selection_count: stats.selected,
        selection_sample: ex.selected_rows().iter().take(SELECTION_SAMPLE).map(|r| r.path.clone()).collect(),
        focused_path: Some(focused).filter(|p| !p.is_empty()),
        filters: stats.filters.into_iter().map(|c| c.label).collect(),
        collection: crate::command::context::active_collection(),
        updated_ms: 0,
    }
}

/// Publishes the application's state from the frame loop, at most a few times a second.
pub fn publish_frame(app: &mut crate::app::SmartMediaApp) {
    let now = Instant::now();
    {
        let mut last = LAST_PUBLISH.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|at| now.saturating_duration_since(at) < PUBLISH_EVERY) {
            return;
        }
        *last = Some(now);
    }
    let snapshot = from_app(app);
    use super::clock::Clock;
    store(&GLOBAL, snapshot, super::clock::SystemClock.unix_ms());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> SnapshotCell {
        Arc::new(RwLock::new(AppSnapshot::default()))
    }

    #[test]
    fn a_new_snapshot_is_stored_with_its_time_and_an_equal_one_is_not() {
        let cell = cell();
        let snap = AppSnapshot { tab: Some("Explorer".into()), selection_count: 3, ..Default::default() };
        assert!(store(&cell, snap.clone(), 1000));
        assert_eq!(current(&cell).updated_ms, 1000);
        assert!(!store(&cell, snap.clone(), 2000), "unchanged content keeps the old time");
        assert_eq!(current(&cell).updated_ms, 1000);
        let changed = AppSnapshot { selection_count: 4, ..snap };
        assert!(store(&cell, changed, 3000));
        assert_eq!((current(&cell).selection_count, current(&cell).updated_ms), (4, 3000));
    }

    #[test]
    fn the_first_publish_always_stores_even_when_the_content_is_the_default() {
        let cell = cell();
        assert!(store(&cell, AppSnapshot::default(), 5));
        assert_eq!(current(&cell).updated_ms, 5);
    }

    #[test]
    fn the_global_cell_is_shared() {
        assert!(Arc::ptr_eq(&global(), &global()));
    }

    #[test]
    fn the_real_app_publishes_its_focused_view() {
        use crate::database::ui_themes::tests::{Fixture, RT};
        let fixture = Fixture::new();
        let _runtime = RT.enter();
        let ctx = crate::ui::test_support::test_ctx();
        let appearance = crate::theme::Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let chrome = crate::theme::chrome::Chrome { frameless: false, available: true };
        let mut app = crate::app::SmartMediaApp::assemble(ctx, appearance, chrome, crate::ui::file_table::FileExplorer::new(true));
        let snapshot = from_app(&mut app);
        assert_eq!(snapshot.tab, Some(app.context.active_explorer().tab_title()));
        assert_eq!((snapshot.row_count, snapshot.selection_count, snapshot.focused_path.clone()), (Some(0), 0, None));
        assert_eq!(snapshot.db_connected, crate::database::is_connected());
        publish_frame(&mut app);
        let published = current(&global());
        assert!(published.updated_ms > 0);
        assert_eq!(published.row_count, Some(0));
    }
}
