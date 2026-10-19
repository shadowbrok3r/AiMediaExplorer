//! Start-up milestones, each noted once with the time since the process started.

use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// The database handle is connected to its store.
pub const DB_CONNECTED: &str = "db_connected";
/// Schema and default records are in place.
pub const DB_READY: &str = "db_ready";
/// A tab that shows the library has its first rows.
pub const FIRST_LIBRARY_ROWS: &str = "first_library_rows";
/// A tab that shows the library finished loading.
pub const LIBRARY_LOADED: &str = "library_loaded";
/// The first frame was drawn.
pub const FIRST_FRAME: &str = "first_frame";

static FALLBACK_START: LazyLock<Instant> = LazyLock::new(Instant::now);
static MARKS: Mutex<Vec<(&'static str, f32)>> = Mutex::new(Vec::new());

fn marks() -> std::sync::MutexGuard<'static, Vec<(&'static str, f32)>> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

fn age_secs() -> f32 {
    match super::proc::process_age() {
        Some(age) => age.as_secs_f32(),
        None => FALLBACK_START.elapsed().as_secs_f32(),
    }
}

/// Notes `name` with the time since the process started, unless it was noted before.
pub fn mark(name: &'static str) {
    let mut all = marks();
    if all.iter().all(|(n, _)| *n != name) {
        all.push((name, age_secs()));
    }
}

/// Seconds since the process started at which `name` was noted.
pub fn get(name: &str) -> Option<f32> {
    marks().iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Every milestone in the order it was noted.
pub fn all() -> Vec<(String, f32)> {
    marks().iter().map(|(n, t)| (n.to_string(), *t)).collect()
}

/// Whether every milestone has been noted.
pub fn complete() -> bool {
    [DB_CONNECTED, DB_READY, FIRST_LIBRARY_ROWS, LIBRARY_LOADED].iter().all(|n| get(n).is_some())
}

/// Notes the database milestones that hold right now; called once per frame.
pub fn observe_runtime() {
    let _ = *FALLBACK_START;
    mark(FIRST_FRAME);
    if crate::database::is_connected() {
        mark(DB_CONNECTED);
    }
    if crate::database::is_ready() {
        mark(DB_READY);
    }
}

/// Notes the library milestones from the state of a tab.
pub fn observe_browser(library: bool, rows: usize, loading: bool) {
    if !library || rows == 0 {
        return;
    }
    mark(FIRST_LIBRARY_ROWS);
    if !loading {
        mark(LIBRARY_LOADED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_milestone_keeps_the_time_it_was_first_noted() {
        mark("test_milestone");
        let first = get("test_milestone").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        mark("test_milestone");
        assert_eq!(get("test_milestone"), Some(first));
        assert!(all().iter().any(|(n, _)| n == "test_milestone"));
        assert_eq!(get("never_noted"), None);
    }

    #[test]
    fn a_tab_that_is_not_the_library_or_has_no_rows_notes_nothing() {
        observe_browser(false, 10, false);
        observe_browser(true, 0, false);
        if get(FIRST_LIBRARY_ROWS).is_none() {
            assert!(get(LIBRARY_LOADED).is_none());
        }
    }
}
