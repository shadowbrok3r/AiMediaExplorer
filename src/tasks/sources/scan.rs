//! Scans: one task per scan that still has a delivery route.

use super::{ExplorerProbe, Report, SourceCtx, TaskSource};
use crate::tasks::model::{CancelHandle, Observation, TaskKind, TaskSpec};
use crate::utilities::scan::ActiveScan;

pub const SOURCE: &str = "scan";
/// Age below which a scan shows no chip.
const VISIBLE_AFTER: std::time::Duration = std::time::Duration::from_millis(400);
/// Shortest scan run whose end toasts.
const TOAST_AFTER: std::time::Duration = std::time::Duration::from_secs(3);

type ActiveScans = Box<dyn FnMut() -> Vec<ActiveScan> + Send>;
type Canceller = std::sync::Arc<dyn Fn(u64) + Send + Sync>;

pub struct ScanSource {
    active: ActiveScans,
    cancel: Canceller,
}

impl ScanSource {
    pub fn new(active: ActiveScans, cancel: Canceller) -> Self {
        Self { active, cancel }
    }

    /// Reads the scan engine and stops scans through `cancel_scan`.
    pub fn live() -> Self {
        Self::new(Box::new(crate::utilities::scan::active_scans), std::sync::Arc::new(crate::utilities::scan::cancel_scan))
    }
}

/// The explorer showing scan `id`; a lone scan is matched with a lone scanning explorer that does not expose its id.
fn explorer_of<'a>(id: u64, scans: usize, explorers: &'a [ExplorerProbe]) -> Option<&'a ExplorerProbe> {
    if let Some(found) = explorers.iter().find(|e| e.scan_id == Some(id)) {
        return Some(found);
    }
    let mut anonymous = explorers.iter().filter(|e| e.scan_id.is_none() && e.is_scanning());
    match (scans, anonymous.next(), anonymous.next()) {
        (1, Some(only), None) => Some(only),
        _ => None,
    }
}

impl TaskSource for ScanSource {
    fn name(&self) -> &'static str {
        SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        let scans = (self.active)();
        let count = scans.len();
        for scan in scans {
            let explorer = explorer_of(scan.id, count, &cx.app.explorers);
            let title = match explorer {
                Some(e) if !e.tab.is_empty() => format!("Scan: {}", e.tab),
                _ => format!("Scan #{}", scan.id),
            };
            let detail = match (scan.cancelled, scan.walk_done) {
                (true, _) => "Cancelling".to_string(),
                (false, true) => format!("{} thumbnails left", scan.queued_thumbs),
                (false, false) => "Walking folders".to_string(),
            };
            let (done, total) = match explorer {
                Some(e) if !scan.walk_done && e.scan_progress > 0.0 => ((e.scan_progress.clamp(0.0, 1.0) * 1000.0).round() as u64, 1000),
                _ => (0, 0),
            };
            let id = scan.id;
            let cancel = self.cancel.clone();
            let spec = TaskSpec::new(TaskKind::Scan, title)
                .total(total)
                .detail(detail)
                .cancel(CancelHandle::new(move || cancel(id)))
                .user_initiated()
                .visible_after(VISIBLE_AFTER)
                .toast_after(TOAST_AFTER);
            out.observations.push(Observation::new(format!("scan:{id}"), spec).done(done));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::sources::AppProbe;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    fn scan(id: u64, queued: usize, walk_done: bool) -> ActiveScan {
        ActiveScan { id, queued_thumbs: queued, walk_done, cancelled: false }
    }

    fn explorer(tab: &str, id: Option<u64>, progress: f32) -> ExplorerProbe {
        ExplorerProbe { tab: tab.into(), scan_id: id, scan_progress: progress, ..Default::default() }
    }

    fn poll_with(scans: Vec<ActiveScan>, explorers: Vec<ExplorerProbe>) -> (Vec<Observation>, Arc<Mutex<Vec<u64>>>, ScanSource) {
        let cancelled = Arc::new(Mutex::new(Vec::new()));
        let sink = cancelled.clone();
        let mut source = ScanSource::new(Box::new(move || scans.clone()), Arc::new(move |id| sink.lock().unwrap().push(id)));
        let app = AppProbe { explorers, ..Default::default() };
        let mut out = Report::default();
        source.poll(&SourceCtx { now: Instant::now(), app: &app }, &mut out);
        (out.observations, cancelled, source)
    }

    #[test]
    fn an_explorer_with_the_scan_id_supplies_title_and_progress() {
        let (obs, _, _) = poll_with(vec![scan(3, 0, false), scan(4, 0, false)], vec![explorer("Photos", Some(3), 0.42), explorer("Other", Some(4), 0.1)]);
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].key, "scan:3");
        assert_eq!(obs[0].spec.title, "Scan: Photos");
        assert_eq!((obs[0].done, obs[0].spec.total), (420, 1000));
        assert_eq!(obs[1].spec.title, "Scan: Other");
    }

    #[test]
    fn a_lone_scan_is_matched_with_a_lone_scanning_explorer() {
        let (obs, _, _) = poll_with(vec![scan(9, 0, false)], vec![explorer("Idle", None, 0.0), explorer("Busy", None, 0.5)]);
        assert_eq!(obs[0].spec.title, "Scan: Busy");
        assert_eq!(obs[0].done, 500);
    }

    #[test]
    fn two_unidentified_candidates_stay_unmatched() {
        let (obs, _, _) = poll_with(vec![scan(9, 0, false)], vec![explorer("A", None, 0.5), explorer("B", None, 0.5)]);
        assert_eq!(obs[0].spec.title, "Scan #9");
        assert_eq!(obs[0].spec.total, 0, "no progress without a match");
    }

    #[test]
    fn after_the_walk_the_remaining_thumbnails_are_the_detail() {
        let (obs, _, _) = poll_with(vec![scan(1, 120, true)], vec![]);
        assert_eq!(obs[0].spec.detail, "120 thumbnails left");
        let (obs, _, _) = poll_with(vec![scan(1, 0, false)], vec![]);
        assert_eq!(obs[0].spec.detail, "Walking folders");
    }

    #[test]
    fn a_cancelled_scan_says_so() {
        let (obs, _, _) = poll_with(vec![ActiveScan { cancelled: true, ..scan(5, 3, false) }], vec![]);
        assert_eq!(obs[0].spec.detail, "Cancelling");
    }

    #[test]
    fn the_cancel_handle_stops_exactly_that_scan() {
        let (obs, cancelled, _) = poll_with(vec![scan(3, 0, false), scan(4, 0, false)], vec![]);
        obs[1].spec.cancel.as_ref().expect("scans can be cancelled").call();
        assert_eq!(*cancelled.lock().unwrap(), vec![4]);
    }

    #[test]
    fn scans_are_user_tasks_that_hide_blinks_and_toast_only_when_long() {
        let (obs, _, _) = poll_with(vec![scan(1, 0, false)], vec![]);
        assert!(obs[0].spec.user_initiated);
        assert_eq!(obs[0].spec.kind, TaskKind::Scan);
        assert_eq!(obs[0].spec.visible_after, VISIBLE_AFTER);
        assert_eq!(obs[0].spec.toast_after, TOAST_AFTER);
    }

    #[test]
    fn no_scans_means_no_observations() {
        let (obs, _, _) = poll_with(vec![], vec![explorer("Photos", Some(3), 0.9)]);
        assert!(obs.is_empty());
    }
}
