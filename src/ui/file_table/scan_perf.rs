//! Scan batch timings of an explorer, read by the Perf devtools window.

use std::time::Duration;

/// Batches kept in a snapshot.
pub const SNAPSHOT_BATCHES: usize = 10;

/// Timings of the last scan: rows per batch, wait for the batch and time to process it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanPerf {
    pub last_total: Option<Duration>,
    pub recent: Vec<(usize, Duration, Duration)>,
    pub batch_count: usize,
}

impl ScanPerf {
    /// Mean receive and UI seconds per 1000 rows over `recent`; `None` without rows.
    pub fn per_1k(&self) -> Option<(f32, f32)> {
        let rows: usize = self.recent.iter().map(|(len, _, _)| *len).sum();
        if rows == 0 {
            return None;
        }
        let scale = 1000.0 / rows as f32;
        let recv: f32 = self.recent.iter().map(|(_, r, _)| r.as_secs_f32()).sum();
        let ui: f32 = self.recent.iter().map(|(_, _, u)| u.as_secs_f32()).sum();
        Some((recv * scale, ui * scale))
    }
}

impl super::FileExplorer {
    /// The latest scan timings of this explorer.
    pub fn scan_perf(&self) -> ScanPerf {
        let from = self.perf_batches.len().saturating_sub(SNAPSHOT_BATCHES);
        ScanPerf { last_total: self.perf_last_total, recent: self.perf_batches[from..].to_vec(), batch_count: self.perf_batches.len() }
    }

    pub fn clear_scan_perf(&mut self) {
        self.perf_batches.clear();
        self.perf_last_total = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_1k_scales_by_rows() {
        let batch = (500, Duration::from_millis(100), Duration::from_millis(50));
        let perf = ScanPerf { last_total: None, recent: vec![batch, batch], batch_count: 2 };
        let (recv, ui) = perf.per_1k().unwrap();
        assert!((recv - 0.2).abs() < 1e-4 && (ui - 0.1).abs() < 1e-4, "{recv} {ui}");
        assert_eq!(ScanPerf::default().per_1k(), None);
    }

    #[tokio::test]
    async fn the_snapshot_keeps_the_newest_batches_and_clearing_empties_it() {
        let mut ex = crate::ui::file_table::FileExplorer::new(true);
        for i in 0..25 {
            ex.perf_batches.push((i + 1, Duration::from_millis(i as u64), Duration::from_millis(1)));
        }
        ex.perf_last_total = Some(Duration::from_secs(2));
        let perf = ex.scan_perf();
        assert_eq!((perf.recent.len(), perf.batch_count), (SNAPSHOT_BATCHES, 25));
        assert_eq!(perf.recent.last().map(|b| b.0), Some(25));
        assert_eq!(perf.last_total, Some(Duration::from_secs(2)));
        ex.clear_scan_perf();
        assert_eq!(ex.scan_perf(), ScanPerf::default());
    }
}
