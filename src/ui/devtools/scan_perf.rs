//! Scan batch timings of the focused tab, shown inside the Perf window.

use eframe::egui::{Context, Id, Ui};

use crate::ui::file_table::scan_perf::ScanPerf;

fn data_id() -> Id {
    Id::new("ame_devtools_scan_perf")
}

fn clear_id() -> Id {
    Id::new("ame_devtools_scan_perf_clear")
}

fn per_1k_id() -> Id {
    Id::new("ame_devtools_scan_perf_per_1k")
}

/// Records the timings the section shows.
pub fn publish(ctx: &Context, perf: ScanPerf) {
    ctx.data_mut(|d| d.insert_temp(data_id(), perf));
}

/// True once after the clear button was pressed.
pub fn take_clear(ctx: &Context) -> bool {
    ctx.data_mut(|d| d.remove_temp::<bool>(clear_id())).unwrap_or(false)
}

/// Lines describing `perf`, one per recent batch; with `per_1k` each carries the rate per 1000 rows.
pub fn batch_lines(perf: &ScanPerf, per_1k: bool) -> Vec<String> {
    perf.recent
        .iter()
        .map(|(len, recv, ui)| {
            let base = format!("len={len} recv={:.3}s ui={:.3}s", recv.as_secs_f32(), ui.as_secs_f32());
            if per_1k && *len > 0 {
                let scale = 1000.0 / *len as f32;
                format!("{base} | recv/1k={:.3}s ui/1k={:.3}s", recv.as_secs_f32() * scale, ui.as_secs_f32() * scale)
            } else {
                base
            }
        })
        .collect()
}

/// Draws the scan timings of the focused tab.
pub fn section(ctx: &Context, ui: &mut Ui) {
    let perf = ctx.data(|d| d.get_temp::<ScanPerf>(data_id())).unwrap_or_default();
    ui.separator();
    ui.strong("Scan batches (focused tab)");
    if perf.last_total.is_none() && perf.recent.is_empty() {
        ui.weak("No scan timings recorded");
        return;
    }
    if let Some(total) = perf.last_total {
        ui.label(format!("Last scan total: {:.3}s over {} batches", total.as_secs_f32(), perf.batch_count));
    }
    let mut per_1k = ctx.data(|d| d.get_temp::<bool>(per_1k_id())).unwrap_or(false);
    if ui.checkbox(&mut per_1k, "Show per-1k rates").changed() {
        ctx.data_mut(|d| d.insert_temp(per_1k_id(), per_1k));
    }
    for line in batch_lines(&perf, per_1k) {
        ui.monospace(line);
    }
    if let Some((recv, ui_time)) = perf.per_1k() {
        ui.label(format!("Average over the last {}: recv/1k={recv:.3}s ui/1k={ui_time:.3}s", perf.recent.len()));
    }
    if ui.button("Clear timings").clicked() {
        ctx.data_mut(|d| d.insert_temp(clear_id(), true));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn lines_add_the_per_1k_rates_on_request() {
        let perf = ScanPerf { last_total: None, recent: vec![(500, Duration::from_millis(100), Duration::from_millis(50))], batch_count: 1 };
        assert_eq!(batch_lines(&perf, false), vec!["len=500 recv=0.100s ui=0.050s".to_string()]);
        assert_eq!(batch_lines(&perf, true), vec!["len=500 recv=0.100s ui=0.050s | recv/1k=0.200s ui/1k=0.100s".to_string()]);
    }
}
