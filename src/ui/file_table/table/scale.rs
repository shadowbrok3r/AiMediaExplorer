//! Connects a scripted performance run of the Scale window to a browser tab.

use std::path::PathBuf;
use std::time::Instant;

use egui::Context;

use super::columns::Col;
use super::grid::Scroll;
use super::view_mode::ViewMode;
use crate::command::Scope;
use crate::ui::devtools::scale::bench::{self, Bench, LoadSummary, Repaint, Resolved, SortBy, Target, Tick};
use crate::ui::devtools::scale::resolve::TopNames;
use crate::ui::devtools::scale::{self, Facts};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::{SortSpec, ViewRef};

/// Time between checks while a run waits for the database to open.
const WAIT_FOR_DATABASE: std::time::Duration = std::time::Duration::from_millis(100);

/// A run in progress on a tab.
pub struct Run {
    bench: Bench,
    top: TopNames,
    report: Option<PathBuf>,
}

fn column(by: SortBy) -> Col {
    match by {
        SortBy::Name => Col::Name,
        SortBy::Size => Col::Size,
        SortBy::Modified => Col::Modified,
    }
}

/// The explorer as the target of a run.
struct Driver<'a> {
    ex: &'a mut FileExplorer,
    ctx: &'a Context,
    top: &'a TopNames,
}

impl Target for Driver<'_> {
    fn view_rows(&self) -> usize {
        self.ex.view.len()
    }

    fn selected(&self) -> usize {
        self.ex.selection.len()
    }

    fn is_loading(&self) -> bool {
        self.ex.is_loading()
    }

    fn open(&mut self, scope: &str) -> Result<(), String> {
        let scope = Scope::parse(scope).map_err(|e| e.to_string())?;
        self.ex.open_scope(scope)
    }

    fn last_load(&self) -> Option<LoadSummary> {
        let r = self.ex.nav.log.back()?;
        Some(LoadSummary {
            scope: r.scope.clone(),
            rows: r.rows,
            batches: r.batches,
            first_batch_ms: r.first_batch.map(|d| d.as_secs_f32() * 1000.0),
            elapsed_ms: r.elapsed.as_secs_f32() * 1000.0,
            page_ms: r.stages.page.as_secs_f32() * 1000.0,
            embed_ms: r.stages.embed.as_secs_f32() * 1000.0,
            total: r.total,
            error: r.error.clone(),
            cancelled: r.cancelled,
        })
    }

    fn scroll_to(&mut self, fraction: f32) {
        let ex = &mut *self.ex;
        match ex.browse.mode {
            ViewMode::Grid => {
                let max = (ex.browse.grid.content_height() - ex.browse.grid.viewport_height()).max(0.0);
                ex.browse.grid.scroll(Scroll::Offset(max * fraction));
            }
            ViewMode::Details => ex.details.scroll_row_to_top(((ex.view.len().saturating_sub(1)) as f32 * fraction) as usize),
        }
    }

    fn set_tile(&mut self, px: f32) {
        self.ex.set_tile_size(px);
    }

    fn set_mode(&mut self, mode: ViewMode) {
        self.ex.set_view_mode(mode);
    }

    fn sort(&mut self, by: Option<(SortBy, bool)>) {
        self.ex.view.set_sort(match by {
            Some((by, descending)) => SortSpec::single(column(by).id(), descending),
            None => SortSpec::new(),
        });
    }

    fn filter(&mut self, patch: &serde_json::Value) -> Result<(), String> {
        self.ex.apply_filter_patch(patch, false).map(|_| ())
    }

    fn clear_filters(&mut self) {
        self.ex.clear_filters();
    }

    fn select_all(&mut self) {
        let ex = &mut *self.ex;
        ex.sync_view();
        let view = ViewRef::new(&ex.table, &ex.view);
        ex.selection.select_all(&view);
        ex.settle_selection();
    }

    fn clear_selection(&mut self) {
        self.ex.clear_selection();
    }

    fn task_center(&mut self, open: bool) {
        crate::ui::taskbar::set_center_open(self.ctx, open);
    }

    fn resolve(&mut self, scope: &str) -> Resolved {
        self.top.resolve(scope)
    }
}

impl FileExplorer {
    /// Starts a scripted run on this tab; the report goes to `report` as JSON when given.
    pub fn start_bench(&mut self, suite: bench::Suite, seconds: Option<f32>, scopes: &[String], report: Option<PathBuf>) -> Result<String, String> {
        if self.browse.bench.is_some() {
            return Err("a scale run is already in progress".to_string());
        }
        for scope in scopes {
            if !scope.contains("@top") {
                Scope::parse(scope).map_err(|e| format!("{scope}: {e}"))?;
            }
        }
        let steps = bench::plan(suite, scopes, seconds);
        let count = steps.len();
        self.browse.bench = Some(Run { bench: Bench::new(suite, steps, Instant::now()), top: TopNames::start(), report });
        scale::set_progress(Some(bench::Progress { step: 0, steps: count, label: String::new(), elapsed: 0.0, quiet: false }));
        Ok(format!("{} run started with {count} steps", suite.label()))
    }

    /// True while a scripted run drives this tab.
    pub fn bench_running(&self) -> bool {
        self.browse.bench.is_some()
    }

    /// Advances the scripted run of this tab by one frame.
    pub(crate) fn run_bench(&mut self, ctx: &Context) {
        let Some(mut run) = self.browse.bench.take() else { return };
        if scale::take_cancel() {
            scale::set_progress(None);
            log::info!("[scale] run cancelled");
            return;
        }
        if !crate::database::is_ready() {
            ctx.request_repaint_after(WAIT_FOR_DATABASE);
            self.browse.bench = Some(run);
            return;
        }
        let now = Instant::now();
        let tick = run.bench.tick(&mut Driver { ex: self, ctx, top: &run.top }, now);
        match tick {
            Tick::Running(repaint) => {
                match repaint {
                    Repaint::Now => ctx.request_repaint(),
                    Repaint::After(wait) => ctx.request_repaint_after(wait),
                    Repaint::Never => {}
                }
                scale::set_progress(Some(run.bench.progress(now)));
                self.browse.bench = Some(run);
            }
            Tick::Done(report) => {
                scale::set_progress(None);
                log::info!("[scale] {}", report.to_text());
                if let Some(path) = &run.report {
                    match std::fs::write(path, report.to_json()) {
                        Ok(()) => log::info!("[scale] report written to {}", path.display()),
                        Err(e) => log::warn!("[scale] cannot write {}: {e}", path.display()),
                    }
                }
                scale::set_report(*report);
                ctx.request_repaint();
            }
        }
    }

    /// Notes the start-up milestones and, while the Scale window is open, publishes this tab's numbers.
    pub(crate) fn observe_scale(&self, ctx: &Context) {
        scale::marks::observe_browser(matches!(self.nav.scope, Scope::Library), self.view.len(), self.is_loading());
        if !scale::is_open(ctx) {
            return;
        }
        scale::publish(
            ctx,
            Facts {
                label: self.current_path.clone(),
                scope: self.nav.scope.format(),
                library: matches!(self.nav.scope, Scope::Library),
                details: self.browse.mode == ViewMode::Details,
                store_len: self.table.len(),
                view_len: self.view.len(),
                selected: self.selection.len(),
                stats: *self.view.stats(),
                store_revision: self.table.revision(),
                view_revision: self.view.revision(),
                refreshes: self.view.counts(),
                loading: self.is_loading(),
                row_bytes: self.row_bytes(ctx),
            },
        );
    }

    // The memory of the rows, recomputed only when the store changed.
    fn row_bytes(&self, ctx: &Context) -> usize {
        let id = egui::Id::new(("ame_scale_row_bytes", self.viewer.scope));
        let revision = self.table.revision();
        if let Some(bytes) = ctx.data(|d| d.get_temp::<(u64, usize)>(id)).filter(|(seen, _)| *seen == revision).map(|(_, bytes)| bytes) {
            return bytes;
        }
        let bytes = self.table.estimate_bytes(crate::ui::file_table::rows::thumbnail::heap_bytes).total();
        ctx.data_mut(|d| d.insert_temp(id, (revision, bytes)));
        bytes
    }
}

#[cfg(test)]
#[path = "scale_tests.rs"]
mod tests;
