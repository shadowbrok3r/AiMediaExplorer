//! The browser content of an explorer: Grid or Details, the grid's selection effects, view mode switches and the saved view prefs.

use egui::{Context, Id, Ui};

use super::details::{DetailsEvent, DetailsInputs};
use super::grid::{BandChange, GridEvent, GridStyle, GridView, Scroll};
use super::grid_layout::{Motion, motion_target, step_between};
use super::keys::KeyAction;
use super::view_mode::{self, TileFit, ViewMode, ViewPrefs, clamp_tile};
use super::view_strip;
use crate::ui::devtools::grid::{self as grid_dev, Command};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::{PathKey, Step, ViewRef};

/// View state of one explorer tab.
pub struct Browse {
    pub mode: ViewMode,
    pub grid: GridView,
    pub tile: f32,
    pub fit: TileFit,
    pub captions: bool,
    pub badges: bool,
    band_base: Vec<PathKey>,
}

impl Default for Browse {
    fn default() -> Self {
        Self::from_prefs(view_mode::app_prefs())
    }
}

impl Browse {
    /// A new tab's view state from the app-wide prefs.
    pub fn from_prefs(prefs: ViewPrefs) -> Self {
        Self { mode: prefs.mode, grid: GridView::new(), tile: prefs.tile, fit: prefs.fit, captions: prefs.captions, badges: prefs.badges, band_base: Vec::new() }
    }

    pub fn style(&self) -> GridStyle {
        GridStyle { tile: self.tile, fit: self.fit, captions: self.captions, badges: self.badges }
    }

    /// The tab's settings as prefs.
    pub fn prefs(&self) -> ViewPrefs {
        ViewPrefs { mode: self.mode, tile: self.tile, fit: self.fit, captions: self.captions, badges: self.badges }
    }

    /// Makes the tab's settings the ones new tabs start from.
    pub fn remember(&self) {
        let prefs = self.prefs();
        view_mode::update_app_prefs(|p| *p = prefs);
    }
}

impl FileExplorer {
    /// Draws the view strip and the rows as a grid or a table.
    pub fn browse_panel(&mut self, ui: &mut Ui) {
        view_strip::show(ui, self);
        match self.browse.mode {
            ViewMode::Grid => self.grid_panel(ui),
            ViewMode::Details => self.details_panel(ui),
        }
        let ctx = ui.ctx();
        if view_mode::app_prefs_dirty() && !ctx.input(|i| i.pointer.any_down()) {
            view_mode::flush_app_prefs();
        }
    }

    /// Draws the grid and applies what the user did in it.
    pub fn grid_panel(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        self.load_details_prefs_once(&ctx);
        self.run_row_commands(&ctx);
        self.run_grid_commands(&ctx);
        self.sync_view();
        let style = self.browse.style();
        let events = {
            let input = DetailsInputs { store: &self.table, view: &self.view, selection: &self.selection, viewer: &self.viewer };
            self.browse.grid.show(ui, &input, &style)
        };
        if !events.is_empty() {
            self.apply_grid_events(&ctx, events);
        }
        self.fire_settled_focus(&ctx);
        self.publish_row_debug(&ctx);
        self.publish_grid_debug(&ctx);
    }

    /// Switches between Grid and Details, keeping the row at the top in sight.
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        if self.browse.mode == mode {
            return;
        }
        let top = match self.browse.mode {
            ViewMode::Grid => Some(self.browse.grid.top_pos()),
            ViewMode::Details => self.details.row_rects().iter().map(|(pos, _)| *pos).min(),
        };
        self.browse.mode = mode;
        if let Some(pos) = top {
            match mode {
                ViewMode::Grid => self.browse.grid.scroll(Scroll::Pos(pos)),
                ViewMode::Details => self.details.scroll_row_to_top(pos),
            }
        }
        self.browse.remember();
    }

    /// Sets the grid's tile size and remembers it for new tabs.
    pub fn set_tile_size(&mut self, tile: f32) {
        let tile = clamp_tile(tile);
        if (self.browse.tile - tile).abs() > f32::EPSILON {
            self.browse.tile = tile;
            self.browse.remember();
        }
    }

    pub fn set_tile_fit(&mut self, fit: TileFit) {
        if self.browse.fit != fit {
            self.browse.fit = fit;
            self.browse.remember();
        }
    }

    pub fn set_captions(&mut self, on: bool) {
        if self.browse.captions != on {
            self.browse.captions = on;
            self.browse.remember();
        }
    }

    pub fn set_badges(&mut self, on: bool) {
        if self.browse.badges != on {
            self.browse.badges = on;
            self.browse.remember();
        }
    }

    pub(crate) fn apply_grid_events(&mut self, ctx: &Context, events: Vec<GridEvent>) {
        for event in events {
            match event {
                GridEvent::Details(event) => {
                    let moves = matches!(event, DetailsEvent::Key(KeyAction::Move { .. }));
                    self.apply_details_events(ctx, vec![event]);
                    if moves {
                        self.reveal_focus();
                    }
                }
                GridEvent::Move { motion, extend } => {
                    self.move_focus(motion, extend);
                }
                GridEvent::Band(change) => self.apply_band(change),
                GridEvent::TileSize(tile) => self.set_tile_size(tile),
            }
        }
        self.settle_selection();
        ctx.request_repaint();
    }

    /// Scrolls the grid just far enough to show the focused tile.
    fn reveal_focus(&mut self) {
        let view = ViewRef::new(&self.table, &self.view);
        if let Some(pos) = self.selection.focus().and_then(|k| view.pos_of(k)) {
            self.browse.grid.scroll(Scroll::Reveal(pos));
        }
    }

    /// Moves the keyboard focus by one grid motion.
    pub fn move_focus(&mut self, motion: Motion, extend: bool) -> Option<PathKey> {
        self.sync_view();
        let view = ViewRef::new(&self.table, &self.view);
        let cur = self.selection.focus().and_then(|k| view.pos_of(k));
        let target = motion_target(cur, view.len(), self.browse.grid.cols(), self.browse.grid.page_rows(), motion)?;
        let step = match cur {
            Some(cur) => step_between(cur, target),
            None if target == 0 => Some(Step::Down(1)),
            None => Some(Step::Up(1)),
        };
        let moved = step.and_then(|step| self.selection.step_focus(&view, step, extend));
        self.settle_selection();
        if moved.is_some() {
            self.reveal_focus();
        }
        moved
    }

    fn grid_owner(&self) -> Id {
        Id::new(("ame_grid_owner", self.viewer.scope))
    }

    /// Runs the commands the Grid devtools window queued for this explorer.
    pub(crate) fn run_grid_commands(&mut self, ctx: &Context) {
        for command in grid_dev::take_commands(ctx, self.grid_owner()) {
            match command {
                Command::SetStress(rate) => self.browse.grid.set_stress(rate),
                Command::SetMode(mode) => self.set_view_mode(mode),
                Command::SetTile(tile) => self.set_tile_size(tile),
                Command::ScrollTop => self.browse.grid.scroll(Scroll::Top),
                Command::ScrollEnd => self.browse.grid.scroll(Scroll::Offset(f32::MAX)),
                Command::ResetFrameStats => self.browse.grid.clear_frame_times(),
                Command::ResetFacts => self.browse.grid.facts.reset(),
            }
        }
    }

    /// Publishes the grid's numbers for the Devtools window when it is open.
    pub(crate) fn publish_grid_debug(&self, ctx: &Context) {
        if !grid_dev::is_open(ctx) {
            return;
        }
        let grid = &self.browse.grid;
        let layout = grid.layout();
        let facts = &grid.facts;
        grid_dev::publish(
            ctx,
            grid_dev::Snapshot {
                owner: self.grid_owner(),
                label: self.current_path.clone(),
                mode: self.browse.mode,
                total: self.view.len(),
                selected: self.selection.len(),
                cols: grid.cols(),
                tile_target: self.browse.tile,
                tile_px: layout.map_or(0.0, |l| l.tile),
                row_h: layout.map_or(0.0, |l| l.row_h()),
                caption_h: layout.map_or(0.0, |l| l.caption_h),
                fit: self.browse.fit,
                offset: grid.offset(),
                viewport_h: grid.viewport_height(),
                content_h: grid.content_height(),
                visible: grid.visible(),
                ahead: grid.ahead(),
                stats: grid.stats(),
                frame_ms: grid.frame_times().iter().copied().collect(),
                stress: grid.stress().map(|s| s.rows_per_s),
                facts_queued: facts.queued(),
                facts_inflight: facts.inflight(),
                facts_known: facts.known_len(),
            },
        );
    }

    // Selects the base selection plus the tiles in the band's rows and columns.
    fn apply_band(&mut self, change: BandChange) {
        if change.begin {
            self.browse.band_base = if change.additive { self.selection.keys().to_vec() } else { Vec::new() };
        }
        let view = ViewRef::new(&self.table, &self.view);
        let mut keys = self.browse.band_base.clone();
        for row in change.rows.clone() {
            for col in change.cols.clone() {
                if let Some(key) = row.checked_mul(change.grid_cols).and_then(|p| p.checked_add(col)).and_then(|pos| view.key_at(pos)) {
                    keys.push(key.clone());
                }
            }
        }
        self.selection.select_keys(&view, keys);
        self.settle_selection();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_tab_starts_from_the_app_prefs() {
        view_mode::update_app_prefs(|p| {
            p.mode = ViewMode::Details;
            p.tile = 200.0;
            p.fit = TileFit::Fit;
            p.captions = false;
        });
        let b = Browse::default();
        assert_eq!((b.mode, b.tile, b.fit, b.captions), (ViewMode::Details, 200.0, TileFit::Fit, false));
    }

    #[test]
    fn the_defaults_open_the_grid() {
        let b = Browse::from_prefs(ViewPrefs::default());
        assert_eq!((b.mode, b.tile, b.captions, b.badges), (ViewMode::Grid, 160.0, true, true));
        assert_eq!(b.style().tile, 160.0);
        assert_eq!(b.prefs(), ViewPrefs::default());
    }
}
