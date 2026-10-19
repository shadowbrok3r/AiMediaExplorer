//! Per-tab filter state and the explorer's filter API: the spec, the lookups it reads, saved views and the Filters window hooks.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use egui::Context;
use serde_json::Value;

use super::FileExplorer;
use super::rows::thumbnail::{DIR_TYPE, kind_of_path};
use super::rows::{DEFAULT_MIN_SIZE, FactSource, FilterSpec, HiddenReport, PreparedFilter, Probe, ProbeCache, ProbeJob, RowKind, TypeSet};
use super::table::{ExplorerMode, FileTableViewer, cells};
use crate::Thumbnail;
use crate::database::FilterGroup;
use crate::ui::devtools::filters::{self as devtools, Command, ExplainReport, Snapshot};

/// Shortest time between counts of what each condition hides while rows keep arriving.
const REPORT_EVERY: std::time::Duration = std::time::Duration::from_millis(400);

/// Which paths the library holds.
#[derive(Debug, Default)]
pub enum IndexedSet {
    /// Every row of the tab comes from the library.
    #[default]
    Everything,
    Paths(HashSet<String>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ScanKey {
    layout: u64,
    edits: u64,
    offline: bool,
    dims: bool,
    limit: u64,
}

/// The filter of one tab and everything its conditions read besides the rows.
pub struct FilterState {
    /// Conditions of the tab; `query` and the three type toggles live on the viewer.
    pub spec: FilterSpec,
    base: FilterSpec,
    pub indexed: IndexedSet,
    pub indexed_rev: u64,
    indexed_len: usize,
    pub probes: ProbeCache,
    scan: Option<(ScanKey, usize)>,
    /// Times the filter signature changed.
    pub changes: u64,
    pub changed_at: Option<Instant>,
    last_sig: u64,
    report: Option<((u64, u64, u64), HiddenReport)>,
    report_at: Option<Instant>,
    explain: Option<ExplainReport>,
    last_error: Option<String>,
}

/// The settings-backed conditions: those of the settings once they are loaded, none before.
fn base_for(settings: &crate::UiSettings) -> FilterSpec {
    match crate::settings::store().load_state() {
        crate::settings::LoadState::Pending => FilterSpec::none(),
        _ => FilterSpec::from_settings(settings),
    }
}

impl FilterState {
    pub fn new(settings: &crate::UiSettings) -> Self {
        Self::with_base(base_for(settings))
    }

    /// State whose conditions start as `base`.
    pub fn with_base(base: FilterSpec) -> Self {
        Self {
            spec: base.clone(),
            base,
            indexed: IndexedSet::default(),
            indexed_rev: 0,
            indexed_len: 0,
            probes: ProbeCache::new(),
            scan: None,
            changes: 0,
            changed_at: None,
            last_sig: 0,
            report: None,
            report_at: None,
            explain: None,
            last_error: None,
        }
    }

    /// Moves the settings-backed conditions the tab has not changed to the new settings.
    pub fn follow_settings(&mut self, settings: &crate::UiSettings) {
        self.follow_base(base_for(settings));
    }

    /// Moves the conditions the tab has not changed from the old base to `next`.
    pub fn follow_base(&mut self, next: FilterSpec) {
        if next != self.base {
            self.spec.follow(&self.base, &next);
            self.base = next;
        }
    }

    /// The conditions the settings hold.
    pub fn base(&self) -> &FilterSpec {
        &self.base
    }

    /// Whether the conditions read file system probes.
    pub fn uses_probes(spec: &FilterSpec) -> bool {
        !spec.offline.is_any() || spec.skip_icons
    }

    pub fn error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

#[cfg(test)]
mod tests;

impl FactSource for FileTableViewer {
    fn has_clip(&self, row: &Thumbnail) -> bool {
        cells::has_clip(self, row)
    }

    fn is_indexed(&self, row: &Thumbnail) -> bool {
        match &self.filters.indexed {
            IndexedSet::Everything => true,
            IndexedSet::Paths(paths) => paths.contains(&row.path),
        }
    }

    fn probe(&self, path: &str) -> Option<Probe> {
        self.filters.probes.get(path)
    }
}

impl FileTableViewer {
    fn effective_types(&self) -> TypeSet {
        TypeSet { images: self.types_show_images, videos: self.types_show_videos, dirs: self.types_show_dirs, ..self.filters.spec.types }
    }

    /// The tab's conditions, with the search text and type toggles of the viewer.
    pub fn filter_spec(&self) -> FilterSpec {
        let mut spec = self.filters.spec.clone();
        spec.query = self.filter.clone();
        spec.types = self.effective_types();
        spec
    }

    /// Replaces the tab's conditions, including the search text and type toggles.
    pub fn set_filter_spec(&mut self, spec: FilterSpec) {
        self.filter = spec.query.clone();
        self.types_show_images = spec.types.images;
        self.types_show_videos = spec.types.videos;
        self.types_show_dirs = spec.types.dirs;
        self.filters.spec = spec;
    }

    /// The conditions compiled for testing many rows.
    pub fn row_filter(&self) -> PreparedFilter<'_> {
        PreparedFilter::new(&self.filter_spec(), self)
    }

    /// Changes whenever [`row_filter`](Self::row_filter) may answer differently for the same rows.
    pub fn filter_signature(&self) -> u64 {
        let st = &self.filters;
        let spec = &st.spec;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (spec.signature_with(&self.filter, self.effective_types()), self.mode as u8).hash(&mut h);
        if !spec.clip.is_any() {
            (self.clip_presence.len(), self.clip_presence_hashes.len()).hash(&mut h);
        }
        if !spec.indexed.is_any() {
            st.indexed_rev.hash(&mut h);
        }
        if FilterState::uses_probes(spec) {
            st.probes.generation().hash(&mut h);
        }
        h.finish()
    }
}

/// The saved view that stands for `spec`; the older columns mirror what they can.
pub fn group_from_spec(name: &str, spec: &FilterSpec) -> FilterGroup {
    FilterGroup {
        id: None,
        name: name.to_string(),
        include_images: spec.types.images,
        include_videos: spec.types.videos,
        include_dirs: spec.types.dirs,
        skip_icons: spec.skip_icons,
        min_size_bytes: spec.size.min,
        max_size_bytes: spec.size.max,
        excluded_terms: spec.excluded_terms.clone(),
        spec: Some(spec.to_json()),
        created: None,
        updated: None,
    }
}

/// The spec a saved view holds: its JSON, or for older views the columns on top of no other condition.
pub fn spec_from_group(g: &FilterGroup) -> FilterSpec {
    if let Some(spec) = g.spec.as_deref().and_then(|text| FilterSpec::from_json(text).ok()) {
        return spec;
    }
    let mut spec = FilterSpec::none();
    spec.types.images = g.include_images;
    spec.types.videos = g.include_videos;
    spec.types.dirs = g.include_dirs;
    spec.skip_icons = g.skip_icons;
    spec.size.min = g.min_size_bytes;
    spec.size.max = g.max_size_bytes;
    spec.excluded_terms = g.excluded_terms.clone();
    spec.normalized()
}

fn probe_job(row: &Thumbnail, offline: bool, dims: bool, limit: u64) -> Option<ProbeJob> {
    if row.file_type == DIR_TYPE || row.path.is_empty() || row.path.contains("://") {
        return None;
    }
    let is_ico = row.path.rsplit('.').next().is_some_and(|e| e.eq_ignore_ascii_case("ico"));
    let want_dims = dims && !is_ico && row.size <= limit && kind_of_path(&row.path) == RowKind::Image;
    (offline || want_dims).then(|| ProbeJob { path: row.path.clone(), dims: want_dims })
}

impl FileExplorer {
    /// The tab's conditions.
    pub fn filter_spec(&self) -> FilterSpec {
        self.viewer.filter_spec()
    }

    /// Replaces the tab's conditions; rows are never dropped, only listed or not.
    pub fn set_filter_spec(&mut self, spec: FilterSpec) {
        let spec = spec.normalized();
        if spec == self.viewer.filter_spec() {
            return;
        }
        self.excluded_terms = spec.excluded_terms.clone();
        self.active_filter_group = None;
        self.viewer.set_filter_spec(spec);
    }

    /// Applies `edit` to a copy of the conditions and stores the result.
    pub fn edit_filters(&mut self, edit: impl FnOnce(&mut FilterSpec)) {
        let mut spec = self.filter_spec();
        edit(&mut spec);
        self.set_filter_spec(spec);
    }

    /// Removes every condition, the app defaults included.
    pub fn clear_filters(&mut self) {
        self.set_filter_spec(FilterSpec::none());
    }

    /// Goes back to the conditions the settings define.
    pub fn reset_filters(&mut self) {
        let base = self.viewer.filters.base().clone();
        self.set_filter_spec(base);
    }

    /// Saves the settings-backed conditions of the tab as the defaults of every tab.
    pub fn make_filters_default(&mut self) {
        let spec = self.filter_spec();
        crate::settings::update("filters:make-default", |s| spec.write_settings(s));
        self.viewer.sync_settings();
    }

    /// True when the settings-backed conditions differ from the settings.
    pub fn filters_differ_from_defaults(&self) -> bool {
        let spec = &self.viewer.filters.spec;
        let base = self.viewer.filters.base();
        (&spec.size, &spec.modified, spec.skip_icons, &spec.ext_deny, &spec.excluded_dirs) != (&base.size, &base.modified, base.skip_icons, &base.ext_deny, &base.excluded_dirs)
    }

    /// Merges a JSON patch into the conditions, or into the defaults when `replace`.
    pub fn apply_filter_patch(&mut self, patch: &Value, replace: bool) -> Result<String, String> {
        let next = self.filter_spec().merged(patch, replace)?;
        let count = next.chips().len();
        self.set_filter_spec(next);
        Ok(format!("{count} active filters"))
    }

    /// Replaces the conditions with the ones in `text` (JSON).
    pub fn apply_filter_json(&mut self, text: &str) -> Result<(), String> {
        let spec = FilterSpec::from_json(text)?;
        self.set_filter_spec(spec);
        Ok(())
    }

    /// Saved views as last loaded.
    pub fn saved_views(&self) -> &[FilterGroup] {
        &self.filter_groups
    }

    #[cfg(test)]
    pub(crate) fn set_saved_views(&mut self, groups: Vec<FilterGroup>) {
        self.filter_groups = groups;
    }

    /// Applies a saved view and remembers its name.
    pub fn apply_view(&mut self, group: &FilterGroup) {
        let spec = spec_from_group(group);
        self.set_filter_spec(spec);
        self.active_filter_group = Some(group.name.clone());
    }

    /// Applies the saved view called `name` (ignoring case).
    pub fn apply_saved_view(&mut self, name: &str) -> Result<String, String> {
        let wanted = name.trim().to_lowercase();
        let found = self.filter_groups.iter().find(|g| g.name.to_lowercase() == wanted).cloned();
        match found {
            Some(group) => {
                self.apply_view(&group);
                Ok(format!("applied view '{}'", group.name))
            }
            None if self.filter_groups.is_empty() => Err(format!("no saved view '{name}' (none are loaded)")),
            None => Err(format!("no saved view '{name}' (have: {})", self.filter_groups.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", "))),
        }
    }

    /// Drops the active view's name once the filters no longer equal the view.
    pub(crate) fn forget_changed_view(&mut self) {
        let Some(name) = self.active_filter_group.as_deref() else { return };
        let Some(group) = self.filter_groups.iter().find(|g| g.name == name) else { return };
        if spec_from_group(group) != self.filter_spec() {
            self.active_filter_group = None;
        }
    }

    /// Reloads the list of saved views in the background.
    pub fn refresh_saved_views(&self) {
        let tx = self.filter_groups_tx.clone();
        tokio::spawn(async move {
            match crate::database::list_filter_groups().await {
                Ok(groups) => {
                    let _ = tx.try_send(groups);
                }
                Err(e) => log::warn!("list_filter_groups failed: {e:?}"),
            }
        });
    }

    /// Saves the tab's conditions as the view `name`, replacing a view of that name.
    pub fn save_view(&mut self, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("give the view a name".to_string());
        }
        let group = group_from_spec(name, &self.filter_spec());
        self.active_filter_group = Some(name.to_string());
        let tx = self.filter_groups_tx.clone();
        tokio::spawn(async move {
            match crate::database::save_filter_group(group).await {
                Ok(_) => match crate::database::list_filter_groups().await {
                    Ok(groups) => {
                        let _ = tx.try_send(groups);
                    }
                    Err(e) => log::error!("list_filter_groups after save failed: {e:?}"),
                },
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Saving the view failed: {e}")),
            }
        });
        Ok(())
    }

    /// Renames a saved view.
    pub fn rename_view(&mut self, old: &str, new: &str) -> Result<(), String> {
        let new = new.trim();
        if new.is_empty() {
            return Err("give the view a name".to_string());
        }
        if self.active_filter_group.as_deref() == Some(old) {
            self.active_filter_group = Some(new.to_string());
        }
        let (old, new, tx) = (old.to_string(), new.to_string(), self.filter_groups_tx.clone());
        tokio::spawn(async move {
            match crate::database::rename_filter_group(&old, &new).await {
                Ok(()) => match crate::database::list_filter_groups().await {
                    Ok(groups) => {
                        let _ = tx.try_send(groups);
                    }
                    Err(e) => log::error!("list_filter_groups after rename failed: {e:?}"),
                },
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Renaming the view failed: {e}")),
            }
        });
        Ok(())
    }

    /// Deletes a saved view.
    pub fn delete_view(&mut self, name: &str) {
        if self.active_filter_group.as_deref() == Some(name) {
            self.active_filter_group = None;
        }
        let (name, tx) = (name.to_string(), self.filter_groups_tx.clone());
        tokio::spawn(async move {
            match crate::database::delete_filter_group_by_name(&name).await {
                Ok(()) => match crate::database::list_filter_groups().await {
                    Ok(groups) => {
                        let _ = tx.try_send(groups);
                    }
                    Err(e) => log::error!("list_filter_groups after delete failed: {e:?}"),
                },
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Deleting the view failed: {e}")),
            }
        });
    }

    /// Rows listed and rows in the tab as the status bar counts them.
    pub fn filter_counts(&self) -> (usize, usize) {
        if self.is_recursive_scan() {
            (self.recursive_total_filtered(), self.recursive_total_unfiltered())
        } else {
            let stats = self.view.stats();
            (stats.visible, stats.total)
        }
    }

    /// Updates what the conditions read: the indexed paths, the file probes and the change record.
    pub(crate) fn refresh_filter_facts(&mut self) {
        let (wants_indexed, offline, skip_icons, limit) = {
            let s = &self.viewer.filters.spec;
            (!s.indexed.is_any(), !s.offline.is_any(), s.skip_icons, s.size.min.unwrap_or(DEFAULT_MIN_SIZE))
        };
        if wants_indexed {
            self.refresh_indexed();
        }
        if offline || skip_icons {
            self.queue_probes(offline, skip_icons, limit);
        } else if self.table.is_empty() && !self.viewer.filters.probes.is_empty() {
            self.viewer.filters.probes.clear();
            self.viewer.filters.scan = None;
        }
        let sig = self.viewer.filter_signature();
        let st = &mut self.viewer.filters;
        if sig != st.last_sig {
            st.last_sig = sig;
            st.changes += 1;
            st.changed_at = Some(Instant::now());
        }
    }

    fn refresh_indexed(&mut self) {
        let st = &mut self.viewer.filters;
        if self.viewer.mode == ExplorerMode::Database {
            if !matches!(st.indexed, IndexedSet::Everything) {
                st.indexed = IndexedSet::Everything;
                st.indexed_rev += 1;
            }
            return;
        }
        if matches!(st.indexed, IndexedSet::Paths(_)) && st.indexed_len == self.db_lookup.len() {
            return;
        }
        st.indexed = IndexedSet::Paths(self.db_lookup.keys().cloned().collect());
        st.indexed_len = self.db_lookup.len();
        st.indexed_rev += 1;
    }

    fn queue_probes(&mut self, offline: bool, dims: bool, limit: u64) {
        let v = self.table.versions();
        let key = ScanKey { layout: v.layout, edits: v.edits, offline, dims, limit };
        let rows = self.table.rows();
        let st = &mut self.viewer.filters;
        let from = match st.scan {
            Some((k, scanned)) if k == key && scanned <= rows.len() => scanned,
            _ => 0,
        };
        st.scan = Some((key, rows.len()));
        if from >= rows.len() {
            return;
        }
        let jobs: Vec<ProbeJob> = rows[from..].iter().filter_map(|r| probe_job(r, offline, dims, limit)).collect();
        st.probes.request(jobs);
    }

    /// Makes file probe results repaint the window.
    pub(crate) fn bind_filter_waker(&self, ctx: &Context) {
        let probes = &self.viewer.filters.probes;
        if !probes.has_waker() {
            let ctx = ctx.clone();
            probes.set_waker(move || ctx.request_repaint());
        }
    }

    /// Runs the commands the Filters window queued for this tab.
    pub(crate) fn run_filter_commands(&mut self, ctx: &Context) {
        for command in devtools::take_commands(ctx, self.filter_debug_id()) {
            self.viewer.filters.last_error = None;
            match command {
                Command::ApplyJson(text) => {
                    if let Err(e) = self.apply_filter_json(&text) {
                        self.viewer.filters.last_error = Some(e);
                    }
                }
                Command::Clear => self.clear_filters(),
                Command::Reset => self.reset_filters(),
                Command::MakeDefault => self.make_filters_default(),
                Command::Explain(path) => self.explain_path(&path),
            }
        }
    }

    /// Records which conditions list or hide the row of `path`.
    pub fn explain_path(&mut self, path: &str) {
        let path = path.trim();
        let report = match self.table.get_by_path(path) {
            Some(row) => {
                let filter = self.viewer.row_filter();
                let verdicts = filter.explain(row);
                ExplainReport { path: path.to_string(), found: true, listed: verdicts.iter().all(|v| v.passed), verdicts }
            }
            None => ExplainReport { path: path.to_string(), found: false, listed: false, verdicts: Vec::new() },
        };
        self.viewer.filters.explain = Some(report);
    }

    fn filter_debug_id(&self) -> egui::Id {
        egui::Id::unique(("ame_filters_owner", self.viewer.scope))
    }

    /// Shows the tab in the Filters window when it is open.
    pub(crate) fn publish_filter_debug(&mut self, ctx: &Context) {
        if !devtools::is_open(ctx) {
            return;
        }
        let key = (self.view.revision(), self.viewer.filter_signature(), self.table.revision());
        let st = &self.viewer.filters;
        let stale = match &st.report {
            None => true,
            Some((k, _)) => *k != key && (k.1 != key.1 || st.report_at.is_none_or(|t| t.elapsed() >= REPORT_EVERY)),
        };
        if stale {
            let report = self.viewer.row_filter().report(self.table.iter());
            self.viewer.filters.report = Some((key, report));
            self.viewer.filters.report_at = Some(Instant::now());
        }
        let st = &self.viewer.filters;
        let report = st.report.as_ref().map(|(_, r)| r.clone()).unwrap_or_default();
        devtools::publish(
            ctx,
            Snapshot {
                owner: self.filter_debug_id(),
                label: self.current_path.clone(),
                spec_json: self.filter_spec().to_json(),
                base_json: st.base().to_json(),
                signature: key.1,
                chips: self.filter_spec().chips().len(),
                rows_total: self.table.len(),
                rows_visible: self.view.len(),
                view_revision: self.view.revision(),
                refresh: self.view.counts(),
                last_refresh: self.view.last_refresh(),
                changes: st.changes,
                since_change: st.changed_at.map(|t| t.elapsed()),
                report,
                probe_entries: st.probes.len(),
                probe_pending: st.probes.pending(),
                probe_generation: st.probes.generation(),
                indexed: match &st.indexed {
                    IndexedSet::Everything => "every row".to_string(),
                    IndexedSet::Paths(p) => format!("{} paths", p.len()),
                },
                explain: st.explain.clone(),
                views: self.filter_groups.iter().map(|g| g.name.clone()).collect(),
                active_view: self.active_filter_group.clone(),
                error: st.last_error.clone(),
            },
        );
    }
}
