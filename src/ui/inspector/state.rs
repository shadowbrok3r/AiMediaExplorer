//! Inspector state: follows the focused browser, runs the background lookups and saves, and applies their results.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crossbeam::channel::{Receiver, Sender, unbounded};
use egui::Context;

use super::backend::{Backend, CollectionInfo, CollectionOp, DbBackend, Library};
use super::facts::{FileFacts, LookupCache, read_facts};
use super::meta::{Applied, Field, MetaDraft, MetaEditor, SaveRequest, SaveState};
use super::prefs::{InspectorPrefs, PrefsStore};
use super::target::{BatchSummary, CONFIRM_ABOVE, Target, is_dir, is_image_path, is_video_path, resolve};
use super::view::ViewState;
use crate::Thumbnail;
use crate::command::ToastLevel;
use crate::database::metadata_edit::{BatchReport, MetaPatch, RowSeed, SaveOutcome, TagChange, apply_tag_change};
use crate::ui::file_table::FileExplorer;

/// How soon a failed library lookup is tried again.
const LIBRARY_RETRY: std::time::Duration = std::time::Duration::from_secs(3);
/// How long the lists for suggestions stay valid.
const LIST_STALE: std::time::Duration = std::time::Duration::from_secs(60);

/// A change to rows already shown in the browsers.
#[derive(Clone, Debug, PartialEq)]
pub enum RowEdit {
    Patch { path: String, patch: MetaPatch },
    Tags(HashMap<String, Vec<String>>),
    Category { paths: Vec<String>, category: Option<String> },
}

impl RowEdit {
    /// Applies the edit to `row` when it concerns it.
    pub fn apply_to(&self, row: &mut Thumbnail) {
        match self {
            RowEdit::Patch { path, patch } if *path == row.path => {
                if let Some(v) = &patch.description {
                    row.description = v.clone().filter(|s| !s.trim().is_empty());
                }
                if let Some(v) = &patch.caption {
                    row.caption = v.clone().filter(|s| !s.trim().is_empty());
                }
                if let Some(v) = &patch.category {
                    row.category = v.clone().filter(|s| !s.trim().is_empty());
                }
                if let Some(v) = &patch.tags {
                    row.tags = v.clone();
                }
            }
            RowEdit::Tags(map) => {
                if let Some(tags) = map.get(&row.path) {
                    row.tags = tags.clone();
                }
            }
            RowEdit::Category { paths, category } if paths.iter().any(|p| *p == row.path) => row.category = category.clone(),
            _ => {}
        }
    }

    fn touches(&self, path: &str) -> bool {
        match self {
            RowEdit::Patch { path: p, .. } => p == path,
            RowEdit::Tags(map) => map.contains_key(path),
            RowEdit::Category { paths, .. } => paths.iter().any(|p| p == path),
        }
    }
}

/// Applies `edit` to the rows of `ex` it concerns and to its selection copy.
pub fn apply_edit(ex: &mut FileExplorer, edit: &RowEdit) {
    let paths: Vec<String> = match edit {
        RowEdit::Patch { path, .. } => vec![path.clone()],
        RowEdit::Tags(map) => map.keys().cloned().collect(),
        RowEdit::Category { paths, .. } => paths.clone(),
    };
    for path in &paths {
        if let Some(mut row) = ex.table.get_mut_by_path(path) {
            edit.apply_to(&mut row);
        }
    }
    if edit.touches(&ex.current_thumb.path) {
        edit.apply_to(&mut ex.current_thumb);
    }
}

/// Result of a batch run.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchOutcome {
    pub message: String,
    pub edit: Option<RowEdit>,
    /// Paths whose library facts changed.
    pub library_paths: Vec<String>,
    pub refresh_lists: bool,
}

/// What came back from a background task.
#[derive(Debug)]
pub enum Event {
    Saved { path: String, rev: u64, patch: MetaPatch, result: Result<SaveOutcome, String> },
    Facts { path: String, facts: FileFacts },
    Library { path: String, result: Result<Library, String> },
    Tags(Result<Vec<String>, String>),
    Categories(Result<Vec<String>, String>),
    Collections(Result<Vec<CollectionInfo>, String>),
    Batch { label: String, result: Result<BatchOutcome, String> },
}

/// An action on the selected files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchAction {
    AddTags(Vec<String>),
    RemoveTags(Vec<String>),
    SetCategory(Option<String>),
    Describe { regenerate: bool },
    GenerateClip,
    Refine,
    AttachToAssistant,
}

impl BatchAction {
    pub fn label(&self) -> &'static str {
        match self {
            BatchAction::AddTags(_) => "Add tags",
            BatchAction::RemoveTags(_) => "Remove tags",
            BatchAction::SetCategory(Some(_)) => "Set category",
            BatchAction::SetCategory(None) => "Clear category",
            BatchAction::Describe { regenerate: false } => "Describe",
            BatchAction::Describe { regenerate: true } => "Regenerate descriptions",
            BatchAction::GenerateClip => "Generate CLIP",
            BatchAction::Refine => "Refine",
            BatchAction::AttachToAssistant => "Ask AI",
        }
    }

    /// Work that runs the models on every file and so asks first on large selections.
    pub fn is_heavy(&self) -> bool {
        matches!(self, BatchAction::Describe { .. } | BatchAction::GenerateClip)
    }
}

/// Whether `action` on `count` files asks for a confirmation first.
pub fn needs_confirm(action: &BatchAction, count: usize) -> bool {
    action.is_heavy() && count > CONFIRM_ABOVE
}

/// A heavy action waiting for the user's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingBatch {
    pub action: BatchAction,
    pub count: usize,
}

/// Inputs of the batch editor.
#[derive(Debug, Default)]
pub struct BatchForm {
    pub tag_input: String,
    pub category_input: String,
    pub collection_pick: Option<String>,
    pub new_collection: String,
    pub confirm: Option<PendingBatch>,
}

/// What the preview drew last, for the devtools window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreviewReport {
    pub source: &'static str,
    pub texture: Option<[usize; 2]>,
    pub image: Option<[f32; 2]>,
}

impl PreviewReport {
    /// Memory the shown texture takes on the GPU.
    pub fn texture_bytes(&self) -> usize {
        self.texture.map_or(0, |[w, h]| w * h * 4)
    }
}

/// Counters of background work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub saves_started: usize,
    pub saves_ok: usize,
    pub saves_failed: usize,
    pub batches: usize,
    pub lookups: usize,
}

/// The lightbox viewer.
#[derive(Debug, Default)]
pub struct Lightbox {
    pub open: bool,
    pub view: ViewState,
}

impl Lightbox {
    /// Opens the viewer on a fresh view of the picture.
    pub fn show(&mut self) {
        self.open = true;
        self.view = ViewState::default();
    }
}

pub struct Inspector {
    pub prefs: PrefsStore,
    pub meta: MetaEditor,
    pub backend: Arc<dyn Backend>,
    tx: Sender<Event>,
    rx: Receiver<Event>,
    ctx: Option<Context>,
    seed: Option<RowSeed>,
    pub facts: LookupCache<FileFacts>,
    pub library: LookupCache<Library>,
    pub known_tags: Vec<String>,
    pub known_categories: Vec<String>,
    pub collections: Vec<CollectionInfo>,
    lists_at: Option<Instant>,
    lists_loading: bool,
    collections_at: Option<Instant>,
    collections_loading: bool,
    pub view: ViewState,
    pub lightbox: Lightbox,
    pub rail_focused: bool,
    pub tag_input: String,
    /// The text field whose suggestions are shown.
    pub suggest_field: Option<Field>,
    /// Tab title and the value of its preview flag as last synced with the global state.
    pub flag_sync: Option<(String, bool)>,
    pub batch: BatchForm,
    summary: Option<((u64, u64), BatchSummary)>,
    pub preview: PreviewReport,
    pub counters: Counters,
    pub force_overlay: Option<bool>,
    pub overlay_active: bool,
    pub last_target: String,
    notices: Vec<(ToastLevel, String)>,
    pub last_batch: Option<String>,
}

impl Default for Inspector {
    fn default() -> Self {
        Self::new(Arc::new(DbBackend::global()), PrefsStore::default())
    }
}

impl Inspector {
    pub fn new(backend: Arc<dyn Backend>, prefs: PrefsStore) -> Self {
        let (tx, rx) = unbounded();
        let mode = prefs.prefs.preview_mode;
        Self {
            prefs,
            meta: MetaEditor::default(),
            backend,
            tx,
            rx,
            ctx: None,
            seed: None,
            facts: LookupCache::default(),
            library: LookupCache::default(),
            known_tags: Vec::new(),
            known_categories: Vec::new(),
            collections: Vec::new(),
            lists_at: None,
            lists_loading: false,
            collections_at: None,
            collections_loading: false,
            view: ViewState::with_mode(mode),
            lightbox: Lightbox::default(),
            rail_focused: false,
            tag_input: String::new(),
            suggest_field: None,
            flag_sync: None,
            batch: BatchForm::default(),
            summary: None,
            preview: PreviewReport::default(),
            counters: Counters::default(),
            force_overlay: None,
            overlay_active: false,
            last_target: "empty".to_string(),
            notices: Vec::new(),
            last_batch: None,
        }
    }

    /// Loads the saved preferences from the config directory.
    pub fn load_prefs(&mut self) {
        self.prefs = PrefsStore::open(super::prefs::default_path());
        self.view = ViewState::with_mode(self.prefs.prefs.preview_mode);
    }

    pub fn is_open(&self) -> bool {
        self.prefs.prefs.open
    }

    pub fn set_open(&mut self, open: bool) {
        self.prefs.update(Instant::now(), |p| p.open = open);
        self.repaint();
    }

    pub fn toggle(&mut self) -> bool {
        let open = !self.is_open();
        self.set_open(open);
        open
    }

    pub fn edit_prefs(&mut self, edit: impl FnOnce(&mut InspectorPrefs)) {
        self.prefs.update(Instant::now(), edit);
    }

    pub fn repaint(&self) {
        if let Some(ctx) = &self.ctx {
            ctx.request_repaint();
        }
    }

    pub fn take_notices(&mut self) -> Vec<(ToastLevel, String)> {
        std::mem::take(&mut self.notices)
    }

    pub fn notify(&mut self, level: ToastLevel, text: impl Into<String>) {
        self.notices.push((level, text.into()));
    }

    pub(super) fn spawn<F: std::future::Future<Output = ()> + Send + 'static>(&self, fut: F) -> bool {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(fut);
                true
            }
            Err(_) => {
                log::warn!("[inspector] no async runtime; background work skipped");
                false
            }
        }
    }

    fn sender(&self) -> impl Fn(Event) + Send + Sync + 'static {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        move |event| {
            let _ = tx.send(event);
            if let Some(ctx) = &ctx {
                ctx.request_repaint();
            }
        }
    }

    /// Takes in what the background tasks finished; the host applies the returned edits to the browsers before the next step.
    pub fn pump(&mut self, ctx: &Context, now: Instant) -> Vec<RowEdit> {
        self.ctx.get_or_insert_with(|| ctx.clone());
        self.drain_events(now)
    }

    /// Runs one frame of bookkeeping for `ex` and returns the target; an inactive rail looks nothing up and writes unsaved edits out.
    pub fn begin_frame(&mut self, ctx: &Context, ex: &FileExplorer, now: Instant, active: bool) -> Target {
        self.ctx.get_or_insert_with(|| ctx.clone());
        let target = resolve(ex);
        self.last_target = target.label();
        let editable = if active { target.editable() } else { None };
        match editable {
            Some(row) => {
                let draft = MetaDraft::from_row(row);
                let seed = RowSeed::from_row(row);
                let previous = self.seed.replace(seed);
                let moved = self.meta.path() != Some(row.path.as_str());
                if let Some(req) = self.meta.bind(&row.path, &draft) {
                    self.start_save(req, previous);
                }
                if moved {
                    self.tag_input.clear();
                    self.suggest_field = None;
                }
                self.meta.sync_row(&draft);
            }
            None => {
                if let Some(req) = self.meta.unbind() {
                    let seed = self.seed.take();
                    self.start_save(req, seed);
                }
                self.seed = None;
                self.tag_input.clear();
                self.suggest_field = None;
            }
        }
        if let Some(row) = focus_row(&target).filter(|_| active) {
            self.want_lookups(row, now);
        }
        if let Some(req) = self.meta.poll(now) {
            let seed = self.seed.clone();
            self.start_save(req, seed);
        }
        if let Some(wait) = self.meta.due_in(now) {
            ctx.request_repaint_after(wait.max(std::time::Duration::from_millis(30)));
        }
        if let Some(wait) = self.prefs.due_in(now) {
            ctx.request_repaint_after(wait.max(std::time::Duration::from_millis(30)));
        }
        self.prefs.save_if_due(now);
        target
    }

    fn want_lookups(&mut self, row: &Thumbnail, now: Instant) {
        if is_dir(row) && row.path.contains("://") {
            return;
        }
        if self.facts.want(&row.path, now) {
            let (path, image, video) = (row.path.clone(), is_image_path(&row.path), is_video_path(&row.path));
            let send = self.sender();
            self.counters.lookups += 1;
            self.spawn(async move {
                let key = path.clone();
                let facts = tokio::task::spawn_blocking(move || read_facts(&key, image, video)).await.unwrap_or_default();
                send(Event::Facts { path, facts });
            });
        }
        if !is_dir(row) && self.library.want(&row.path, now) {
            let (path, backend, send) = (row.path.clone(), self.backend.clone(), self.sender());
            self.counters.lookups += 1;
            self.spawn(async move {
                let result = backend.library(path.clone()).await;
                send(Event::Library { path, result });
            });
        }
    }

    fn start_save(&mut self, req: SaveRequest, seed: Option<RowSeed>) {
        let Some(seed) = seed.filter(|s| s.path == req.path) else {
            log::warn!("[inspector] no row data for {}; save skipped", req.path);
            return;
        };
        self.counters.saves_started += 1;
        let (backend, send) = (self.backend.clone(), self.sender());
        let (path, rev, patch) = (req.path, req.rev, req.patch);
        let failed_path = path.clone();
        if !self.spawn(async move {
            let result = backend.save(seed, patch.clone()).await;
            send(Event::Saved { path, rev, patch, result });
        }) {
            self.meta.finished(&failed_path, rev, Err("no async runtime".to_string()));
        }
    }

    /// Starts the save of pending edits at once (a blur, Enter or a discrete change).
    pub fn flush_now(&mut self, now: Instant) {
        self.meta.request_flush();
        if let Some(req) = self.meta.poll(now) {
            let seed = self.seed.clone();
            self.start_save(req, seed);
        }
    }

    fn drain_events(&mut self, now: Instant) -> Vec<RowEdit> {
        let mut edits = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            self.handle(event, now, &mut edits);
        }
        edits
    }

    fn handle(&mut self, event: Event, now: Instant, edits: &mut Vec<RowEdit>) {
        match event {
            Event::Saved { path, rev, patch, result } => {
                let name = file_name(&path);
                match &result {
                    Ok(outcome) => {
                        self.counters.saves_ok += 1;
                        edits.push(RowEdit::Patch { path: path.clone(), patch: patch.clone() });
                        self.lists_at = None;
                        if outcome.created {
                            self.library.invalidate(&path);
                            self.notify(ToastLevel::Info, format!("Added {name} to the library"));
                        }
                    }
                    Err(e) => {
                        self.counters.saves_failed += 1;
                        self.notify(ToastLevel::Error, format!("Could not save the metadata of {name}: {e}"));
                    }
                }
                let applied = self.meta.finished(&path, rev, result);
                if applied == Applied::Current {
                    self.repaint();
                }
            }
            Event::Facts { path, facts } => self.facts.put(&path, facts, now),
            Event::Library { path, result } => match result {
                Ok(lib) => self.library.put(&path, lib, now),
                Err(e) => {
                    log::debug!("[inspector] library lookup failed for {path}: {e}");
                    self.library.put_for(&path, Library { error: Some(e), ..Default::default() }, now, LIBRARY_RETRY);
                }
            },
            Event::Tags(r) => {
                self.lists_loading = false;
                self.lists_at = Some(now);
                match r {
                    Ok(list) => self.known_tags = list,
                    Err(e) => log::debug!("[inspector] tag list failed: {e}"),
                }
            }
            Event::Categories(r) => {
                if let Ok(list) = r {
                    self.known_categories = list;
                }
            }
            Event::Collections(r) => {
                self.collections_loading = false;
                self.collections_at = Some(now);
                match r {
                    Ok(list) => self.collections = list,
                    Err(e) => log::debug!("[inspector] collection list failed: {e}"),
                }
            }
            Event::Batch { label, result } => match result {
                Ok(done) => {
                    self.last_batch = Some(done.message.clone());
                    self.notify(ToastLevel::Success, done.message);
                    edits.extend(done.edit);
                    for p in done.library_paths {
                        self.library.invalidate(&p);
                    }
                    if done.refresh_lists {
                        self.lists_at = None;
                        self.collections_at = None;
                    }
                }
                Err(e) => {
                    self.last_batch = Some(format!("{label} failed: {e}"));
                    self.notify(ToastLevel::Error, format!("{label} failed: {e}"));
                }
            },
        }
    }

    /// Loads the tag and category suggestion lists when they are missing or old.
    pub fn want_lists(&mut self, now: Instant) {
        let stale = self.lists_at.is_none_or(|t| now.saturating_duration_since(t) > LIST_STALE);
        if !stale || self.lists_loading {
            return;
        }
        self.lists_loading = true;
        let (b1, b2, send) = (self.backend.clone(), self.backend.clone(), self.sender());
        let send2 = self.sender();
        if !self.spawn(async move {
            send(Event::Tags(b1.tags().await));
        }) {
            self.lists_loading = false;
            return;
        }
        self.spawn(async move {
            send2(Event::Categories(b2.categories().await));
        });
    }

    /// Loads the collection list when it is missing or old.
    pub fn want_collections(&mut self, now: Instant) {
        let stale = self.collections_at.is_none_or(|t| now.saturating_duration_since(t) > LIST_STALE);
        if !stale || self.collections_loading {
            return;
        }
        self.collections_loading = true;
        let (backend, send) = (self.backend.clone(), self.sender());
        if !self.spawn(async move {
            send(Event::Collections(backend.collections().await));
        }) {
            self.collections_loading = false;
        }
    }

    /// Numbers over the selected rows, recomputed only when the selection or the rows changed.
    pub fn summary(&mut self, ex: &FileExplorer) -> &BatchSummary {
        let key = (ex.selection.revision(), ex.table.versions().revision);
        if self.summary.as_ref().is_none_or(|(k, _)| *k != key) {
            let rows = ex.selected_rows();
            self.summary = Some((key, BatchSummary::of(&rows)));
        }
        &self.summary.as_ref().expect("computed above").1
    }

    /// Runs `action` on the selected files of `ex`; heavy work on a large selection waits in `batch.confirm` unless `confirmed`.
    pub fn run_batch(&mut self, ex: &mut FileExplorer, action: BatchAction, confirmed: bool) {
        let files: Vec<Thumbnail> = ex.selected_rows().into_iter().filter(|r| !is_dir(r)).cloned().collect();
        if files.is_empty() {
            self.notify(ToastLevel::Warning, format!("{}: no files are selected", action.label()));
            return;
        }
        if !confirmed && needs_confirm(&action, files.len()) {
            self.batch.confirm = Some(PendingBatch { action, count: files.len() });
            return;
        }
        self.batch.confirm = None;
        let paths: Vec<String> = files.iter().map(|r| r.path.clone()).collect();
        let count = files.len();
        match action {
            BatchAction::Describe { regenerate } => {
                let refs: Vec<&Thumbnail> = files.iter().collect();
                let menu = if regenerate { crate::ui::file_table::table::actions::MenuAction::RegenerateDescription } else { crate::ui::file_table::table::actions::MenuAction::GenerateDescription };
                ex.viewer.run_action(menu, &refs);
                self.notify(ToastLevel::Info, format!("Describing {count} files"));
            }
            BatchAction::GenerateClip => crate::command::send(crate::command::AppCommand::Run { action: crate::command::ActionId::GenerateClip, on: crate::command::Selector::Paths(paths) }),
            BatchAction::Refine => crate::ui::refine::request_refine_for_paths(paths),
            BatchAction::AttachToAssistant => crate::ui::assistant::request_attach_to_chat(paths),
            BatchAction::AddTags(tags) => self.spawn_tag_batch(files, TagChange { add: tags, remove: Vec::new() }),
            BatchAction::RemoveTags(tags) => self.spawn_tag_batch(files, TagChange { add: Vec::new(), remove: tags }),
            BatchAction::SetCategory(category) => {
                let (backend, send) = (self.backend.clone(), self.sender());
                let seeds: Vec<RowSeed> = files.iter().map(RowSeed::from_row).collect();
                self.counters.batches += 1;
                self.spawn(async move {
                    let label = if category.is_some() { "Set category" } else { "Clear category" };
                    let result = backend.batch_category(seeds, category.clone()).await.map(|r| {
                        let verb = match &category {
                            Some(c) => format!("Set the category \"{c}\" on {} files", r.updated),
                            None => format!("Cleared the category of {} files", r.updated),
                        };
                        BatchOutcome { message: with_created(verb, r.created), edit: Some(RowEdit::Category { paths: paths.clone(), category }), library_paths: paths, refresh_lists: true }
                    });
                    send(Event::Batch { label: label.to_string(), result });
                });
            }
        }
    }

    fn spawn_tag_batch(&mut self, files: Vec<Thumbnail>, change: TagChange) {
        let (backend, send) = (self.backend.clone(), self.sender());
        let seeds: Vec<RowSeed> = files.iter().map(RowSeed::from_row).collect();
        let paths: Vec<String> = files.iter().map(|r| r.path.clone()).collect();
        self.counters.batches += 1;
        let local = local_tags(&files, &change);
        self.spawn(async move {
            let label = if change.remove.is_empty() { "Add tags" } else { "Remove tags" };
            let result = backend.batch_tags(seeds, change.clone()).await.map(|r: BatchReport| {
                let tags = if r.tags.is_empty() { local } else { r.tags };
                let verb = if change.remove.is_empty() { format!("Added {} to {} files", describe_tags(&change.add), r.updated) } else { format!("Removed {} from {} files", describe_tags(&change.remove), r.updated) };
                BatchOutcome { message: with_created(verb, r.created), edit: Some(RowEdit::Tags(tags)), library_paths: paths, refresh_lists: true }
            });
            send(Event::Batch { label: label.to_string(), result });
        });
    }

    /// Adds the selected files to a collection or removes them from it.
    pub fn run_collection(&mut self, ex: &FileExplorer, name: &str, add: bool) {
        let paths: Vec<String> = ex.selected_rows().into_iter().filter(|r| !is_dir(r)).map(|r| r.path.clone()).collect();
        if paths.is_empty() {
            self.notify(ToastLevel::Warning, "Collections: no files are selected");
            return;
        }
        let name = name.trim().to_string();
        if name.is_empty() {
            self.notify(ToastLevel::Warning, "Choose a collection first");
            return;
        }
        let (backend, send) = (self.backend.clone(), self.sender());
        self.counters.batches += 1;
        self.spawn(async move {
            let op = if add { CollectionOp::Add { name, paths: paths.clone() } } else { CollectionOp::Remove { name, paths: paths.clone() } };
            let result = backend.collection_edit(op).await.map(|message| BatchOutcome { message, edit: None, library_paths: paths, refresh_lists: true });
            send(Event::Batch { label: if add { "Add to collection" } else { "Remove from collection" }.to_string(), result });
        });
    }

    pub fn edit_state(&self) -> SaveState {
        self.meta.state()
    }
}

fn focus_row(target: &Target) -> Option<&Thumbnail> {
    match target {
        Target::Empty => None,
        Target::Single(row) => Some(row),
        Target::Multi { focus, .. } => focus.as_deref(),
    }
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path).to_string()
}

fn describe_tags(tags: &[String]) -> String {
    match tags {
        [one] => format!("the tag \"{one}\""),
        many => format!("{} tags", many.len()),
    }
}

fn with_created(message: String, created: usize) -> String {
    if created == 0 { message } else { format!("{message}; {created} added to the library") }
}

/// The tags each file would have after `change`, from the rows as shown.
pub fn local_tags(files: &[Thumbnail], change: &TagChange) -> HashMap<String, Vec<String>> {
    files.iter().map(|r| (r.path.clone(), apply_tag_change(&r.tags, change))).collect()
}

#[cfg(test)]
pub(crate) mod tests;
