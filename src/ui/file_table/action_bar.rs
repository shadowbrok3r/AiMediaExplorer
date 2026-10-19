//! The selection action bar over the bottom of the browser, and the executor that the bar, the context menu, the keys, the commands and Devtools share.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender};
use eframe::egui::*;
use humansize::DECIMAL;

use super::FileExplorer;
use super::batch::{self, files_noun};
use super::table::actions::{MenuAction, MenuTargets};
use crate::Thumbnail;
use crate::command::actions::{Confirm, Facts, OPEN_LIMIT, SPECS, availability, needs_confirmation, plan_bar, spec_of, target_count};
use crate::command::{ActionId, CommandError, CommandResult};
use crate::database::metadata_edit::TagChange;
use crate::icons::{self, svg};
use crate::ui::file_table::rows::thumbnail::DIR_TYPE;

mod popovers;
#[cfg(test)]
mod tests;

pub use popovers::{CategoryForm, CollectionForm, Forms, Intent, Pending, Popover, TagForm};

/// How long the vision model check is reused.
const AI_RECHECK: Duration = Duration::from_secs(2);
/// Runs kept for the Devtools window.
const RUN_LOG: usize = 40;
/// Longest time library lists for suggestions stay valid.
const LISTS_STALE: Duration = Duration::from_secs(30);
/// Names listed in a confirmation before the rest are summarised as a count.
const NAME_PREVIEW: usize = 5;

/// Where a request to run an action came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Bar,
    Menu,
    Key,
    Command,
    Devtools,
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bar => "action bar",
            Self::Menu => "context menu",
            Self::Key => "key",
            Self::Command => "command",
            Self::Devtools => "devtools",
        }
    }
}

/// What a request to run an action led to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Done {
    Ran(String),
    /// A popover or dialog is waiting for the user.
    Asked(String),
}

impl Done {
    pub fn text(&self) -> &str {
        match self {
            Done::Ran(t) | Done::Asked(t) => t,
        }
    }
}

/// One request, for the Devtools history.
#[derive(Clone, Debug)]
pub struct ActionRun {
    pub at: Instant,
    pub action: ActionId,
    pub origin: Origin,
    pub count: usize,
    pub outcome: Result<Done, String>,
}

/// What a background library edit reports back to the tab.
#[derive(Debug)]
pub enum BarEvent {
    Tags(HashMap<String, Vec<String>>),
    Category { paths: Vec<String>, category: Option<String> },
    Removed(Vec<String>),
    Lists { tags: Vec<String>, categories: Vec<String>, collections: Vec<(String, u64)> },
}

/// Where the parts of the bar were drawn in the last frame.
#[derive(Clone, Debug)]
pub struct BarLayout {
    pub rect: Rect,
    pub browser: Rect,
    pub label: Rect,
    pub buttons: Vec<(ActionId, Rect, bool)>,
    pub more: Option<Rect>,
    pub clear: Rect,
    pub hidden: usize,
    /// Rectangle of the open popover.
    pub popover: Option<Rect>,
}

impl Default for BarLayout {
    fn default() -> Self {
        Self { rect: Rect::ZERO, browser: Rect::ZERO, label: Rect::ZERO, buttons: Vec::new(), more: None, clear: Rect::ZERO, hidden: 0, popover: None }
    }
}

/// Edge of a medium icon in `style`, as the icon buttons size it.
fn icon_md(style: &Style) -> f32 {
    (style.text_styles.get(&TextStyle::Body).map_or(13.0, |f| f.size) * 1.08).round().max(8.0)
}

/// State of the action bar of one tab.
pub struct BarState {
    pub popover: Option<Popover>,
    pub forms: Forms,
    /// Rows the open popover acts on.
    pub target: Vec<Thumbnail>,
    /// Selection revision the popover is bound to; it closes when the selection changes.
    bound: Option<u64>,
    tx: Sender<BarEvent>,
    rx: Receiver<BarEvent>,
    facts: Option<((u64, u64, bool), Facts)>,
    ai: Option<(Instant, bool)>,
    pub known_tags: Vec<String>,
    pub known_categories: Vec<String>,
    pub collections: Vec<(String, u64)>,
    lists_at: Option<Instant>,
    pub layout: Option<BarLayout>,
    pub runs: VecDeque<ActionRun>,
}

impl Default for BarState {
    fn default() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self {
            popover: None,
            forms: Forms::default(),
            target: Vec::new(),
            bound: None,
            tx,
            rx,
            facts: None,
            ai: None,
            known_tags: Vec::new(),
            known_categories: Vec::new(),
            collections: Vec::new(),
            lists_at: None,
            layout: None,
            runs: VecDeque::new(),
        }
    }
}

impl BarState {
    fn record(&mut self, run: ActionRun) {
        if self.runs.len() >= RUN_LOG {
            self.runs.pop_front();
        }
        self.runs.push_back(run);
    }
}

#[cfg(test)]
thread_local! {
    static PINNED: std::cell::Cell<(Option<bool>, Option<bool>)> = const { std::cell::Cell::new((None, None)) };
}

/// Pins the library and the vision model checks for the current test thread.
#[cfg(test)]
pub(crate) fn pin_environment(db: Option<bool>, ai: Option<bool>) {
    PINNED.with(|c| c.set((db, ai)));
}

/// True when the library is connected.
pub fn db_ready() -> bool {
    #[cfg(test)]
    if let Some(v) = PINNED.with(|c| c.get().0) {
        return v;
    }
    crate::database::is_connected()
}

/// True when the vision model folder exists.
pub fn ai_ready() -> bool {
    #[cfg(test)]
    if let Some(v) = PINNED.with(|c| c.get().1) {
        return v;
    }
    let settings = crate::settings::snapshot();
    let chosen = settings.joycaption_model_dir.as_deref().map(str::trim).filter(|d| !d.is_empty());
    std::path::Path::new(chosen.unwrap_or(crate::app::DEFAULT_JOYCAPTION_PATH)).is_dir()
}

fn is_image_path(path: &str) -> bool {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_image(&e.to_ascii_lowercase()))
}

/// The facts over `rows`, which are the selected rows in view order.
pub fn facts_of(rows: &[&Thumbnail], db_ready: bool, ai_ready: bool) -> Facts {
    let mut f = Facts { selected: rows.len(), db_ready, ai_ready, ..Default::default() };
    for r in rows {
        if r.file_type == DIR_TYPE {
            f.dirs += 1;
            continue;
        }
        f.files += 1;
        f.size += r.size;
        if !r.path.contains("://") {
            f.local += 1;
            if is_image_path(&r.path) {
                f.images += 1;
            }
        }
    }
    f
}

/// The count line of the bar: `12 selected (48.2 MB)`.
pub fn count_text(f: &Facts) -> String {
    let mut text = format!("{} selected", f.selected);
    if f.dirs > 0 && f.files > 0 {
        text.push_str(&format!(": {}, {}", files_noun(f.files), if f.dirs == 1 { "1 folder".to_string() } else { format!("{} folders", f.dirs) }));
    }
    if f.size > 0 {
        text.push_str(&format!(" ({})", humansize::format_size(f.size, DECIMAL)));
    }
    text
}

/// The first names joined, then "and N more".
pub fn names_text(names: &[String]) -> String {
    let shown = names.iter().take(NAME_PREVIEW).cloned().collect::<Vec<_>>().join(", ");
    match names.len().saturating_sub(NAME_PREVIEW) {
        0 => shown,
        more => format!("{shown} and {more} more"),
    }
}

/// Title, body and button of the confirmation for `action` on `count` files.
pub fn confirm_text(action: ActionId, count: usize) -> (String, String, String) {
    let n = files_noun(count);
    match action {
        ActionId::RemoveFromLibrary => (
            "Remove from library".to_string(),
            format!("Remove {n} from the library? Their rows and CLIP embeddings are deleted. Files on disk are not touched."),
            format!("Remove {count}"),
        ),
        ActionId::MoveTo => ("Move files".to_string(), format!("Move {n} out of their folders into a folder you choose next?"), format!("Choose folder and move {count}")),
        ActionId::AddToLibrary => ("Add to library".to_string(), format!("Add {n} to the library?"), format!("Add {count}")),
        ActionId::CopyTo => ("Copy files".to_string(), format!("Copy {n} into a folder you choose next?"), format!("Choose folder and copy {count}")),
        ActionId::Describe | ActionId::RegenerateDescription => {
            let overwrite = if action == ActionId::RegenerateDescription { " Existing descriptions are replaced." } else { "" };
            ("Describe images".to_string(), format!("Run the vision model on {n}? This can take a long time.{overwrite}"), format!("Describe {count}"))
        }
        _ => {
            let label = spec_of(action).label;
            (label.trim_end_matches("...").to_string(), format!("{} on {n}? This can take a long time.", label.trim_end_matches("...")), format!("{} {count}", spec_of(action).short.trim_end_matches("...")))
        }
    }
}

impl FileExplorer {
    /// Facts of the selection, recomputed only when the selection, the rows or the connections changed.
    pub fn selection_facts(&mut self) -> Facts {
        self.sync_view();
        let db = db_ready();
        let key = (self.selection.revision(), self.table.versions().revision, db);
        let bar = &mut self.toolbar.bar;
        let ai_stale = bar.ai.is_none_or(|(at, _)| at.elapsed() > AI_RECHECK);
        if ai_stale {
            bar.ai = Some((Instant::now(), ai_ready()));
        }
        let ai = bar.ai.map_or(false, |(_, v)| v);
        if let Some((k, f)) = &self.toolbar.bar.facts {
            if *k == key && f.ai_ready == ai {
                return f.clone();
            }
        }
        let facts = facts_of(&self.selected_rows(), db, ai);
        self.toolbar.bar.facts = Some((key, facts.clone()));
        facts
    }

    fn selected_clones(&mut self) -> Vec<Thumbnail> {
        self.sync_view();
        self.selected_rows().into_iter().cloned().collect()
    }

    /// The row of `path`: the one in the tab, or one made from the file itself.
    fn row_for_path(&self, path: &str) -> Thumbnail {
        if let Some(row) = self.table.get_by_path(path) {
            return row.clone();
        }
        let p = std::path::Path::new(path);
        let mut row = Thumbnail::new(p.file_name().and_then(|n| n.to_str()).unwrap_or(path));
        row.path = path.to_string();
        row.parent_dir = p.parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_default();
        row.file_type = if p.is_dir() { DIR_TYPE.to_string() } else { p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default() };
        row.size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        row
    }

    /// Runs `action` on the selection the way the bar, the menu and the keys do; a reason it cannot run comes back as `Err`.
    pub fn run_action_ui(&mut self, ctx: &Context, action: ActionId, origin: Origin) -> Result<Done, String> {
        let rows = self.selected_clones();
        self.run_action_rows(ctx, action, rows, origin, false)
    }

    /// Runs `action` on explicit rows; `confirmed` skips the confirmation.
    pub fn run_action_rows(&mut self, ctx: &Context, action: ActionId, rows: Vec<Thumbnail>, origin: Origin, confirmed: bool) -> Result<Done, String> {
        let refs: Vec<&Thumbnail> = rows.iter().collect();
        let facts = facts_of(&refs, db_ready(), ai_ready());
        let count = target_count(action, &facts);
        let outcome = self.perform(ctx, action, rows, &facts, origin, confirmed);
        self.toolbar.bar.record(ActionRun { at: Instant::now(), action, origin, count, outcome: outcome.clone() });
        if let (Err(reason), true) = (&outcome, origin != Origin::Command) {
            crate::ui::dialogs::toast(egui_toast::ToastKind::Warning, format!("{}: {reason}", spec_of(action).short.trim_end_matches("...")));
        }
        outcome
    }

    /// Runs `action` for a command on explicit `paths`: only destructive actions ask first.
    pub fn run_action_paths(&mut self, ctx: &Context, action: ActionId, paths: Vec<String>) -> CommandResult {
        let rows: Vec<Thumbnail> = paths.iter().map(|p| self.row_for_path(p)).collect();
        self.run_action_rows(ctx, action, rows, Origin::Command, false).map(|d| d.text().to_string()).map_err(|reason| CommandError::invalid(format!("{}: {reason}", action.label())))
    }

    /// Runs the context menu item `action` on the selection.
    pub fn run_menu_action(&mut self, ctx: &Context, action: MenuAction) {
        let _ = self.run_action_ui(ctx, action.action_id(), Origin::Menu);
    }

    fn perform(&mut self, ctx: &Context, action: ActionId, rows: Vec<Thumbnail>, facts: &Facts, origin: Origin, confirmed: bool) -> Result<Done, String> {
        availability(action, facts)?;
        let count = target_count(action, facts);
        let asks = origin != Origin::Command || spec_of(action).confirm == Confirm::Always;
        if !confirmed && asks && needs_confirmation(action, count) {
            self.open_confirm(action, rows, count);
            return Ok(Done::Asked(format!("asked to confirm {} on {}", action.label(), files_noun(count))));
        }
        let refs: Vec<&Thumbnail> = rows.iter().collect();
        let files = batch::files_of(&refs);
        let paths: Vec<String> = refs.iter().map(|r| r.path.clone()).collect();
        let text = match action {
            ActionId::Open => {
                if let [only] = refs.as_slice() {
                    let key = self.table.key_for(&only.path);
                    self.open_row(&key);
                } else {
                    for r in refs.iter().filter(|r| r.file_type != DIR_TYPE && !r.path.contains("://")).take(OPEN_LIMIT) {
                        self.open_external(&r.path);
                    }
                }
                format!("opened {} items", refs.len())
            }
            ActionId::Reveal => {
                for p in paths.iter().filter(|p| !p.contains("://")).take(OPEN_LIMIT) {
                    if let Some(parent) = std::path::Path::new(p).parent() {
                        self.open_external(&parent.to_string_lossy());
                    }
                }
                format!("showed the folders of {} items", refs.len())
            }
            ActionId::CopyPath => {
                ctx.copy_text(paths.join("\n"));
                format!("copied {} paths", paths.len())
            }
            ActionId::AddToLibrary => {
                self.save_rows_to_library(files.clone());
                format!("saving {} to the library", files_noun(files.len()))
            }
            ActionId::RemoveFromLibrary => {
                self.spawn_remove(files.clone());
                format!("removing {} from the library", files_noun(files.len()))
            }
            ActionId::Tag | ActionId::Category | ActionId::AddToCollection | ActionId::RemoveFromCollection => {
                self.open_popover(action, rows);
                return Ok(Done::Asked(format!("{} popover opened for {}", action.label(), files_noun(files.len()))));
            }
            ActionId::CopyTo | ActionId::MoveTo => {
                let moving = action == ActionId::MoveTo;
                let local: Vec<std::path::PathBuf> = files.iter().filter(|r| !r.path.contains("://")).map(|r| std::path::PathBuf::from(&r.path)).collect();
                let n = local.len();
                super::table::actions::pick_destination(ctx, moving, local);
                return Ok(Done::Asked(format!("choose the folder to {} {} into", if moving { "move" } else { "copy" }, files_noun(n))));
            }
            ActionId::ExcludeFolder => {
                let dirs: Vec<String> = refs.iter().filter(|r| r.file_type == DIR_TYPE).map(|r| r.path.clone()).collect();
                self.exclude_folders(&dirs);
                format!("excluded {} from scans", if dirs.len() == 1 { "1 folder".to_string() } else { format!("{} folders", dirs.len()) })
            }
            ActionId::Describe | ActionId::RegenerateDescription | ActionId::GenerateClip | ActionId::RegenerateClip | ActionId::FindSimilar | ActionId::OpenInNewTab | ActionId::Refine | ActionId::AttachToAssistant => {
                self.run_viewer_action(action, &refs, facts)
            }
        };
        Ok(Done::Ran(text))
    }

    /// The actions the table's viewer runs itself: AI work, tabs and hand-offs.
    fn run_viewer_action(&mut self, action: ActionId, rows: &[&Thumbnail], facts: &Facts) -> String {
        let menu = MenuAction::from_action(action);
        match action {
            ActionId::Describe | ActionId::RegenerateDescription => {
                let overwrite = action == ActionId::RegenerateDescription || self.viewer.ui_settings.overwrite_descriptions;
                let images: Vec<&Thumbnail> = rows.iter().copied().filter(|r| r.file_type != DIR_TYPE && !r.path.contains("://") && is_image_path(&r.path)).collect();
                let have = images.iter().filter(|r| r.caption.is_some() || r.description.is_some()).count();
                let queued = if overwrite { images.len() } else { images.len() - have };
                self.viewer.run_action(menu, rows);
                if queued == 0 {
                    let text = format!("Nothing to describe: {} already described (use Regenerate to replace)", files_noun(have));
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Info, text.clone());
                    return text;
                }
                let text = if overwrite || have == 0 { format!("Describing {}", files_noun(queued)) } else { format!("Describing {}; {} already described", files_noun(queued), files_noun(have)) };
                crate::ui::dialogs::toast(egui_toast::ToastKind::Info, text.clone());
                text
            }
            ActionId::GenerateClip | ActionId::RegenerateClip => {
                self.viewer.run_action(menu, rows);
                let text = format!("Generating CLIP embeddings for {}", files_noun(facts.images));
                crate::ui::dialogs::toast(egui_toast::ToastKind::Info, text.clone());
                text
            }
            ActionId::FindSimilar => {
                self.viewer.run_action(menu, rows);
                "searching for similar files".to_string()
            }
            ActionId::OpenInNewTab => {
                self.viewer.run_action(menu, rows);
                format!("opened {} rows in a new tab", rows.len())
            }
            ActionId::Refine => {
                self.viewer.run_action(menu, rows);
                format!("refining {}", files_noun(facts.files))
            }
            _ => {
                self.viewer.run_action(menu, rows);
                format!("attached {} to the assistant", files_noun(facts.files))
            }
        }
    }

    fn exclude_folders(&mut self, dirs: &[String]) {
        let add: Vec<String> = dirs.to_vec();
        crate::settings::update("action-bar:exclude-dirs", |s| {
            let list = s.excluded_dirs.get_or_insert_with(Vec::new);
            for d in &add {
                if !list.contains(d) {
                    list.push(d.clone());
                }
            }
        });
        self.edit_filters(|spec| {
            for d in dirs {
                if !spec.excluded_dirs.contains(d) {
                    spec.excluded_dirs.push(d.clone());
                }
            }
        });
        self.viewer.sync_settings();
    }

    fn open_confirm(&mut self, action: ActionId, rows: Vec<Thumbnail>, count: usize) {
        let names: Vec<String> = rows.iter().filter(|r| r.file_type != DIR_TYPE).map(|r| r.filename.clone()).collect();
        self.bind_popover(Popover::Confirm(Pending { action, count, names }), rows);
    }

    fn open_popover(&mut self, action: ActionId, rows: Vec<Thumbnail>) {
        let popover = match action {
            ActionId::Tag => Popover::Tag,
            ActionId::Category => Popover::Category,
            ActionId::AddToCollection => Popover::AddToCollection,
            _ => Popover::RemoveFromCollection,
        };
        self.toolbar.bar.forms = Forms::for_rows(&rows);
        self.bind_popover(popover, rows);
        self.request_lists();
    }

    fn bind_popover(&mut self, popover: Popover, rows: Vec<Thumbnail>) {
        let revision = self.selection.revision();
        let bar = &mut self.toolbar.bar;
        bar.target = rows;
        bar.bound = Some(revision);
        bar.popover = Some(popover);
    }

    /// Closes the open popover and forgets its rows.
    pub fn close_popover(&mut self) {
        let bar = &mut self.toolbar.bar;
        bar.popover = None;
        bar.target.clear();
        bar.bound = None;
    }

    /// Reads the tags, categories and collections the popovers suggest, at most every half minute.
    fn request_lists(&mut self) {
        let bar = &mut self.toolbar.bar;
        if !db_ready() || bar.lists_at.is_some_and(|t| t.elapsed() < LISTS_STALE) {
            return;
        }
        bar.lists_at = Some(Instant::now());
        let tx = bar.tx.clone();
        tokio::spawn(async move {
            let db = &*crate::database::DB;
            let tags = crate::database::metadata_edit::all_tags(db).await.unwrap_or_default();
            let categories = crate::database::metadata_edit::all_categories(db).await.unwrap_or_default();
            let collections = crate::database::collections::list_with_counts(db).await.map(|l| l.into_iter().map(|c| (c.collection.name, c.count)).collect()).unwrap_or_default();
            let _ = tx.send(BarEvent::Lists { tags, categories, collections });
            crate::command::request_repaint();
        });
    }

    fn spawn_tags(&mut self, files: Vec<Thumbnail>, change: TagChange) {
        let tx = self.toolbar.bar.tx.clone();
        tokio::spawn(async move {
            match batch::tag_files(&*crate::database::DB, &files, &change).await {
                Ok(done) => {
                    let _ = tx.send(BarEvent::Tags(done.tags));
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Success, done.message);
                }
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Tagging failed: {e:#}")),
            }
            crate::ui::sidebar::request_refresh();
            crate::command::request_repaint();
        });
    }

    fn spawn_category(&mut self, files: Vec<Thumbnail>, category: Option<String>) {
        let tx = self.toolbar.bar.tx.clone();
        tokio::spawn(async move {
            match batch::categorize_files(&*crate::database::DB, &files, category.as_deref()).await {
                Ok(message) => {
                    let paths = files.iter().map(|r| r.path.clone()).collect();
                    let _ = tx.send(BarEvent::Category { paths, category: category.map(|c| c.trim().to_string()).filter(|c| !c.is_empty()) });
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Success, message);
                }
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Setting the category failed: {e:#}")),
            }
            crate::ui::sidebar::request_refresh();
            crate::command::request_repaint();
        });
    }

    fn spawn_collection(&mut self, files: Vec<Thumbnail>, name: String, add: bool) {
        tokio::spawn(async move {
            match batch::collect_files(&crate::database::DB, files, &name, add).await {
                Ok(message) => crate::ui::dialogs::toast(egui_toast::ToastKind::Success, message),
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Collection change failed: {e:#}")),
            }
            crate::ui::sidebar::request_refresh();
            crate::command::request_repaint();
        });
    }

    fn spawn_remove(&mut self, files: Vec<Thumbnail>) {
        let tx = self.toolbar.bar.tx.clone();
        tokio::spawn(async move {
            match batch::remove_files(&files).await {
                Ok(message) => {
                    let _ = tx.send(BarEvent::Removed(files.iter().map(|r| r.path.clone()).collect()));
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Success, message);
                }
                Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Removing from the library failed: {e:#}")),
            }
            crate::ui::sidebar::request_refresh();
            crate::command::request_repaint();
        });
    }

    /// Applies what finished library edits report to the rows in view.
    fn drain_bar_events(&mut self) {
        while let Ok(event) = self.toolbar.bar.rx.try_recv() {
            match event {
                BarEvent::Tags(map) => {
                    for (path, tags) in &map {
                        if let Some(mut row) = self.table.get_mut_by_path(path) {
                            row.tags = tags.clone();
                        }
                        if self.current_thumb.path == *path {
                            self.current_thumb.tags = tags.clone();
                        }
                    }
                }
                BarEvent::Category { paths, category } => {
                    for path in &paths {
                        if let Some(mut row) = self.table.get_mut_by_path(path) {
                            row.category = category.clone();
                        }
                        if self.current_thumb.path == *path {
                            self.current_thumb.category = category.clone();
                        }
                    }
                }
                BarEvent::Removed(paths) => {
                    if self.nav.scope.is_filesystem() {
                        self.viewer.badges.unmark_indexed(paths.iter().map(String::as_str));
                    } else {
                        for path in &paths {
                            self.table.remove_path(path);
                        }
                    }
                }
                BarEvent::Lists { tags, categories, collections } => {
                    let bar = &mut self.toolbar.bar;
                    bar.known_tags = tags;
                    bar.known_categories = categories;
                    bar.collections = collections;
                }
            }
        }
    }

    /// Carries out what a popover asked for.
    fn apply_intent(&mut self, ctx: &Context, intent: Intent) {
        let rows = std::mem::take(&mut self.toolbar.bar.target);
        let refs: Vec<&Thumbnail> = rows.iter().collect();
        let files = batch::files_of(&refs);
        match intent {
            Intent::Close => {}
            Intent::Tags(change) => self.spawn_tags(files, change),
            Intent::Category(category) => self.spawn_category(files, category),
            Intent::Collection { name, add } => self.spawn_collection(files, name, add),
            Intent::Confirmed(action) => {
                let facts = facts_of(&refs, db_ready(), ai_ready());
                self.close_popover();
                let _ = self.run_action_rows_confirmed(ctx, action, rows, &facts);
                return;
            }
        }
        self.close_popover();
    }

    fn run_action_rows_confirmed(&mut self, ctx: &Context, action: ActionId, rows: Vec<Thumbnail>, _facts: &Facts) -> Result<Done, String> {
        self.run_action_rows(ctx, action, rows, Origin::Bar, true)
    }

    /// Runs what the Browser actions devtools window asked for and publishes what it shows while it is open.
    pub(crate) fn run_actions_dev(&mut self, ctx: &Context) {
        use crate::ui::devtools::browser_actions as dev;
        let owner = Id::unique(("ame_browser_actions_dev_owner", self.viewer.scope));
        for command in dev::take_commands(ctx, owner) {
            let dev::Command::Run(action) = command;
            let _ = self.run_action_ui(ctx, action, Origin::Devtools);
        }
        if !dev::is_open(ctx) {
            return;
        }
        let facts = self.selection_facts();
        let rows = SPECS
            .iter()
            .map(|s| dev::Row {
                id: s.id,
                menu_id: s.menu_id,
                group: s.group.label(),
                shortcut: s.shortcut,
                in_bar: s.in_bar,
                confirm: match s.confirm {
                    Confirm::Never => "never",
                    Confirm::Bulk => "above 25",
                    Confirm::Always => "always",
                },
                needs: s.needs.label(),
                count: target_count(s.id, &facts),
                enabled: availability(s.id, &facts),
            })
            .collect();
        let now = Instant::now();
        let history = self
            .toolbar
            .bar
            .runs
            .iter()
            .map(|r| dev::RunRow {
                age: now.saturating_duration_since(r.at),
                action: r.action,
                origin: r.origin.label(),
                count: r.count,
                ok: r.outcome.is_ok(),
                outcome: match &r.outcome {
                    Ok(done) => done.text().to_string(),
                    Err(reason) => reason.clone(),
                },
            })
            .collect();
        let popover = match &self.toolbar.bar.popover {
            None => "none".to_string(),
            Some(p) => format!("{p:?}"),
        };
        dev::publish(ctx, dev::Snapshot { owner, label: self.tab_title(), facts, popover, rows, history });
    }

    /// Draws the action bar over the bottom of `browser` while rows are selected, and the popover of a batch action.
    pub fn show_action_bar(&mut self, ctx: &Context, browser: Rect) {
        self.drain_bar_events();
        let facts = self.selection_facts();
        let bound = self.toolbar.bar.bound;
        if bound.is_some_and(|rev| rev != self.selection.revision()) {
            self.close_popover();
        }
        if facts.selected == 0 {
            self.toolbar.bar.layout = None;
            if self.toolbar.bar.popover.is_some() && bound.is_some() {
                self.close_popover();
            }
            return;
        }
        let layout = self.draw_bar(ctx, browser, &facts);
        let popover_rect = self.show_popover(ctx, layout.rect);
        let mut layout = layout;
        layout.popover = popover_rect;
        self.close_popover_on_outside_press(ctx, &layout);
        self.toolbar.bar.layout = Some(layout);
    }

    fn close_popover_on_outside_press(&mut self, ctx: &Context, layout: &BarLayout) {
        if self.toolbar.bar.popover.is_none() {
            return;
        }
        let (pressed, at, escape) = ctx.input_mut(|i| (i.pointer.any_pressed(), i.pointer.interact_pos(), i.consume_key(Modifiers::NONE, Key::Escape)));
        let inside = at.is_some_and(|p| layout.rect.contains(p) || layout.popover.is_some_and(|r| r.contains(p)));
        if escape || (pressed && !inside) {
            self.close_popover();
        }
    }

    fn draw_bar(&mut self, ctx: &Context, browser: Rect, facts: &Facts) -> BarLayout {
        const MARGIN: f32 = 10.0;
        let style = ctx.global_style();
        let side = style.spacing.interact_size.y.max(icon_md(&style) + 4.0);
        let gap = style.spacing.item_spacing.x;
        let frame_pad = vec2(8.0, 5.0);
        let font = TextStyle::Button.resolve(&style);
        let painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::unique("ame_action_bar_measure")));
        let measure = |text: &str| painter.layout_no_wrap(text.to_string(), font.clone(), Color32::PLACEHOLDER).size().x;
        let button_w = |short: &str| icon_md(&style) + 5.0 + measure(short) + style.spacing.button_padding.x * 2.0 + 4.0;
        let label = count_text(facts);
        let label_w = measure(&label) + gap;
        let max_w = (browser.width() - 2.0 * MARGIN).max(120.0);
        let avail = max_w - 2.0 * frame_pad.x - label_w - side - gap * 2.0;
        let items: Vec<(u8, f32)> = SPECS.iter().filter(|s| s.in_bar).map(|s| (s.priority, button_w(s.short))).collect();
        let shown = plan_bar(&items, gap, side, avail);
        let used: f32 = items.iter().zip(&shown).filter(|(_, on)| **on).map(|(i, _)| i.1 + gap).sum();
        let hidden = shown.iter().filter(|s| !**s).count();
        let more_w = if hidden > 0 { side + gap } else { 0.0 };
        let bar_w = (2.0 * frame_pad.x + label_w + used + more_w + side + gap).min(max_w);
        let bar_h = side + 2.0 * frame_pad.y;
        let pos = pos2(browser.center().x - bar_w / 2.0, browser.bottom() - MARGIN - bar_h);
        let mut clicked: Option<ActionId> = None;
        let mut clear = false;
        let mut layout = BarLayout { browser, hidden, ..Default::default() };
        let id = Id::unique(("ame_action_bar", self.viewer.scope));
        let area = Area::new(id).order(Order::Middle).fixed_pos(pos).constrain(false).show(ctx, |ui| {
            Frame::popup(ui.style()).inner_margin(Margin::symmetric(frame_pad.x as i8, frame_pad.y as i8)).show(ui, |ui| {
                ui.set_width(bar_w - 2.0 * frame_pad.x);
                ui.set_max_height(side);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    let text = ui.add(Label::new(RichText::new(&label).strong()).truncate().selectable(false));
                    layout.label = text.rect;
                    let specs: Vec<&crate::command::actions::ActionSpec> = SPECS.iter().filter(|s| s.in_bar).collect();
                    for (spec, on) in specs.iter().zip(&shown) {
                        if !*on {
                            continue;
                        }
                        let verdict = availability(spec.id, facts);
                        let button = icons::button_widget(ui, spec.icon, spec.short);
                        let response = ui.add_enabled(verdict.is_ok(), button);
                        let tip = match (&verdict, spec.shortcut) {
                            (Ok(()), Some(k)) => format!("{} ({k})", spec.label),
                            (Ok(()), None) => spec.label.to_string(),
                            (Err(reason), _) => format!("{}: {reason}", spec.label),
                        };
                        let response = if verdict.is_ok() { response.on_hover_text(tip) } else { response.on_disabled_hover_text(tip) };
                        layout.buttons.push((spec.id, response.rect, verdict.is_ok()));
                        if response.clicked() {
                            clicked = Some(spec.id);
                        }
                    }
                    if hidden > 0 {
                        let more = icons::icon_menu_button(ui, svg::DOTS_THREE, "More actions", |ui| {
                            for (spec, on) in specs.iter().zip(&shown) {
                                if *on {
                                    continue;
                                }
                                let verdict = availability(spec.id, facts);
                                let response = ui.add_enabled(verdict.is_ok(), icons::button_widget(ui, spec.icon, spec.label));
                                let response = match &verdict {
                                    Ok(()) => response,
                                    Err(reason) => response.on_disabled_hover_text(reason),
                                };
                                if response.clicked() {
                                    clicked = Some(spec.id);
                                    ui.close();
                                }
                            }
                        });
                        layout.more = Some(more.response.rect);
                    }
                    let x = icons::icon_button(ui, svg::X, "Clear the selection (Esc)");
                    layout.clear = x.rect;
                    clear = x.clicked();
                });
            });
        });
        layout.rect = area.response.rect;
        if clear {
            self.clear_selection();
        }
        if let Some(action) = clicked {
            let _ = self.run_action_ui(ctx, action, Origin::Bar);
        }
        layout
    }

    /// Draws the popover of the open batch action above the bar; returns its rectangle.
    fn show_popover(&mut self, ctx: &Context, bar: Rect) -> Option<Rect> {
        let popover = self.toolbar.bar.popover.clone()?;
        let mut forms = std::mem::take(&mut self.toolbar.bar.forms);
        let view = popovers::View { popover: &popover, rows: &self.toolbar.bar.target, known_tags: &self.toolbar.bar.known_tags, known_categories: &self.toolbar.bar.known_categories, collections: &self.toolbar.bar.collections };
        let (rect, intent) = popovers::show(ctx, &mut forms, &view, pos2(bar.center().x, bar.top() - 6.0), Id::unique(("ame_action_popover", self.viewer.scope)));
        self.toolbar.bar.forms = forms;
        if let Some(intent) = intent {
            self.apply_intent(ctx, intent);
        }
        Some(rect)
    }
}

/// Draws the items of the context menu: one per action, grouped, each with the reason it is off.
pub fn context_menu(ui: &mut Ui, targets: MenuTargets, mut emit: impl FnMut(MenuAction)) {
    let facts = Facts::from_counts(targets.rows, targets.dirs, db_ready(), ai_ready());
    ui.label(RichText::new(if targets.rows == 1 { "1 item".to_string() } else { format!("{} items", targets.rows) }).weak());
    for (i, group) in crate::command::actions::Group::ALL.into_iter().enumerate() {
        if i > 0 {
            ui.separator();
        }
        for spec in SPECS.iter().filter(|s| s.group == group) {
            let verdict = availability(spec.id, &facts);
            let mut button = icons::button_widget(ui, spec.icon, spec.label);
            if let Some(key) = spec.shortcut {
                button = button.shortcut_text(key);
            }
            let response = ui.add_enabled(verdict.is_ok(), button);
            let response = match &verdict {
                Ok(()) => response,
                Err(reason) => response.on_disabled_hover_text(reason),
            };
            if response.clicked() {
                emit(MenuAction::from_action(spec.id));
                ui.close();
            }
        }
    }
}

/// Keeps the bar's key handling next to it: the plain keys act on the selection.
pub const KEYS: [(Key, ActionId); 4] = [(Key::T, ActionId::Tag), (Key::C, ActionId::AddToCollection), (Key::D, ActionId::Describe), (Key::Delete, ActionId::RemoveFromLibrary)];
