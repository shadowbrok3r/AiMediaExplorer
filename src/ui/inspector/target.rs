//! What the inspector shows: derived from the focused browser's selection, plus the logic that follows the focus.

use std::collections::HashMap;

use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::key::{KeyPolicy, paths_equivalent};
use crate::ui::file_table::rows::thumbnail::DIR_TYPE;
use crate::ui::file_table::rows::{PathKey, Step};

use super::tags::TagTally;

/// Selections larger than this ask before starting AI work on every file.
pub const CONFIRM_ABOVE: usize = 25;
/// Folders skipped in a row before navigation gives up.
const NAVIGATE_LIMIT: usize = 256;

/// The thing the rail is about.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// Nothing is selected; never stands for "everything".
    Empty,
    /// One row is selected: a file or a folder.
    Single(Box<Thumbnail>),
    /// Several rows are selected; `focus` is the row the preview follows.
    Multi { count: usize, focus: Option<Box<Thumbnail>> },
}

impl Target {
    /// Path of the row the preview and the editor follow.
    pub fn focus_path(&self) -> Option<&str> {
        match self {
            Target::Empty => None,
            Target::Single(row) => Some(row.path.as_str()),
            Target::Multi { focus, .. } => focus.as_ref().map(|r| r.path.as_str()),
        }
    }

    /// The row that is edited: a single selected file.
    pub fn editable(&self) -> Option<&Thumbnail> {
        match self {
            Target::Single(row) if !is_dir(row) => Some(row),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Target::Empty => "empty".to_string(),
            Target::Single(row) if is_dir(row) => format!("folder {}", row.path),
            Target::Single(row) => format!("file {}", row.path),
            Target::Multi { count, .. } => format!("{count} selected"),
        }
    }
}

pub fn is_dir(row: &Thumbnail) -> bool {
    row.file_type == DIR_TYPE
}

/// Reads the selection of `ex` into a target; the focus drives it, not the order of clicks.
pub fn resolve(ex: &FileExplorer) -> Target {
    let count = ex.selection.len();
    let focus = ex.selection.preview_target().and_then(|k| ex.table.get_by_key(k));
    match (count, focus) {
        (0, _) => Target::Empty,
        (1, Some(row)) => Target::Single(Box::new(row.clone())),
        (_, None) if count == 1 => Target::Empty,
        _ => Target::Multi { count, focus: focus.map(|r| Box::new(r.clone())) },
    }
}

/// The interim text to show for `focus`: only the focused file's, never another file's.
pub fn stream_view<'a>(focus: Option<&str>, interim: &'a HashMap<String, String>) -> Option<&'a str> {
    let focus = focus?;
    if let Some(text) = interim.get(focus) {
        return Some(text.as_str());
    }
    interim.iter().find(|(path, _)| paths_equivalent(path, focus, KeyPolicy::NATIVE)).map(|(_, text)| text.as_str())
}

/// Numbers over the selected rows for the batch editor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchSummary {
    pub files: usize,
    pub dirs: usize,
    pub images: usize,
    pub videos: usize,
    pub size: u64,
    pub tally: TagTally,
    /// Categories with the number of rows that have them, most used first.
    pub categories: Vec<(String, usize)>,
    pub with_description: usize,
    /// Rows no category is set on.
    pub uncategorized: usize,
}

impl BatchSummary {
    pub fn of(rows: &[&Thumbnail]) -> Self {
        let files: Vec<&&Thumbnail> = rows.iter().filter(|r| !is_dir(r)).collect();
        let mut categories: HashMap<String, usize> = HashMap::new();
        let mut images = 0;
        let mut videos = 0;
        let mut size = 0u64;
        let mut with_description = 0;
        let mut uncategorized = 0;
        for r in &files {
            match category_of(r) {
                Some(c) => *categories.entry(c).or_default() += 1,
                None => uncategorized += 1,
            }
            if has_description(r) {
                with_description += 1;
            }
            size = size.saturating_add(r.size);
            let ext = std::path::Path::new(&r.path).extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default();
            if crate::is_image(&ext) {
                images += 1;
            } else if crate::is_video(&ext) {
                videos += 1;
            }
        }
        let mut categories: Vec<(String, usize)> = categories.into_iter().collect();
        categories.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));
        Self {
            files: files.len(),
            dirs: rows.len() - files.len(),
            images,
            videos,
            size,
            tally: TagTally::of(files.iter().map(|r| r.tags.as_slice())),
            categories,
            with_description,
            uncategorized,
        }
    }
}

fn category_of(row: &Thumbnail) -> Option<String> {
    row.category.as_deref().map(str::trim).filter(|c| !c.is_empty()).map(str::to_string)
}

pub fn has_description(row: &Thumbnail) -> bool {
    row.description.as_deref().is_some_and(|d| !d.trim().is_empty())
}

/// Position of the focus in the view, 1-based, and the number of rows.
pub fn position(ex: &FileExplorer) -> Option<(usize, usize)> {
    let key = ex.selection.preview_target()?;
    let view = ex.view_ref();
    Some((view.pos_of(key)? + 1, view.len()))
}

/// Moves the focus to the next (`forward`) or previous file of the view, skipping folders; stays put at the ends.
pub fn navigate(ex: &mut FileExplorer, forward: bool) -> Option<PathKey> {
    let start = ex.selection.preview_target().cloned()?;
    let step = if forward { Step::Down(1) } else { Step::Up(1) };
    let mut last = start.clone();
    for _ in 0..NAVIGATE_LIMIT {
        let Some(moved) = ex.step_focus(step, false) else { break };
        if moved == last {
            break;
        }
        last = moved.clone();
        if ex.table.get_by_key(&moved).is_some_and(|r| !is_dir(r)) {
            if let Some(pos) = ex.view_ref().pos_of(&moved) {
                ex.details.scroll_to(pos);
            }
            return Some(moved);
        }
    }
    if ex.selection.preview_target() != Some(&start) {
        ex.select_paths([start.as_str()]);
    }
    Some(start)
}

/// Whether the vision model is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiState {
    Ready,
    /// No model folder is chosen or it does not exist.
    NotConfigured,
}

/// AI state from the chosen model folder; the built-in default folder counts when it exists.
pub fn ai_state(chosen: Option<&str>, default_dir: &str, exists: impl Fn(&std::path::Path) -> bool) -> AiState {
    let chosen = chosen.map(str::trim).filter(|d| !d.is_empty());
    let dir = chosen.unwrap_or(default_dir);
    if exists(std::path::Path::new(dir)) { AiState::Ready } else { AiState::NotConfigured }
}

/// AI state from the live settings.
pub fn current_ai_state() -> AiState {
    let settings = crate::settings::snapshot();
    ai_state(settings.joycaption_model_dir.as_deref(), crate::app::DEFAULT_JOYCAPTION_PATH, |p| p.is_dir())
}

/// Whether the file can be described by the vision model.
pub fn can_describe(row: &Thumbnail) -> bool {
    !is_dir(row) && !row.path.contains("://") && std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_image(&e.to_ascii_lowercase()))
}

pub fn is_video_path(path: &str) -> bool {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_video(&e.to_ascii_lowercase()))
}

pub fn is_image_path(path: &str) -> bool {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_image(&e.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, ty: &str) -> Thumbnail {
        let mut t = Thumbnail::new(path.rsplit('/').next().unwrap());
        t.path = path.into();
        t.file_type = ty.into();
        t
    }

    fn explorer(rows: Vec<Thumbnail>) -> FileExplorer {
        let mut ex = FileExplorer::new(true);
        ex.table.bulk_load(rows);
        ex.sync_view();
        ex
    }

    fn five() -> Vec<Thumbnail> {
        vec![row("/d/a.jpg", "jpg"), row("/d/sub", "<DIR>"), row("/d/b.jpg", "jpg"), row("/d/c.mp4", "mp4"), row("/d/d.png", "png")]
    }

    #[tokio::test]
    async fn the_target_follows_the_selection_and_never_means_everything() {
        let mut ex = explorer(five());
        assert_eq!(resolve(&ex), Target::Empty, "an unselected table is not a selection of everything");
        ex.select_paths(["/d/a.jpg"]);
        assert!(matches!(resolve(&ex), Target::Single(r) if r.path == "/d/a.jpg"));
        ex.select_paths(["/d/a.jpg", "/d/b.jpg", "/d/d.png"]);
        match resolve(&ex) {
            Target::Multi { count, focus } => {
                assert_eq!(count, 3);
                assert!(focus.is_some());
            }
            other => panic!("{other:?}"),
        }
        ex.clear_selection();
        assert_eq!(resolve(&ex), Target::Empty);
    }

    #[tokio::test]
    async fn a_folder_is_a_single_target_that_cannot_be_edited() {
        let mut ex = explorer(five());
        ex.select_paths(["/d/sub"]);
        let t = resolve(&ex);
        assert!(matches!(&t, Target::Single(r) if is_dir(r)));
        assert!(t.editable().is_none());
        assert_eq!(t.focus_path(), Some("/d/sub"));
        ex.select_paths(["/d/b.jpg"]);
        assert_eq!(resolve(&ex).editable().map(|r| r.path.as_str()), Some("/d/b.jpg"));
    }

    #[tokio::test]
    async fn the_focus_not_the_last_click_picks_the_previewed_row_of_a_multi_selection() {
        let mut ex = explorer(five());
        ex.select_paths(["/d/a.jpg", "/d/b.jpg", "/d/d.png"]);
        let focus = ex.selection.focus().cloned();
        let t = resolve(&ex);
        let expected = ex.selection.preview_target().map(|k| k.as_str().to_string());
        assert_eq!(t.focus_path().map(str::to_string), expected);
        assert!(focus.is_some() || expected.is_some());
    }

    #[test]
    fn only_the_focused_file_shows_streaming_text() {
        let mut interim = HashMap::new();
        interim.insert("/d/a.jpg".to_string(), "A cat sitting".to_string());
        interim.insert("/d/b.jpg".to_string(), "A dog".to_string());
        assert_eq!(stream_view(Some("/d/a.jpg"), &interim), Some("A cat sitting"));
        assert_eq!(stream_view(Some("/d/b.jpg"), &interim), Some("A dog"));
        assert_eq!(stream_view(Some("/d/c.jpg"), &interim), None, "another file's stream is not shown");
        assert_eq!(stream_view(None, &interim), None, "no focus, no stream");
        interim.remove("/d/a.jpg");
        assert_eq!(stream_view(Some("/d/a.jpg"), &interim), None, "a finished stream disappears");
    }

    #[tokio::test]
    async fn a_focus_change_hides_the_stale_stream_and_coming_back_shows_it_again() {
        let mut ex = explorer(five());
        ex.streaming_interim.insert("/d/a.jpg".into(), "streaming...".into());
        ex.select_paths(["/d/a.jpg"]);
        assert_eq!(stream_view(resolve(&ex).focus_path(), &ex.streaming_interim), Some("streaming..."));
        ex.select_paths(["/d/b.jpg"]);
        assert_eq!(stream_view(resolve(&ex).focus_path(), &ex.streaming_interim), None);
        ex.select_paths(["/d/a.jpg"]);
        assert_eq!(stream_view(resolve(&ex).focus_path(), &ex.streaming_interim), Some("streaming..."));
    }

    #[tokio::test]
    async fn navigation_steps_over_folders_and_stops_at_the_ends() {
        let mut ex = explorer(five());
        ex.select_paths(["/d/a.jpg"]);
        let path = |ex: &FileExplorer| resolve(ex).focus_path().map(str::to_string);
        assert_eq!(position(&ex), Some((1, 5)));
        navigate(&mut ex, true);
        assert_eq!(path(&ex).as_deref(), Some("/d/b.jpg"), "the folder between is skipped");
        navigate(&mut ex, true);
        assert_eq!(path(&ex).as_deref(), Some("/d/c.mp4"));
        navigate(&mut ex, true);
        navigate(&mut ex, true);
        assert_eq!(path(&ex).as_deref(), Some("/d/d.png"), "the last file stays");
        navigate(&mut ex, false);
        assert_eq!(path(&ex).as_deref(), Some("/d/c.mp4"));
        navigate(&mut ex, false);
        navigate(&mut ex, false);
        navigate(&mut ex, false);
        assert_eq!(path(&ex).as_deref(), Some("/d/a.jpg"), "the first file stays");
        assert_eq!(ex.selection.len(), 1, "navigating selects exactly the new focus");
    }

    #[tokio::test]
    async fn the_batch_summary_counts_files_sizes_and_tags_and_ignores_folders() {
        let mut a = row("/d/a.jpg", "jpg");
        a.size = 100;
        a.tags = vec!["cat".into(), "pet".into()];
        a.category = Some("animals".into());
        a.description = Some("a cat".into());
        let mut b = row("/d/b.png", "png");
        b.size = 50;
        b.tags = vec!["Cat".into()];
        b.category = Some("animals".into());
        let mut c = row("/d/c.mp4", "mp4");
        c.size = 1000;
        let dir = row("/d/sub", "<DIR>");
        let rows = vec![&a, &b, &c, &dir];
        let s = BatchSummary::of(&rows);
        assert_eq!((s.files, s.dirs, s.images, s.videos, s.size), (3, 1, 2, 1, 1150));
        assert_eq!(s.categories, vec![("animals".to_string(), 2)]);
        assert_eq!((s.with_description, s.uncategorized), (1, 1));
        assert_eq!(s.tally.rows, 3);
        assert!(s.tally.common().is_empty(), "the video has no tags");
        assert_eq!(s.tally.partial().iter().map(|t| (t.tag.as_str(), t.count)).collect::<Vec<_>>(), vec![("cat", 2), ("pet", 1)]);
        assert_eq!(BatchSummary::of(&[]), BatchSummary::default());
    }

    #[test]
    fn ai_is_ready_when_a_model_folder_exists() {
        let exists = |p: &std::path::Path| p.to_str() == Some("/models/joy");
        assert_eq!(ai_state(Some("/models/joy"), "/default", exists), AiState::Ready);
        assert_eq!(ai_state(Some("/nowhere"), "/models/joy", exists), AiState::NotConfigured, "a chosen folder replaces the default");
        assert_eq!(ai_state(None, "/models/joy", exists), AiState::Ready);
        assert_eq!(ai_state(Some("  "), "/models/joy", exists), AiState::Ready, "a blank choice falls back to the default");
        assert_eq!(ai_state(None, "/nowhere", exists), AiState::NotConfigured);
    }

    #[test]
    fn describing_needs_a_local_picture() {
        assert!(can_describe(&row("/d/a.jpg", "jpg")));
        assert!(!can_describe(&row("/d/a.mp4", "mp4")));
        assert!(!can_describe(&row("/d/sub", "<DIR>")));
        assert!(!can_describe(&row("zip:///a.zip!/x.jpg", "jpg")));
        assert!(is_video_path("/x/y.MP4") && !is_video_path("/x/y.jpg"));
        assert!(is_image_path("/x/y.JPG"));
    }
}
