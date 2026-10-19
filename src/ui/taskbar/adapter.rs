//! The only place the status bar and the task sources read an explorer: swap these functions when the view model changes.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use super::view_stats::{FilterChip, FilterKind, Signature, ViewStats};
use crate::UiSettings;
use crate::app::SmartMediaContext;
use crate::database::Thumbnail;
use crate::tasks::{AppProbe, ExplorerProbe};
use crate::ui::file_table::FileExplorer;

/// Selections up to this size are hashed in full for the change signature.
const SELECTION_HASH_LIMIT: usize = 256;
const NAME_MAX_CHARS: usize = 18;

fn hash_of(value: &(impl Hash + ?Sized)) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// Cheap fingerprint of the explorer state the stats depend on; it never walks the rows.
pub fn view_signature(ex: &FileExplorer) -> Signature {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ex.table.len().hash(&mut h);
    ex.table.first().map(|r| r.path.as_str()).hash(&mut h);
    ex.table.last().map(|r| r.path.as_str()).hash(&mut h);
    ex.current_path.hash(&mut h);
    ex.viewer.filter.hash(&mut h);
    (ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs).hash(&mut h);
    (ex.viewer.mode as u8, ex.viewer.showing_similarity).hash(&mut h);
    (ex.is_recursive_scan(), ex.recursive_total_filtered(), ex.recursive_total_unfiltered()).hash(&mut h);
    selection_signature(&ex.viewer.selected, &ex.current_thumb.path).hash(&mut h);
    crate::settings::store().generation().hash(&mut h);
    h.finish()
}

/// Order-independent hash of a small selection; for a large one its size and the focused path.
fn selection_signature(selected: &HashSet<String>, focus: &str) -> u64 {
    if selected.len() <= SELECTION_HASH_LIMIT {
        selected.iter().fold(selected.len() as u64, |acc, p| acc ^ hash_of(p.as_str()))
    } else {
        hash_of(&(selected.len(), focus))
    }
}

/// True while the explorer still receives rows from a scan.
pub fn is_streaming(ex: &FileExplorer) -> bool {
    ex.file_scan_progress > 0.0 && !ex.scan_done
}

/// Stats of the explorer's view; walks the rows once, so callers go through the cache.
pub fn compute_view_stats(ex: &FileExplorer) -> ViewStats {
    let tally = tally(ex.table.iter(), |r| ex.viewer.row_passes_filter(r), &ex.viewer.selected);
    let (shown, total) = if ex.is_recursive_scan() { (ex.recursive_total_filtered(), ex.recursive_total_unfiltered()) } else { (tally.visible, ex.table.len()) };
    let filters = crate::settings::store().read(|s| {
        active_filters(s, &ex.viewer.filter, ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs)
    });
    ViewStats {
        items: shown,
        hidden: total.saturating_sub(shown),
        images: tally.images,
        videos: tally.videos,
        dirs: tally.dirs,
        total_bytes: tally.total_bytes,
        selected: tally.selected,
        selected_bytes: tally.selected_bytes,
        filters,
    }
}

/// Removes every filter the stats list: the settings filters and the explorer's search and type toggles.
pub fn clear_filters(ex: &mut FileExplorer) {
    crate::settings::update("taskbar:clear-filters", |s| {
        s.db_min_size_bytes = None;
        s.db_max_size_bytes = None;
        s.filter_modified_after = None;
        s.filter_modified_before = None;
        s.filter_skip_icons = false;
    });
    ex.viewer.filter.clear();
    ex.viewer.types_show_images = true;
    ex.viewer.types_show_videos = true;
    ex.viewer.types_show_dirs = true;
}

/// What one explorer tab tells the task sources.
pub fn explorer_probe(tab: &str, ex: &FileExplorer) -> ExplorerProbe {
    ExplorerProbe {
        tab: tab.to_string(),
        scan_id: None,
        scan_progress: ex.file_scan_progress,
        scan_done: ex.scan_done,
        failed_thumbs: ex.failed_thumbs.len(),
        archive_prompts: 0,
    }
}

/// What the task sources need to know about the application this frame.
pub fn app_probe(ctx: &mut SmartMediaContext) -> AppProbe {
    let mut explorers = vec![explorer_probe(crate::app::MAIN_EXPLORER_TAB, &ctx.file_explorer)];
    explorers.extend(ctx.filtered_tabs.iter().map(|(title, ex)| explorer_probe(title, ex)));
    AppProbe { explorers, image_edit_cancel: ctx.image_edit.cancel_handle() }
}

/// Counts of one pass over the rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub visible: usize,
    pub images: usize,
    pub videos: usize,
    pub dirs: usize,
    pub total_bytes: u64,
    pub selected: usize,
    pub selected_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Image,
    Video,
    Other,
}

fn kind_of(path: &str) -> Kind {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let Some((stem, ext)) = name.rsplit_once('.') else { return Kind::Other };
    if stem.is_empty() || ext.is_empty() || ext.len() > 8 {
        return Kind::Other;
    }
    let mut buf = [0u8; 8];
    for (slot, b) in buf.iter_mut().zip(ext.bytes()) {
        *slot = b.to_ascii_lowercase();
    }
    match std::str::from_utf8(&buf[..ext.len()]) {
        Ok(lower) if crate::is_image(lower) => Kind::Image,
        Ok(lower) if crate::is_video(lower) => Kind::Video,
        _ => Kind::Other,
    }
}

/// One pass over `rows`: `passes` is the view filter; selected rows are counted whether or not they pass it.
pub fn tally<'a>(rows: impl Iterator<Item = &'a Thumbnail>, passes: impl Fn(&Thumbnail) -> bool, selected: &HashSet<String>) -> Tally {
    let mut t = Tally::default();
    for r in rows {
        let is_dir = r.file_type == "<DIR>";
        let kind = if is_dir { Kind::Other } else { kind_of(&r.path) };
        if passes(r) {
            t.visible += 1;
            if is_dir {
                t.dirs += 1;
            } else {
                t.total_bytes = t.total_bytes.saturating_add(r.size);
                match kind {
                    Kind::Image => t.images += 1,
                    Kind::Video => t.videos += 1,
                    Kind::Other => {}
                }
            }
        }
        if !selected.is_empty() && !is_dir && selected.contains(&r.path) {
            t.selected += 1;
            t.selected_bytes = t.selected_bytes.saturating_add(r.size);
        }
    }
    t
}

fn fmt_bytes(b: u64) -> String {
    humansize::format_size(b, humansize::BINARY)
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
}

/// Filters currently narrowing the view, including the default minimum size.
pub fn active_filters(s: &UiSettings, search: &str, show_images: bool, show_videos: bool, show_dirs: bool) -> Vec<FilterChip> {
    let mut chips = Vec::new();
    let size = match (s.db_min_size_bytes, s.db_max_size_bytes) {
        (Some(a), Some(b)) => Some(format!("Size {} - {}", fmt_bytes(a), fmt_bytes(b))),
        (Some(a), None) => Some(format!("Size >= {}", fmt_bytes(a))),
        (None, Some(b)) => Some(format!("Size <= {}", fmt_bytes(b))),
        (None, None) => None,
    };
    if let Some(label) = size {
        chips.push(FilterChip { kind: FilterKind::Size, label });
    }
    let dates = match (s.filter_modified_after.as_deref(), s.filter_modified_before.as_deref()) {
        (Some(a), Some(b)) => Some(format!("{a} to {b}")),
        (Some(a), None) => Some(format!("Since {a}")),
        (None, Some(b)) => Some(format!("Until {b}")),
        (None, None) => None,
    };
    if let Some(label) = dates {
        chips.push(FilterChip { kind: FilterKind::Dates, label });
    }
    if s.filter_skip_icons {
        chips.push(FilterChip { kind: FilterKind::SkipIcons, label: "Skip icons".into() });
    }
    if !search.trim().is_empty() {
        chips.push(FilterChip { kind: FilterKind::Search, label: format!("Search: {}", truncate_chars(search.trim(), NAME_MAX_CHARS)) });
    }
    let hidden: Vec<&str> = [(!show_images, "Images"), (!show_videos, "Videos"), (!show_dirs, "Folders")]
        .into_iter()
        .filter_map(|(hide, name)| hide.then_some(name))
        .collect();
    if !hidden.is_empty() {
        chips.push(FilterChip { kind: FilterKind::HiddenTypes, label: format!("Hiding: {}", hidden.join(", ")) });
    }
    chips
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, kind: &str, size: u64) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit('/').next().unwrap_or(path).into();
        t.file_type = kind.into();
        t.size = size;
        t
    }

    fn sample_rows() -> Vec<Thumbnail> {
        vec![
            row("/p/a.JPG", "jpg", 100),
            row("/p/b.png", "png", 200),
            row("/p/c.mp4", "mp4", 4000),
            row("/p/notes.txt", "txt", 7),
            row("/p/sub", "<DIR>", 0),
        ]
    }

    #[test]
    fn kind_detection_ignores_case_and_dotfiles() {
        assert_eq!(kind_of("/x/A.JPG"), Kind::Image);
        assert_eq!(kind_of("C:\\x\\clip.MP4"), Kind::Video);
        assert_eq!(kind_of("/x/.png"), Kind::Other);
        assert_eq!(kind_of("/x/noext"), Kind::Other);
        assert_eq!(kind_of("/x/file."), Kind::Other);
        assert_eq!(kind_of("/x/a.verylongextension"), Kind::Other);
    }

    #[test]
    fn one_pass_counts_kinds_sizes_and_the_selection() {
        let rows = sample_rows();
        let selected: HashSet<String> = ["/p/b.png".to_string(), "/p/c.mp4".to_string(), "/p/sub".to_string()].into();
        let t = tally(rows.iter(), |_| true, &selected);
        assert_eq!((t.visible, t.images, t.videos, t.dirs, t.total_bytes), (5, 2, 1, 1, 4307));
        assert_eq!((t.selected, t.selected_bytes), (2, 4200), "folders are never counted as selected");
    }

    #[test]
    fn the_view_filter_limits_the_visible_counts_but_not_the_selection() {
        let rows = sample_rows();
        let selected: HashSet<String> = ["/p/c.mp4".to_string()].into();
        let t = tally(rows.iter(), |r| r.file_type != "mp4", &selected);
        assert_eq!((t.visible, t.videos), (4, 0));
        assert_eq!(t.selected, 1);
    }

    #[test]
    fn the_default_settings_show_the_hidden_minimum_size_filter() {
        let chips = active_filters(&UiSettings::default(), "", true, true, true);
        assert_eq!(chips, vec![FilterChip { kind: FilterKind::Size, label: "Size >= 10 KiB".into() }]);
    }

    #[test]
    fn filters_cover_every_active_narrowing_and_none_when_clear() {
        let mut s = UiSettings::default();
        s.db_min_size_bytes = None;
        assert!(active_filters(&s, "", true, true, true).is_empty());
        s.db_max_size_bytes = Some(5 * 1024 * 1024);
        s.filter_modified_after = Some("2024-01-01".into());
        s.filter_skip_icons = true;
        let chips = active_filters(&s, "cat", false, true, true);
        let kinds: Vec<_> = chips.iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![FilterKind::Size, FilterKind::Dates, FilterKind::SkipIcons, FilterKind::Search, FilterKind::HiddenTypes]);
        assert_eq!(chips[0].label, "Size <= 5 MiB");
        assert_eq!(chips[1].label, "Since 2024-01-01");
        assert_eq!(chips[4].label, "Hiding: Images");
    }

    #[test]
    fn a_long_search_is_shortened_in_its_label() {
        let chips = active_filters(&UiSettings { db_min_size_bytes: None, ..Default::default() }, "a very long search phrase indeed", true, true, true);
        assert_eq!(chips[0].label, "Search: a very long sea...");
    }

    #[test]
    fn the_selection_signature_sees_swapped_members_of_equal_size() {
        let a: HashSet<String> = ["/a".to_string()].into();
        let b: HashSet<String> = ["/b".to_string()].into();
        assert_ne!(selection_signature(&a, ""), selection_signature(&b, ""));
        assert_eq!(selection_signature(&a, ""), selection_signature(&a.clone(), ""));
        assert_ne!(selection_signature(&HashSet::new(), ""), selection_signature(&a, ""));
    }

    #[test]
    fn a_large_selection_is_fingerprinted_without_hashing_every_member() {
        let big: HashSet<String> = (0..1000).map(|i| format!("/p/{i}")).collect();
        let mut more = big.clone();
        more.insert("/extra".into());
        assert_ne!(selection_signature(&big, "/f"), selection_signature(&more, "/f"));
        assert_ne!(selection_signature(&big, "/f"), selection_signature(&big, "/g"));
    }

    #[tokio::test]
    async fn the_signature_moves_with_rows_selection_and_filters_and_not_with_idle_frames() {
        let mut ex = FileExplorer::new(true);
        let idle = view_signature(&ex);
        assert_eq!(view_signature(&ex), idle);
        ex.table.push(row("/p/a.jpg", "jpg", 1));
        let one_row = view_signature(&ex);
        assert_ne!(one_row, idle);
        ex.viewer.selected.insert("/p/a.jpg".into());
        let selected = view_signature(&ex);
        assert_ne!(selected, one_row);
        ex.viewer.filter = "cat".into();
        assert_ne!(view_signature(&ex), selected);
    }

    #[tokio::test]
    async fn the_computed_stats_follow_the_explorer() {
        let mut ex = FileExplorer::new(true);
        for r in sample_rows() {
            ex.table.push(r);
        }
        ex.viewer.selected.insert("/p/b.png".into());
        let stats = compute_view_stats(&ex);
        assert_eq!((stats.images, stats.videos, stats.dirs), (2, 1, 1));
        assert_eq!((stats.selected, stats.selected_bytes), (1, 200));
        assert_eq!(stats.items + stats.hidden, ex.table.len());
    }

    #[tokio::test]
    async fn clearing_the_filters_resets_the_search_and_the_type_toggles() {
        let mut ex = FileExplorer::new(true);
        ex.viewer.filter = "cat".into();
        ex.viewer.types_show_images = false;
        crate::settings::store().update("test:seed", |s| {
            s.db_min_size_bytes = Some(10 * 1024);
            s.filter_skip_icons = true;
        });
        clear_filters(&mut ex);
        assert!(ex.viewer.filter.is_empty() && ex.viewer.types_show_images);
        assert_eq!(crate::settings::store().read(|s| (s.db_min_size_bytes, s.filter_skip_icons)), (None, false));
    }

    #[tokio::test]
    async fn the_probe_reports_scan_state_and_failed_thumbnails() {
        let mut ex = FileExplorer::new(true);
        ex.file_scan_progress = 0.5;
        ex.failed_thumbs.push(("/a".into(), "bad".into()));
        let probe = explorer_probe("Tab", &ex);
        assert_eq!((probe.tab.as_str(), probe.scan_progress, probe.scan_done, probe.failed_thumbs), ("Tab", 0.5, false, 1));
        assert!(is_streaming(&ex));
    }
}
