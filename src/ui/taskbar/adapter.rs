//! The only place the status bar and the task sources read an explorer: swap these functions when the view model changes.

use std::hash::{Hash, Hasher};

use super::view_stats::{FilterChip, FilterKind, Signature, ViewStats};
use crate::UiSettings;
use crate::app::SmartMediaContext;
use crate::tasks::{AppProbe, ExplorerProbe};
use crate::ui::file_table::FileExplorer;

const NAME_MAX_CHARS: usize = 18;

fn hash_of(value: &(impl Hash + ?Sized)) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// Cheap fingerprint of the explorer state the stats depend on; it never walks the rows.
pub fn view_signature(ex: &FileExplorer) -> Signature {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (ex.view.revision(), ex.selection.revision()).hash(&mut h);
    ex.current_path.hash(&mut h);
    ex.viewer.filter.hash(&mut h);
    (ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs).hash(&mut h);
    (ex.viewer.mode as u8, ex.viewer.showing_similarity).hash(&mut h);
    ex.scope().format().hash(&mut h);
    (ex.is_recursive_scan(), ex.recursive_total_filtered(), ex.recursive_total_unfiltered()).hash(&mut h);
    crate::settings::store().generation().hash(&mut h);
    h.finish()
}

/// True while the explorer still receives rows from a scan.
pub fn is_streaming(ex: &FileExplorer) -> bool {
    ex.file_scan_progress > 0.0 && !ex.scan_done
}

/// Stats of the explorer's view as of its last sync; reads no rows.
pub fn compute_view_stats(ex: &FileExplorer) -> ViewStats {
    let cached = crate::ui::file_table::view_stats(ex);
    let rows = cached.rows;
    let (shown, total) = if ex.is_recursive_scan() { (ex.recursive_total_filtered(), ex.recursive_total_unfiltered()) } else { (rows.visible, rows.total) };
    let filters = crate::settings::store().read(|s| {
        active_filters(s, &ex.viewer.filter, ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs)
    });
    ViewStats {
        items: shown,
        hidden: total.saturating_sub(shown),
        images: rows.images,
        videos: rows.videos,
        dirs: rows.dirs,
        total_bytes: rows.total_size,
        selected: cached.selection.count.saturating_sub(cached.selection.dirs),
        selected_bytes: cached.selection.size,
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
        scan_id: ex.scan_id(),
        scan_progress: ex.file_scan_progress,
        scan_done: ex.scan_done,
        failed_thumbs: ex.failed_thumbs.len(),
        archive_prompts: ex.archive_prompt_count(),
    }
}

/// What the task sources need to know about the application this frame.
pub fn app_probe(ctx: &mut SmartMediaContext) -> AppProbe {
    let mut explorers = vec![explorer_probe(&ctx.file_explorer.tab_title(), &ctx.file_explorer)];
    explorers.extend(ctx.filtered_tabs.values().map(|ex| explorer_probe(&ex.tab_title(), ex)));
    AppProbe { explorers, image_edit_cancel: ctx.image_edit.cancel_handle() }
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
    use crate::database::Thumbnail;

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

    #[tokio::test]
    async fn the_signature_moves_with_rows_selection_and_filters_and_not_with_idle_frames() {
        let mut ex = FileExplorer::new(true);
        ex.sync_view();
        let idle = view_signature(&ex);
        ex.sync_view();
        assert_eq!(view_signature(&ex), idle);
        ex.table.push(row("/p/a.jpg", "jpg", 1));
        ex.sync_view();
        let one_row = view_signature(&ex);
        assert_ne!(one_row, idle);
        ex.select_paths(["/p/a.jpg"]);
        ex.sync_view();
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
        ex.sync_view();
        ex.select_paths(["/p/b.png", "/p/sub"]);
        ex.sync_view();
        let stats = compute_view_stats(&ex);
        assert_eq!((stats.images, stats.videos, stats.dirs), (2, 1, 1));
        assert_eq!((stats.selected, stats.selected_bytes), (1, 200), "folders are never counted as selected");
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
