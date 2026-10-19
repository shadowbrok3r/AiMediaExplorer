//! The only place the status bar and the task sources read an explorer: swap these functions when the view model changes.

use std::hash::{Hash, Hasher};

use super::view_stats::{FilterChip, FilterKind, Signature, ViewStats};
use crate::app::SmartMediaContext;
use crate::tasks::{AppProbe, ExplorerProbe};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::{ChipKind, FilterSpec};

/// Cheap fingerprint of the explorer state the stats depend on; it never walks the rows.
pub fn view_signature(ex: &FileExplorer) -> Signature {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (ex.view.revision(), ex.selection.revision()).hash(&mut h);
    ex.current_path.hash(&mut h);
    ex.viewer.filter_signature().hash(&mut h);
    (ex.viewer.mode as u8, ex.viewer.showing_similarity).hash(&mut h);
    ex.scope().format().hash(&mut h);
    (ex.is_recursive_scan(), ex.recursive_total_filtered(), ex.recursive_total_unfiltered()).hash(&mut h);
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
    let filters = active_filters(&ex.filter_spec());
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

/// Removes every filter of the explorer's tab; its rows stay and are listed again.
pub fn clear_filters(ex: &mut FileExplorer) {
    ex.clear_filters();
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

/// Filters currently narrowing the view, including the default minimum size.
pub fn active_filters(spec: &FilterSpec) -> Vec<FilterChip> {
    spec.chips().into_iter().map(|c| FilterChip { kind: filter_kind(&c.kind), label: c.label }).collect()
}

fn filter_kind(kind: &ChipKind) -> FilterKind {
    match kind {
        ChipKind::Size => FilterKind::Size,
        ChipKind::Modified | ChipKind::Created => FilterKind::Dates,
        ChipKind::SkipIcons => FilterKind::SkipIcons,
        ChipKind::Query | ChipKind::Name => FilterKind::Search,
        ChipKind::Types => FilterKind::HiddenTypes,
        _ => FilterKind::Other,
    }
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
    fn the_default_spec_shows_the_hidden_minimum_size_filter() {
        let chips = active_filters(&FilterSpec::default());
        assert_eq!(chips, vec![FilterChip { kind: FilterKind::Size, label: "Size >= 10 KiB".into() }]);
    }

    #[test]
    fn filters_cover_every_active_narrowing_and_none_when_clear() {
        let mut s = FilterSpec::none();
        assert!(active_filters(&s).is_empty());
        s.size.max = Some(5 * 1024 * 1024);
        s.modified.after = Some("2024-01-01".into());
        s.skip_icons = true;
        s.query = "cat".into();
        s.types.images = false;
        s.excluded_terms.push("tmp".into());
        let chips = active_filters(&s);
        let kinds: Vec<_> = chips.iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![FilterKind::Search, FilterKind::HiddenTypes, FilterKind::Size, FilterKind::Dates, FilterKind::SkipIcons, FilterKind::Other]);
        assert_eq!(chips[1].label, "Hiding: Images");
        assert_eq!(chips[2].label, "Size <= 5 MiB");
        assert_eq!(chips[3].label, "Modified since 2024-01-01");
        assert_eq!(chips[5].label, "Not matching: tmp");
    }

    #[test]
    fn a_long_search_is_shortened_in_its_label() {
        let mut s = FilterSpec::none();
        s.query = "a very long search phrase indeed".into();
        assert_eq!(active_filters(&s)[0].label, "Search: a very long search ph...");
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
    async fn clearing_the_filters_resets_the_spec_and_keeps_every_row() {
        let mut ex = FileExplorer::new(true);
        for r in sample_rows() {
            ex.table.push(r);
        }
        ex.viewer.filter = "cat".into();
        ex.viewer.types_show_images = false;
        ex.edit_filters(|s| s.size.min = Some(10 * 1024));
        ex.sync_view();
        assert_eq!(ex.view.len(), 0);
        clear_filters(&mut ex);
        ex.sync_view();
        assert!(ex.viewer.filter.is_empty() && ex.viewer.types_show_images);
        assert!(ex.filter_spec().is_empty());
        assert_eq!((ex.table.len(), ex.view.len()), (5, 5));
    }

    #[tokio::test]
    async fn the_bar_chips_come_from_the_tab_spec() {
        let mut ex = FileExplorer::new(true);
        ex.edit_filters(|s| {
            s.excluded_terms.push("tmp".into());
            s.tags.push("sun".into());
        });
        let labels: Vec<String> = compute_view_stats(&ex).filters.into_iter().map(|c| c.label).collect();
        assert_eq!(labels, vec!["Tag: sun".to_string(), "Not matching: tmp".to_string()]);
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
