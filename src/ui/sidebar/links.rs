//! Sidebar entries as plain values: each link names a scope and the command a click sends.

use humansize::{DECIMAL, format_size};

use crate::command::{AppCommand, Scope, SmartView, Target};
use crate::icons::{self, Icon, svg};
use crate::utilities::explorer::DriveInfo;
use crate::utilities::types::QuickAccess;

/// Tags or categories shown before "show more".
pub const TOP_N: usize = 12;
/// Most tags or categories listed at once, filtered or expanded.
pub const MAX_SHOWN: usize = 100;
/// Recent folders listed under Places.
pub const MAX_RECENT: usize = 8;

/// One clickable sidebar row.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub key: String,
    pub label: String,
    /// Small right-aligned text such as a count.
    pub detail: String,
    pub tooltip: String,
    pub icon: Icon,
    pub scope: Scope,
}

impl Link {
    fn new(key: impl Into<String>, label: impl Into<String>, icon: Icon, scope: Scope) -> Self {
        let label = label.into();
        Self { key: key.into(), tooltip: label.clone(), label, detail: String::new(), icon, scope }
    }

    fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    fn tip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = tooltip.into();
        self
    }

    pub fn is_folder(&self) -> bool {
        matches!(self.scope, Scope::Folder(_))
    }

    /// The command a click sends.
    pub fn command(&self, target: Target, recursive: bool) -> AppCommand {
        AppCommand::OpenScope { scope: self.scope.clone(), target, title: None, recursive: recursive && self.is_folder() }
    }

    /// True when the focused browser shows this link's scope.
    pub fn is_active(&self, active: Option<&Scope>) -> bool {
        active.is_some_and(|a| same_scope(a, &self.scope))
    }
}

/// Scope equality that ignores a trailing separator on folders.
pub fn same_scope(a: &Scope, b: &Scope) -> bool {
    match (a, b) {
        (Scope::Folder(x), Scope::Folder(y)) => trim_sep(x) == trim_sep(y),
        _ => a == b,
    }
}

fn trim_sep(path: &str) -> &str {
    let trimmed = path.trim_end_matches(['/', '\\']);
    if trimmed.is_empty() { path } else { trimmed }
}

/// Click on a link: the target of the open and whether it scans recursively.
pub fn click_target(ctrl: bool, middle: bool, shift: bool) -> (Target, bool) {
    let target = if ctrl || middle { Target::NewTab } else { Target::Current };
    (target, shift)
}

/// Places: user folders, drives and recent folders.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlacesModel {
    pub quick: Vec<Link>,
    pub drives: Vec<Link>,
    pub wsl: Vec<Link>,
    pub recent: Vec<Link>,
}

pub fn places_links(quick: &[QuickAccess], drives: &[DriveInfo], wsl: &[QuickAccess], recent: &[String]) -> PlacesModel {
    let quick_links: Vec<Link> = quick
        .iter()
        .map(|q| {
            let path = q.path.to_string_lossy().to_string();
            Link::new(format!("place:{path}"), q.label.clone(), icons::folder_shortcut_svg(&q.label), Scope::Folder(path.clone())).tip(path)
        })
        .collect();
    let drive_links = drives
        .iter()
        .map(|d| {
            let label = if d.label.trim().is_empty() { d.root.clone() } else { d.label.clone() };
            let free = format_size(d.free, DECIMAL);
            let total = format_size(d.total, DECIMAL);
            Link::new(format!("drive:{}", d.root), label, svg::HARD_DRIVES, Scope::Folder(d.root.clone()))
                .detail(format!("{free} free"))
                .tip(format!("{} ({}): {free} free of {total}", d.root, d.drive_type))
        })
        .collect();
    let known: Vec<&str> = quick_links.iter().filter_map(|l| folder_path(&l.scope)).collect();
    let recent_links = recent
        .iter()
        .filter(|p| !p.trim().is_empty() && !known.iter().any(|k| trim_sep(k) == trim_sep(p)))
        .take(MAX_RECENT)
        .map(|p| {
            let scope = Scope::Folder(p.clone());
            Link::new(format!("recent:{p}"), scope.default_title(), svg::CLOCK_COUNTER_CLOCKWISE, scope).tip(p.clone())
        })
        .collect();
    let wsl_links = wsl
        .iter()
        .map(|w| {
            let path = w.path.to_string_lossy().to_string();
            Link::new(format!("wsl:{path}"), w.label.clone(), svg::LINUX_LOGO, Scope::Folder(path.clone())).tip(path)
        })
        .collect();
    PlacesModel { quick: quick_links, drives: drive_links, wsl: wsl_links, recent: recent_links }
}

fn folder_path(scope: &Scope) -> Option<&str> {
    match scope {
        Scope::Folder(p) => Some(p),
        _ => None,
    }
}

/// Library: everything, recent files and the smart views.
pub fn library_links(total: Option<u64>, failed_thumbs: usize) -> Vec<Link> {
    let mut links = vec![
        Link::new("lib:all", "All media", svg::AME_THUMBNAIL_GRID, Scope::Library)
            .detail(total.map(|n| n.to_string()).unwrap_or_default())
            .tip("Every file stored in the library"),
        Link::new("lib:recent", "Recent", svg::CLOCK_COUNTER_CLOCKWISE, Scope::Recent).tip("Recently scanned or added files"),
    ];
    links.extend(SmartView::ALL.into_iter().map(|view| {
        let (label, icon) = smart_label(view);
        let detail = if view == SmartView::FailedThumbs && failed_thumbs > 0 { failed_thumbs.to_string() } else { String::new() };
        Link::new(format!("smart:{}", view.slug()), label, icon, Scope::Smart(view)).detail(detail).tip(view.label())
    }));
    links
}

/// Short name and icon of a smart view.
pub fn smart_label(view: SmartView) -> (&'static str, Icon) {
    match view {
        SmartView::NoClip => ("No CLIP", svg::AME_CLIP_EMBED),
        SmartView::Undescribed => ("Undescribed", svg::AME_AI_DESCRIBE),
        SmartView::Untagged => ("Untagged", svg::TAG),
        SmartView::FailedThumbs => ("Failed thumbnails", svg::WARNING),
        SmartView::Duplicates => ("Duplicates", svg::AME_DUPLICATES),
    }
}

/// A collection with its member count.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionEntry {
    pub id: surrealdb::types::RecordId,
    pub name: String,
    pub count: u64,
}

/// A tag or category with the number of files that carry it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountEntry {
    pub name: String,
    pub count: i64,
}

/// A saved scan that can be reopened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanEntry {
    pub scan_id: u64,
    pub title: String,
    pub root: String,
    pub started: String,
    pub total: Option<u64>,
}

pub fn collection_links(entries: &[CollectionEntry]) -> Vec<Link> {
    entries
        .iter()
        .map(|c| {
            Link::new(format!("col:{}", c.name), c.name.clone(), svg::AME_LOGICAL_GROUP, Scope::Collection(c.name.clone()))
                .detail(c.count.to_string())
                .tip(format!("{} files in {}", c.count, c.name))
        })
        .collect()
}

pub fn tag_link(entry: &CountEntry) -> Link {
    Link::new(format!("tag:{}", entry.name), entry.name.clone(), svg::TAG, Scope::Tag(entry.name.clone())).detail(entry.count.to_string())
}

pub fn category_link(entry: &CountEntry) -> Link {
    Link::new(format!("cat:{}", entry.name), entry.name.clone(), svg::AME_CATEGORY, Scope::Category(entry.name.clone())).detail(entry.count.to_string())
}

pub fn scan_links(entries: &[ScanEntry]) -> Vec<Link> {
    entries
        .iter()
        .map(|s| {
            let detail = s.total.map(|n| n.to_string()).unwrap_or_default();
            Link::new(format!("scan:{}", s.scan_id), s.title.clone(), svg::AME_SCAN, Scope::Scan(s.scan_id))
                .detail(detail)
                .tip(format!("{}\n{}", s.root, s.started))
        })
        .collect()
}

/// Entries containing `filter` (ignoring case), capped at [`TOP_N`] unless filtering or `show_all`, and how many the cap hid.
pub fn top_entries<'a>(entries: &'a [CountEntry], filter: &str, show_all: bool) -> (Vec<&'a CountEntry>, usize) {
    let needle = filter.trim().to_lowercase();
    let matching: Vec<&CountEntry> = if needle.is_empty() {
        entries.iter().collect()
    } else {
        entries.iter().filter(|e| e.name.to_lowercase().contains(&needle)).collect()
    };
    let limit = if show_all || !needle.is_empty() { MAX_SHOWN } else { TOP_N };
    let hidden = matching.len().saturating_sub(limit);
    (matching.into_iter().take(limit).collect(), hidden)
}

/// Counts ordered by count descending, then name.
pub fn sort_counts(entries: &mut [CountEntry]) {
    entries.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())).then_with(|| a.name.cmp(&b.name)));
}
