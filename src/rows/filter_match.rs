//! Decides whether a row meets a [`FilterSpec`], using only the row and cached lookups.

use std::cell::OnceCell;

use chrono::{LocalResult, NaiveDate, TimeZone};

use super::filter::{DEFAULT_MIN_SIZE, DateRange, FilterSpec, Tri, format_size, parse_date};
use super::filter_probe::Probe;
use super::thumbnail::{ARCHIVE_TYPE, DIR_TYPE, short_extension, with_lowercase};
use crate::Thumbnail;

/// Facts about a row that are not stored on it.
pub trait FactSource {
    /// A CLIP embedding is known for the row, by path or by content hash.
    fn has_clip(&self, row: &Thumbnail) -> bool;
    /// The library holds the row.
    fn is_indexed(&self, row: &Thumbnail) -> bool;
    /// What probing the file found, when it was probed.
    fn probe(&self, path: &str) -> Option<Probe>;
}

/// Facts for a view that knows nothing besides its rows: no CLIP, everything indexed, nothing probed.
pub struct NoFacts;

impl FactSource for NoFacts {
    fn has_clip(&self, _row: &Thumbnail) -> bool {
        false
    }

    fn is_indexed(&self, _row: &Thumbnail) -> bool {
        true
    }

    fn probe(&self, _path: &str) -> Option<Probe> {
        None
    }
}

/// One kind of condition of a spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Condition {
    Types,
    Size,
    ExtAllow,
    ExtDeny,
    ExcludedDirs,
    ExcludedTerms,
    Modified,
    Created,
    Query,
    Name,
    Description,
    Category,
    Tags,
    Indexed,
    Clip,
    Offline,
    SkipIcons,
}

impl Condition {
    /// Every condition in the order the matcher tries them, cheapest first.
    pub const ALL: [Condition; 17] = [
        Self::Types,
        Self::Size,
        Self::ExtAllow,
        Self::ExtDeny,
        Self::ExcludedDirs,
        Self::ExcludedTerms,
        Self::Modified,
        Self::Created,
        Self::Query,
        Self::Name,
        Self::Description,
        Self::Category,
        Self::Tags,
        Self::Indexed,
        Self::Clip,
        Self::Offline,
        Self::SkipIcons,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Types => "Types",
            Self::Size => "Size",
            Self::ExtAllow => "Only extensions",
            Self::ExtDeny => "Excluded extensions",
            Self::ExcludedDirs => "Excluded folders",
            Self::ExcludedTerms => "Excluded terms",
            Self::Modified => "Modified date",
            Self::Created => "Added date",
            Self::Query => "Search text",
            Self::Name => "Name contains",
            Self::Description => "Description",
            Self::Category => "Category",
            Self::Tags => "Tags",
            Self::Indexed => "Indexed",
            Self::Clip => "CLIP",
            Self::Offline => "Offline",
            Self::SkipIcons => "Skip icons",
        }
    }
}

/// How one active condition judged one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub condition: Condition,
    pub passed: bool,
    pub detail: String,
}

/// How many rows each active condition hides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HiddenReport {
    pub total: usize,
    pub visible: usize,
    pub per_condition: Vec<ConditionCount>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditionCount {
    pub condition: Condition,
    /// Rows that fail this condition.
    pub failing: usize,
    /// Rows that fail this condition and no other, so removing it lists them.
    pub only: usize,
}

struct Needle {
    text: String,
    ascii: bool,
}

impl Needle {
    fn new(text: &str) -> Self {
        let text = text.to_lowercase();
        Self { ascii: text.is_ascii(), text }
    }

    fn in_text(&self, hay: &str) -> bool {
        if self.text.is_empty() {
            return true;
        }
        if self.ascii && hay.is_ascii() {
            let (h, n) = (hay.as_bytes(), self.text.as_bytes());
            return n.len() <= h.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n));
        }
        hay.to_lowercase().contains(&self.text)
    }

    fn is_text(&self, other: &str) -> bool {
        if self.ascii && other.is_ascii() { other.eq_ignore_ascii_case(&self.text) } else { other.to_lowercase() == self.text }
    }
}

#[derive(Default)]
struct Bounds {
    after: Option<i64>,
    before: Option<i64>,
}

impl Bounds {
    fn of(range: &DateRange) -> Self {
        let day = |text: &Option<String>, end: bool| text.as_deref().and_then(parse_date).map(|d| local_millis(d, end));
        Self { after: day(&range.after, false), before: day(&range.before, true) }
    }

    fn admits(&self, at: Option<i64>) -> bool {
        if self.after.is_none() && self.before.is_none() {
            return true;
        }
        let Some(at) = at else { return false };
        self.after.is_none_or(|a| at >= a) && self.before.is_none_or(|b| at <= b)
    }
}

/// UTC milliseconds of the first or last instant of `date` in local time.
fn local_millis(date: NaiveDate, end: bool) -> i64 {
    let time = if end { date.and_hms_milli_opt(23, 59, 59, 999) } else { date.and_hms_opt(0, 0, 0) }.unwrap_or_default();
    match (end, chrono::Local.from_local_datetime(&time)) {
        (false, LocalResult::Single(t)) | (false, LocalResult::Ambiguous(t, _)) => t.timestamp_millis(),
        (true, LocalResult::Single(t)) | (true, LocalResult::Ambiguous(_, t)) => t.timestamp_millis(),
        (_, LocalResult::None) => time.and_utc().timestamp_millis(),
    }
}

/// Last component of `path` split at the first dot from the right, when it has a stem.
fn full_extension(source: &str) -> Option<&str> {
    let name = source.rsplit(['/', '\\']).next().unwrap_or(source);
    let (stem, ext) = name.rsplit_once('.')?;
    (!stem.is_empty() && !ext.is_empty()).then_some(ext)
}

/// True when `path` is `dir` or lies below it.
pub fn under_dir(path: &str, dir: &str) -> bool {
    let starts = if cfg!(windows) {
        path.len() >= dir.len() && path.is_char_boundary(dir.len()) && path[..dir.len()].eq_ignore_ascii_case(dir)
    } else {
        path.starts_with(dir)
    };
    starts && (dir.ends_with(['/', '\\']) || path.len() == dir.len() || path[dir.len()..].starts_with(['/', '\\']))
}

fn ext_in(ext: Option<&str>, list: &[String]) -> bool {
    let Some(ext) = ext else { return false };
    if ext.is_ascii() { list.iter().any(|e| e.eq_ignore_ascii_case(ext)) } else { list.contains(&ext.to_lowercase()) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Image,
    Video,
    Archive,
    Other,
}

struct RowCtx<'r> {
    row: &'r Thumbnail,
    is_dir: bool,
    is_virtual: bool,
    /// What names the file: the entry name for archive entries, else the path.
    source: &'r str,
    kind: Kind,
    lower_path: OnceCell<String>,
}

impl<'r> RowCtx<'r> {
    fn new(row: &'r Thumbnail) -> Self {
        let is_dir = row.file_type == DIR_TYPE;
        let source = if row.path.starts_with("zip://") { row.filename.as_str() } else { row.path.as_str() };
        let kind = if is_dir {
            Kind::Other
        } else {
            match short_extension(source) {
                Some(ext) => with_lowercase(ext, |lower| {
                    if crate::is_image(lower) {
                        Kind::Image
                    } else if crate::is_video(lower) {
                        Kind::Video
                    } else if crate::is_archive(lower) {
                        Kind::Archive
                    } else {
                        Kind::Other
                    }
                }),
                None => Kind::Other,
            }
        };
        Self { row, is_dir, is_virtual: row.path.contains("://"), source, kind, lower_path: OnceCell::new() }
    }

    fn path_has(&self, needle: &Needle) -> bool {
        if self.row.path.is_ascii() && needle.ascii {
            return needle.in_text(&self.row.path);
        }
        self.lower_path.get_or_init(|| self.row.path.to_lowercase()).contains(&needle.text)
    }
}

/// A [`FilterSpec`] compiled for testing many rows against.
pub struct PreparedFilter<'a> {
    spec: FilterSpec,
    active: Vec<Condition>,
    facts: &'a dyn FactSource,
    query: Needle,
    name: Needle,
    terms: Vec<Needle>,
    categories: Vec<Needle>,
    tags: Vec<Needle>,
    modified: Bounds,
    created: Bounds,
    tiny_limit: u64,
}

impl<'a> PreparedFilter<'a> {
    pub fn new(spec: &FilterSpec, facts: &'a dyn FactSource) -> Self {
        let query = if spec.query.trim().is_empty() { Needle::new("") } else { Needle::new(&spec.query) };
        let name = Needle::new(spec.name.trim());
        let active = Condition::ALL.into_iter().filter(|c| is_active(spec, *c)).collect();
        Self {
            terms: spec.excluded_terms.iter().map(|t| Needle::new(t)).collect(),
            categories: spec.categories.iter().map(|t| Needle::new(t)).collect(),
            tags: spec.tags.iter().map(|t| Needle::new(t)).collect(),
            modified: Bounds::of(&spec.modified),
            created: Bounds::of(&spec.created),
            tiny_limit: spec.size.min.unwrap_or(DEFAULT_MIN_SIZE),
            spec: spec.clone(),
            active,
            facts,
            query,
            name,
        }
    }

    pub fn spec(&self) -> &FilterSpec {
        &self.spec
    }

    /// The conditions the spec has active, in matching order.
    pub fn active(&self) -> &[Condition] {
        &self.active
    }

    pub fn matches(&self, row: &Thumbnail) -> bool {
        if self.active.is_empty() {
            return true;
        }
        let cx = RowCtx::new(row);
        self.active.iter().all(|c| self.passes(*c, &cx))
    }

    /// How each active condition judges `row`.
    pub fn explain(&self, row: &Thumbnail) -> Vec<Verdict> {
        let cx = RowCtx::new(row);
        self.active.iter().map(|c| self.verdict(*c, &cx)).collect()
    }

    /// Counts the rows each condition hides.
    pub fn report<'r>(&self, rows: impl Iterator<Item = &'r Thumbnail>) -> HiddenReport {
        let mut report = HiddenReport { per_condition: self.active.iter().map(|c| ConditionCount { condition: *c, failing: 0, only: 0 }).collect(), ..Default::default() };
        let mut failed = vec![false; self.active.len()];
        for row in rows {
            let cx = RowCtx::new(row);
            report.total += 1;
            for (slot, c) in failed.iter_mut().zip(&self.active) {
                *slot = !self.passes(*c, &cx);
            }
            match failed.iter().filter(|f| **f).count() {
                0 => report.visible += 1,
                1 => {
                    let i = failed.iter().position(|f| *f).unwrap_or(0);
                    report.per_condition[i].only += 1;
                    report.per_condition[i].failing += 1;
                }
                _ => {
                    for (count, f) in report.per_condition.iter_mut().zip(&failed) {
                        count.failing += usize::from(*f);
                    }
                }
            }
        }
        report
    }

    fn verdict(&self, c: Condition, cx: &RowCtx<'_>) -> Verdict {
        let passed = self.passes(c, cx);
        Verdict { condition: c, passed, detail: self.detail(c, cx, passed) }
    }

    fn passes(&self, c: Condition, cx: &RowCtx<'_>) -> bool {
        let row = cx.row;
        if cx.is_dir && !matches!(c, Condition::Types | Condition::ExcludedDirs | Condition::Query | Condition::Name) {
            return true;
        }
        match c {
            Condition::Types => {
                let t = &self.spec.types;
                if cx.is_dir {
                    return t.dirs;
                }
                match cx.kind {
                    Kind::Image => t.images,
                    Kind::Video => t.videos,
                    Kind::Archive => t.archives,
                    Kind::Other if row.file_type == ARCHIVE_TYPE => t.archives,
                    Kind::Other => t.other,
                }
            }
            Condition::Size => {
                if row.size == 0 && cx.is_virtual {
                    return true;
                }
                let s = &self.spec.size;
                s.min.is_none_or(|m| row.size >= m) && s.max.is_none_or(|m| row.size <= m)
            }
            Condition::ExtAllow => ext_in(full_extension(cx.source), &self.spec.ext_allow),
            Condition::ExtDeny => !ext_in(full_extension(cx.source), &self.spec.ext_deny),
            Condition::ExcludedDirs => !self.spec.excluded_dirs.iter().any(|d| under_dir(&row.path, d)),
            Condition::ExcludedTerms => !self.terms.iter().any(|t| cx.path_has(t)),
            Condition::Modified => self.modified.admits(row.modified.as_ref().map(|d| d.timestamp_millis())),
            Condition::Created => self.created.admits(row.db_created.as_ref().map(|d| d.timestamp_millis())),
            Condition::Query => self.query_hit(row),
            Condition::Name => self.name.in_text(&row.filename),
            Condition::Description => self.spec.description.wants(row.description.as_deref().is_some_and(|s| !s.trim().is_empty())),
            Condition::Category => row.category.as_deref().is_some_and(|c| self.categories.iter().any(|n| n.is_text(c.trim()))),
            Condition::Tags => {
                let has = |n: &Needle| row.tags.iter().any(|t| n.is_text(t.trim()));
                if self.spec.tags_all { self.tags.iter().all(has) } else { self.tags.iter().any(has) }
            }
            Condition::Indexed => self.spec.indexed.wants(self.facts.is_indexed(row)),
            Condition::Clip => self.spec.clip.wants(self.facts.has_clip(row)),
            Condition::Offline => {
                let offline = if cx.is_virtual { Some(false) } else { self.facts.probe(&row.path).map(|p| !p.exists) };
                match (self.spec.offline, offline) {
                    (Tri::Any, _) => true,
                    (Tri::Yes, known) => known == Some(true),
                    (Tri::No, known) => known != Some(true),
                }
            }
            Condition::SkipIcons => {
                if full_extension(cx.source).is_some_and(|e| e.eq_ignore_ascii_case("ico")) {
                    return false;
                }
                if cx.kind != Kind::Image || cx.is_virtual {
                    return true;
                }
                !(row.size <= self.tiny_limit && self.facts.probe(&row.path).is_some_and(|p| p.dims_checked && p.tiny))
            }
        }
    }

    fn query_hit(&self, row: &Thumbnail) -> bool {
        let n = &self.query;
        n.in_text(&row.filename)
            || n.in_text(&row.file_type)
            || n.in_text(&row.path)
            || (!row.tags.is_empty() && (row.tags.iter().any(|t| n.in_text(t)) || (n.text.contains(',') && n.in_text(&row.tags.join(",")))))
            || row.category.as_deref().is_some_and(|c| n.in_text(c))
            || row.description.as_deref().is_some_and(|d| n.in_text(d))
    }

    fn detail(&self, c: Condition, cx: &RowCtx<'_>, passed: bool) -> String {
        let row = cx.row;
        let spec = &self.spec;
        let yes_no = |b: bool| if b { "yes" } else { "no" };
        match c {
            Condition::Types => format!("{} row, {}", if cx.is_dir { "folder" } else { kind_name(cx.kind) }, if passed { "type is listed" } else { "type is hidden" }),
            Condition::Size => format!("size {} (limits {})", format_size(row.size), bounds_text(spec.size.min.map(format_size), spec.size.max.map(format_size))),
            Condition::ExtAllow => format!("extension {} (allowed: {})", full_extension(cx.source).unwrap_or("none"), spec.ext_allow.join(", ")),
            Condition::ExtDeny => format!("extension {} (denied: {})", full_extension(cx.source).unwrap_or("none"), spec.ext_deny.join(", ")),
            Condition::ExcludedDirs => match spec.excluded_dirs.iter().find(|d| under_dir(&row.path, d)) {
                Some(d) => format!("below excluded folder {d}"),
                None => "not below an excluded folder".to_string(),
            },
            Condition::ExcludedTerms => match self.terms.iter().find(|t| cx.path_has(t)) {
                Some(t) => format!("path contains '{}'", t.text),
                None => "path has no excluded term".to_string(),
            },
            Condition::Modified => format!("modified {} (range {})", date_text(row.modified.as_ref().map(|d| d.timestamp_millis())), range_text(&spec.modified)),
            Condition::Created => format!("added {} (range {})", date_text(row.db_created.as_ref().map(|d| d.timestamp_millis())), range_text(&spec.created)),
            Condition::Query => format!("search '{}' {}", spec.query, if passed { "found" } else { "not found in name, type, path, tags, category or description" }),
            Condition::Name => format!("name '{}' {}", row.filename, if passed { "contains the text" } else { "lacks the text" }),
            Condition::Description => format!("has description: {}", yes_no(row.description.as_deref().is_some_and(|s| !s.trim().is_empty()))),
            Condition::Category => format!("category {} (wanted: {})", row.category.as_deref().unwrap_or("none"), spec.categories.join(", ")),
            Condition::Tags => format!("tags [{}] (wanted {}: {})", row.tags.join(", "), if spec.tags_all { "all" } else { "any" }, spec.tags.join(", ")),
            Condition::Indexed => format!("indexed: {}", yes_no(self.facts.is_indexed(row))),
            Condition::Clip => format!("has CLIP: {}", yes_no(self.facts.has_clip(row))),
            Condition::Offline => match self.facts.probe(&row.path) {
                Some(p) => format!("file {}", if p.exists { "found" } else { "missing" }),
                None if cx.is_virtual => "virtual path".to_string(),
                None => "not probed yet".to_string(),
            },
            Condition::SkipIcons => match self.facts.probe(&row.path) {
                Some(p) if p.dims_checked => format!("image size {}", if p.tiny { "is icon-sized" } else { "is not icon-sized" }),
                _ => "icon check applies to small images".to_string(),
            },
        }
    }
}

fn is_active(spec: &FilterSpec, c: Condition) -> bool {
    match c {
        Condition::Types => !spec.types.is_all(),
        Condition::Size => spec.size.min.is_some() || spec.size.max.is_some(),
        Condition::ExtAllow => !spec.ext_allow.is_empty(),
        Condition::ExtDeny => !spec.ext_deny.is_empty(),
        Condition::ExcludedDirs => !spec.excluded_dirs.is_empty(),
        Condition::ExcludedTerms => !spec.excluded_terms.is_empty(),
        Condition::Modified => !spec.modified.is_empty(),
        Condition::Created => !spec.created.is_empty(),
        Condition::Query => !spec.query.trim().is_empty(),
        Condition::Name => !spec.name.trim().is_empty(),
        Condition::Description => !spec.description.is_any(),
        Condition::Category => !spec.categories.is_empty(),
        Condition::Tags => !spec.tags.is_empty(),
        Condition::Indexed => !spec.indexed.is_any(),
        Condition::Clip => !spec.clip.is_any(),
        Condition::Offline => !spec.offline.is_any(),
        Condition::SkipIcons => spec.skip_icons,
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Image => "image",
        Kind::Video => "video",
        Kind::Archive => "archive",
        Kind::Other => "other",
    }
}

fn bounds_text(min: Option<String>, max: Option<String>) -> String {
    match (min, max) {
        (Some(a), Some(b)) => format!("{a} to {b}"),
        (Some(a), None) => format!(">= {a}"),
        (None, Some(b)) => format!("<= {b}"),
        (None, None) => "none".to_string(),
    }
}

fn range_text(range: &DateRange) -> String {
    bounds_text(range.after.clone(), range.before.clone())
}

fn date_text(millis: Option<i64>) -> String {
    match millis.and_then(chrono::DateTime::from_timestamp_millis) {
        Some(utc) => utc.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string(),
        None => "unknown".to_string(),
    }
}
