//! The conditions a row must meet to be listed: one serializable spec per view, with its chips, defaults and signature.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::UiSettings;

/// Smallest file the app lists unless the user changes the setting.
pub const DEFAULT_MIN_SIZE: u64 = 10 * 1024;

/// A yes/no/either condition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tri {
    #[default]
    Any,
    Yes,
    No,
}

impl Tri {
    pub fn is_any(self) -> bool {
        self == Self::Any
    }

    /// Whether a row for which the fact is `has` satisfies the condition.
    pub fn wants(self, has: bool) -> bool {
        match self {
            Self::Any => true,
            Self::Yes => has,
            Self::No => !has,
        }
    }
}

/// Which kinds of rows are listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TypeSet {
    pub images: bool,
    pub videos: bool,
    pub dirs: bool,
    pub archives: bool,
    pub other: bool,
}

impl Default for TypeSet {
    fn default() -> Self {
        Self { images: true, videos: true, dirs: true, archives: true, other: true }
    }
}

impl TypeSet {
    pub fn is_all(&self) -> bool {
        *self == Self::default()
    }

    /// Names of the kinds that are not listed.
    pub fn hidden_names(&self) -> Vec<&'static str> {
        [(self.images, "Images"), (self.videos, "Videos"), (self.dirs, "Folders"), (self.archives, "Archives"), (self.other, "Other files")]
            .into_iter()
            .filter_map(|(shown, name)| (!shown).then_some(name))
            .collect()
    }
}

/// Inclusive size bounds in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SizeRange {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

/// Inclusive day bounds as `YYYY-MM-DD` in local time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DateRange {
    pub after: Option<String>,
    pub before: Option<String>,
}

impl DateRange {
    pub fn is_empty(&self) -> bool {
        self.after.is_none() && self.before.is_none()
    }

    fn label(&self, what: &str) -> Option<String> {
        match (self.after.as_deref(), self.before.as_deref()) {
            (Some(a), Some(b)) => Some(format!("{what} {a} to {b}")),
            (Some(a), None) => Some(format!("{what} since {a}")),
            (None, Some(b)) => Some(format!("{what} until {b}")),
            (None, None) => None,
        }
    }
}

/// Every condition of a view. A row is listed when it meets all of the active ones.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FilterSpec {
    pub types: TypeSet,
    pub size: SizeRange,
    /// Modified date of the file.
    pub modified: DateRange,
    /// Date the row was added to the library.
    pub created: DateRange,
    /// Lowercase substrings; a file whose path contains one is not listed.
    pub excluded_terms: Vec<String>,
    /// Folders; everything below one is not listed.
    pub excluded_dirs: Vec<String>,
    /// Text that must appear in the name, type, path, tags, category or description.
    pub query: String,
    /// Text that must appear in the file name.
    pub name: String,
    pub skip_icons: bool,
    /// Lowercase extensions without the dot; when set, only these are listed.
    pub ext_allow: Vec<String>,
    /// Lowercase extensions without the dot that are not listed.
    pub ext_deny: Vec<String>,
    pub indexed: Tri,
    pub description: Tri,
    pub clip: Tri,
    /// Any of these categories.
    pub categories: Vec<String>,
    pub tags: Vec<String>,
    /// All of `tags` instead of any.
    pub tags_all: bool,
    pub offline: Tri,
}

impl FilterSpec {
    /// No condition at all.
    pub fn none() -> Self {
        Self {
            types: TypeSet::default(),
            size: SizeRange::default(),
            modified: DateRange::default(),
            created: DateRange::default(),
            excluded_terms: Vec::new(),
            excluded_dirs: Vec::new(),
            query: String::new(),
            name: String::new(),
            skip_icons: false,
            ext_allow: Vec::new(),
            ext_deny: Vec::new(),
            indexed: Tri::Any,
            description: Tri::Any,
            clip: Tri::Any,
            categories: Vec::new(),
            tags: Vec::new(),
            tags_all: false,
            offline: Tri::Any,
        }
    }

    /// The conditions the settings hold: sizes, dates, icons, excluded extensions and folders.
    pub fn from_settings(s: &UiSettings) -> Self {
        let mut spec = Self::none();
        spec.size = SizeRange { min: s.db_min_size_bytes, max: s.db_max_size_bytes };
        spec.modified = DateRange { after: s.filter_modified_after.clone(), before: s.filter_modified_before.clone() };
        spec.skip_icons = s.filter_skip_icons;
        spec.ext_deny = s.db_excluded_exts.clone().unwrap_or_default();
        spec.excluded_dirs = s.excluded_dirs.clone().unwrap_or_default();
        spec.normalized()
    }

    /// Writes the settings-backed conditions into `s`.
    pub fn write_settings(&self, s: &mut UiSettings) {
        s.db_min_size_bytes = self.size.min;
        s.db_max_size_bytes = self.size.max;
        s.filter_modified_after = self.modified.after.clone();
        s.filter_modified_before = self.modified.before.clone();
        s.filter_skip_icons = self.skip_icons;
        s.db_excluded_exts = (!self.ext_deny.is_empty()).then(|| self.ext_deny.clone());
        s.excluded_dirs = (!self.excluded_dirs.is_empty()).then(|| self.excluded_dirs.clone());
    }

    /// Moves each settings-backed group that still equals `old` to its value in `new`.
    pub fn follow(&mut self, old: &FilterSpec, new: &FilterSpec) {
        if self.size == old.size {
            self.size = new.size;
        }
        if self.modified == old.modified {
            self.modified = new.modified.clone();
        }
        if self.skip_icons == old.skip_icons {
            self.skip_icons = new.skip_icons;
        }
        if self.ext_deny == old.ext_deny {
            self.ext_deny = new.ext_deny.clone();
        }
        if self.excluded_dirs == old.excluded_dirs {
            self.excluded_dirs = new.excluded_dirs.clone();
        }
    }

    /// True when no condition is active.
    pub fn is_empty(&self) -> bool {
        *self == Self::none()
    }

    /// Lists the spec's conditions as removable chips.
    pub fn chips(&self) -> Vec<Chip> {
        let mut chips = Vec::new();
        let mut push = |kind: ChipKind, label: String| chips.push(Chip { kind, label });
        if !self.query.trim().is_empty() {
            push(ChipKind::Query, format!("Search: {}", shorten(self.query.trim(), CHIP_TEXT_MAX)));
        }
        if !self.name.trim().is_empty() {
            push(ChipKind::Name, format!("Name: {}", shorten(self.name.trim(), CHIP_TEXT_MAX)));
        }
        let hidden = self.types.hidden_names();
        if !hidden.is_empty() {
            push(ChipKind::Types, format!("Hiding: {}", hidden.join(", ")));
        }
        let size = match (self.size.min, self.size.max) {
            (Some(a), Some(b)) => Some(format!("Size {} - {}", format_size(a), format_size(b))),
            (Some(a), None) => Some(format!("Size >= {}", format_size(a))),
            (None, Some(b)) => Some(format!("Size <= {}", format_size(b))),
            (None, None) => None,
        };
        if let Some(label) = size {
            push(ChipKind::Size, label);
        }
        if let Some(label) = self.modified.label("Modified") {
            push(ChipKind::Modified, label);
        }
        if let Some(label) = self.created.label("Added") {
            push(ChipKind::Created, label);
        }
        if self.skip_icons {
            push(ChipKind::SkipIcons, "Skip icons".to_string());
        }
        if !self.ext_allow.is_empty() {
            let list = self.ext_allow.iter().map(|e| format!(".{e}")).collect::<Vec<_>>().join(", ");
            push(ChipKind::ExtAllow, format!("Only {}", shorten(&list, CHIP_LIST_MAX)));
        }
        for e in &self.ext_deny {
            push(ChipKind::ExtDeny(e.clone()), format!("Not .{e}"));
        }
        for (tri, kind, yes, no) in [
            (self.indexed, ChipKind::Indexed, "Indexed", "Not indexed"),
            (self.description, ChipKind::Description, "Has description", "No description"),
            (self.clip, ChipKind::Clip, "Has CLIP", "No CLIP"),
            (self.offline, ChipKind::Offline, "Offline", "Online"),
        ] {
            match tri {
                Tri::Any => {}
                Tri::Yes => push(kind, yes.to_string()),
                Tri::No => push(kind, no.to_string()),
            }
        }
        if !self.categories.is_empty() {
            push(ChipKind::Categories, format!("Category: {}", shorten(&self.categories.join(", "), CHIP_LIST_MAX)));
        }
        if self.tags_all && self.tags.len() > 1 {
            for t in &self.tags {
                push(ChipKind::Tag(t.clone()), format!("Tag (all): {}", shorten(t, CHIP_TEXT_MAX)));
            }
        } else if !self.tags.is_empty() {
            push(ChipKind::Tags, format!("Tag: {}", shorten(&self.tags.join(", "), CHIP_LIST_MAX)));
        }
        for t in &self.excluded_terms {
            push(ChipKind::ExcludedTerm(t.clone()), format!("Not matching: {}", shorten(t, CHIP_TEXT_MAX)));
        }
        for d in &self.excluded_dirs {
            push(ChipKind::ExcludedDir(d.clone()), format!("Excluding folder: {}", shorten(last_component(d), CHIP_TEXT_MAX)));
        }
        chips
    }

    /// Removes the condition a chip stands for.
    pub fn remove(&mut self, kind: &ChipKind) {
        match kind {
            ChipKind::Query => self.query.clear(),
            ChipKind::Name => self.name.clear(),
            ChipKind::Types => self.types = TypeSet::default(),
            ChipKind::Size => self.size = SizeRange::default(),
            ChipKind::Modified => self.modified = DateRange::default(),
            ChipKind::Created => self.created = DateRange::default(),
            ChipKind::SkipIcons => self.skip_icons = false,
            ChipKind::ExtAllow => self.ext_allow.clear(),
            ChipKind::ExtDeny(e) => self.ext_deny.retain(|x| x != e),
            ChipKind::Indexed => self.indexed = Tri::Any,
            ChipKind::Description => self.description = Tri::Any,
            ChipKind::Clip => self.clip = Tri::Any,
            ChipKind::Offline => self.offline = Tri::Any,
            ChipKind::Categories => self.categories.clear(),
            ChipKind::Tags => {
                self.tags.clear();
                self.tags_all = false;
            }
            ChipKind::Tag(t) => {
                self.tags.retain(|x| x != t);
                if self.tags.len() < 2 {
                    self.tags_all = false;
                }
            }
            ChipKind::ExcludedTerm(t) => self.excluded_terms.retain(|x| x != t),
            ChipKind::ExcludedDir(d) => self.excluded_dirs.retain(|x| x != d),
        }
    }

    /// Cleans the spec: trims and lowercases terms and extensions, drops blanks and repeats, orders the size bounds.
    pub fn normalized(mut self) -> Self {
        normalize_terms(&mut self.excluded_terms);
        for list in [&mut self.ext_allow, &mut self.ext_deny] {
            for e in list.iter_mut() {
                *e = e.trim().trim_start_matches("*.").trim_start_matches('.').to_lowercase();
            }
            list.retain(|e| !e.is_empty());
            dedup_keep_order(list);
        }
        normalize_names(&mut self.categories);
        normalize_names(&mut self.tags);
        for d in &mut self.excluded_dirs {
            *d = d.trim().to_string();
        }
        self.excluded_dirs.retain(|d| !d.is_empty());
        dedup_keep_order(&mut self.excluded_dirs);
        self.name = self.name.trim().to_string();
        for range in [&mut self.modified, &mut self.created] {
            for bound in [&mut range.after, &mut range.before] {
                *bound = bound.take().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
            }
        }
        if let (Some(a), Some(b)) = (self.size.min, self.size.max) {
            if a > b {
                self.size = SizeRange { min: Some(b), max: Some(a) };
            }
        }
        if self.tags.len() < 2 {
            self.tags_all = false;
        }
        self
    }

    /// Checks that the dates read as `YYYY-MM-DD`.
    pub fn check(&self) -> Result<(), String> {
        for (what, range) in [("modified", &self.modified), ("created", &self.created)] {
            for (side, bound) in [("after", &range.after), ("before", &range.before)] {
                if let Some(text) = bound {
                    parse_date(text).ok_or_else(|| format!("{what}.{side}: '{text}' is not a date like 2024-01-31"))?;
                }
            }
        }
        Ok(())
    }

    /// Hash that changes whenever the spec does.
    pub fn signature(&self) -> u64 {
        self.signature_with(&self.query, self.types)
    }

    /// The signature the spec would have with `query` and `types` in place of its own.
    pub fn signature_with(&self, query: &str, types: TypeSet) -> u64 {
        let Self { types: _, size, modified, created, excluded_terms, excluded_dirs, query: _, name, skip_icons, ext_allow, ext_deny, indexed, description, clip, categories, tags, tags_all, offline } = self;
        let mut h = DefaultHasher::new();
        (query, types, size, modified, created, excluded_terms, excluded_dirs, name, skip_icons).hash(&mut h);
        (ext_allow, ext_deny, indexed, description, clip, categories, tags, tags_all, offline).hash(&mut h);
        h.finish()
    }

    /// Pretty JSON of the whole spec.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Reads a full or partial spec; fields left out take the app defaults.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let spec: Self = serde_json::from_str(text).map_err(|e| format!("bad filter: {e}"))?;
        spec.check()?;
        Ok(spec.normalized())
    }

    /// The spec after merging `patch` (a JSON object) into this one, or into the defaults when `replace`. A `null` turns that part off.
    pub fn merged(&self, patch: &Value, replace: bool) -> Result<Self, String> {
        if !patch.is_object() {
            return Err("bad filter: expected a JSON object".to_string());
        }
        let base = if replace { Self::default() } else { self.clone() };
        let mut value = serde_json::to_value(&base).map_err(|e| e.to_string())?;
        let off = serde_json::to_value(Self::none()).map_err(|e| e.to_string())?;
        merge_patch(&mut value, patch, &off);
        let spec: Self = serde_json::from_value(value).map_err(|e| format!("bad filter: {e}"))?;
        spec.check()?;
        Ok(spec.normalized())
    }
}

impl Default for FilterSpec {
    /// The app defaults: everything listed except files under 10 KiB.
    fn default() -> Self {
        let mut spec = Self::none();
        spec.size.min = Some(DEFAULT_MIN_SIZE);
        spec
    }
}

fn merge_patch(target: &mut Value, patch: &Value, off: &Value) {
    match (target, patch) {
        (Value::Object(t), Value::Object(p)) => {
            for (key, value) in p {
                let off_value = off.get(key).unwrap_or(&Value::Null);
                if value.is_null() {
                    t.insert(key.clone(), off_value.clone());
                } else if let Some(existing) = t.get_mut(key) {
                    merge_patch(existing, value, off_value);
                } else {
                    t.insert(key.clone(), value.clone());
                }
            }
        }
        (target, patch) => *target = patch.clone(),
    }
}

/// Longest text a chip label shows before it is cut.
const CHIP_TEXT_MAX: usize = 24;
/// Longest list a chip label shows before it is cut.
const CHIP_LIST_MAX: usize = 36;

/// What a chip removes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ChipKind {
    Query,
    Name,
    Types,
    Size,
    Modified,
    Created,
    SkipIcons,
    /// The whole allow-list, since each entry only widens it.
    ExtAllow,
    ExtDeny(String),
    Indexed,
    Description,
    Clip,
    Offline,
    /// The whole list of categories.
    Categories,
    /// The whole list of tags, when any of them is enough.
    Tags,
    /// One tag of a list that must all be present.
    Tag(String),
    ExcludedTerm(String),
    ExcludedDir(String),
}

/// One active condition, ready to draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chip {
    pub kind: ChipKind,
    pub label: String,
}

pub fn format_size(bytes: u64) -> String {
    humansize::format_size(bytes, humansize::BINARY)
}

fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
}

fn last_component(path: &str) -> &str {
    let trimmed = path.trim_end_matches(['/', '\\']);
    trimmed.rsplit(['/', '\\']).next().filter(|s| !s.is_empty()).unwrap_or(path)
}

fn dedup_keep_order(list: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    list.retain(|s| seen.insert(s.clone()));
}

fn normalize_terms(list: &mut Vec<String>) {
    for t in list.iter_mut() {
        *t = t.trim().to_lowercase();
    }
    list.retain(|t| !t.is_empty());
    dedup_keep_order(list);
}

fn normalize_names(list: &mut Vec<String>) {
    for t in list.iter_mut() {
        *t = t.trim().to_string();
    }
    list.retain(|t| !t.is_empty());
    let mut seen = std::collections::HashSet::new();
    list.retain(|t| seen.insert(t.to_lowercase()));
}

/// Parses `YYYY-MM-DD`.
pub fn parse_date(text: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d").ok()
}

/// Reads a size such as `10 KiB`, `2mb`, `1.5 GiB` or `4096`; units are powers of 1024. Blank, `none` and `-` mean no bound.
pub fn parse_size(text: &str) -> Result<Option<u64>, String> {
    let t = text.trim().to_ascii_lowercase();
    if t.is_empty() || t == "none" || t == "-" || t == "any" {
        return Ok(None);
    }
    let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let (number, unit) = t.split_at(split);
    let value: f64 = number.parse().map_err(|_| format!("'{text}' is not a size like 10 KiB"))?;
    let factor: f64 = match unit.trim() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown size unit '{other}'")),
    };
    let bytes = value * factor;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(format!("'{text}' is out of range"));
    }
    Ok(Some(bytes as u64))
}

/// Splits user text into extension, term or name tokens at commas, semicolons and whitespace.
pub fn split_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string).collect()
}
