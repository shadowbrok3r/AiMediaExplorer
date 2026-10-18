//! Windows to local path remapping with longest-prefix rules and a case-insensitive per-directory probe.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    #[default]
    Keep,
    Drop,
    Tag,
}

impl Policy {
    pub fn label(self) -> &'static str {
        match self {
            Policy::Keep => "keep",
            Policy::Drop => "drop",
            Policy::Tag => "tag",
        }
    }
}

impl FromStr for Policy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "keep" => Ok(Policy::Keep),
            "drop" => Ok(Policy::Drop),
            "tag" => Ok(Policy::Tag),
            other => Err(format!("expected keep, drop or tag, not '{other}'")),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rule {
    pub from: String,
    pub to: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

fn d_missing_tag() -> String {
    "legacy-missing".into()
}

fn d_unmapped_tag() -> String {
    "legacy-unmapped".into()
}

fn d_true() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Options {
    #[serde(default)]
    pub unmapped: Policy,
    #[serde(default)]
    pub missing: Policy,
    #[serde(default = "d_missing_tag")]
    pub missing_tag: String,
    #[serde(default = "d_unmapped_tag")]
    pub unmapped_tag: String,
    #[serde(default = "d_true")]
    pub case_fix: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { unmapped: Policy::Keep, missing: Policy::Keep, missing_tag: d_missing_tag(), unmapped_tag: d_unmapped_tag(), case_fix: true }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Config {
    #[serde(default)]
    pub options: Options,
    #[serde(default, rename = "rule")]
    pub rules: Vec<Rule>,
}

impl Config {
    pub fn parse(text: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(text).context("path map is not valid TOML")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read path map {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("path map {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        for (i, r) in self.rules.iter().enumerate() {
            if parse_win(&r.from).is_none() {
                bail!("rule {}: `from` must start with a drive letter or a \\\\host\\share root, got {:?}", i + 1, r.from);
            }
            if !r.to.is_empty() && !Path::new(&r.to).is_absolute() {
                bail!("rule {}: `to` must be an absolute path or empty, got {:?}", i + 1, r.to);
            }
        }
        Ok(())
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}

/// A Windows path split into its root (`C:` or `\\HOST`) and components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinPath {
    pub root: String,
    pub comps: Vec<String>,
}

fn split_comps(rest: &str) -> Vec<String> {
    rest.split(['\\', '/']).filter(|c| !c.is_empty()).map(str::to_string).collect()
}

/// Parses `X:\a\b`, `X:/a/b`, `\\host\share\a` and `\\?\` prefixed forms; None for anything else.
pub fn parse_win(p: &str) -> Option<WinPath> {
    let mut s = p;
    if let Some(r) = s.strip_prefix("\\\\?\\") {
        s = r;
        if s.len() >= 4 && s[..4].eq_ignore_ascii_case("UNC\\") {
            let mut comps = split_comps(&s[4..]);
            if comps.is_empty() {
                return None;
            }
            let host = comps.remove(0);
            return Some(WinPath { root: format!("\\\\{}", host.to_uppercase()), comps });
        }
    } else if let Some(r) = s.strip_prefix("\\\\") {
        let mut comps = split_comps(r);
        if comps.is_empty() {
            return None;
        }
        let host = comps.remove(0);
        return Some(WinPath { root: format!("\\\\{}", host.to_uppercase()), comps });
    }
    let b = s.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b.len() == 2 || b[2] == b'\\' || b[2] == b'/') {
        let drive = (b[0] as char).to_ascii_uppercase();
        let rest = if b.len() > 3 { &s[3..] } else { "" };
        return Some(WinPath { root: format!("{drive}:"), comps: split_comps(rest) });
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Status {
    Exists,
    ExistsCaseFixed,
    Missing,
    Unmapped,
    Native,
}

impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Status::Exists => "exists",
            Status::ExistsCaseFixed => "exists_case_fixed",
            Status::Missing => "missing",
            Status::Unmapped => "unmapped",
            Status::Native => "native",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mapped {
    pub path: String,
    pub status: Status,
    pub rule: Option<usize>,
    pub first_missing: Option<String>,
}

struct DirList {
    exact: HashSet<String>,
    lower: HashMap<String, Vec<String>>,
}

pub struct Mapper {
    pub cfg: Config,
    rule_comps: Vec<(String, Vec<String>, usize)>,
    dirs: HashMap<PathBuf, Option<DirList>>,
    memo: HashMap<String, Mapped>,
    pub dirs_listed: u64,
}

impl Mapper {
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        let mut rule_comps = Vec::new();
        for (i, r) in cfg.rules.iter().enumerate() {
            let w = parse_win(&r.from).with_context(|| format!("rule {}: bad `from`", i + 1))?;
            rule_comps.push((w.root, w.comps.iter().map(|c| c.to_lowercase()).collect::<Vec<_>>(), i));
        }
        rule_comps.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.2.cmp(&b.2)));
        Ok(Self { cfg, rule_comps, dirs: HashMap::new(), memo: HashMap::new(), dirs_listed: 0 })
    }

    fn list(&mut self, dir: &Path) -> Option<&DirList> {
        if !self.dirs.contains_key(dir) {
            let mut out = None;
            if let Ok(rd) = std::fs::read_dir(dir) {
                let mut exact = HashSet::new();
                let mut lower: HashMap<String, Vec<String>> = HashMap::new();
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    lower.entry(n.to_lowercase()).or_default().push(n.clone());
                    exact.insert(n);
                }
                out = Some(DirList { exact, lower });
                self.dirs_listed += 1;
            }
            self.dirs.insert(dir.to_path_buf(), out);
        }
        self.dirs.get(dir).and_then(|o| o.as_ref())
    }

    /// Maps one stored path; deterministic and memoized.
    pub fn map(&mut self, p: &str) -> Mapped {
        if let Some(m) = self.memo.get(p) {
            return m.clone();
        }
        let m = self.map_uncached(p);
        self.memo.insert(p.to_string(), m.clone());
        m
    }

    fn map_uncached(&mut self, p: &str) -> Mapped {
        let plain = |status, rule| Mapped { path: p.to_string(), status, rule, first_missing: None };
        if p.contains("://") || p.starts_with('/') {
            return plain(Status::Native, None);
        }
        let Some(win) = parse_win(p) else { return plain(Status::Unmapped, None) };
        let lowered: Vec<String> = win.comps.iter().map(|c| c.to_lowercase()).collect();
        let hit = self
            .rule_comps
            .iter()
            .find(|(root, need, _)| *root == win.root && need.len() <= lowered.len() && lowered[..need.len()] == need[..])
            .map(|(_, need, idx)| (*idx, need.len()));
        let Some((ri, consumed)) = hit else { return plain(Status::Unmapped, None) };
        let to = self.cfg.rules[ri].to.clone();
        if to.is_empty() {
            return plain(Status::Unmapped, Some(ri));
        }
        let case_fix = self.cfg.options.case_fix;
        let mut cur = PathBuf::from(if to.len() > 1 { to.trim_end_matches('/') } else { to.as_str() });
        let mut fixed = false;
        let mut missing: Option<String> = None;
        for (i, c) in win.comps[consumed..].iter().enumerate() {
            if missing.is_some() {
                cur.push(c);
                continue;
            }
            let found = self.list(&cur).and_then(|dl| {
                if dl.exact.contains(c) {
                    Some((c.clone(), false))
                } else if case_fix {
                    match dl.lower.get(&c.to_lowercase()) {
                        Some(v) if v.len() == 1 => Some((v[0].clone(), true)),
                        _ => None,
                    }
                } else {
                    None
                }
            });
            match found {
                Some((real, f)) => {
                    fixed |= f;
                    cur.push(real);
                }
                None => {
                    let shown = win.comps[..=consumed + i].join("\\");
                    missing = Some(format!("{}\\{shown}", win.root));
                    cur.push(c);
                }
            }
        }
        let status = if missing.is_some() {
            Status::Missing
        } else if fixed {
            Status::ExistsCaseFixed
        } else {
            Status::Exists
        };
        Mapped { path: cur.to_string_lossy().into_owned(), status, rule: Some(ri), first_missing: missing }
    }
}

/// Per-rule aggregate used by the path-mapping table.
#[derive(Default, Debug, Clone, Serialize)]
pub struct RuleStats {
    pub rows: u64,
    pub by_status: BTreeMap<String, u64>,
    pub missing_by_prefix: BTreeMap<String, u64>,
    pub missing_examples: Vec<String>,
    pub case_fixed_examples: Vec<String>,
}

pub const DEFAULT_PATHMAP_FILE: &str = "legacy-pathmap.toml";

/// Location of the editable path map next to the store folder.
pub fn default_config_path(store: &Path) -> PathBuf {
    crate::database::snapshot::db_parent(store).join(DEFAULT_PATHMAP_FILE)
}
