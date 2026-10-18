//! Streaming conversion of a SurrealDB 2.x export into import files for the current schema.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

use super::job::{Job, PhaseId};
use super::pathmap::{Config, Mapped, Mapper, Policy, RuleStats, Status, parse_win};
use super::sq::{ExportReader, Item, RecId, V, obj_get, obj_str, obj_take, write_recid, write_value};

pub const SETTINGS_KEY: &str = "ShadowbrokerPC";
const SAMPLE_PER_BUCKET: usize = 200;
const PATH_LIKE_SETTINGS: [&str; 4] = ["recent_paths", "excluded_dirs", "last_root", "joycaption_model_dir"];

/// Import order of the converted tables.
pub const TABLE_ORDER: [&str; 9] = ["logical_groups", "filter_groups", "thumbnails", "clip_embeddings", "cached_scans", "cached_scan_items", "assistant_sessions", "assistant_messages", "user_settings"];

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub input: PathBuf,
    pub out_dir: Option<PathBuf>,
    pub drop_b64: bool,
    pub merge: bool,
    pub include_chat: bool,
    pub settings_fields: Vec<String>,
    pub settings_key: String,
    pub dim: usize,
    pub split_bytes: u64,
    pub batch_rows: usize,
    pub batch_bytes: usize,
}

impl ConvertOptions {
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            out_dir: None,
            drop_b64: false,
            merge: false,
            include_chat: true,
            settings_fields: Vec::new(),
            settings_key: SETTINGS_KEY.to_string(),
            dim: schema_dimension(crate::database::SCHEMA_DDL),
            split_bytes: 128 << 20,
            batch_rows: 100,
            batch_bytes: 6 << 20,
        }
    }
}

/// Rows already present in the target store, used for merge semantics.
#[derive(Clone, Debug, Default)]
pub struct ExistingIndex {
    pub thumb_by_path: HashMap<String, String>,
    pub thumb_ids: HashSet<String>,
    pub clip_paths: HashSet<String>,
    pub clip_ids: HashSet<String>,
    pub group_by_name: HashMap<String, String>,
    pub session_ids: HashSet<String>,
    pub message_ids: HashSet<String>,
    pub filter_names: HashSet<String>,
    pub settings_present: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConvertedFile {
    pub table: String,
    pub file: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TableReport {
    pub rows_in: u64,
    pub rows_out: u64,
    pub skipped: BTreeMap<String, u64>,
    pub fixed: BTreeMap<String, u64>,
    pub dropped_fields: BTreeMap<String, u64>,
    pub bytes_out: u64,
}

impl TableReport {
    fn skip(&mut self, why: &str) {
        *self.skipped.entry(why.into()).or_default() += 1;
    }

    fn fix(&mut self, what: &str) {
        *self.fixed.entry(what.into()).or_default() += 1;
    }

    fn drop_field(&mut self, name: &str) {
        *self.dropped_fields.entry(name.into()).or_default() += 1;
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuleReport {
    pub field: String,
    pub rule: Option<usize>,
    pub from: String,
    pub to: String,
    pub note: String,
    pub rows: u64,
    pub by_status: BTreeMap<String, u64>,
    pub exists_rate: f64,
    pub case_fixed_rows: u64,
    pub missing_top_prefixes: Vec<(String, u64)>,
    pub missing_examples: Vec<String>,
    pub case_fixed_examples: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SettingsReport {
    pub present: bool,
    pub field_names: Vec<String>,
    pub secret_field_names: Vec<String>,
    pub merged_fields: Vec<String>,
    pub merged_secret_fields: Vec<String>,
    pub unknown_requested_fields: Vec<String>,
    pub note: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ConvertReport {
    pub input: String,
    pub dry_run: bool,
    pub merge: bool,
    pub elapsed_ms: u128,
    pub dirs_listed: u64,
    pub tables: BTreeMap<String, TableReport>,
    pub other_tables_dropped: BTreeMap<String, u64>,
    pub path_rules: Vec<RuleReport>,
    pub unmapped_prefixes: BTreeMap<String, u64>,
    pub files: Vec<ConvertedFile>,
    pub settings: SettingsReport,
    pub thumbnail_rows: u64,
    pub thumbnail_rows_existing: u64,
    pub warnings: Vec<String>,
}

impl ConvertReport {
    pub fn rows_out(&self, table: &str) -> u64 {
        self.tables.get(table).map_or(0, |t| t.rows_out)
    }

    pub fn exists_rate(&self) -> f64 {
        if self.thumbnail_rows == 0 { 0.0 } else { self.thumbnail_rows_existing as f64 / self.thumbnail_rows as f64 }
    }
}

/// Fields the current schema defines per table, parsed from the DDL text.
pub fn schema_fields(ddl: &str) -> HashMap<String, HashSet<String>> {
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for line in ddl.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("DEFINE FIELD IF NOT EXISTS ") {
            let mut it = rest.split_whitespace();
            if let (Some(name), Some("ON"), Some(table)) = (it.next(), it.next(), it.next()) {
                out.entry(table.to_string()).or_default().insert(name.to_string());
            }
        }
        if let Some(rest) = l.strip_prefix("DEFINE TABLE IF NOT EXISTS ") {
            out.entry(rest.split_whitespace().next().unwrap_or("").to_string()).or_default();
        }
    }
    out
}

/// HNSW dimension declared in the DDL.
pub fn schema_dimension(ddl: &str) -> usize {
    ddl.split("DIMENSION ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok()).unwrap_or(1024)
}

pub fn is_secret_field(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["key", "token", "secret", "password", "passwd", "credential"].iter().any(|w| n.contains(w))
}

pub fn dirname(p: &str) -> String {
    match p.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => p[..i].to_string(),
        None => String::new(),
    }
}

pub fn basename(p: &str) -> &str {
    p.rsplit(['/', '\\']).next().unwrap_or(p)
}

/// Deterministic 20-character record key derived from `seed`.
pub fn derived_key(seed: &str) -> String {
    const CH: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    blake3::hash(seed.as_bytes()).as_bytes().iter().take(20).map(|b| CH[*b as usize % CH.len()] as char).collect()
}

#[derive(Debug)]
struct TableOut {
    dir: Option<PathBuf>,
    table: String,
    part: u32,
    part_bytes: u64,
    split_bytes: u64,
    w: Option<BufWriter<File>>,
    batch: Vec<u8>,
    batch_rows: usize,
    max_rows: usize,
    max_bytes: usize,
    ignore: bool,
    bytes: u64,
    files: Vec<ConvertedFile>,
    cur_name: String,
}

impl TableOut {
    fn new(dir: Option<&Path>, table: &str, o: &ConvertOptions) -> Self {
        Self {
            dir: dir.map(Path::to_path_buf),
            table: table.to_string(),
            part: 0,
            part_bytes: 0,
            split_bytes: o.split_bytes,
            w: None,
            batch: Vec::new(),
            batch_rows: 0,
            max_rows: o.batch_rows,
            max_bytes: o.batch_bytes,
            ignore: o.merge,
            bytes: 0,
            files: Vec::new(),
            cur_name: String::new(),
        }
    }

    fn open_next(&mut self) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        self.part += 1;
        let name = format!("{}.{:03}.surql", self.table, self.part);
        let path = dir.join(&name);
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut w = BufWriter::with_capacity(1 << 20, opts.open(&path).with_context(|| format!("cannot create {}", path.display()))?);
        let head = format!("-- legacy import: table {} part {}\nOPTION IMPORT;\n", self.table, self.part);
        w.write_all(head.as_bytes())?;
        self.part_bytes = head.len() as u64;
        self.bytes += head.len() as u64;
        self.w = Some(w);
        self.cur_name = name.clone();
        self.files.push(ConvertedFile { table: self.table.clone(), file: name, bytes: head.len() as u64 });
        Ok(())
    }

    fn push(&mut self, row: &V) -> Result<()> {
        if self.dir.is_none() {
            return Ok(());
        }
        if self.batch_rows == 0 {
            self.batch.extend_from_slice(if self.ignore { b"INSERT IGNORE [ " } else { b"INSERT [ " });
        } else {
            self.batch.extend_from_slice(b",\n");
        }
        write_value(&mut self.batch, row);
        self.batch_rows += 1;
        if self.batch_rows >= self.max_rows || self.batch.len() >= self.max_bytes {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn push_raw(&mut self, stmt: &[u8]) -> Result<()> {
        if self.dir.is_none() {
            return Ok(());
        }
        self.flush_batch()?;
        if self.w.is_none() {
            self.open_next()?;
        }
        self.write_out(stmt)
    }

    fn write_out(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(w) = self.w.as_mut() {
            w.write_all(bytes)?;
        }
        let n = bytes.len() as u64;
        self.part_bytes += n;
        self.bytes += n;
        if let Some(f) = self.files.last_mut() {
            f.bytes += n;
        }
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.batch_rows == 0 {
            return Ok(());
        }
        self.batch.extend_from_slice(b" ];\n");
        if self.w.is_none() || self.part_bytes >= self.split_bytes {
            if let Some(mut w) = self.w.take() {
                w.flush()?;
            }
            self.open_next()?;
        }
        let batch = std::mem::take(&mut self.batch);
        self.write_out(&batch)?;
        self.batch = batch;
        self.batch.clear();
        self.batch_rows = 0;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush_batch()?;
        if let Some(mut w) = self.w.take() {
            w.flush()?;
        }
        Ok(())
    }
}

struct Decision {
    mapped: Mapped,
    keep: bool,
    reason: Option<&'static str>,
    tag: Option<String>,
}

fn decide(mapper: &mut Mapper, path: &str) -> Decision {
    let mapped = mapper.map(path);
    let o = &mapper.cfg.options;
    let (keep, reason, tag) = match mapped.status {
        Status::Unmapped => match o.unmapped {
            Policy::Drop => (false, Some("unmapped_path"), None),
            Policy::Tag => (true, None, Some(o.unmapped_tag.clone())),
            Policy::Keep => (true, None, None),
        },
        Status::Missing => match o.missing {
            Policy::Drop => (false, Some("mapped_file_missing"), None),
            Policy::Tag => (true, None, Some(o.missing_tag.clone())),
            Policy::Keep => (true, None, None),
        },
        _ => (true, None, None),
    };
    Decision { mapped, keep, reason, tag }
}

#[derive(Clone, Debug)]
enum Dec {
    Keep(RecId),
    Redirect(RecId),
    Existing(RecId),
    Drop,
}

#[derive(Default)]
struct RuleAgg {
    stats: RuleStats,
}

struct GroupRow {
    old_key: String,
    name: String,
    created: Option<V>,
    updated: Option<V>,
}

fn thumb_rec(key: &str) -> RecId {
    RecId::named("thumbnails", key)
}

fn tally(agg: &mut BTreeMap<(String, usize), RuleAgg>, field: &str, d: &Decision, path: &str) {
    let ag = agg.entry((field.to_string(), d.mapped.rule.unwrap_or(usize::MAX))).or_default();
    ag.stats.rows += 1;
    *ag.stats.by_status.entry(d.mapped.status.name().to_string()).or_default() += 1;
    if d.mapped.status == Status::Missing {
        if let Some(fm) = &d.mapped.first_missing {
            *ag.stats.missing_by_prefix.entry(fm.clone()).or_default() += 1;
        }
        if ag.stats.missing_examples.len() < 5 {
            ag.stats.missing_examples.push(path.to_string());
        }
    }
    if d.mapped.status == Status::ExistsCaseFixed && ag.stats.case_fixed_examples.len() < 5 {
        ag.stats.case_fixed_examples.push(format!("{path} -> {}", d.mapped.path));
    }
}

fn normalize_size(v: V, c: &mut TableReport) -> V {
    match v {
        V::Num(n) if n.ends_with('f') || n.contains(['.', 'e', 'E']) => {
            let x = n.trim_end_matches('f').parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0).unwrap_or(0.0);
            c.fix("size_normalized_to_integer");
            V::Num((x.round() as u64).to_string())
        }
        other => other,
    }
}

fn map_path_value(mapper: &mut Mapper, v: V) -> V {
    match v {
        V::Str(p) => V::Str(decide(mapper, &p).mapped.path),
        V::Arr(items) => V::Arr(items.into_iter().map(|x| map_path_value(mapper, x)).collect()),
        other => other,
    }
}

fn open_reader(path: &Path) -> Result<ExportReader<File>> {
    Ok(ExportReader::new(File::open(path).with_context(|| format!("cannot open {}", path.display()))?))
}

/// Converts the export at `opts.input`; writes files when `opts.out_dir` is set, otherwise only reports.
pub fn convert(opts: &ConvertOptions, cfg: &Config, existing: Option<&ExistingIndex>, job: &Job) -> Result<ConvertReport> {
    let started = Instant::now();
    let mut mapper = Mapper::new(cfg.clone())?;
    let allowed = schema_fields(crate::database::SCHEMA_DDL);
    let none = ExistingIndex::default();
    let ex = if opts.merge { existing.unwrap_or(&none) } else { &none };
    if let Some(dir) = &opts.out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    let total_bytes = std::fs::metadata(&opts.input).with_context(|| format!("cannot read {}", opts.input.display()))?.len();
    let mut report = ConvertReport { input: opts.input.display().to_string(), dry_run: opts.out_dir.is_none(), merge: opts.merge, ..Default::default() };
    job.begin(PhaseId::Convert, total_bytes * 2, "reading the export");
    job.check_cancel()?;

    let mut groups: Vec<GroupRow> = Vec::new();
    let mut group_map: HashMap<String, RecId> = HashMap::new();
    let mut thumb_dec: HashMap<String, Dec> = HashMap::new();
    let mut seen_paths: HashMap<String, RecId> = HashMap::new();
    let mut used_ids: HashSet<String> = HashSet::new();
    let mut rule_agg: BTreeMap<(String, usize), RuleAgg> = BTreeMap::new();
    let mut unmapped_prefix: BTreeMap<String, u64> = BTreeMap::new();
    let mut thumbs_total = 0u64;
    let mut thumbs_existing = 0u64;
    let mut rows_seen = 0u64;

    let group_target = |name: &str| -> RecId {
        match ex.group_by_name.get(name) {
            Some(k) => RecId::named("logical_groups", k),
            None => RecId::named("logical_groups", name),
        }
    };

    {
        let mut rd = open_reader(&opts.input)?;
        while let Some(item) = rd.next_item()? {
            let Item::Row(V::Obj(fields)) = item else { continue };
            let Some(V::Rec(id)) = obj_get(&fields, "id") else { continue };
            rows_seen += 1;
            if rows_seen % 512 == 0 {
                job.check_cancel()?;
                job.update(PhaseId::Convert, rd.src.offset(), total_bytes * 2, "reading the export");
            }
            match id.table.as_str() {
                "logical_groups" => {
                    let name = obj_str(&fields, "name").map(str::to_string).unwrap_or_else(|| id.key.clone());
                    group_map.insert(id.key.clone(), group_target(&name));
                    groups.push(GroupRow { old_key: id.key.clone(), name, created: obj_get(&fields, "created").cloned(), updated: obj_get(&fields, "updated").cloned() });
                }
                "thumbnails" => {
                    thumbs_total += 1;
                    let Some(path) = obj_str(&fields, "path").map(str::to_string) else {
                        thumb_dec.insert(id.key.clone(), Dec::Drop);
                        continue;
                    };
                    let d = decide(&mut mapper, &path);
                    tally(&mut rule_agg, "thumbnails.path", &d, &path);
                    if matches!(d.mapped.status, Status::Exists | Status::ExistsCaseFixed) {
                        thumbs_existing += 1;
                    }
                    if d.mapped.status == Status::Unmapped {
                        let pre: String = path.split('\\').take(2).collect::<Vec<_>>().join("\\");
                        *unmapped_prefix.entry(pre).or_default() += 1;
                    }
                    if !d.keep {
                        thumb_dec.insert(id.key.clone(), Dec::Drop);
                        continue;
                    }
                    if let Some(first) = seen_paths.get(&d.mapped.path) {
                        thumb_dec.insert(id.key.clone(), Dec::Redirect(first.clone()));
                        continue;
                    }
                    if let Some(k) = ex.thumb_by_path.get(&d.mapped.path) {
                        let target = thumb_rec(k);
                        seen_paths.insert(d.mapped.path.clone(), target.clone());
                        thumb_dec.insert(id.key.clone(), Dec::Existing(target));
                        continue;
                    }
                    let final_id = if ex.thumb_ids.contains(&id.key) || used_ids.contains(&id.key) { thumb_rec(&derived_key(&d.mapped.path)) } else { id.clone() };
                    used_ids.insert(final_id.key.clone());
                    seen_paths.insert(d.mapped.path.clone(), final_id.clone());
                    thumb_dec.insert(id.key.clone(), Dec::Keep(final_id));
                }
                _ => {}
            }
        }
    }

    let mut outs: BTreeMap<String, TableOut> = BTreeMap::new();
    let mut cnt: BTreeMap<String, TableReport> = BTreeMap::new();
    let mut other_tables: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen_clip_paths: HashSet<String> = HashSet::new();
    let mut used_clip_ids: HashSet<String> = HashSet::new();
    let mut settings = SettingsReport::default();
    let mut settings_chosen = false;
    let out_dir = opts.out_dir.clone();

    macro_rules! out {
        ($t:expr) => {{
            let t: &str = $t;
            outs.entry(t.to_string()).or_insert_with(|| TableOut::new(out_dir.as_deref(), t, opts))
        }};
    }

    {
        let mut seen_names: HashSet<String> = HashSet::new();
        for g in &groups {
            let c = cnt.entry("logical_groups".into()).or_default();
            c.rows_in += 1;
            if !seen_names.insert(g.name.clone()) {
                c.skip("duplicate_group_name");
                continue;
            }
            if ex.group_by_name.contains_key(&g.name) {
                c.skip("already_in_target");
                continue;
            }
            let nid = group_map.get(&g.old_key).cloned().unwrap_or_else(|| group_target(&g.name));
            let mut f: Vec<(String, V)> = vec![("id".into(), V::Rec(nid)), ("name".into(), V::Str(g.name.clone()))];
            if let Some(v) = &g.created {
                f.push(("created".into(), v.clone()));
            }
            if let Some(v) = &g.updated {
                f.push(("updated".into(), v.clone()));
            }
            c.fix("group_id_normalized_to_name");
            c.rows_out += 1;
            out!("logical_groups").push(&V::Obj(f))?;
        }
        if !groups.iter().any(|g| g.name == "Default") && !ex.group_by_name.contains_key("Default") {
            let f = vec![("id".into(), V::Rec(RecId::named("logical_groups", "Default"))), ("name".into(), V::Str("Default".into()))];
            let c = cnt.entry("logical_groups".into()).or_default();
            c.fix("default_group_created");
            c.rows_out += 1;
            out!("logical_groups").push(&V::Obj(f))?;
        }
    }
    let default_group = group_target("Default");

    let mut rd = open_reader(&opts.input)?;
    let mut seen_rows = 0u64;
    while let Some(item) = rd.next_item()? {
        let Item::Row(V::Obj(mut fields)) = item else { continue };
        let Some(V::Rec(id)) = obj_get(&fields, "id").cloned() else { continue };
        let table = id.table.clone();
        if table == "logical_groups" {
            continue;
        }
        seen_rows += 1;
        if seen_rows % 256 == 0 {
            job.check_cancel()?;
            job.update(PhaseId::Convert, total_bytes + rd.src.offset(), total_bytes * 2, "writing import files");
        }
        let c = cnt.entry(table.clone()).or_default();
        c.rows_in += 1;
        let allow = allowed.get(&table);
        let strip = |k: &str, c: &mut TableReport| -> bool {
            match allow {
                Some(al) if k != "id" && !al.contains(k) => {
                    c.drop_field(k);
                    true
                }
                _ => false,
            }
        };

        match table.as_str() {
            "thumbnails" => {
                let final_id = match thumb_dec.get(&id.key) {
                    Some(Dec::Keep(fid)) => fid.clone(),
                    Some(Dec::Redirect(_)) => {
                        c.skip("duplicate_mapped_path");
                        continue;
                    }
                    Some(Dec::Existing(_)) => {
                        c.skip("already_in_target");
                        continue;
                    }
                    _ => {
                        c.skip("path_policy_or_no_path");
                        continue;
                    }
                };
                let path = obj_str(&fields, "path").unwrap_or_default().to_string();
                let d = decide(&mut mapper, &path);
                let mapped_path = d.mapped.path.clone();
                let mut row: Vec<(String, V)> = vec![("id".into(), V::Rec(final_id))];
                let mut tags: Vec<V> = match obj_take(&mut fields, "tags") {
                    Some(V::Arr(t)) => t,
                    _ => {
                        c.fix("tags_defaulted_empty");
                        Vec::new()
                    }
                };
                if let Some(t) = &d.tag {
                    tags.push(V::Str(t.clone()));
                    c.fix("tagged_missing_or_unmapped");
                }
                let old_group = obj_get(&fields, "logical_group").cloned();
                let old_parent = obj_str(&fields, "parent_dir").map(str::to_string);
                let (mut have_size, mut have_ft) = (false, false);
                for (k, v) in fields {
                    match k.as_str() {
                        "id" | "path" | "parent_dir" | "logical_group" => continue,
                        "thumbnail_b64" if opts.drop_b64 => {
                            c.fix("thumbnail_b64_omitted");
                            continue;
                        }
                        "filename" => {
                            if d.mapped.status == Status::ExistsCaseFixed {
                                row.push((k, V::Str(basename(&mapped_path).to_string())));
                                c.fix("filename_case_fixed");
                            } else {
                                row.push((k, v));
                            }
                            continue;
                        }
                        "size" => {
                            have_size = true;
                            row.push((k, normalize_size(v, c)));
                            continue;
                        }
                        "file_type" => have_ft = true,
                        _ => {}
                    }
                    if strip(&k, c) {
                        continue;
                    }
                    row.push((k, v));
                }
                if !have_size {
                    let real = matches!(d.mapped.status, Status::Exists | Status::ExistsCaseFixed).then(|| std::fs::metadata(&mapped_path).map(|m| m.len()).ok()).flatten();
                    match real {
                        Some(n) => {
                            row.push(("size".into(), V::Num(n.to_string())));
                            c.fix("size_from_disk");
                        }
                        None => {
                            row.push(("size".into(), V::Num("0".into())));
                            c.fix("size_defaulted_zero");
                        }
                    }
                }
                if !have_ft {
                    let ext = basename(&mapped_path).rsplit_once('.').map(|x| x.1.to_lowercase()).unwrap_or_default();
                    row.push(("file_type".into(), V::Str(ext)));
                    c.fix("file_type_from_extension");
                }
                row.push(("path".into(), V::Str(mapped_path.clone())));
                let parent = match d.mapped.status {
                    Status::Unmapped | Status::Native => old_parent.unwrap_or_default(),
                    _ => dirname(&mapped_path),
                };
                row.push(("parent_dir".into(), V::Str(parent)));
                let lg = match old_group {
                    Some(V::Rec(r)) => {
                        if r.key == "Default" {
                            default_group.clone()
                        } else if let Some(n) = group_map.get(&r.key) {
                            c.fix("logical_group_remapped");
                            n.clone()
                        } else {
                            c.fix("logical_group_unknown_to_default");
                            default_group.clone()
                        }
                    }
                    _ => {
                        c.fix("logical_group_defaulted");
                        default_group.clone()
                    }
                };
                row.push(("logical_group".into(), V::Rec(lg)));
                row.push(("tags".into(), V::Arr(tags)));
                c.fix(match d.mapped.status {
                    Status::Exists => "path_remapped_exists",
                    Status::ExistsCaseFixed => "path_remapped_case_fixed",
                    Status::Missing => "path_remapped_file_missing",
                    Status::Unmapped => "path_kept_unmapped",
                    Status::Native => "path_native",
                });
                c.rows_out += 1;
                out!("thumbnails").push(&V::Obj(row))?;
            }
            "clip_embeddings" => {
                let Some(path) = obj_str(&fields, "path").map(str::to_string) else {
                    c.skip("no_path");
                    continue;
                };
                let d = decide(&mut mapper, &path);
                if !d.keep {
                    c.skip(d.reason.unwrap_or("path_policy"));
                    continue;
                }
                let mapped = d.mapped.path.clone();
                if ex.clip_paths.contains(&mapped) {
                    c.skip("already_in_target");
                    continue;
                }
                if !seen_clip_paths.insert(mapped.clone()) {
                    c.skip("duplicate_mapped_path");
                    continue;
                }
                match obj_get(&fields, "embedding") {
                    Some(V::Arr(e)) if e.len() == opts.dim && e.iter().all(|x| matches!(x, V::Num(_))) => {}
                    Some(V::Arr(_)) => {
                        c.skip("wrong_embedding_dimension");
                        continue;
                    }
                    _ => {
                        c.skip("no_embedding");
                        continue;
                    }
                }
                let final_id = if ex.clip_ids.contains(&id.key) || used_clip_ids.contains(&id.key) { RecId::named("clip_embeddings", &derived_key(&mapped)) } else { id.clone() };
                used_clip_ids.insert(final_id.key.clone());
                let mut row: Vec<(String, V)> = vec![("id".into(), V::Rec(final_id))];
                let mut linked = false;
                for (k, v) in fields {
                    match k.as_str() {
                        "id" | "path" => continue,
                        "thumb_ref" => {
                            if let V::Rec(r) = v {
                                match thumb_dec.get(&r.key) {
                                    Some(Dec::Keep(t)) => {
                                        row.push((k, V::Rec(t.clone())));
                                        linked = true;
                                    }
                                    Some(Dec::Redirect(t)) | Some(Dec::Existing(t)) => {
                                        row.push((k, V::Rec(t.clone())));
                                        linked = true;
                                        c.fix("thumb_ref_redirected_to_existing_row");
                                    }
                                    _ => c.fix("thumb_ref_dropped_target_removed"),
                                }
                            }
                            continue;
                        }
                        _ => {}
                    }
                    if strip(&k, c) {
                        continue;
                    }
                    row.push((k, v));
                }
                if !linked {
                    if let Some(t) = seen_paths.get(&mapped) {
                        row.push(("thumb_ref".into(), V::Rec(t.clone())));
                        c.fix("thumb_ref_linked_by_path");
                    }
                }
                row.push(("path".into(), V::Str(mapped)));
                c.rows_out += 1;
                out!("clip_embeddings").push(&V::Obj(row))?;
            }
            "cached_scans" | "cached_scan_items" => {
                let key = if table == "cached_scans" { "root" } else { "path" };
                let mut row: Vec<(String, V)> = Vec::new();
                for (k, v) in fields {
                    if k == key {
                        if let V::Str(p) = &v {
                            row.push((k, V::Str(decide(&mut mapper, p).mapped.path)));
                            continue;
                        }
                    }
                    if strip(&k, c) {
                        continue;
                    }
                    row.push((k, v));
                }
                c.rows_out += 1;
                out!(&table).push(&V::Obj(row))?;
            }
            "assistant_sessions" | "assistant_messages" | "filter_groups" => {
                if !opts.include_chat && table.starts_with("assistant") {
                    c.skip("chat_skipped_by_option");
                    continue;
                }
                let already = match table.as_str() {
                    "assistant_sessions" => ex.session_ids.contains(&id.key),
                    "assistant_messages" => ex.message_ids.contains(&id.key),
                    _ => obj_str(&fields, "name").is_some_and(|n| ex.filter_names.contains(n)),
                };
                if already {
                    c.skip("already_in_target");
                    continue;
                }
                let mut row: Vec<(String, V)> = Vec::new();
                for (k, v) in fields {
                    if table == "assistant_messages" && k == "attachments" {
                        if let V::Arr(items) = v {
                            let mapped: Vec<V> = items.into_iter().map(|x| match x {
                                V::Str(p) => V::Str(decide(&mut mapper, &p).mapped.path),
                                o => o,
                            }).collect();
                            row.push((k, V::Arr(mapped)));
                            c.fix("attachment_paths_remapped");
                            continue;
                        }
                    }
                    if table == "assistant_messages" && k == "attachments_refs" {
                        if let V::Arr(items) = v {
                            let before = items.len();
                            let kept: Vec<V> = items.into_iter().filter_map(|x| match x {
                                V::Rec(r) => match thumb_dec.get(&r.key) {
                                    Some(Dec::Keep(t)) | Some(Dec::Redirect(t)) | Some(Dec::Existing(t)) => Some(V::Rec(t.clone())),
                                    _ => None,
                                },
                                o => Some(o),
                            }).collect();
                            if kept.len() != before {
                                c.fix("attachment_refs_pruned");
                            }
                            row.push((k, V::Arr(kept)));
                            continue;
                        }
                    }
                    if strip(&k, c) {
                        continue;
                    }
                    row.push((k, v));
                }
                c.rows_out += 1;
                out!(&table).push(&V::Obj(row))?;
            }
            "user_settings" => {
                let names: Vec<String> = fields.iter().map(|(k, _)| k.clone()).filter(|k| k != "id").collect();
                if settings_chosen && id.key != opts.settings_key {
                    c.skip("settings_extra_record");
                    continue;
                }
                settings_chosen = true;
                settings.present = true;
                settings.secret_field_names = names.iter().filter(|n| is_secret_field(n)).cloned().collect();
                settings.field_names = names;
                if opts.settings_fields.is_empty() {
                    c.skip("settings_skipped_keep_target_settings");
                    settings.note = "user settings are not imported; pass --settings fields:<names> to merge selected fields".into();
                    continue;
                }
                if opts.merge && !ex.settings_present {
                    c.skip("settings_target_missing");
                    settings.note = "the target store has no user settings record to merge into".into();
                    continue;
                }
                if !opts.merge {
                    c.skip("settings_no_target_in_new_store");
                    settings.note = "a new store has no settings record to merge into; start the app once, then merge with --merge".into();
                    continue;
                }
                let mut sel: Vec<(String, V)> = Vec::new();
                for (k, v) in fields {
                    if k == "id" || !opts.settings_fields.iter().any(|f| *f == k) {
                        continue;
                    }
                    let nv = if PATH_LIKE_SETTINGS.contains(&k.as_str()) { map_path_value(&mut mapper, v) } else { v };
                    if is_secret_field(&k) {
                        settings.merged_secret_fields.push(k.clone());
                    }
                    settings.merged_fields.push(k.clone());
                    sel.push((k, nv));
                }
                settings.unknown_requested_fields = opts.settings_fields.iter().filter(|f| !settings.merged_fields.contains(f)).cloned().collect();
                if !sel.is_empty() {
                    let mut stmt: Vec<u8> = b"UPDATE ".to_vec();
                    write_recid(&mut stmt, &RecId::named("user_settings", &opts.settings_key));
                    stmt.extend_from_slice(b" MERGE ");
                    write_value(&mut stmt, &V::Obj(sel));
                    stmt.extend_from_slice(b";\n");
                    c.fix("settings_fields_merge_statement");
                    c.rows_out += 1;
                    out!("user_settings").push_raw(&stmt)?;
                }
            }
            other => {
                *other_tables.entry(other.to_string()).or_default() += 1;
                c.skip("table_not_in_current_schema");
            }
        }
    }

    for o in outs.values_mut() {
        o.finish()?;
    }
    for (t, o) in &outs {
        let c = cnt.entry(t.clone()).or_default();
        c.bytes_out = o.bytes;
    }
    if !settings.unknown_requested_fields.is_empty() {
        report.warnings.push(format!("requested settings fields not found in the legacy record: {}", settings.unknown_requested_fields.join(", ")));
    }
    if !settings.merged_secret_fields.is_empty() {
        report.warnings.push(format!("secret settings fields were merged because they were named: {}", settings.merged_secret_fields.join(", ")));
    }

    let mut grouped: BTreeMap<usize, Vec<(&String, &RuleAgg)>> = BTreeMap::new();
    for ((field, ri), ag) in &rule_agg {
        grouped.entry(*ri).or_default().push((field, ag));
    }
    for (ri, list) in &grouped {
        let (from, to, note) = match cfg.rules.get(*ri) {
            Some(r) => (r.from.clone(), r.to.clone(), r.note.clone()),
            None => ("(no rule matched)".to_string(), String::new(), String::new()),
        };
        for (field, ag) in list {
            let s = &ag.stats;
            let n = |k: &str| s.by_status.get(k).copied().unwrap_or(0);
            let mut mp: Vec<(String, u64)> = s.missing_by_prefix.iter().map(|(k, v)| (k.clone(), *v)).collect();
            mp.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            mp.truncate(15);
            report.path_rules.push(RuleReport {
                field: (*field).clone(),
                rule: (*ri != usize::MAX).then_some(*ri),
                from: from.clone(),
                to: to.clone(),
                note: note.clone(),
                rows: s.rows,
                by_status: s.by_status.clone(),
                exists_rate: if s.rows > 0 { (n("exists") + n("exists_case_fixed")) as f64 / s.rows as f64 } else { 0.0 },
                case_fixed_rows: n("exists_case_fixed"),
                missing_top_prefixes: mp,
                missing_examples: s.missing_examples.clone(),
                case_fixed_examples: s.case_fixed_examples.clone(),
            });
        }
    }
    report.files = TABLE_ORDER.iter().filter_map(|t| outs.get(*t)).flat_map(|o| o.files.clone()).collect();
    if let Some(dir) = &opts.out_dir {
        let names: Vec<&str> = report.files.iter().map(|f| f.file.as_str()).collect();
        std::fs::write(dir.join("import-order.json"), serde_json::to_string_pretty(&names)?)?;
    }
    report.tables = cnt;
    report.other_tables_dropped = other_tables;
    report.unmapped_prefixes = unmapped_prefix;
    report.settings = settings;
    report.thumbnail_rows = thumbs_total;
    report.thumbnail_rows_existing = thumbs_existing;
    report.dirs_listed = mapper.dirs_listed;
    report.elapsed_ms = started.elapsed().as_millis();
    job.finish(PhaseId::Convert);
    Ok(report)
}

#[derive(Clone, Debug, Serialize)]
pub struct Bucket {
    pub root: String,
    pub first: String,
    pub rows: u64,
    pub samples: Vec<(String, Option<u64>)>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Analysis {
    pub tables: BTreeMap<String, u64>,
    pub buckets: Vec<Bucket>,
    pub settings_fields: Vec<String>,
    pub settings_paths: Vec<String>,
    pub native_paths: u64,
    pub unparsed_paths: u64,
}

fn collect_paths(v: &V, out: &mut Vec<String>) {
    match v {
        V::Str(s) if parse_win(s).is_some() => out.push(s.clone()),
        V::Arr(items) => items.iter().for_each(|i| collect_paths(i, out)),
        V::Obj(fields) => fields.iter().for_each(|(_, i)| collect_paths(i, out)),
        _ => {}
    }
}

/// One cheap pass over the export: table row counts, Windows path buckets with samples, settings field names.
pub fn analyze(input: &Path, job: &Job) -> Result<Analysis> {
    let total = std::fs::metadata(input).with_context(|| format!("cannot read {}", input.display()))?.len();
    job.check_cancel()?;
    let mut rd = open_reader(input)?;
    let mut a = Analysis::default();
    let mut buckets: HashMap<(String, String), Bucket> = HashMap::new();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut seen = 0u64;
    while let Some(item) = rd.next_item()? {
        let Item::Row(V::Obj(fields)) = item else { continue };
        let Some(V::Rec(id)) = obj_get(&fields, "id") else { continue };
        *a.tables.entry(id.table.clone()).or_default() += 1;
        seen += 1;
        if seen % 1024 == 0 {
            job.check_cancel()?;
            job.update(PhaseId::Convert, rd.src.offset(), total.max(1), "analysing the export");
        }
        match id.table.as_str() {
            "thumbnails" => {
                let Some(p) = obj_str(&fields, "path") else { continue };
                let size = match obj_get(&fields, "size") {
                    Some(V::Num(n)) => n.trim_end_matches('f').parse::<f64>().ok().map(|f| f as u64),
                    _ => None,
                };
                match parse_win(p) {
                    Some(w) => {
                        let first = if w.comps.len() > 1 { w.comps[0].clone() } else { String::new() };
                        let b = buckets.entry((w.root.clone(), first.to_lowercase())).or_insert_with(|| Bucket { root: w.root.clone(), first: first.clone(), rows: 0, samples: Vec::new() });
                        b.rows += 1;
                        if b.samples.len() < SAMPLE_PER_BUCKET {
                            b.samples.push((p.to_string(), size));
                        } else {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            let j = (rng >> 33) % b.rows;
                            if (j as usize) < SAMPLE_PER_BUCKET {
                                b.samples[j as usize] = (p.to_string(), size);
                            }
                        }
                    }
                    None if p.starts_with('/') || p.contains("://") => a.native_paths += 1,
                    None => a.unparsed_paths += 1,
                }
            }
            "user_settings" => {
                let mut names: Vec<String> = fields.iter().map(|(k, _)| k.clone()).filter(|k| k != "id").collect();
                names.sort();
                a.settings_fields = names;
                for (_, v) in &fields {
                    collect_paths(v, &mut a.settings_paths);
                }
            }
            _ => {}
        }
    }
    let mut list: Vec<Bucket> = buckets.into_values().collect();
    list.sort_by(|x, y| y.rows.cmp(&x.rows).then(x.root.cmp(&y.root)).then(x.first.cmp(&y.first)));
    a.buckets = list;
    a.settings_paths.sort();
    a.settings_paths.dedup();
    job.finish(PhaseId::Convert);
    Ok(a)
}
