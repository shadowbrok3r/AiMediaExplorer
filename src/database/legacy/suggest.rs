//! Suggests a Windows to local path map from mounted volumes, store evidence and size-matched sample files.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::convert::{Analysis, Bucket};
use super::pathmap::{Config, Mapper, Options, Rule, Status, parse_win};

const CONFIRM_RATE: f64 = 0.5;
const DATA_FS: [&str; 7] = ["ntfs", "ntfs3", "fuseblk", "exfat", "vfat", "msdos", "hfsplus"];
const MOUNT_ROOTS: [&str; 4] = ["/run/media/", "/media/", "/mnt/", "/Volumes/"];

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Volume {
    pub mount: PathBuf,
    pub fstype: String,
    pub device: String,
    pub label: Option<String>,
    pub uuid: Option<String>,
}

impl Volume {
    pub fn is_automount(&self) -> bool {
        is_automount(&self.mount)
    }

    pub fn display_name(&self) -> String {
        self.label.clone().unwrap_or_else(|| self.mount.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
    }
}

/// True for udisks-style mount points whose name follows the volume label.
pub fn is_automount(path: &Path) -> bool {
    let p = path.to_string_lossy();
    p.starts_with("/run/media/") || p.starts_with("/media/") || p.starts_with("/Volumes/")
}

fn unescape_mount(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
            let v = (b[i + 1] - b'0') as u32 * 64 + (b[i + 2] - b'0') as u32 * 8 + (b[i + 3] - b'0') as u32;
            out.push(v as u8);
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parses `/proc/self/mounts` text into candidate data volumes.
pub fn parse_mounts(text: &str) -> Vec<Volume> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(dev), Some(mnt), Some(fs)) = (it.next(), it.next(), it.next()) else { continue };
        let mount = unescape_mount(mnt);
        let under_root = MOUNT_ROOTS.iter().any(|r| mount.starts_with(r));
        let data_fs = DATA_FS.contains(&fs) && mount != "/boot" && !mount.starts_with("/boot/");
        if (under_root || data_fs) && dev.starts_with('/') {
            out.push(Volume { mount: PathBuf::from(mount), fstype: fs.to_string(), device: unescape_mount(dev), label: None, uuid: None });
        }
    }
    out.sort_by(|a, b| a.mount.cmp(&b.mount));
    out.dedup_by(|a, b| a.mount == b.mount);
    out
}

fn enrich_with_lsblk(vols: &mut [Volume]) {
    let Ok(out) = std::process::Command::new("lsblk").args(["-J", "-o", "PATH,LABEL,UUID,MOUNTPOINT"]).output() else { return };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else { return };
    let Some(devs) = v.get("blockdevices").and_then(|d| d.as_array()) else { return };
    let mut stack: Vec<&serde_json::Value> = devs.iter().collect();
    while let Some(d) = stack.pop() {
        if let Some(ch) = d.get("children").and_then(|c| c.as_array()) {
            stack.extend(ch.iter());
        }
        let Some(mp) = d.get("mountpoint").and_then(|m| m.as_str()) else { continue };
        if let Some(vol) = vols.iter_mut().find(|x| x.mount == Path::new(mp)) {
            vol.label = d.get("label").and_then(|l| l.as_str()).map(str::to_string);
            vol.uuid = d.get("uuid").and_then(|l| l.as_str()).map(str::to_string);
        }
    }
}

/// Mounted data volumes on this machine (Linux and macOS); empty elsewhere.
pub fn list_volumes() -> Vec<Volume> {
    #[cfg(target_os = "linux")]
    {
        let mut v = std::fs::read_to_string("/proc/self/mounts").map(|t| parse_mounts(&t)).unwrap_or_default();
        enrich_with_lsblk(&mut v);
        v
    }
    #[cfg(target_os = "macos")]
    {
        std::fs::read_dir("/Volumes")
            .map(|rd| rd.flatten().map(|e| Volume { mount: e.path(), fstype: String::new(), device: String::new(), label: Some(e.file_name().to_string_lossy().into_owned()), uuid: None }).collect())
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Vec::new()
    }
}

/// Volume whose mount point contains `path`, longest mount wins.
pub fn volume_of<'a>(path: &Path, vols: &'a [Volume]) -> Option<&'a Volume> {
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    vols.iter().filter(|v| canon.starts_with(&v.mount)).max_by_key(|v| v.mount.components().count())
}

/// Windows paths embedded in raw bytes (manifest files), longest printable runs after `X:\`.
pub fn windows_paths_in_bytes(bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if boundary && bytes[i].is_ascii_alphabetic() && bytes[i + 1] == b':' && bytes[i + 2] == b'\\' {
            let mut j = i;
            while j < bytes.len() && (0x20..0x7f).contains(&bytes[j]) {
                j += 1;
            }
            if let Ok(s) = std::str::from_utf8(&bytes[i..j]) {
                out.push(s.to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Windows paths recorded in an old store's manifest files.
pub fn manifest_paths(store: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for dir in ["manifest", "."] {
        let Ok(rd) = std::fs::read_dir(store.join(dir)) else { continue };
        for e in rd.flatten() {
            let md = e.metadata();
            if md.as_ref().is_ok_and(|m| m.is_file() && m.len() <= 4 << 20) && e.path().extension().is_some_and(|x| x == "manifest" || x == "json") {
                if let Ok(b) = std::fs::read(e.path()) {
                    out.extend(windows_paths_in_bytes(&b));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

#[derive(Clone, Debug, Default)]
pub struct Evidence {
    pub hints: Vec<String>,
    pub old_store_volume: Option<PathBuf>,
    pub home: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub from: String,
    pub to: String,
    pub hits: u32,
    pub sampled: u32,
    pub via: String,
}

impl Candidate {
    pub fn rate(&self) -> f64 {
        if self.sampled == 0 { 0.0 } else { self.hits as f64 / self.sampled as f64 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BucketResult {
    pub root: String,
    pub first: String,
    pub rows: u64,
    pub chosen: Option<Candidate>,
    pub confirmed: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Suggestion {
    pub config: Config,
    pub buckets: Vec<BucketResult>,
    pub warnings: Vec<String>,
    pub volumes: Vec<Volume>,
    pub unresolved: Vec<String>,
}

fn probe(from: &str, to: &str, samples: &[(String, Option<u64>)]) -> (u32, u32) {
    let cfg = Config { options: Options::default(), rules: vec![Rule { from: from.to_string(), to: to.to_string(), note: String::new() }] };
    let Ok(mut mapper) = Mapper::new(cfg) else { return (0, samples.len() as u32) };
    let mut hits = 0;
    for (p, size) in samples {
        let m = mapper.map(p);
        if matches!(m.status, Status::Exists | Status::ExistsCaseFixed) {
            let ok = match size {
                Some(s) => std::fs::metadata(&m.path).map(|md| md.len() == *s).unwrap_or(false),
                None => true,
            };
            if ok {
                hits += 1;
            }
        }
    }
    (hits, samples.len() as u32)
}

fn child_dir(parent: &Path, name: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir(parent).ok()?;
    let mut ci = None;
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n == name {
            return Some(e.path());
        }
        if n.eq_ignore_ascii_case(name) && e.path().is_dir() {
            ci = Some(e.path());
        }
    }
    ci
}

fn bucket_from(b: &Bucket) -> String {
    if b.first.is_empty() { format!("{}\\", b.root) } else { format!("{}\\{}", b.root, b.first) }
}

fn candidates_for(b: &Bucket, targets: &[(PathBuf, String)], home: Option<&Path>) -> Vec<Candidate> {
    let from = bucket_from(b);
    let mut list = Vec::new();
    for (dir, via) in targets {
        let mut tos: Vec<(PathBuf, String)> = vec![(dir.clone(), format!("{via} root"))];
        if !b.first.is_empty() {
            if let Some(c) = child_dir(dir, &b.first) {
                tos.push((c, format!("{via} subfolder")));
            }
        }
        for (to, how) in tos {
            let to_s = to.to_string_lossy().into_owned();
            let (hits, sampled) = probe(&from, &to_s, &b.samples);
            list.push(Candidate { from: from.clone(), to: to_s, hits, sampled, via: how });
        }
    }
    if let (Some(home), true) = (home, b.first.eq_ignore_ascii_case("Users")) {
        let mut users: Vec<String> = b.samples.iter().filter_map(|(p, _)| parse_win(p)).filter_map(|w| w.comps.get(1).cloned()).collect();
        users.sort();
        users.dedup();
        for u in users {
            let from_u = format!("{}\\{}\\{}", b.root, b.first, u);
            let sub: Vec<(String, Option<u64>)> = b.samples.iter().filter(|(p, _)| parse_win(p).is_some_and(|w| w.comps.get(1).is_some_and(|c| c.eq_ignore_ascii_case(&u)))).cloned().collect();
            let to_s = home.to_string_lossy().into_owned();
            let (hits, sampled) = probe(&from_u, &to_s, &sub);
            list.push(Candidate { from: from_u, to: to_s, hits, sampled, via: "home folder".into() });
        }
    }
    list
}

/// Builds a path map: confirmed per-folder rules first, then drive rules for the folders that stay unresolved.
pub fn suggest(analysis: &Analysis, volumes: &[Volume], evidence: &Evidence) -> Suggestion {
    let mut targets: Vec<(PathBuf, String)> = volumes.iter().map(|v| (v.mount.clone(), format!("volume {}", v.display_name()))).collect();
    if let Some(m) = &evidence.old_store_volume {
        if !targets.iter().any(|(p, _)| p == m) {
            targets.push((m.clone(), "volume holding the old store".into()));
        }
    }
    let mut out = Suggestion { volumes: volumes.to_vec(), ..Default::default() };
    let mut rules: Vec<Rule> = Vec::new();
    let mut weak: Vec<&Bucket> = Vec::new();
    for b in &analysis.buckets {
        let mut cands = candidates_for(b, &targets, evidence.home.as_deref());
        cands.sort_by(|x, y| y.rate().total_cmp(&x.rate()).then(y.hits.cmp(&x.hits)));
        let best = cands.into_iter().next().filter(|c| c.hits > 0);
        let confirmed = best.as_ref().is_some_and(|c| c.rate() >= CONFIRM_RATE);
        if let (true, Some(c)) = (confirmed, &best) {
            rules.push(Rule { from: c.from.clone(), to: c.to.clone(), note: format!("{} rows; {} of {} sampled files exist with the recorded size ({})", b.rows, c.hits, c.sampled, c.via) });
        } else {
            weak.push(b);
        }
        out.buckets.push(BucketResult { root: b.root.clone(), first: b.first.clone(), rows: b.rows, chosen: best, confirmed });
    }
    let mut roots: Vec<String> = weak.iter().map(|b| b.root.clone()).collect();
    roots.sort();
    roots.dedup();
    for root in roots {
        let samples: Vec<(String, Option<u64>)> = weak.iter().filter(|b| b.root == root).flat_map(|b| b.samples.clone()).collect();
        let rows: u64 = weak.iter().filter(|b| b.root == root).map(|b| b.rows).sum();
        let from = format!("{root}\\");
        if rules.iter().any(|r| r.from.eq_ignore_ascii_case(&from)) {
            continue;
        }
        let mut best: Option<Candidate> = None;
        for (dir, via) in &targets {
            let (hits, sampled) = probe(&from, &dir.to_string_lossy(), &samples);
            if hits > 0 && best.as_ref().is_none_or(|b| hits > b.hits) {
                best = Some(Candidate { from: from.clone(), to: dir.to_string_lossy().into_owned(), hits, sampled, via: via.clone() });
            }
        }
        let by_hint = || {
            evidence.hints.iter().filter_map(|h| parse_win(h)).filter(|w| w.root == root).find_map(|w| {
                let depth: Vec<&String> = w.comps.iter().take(2).collect();
                targets.iter().find(|(dir, _)| {
                    let mut cur = dir.clone();
                    !depth.is_empty() && depth.iter().all(|c| child_dir(&cur.clone(), c).map(|n| cur = n).is_some())
                }).map(|(dir, via)| Candidate { from: from.clone(), to: dir.to_string_lossy().into_owned(), hits: 0, sampled: samples.len() as u32, via: format!("{via}, matches a path recorded in the old store") })
            })
        };
        match best.or_else(by_hint) {
            Some(c) => {
                let note = if c.hits > 0 {
                    format!("{rows} rows in unresolved folders; {} of {} sampled files exist with the recorded size ({})", c.hits, c.sampled, c.via)
                } else {
                    format!("{rows} rows in unresolved folders; chosen from path evidence only ({})", c.via)
                };
                rules.push(Rule { from: c.from.clone(), to: c.to.clone(), note });
            }
            None => out.unresolved.push(format!("{root}\\ ({rows} rows)")),
        }
    }
    rules.sort_by(|a, b| a.from.to_lowercase().cmp(&b.from.to_lowercase()));
    out.warnings = stability_warnings(&rules, volumes);
    out.config = Config { options: Options::default(), rules };
    out
}

fn stability_warnings(rules: &[Rule], volumes: &[Volume]) -> Vec<String> {
    let mut w = Vec::new();
    for r in rules {
        if is_automount(Path::new(&r.to)) {
            let vol = volumes.iter().find(|v| Path::new(&r.to).starts_with(&v.mount));
            let hint = vol
                .and_then(|v| v.uuid.as_ref().map(|u| format!(" (volume UUID {u})")))
                .unwrap_or_default();
            w.push(format!("{} is an automount path{hint}; its name can change when the volume label changes or collides. Mount the volume at a fixed place (fstab) or create a symlink and use that path as `to`.", r.to));
        }
    }
    w.dedup();
    w
}

fn toml_str(s: &str) -> String {
    toml::Value::String(s.to_string()).to_string()
}

/// Renders the editable `legacy-pathmap.toml` with explanatory comments.
pub fn render_toml(s: &Suggestion) -> String {
    let mut t = String::new();
    t.push_str("# Windows to local path map for the legacy database import.\n");
    t.push_str("# Rules match path components case-insensitively; the longest matching prefix wins; an empty `to` leaves matching rows unmapped.\n");
    t.push_str("# Review this file, then run the import again. Thumbnail blobs are keyed by the stored path, so settle the map before migrating thumbnails.\n");
    for w in &s.warnings {
        t.push_str(&format!("# WARNING: {w}\n"));
    }
    t.push_str("\n[options]\n");
    let o = &s.config.options;
    t.push_str(&format!("unmapped = {}\nmissing = {}\nmissing_tag = {}\nunmapped_tag = {}\ncase_fix = {}\n", toml_str(o.unmapped.label()), toml_str(o.missing.label()), toml_str(&o.missing_tag), toml_str(&o.unmapped_tag), o.case_fix));
    for r in &s.config.rules {
        t.push_str("\n[[rule]]\n");
        t.push_str(&format!("from = {}\nto = {}\n", toml_str(&r.from), toml_str(&r.to)));
        if !r.note.is_empty() {
            t.push_str(&format!("note = {}\n", toml_str(&r.note)));
        }
    }
    for u in &s.unresolved {
        t.push_str(&format!("\n# No location found for {u}; add a rule if the files still exist somewhere.\n"));
    }
    if !s.volumes.is_empty() {
        t.push_str("\n# Volumes seen when this file was generated:\n");
        for v in &s.volumes {
            t.push_str(&format!("#   {} ({}{}{})\n", v.mount.display(), if v.fstype.is_empty() { "?" } else { &v.fstype }, v.label.as_ref().map(|l| format!(", label {l}")).unwrap_or_default(), v.uuid.as_ref().map(|u| format!(", uuid {u}")).unwrap_or_default()));
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mounts_are_filtered_and_unescaped() {
        let text = "/dev/nvme0n1p2 / ext4 rw 0 0\n/dev/sda2 /run/media/u/HDD\\040Games ntfs3 rw 0 0\ntmpfs /run/user/1000 tmpfs rw 0 0\n/dev/sdb1 /mnt/data ext4 rw 0 0\n/dev/sdc1 /boot/efi vfat rw 0 0\n/dev/sdd1 /srv/ntfs ntfs3 rw 0 0\n";
        let v = parse_mounts(text);
        let m: Vec<String> = v.iter().map(|x| x.mount.display().to_string()).collect();
        assert_eq!(m, vec!["/mnt/data", "/run/media/u/HDD Games", "/srv/ntfs"]);
        assert!(v[1].is_automount());
        assert!(!v[0].is_automount());
    }

    #[test]
    fn manifest_scan_finds_windows_paths() {
        let bytes = b"\x00\x01db\\ai_search\x00\x07/G:\\Users\\Owner\\Desktop\\smart-media\\db\\ai_search\x00xx E:\\Backups\\a\x00";
        let p = windows_paths_in_bytes(bytes);
        assert!(p.contains(&"G:\\Users\\Owner\\Desktop\\smart-media\\db\\ai_search".to_string()), "{p:?}");
        assert!(p.contains(&"E:\\Backups\\a".to_string()));
    }

    fn bucket(root: &str, first: &str, files: &[(&str, Option<u64>)]) -> Bucket {
        Bucket { root: root.into(), first: first.into(), rows: files.len() as u64, samples: files.iter().map(|(p, s)| (p.to_string(), *s)).collect() }
    }

    #[test]
    fn size_matched_folders_become_specific_rules_and_weak_ones_a_drive_rule() {
        let vol = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(vol.path().join("Backups/photos")).unwrap();
        std::fs::write(vol.path().join("Backups/photos/a.jpg"), vec![0u8; 10]).unwrap();
        std::fs::write(vol.path().join("Backups/photos/b.jpg"), vec![0u8; 20]).unwrap();
        std::fs::create_dir_all(vol.path().join("Users/Owner/Pictures")).unwrap();
        std::fs::write(vol.path().join("Users/Owner/Pictures/c.jpg"), vec![0u8; 5]).unwrap();
        let analysis = Analysis {
            buckets: vec![
                bucket("E:", "Backups", &[("E:\\backups\\photos\\a.jpg", Some(10)), ("E:\\Backups\\photos\\b.jpg", Some(20))]),
                bucket("G:", "Users", &[("G:\\Users\\Owner\\Pictures\\c.jpg", Some(5)), ("G:\\Users\\Owner\\Comfy\\gone1.png", Some(1)), ("G:\\Users\\Owner\\Comfy\\gone2.png", Some(1)), ("G:\\Users\\Owner\\Comfy\\gone3.png", Some(1))]),
                bucket("I:", "DCIM", &[("I:\\DCIM\\x.jpg", Some(3))]),
            ],
            ..Default::default()
        };
        let volumes = vec![Volume { mount: vol.path().to_path_buf(), fstype: "ntfs3".into(), device: "/dev/x".into(), label: Some("DATA".into()), uuid: None }];
        let s = suggest(&analysis, &volumes, &Evidence::default());
        let rules: Vec<(String, String)> = s.config.rules.iter().map(|r| (r.from.clone(), r.to.clone())).collect();
        let to_vol = vol.path().to_string_lossy().into_owned();
        assert!(rules.contains(&("E:\\Backups".into(), format!("{to_vol}/Backups"))), "{rules:?}");
        assert!(rules.contains(&("G:\\".into(), to_vol.clone())), "{rules:?}");
        assert!(s.unresolved.iter().any(|u| u.starts_with("I:")));
        let text = render_toml(&s);
        let reparsed = Config::parse(&text).unwrap();
        assert_eq!(reparsed.rules.len(), s.config.rules.len());
    }

    #[test]
    fn path_evidence_alone_can_pick_a_drive() {
        let vol = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(vol.path().join("Users/Owner")).unwrap();
        let analysis = Analysis { buckets: vec![bucket("G:", "Users", &[("G:\\Users\\Owner\\gone.png", Some(1))])], ..Default::default() };
        let volumes = vec![Volume { mount: vol.path().to_path_buf(), fstype: "ntfs3".into(), device: "/dev/x".into(), label: None, uuid: None }];
        let ev = Evidence { hints: vec!["G:\\Users\\Owner\\Desktop\\proj".into()], ..Default::default() };
        let s = suggest(&analysis, &volumes, &ev);
        assert_eq!(s.config.rules.len(), 1);
        assert_eq!(s.config.rules[0].from, "G:\\");
        assert!(s.config.rules[0].note.contains("path evidence"));
    }
}
