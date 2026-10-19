//! Offline tests for the legacy import pipeline: tokenizer, path map, converter, merge semantics and a kv-mem import.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem};

use super::convert::{self, ConvertOptions, ConvertReport, ExistingIndex};
use super::job::Job;
use super::pathmap::{Config, Mapper, Options, Policy, Rule, Status, parse_win};
use super::sq::{ExportReader, Item, KeyKind, RecId, V, obj_get, obj_str, write_value};
use super::{import, verify};
use crate::database::{DB_NAME, NS, define_schema};

struct Chunked<'a> {
    data: &'a [u8],
    pos: usize,
    rng: u64,
    max: usize,
}

impl<'a> Chunked<'a> {
    fn new(data: &'a [u8], seed: u64, max: usize) -> Self {
        Self { data, pos: 0, rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, max }
    }
}

impl Read for Chunked<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let n = (1 + (self.rng as usize) % self.max).min(buf.len()).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

struct SplitAt<'a> {
    parts: Vec<&'a [u8]>,
}

impl Read for SplitAt<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while let Some(first) = self.parts.first_mut() {
            if first.is_empty() {
                self.parts.remove(0);
                continue;
            }
            let n = buf.len().min(first.len());
            buf[..n].copy_from_slice(&first[..n]);
            *first = &first[n..];
            return Ok(n);
        }
        Ok(0)
    }
}

fn items_of<R: Read>(mut rd: ExportReader<R>) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(item) = rd.next_item().expect("parse") {
        out.push(match item {
            Item::Statement(s) => format!("S:{s}"),
            Item::Row(v) => {
                let mut b = Vec::new();
                write_value(&mut b, &v);
                format!("R:{}", String::from_utf8_lossy(&b))
            }
            Item::EndBatch => "E".to_string(),
        });
    }
    out
}

const TRICKY: &str = "-- ------------------------------\n-- OPTION\n-- ------------------------------\n\nOPTION IMPORT;\n\n-- TABLE: thumbnails\nDEFINE TABLE thumbnails TYPE NORMAL SCHEMAFULL PERMISSIONS NONE;\nDEFINE FIELD embedding[*] ON clip_embeddings TYPE float PERMISSIONS FULL;\nDEFINE INDEX idx_clip_hnsw ON clip_embeddings FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32 EFC 120 M 12 M0 24 LM 0.402;\nDEFINE FIELD note ON thumbnails TYPE string DEFAULT 'a;b' PERMISSIONS FULL;\n\nINSERT [ { id: thumbnails:abc, path: 'E:\\\\Backups\\\\a\\n\\'b.jpg', size: 1f, when: d'2025-09-07T01:02:03.456789Z', u: u'8424486b-85b3-4448-ac8d-5d51083391c7', ratio: 0.5f, nothing: NONE, nil: NULL, ok: true, tags: [ 'x', \"y\" ], nested: { k: [ 1, 2.5f, -3 ] }, em: 'caf\\u00e9 \\ud83d\\ude00' }, { id: thumbnails:⟨8424486b-85b3-4448-ac8d-5d51083391c7⟩, path: 'p' }, { id: thumbnails:`odd key`, path: 'q' }, { id: thumbnails:12345, path: 'r', ref: logical_groups:xnoss } ];\n\nINSERT [ { id: user_settings:ShadowbrokerPC, k: 'v' } ];\n";

#[test]
fn tokenizer_parses_all_value_forms() {
    let items = items_of(ExportReader::new(TRICKY.as_bytes()));
    let rows: Vec<&String> = items.iter().filter(|i| i.starts_with("R:")).collect();
    assert_eq!(rows.len(), 5);
    assert!(items.iter().filter(|i| i.starts_with("S:")).count() >= 5, "{items:?}");
    assert!(rows[0].contains("d'2025-09-07T01:02:03.456789Z'"));
    assert!(rows[0].contains("u'8424486b-85b3-4448-ac8d-5d51083391c7'"));
    assert!(rows[0].contains("ratio: 0.5f"));
    assert!(rows[0].contains("nothing: NONE, nil: NULL, ok: true"));
    assert!(rows[0].contains("caf\u{e9} \u{1F600}"));
    assert!(rows[1].contains("thumbnails:`8424486b-85b3-4448-ac8d-5d51083391c7`"), "{}", rows[1]);
    assert!(rows[2].contains("thumbnails:`odd key`"));
    assert!(rows[3].contains("id: thumbnails:12345"));
    assert!(rows[3].contains("ref: logical_groups:xnoss"));
}

#[test]
fn string_escapes_round_trip_through_writer_and_reader() {
    let mut rd = ExportReader::new(TRICKY.as_bytes());
    let mut first = None;
    while let Some(item) = rd.next_item().unwrap() {
        if let Item::Row(V::Obj(f)) = item {
            first = Some(f);
            break;
        }
    }
    let f = first.unwrap();
    assert_eq!(obj_str(&f, "path"), Some("E:\\Backups\\a\n'b.jpg"));
    let mut buf = Vec::new();
    write_value(&mut buf, &V::Obj(f.clone()));
    let text = format!("INSERT [ {} ];", String::from_utf8(buf).unwrap());
    let mut again = ExportReader::new(text.as_bytes());
    let Item::Row(V::Obj(g)) = again.next_item().unwrap().unwrap() else { panic!("row expected") };
    assert_eq!(f, g);
}

#[test]
fn tokenizer_is_stable_under_random_chunking_and_tiny_buffers() {
    let reference = items_of(ExportReader::new(TRICKY.as_bytes()));
    for seed in 1..60u64 {
        let max = 1 + (seed as usize % 23);
        let cap = 1 + (seed as usize * 7 % 41);
        let got = items_of(ExportReader::with_capacity(Chunked::new(TRICKY.as_bytes(), seed, max), cap));
        assert_eq!(got, reference, "seed {seed} chunk<={max} cap {cap}");
    }
}

#[test]
fn rows_split_at_every_byte_boundary_parse_identically() {
    let reference = items_of(ExportReader::new(TRICKY.as_bytes()));
    let bytes = TRICKY.as_bytes();
    for k in 1..bytes.len() {
        let rd = SplitAt { parts: vec![&bytes[..k], &bytes[k..]] };
        assert_eq!(items_of(ExportReader::with_capacity(rd, 5)), reference, "split at {k}");
    }
}

#[test]
fn huge_single_line_rows_stream_through_small_buffers() {
    let big = "ab\\n".repeat(1_500_000);
    let text = format!("INSERT [ {{ id: thumbnails:big, thumbnail_b64: '{big}', n: 1 }}, {{ id: thumbnails:small, n: 2 }} ];");
    let mut rd = ExportReader::with_capacity(Chunked::new(text.as_bytes(), 9, 4093), 8192);
    let Item::Row(V::Obj(f)) = rd.next_item().unwrap().unwrap() else { panic!("row expected") };
    assert_eq!(obj_str(&f, "thumbnail_b64").unwrap().len(), 3 * 1_500_000);
    let Item::Row(V::Obj(g)) = rd.next_item().unwrap().unwrap() else { panic!("row expected") };
    assert!(matches!(obj_get(&g, "n"), Some(V::Num(n)) if n == "2"));
    assert!(matches!(rd.next_item().unwrap(), Some(Item::EndBatch)));
    assert!(rd.next_item().unwrap().is_none());
}

#[test]
fn insert_ignore_batches_and_unterminated_input() {
    let items = items_of(ExportReader::new("OPTION IMPORT;\nINSERT IGNORE [ { id: a:1 }, { id: a:2 } ];\nUPDATE user_settings:X MERGE { recent_paths: [ 'p;q' ] };\n".as_bytes()));
    assert_eq!(items.iter().filter(|i| i.starts_with("R:")).count(), 2);
    assert!(items.last().unwrap().starts_with("S:UPDATE user_settings:X MERGE"));
    let mut rd = ExportReader::new("INSERT [ { id: a:1 ".as_bytes());
    let mut failed = false;
    for _ in 0..4 {
        match rd.next_item() {
            Err(_) => {
                failed = true;
                break;
            }
            Ok(None) => break,
            Ok(Some(_)) => {}
        }
    }
    assert!(failed);
}

#[test]
fn record_id_writer_quotes_unsafe_keys() {
    let w = |r: &RecId| {
        let mut b = Vec::new();
        super::sq::write_recid(&mut b, r);
        String::from_utf8(b).unwrap()
    };
    assert_eq!(w(&RecId::named("t", "plain_1")), "t:plain_1");
    assert_eq!(w(&RecId::named("t", "has-dash")), "t:`has-dash`");
    assert_eq!(w(&RecId::named("t", "9lead")), "t:`9lead`");
    assert_eq!(w(&RecId { table: "t".into(), key_raw: "42".into(), kind: KeyKind::Int, key: "42".into() }), "t:42");
    assert_eq!(w(&RecId { table: "t".into(), key_raw: "007".into(), kind: KeyKind::Int, key: "007".into() }), "t:`007`");
    assert_eq!(w(&RecId { table: "t".into(), key_raw: "99999999999999999999".into(), kind: KeyKind::Int, key: "99999999999999999999".into() }), "t:`99999999999999999999`");
    assert_eq!(w(&RecId::named("t", "a`b")), "t:`a\\`b`");
}

fn cfg_of(rules: &[(&str, &str)]) -> Config {
    Config { options: Options::default(), rules: rules.iter().map(|(f, t)| Rule { from: f.to_string(), to: t.to_string(), note: String::new() }).collect() }
}

#[test]
fn windows_path_forms_are_parsed() {
    let p = parse_win("e:/Backups\\x/y.jpg").unwrap();
    assert_eq!((p.root.as_str(), p.comps.as_slice()), ("E:", ["Backups".to_string(), "x".to_string(), "y.jpg".to_string()].as_slice()));
    assert_eq!(parse_win("\\\\nas\\Photos\\a.jpg").unwrap().root, "\\\\NAS");
    assert_eq!(parse_win("\\\\?\\C:\\a\\b").unwrap().root, "C:");
    assert_eq!(parse_win("\\\\?\\UNC\\host\\share\\f").unwrap().comps, vec!["share".to_string(), "f".to_string()]);
    assert!(parse_win("/home/u/a.jpg").is_none());
    assert!(parse_win("relative\\x").is_none());
    assert!(parse_win("C:file").is_none());
    assert_eq!(parse_win("D:").unwrap().comps.len(), 0);
}

#[test]
fn longest_prefix_wins_and_ties_go_to_the_first_rule() {
    let d = tempfile::tempdir().unwrap();
    for sub in ["all/Backups/x", "bk/x", "first/y", "second/y"] {
        std::fs::create_dir_all(d.path().join(sub)).unwrap();
    }
    let all = d.path().join("all").to_string_lossy().into_owned();
    let bk = d.path().join("bk").to_string_lossy().into_owned();
    let first = d.path().join("first").to_string_lossy().into_owned();
    let second = d.path().join("second").to_string_lossy().into_owned();
    let mut m = Mapper::new(cfg_of(&[("E:\\", &all), ("E:\\Backups", &bk), ("F:\\z", &first), ("F:\\Z", &second)])).unwrap();
    let a = m.map("E:\\Backups\\x");
    assert_eq!((a.status, a.rule), (Status::Exists, Some(1)));
    assert_eq!(a.path, format!("{bk}/x"));
    let b = m.map("E:\\Other");
    assert_eq!((b.rule, b.status), (Some(0), Status::Missing));
    let c = m.map("F:\\z\\y");
    assert_eq!(c.rule, Some(2));
    assert_eq!(c.path, format!("{first}/y"));
}

#[test]
fn case_differences_are_repaired_per_directory_and_ambiguity_is_missing() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("Photos/Sub Dir")).unwrap();
    std::fs::write(d.path().join("Photos/Sub Dir/A.JPG"), b"x").unwrap();
    std::fs::create_dir_all(d.path().join("Dup")).unwrap();
    std::fs::create_dir_all(d.path().join("dup")).unwrap();
    let root = d.path().to_string_lossy().into_owned();
    let mut m = Mapper::new(cfg_of(&[("D:\\", &root)])).unwrap();
    let ok = m.map("d:/photos\\sub dir\\a.jpg");
    assert_eq!(ok.status, Status::ExistsCaseFixed);
    assert_eq!(ok.path, format!("{root}/Photos/Sub Dir/A.JPG"));
    let exact = m.map("D:\\Photos\\Sub Dir\\A.JPG");
    assert_eq!(exact.status, Status::Exists);
    let missing = m.map("D:\\Photos\\nope\\a.jpg");
    assert_eq!(missing.status, Status::Missing);
    assert_eq!(missing.first_missing.as_deref(), Some("D:\\Photos\\nope"));
    assert_eq!(missing.path, format!("{root}/Photos/nope/a.jpg"));
    let ambiguous = m.map("D:\\DUP\\f");
    assert_eq!(ambiguous.status, Status::Missing);
    assert_eq!(m.map("/already/posix").status, Status::Native);
    assert_eq!(m.map("https://x/y").status, Status::Native);
    assert_eq!(m.map("Z:\\x").status, Status::Unmapped);
}

#[test]
fn empty_target_is_an_explicit_unmapped_rule_and_unc_roots_map() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("photos/2020")).unwrap();
    std::fs::write(d.path().join("photos/2020/a.jpg"), b"1").unwrap();
    let root = d.path().to_string_lossy().into_owned();
    let mut m = Mapper::new(cfg_of(&[("\\\\NAS\\Photos", &format!("{root}/photos")), ("I:\\", "")])).unwrap();
    let u = m.map("\\\\nas\\photos\\2020\\a.jpg");
    assert_eq!(u.path, format!("{root}/photos/2020/a.jpg"));
    assert!(matches!(u.status, Status::Exists | Status::ExistsCaseFixed));
    let i = m.map("I:\\DCIM\\x.jpg");
    assert_eq!((i.status, i.rule, i.path.as_str()), (Status::Unmapped, Some(1), "I:\\DCIM\\x.jpg"));
}

#[test]
fn config_validation_and_toml_round_trip() {
    let text = "[options]\nunmapped = \"tag\"\nmissing = \"drop\"\n\n[[rule]]\nfrom = \"E:\\\\Backups\"\nto = \"/mnt/b\"\nnote = \"n\"\n\n[[rule]]\nfrom = \"I:\\\\\"\nto = \"\"\n";
    let c = Config::parse(text).unwrap();
    assert_eq!((c.options.unmapped, c.options.missing, c.options.case_fix), (Policy::Tag, Policy::Drop, true));
    assert_eq!(c.rules.len(), 2);
    assert_eq!(Config::parse(&c.to_toml().unwrap()).unwrap(), c);
    assert!(Config::parse("[[rule]]\nfrom = \"/posix\"\nto = \"/x\"\n").is_err());
    assert!(Config::parse("[[rule]]\nfrom = \"E:\\\\\"\nto = \"relative/x\"\n").is_err());
    assert!(Config::parse("[options]\nmissing = \"bogus\"\n").is_err());
    assert_eq!("DROP".parse::<Policy>().unwrap(), Policy::Drop);
    assert!("x".parse::<Policy>().is_err());
}

struct Fixture {
    dir: tempfile::TempDir,
    vol: PathBuf,
    nas: PathBuf,
    secret: &'static str,
    b64: String,
}

fn png_uri() -> String {
    use base64::Engine;
    let img = image::RgbImage::from_fn(16, 16, |x, y| image::Rgb([(x * 8) as u8, (y * 8) as u8, 100]));
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf))
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let vol = dir.path().join("vol");
        let nas = dir.path().join("nas");
        for (p, len) in [("Backups/Photos/a.jpg", 11usize), ("Backups/Photos/b.PNG", 22), ("Backups/other/c.mp4", 33), ("Users/Owner/Pictures/g.jpg", 44), ("Users/Owner/Pictures/h.jpg", 55)] {
            let f = vol.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(&f, vec![1u8; len]).unwrap();
        }
        std::fs::create_dir_all(nas.join("photos")).unwrap();
        std::fs::write(nas.join("photos/n.jpg"), b"nas").unwrap();
        Self { dir, vol, nas, secret: "sk-SECRET-TOKEN-123", b64: format!("data:image/png;base64,{}", "QUJD".repeat(300)) }
    }

    fn with_png() -> Self {
        Self { b64: png_uri(), ..Self::new() }
    }

    fn cfg(&self) -> Config {
        let v = self.vol.to_string_lossy().into_owned();
        let n = self.nas.to_string_lossy().into_owned();
        cfg_of(&[("E:\\Backups", &format!("{v}/Backups")), ("G:\\", &v), ("\\\\NAS\\Photos", &format!("{n}/photos")), ("I:\\", "")])
    }

    fn vec_text(seed: usize, dim: usize) -> String {
        (0..dim).map(|i| format!("{:.3}f", ((i * 31 + seed * 17) % 1000) as f32 / 1000.0 - 0.5)).collect::<Vec<_>>().join(", ")
    }

    fn export(&self) -> String {
        let b64 = self.b64.clone();
        let thumb = |id: &str, path: &str, extra: &str| format!("{{ id: thumbnails:{id}, path: '{path}', filename: 'f', parent_dir: 'X', size: 1f, file_type: 'jpg', modified: d'2025-09-07T01:02:03Z', db_created: d'2025-09-07T01:02:03Z', thumbnail_b64: '{b64}'{extra} }}");
        let mut rows = vec![
            thumb("aaaaaaaaaaaaaaaaaaaa", "E:\\\\Backups\\\\Photos\\\\a.jpg", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c, tags: [ 'old' ]"),
            thumb("bbbbbbbbbbbbbbbbbbbb", "e:/backups\\\\photos\\\\B.png", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c, legacy_x: 'drop me'"),
            thumb("cccccccccccccccccccc", "E:\\\\Backups\\\\other\\\\c.mp4", ", logical_group: logical_groups:Default"),
            thumb("dddddddddddddddddddd", "G:\\\\Users\\\\Owner\\\\Pictures\\\\g.jpg", ", tags: [], logical_group: logical_groups:qaozwzjnybe3lkgko25m"),
            thumb("eeeeeeeeeeeeeeeeeeee", "G:\\\\Users\\\\Owner\\\\Pictures\\\\gone.jpg", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c"),
            thumb("ffffffffffffffffffff", "I:\\\\DCIM\\\\100MSDCF\\\\x.jpg", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c"),
            thumb("gggggggggggggggggggg", "\\\\\\\\NAS\\\\Photos\\\\n.jpg", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c"),
            thumb("hhhhhhhhhhhhhhhhhhhh", "E:\\\\BACKUPS\\\\PHOTOS\\\\A.JPG", ", logical_group: logical_groups:xnossbcr1k5twgu7dr3c"),
            "{ id: thumbnails:⟨8424486b-85b3-4448-ac8d-5d51083391c7⟩, path: 'G:\\\\Users\\\\Owner\\\\Pictures\\\\h.jpg', filename: 'h.jpg', parent_dir: 'G:\\\\Users', caption: 'a caption', description: 'desc', category: 'cat', tags: [ 't1' ], logical_group: logical_groups:Default, modified: d'2025-11-20T01:02:03Z' }".to_string(),
        ];
        let thumbs = rows.drain(..).collect::<Vec<_>>().join(",\n");
        let clip = |id: &str, thumb_ref: &str, path: &str, dim: usize| format!("{{ id: clip_embeddings:{id}, path: '{path}', {thumb_ref} embedding: [ {} ], created: d'2025-09-07T01:02:03Z', updated: d'2025-09-07T01:02:03Z', hash: 'h' }}", Self::vec_text(id.len(), dim));
        let clips = [
            clip("c1c1c1c1c1c1c1c1c1c1", "thumb_ref: thumbnails:aaaaaaaaaaaaaaaaaaaa,", "E:\\\\Backups\\\\Photos\\\\a.jpg", 1024),
            clip("c2c2c2c2c2c2c2c2c2c2", "thumb_ref: thumbnails:bbbbbbbbbbbbbbbbbbbb,", "e:/backups\\\\photos\\\\B.png", 1024),
            clip("c3c3c3c3c3c3c3c3c3c3", "thumb_ref: thumbnails:hhhhhhhhhhhhhhhhhhhh,", "E:\\\\BACKUPS\\\\PHOTOS\\\\A.JPG", 1024),
            clip("c4c4c4c4c4c4c4c4c4c4", "", "G:\\\\Users\\\\Owner\\\\Pictures\\\\g.jpg", 1024),
            clip("c5c5c5c5c5c5c5c5c5c5", "thumb_ref: thumbnails:dddddddddddddddddddd,", "G:\\\\Users\\\\Owner\\\\Pictures\\\\bad.jpg", 512),
        ]
        .join(",\n");
        format!(
            "-- ------------------------------\n-- OPTION\n-- ------------------------------\n\nOPTION IMPORT;\n\n-- TABLE: categories\nDEFINE TABLE categories TYPE NORMAL SCHEMAFULL PERMISSIONS NONE;\n\n-- TABLE DATA: logical_groups\nINSERT [ {{ id: logical_groups:xnossbcr1k5twgu7dr3c, name: 'Default', created: d'2025-09-07T00:00:00Z', updated: d'2025-09-07T00:00:00Z', thumbnails: [] }}, {{ id: logical_groups:qaozwzjnybe3lkgko25m, name: 'FileHistory-OldGamesHdd', thumbnails: [] }} ];\n\nINSERT [ {{ id: categories:one, name: 'x' }} ];\n\nINSERT [ {clips} ];\n\nINSERT [ {thumbs} ];\n\nINSERT [ {{ id: assistant_sessions:sess1, title: 'chat', created: d'2025-09-07T00:00:00Z', updated: d'2025-09-07T00:00:00Z' }} ];\n\nINSERT [ {{ id: assistant_messages:msg1, session_ref: assistant_sessions:sess1, role: 'user', content: 'hello', attachments: [ 'E:\\\\Backups\\\\Photos\\\\a.jpg' ], attachments_refs: [ thumbnails:aaaaaaaaaaaaaaaaaaaa, thumbnails:nonexistent ], created: d'2025-09-07T00:00:00Z' }} ];\n\nINSERT [ {{ id: user_settings:ShadowbrokerPC, openai_api_key: '{}', recent_paths: [ 'E:\\\\Backups\\\\Photos' ], excluded_dirs: [ 'G:\\\\Users\\\\Owner\\\\Pictures' ], group_by_category: true, last_used_model: 'm' }} ];\n",
            self.secret
        )
    }

    fn write_export(&self) -> PathBuf {
        let p = self.dir.path().join("export.surql");
        std::fs::write(&p, self.export()).unwrap();
        p
    }
}

fn run_convert(f: &Fixture, mutate: impl FnOnce(&mut ConvertOptions, &mut Config)) -> (ConvertReport, PathBuf) {
    let out = f.dir.path().join(format!("out-{}", rand_suffix()));
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = Some(out.clone());
    let mut cfg = f.cfg();
    mutate(&mut opts, &mut cfg);
    let rep = convert::convert(&opts, &cfg, None, &Job::new()).unwrap();
    (rep, out)
}

fn rand_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::SeqCst).to_string()
}

fn rows_of(dir: &Path, table: &str) -> Vec<Vec<(String, V)>> {
    let mut out = Vec::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(&format!("{table}."))).collect();
    files.sort();
    for p in files {
        let mut rd = ExportReader::new(std::fs::File::open(p).unwrap());
        while let Some(item) = rd.next_item().unwrap() {
            if let Item::Row(V::Obj(f)) = item {
                out.push(f);
            }
        }
    }
    out
}

fn rec_text(v: Option<&V>) -> String {
    match v {
        Some(V::Rec(r)) => format!("{}:{}", r.table, r.key),
        other => format!("{other:?}"),
    }
}

fn dir_text(dir: &Path) -> String {
    let mut s = String::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.path().is_file() {
            s.push_str(&String::from_utf8_lossy(&std::fs::read(e.path()).unwrap()));
        }
    }
    s
}

#[test]
fn converter_fixes_the_tricky_export() {
    let f = Fixture::new();
    let (rep, out) = run_convert(&f, |_, _| {});
    let v = f.vol.to_string_lossy().into_owned();
    let groups = rows_of(&out, "logical_groups");
    let names: Vec<&str> = groups.iter().filter_map(|g| obj_str(g, "name")).collect();
    assert_eq!(names, vec!["Default", "FileHistory-OldGamesHdd"]);
    assert_eq!(rec_text(obj_get(&groups[0], "id")), "logical_groups:Default");
    assert!(groups.iter().all(|g| obj_get(g, "thumbnails").is_none()));

    let thumbs = rows_of(&out, "thumbnails");
    assert_eq!(thumbs.len(), 8, "one duplicate is dropped: {:?}", rep.tables["thumbnails"].skipped);
    let by_path: BTreeMap<String, &Vec<(String, V)>> = thumbs.iter().map(|t| (obj_str(t, "path").unwrap().to_string(), t)).collect();
    let a = by_path[&format!("{v}/Backups/Photos/a.jpg")];
    assert_eq!(rec_text(obj_get(a, "id")), "thumbnails:aaaaaaaaaaaaaaaaaaaa");
    assert_eq!(obj_str(a, "parent_dir"), Some(format!("{v}/Backups/Photos").as_str()));
    assert_eq!(rec_text(obj_get(a, "logical_group")), "logical_groups:Default");
    assert!(matches!(obj_get(a, "tags"), Some(V::Arr(t)) if t.len() == 1));
    let b = by_path[&format!("{v}/Backups/Photos/b.PNG")];
    assert_eq!(obj_str(b, "filename"), Some("b.PNG"));
    assert!(obj_get(b, "legacy_x").is_none());
    assert!(matches!(obj_get(b, "tags"), Some(V::Arr(t)) if t.is_empty()));
    let c = by_path[&format!("{v}/Backups/other/c.mp4")];
    assert_eq!(rec_text(obj_get(c, "logical_group")), "logical_groups:Default");
    let d = by_path[&format!("{v}/Users/Owner/Pictures/g.jpg")];
    assert_eq!(rec_text(obj_get(d, "logical_group")), "logical_groups:FileHistory-OldGamesHdd");
    let missing = by_path[&format!("{v}/Users/Owner/Pictures/gone.jpg")];
    assert_eq!(rec_text(obj_get(missing, "id")), "thumbnails:eeeeeeeeeeeeeeeeeeee");
    let sd = by_path["I:\\DCIM\\100MSDCF\\x.jpg"];
    assert_eq!(obj_str(sd, "parent_dir"), Some("X"));
    let nas = f.nas.to_string_lossy().into_owned();
    assert!(by_path.contains_key(&format!("{nas}/photos/n.jpg")));
    let uuid = thumbs.iter().find(|t| matches!(obj_get(t, "id"), Some(V::Rec(r)) if r.key == "8424486b-85b3-4448-ac8d-5d51083391c7")).expect("uuid row");
    assert!(obj_get(uuid, "thumbnail_b64").is_none());
    assert!(matches!(obj_get(uuid, "size"), Some(V::Num(n)) if n == "55"), "size is read from the file: {:?}", obj_get(uuid, "size"));
    assert_eq!(obj_str(uuid, "file_type"), Some("jpg"));
    assert_eq!(obj_str(uuid, "caption"), Some("a caption"));
    assert!(dir_text(&out).contains("thumbnails:`8424486b-85b3-4448-ac8d-5d51083391c7`"));

    let t = &rep.tables["thumbnails"];
    assert_eq!((t.rows_in, t.rows_out), (9, 8));
    assert_eq!(t.skipped.get("duplicate_mapped_path"), Some(&1));
    assert_eq!(t.dropped_fields.get("legacy_x"), Some(&1));
    assert_eq!(t.fixed.get("size_normalized_to_integer"), Some(&7));
    assert!(t.fixed.contains_key("logical_group_unknown_to_default") || t.fixed.contains_key("logical_group_remapped"));
    assert_eq!(rep.other_tables_dropped.get("categories"), Some(&1));
    assert_eq!(rep.rows_out("logical_groups"), 2);
}

#[test]
fn clip_rows_follow_their_thumbnails_and_bad_rows_are_skipped() {
    let f = Fixture::new();
    let (rep, out) = run_convert(&f, |_, _| {});
    let clips = rows_of(&out, "clip_embeddings");
    assert_eq!(clips.len(), 3, "{:?}", rep.tables["clip_embeddings"]);
    let c = &rep.tables["clip_embeddings"];
    assert_eq!(c.skipped.get("wrong_embedding_dimension"), Some(&1));
    assert_eq!(c.skipped.get("duplicate_mapped_path"), Some(&1));
    let v = f.vol.to_string_lossy().into_owned();
    let by_path: BTreeMap<String, &Vec<(String, V)>> = clips.iter().map(|t| (obj_str(t, "path").unwrap().to_string(), t)).collect();
    assert_eq!(rec_text(obj_get(by_path[&format!("{v}/Backups/Photos/a.jpg")], "thumb_ref")), "thumbnails:aaaaaaaaaaaaaaaaaaaa");
    assert_eq!(rec_text(obj_get(by_path[&format!("{v}/Backups/Photos/b.PNG")], "thumb_ref")), "thumbnails:bbbbbbbbbbbbbbbbbbbb");
    let unlinked = by_path[&format!("{v}/Users/Owner/Pictures/g.jpg")];
    assert_eq!(rec_text(obj_get(unlinked, "thumb_ref")), "thumbnails:dddddddddddddddddddd");
    assert_eq!(c.fixed.get("thumb_ref_linked_by_path"), Some(&1));
    assert!(matches!(obj_get(unlinked, "embedding"), Some(V::Arr(e)) if e.len() == 1024));
}

#[test]
fn chat_rows_are_remapped_and_dangling_refs_pruned() {
    let f = Fixture::new();
    let (rep, out) = run_convert(&f, |_, _| {});
    let msgs = rows_of(&out, "assistant_messages");
    assert_eq!(msgs.len(), 1);
    let v = f.vol.to_string_lossy().into_owned();
    assert!(matches!(obj_get(&msgs[0], "attachments"), Some(V::Arr(a)) if matches!(&a[0], V::Str(s) if *s == format!("{v}/Backups/Photos/a.jpg"))));
    assert!(matches!(obj_get(&msgs[0], "attachments_refs"), Some(V::Arr(a)) if a.len() == 1));
    assert_eq!(rep.tables["assistant_messages"].fixed.get("attachment_refs_pruned"), Some(&1));
    assert_eq!(rows_of(&out, "assistant_sessions").len(), 1);
}

#[test]
fn secrets_are_never_emitted_and_settings_are_skipped_by_default() {
    let f = Fixture::new();
    let (rep, out) = run_convert(&f, |_, _| {});
    assert!(!dir_text(&out).contains("SECRET"));
    assert!(rows_of(&out, "user_settings").is_empty());
    assert!(rep.settings.present);
    assert!(rep.settings.secret_field_names.contains(&"openai_api_key".to_string()));
    assert!(!serde_json::to_string(&rep).unwrap().contains("SECRET"));
    assert!(rep.settings.merged_fields.is_empty());

    let (rep2, out2) = run_convert(&f, |o, _| {
        o.merge = true;
        o.settings_fields = vec!["recent_paths".into(), "excluded_dirs".into()];
    });
    assert_eq!(rep2.tables["user_settings"].skipped.get("settings_target_missing"), Some(&1));
    assert!(!dir_text(&out2).contains("SECRET"));

    let ex = ExistingIndex { settings_present: true, ..Default::default() };
    let out3 = f.dir.path().join("out-settings");
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = Some(out3.clone());
    opts.merge = true;
    opts.settings_fields = vec!["recent_paths".into(), "excluded_dirs".into(), "nope".into()];
    let rep3 = convert::convert(&opts, &f.cfg(), Some(&ex), &Job::new()).unwrap();
    let text = dir_text(&out3);
    assert!(text.contains("UPDATE user_settings:ShadowbrokerPC MERGE"));
    assert!(text.contains(&format!("{}/Backups/Photos", f.vol.display())));
    assert!(!text.contains("SECRET") && !text.contains("openai_api_key"));
    assert_eq!(rep3.settings.merged_fields, vec!["recent_paths".to_string(), "excluded_dirs".to_string()]);
    assert_eq!(rep3.settings.unknown_requested_fields, vec!["nope".to_string()]);

    opts.out_dir = Some(f.dir.path().join("out-named"));
    opts.settings_fields = vec!["openai_api_key".into()];
    let rep4 = convert::convert(&opts, &f.cfg(), Some(&ex), &Job::new()).unwrap();
    assert!(dir_text(&f.dir.path().join("out-named")).contains("SECRET"));
    assert_eq!(rep4.settings.merged_secret_fields, vec!["openai_api_key".to_string()]);
    assert!(rep4.warnings.iter().any(|w| w.contains("openai_api_key")));
    assert!(!serde_json::to_string(&rep4).unwrap().contains("SECRET"));
}

#[test]
fn missing_and_unmapped_policies_keep_drop_or_tag() {
    let f = Fixture::new();
    let (_, keep) = run_convert(&f, |_, _| {});
    assert_eq!(rows_of(&keep, "thumbnails").len(), 8);
    let (rep_drop, drop) = run_convert(&f, |_, c| {
        c.options.missing = Policy::Drop;
        c.options.unmapped = Policy::Drop;
    });
    let kept: Vec<String> = rows_of(&drop, "thumbnails").iter().map(|t| obj_str(t, "path").unwrap().to_string()).collect();
    assert_eq!(kept.len(), 6, "{kept:?}");
    assert!(kept.iter().all(|p| p.starts_with('/')));
    assert_eq!(rep_drop.tables["thumbnails"].skipped.get("path_policy_or_no_path"), Some(&2));
    assert!(rows_of(&drop, "clip_embeddings").len() <= 3);
    let (_, tag) = run_convert(&f, |_, c| {
        c.options.missing = Policy::Tag;
        c.options.unmapped = Policy::Tag;
    });
    let tagged: Vec<(String, usize)> = rows_of(&tag, "thumbnails")
        .iter()
        .filter(|t| matches!(obj_get(t, "tags"), Some(V::Arr(a)) if a.iter().any(|x| matches!(x, V::Str(s) if s.starts_with("legacy-")))))
        .map(|t| (obj_str(t, "path").unwrap().to_string(), 1))
        .collect();
    assert_eq!(tagged.len(), 2);
}

#[test]
fn dry_run_writes_nothing_and_reports_the_mapping_table() {
    let f = Fixture::new();
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = None;
    let rep = convert::convert(&opts, &f.cfg(), None, &Job::new()).unwrap();
    assert!(rep.dry_run && rep.files.is_empty());
    let rows: BTreeMap<String, u64> = rep.path_rules.iter().map(|r| (r.from.clone(), r.rows)).collect();
    assert_eq!(rows["E:\\Backups"], 4);
    assert_eq!(rows["G:\\"], 3);
    assert_eq!(rows["I:\\"], 1);
    let e = rep.path_rules.iter().find(|r| r.from == "E:\\Backups").unwrap();
    assert!(e.exists_rate > 0.99);
    assert!(e.case_fixed_rows >= 2);
    let g = rep.path_rules.iter().find(|r| r.from == "G:\\").unwrap();
    assert_eq!(g.by_status.get("missing"), Some(&1));
    assert!(!g.missing_top_prefixes.is_empty());
    assert!(rep.exists_rate() > 0.5);
    assert_eq!(rep.unmapped_prefixes.len(), 1);
}

#[test]
fn conversion_is_deterministic_and_parts_split_by_size() {
    let f = Fixture::new();
    let (_, a) = run_convert(&f, |o, _| {
        o.split_bytes = 2500;
        o.batch_rows = 2;
    });
    let (_, b) = run_convert(&f, |o, _| {
        o.split_bytes = 2500;
        o.batch_rows = 2;
    });
    let names = |d: &Path| {
        let mut n: Vec<String> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        n.sort();
        n
    };
    assert_eq!(names(&a), names(&b));
    assert!(names(&a).iter().filter(|n| n.starts_with("thumbnails.")).count() >= 3, "{:?}", names(&a));
    for n in names(&a) {
        assert_eq!(std::fs::read(a.join(&n)).unwrap(), std::fs::read(b.join(&n)).unwrap(), "{n}");
    }
    assert_eq!(rows_of(&a, "thumbnails").len(), 8);
    let order: Vec<String> = serde_json::from_slice(&std::fs::read(a.join("import-order.json")).unwrap()).unwrap();
    let first_thumb = order.iter().position(|f| f.starts_with("thumbnails.")).unwrap();
    let first_clip = order.iter().position(|f| f.starts_with("clip_embeddings.")).unwrap();
    assert!(order[0].starts_with("logical_groups.") && first_thumb < first_clip);
}

#[test]
fn drop_b64_omits_the_inline_thumbnail() {
    let f = Fixture::new();
    let (rep, out) = run_convert(&f, |o, _| o.drop_b64 = true);
    assert!(rows_of(&out, "thumbnails").iter().all(|t| obj_get(t, "thumbnail_b64").is_none()));
    assert_eq!(rep.tables["thumbnails"].fixed.get("thumbnail_b64_omitted"), Some(&7));
}

#[test]
fn merge_skips_rows_the_target_has_and_relinks_embeddings() {
    let f = Fixture::new();
    let v = f.vol.to_string_lossy().into_owned();
    let mut ex = ExistingIndex::default();
    ex.thumb_by_path.insert(format!("{v}/Backups/Photos/a.jpg"), "existingaaaaaaaaaaaaa".into());
    ex.thumb_ids.insert("existingaaaaaaaaaaaaa".into());
    ex.thumb_ids.insert("bbbbbbbbbbbbbbbbbbbb".into());
    ex.clip_paths.insert(format!("{v}/Backups/Photos/b.PNG"));
    ex.group_by_name.insert("Default".into(), "Default".into());
    ex.session_ids.insert("sess1".into());
    let out = f.dir.path().join("merge-out");
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = Some(out.clone());
    opts.merge = true;
    let rep = convert::convert(&opts, &f.cfg(), Some(&ex), &Job::new()).unwrap();
    assert!(dir_text(&out).contains("INSERT IGNORE ["));
    let thumbs = rows_of(&out, "thumbnails");
    let paths: Vec<&str> = thumbs.iter().map(|t| obj_str(t, "path").unwrap()).collect();
    assert!(!paths.contains(&format!("{v}/Backups/Photos/a.jpg").as_str()));
    assert_eq!(rep.tables["thumbnails"].skipped.get("already_in_target"), Some(&1));
    let b = thumbs.iter().find(|t| obj_str(t, "path") == Some(format!("{v}/Backups/Photos/b.PNG").as_str())).unwrap();
    let new_id = rec_text(obj_get(b, "id"));
    assert_ne!(new_id, "thumbnails:bbbbbbbbbbbbbbbbbbbb", "a colliding id gets a fresh derived key");
    assert_eq!(new_id, format!("thumbnails:{}", convert::derived_key(&format!("{v}/Backups/Photos/b.PNG"))));
    let clips = rows_of(&out, "clip_embeddings");
    let a_clip = clips.iter().find(|c| obj_str(c, "path") == Some(format!("{v}/Backups/Photos/a.jpg").as_str()));
    assert!(a_clip.is_some());
    assert_eq!(rec_text(obj_get(a_clip.unwrap(), "thumb_ref")), "thumbnails:existingaaaaaaaaaaaaa");
    assert!(clips.iter().all(|c| obj_str(c, "path") != Some(format!("{v}/Backups/Photos/b.PNG").as_str())));
    let b_clip_skipped = rep.tables["clip_embeddings"].skipped.get("already_in_target");
    assert_eq!(b_clip_skipped, Some(&1));
    assert_eq!(rows_of(&out, "logical_groups").len(), 1, "Default already exists in the target");
    assert_eq!(rep.tables["assistant_sessions"].skipped.get("already_in_target"), Some(&1));
    let re_linked_msg = rows_of(&out, "assistant_messages");
    assert!(matches!(obj_get(&re_linked_msg[0], "attachments_refs"), Some(V::Arr(a)) if a.len() == 1 && matches!(&a[0], V::Rec(r) if r.key == "existingaaaaaaaaaaaaa")));
}

#[test]
fn analysis_reports_buckets_counts_and_settings_fields() {
    let f = Fixture::new();
    let a = convert::analyze(&f.write_export(), &Job::new()).unwrap();
    assert_eq!(a.tables["thumbnails"], 9);
    assert_eq!(a.tables["clip_embeddings"], 5);
    let e = a.buckets.iter().find(|b| b.root == "E:").unwrap();
    assert_eq!((e.first.to_lowercase().as_str(), e.rows), ("backups", 4));
    assert!(a.buckets.iter().any(|b| b.root == "\\\\NAS"));
    assert!(a.settings_fields.contains(&"openai_api_key".to_string()));
    assert!(a.settings_paths.iter().any(|p| p.starts_with("E:")));
    assert!(!serde_json::to_string(&a).unwrap().contains("SECRET"));
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(NS).use_db(DB_NAME).await.unwrap();
    define_schema(&db).await.unwrap();
    db
}

fn settings_text() -> &'static str {
    "CREATE user_settings:ShadowbrokerPC CONTENT { group_by_category: false, auto_indexing: true, ai_prompt_template: 'p', overwrite_descriptions: false, filter_skip_icons: false, auto_clip_embeddings: false, clip_augment_with_text: false, clip_overwrite_embeddings: false, recent_paths: ['/linux/recent'], recent_models: [], auto_save_to_database: false, openai_api_key: 'linux-key' };"
}

#[tokio::test]
async fn converted_set_imports_into_a_kv_mem_store_and_deserializes_into_app_types() {
    let f = Fixture::new();
    let db = mem_db().await;
    let job = Job::new();
    let out = f.dir.path().join("mem-out");
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = Some(out.clone());
    let conv = convert::convert(&opts, &f.cfg(), None, &job).unwrap();
    let before = import::table_counts(&db).await.unwrap();
    let stats = import::import_files(&db, &out, &conv.files, &job).await.unwrap();
    assert_eq!(stats.files, conv.files.len());
    let mut idem = opts.clone();
    idem.out_dir = None;
    idem.merge = true;
    let spec = verify::VerifySpec { before, settings_digest_before: None, knn_samples: 2, idempotent: Some((idem, f.cfg())), settings_key: convert::SETTINGS_KEY.into(), live: false };
    let rep = verify::verify_import(&db, &spec, &conv, &job).await.unwrap();
    assert!(rep.ok, "{:?}", rep.problems);
    assert_eq!(rep.counts["thumbnails"].after, 8);
    assert_eq!(rep.counts["clip_embeddings"].after, 3);
    assert_eq!(rep.typecheck.thumbnails, 8);
    assert_eq!(rep.typecheck.clip_embeddings, 3);
    assert_eq!(rep.typecheck.logical_groups, 2);
    assert!(rep.indexes_ok);
    assert!(rep.knn.iter().all(|k| k.top1_is_self));
    assert!(rep.idempotent.as_ref().is_some_and(|i| i.ok));
    let g: Vec<crate::LogicalGroup> = db.query("SELECT * FROM logical_groups ORDER BY name").await.unwrap().take(0).unwrap();
    assert_eq!(g.len(), 2);
    let t: Vec<crate::Thumbnail> = db.query("SELECT id, db_created, path, filename, file_type, size, description, caption, tags, category, thumbnail_b64, modified, hash, parent_dir, logical_group FROM thumbnails WHERE caption != NONE").await.unwrap().take(0).unwrap();
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].caption.as_deref(), Some("a caption"));
}

#[tokio::test]
async fn merge_into_a_populated_store_is_additive_and_idempotent() {
    let f = Fixture::new();
    let db = mem_db().await;
    let v = f.vol.to_string_lossy().into_owned();
    db.query(format!(
        "CREATE logical_groups:Default CONTENT {{ name: 'Default' }}; CREATE thumbnails:linuxone CONTENT {{ path: '{v}/Backups/Photos/a.jpg', filename: 'a.jpg', size: 1, parent_dir: '{v}/Backups/Photos', logical_group: logical_groups:Default, tags: [], file_type: 'jpg' }}; CREATE thumbnails:linuxtwo CONTENT {{ path: '/linux/only.jpg', filename: 'only.jpg', size: 2, parent_dir: '/linux', logical_group: logical_groups:Default, tags: [], file_type: 'jpg' }}; {}",
        settings_text()
    ))
    .await
    .unwrap()
    .check()
    .unwrap();
    let job = Job::new();
    let digest = verify::settings_digest(&db, convert::SETTINGS_KEY).await;
    assert!(digest.is_some() && digest.as_deref() != Some("none"));
    let work = f.dir.path().to_path_buf();
    let run = |dir: String, fields: Vec<String>| {
        let (db, f_cfg, export, job, work) = (db.clone(), f.cfg(), f.write_export(), Arc::clone(&job), work.clone());
        async move {
            let existing = import::load_existing(&db, convert::SETTINGS_KEY, &job).await.unwrap();
            let out = work.join(dir);
            let _ = std::fs::remove_dir_all(&out);
            let mut opts = ConvertOptions::new(export);
            opts.out_dir = Some(out.clone());
            opts.merge = true;
            opts.settings_fields = fields;
            let conv = convert::convert(&opts, &f_cfg, Some(&existing), &job).unwrap();
            import::import_files(&db, &out, &conv.files, &job).await.unwrap();
            let _ = std::fs::remove_dir_all(&out);
            conv
        }
    };
    let first = run("merge-a".to_string(), vec!["recent_paths".into()]).await;
    assert_eq!(first.rows_out("thumbnails"), 7);
    assert_eq!(first.tables["thumbnails"].skipped.get("already_in_target"), Some(&1));
    let counts = import::table_counts(&db).await.unwrap();
    assert_eq!(counts["thumbnails"], 2 + 7);
    assert_eq!(counts["logical_groups"], 2);
    assert_eq!(counts["assistant_messages"], 1);
    let linux: Vec<crate::Thumbnail> = db.query("SELECT * FROM thumbnails WHERE id = thumbnails:linuxone").await.unwrap().take(0).unwrap();
    assert_eq!(linux.len(), 1);
    let clip: Vec<crate::ClipEmbeddingRow> = db.query("SELECT * FROM clip_embeddings WHERE path = $p").bind(("p", format!("{v}/Backups/Photos/a.jpg"))).await.unwrap().take(0).unwrap();
    assert_eq!(clip[0].thumb_ref.as_ref().map(|r| r.key.clone()), Some(surrealdb::types::RecordIdKey::String("linuxone".into())));
    let settings: Vec<serde_json::Value> = db.query("SELECT recent_paths, openai_api_key FROM user_settings").await.unwrap().take::<surrealdb::types::Value>(0).map(|v| match v.into_json_value() {
        serde_json::Value::Array(a) => a,
        _ => vec![],
    }).unwrap();
    assert_eq!(settings[0]["openai_api_key"], "linux-key");
    let rp = settings[0]["recent_paths"].to_string();
    assert!(rp.contains(&format!("{v}/Backups/Photos")), "{rp}");

    let second = run("merge-b".to_string(), vec![]).await;
    for t in ["thumbnails", "clip_embeddings", "logical_groups", "assistant_sessions", "assistant_messages"] {
        assert_eq!(second.rows_out(t), 0, "{t} must be a no-op the second time");
    }
    assert_eq!(import::table_counts(&db).await.unwrap(), counts);
}

#[tokio::test]
async fn insert_ignore_survives_id_and_unique_path_conflicts_without_aborting_the_batch() {
    let db = mem_db().await;
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("t.surql");
    let row = |id: &str, path: &str| format!("{{ id: thumbnails:{id}, path: '{path}', filename: 'f', size: 1, parent_dir: '/x', logical_group: logical_groups:Default, tags: [], file_type: 'jpg' }}");
    std::fs::write(&file, format!("OPTION IMPORT;\nINSERT IGNORE [ {}, {}, {} ];\n", row("one", "/x/1.jpg"), row("two", "/x/1.jpg"), row("three", "/x/3.jpg"))).unwrap();
    db.import(&file).await.unwrap();
    db.import(&file).await.unwrap();
    let ids: Vec<surrealdb::types::RecordId> = db.query("SELECT VALUE id FROM thumbnails ORDER BY id").await.unwrap().take(0).unwrap();
    assert_eq!(ids.len(), 2, "same-path row ignored, other rows kept, re-import is a no-op");
}

#[cfg(unix)]
mod engine_process {
    use super::*;
    use crate::database::legacy::engine::{self, Engine};
    use crate::database::legacy::job::EngineSource;
    use std::os::unix::fs::PermissionsExt;

    fn fake_cli(dir: &Path) -> PathBuf {
        let bin = dir.join("surreal");
        std::fs::write(
            &bin,
            "#!/bin/sh\ncase \"$1\" in\n  version) echo \"2.3.10 for linux on x86_64\";;\n  export)\n    for a in \"$@\"; do last=\"$a\"; done\n    env | grep '^SURREAL' > \"$last.env\"\n    printf 'OPTION IMPORT;\\nINSERT [ { id: thumbnails:a, path: \\\"E:\\\\\\\\x\\\\\\\\y.jpg\\\" } ];\\n' >> \"$last\"\n    ;;\n  sql)\n    while read line; do case \"$line\" in\n      \"INFO FOR ROOT;\") echo '[{\"namespaces\":{\"n1\":\"x\"}}]'; echo;;\n      \"USE NS \\`n1\\`; INFO FOR NS;\") echo '[null,{\"databases\":{\"d1\":\"x\"}}]'; echo;;\n      *) echo '[null,{\"tables\":{\"thumbnails\":\"x\"}}]'; echo;;\n    esac; done;;\n  *) exit 3;;\nesac\n",
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    #[test]
    fn child_environment_drops_every_surreal_variable() {
        // SAFETY: the variable name is unique to this test and read only by it.
        unsafe { std::env::set_var("SURREAL_AME_TEST_VAR", "1") };
        let cmd = engine::command(Path::new("/bin/true"));
        let removed = cmd.as_std().get_envs().any(|(k, v)| k == "SURREAL_AME_TEST_VAR" && v.is_none());
        assert!(removed);
    }

    #[tokio::test]
    async fn export_runs_the_cli_with_a_clean_environment_into_a_private_file() {
        // SAFETY: unique variable name, see above.
        unsafe { std::env::set_var("SURREAL_CAPS_ALLOW_EXPERIMENTAL_TEST", "files") };
        let d = tempfile::tempdir().unwrap();
        let bin = fake_cli(d.path());
        let job = Job::new();
        let opts = engine::EngineOptions { explicit: Some(bin.clone()), allow_download: false, cache_dir: Some(d.path().join("cache")) };
        let eng: Engine = engine::ensure_engine(&opts, &job).await.unwrap();
        assert_eq!((eng.version.as_str(), eng.source.clone()), ("2.3.10", EngineSource::Explicit));
        let store = d.path().join("copy");
        std::fs::create_dir_all(store.join("clog")).unwrap();
        std::fs::create_dir_all(store.join("manifest")).unwrap();
        let out = d.path().join("work/export.surql");
        let stats = engine::export(&eng, engine::StoreKind::SurrealKvLegacy, &store, "ns", "db", &out, 1000, &job).await.unwrap();
        assert!(stats.bytes > 10);
        let env = std::fs::read_to_string(format!("{}.env", out.display())).unwrap();
        assert!(!env.contains("SURREAL"), "child saw: {env}");
        let mode = std::fs::metadata(&out).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let found = engine::detect_namespaces(&eng, engine::StoreKind::SurrealKvLegacy, &store, d.path(), &job).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].dbs[0].tables, vec!["thumbnails".to_string()]);
        assert_eq!(engine::pick_namespace(&found).unwrap(), ("n1".into(), "d1".into()));
    }

    #[tokio::test]
    async fn relative_paths_resolve_against_the_callers_directory() {
        let cwd = std::env::current_dir().unwrap();
        let d = tempfile::tempdir_in(&cwd).unwrap();
        let bin = fake_cli(d.path());
        let rel = |p: &Path| p.strip_prefix(&cwd).unwrap().to_path_buf();
        let job = Job::new();
        let eng = engine::ensure_engine(&engine::EngineOptions { explicit: Some(rel(&bin)), allow_download: false, cache_dir: None }, &job).await.unwrap();
        let store = d.path().join("copy");
        std::fs::create_dir_all(store.join("clog")).unwrap();
        let out = rel(&d.path().join("work/export.surql"));
        engine::export(&eng, engine::StoreKind::SurrealKvLegacy, &rel(&store), "ns", "db", &out, 100, &job).await.unwrap();
        assert!(cwd.join(&out).is_file(), "the export lands where the relative path points from the caller's directory");
        assert!(!d.path().join("work").join(&out).exists());
    }

    #[tokio::test]
    async fn a_failing_cli_reports_its_error_and_removes_the_partial_export() {
        let d = tempfile::tempdir().unwrap();
        let bin = d.path().join("surreal");
        std::fs::write(&bin, "#!/bin/sh\nif [ \"$1\" = version ]; then echo '2.3.10 for linux'; exit 0; fi\necho 'boom: cannot open store' >&2\nexit 1\n").unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let job = Job::new();
        let eng = engine::ensure_engine(&engine::EngineOptions { explicit: Some(bin), allow_download: false, cache_dir: None }, &job).await.unwrap();
        let out = d.path().join("e.surql");
        let err = engine::export(&eng, engine::StoreKind::SurrealKvLegacy, d.path(), "a", "b", &out, 10, &job).await.unwrap_err();
        assert!(format!("{err:#}").contains("boom"), "{err:#}");
        assert!(!out.exists());
    }

    #[tokio::test]
    async fn a_three_x_cli_is_rejected_as_the_legacy_engine() {
        let d = tempfile::tempdir().unwrap();
        let bin = d.path().join("surreal");
        std::fs::write(&bin, "#!/bin/sh\necho '3.3.4 for linux on x86_64'\n").unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let job = Job::new();
        let err = engine::ensure_engine(&engine::EngineOptions { explicit: Some(bin), allow_download: false, cache_dir: None }, &job).await.unwrap_err();
        assert!(format!("{err:#}").contains("2.x"));
        assert!(matches!(job.engine_status(), crate::database::legacy::job::EngineStatus::Failed(_)));
    }

    fn tgz_with_cli(script: &[u8]) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        {
            let mut b = tar::Builder::new(&mut tar_bytes);
            let mut h = tar::Header::new_gnu();
            h.set_size(script.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            b.append_data(&mut h, "surreal", script).unwrap();
            b.finish().unwrap();
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut gz, &tar_bytes).unwrap();
        gz.finish().unwrap()
    }

    async fn serve_once(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..2 {
                let Ok((mut s, _)) = listener.accept().await else { return };
                let mut buf = [0u8; 2048];
                let _ = tokio::io::AsyncReadExt::read(&mut s, &mut buf).await;
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = tokio::io::AsyncWriteExt::write_all(&mut s, head.as_bytes()).await;
                let _ = tokio::io::AsyncWriteExt::write_all(&mut s, &body).await;
            }
        });
        format!("http://{addr}/surreal-test.tgz")
    }

    #[tokio::test]
    async fn download_verifies_the_checksum_before_installing() {
        use sha2::{Digest, Sha256};
        let script = b"#!/bin/sh\necho '2.3.10 for linux on x86_64'\n";
        let tgz = tgz_with_cli(script);
        let sha: String = Sha256::digest(&tgz).iter().map(|b| format!("{b:02x}")).collect();
        let url = serve_once(tgz).await;
        let d = tempfile::tempdir().unwrap();
        let dest = d.path().join("2.3.10").join("surreal");
        let job = Job::new();
        let good = engine::Asset { name: "surreal-test.tgz".into(), url: url.clone(), sha256: sha, kind: engine::AssetKind::Tgz, binary_sha256: None };
        engine::download_asset(&good, &dest, &job).await.unwrap();
        assert!(dest.is_file());
        assert_eq!(std::fs::metadata(&dest).unwrap().permissions().mode() & 0o111, 0o111);
        assert!(dest.with_file_name("surreal.sha256").is_file());
        assert_eq!(engine::probe_version(&dest).await.unwrap(), "2.3.10");
        let bad = engine::Asset { sha256: "0".repeat(64), ..good };
        let dest2 = d.path().join("other").join("surreal");
        let err = engine::download_asset(&bad, &dest2, &job).await.unwrap_err();
        assert!(format!("{err:#}").contains("checksum mismatch"), "{err:#}");
        assert!(!dest2.exists());
        assert!(std::fs::read_dir(dest2.parent().unwrap()).unwrap().next().is_none(), "partial download removed");
    }
}

fn request_with_cfg(f: &Fixture, target: super::Target, cfg: &Config) -> super::ImportRequest {
    let mut req = super::ImportRequest::new(f.dir.path().join("old-store"), target);
    req.from_export = Some(f.write_export());
    req.work_dir = Some(f.dir.path().join("work"));
    let map = f.dir.path().join("map.toml");
    std::fs::write(&map, cfg.to_toml().unwrap()).unwrap();
    req.path_map = Some(map);
    req
}

fn request_for(f: &Fixture, target: super::Target) -> super::ImportRequest {
    request_with_cfg(f, target, &f.cfg())
}

#[tokio::test]
async fn pipeline_new_store_then_merge_is_idempotent_and_migrates_thumbnails() {
    use super::{Target, run};
    let f = Fixture::with_png();
    let store = f.dir.path().join("db/ai_search");
    let mut strict = f.cfg();
    strict.options.missing = Policy::Drop;
    strict.options.unmapped = Policy::Drop;
    let first = run(request_with_cfg(&f, Target::NewStore(store.clone()), &strict), Job::new()).await;
    assert!(first.result.is_ok(), "{:?} {:?}", first.result.as_ref().err().map(|e| e.message()), first.report.verify.as_ref().map(|v| v.problems.clone()));
    let rep = &first.report;
    assert!(rep.ok && rep.verify.as_ref().unwrap().ok);
    assert_eq!(rep.convert.as_ref().unwrap().rows_out("thumbnails"), 6);
    assert!(rep.snapshot.is_none(), "a new store needs no rollback copy");
    assert!(matches!(crate::database::snapshot::read_stamp(&store), crate::database::snapshot::StampState::Present(_)), "engine stamp is written after a successful import");
    assert!(rep.next_steps.iter().any(|n| n.contains("--db-thumbs migrate")));
    assert!(rep.next_steps.iter().any(|n| n.contains("--db-path")));
    let json = serde_json::to_string(rep).unwrap();
    assert!(!json.contains("SECRET"));

    let more = run(request_for(&f, Target::Merge(store.clone())), Job::new()).await;
    assert!(more.result.is_ok(), "{:?} {:?}", more.result.as_ref().err().map(|e| e.message()), more.report.verify.as_ref().map(|v| v.problems.clone()));
    let c = more.report.convert.as_ref().unwrap();
    assert_eq!(c.rows_out("thumbnails"), 2, "the rows the strict policy dropped are added now");
    assert_eq!(c.tables["thumbnails"].skipped.get("already_in_target"), Some(&6));
    let snap = more.report.snapshot.clone().expect("a merge that writes rows takes a rollback snapshot");
    assert!(Path::new(&snap).is_dir());

    let again = run(request_for(&f, Target::Merge(store.clone())), Job::new()).await;
    assert!(again.result.is_ok(), "{:?}", again.result.as_ref().err().map(|e| e.message()));
    let c = again.report.convert.as_ref().unwrap();
    for t in ["thumbnails", "clip_embeddings", "logical_groups", "assistant_sessions", "assistant_messages"] {
        assert_eq!(c.rows_out(t), 0, "{t}");
    }
    assert_eq!(c.tables["thumbnails"].skipped.get("already_in_target"), Some(&8));
    assert!(again.report.snapshot.is_none(), "a no-op merge takes no snapshot");
    assert_eq!(again.report.import.as_ref().unwrap().files, 0);

    let mut req = request_for(&f, Target::Merge(store.clone()));
    req.then_migrate_thumbs = true;
    req.skip_snapshot = true;
    let third = run(req, Job::new()).await;
    assert!(third.result.is_ok(), "{:?}", third.result.as_ref().err().map(|e| e.message()));
    let m = third.report.migrate.as_ref().expect("migration ran");
    assert!(m.finish.ok, "{}", m.finish.summary);
    assert_eq!(m.copy.failed, 0);
    assert_eq!(m.copy.migrated, 7, "every row with an inline thumbnail gets a blob keyed by its mapped path");
    let state = crate::database::thumb_migration::MigrationState::load(&store);
    assert!(state.is_finished());

    let mut repeat = request_for(&f, Target::Merge(store.clone()));
    repeat.then_migrate_thumbs = true;
    repeat.skip_snapshot = true;
    let fourth = run(repeat, Job::new()).await;
    assert!(fourth.result.is_ok(), "{:?}", fourth.result.as_ref().err().map(|e| e.message()));
    assert!(fourth.report.migrate.as_ref().is_some_and(|m| m.finish.ok), "a second migration request on a finished store stays harmless");
}

#[tokio::test]
async fn pipeline_turns_imported_groups_into_collections_and_flags_later_imports() {
    use crate::database::collections;
    use crate::database::collections_migration::{self, Phase};
    use super::{Target, run};
    let f = Fixture::with_png();
    let store = f.dir.path().join("db/ai_search");
    let mut strict = f.cfg();
    strict.options.missing = Policy::Drop;
    strict.options.unmapped = Policy::Drop;
    let read_state = |store: PathBuf| async move {
        let opened = crate::database::surql::open_store(&store).await.unwrap();
        let state = collections_migration::load_state(&opened.db).await.unwrap();
        let cols = collections::list_with_counts(&opened.db).await.unwrap();
        let members = match cols.first() {
            Some(c) => collections::members(&opened.db, &c.collection.id, 0, 100).await.unwrap(),
            None => Vec::new(),
        };
        crate::database::surql::close_store(opened, &store).await;
        (state, cols, members)
    };

    let mut first_req = request_with_cfg(&f, Target::NewStore(store.clone()), &strict);
    first_req.then_migrate_collections = true;
    let first = run(first_req, Job::new()).await;
    assert!(first.result.is_ok(), "{:?}", first.result.as_ref().err().map(|e| e.message()));
    let c = first.report.collections.as_ref().expect("the collection migration ran");
    assert!(c.finish.ok, "{}", c.finish.summary);
    assert_eq!((c.copy.collections_created, c.copy.memberships_added), (1, 1), "FileHistory-OldGamesHdd holds the one file under G:\\Users");
    assert!(!first.report.next_steps.iter().any(|n| n.contains("--db-collections migrate")));
    let (state, cols, members) = read_state(store.clone()).await;
    assert_eq!(state.phase(), Phase::Finished);
    assert_eq!(cols.len(), 1);
    assert_eq!(cols[0].collection.name, "FileHistory-OldGamesHdd");
    assert_eq!(cols[0].count, 1);
    assert!(members[0].ends_with("Users/Owner/Pictures/g.jpg"), "{members:?}");

    let more = run(request_for(&f, Target::Merge(store.clone())), Job::new()).await;
    assert!(more.result.is_ok(), "{:?}", more.result.as_ref().err().map(|e| e.message()));
    let (state, ..) = read_state(store.clone()).await;
    assert!(state.needs_resync, "rows imported after the migration flag it as stale");
    assert_eq!(state.phase(), Phase::Finished);

    let mut again = request_for(&f, Target::Merge(store.clone()));
    again.then_migrate_collections = true;
    again.skip_snapshot = true;
    let third = run(again, Job::new()).await;
    assert!(third.result.is_ok(), "{:?}", third.result.as_ref().err().map(|e| e.message()));
    let c = third.report.collections.as_ref().expect("the migration ran again");
    assert!(c.finish.ok);
    assert_eq!((c.copy.collections_created, c.copy.memberships_added, c.copy.memberships_existing), (0, 0, 1), "a repeat reuses the collection and adds nothing");
    let (state, cols, _) = read_state(store).await;
    assert!(!state.needs_resync);
    assert_eq!((state.phase(), cols.len()), (Phase::Finished, 1));
}

#[tokio::test]
async fn collection_migration_requests_are_validated() {
    use super::{LegacyError, Target, run};
    let f = Fixture::new();
    let mut dry = request_for(&f, Target::NewStore(f.dir.path().join("x/store")));
    dry.then_migrate_collections = true;
    dry.dry_run = true;
    assert!(matches!(run(dry, Job::new()).await.result, Err(LegacyError::Usage(_))));
    let mut live = request_for(&f, Target::Live);
    live.then_migrate_collections = true;
    assert!(matches!(run(live, Job::new()).await.result, Err(LegacyError::Usage(_))));
}

#[tokio::test]
async fn pipeline_demands_a_reviewed_path_map_and_refuses_unsafe_requests() {
    use super::{LegacyError, Target, run};
    let f = Fixture::new();
    let store = f.dir.path().join("db/ai_search");
    let mut req = request_for(&f, Target::NewStore(store.clone()));
    req.path_map = None;
    let out = run(req.clone(), Job::new()).await;
    let Err(LegacyError::MapReview(file)) = &out.result else { panic!("review expected: {:?}", out.result.as_ref().err().map(|e| e.message())) };
    assert!(file.is_file() && file.ends_with("legacy-pathmap.toml"));
    assert!(out.report.convert.as_ref().is_some_and(|c| c.dry_run), "the mapping table is still produced");
    assert!(!store.exists() || !crate::database::snapshot::store_has_data(&store));
    Config::load(file).expect("the generated map parses");
    let mut accepted = req.clone();
    accepted.accept_suggested = true;
    accepted.dry_run = true;
    let dry = run(accepted, Job::new()).await;
    assert!(dry.result.is_ok());
    assert!(!crate::database::snapshot::store_has_data(&store), "dry runs never write a store");

    let mut bad = request_for(&f, Target::Live);
    bad.then_migrate_thumbs = true;
    assert!(matches!(run(bad, Job::new()).await.result, Err(LegacyError::Usage(_))));
    let mut same = request_for(&f, Target::Merge(f.dir.path().join("old-store")));
    same.from_export = None;
    std::fs::create_dir_all(f.dir.path().join("old-store/clog")).unwrap();
    assert!(matches!(run(same, Job::new()).await.result, Err(LegacyError::Usage(_))));
    let mut merge_missing = request_for(&f, Target::Merge(f.dir.path().join("nothing/here")));
    merge_missing.accept_suggested = true;
    assert!(matches!(run(merge_missing, Job::new()).await.result, Err(LegacyError::Usage(_))));
    let mut b64_migrate = request_for(&f, Target::NewStore(f.dir.path().join("x/store")));
    b64_migrate.then_migrate_thumbs = true;
    b64_migrate.drop_b64 = true;
    assert!(matches!(run(b64_migrate, Job::new()).await.result, Err(LegacyError::Usage(_))));
}

#[tokio::test]
async fn a_cancelled_pipeline_ends_with_the_cancelled_error_and_creates_no_store() {
    use super::{LegacyError, Target, run};
    let f = Fixture::new();
    let store = f.dir.path().join("db/ai_search");
    let job = Job::new();
    job.cancel();
    let out = run(request_for(&f, Target::NewStore(store.clone())), job).await;
    assert!(matches!(out.result, Err(LegacyError::Cancelled)));
    assert_eq!(out.result.as_ref().err().map(|e| e.exit_code()), Some(super::EXIT_CANCELLED));
    assert!(!crate::database::snapshot::store_has_data(&store));
    assert!(!out.report.ok && out.report.error.as_deref() == Some("cancelled"));
}

#[tokio::test]
async fn cancelling_stops_the_conversion_before_any_output() {
    let f = Fixture::new();
    let job = Job::new();
    job.cancel();
    let out = f.dir.path().join("cancelled");
    let mut opts = ConvertOptions::new(f.write_export());
    opts.out_dir = Some(out.clone());
    let cfg = f.cfg();
    let err = tokio::task::spawn_blocking(move || convert::convert(&opts, &cfg, None, &job)).await.unwrap().unwrap_err();
    assert!(super::job::is_cancelled(&err));
    assert!(std::fs::read_dir(&out).map(|mut d| d.next().is_none()).unwrap_or(true));
}
