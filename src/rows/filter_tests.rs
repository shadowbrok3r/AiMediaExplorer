//! Truth tables and invariants of the filter spec, its matcher, its chips and its JSON forms.

use std::collections::{HashMap, HashSet};

use chrono::{Local, TimeZone};

use super::filter::{DEFAULT_MIN_SIZE, FilterSpec, Tri, parse_size, split_tokens};
use super::filter_match::{Condition, FactSource, NoFacts, PreparedFilter};
use super::filter_probe::{Probe, ProbeCache, ProbeJob, probe_file};
use super::store::RowStore;
use super::thumbnail::{ARCHIVE_TYPE, DIR_TYPE};
use super::view::{Refresh, SortValue, ViewIndex, ViewInputs};
use super::{ChipKind, KeyPolicy};
use crate::Thumbnail;
use crate::UiSettings;

fn row(path: &str, file_type: &str) -> Thumbnail {
    let mut t = Thumbnail::default();
    t.path = path.into();
    t.filename = path.rsplit(['/', '\\']).next().unwrap_or(path).into();
    t.file_type = file_type.into();
    t.size = 50_000;
    t
}

fn sized(path: &str, size: u64) -> Thumbnail {
    let mut t = row(path, "jpg");
    t.size = size;
    t
}

fn local_time(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> surrealdb::types::Datetime {
    let at = |hour| Local.with_ymd_and_hms(y, m, d, hour, min, s).earliest();
    let local = at(h).or_else(|| at(h + 1)).expect("a valid local time");
    local.with_timezone(&chrono::Utc).into()
}

fn spec_with(edit: impl FnOnce(&mut FilterSpec)) -> FilterSpec {
    let mut spec = FilterSpec::none();
    edit(&mut spec);
    spec.normalized()
}

#[derive(Default)]
struct Facts {
    clip: HashSet<String>,
    indexed: Option<HashSet<String>>,
    probes: HashMap<String, Probe>,
}

impl FactSource for Facts {
    fn has_clip(&self, row: &Thumbnail) -> bool {
        self.clip.contains(&row.path)
    }

    fn is_indexed(&self, row: &Thumbnail) -> bool {
        self.indexed.as_ref().is_none_or(|set| set.contains(&row.path))
    }

    fn probe(&self, path: &str) -> Option<Probe> {
        self.probes.get(path).copied()
    }
}

fn passes(spec: &FilterSpec, r: &Thumbnail) -> bool {
    PreparedFilter::new(spec, &NoFacts).matches(r)
}

fn passes_with(spec: &FilterSpec, facts: &Facts, r: &Thumbnail) -> bool {
    PreparedFilter::new(spec, facts).matches(r)
}

/* ----------------------------------------- parity with the previous filter ----------------------------------------- */

/// The filter before the spec existed: type toggles and search text, kept as the reference.
fn legacy(text: &str, images: bool, videos: bool, dirs: bool, row: &Thumbnail) -> bool {
    if row.file_type == "<DIR>" && !dirs {
        return false;
    }
    if row.file_type != "<DIR>" {
        let ext_opt = if row.path.starts_with("zip://") {
            std::path::Path::new(&row.filename).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase())
        } else {
            std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase())
        };
        if let Some(ext) = ext_opt {
            if crate::is_image(ext.as_str()) && !images {
                return false;
            }
            if crate::is_video(ext.as_str()) && !videos {
                return false;
            }
        }
    }
    if text.trim().is_empty() {
        return true;
    }
    let f = text.to_lowercase();
    let tags_join = row.tags.join(",");
    row.filename.to_lowercase().contains(&f)
        || row.file_type.to_lowercase().contains(&f)
        || row.path.to_lowercase().contains(&f)
        || tags_join.to_lowercase().contains(&f)
        || row.category.as_ref().map(|s| s.to_lowercase().contains(&f)).unwrap_or(false)
}

fn type_search_spec(text: &str, images: bool, videos: bool, dirs: bool) -> FilterSpec {
    let mut spec = FilterSpec::none();
    spec.query = text.into();
    spec.types.images = images;
    spec.types.videos = videos;
    spec.types.dirs = dirs;
    spec
}

#[test]
fn the_spec_agrees_with_the_previous_filter_on_types_and_search_text() {
    let paths = ["/a/x.jpg", "/a/Y.PNG", "/a/clip.MP4", "/a/notes.txt", "/a/d", "zip:///z.zip!/in/p.png", "C:\\pics\\q.JPEG", "/a/.hidden", "/a/noext", "/ü/é.jpg"];
    let types = ["jpg", "<DIR>", "<ARCHIVE>", "txt", ""];
    let texts = ["", "x", "PNG", "clip", "a/", "é", "tag1", "cat", "j,t", " ", "  x", "İ"];
    let mut n = 0;
    for p in paths {
        for t in types {
            let mut r = row(p, t);
            r.tags = vec!["Tag1".into(), "two".into()];
            r.category = Some("Cat".into());
            if p.starts_with("zip://") {
                r.filename = "p.png".into();
            }
            for text in texts {
                for flags in 0..8u8 {
                    let (i, v, d) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                    let spec = type_search_spec(text, i, v, d);
                    assert_eq!(passes(&spec, &r), legacy(text, i, v, d, &r), "{p} {t} {text:?} {flags}");
                    n += 1;
                }
            }
        }
    }
    assert!(n > 1000);
}

#[test]
fn type_toggles_hide_images_videos_and_folders_independently() {
    let img = row("/a/x.jpg", "jpg");
    let vid = row("/a/y.mp4", "mp4");
    let dir = row("/a/d", DIR_TYPE);
    let txt = row("/a/t.txt", "txt");
    let arc = row("/a/z.zip", ARCHIVE_TYPE);
    assert!(!passes(&type_search_spec("", false, true, true), &img));
    assert!(passes(&type_search_spec("", false, true, true), &vid));
    assert!(!passes(&type_search_spec("", true, false, true), &vid));
    assert!(!passes(&type_search_spec("", true, true, false), &dir));
    let none_shown = type_search_spec("", false, false, false);
    assert!(passes(&none_shown, &txt) && passes(&none_shown, &arc), "other files and archives are not hidden by those three toggles");
}

#[test]
fn archives_and_other_files_have_their_own_toggles() {
    let txt = row("/a/t.txt", "txt");
    let zip = row("/a/z.zip", ARCHIVE_TYPE);
    let zip_by_ext = row("/a/y.7z", "7z");
    let img = row("/a/x.jpg", "jpg");
    let spec = spec_with(|s| s.types.archives = false);
    assert!(!passes(&spec, &zip) && !passes(&spec, &zip_by_ext));
    assert!(passes(&spec, &txt) && passes(&spec, &img));
    let spec = spec_with(|s| s.types.other = false);
    assert!(!passes(&spec, &txt));
    assert!(passes(&spec, &zip) && passes(&spec, &img));
}

#[test]
fn zip_entries_take_their_type_from_the_file_name() {
    let mut entry = row("zip:///x/a.zip!/inner/pic", "png");
    entry.filename = "pic.png".into();
    assert!(!passes(&type_search_spec("", false, true, true), &entry));
    assert!(passes(&type_search_spec("", true, true, true), &entry));
}

#[test]
fn the_search_text_matches_name_type_path_tags_and_category_ignoring_case() {
    let mut r = row("/pics/Holiday/IMG_001.JPG", "jpg");
    r.tags = vec!["Beach".into(), "sun".into()];
    r.category = Some("Travel".into());
    for hit in ["img_001", "holiday", "JPG", "beach", "SUN", "travel", "beach,sun"] {
        assert!(passes(&spec_with(|s| s.query = hit.into()), &r), "{hit}");
    }
    for miss in ["mountain", "img_002", "snow"] {
        assert!(!passes(&spec_with(|s| s.query = miss.into()), &r), "{miss}");
    }
    assert!(passes(&spec_with(|s| s.query = "   ".into()), &r), "blank text is no filter");
}

/* --------------------------------------------------- size ---------------------------------------------------- */

#[test]
fn size_bounds_are_inclusive_and_folders_are_exempt() {
    let spec = spec_with(|s| {
        s.size.min = Some(10_240);
        s.size.max = Some(20_480);
    });
    let table = [(0, false), (10_239, false), (10_240, true), (15_000, true), (20_480, true), (20_481, false), (u64::MAX, false)];
    for (size, expect) in table {
        assert_eq!(passes(&spec, &sized("/a/x.jpg", size)), expect, "{size}");
    }
    assert!(passes(&spec, &row("/a/sub", DIR_TYPE)), "folders have no size to compare");
}

#[test]
fn a_min_only_and_a_max_only_bound_work_alone() {
    let min = spec_with(|s| s.size.min = Some(100));
    let max = spec_with(|s| s.size.max = Some(100));
    assert!(!passes(&min, &sized("/a/x.jpg", 99)) && passes(&min, &sized("/a/x.jpg", u64::MAX)));
    assert!(passes(&max, &sized("/a/x.jpg", 0)) && !passes(&max, &sized("/a/x.jpg", 101)));
}

#[test]
fn an_unknown_size_of_a_virtual_entry_passes_but_an_empty_file_does_not() {
    let spec = spec_with(|s| s.size.min = Some(10));
    assert!(passes(&spec, &sized("zip:///a.zip!/in/x.jpg", 0)));
    assert!(!passes(&spec, &sized("/a/empty.jpg", 0)));
}

#[test]
fn the_default_spec_hides_small_files_and_nothing_else() {
    let spec = FilterSpec::default();
    assert_eq!(spec.size.min, Some(DEFAULT_MIN_SIZE));
    assert_eq!(spec.chips().len(), 1);
    assert!(!passes(&spec, &sized("/a/tiny.jpg", DEFAULT_MIN_SIZE - 1)));
    assert!(passes(&spec, &sized("/a/ok.jpg", DEFAULT_MIN_SIZE)));
    assert!(FilterSpec::none().is_empty() && !spec.is_empty());
}

#[test]
fn the_default_spec_is_what_the_default_settings_define() {
    assert_eq!(FilterSpec::from_settings(&UiSettings::default()), FilterSpec::default());
}

/* --------------------------------------------------- dates --------------------------------------------------- */

#[test]
fn a_date_range_covers_whole_local_days() {
    let spec = spec_with(|s| {
        s.modified.after = Some("2024-01-10".into());
        s.modified.before = Some("2024-01-20".into());
    });
    let at = |y, m, d, h, min, sec| {
        let mut r = row("/a/x.jpg", "jpg");
        r.modified = Some(local_time(y, m, d, h, min, sec));
        r
    };
    assert!(!passes(&spec, &at(2024, 1, 9, 23, 59, 59)));
    assert!(passes(&spec, &at(2024, 1, 10, 0, 0, 0)));
    assert!(passes(&spec, &at(2024, 1, 15, 12, 0, 0)));
    assert!(passes(&spec, &at(2024, 1, 20, 23, 59, 59)));
    assert!(!passes(&spec, &at(2024, 1, 21, 0, 0, 0)));
}

#[test]
fn one_sided_ranges_and_missing_dates() {
    let since = spec_with(|s| s.modified.after = Some("2024-06-01".into()));
    let until = spec_with(|s| s.modified.before = Some("2024-06-01".into()));
    let mut r = row("/a/x.jpg", "jpg");
    r.modified = Some(local_time(2024, 6, 1, 8, 0, 0));
    assert!(passes(&since, &r) && passes(&until, &r));
    r.modified = Some(local_time(2023, 12, 31, 8, 0, 0));
    assert!(!passes(&since, &r) && passes(&until, &r));
    r.modified = None;
    assert!(!passes(&since, &r) && !passes(&until, &r), "a row without a date cannot meet a date limit");
    assert!(passes(&FilterSpec::none(), &r));
    assert!(passes(&since, &row("/a/sub", DIR_TYPE)), "folders are exempt");
}

#[test]
fn the_created_range_reads_the_library_date() {
    let spec = spec_with(|s| s.created.after = Some("2024-03-01".into()));
    let mut r = row("/a/x.jpg", "jpg");
    r.db_created = Some(local_time(2024, 3, 2, 0, 0, 0));
    r.modified = Some(local_time(2020, 1, 1, 0, 0, 0));
    assert!(passes(&spec, &r));
    r.db_created = Some(local_time(2024, 2, 28, 0, 0, 0));
    assert!(!passes(&spec, &r));
}

#[test]
fn bad_dates_are_rejected_by_the_checks() {
    let mut spec = FilterSpec::none();
    spec.modified.after = Some("31/01/2024".into());
    assert!(spec.check().unwrap_err().contains("modified.after"));
    spec.modified.after = Some("2024-02-30".into());
    assert!(spec.check().is_err());
    spec.modified.after = Some("2024-02-29".into());
    assert!(spec.check().is_ok());
}

/* ----------------------------------------------- terms, folders ----------------------------------------------- */

#[test]
fn excluded_terms_match_any_part_of_the_path_ignoring_case() {
    let spec = spec_with(|s| s.excluded_terms = vec!["TMP".into(), "ünï".into()]);
    assert!(!passes(&spec, &row("/a/tmp/x.jpg", "jpg")));
    assert!(!passes(&spec, &row("/a/x_Tmp.jpg", "jpg")));
    assert!(!passes(&spec, &row("/a/ÜNÏ/x.jpg", "jpg")));
    assert!(passes(&spec, &row("/a/ok/x.jpg", "jpg")));
    assert!(passes(&spec, &row("/a/tmp", DIR_TYPE)), "folders stay listed");
}

#[test]
fn excluded_folders_hide_everything_below_and_the_folder_itself() {
    let spec = spec_with(|s| s.excluded_dirs = vec!["/p/sub".into()]);
    assert!(!passes(&spec, &row("/p/sub/x.jpg", "jpg")));
    assert!(!passes(&spec, &row("/p/sub/deep/er/x.jpg", "jpg")));
    assert!(!passes(&spec, &row("/p/sub", DIR_TYPE)));
    assert!(passes(&spec, &row("/p/subway/x.jpg", "jpg")), "a sibling that shares the prefix stays");
    assert!(passes(&spec, &row("/p/other/x.jpg", "jpg")));
    let slash = spec_with(|s| s.excluded_dirs = vec!["/p/sub/".into()]);
    assert!(!passes(&slash, &row("/p/sub/x.jpg", "jpg")));
    assert!(passes(&slash, &row("/p/subway/x.jpg", "jpg")));
}

#[test]
fn extension_lists_allow_only_or_deny() {
    let allow = spec_with(|s| s.ext_allow = vec!["JPG".into(), ".png".into(), "*.webp".into()]);
    assert_eq!(allow.ext_allow, ["jpg", "png", "webp"]);
    assert!(passes(&allow, &row("/a/x.jpg", "jpg")) && passes(&allow, &row("/a/x.PNG", "png")) && passes(&allow, &row("/a/x.webp", "webp")));
    assert!(!passes(&allow, &row("/a/x.gif", "gif")) && !passes(&allow, &row("/a/noext", "")) && !passes(&allow, &row("/a/.jpg", "")));
    assert!(passes(&allow, &row("/a/sub", DIR_TYPE)));
    let deny = spec_with(|s| s.ext_deny = vec!["gif".into(), "tar.gz".into()]);
    assert!(!passes(&deny, &row("/a/x.GIF", "gif")));
    assert!(passes(&deny, &row("/a/x.jpg", "jpg")) && passes(&deny, &row("/a/noext", "")));
    assert!(passes(&deny, &row("/a/y.tar.gz", "gz")), "only the last extension is compared");
    let mut entry = row("zip:///x.zip!/in/pic", "gif");
    entry.filename = "pic.gif".into();
    assert!(!passes(&deny, &entry), "archive entries use their entry name");
}

#[test]
fn name_and_search_texts_are_separate_conditions() {
    let mut r = row("/pics/Holiday/IMG_001.JPG", "jpg");
    r.tags = vec!["holiday".into()];
    assert!(passes(&spec_with(|s| s.name = "img_0".into()), &r));
    assert!(!passes(&spec_with(|s| s.name = "holiday".into()), &r), "the folder name is not the file name");
    assert!(passes(&spec_with(|s| s.query = "holiday".into()), &r));
    assert!(passes(&spec_with(|s| s.name = "holiday".into()), &row("/x/Holiday", DIR_TYPE)), "folder names are matched");
}

/* ------------------------------------------------- metadata -------------------------------------------------- */

#[test]
fn description_category_and_tags() {
    let mut r = row("/a/x.jpg", "jpg");
    let yes = spec_with(|s| s.description = Tri::Yes);
    let no = spec_with(|s| s.description = Tri::No);
    assert!(!passes(&yes, &r) && passes(&no, &r));
    r.description = Some("   ".into());
    assert!(!passes(&yes, &r) && passes(&no, &r), "blank text is no description");
    r.description = Some("a cat".into());
    assert!(passes(&yes, &r) && !passes(&no, &r));

    let cats = spec_with(|s| s.categories = vec!["Travel".into(), "FOOD".into()]);
    assert!(!passes(&cats, &r));
    r.category = Some(" travel ".into());
    assert!(passes(&cats, &r));
    r.category = Some("food".into());
    assert!(passes(&cats, &r));
    r.category = Some("travels".into());
    assert!(!passes(&cats, &r), "categories match whole names");

    r.tags = vec!["Sun".into(), "sea".into()];
    let any = spec_with(|s| s.tags = vec!["sun".into(), "rain".into()]);
    let all = spec_with(|s| {
        s.tags = vec!["sun".into(), "rain".into()];
        s.tags_all = true;
    });
    assert!(passes(&any, &r) && !passes(&all, &r));
    r.tags.push("RAIN".into());
    assert!(passes(&all, &r));
    assert!(passes(&any, &row("/a/sub", DIR_TYPE)) && passes(&all, &row("/a/sub", DIR_TYPE)));
}

#[test]
fn clip_and_indexed_come_from_the_fact_source() {
    let a = row("/a/a.jpg", "jpg");
    let b = row("/a/b.jpg", "jpg");
    let mut facts = Facts::default();
    facts.clip.insert("/a/a.jpg".into());
    facts.indexed = Some(HashSet::from(["/a/b.jpg".to_string()]));
    for (tri, a_ok, b_ok) in [(Tri::Any, true, true), (Tri::Yes, true, false), (Tri::No, false, true)] {
        let clip = spec_with(|s| s.clip = tri);
        assert_eq!((passes_with(&clip, &facts, &a), passes_with(&clip, &facts, &b)), (a_ok, b_ok), "clip {tri:?}");
    }
    for (tri, a_ok, b_ok) in [(Tri::Any, true, true), (Tri::Yes, false, true), (Tri::No, true, false)] {
        let indexed = spec_with(|s| s.indexed = tri);
        assert_eq!((passes_with(&indexed, &facts, &a), passes_with(&indexed, &facts, &b)), (a_ok, b_ok), "indexed {tri:?}");
    }
}

#[test]
fn offline_needs_a_probe_and_virtual_paths_are_online() {
    let r = row("/gone/x.jpg", "jpg");
    let mut facts = Facts::default();
    let only_offline = spec_with(|s| s.offline = Tri::Yes);
    let only_online = spec_with(|s| s.offline = Tri::No);
    assert!(!passes_with(&only_offline, &facts, &r), "unknown is not offline");
    assert!(passes_with(&only_online, &facts, &r), "unknown is not shown as offline either");
    facts.probes.insert(r.path.clone(), Probe { exists: false, dims_checked: false, tiny: false });
    assert!(passes_with(&only_offline, &facts, &r) && !passes_with(&only_online, &facts, &r));
    facts.probes.insert(r.path.clone(), Probe { exists: true, dims_checked: false, tiny: false });
    assert!(!passes_with(&only_offline, &facts, &r) && passes_with(&only_online, &facts, &r));
    let virt = row("zip:///a.zip!/x.jpg", "jpg");
    assert!(!passes_with(&only_offline, &facts, &virt) && passes_with(&only_online, &facts, &virt));
}

#[test]
fn skip_icons_hides_ico_files_and_probed_tiny_small_images() {
    let spec = spec_with(|s| s.skip_icons = true);
    let mut facts = Facts::default();
    let mut icon = sized("/a/logo.png", 800);
    let ico = sized("/a/favicon.ICO", 80_000);
    let big = sized("/a/photo.jpg", 5_000_000);
    let mut small_photo = sized("/a/small.jpg", 900);
    let doc = sized("/a/notes.txt", 10);
    assert!(!passes_with(&spec, &facts, &ico));
    assert!(passes_with(&spec, &facts, &icon), "an image that was not probed yet is listed");
    facts.probes.insert(icon.path.clone(), Probe { exists: true, dims_checked: true, tiny: true });
    facts.probes.insert(small_photo.path.clone(), Probe { exists: true, dims_checked: true, tiny: false });
    assert!(!passes_with(&spec, &facts, &icon));
    assert!(passes_with(&spec, &facts, &small_photo) && passes_with(&spec, &facts, &big) && passes_with(&spec, &facts, &doc));
    icon.size = 5_000_000;
    assert!(passes_with(&spec, &facts, &icon), "only images under the size limit count as icons");
    small_photo.size = 0;
    assert!(passes_with(&spec, &facts, &small_photo));
}

/* ------------------------------------------- explain and the report -------------------------------------------- */

#[test]
fn explain_names_the_failing_conditions() {
    let spec = spec_with(|s| {
        s.size.min = Some(10_000);
        s.excluded_terms = vec!["tmp".into()];
        s.types.videos = false;
    });
    let facts = NoFacts;
    let filter = PreparedFilter::new(&spec, &facts);
    let verdicts = filter.explain(&sized("/a/tmp/clip.mp4", 5));
    let failing: Vec<Condition> = verdicts.iter().filter(|v| !v.passed).map(|v| v.condition).collect();
    assert_eq!(failing, [Condition::Types, Condition::Size, Condition::ExcludedTerms]);
    assert!(verdicts.iter().find(|v| v.condition == Condition::Size).unwrap().detail.contains("5 B"));
    assert!(verdicts.iter().find(|v| v.condition == Condition::ExcludedTerms).unwrap().detail.contains("tmp"));
    assert!(filter.explain(&sized("/a/ok/pic.jpg", 20_000)).iter().all(|v| v.passed));
}

#[test]
fn the_report_counts_what_each_condition_hides_and_what_removing_it_would_list() {
    let spec = spec_with(|s| {
        s.size.min = Some(1000);
        s.excluded_terms = vec!["tmp".into()];
    });
    let rows = vec![sized("/a/ok.jpg", 5000), sized("/a/small.jpg", 10), sized("/a/tmp/big.jpg", 5000), sized("/a/tmp/small.jpg", 10), sized("/a/tmp2/big.jpg", 9000)];
    let report = PreparedFilter::new(&spec, &NoFacts).report(rows.iter());
    assert_eq!((report.total, report.visible), (5, 1));
    let by = |c: Condition| report.per_condition.iter().find(|x| x.condition == c).copied().unwrap();
    assert_eq!((by(Condition::Size).failing, by(Condition::Size).only), (2, 1));
    assert_eq!((by(Condition::ExcludedTerms).failing, by(Condition::ExcludedTerms).only), (3, 2));
}

#[test]
fn an_empty_spec_lists_everything_and_has_no_active_condition() {
    let filter = PreparedFilter::new(&FilterSpec::none(), &NoFacts);
    assert!(filter.active().is_empty());
    assert!(filter.matches(&row("", "")));
}

/* -------------------------------------------- unusual file names -------------------------------------------- */

fn unusual_names() -> Vec<String> {
    let mut names: Vec<String> = [
        "a b.jpg", "  lead.jpg", "trail .jpg", ".hidden", "noext", "dot.", "..", "UPPER.JPG", "MiXeD.PnG", "ünï.jpg", "日本語.png", "😀.gif", "100%.jpg", "a'b\"c.jpg", "back\\slash.jpg", "tab\tname.jpg",
        "multi.part.name.tar.gz", "ß.jpg", "İstanbul.JPG", "a.b.c.d.e.f.g.h.i.jpg", "-dash.jpg", "[brackets].jpg", "{curly}.jpg", "semi;colon.jpg", "emoji😀name.mp4", "عربى.jpg", "\u{202e}rtl.jpg", "x.jpg.exe", ".jpg",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    names.push(format!("very_long_{}.jpg", "x".repeat(300)));
    names
}

#[test]
fn unusual_names_never_panic_and_follow_the_same_rules_as_the_previous_filter() {
    let texts = ["", "jpg", "ü", "日本", "😀", "%", "'", "\"", "\\", "\t", ".", "..", "ß", "İ", "i", "x", "tar.gz"];
    for name in unusual_names() {
        for dir in ["/p", "C:\\p", "zip:///a.zip!/in", "/ü/日本"] {
            let path = format!("{dir}/{name}");
            for file_type in ["jpg", DIR_TYPE, ARCHIVE_TYPE, ""] {
                let mut r = row(&path, file_type);
                r.filename = name.clone();
                for text in texts {
                    for flags in [0b111u8, 0b110, 0b101, 0b011, 0b000] {
                        let (i, v, d) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                        let spec = type_search_spec(text, i, v, d);
                        assert_eq!(passes(&spec, &r), legacy(text, i, v, d, &r), "{path:?} {file_type} {text:?} {flags}");
                    }
                }
                let everything = spec_with(|s| {
                    s.size.min = Some(1);
                    s.excluded_terms = vec!["ü".into(), "x".into()];
                    s.excluded_dirs = vec!["/p".into()];
                    s.name = "j".into();
                    s.ext_deny = vec!["exe".into()];
                    s.ext_allow = vec!["jpg".into(), "png".into()];
                    s.modified.after = Some("2000-01-01".into());
                });
                let _ = passes(&everything, &r);
                let _ = PreparedFilter::new(&everything, &NoFacts).explain(&r);
            }
        }
    }
}

#[test]
fn terms_with_regex_or_glob_characters_match_literally() {
    let r = row("/a/file[1](x).jpg", "jpg");
    for term in ["[1]", "(x)", ".", "*", "?", "\\", "^", "$"] {
        let hit = r.path.contains(term);
        assert_eq!(!passes(&spec_with(|s| s.excluded_terms = vec![term.into()]), &r), hit, "{term}");
    }
}

/* ----------------------------------------------- parsing helpers ----------------------------------------------- */

#[test]
fn sizes_parse_with_binary_units() {
    let table: [(&str, Option<u64>); 14] = [
        ("", None),
        ("  ", None),
        ("none", None),
        ("-", None),
        ("4096", Some(4096)),
        ("10 KiB", Some(10 * 1024)),
        ("10kb", Some(10 * 1024)),
        ("2k", Some(2048)),
        ("1.5 MiB", Some(1_572_864)),
        ("2mb", Some(2 * 1024 * 1024)),
        ("1 GiB", Some(1 << 30)),
        ("0", Some(0)),
        (" 7 B ", Some(7)),
        ("1tb", Some(1 << 40)),
    ];
    for (text, expect) in table {
        assert_eq!(parse_size(text), Ok(expect), "{text:?}");
    }
    for bad in ["abc", "10 parsecs", "-5", "1..2", "kb"] {
        assert!(parse_size(bad).is_err(), "{bad}");
    }
}

#[test]
fn tokens_split_at_commas_semicolons_and_spaces() {
    assert_eq!(split_tokens("jpg, png;webp  gif\t,"), ["jpg", "png", "webp", "gif"]);
    assert!(split_tokens(" ,; ").is_empty());
}

/* ------------------------------------------------ chips and edits ------------------------------------------------ */

fn busy_spec() -> FilterSpec {
    spec_with(|s| {
        s.query = "cat".into();
        s.name = "img".into();
        s.types.images = false;
        s.types.other = false;
        s.size.min = Some(10 * 1024);
        s.size.max = Some(5 * 1024 * 1024);
        s.modified.after = Some("2024-01-01".into());
        s.created.before = Some("2024-12-31".into());
        s.skip_icons = true;
        s.ext_allow = vec!["jpg".into(), "png".into()];
        s.ext_deny = vec!["gif".into()];
        s.indexed = Tri::Yes;
        s.description = Tri::No;
        s.clip = Tri::Yes;
        s.offline = Tri::No;
        s.categories = vec!["Travel".into()];
        s.tags = vec!["sun".into(), "sea".into()];
        s.tags_all = true;
        s.excluded_terms = vec!["tmp".into(), "cache".into()];
        s.excluded_dirs = vec!["/p/sub".into()];
    })
}

#[test]
fn every_active_condition_has_a_chip_and_removing_it_clears_exactly_that() {
    let spec = busy_spec();
    let chips = spec.chips();
    let labels: Vec<&str> = chips.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "Search: cat",
            "Name: img",
            "Hiding: Images, Other files",
            "Size 10 KiB - 5 MiB",
            "Modified since 2024-01-01",
            "Added until 2024-12-31",
            "Skip icons",
            "Only .jpg, .png",
            "Not .gif",
            "Indexed",
            "No description",
            "Has CLIP",
            "Online",
            "Category: Travel",
            "Tag (all): sun",
            "Tag (all): sea",
            "Not matching: tmp",
            "Not matching: cache",
            "Excluding folder: sub",
        ]
    );
    let mut left = spec.clone();
    for chip in &chips {
        let before = left.chips().len();
        left.remove(&chip.kind);
        assert_eq!(left.chips().len(), before - 1, "{:?}", chip.kind);
        assert!(!left.chips().iter().any(|c| c.kind == chip.kind), "{:?}", chip.kind);
    }
    assert!(left.is_empty(), "{left:?}");
}

#[test]
fn removing_a_tag_chip_drops_the_match_all_flag_when_fewer_than_two_remain() {
    let mut spec = spec_with(|s| {
        s.tags = vec!["a".into(), "b".into()];
        s.tags_all = true;
    });
    spec.remove(&ChipKind::Tag("a".into()));
    assert_eq!(spec.tags, ["b"]);
    assert!(!spec.tags_all);
}

#[test]
fn long_chip_texts_are_shortened() {
    let spec = spec_with(|s| s.query = "a very long search phrase indeed".into());
    assert_eq!(spec.chips()[0].label, "Search: a very long search ph...");
}

#[test]
fn normalizing_cleans_lists_and_orders_the_size_bounds() {
    let spec = FilterSpec {
        excluded_terms: vec!["  Tmp ".into(), "tmp".into(), "".into(), "CACHE".into()],
        ext_allow: vec![" .JPG".into(), "jpg".into(), "*.png".into(), ".".into()],
        categories: vec![" Travel".into(), "travel".into(), " ".into()],
        tags: vec!["Sun".into(), "sun ".into()],
        size: super::filter::SizeRange { min: Some(500), max: Some(100) },
        name: "  x ".into(),
        modified: super::filter::DateRange { after: Some(" 2024-01-01 ".into()), before: Some("  ".into()) },
        ..FilterSpec::none()
    }
    .normalized();
    assert_eq!(spec.excluded_terms, ["tmp", "cache"]);
    assert_eq!(spec.ext_allow, ["jpg", "png"]);
    assert_eq!(spec.categories, ["Travel"]);
    assert_eq!(spec.tags, ["Sun"]);
    assert_eq!((spec.size.min, spec.size.max), (Some(100), Some(500)));
    assert_eq!(spec.name, "x");
    assert_eq!((spec.modified.after.as_deref(), spec.modified.before.as_deref()), (Some("2024-01-01"), None));
    assert_eq!(spec.clone().normalized(), spec, "normalizing twice changes nothing");
}

/* ----------------------------------------------------- JSON ----------------------------------------------------- */

#[test]
fn a_spec_round_trips_through_json() {
    for spec in [FilterSpec::default(), FilterSpec::none(), busy_spec()] {
        assert_eq!(FilterSpec::from_json(&spec.to_json()).unwrap(), spec);
    }
}

#[test]
fn missing_json_fields_take_the_app_defaults_and_unknown_ones_are_errors() {
    assert_eq!(FilterSpec::from_json("{}").unwrap(), FilterSpec::default());
    let partial = FilterSpec::from_json(r#"{"types":{"videos":false}}"#).unwrap();
    assert!(!partial.types.videos && partial.types.images && partial.size.min == Some(DEFAULT_MIN_SIZE));
    assert!(FilterSpec::from_json(r#"{"colour":"red"}"#).unwrap_err().contains("colour"));
    assert!(FilterSpec::from_json(r#"{"types":{"gifs":false}}"#).is_err());
    assert!(FilterSpec::from_json(r#"{"indexed":"maybe"}"#).is_err());
    assert!(FilterSpec::from_json(r#"{"modified":{"after":"yesterday"}}"#).is_err());
    assert!(FilterSpec::from_json("[1]").is_err());
}

#[test]
fn a_merge_patch_changes_only_what_it_names_and_null_turns_a_part_off() {
    let base = spec_with(|s| {
        s.size.min = Some(100);
        s.excluded_terms = vec!["tmp".into()];
        s.types.videos = false;
    });
    let patch = serde_json::json!({"size": {"max": 900}, "tags": ["Sun"], "types": {"images": false}});
    let merged = base.merged(&patch, false).unwrap();
    assert_eq!((merged.size.min, merged.size.max), (Some(100), Some(900)));
    assert_eq!(merged.excluded_terms, ["tmp"]);
    assert_eq!(merged.tags, ["Sun"]);
    assert!(!merged.types.images && !merged.types.videos);

    let off = merged.merged(&serde_json::json!({"size": null, "types": null, "excluded_terms": null, "tags": null}), false).unwrap();
    assert!(off.is_empty(), "{off:?}");
    let min_off = base.merged(&serde_json::json!({"size": {"min": null}}), false).unwrap();
    assert_eq!(min_off.size.min, None);
}

#[test]
fn a_replacing_patch_starts_from_the_defaults() {
    let base = busy_spec();
    let replaced = base.merged(&serde_json::json!({"skip_icons": true}), true).unwrap();
    assert!(replaced.skip_icons);
    assert_eq!(replaced.size.min, Some(DEFAULT_MIN_SIZE));
    assert_eq!(replaced.chips().len(), 2);
    assert_eq!(base.merged(&serde_json::json!({}), true).unwrap(), FilterSpec::default());
}

#[test]
fn bad_patches_are_errors_and_leave_the_spec_alone() {
    let base = busy_spec();
    for bad in [serde_json::json!([1]), serde_json::json!("x"), serde_json::json!({"nope": 1}), serde_json::json!({"size": {"min": "big"}}), serde_json::json!({"modified": {"after": "x"}})] {
        assert!(base.merged(&bad, false).is_err(), "{bad}");
    }
    assert_eq!(base.merged(&serde_json::json!({"query": null}), false).unwrap().query, "", "null clears text");
}

/* ----------------------------------------------- settings link ----------------------------------------------- */

#[test]
fn settings_round_trip_through_the_spec() {
    let mut settings = UiSettings::default();
    settings.db_min_size_bytes = Some(2048);
    settings.db_max_size_bytes = Some(9_999_999);
    settings.filter_modified_after = Some("2023-05-05".into());
    settings.filter_skip_icons = true;
    settings.db_excluded_exts = Some(vec!["TMP".into(), "bak".into()]);
    settings.excluded_dirs = Some(vec!["/x/y".into()]);
    let spec = FilterSpec::from_settings(&settings);
    assert_eq!(spec.ext_deny, ["tmp", "bak"]);
    let mut written = UiSettings::default();
    spec.write_settings(&mut written);
    assert_eq!((written.db_min_size_bytes, written.db_max_size_bytes), (Some(2048), Some(9_999_999)));
    assert_eq!(written.filter_modified_after.as_deref(), Some("2023-05-05"));
    assert!(written.filter_skip_icons);
    assert_eq!(written.db_excluded_exts, Some(vec!["tmp".to_string(), "bak".to_string()]));
    assert_eq!(written.excluded_dirs, Some(vec!["/x/y".to_string()]));
    FilterSpec::none().write_settings(&mut written);
    assert_eq!((written.db_min_size_bytes, written.db_excluded_exts, written.excluded_dirs), (None, None, None));
}

#[test]
fn a_tab_follows_the_settings_for_every_group_it_did_not_change() {
    let old = FilterSpec::default();
    let mut new = old.clone();
    new.size.min = Some(1);
    new.skip_icons = true;
    new.excluded_dirs = vec!["/e".into()];
    let mut untouched = old.clone();
    untouched.follow(&old, &new);
    assert_eq!(untouched, new);
    let mut edited = old.clone();
    edited.size.min = Some(777);
    edited.query = "keep".into();
    edited.follow(&old, &new);
    assert_eq!(edited.size.min, Some(777), "the tab's own size limit stays");
    assert!(edited.skip_icons && edited.excluded_dirs == ["/e"], "the other groups follow");
    assert_eq!(edited.query, "keep");
}

/* ------------------------------------------------- signature -------------------------------------------------- */

#[test]
fn equal_specs_share_a_signature_and_every_field_moves_it() {
    let base = FilterSpec::default();
    assert_eq!(base.signature(), FilterSpec::default().signature());
    let edits: Vec<(&str, Box<dyn Fn(&mut FilterSpec)>)> = vec![
        ("types.images", Box::new(|s| s.types.images = false)),
        ("types.videos", Box::new(|s| s.types.videos = false)),
        ("types.dirs", Box::new(|s| s.types.dirs = false)),
        ("types.archives", Box::new(|s| s.types.archives = false)),
        ("types.other", Box::new(|s| s.types.other = false)),
        ("size.min", Box::new(|s| s.size.min = None)),
        ("size.max", Box::new(|s| s.size.max = Some(5))),
        ("modified.after", Box::new(|s| s.modified.after = Some("2024-01-01".into()))),
        ("modified.before", Box::new(|s| s.modified.before = Some("2024-01-01".into()))),
        ("created.after", Box::new(|s| s.created.after = Some("2024-01-01".into()))),
        ("created.before", Box::new(|s| s.created.before = Some("2024-01-01".into()))),
        ("excluded_terms", Box::new(|s| s.excluded_terms.push("x".into()))),
        ("excluded_dirs", Box::new(|s| s.excluded_dirs.push("/x".into()))),
        ("query", Box::new(|s| s.query = "x".into())),
        ("name", Box::new(|s| s.name = "x".into())),
        ("skip_icons", Box::new(|s| s.skip_icons = true)),
        ("ext_allow", Box::new(|s| s.ext_allow.push("jpg".into()))),
        ("ext_deny", Box::new(|s| s.ext_deny.push("gif".into()))),
        ("indexed", Box::new(|s| s.indexed = Tri::Yes)),
        ("description", Box::new(|s| s.description = Tri::No)),
        ("clip", Box::new(|s| s.clip = Tri::Yes)),
        ("categories", Box::new(|s| s.categories.push("c".into()))),
        ("tags", Box::new(|s| s.tags.push("t".into()))),
        ("tags_all", Box::new(|s| s.tags_all = true)),
        ("offline", Box::new(|s| s.offline = Tri::Yes)),
    ];
    let mut seen = HashSet::from([base.signature()]);
    for (name, edit) in &edits {
        let mut spec = base.clone();
        edit(&mut spec);
        assert!(seen.insert(spec.signature()), "{name} does not move the signature");
    }
}

#[test]
fn the_overlay_signature_replaces_the_text_and_the_types() {
    let mut a = FilterSpec::default();
    a.query = "left".into();
    a.types.videos = false;
    let mut b = FilterSpec::default();
    b.query = "right".into();
    assert_eq!(a.signature_with("right", b.types), b.signature());
    assert_ne!(a.signature(), a.signature_with("right", a.types));
}

/* ------------------------------------------------ probing files ------------------------------------------------ */

fn write_png(path: &std::path::Path, w: u32, h: u32) {
    image::RgbImage::from_pixel(w, h, image::Rgb([10, 20, 30])).save(path).unwrap();
}

#[test]
fn probing_reads_existence_and_image_size_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let small = dir.path().join("small.png");
    let large = dir.path().join("large.png");
    let text = dir.path().join("notes.txt");
    write_png(&small, 16, 16);
    write_png(&large, 300, 200);
    std::fs::write(&text, "hello").unwrap();
    let s = |p: &std::path::Path| p.to_str().unwrap().to_string();
    assert_eq!(probe_file(&s(&small), true), Probe { exists: true, dims_checked: true, tiny: true });
    assert_eq!(probe_file(&s(&large), true), Probe { exists: true, dims_checked: true, tiny: false });
    assert_eq!(probe_file(&s(&small), false), Probe { exists: true, dims_checked: false, tiny: false });
    assert_eq!(probe_file(&s(&text), true), Probe { exists: true, dims_checked: true, tiny: false }, "an unreadable image is not tiny");
    assert_eq!(probe_file(&s(&dir.path().join("missing.png")), true), Probe { exists: false, dims_checked: false, tiny: false });
}

#[test]
fn the_probe_cache_fills_in_the_background_and_skips_answered_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let mut jobs = Vec::new();
    for i in 0..600 {
        let p = dir.path().join(format!("f{i}.png"));
        if i % 2 == 0 {
            write_png(&p, 8, 8);
        }
        jobs.push(ProbeJob { path: p.to_str().unwrap().to_string(), dims: i % 4 == 0 });
    }
    let cache = ProbeCache::new();
    let wakes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = wakes.clone();
    cache.set_waker(move || {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    });
    let g0 = cache.generation();
    cache.request(jobs.clone());
    assert!(cache.wait_idle(std::time::Duration::from_secs(20)));
    assert_eq!(cache.len(), 600);
    assert!(cache.generation() > g0 && wakes.load(std::sync::atomic::Ordering::SeqCst) >= 2, "results arrive in batches");
    assert_eq!(cache.get(&jobs[0].path), Some(Probe { exists: true, dims_checked: true, tiny: true }));
    assert_eq!(cache.get(&jobs[2].path), Some(Probe { exists: true, dims_checked: false, tiny: false }));
    assert_eq!(cache.get(&jobs[1].path).map(|p| p.exists), Some(false));
    let g1 = cache.generation();
    cache.request(jobs.clone());
    assert!(cache.is_idle() && cache.generation() == g1, "nothing left to ask");
    let wants_dims = ProbeJob { path: jobs[2].path.clone(), dims: true };
    assert!(!cache.satisfies(&wants_dims));
    cache.request(vec![wants_dims.clone()]);
    assert!(cache.wait_idle(std::time::Duration::from_secs(5)) && cache.satisfies(&wants_dims));
    cache.clear();
    assert!(cache.is_empty() && cache.generation() > g1);
}

/* ------------------------------------------ non-destructive view index ------------------------------------------ */

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, list: &'a [T]) -> &'a T {
        &list[self.below(list.len())]
    }
}

fn random_rows(n: usize, seed: u64) -> Vec<Thumbnail> {
    let mut rng = Xorshift(seed);
    let dirs = ["/a/b", "/a/bc", "/a", "/ü/日本", "/p/tmp", "/p/sub/deep", "C:\\win\\dir", "zip:///z.zip!/in"];
    let names = unusual_names();
    let exts = ["jpg", "png", "gif", "mp4", "txt", "zip", "ico", ""];
    let sizes = [0u64, 1, 500, 10_239, 10_240, 10_241, 1 << 20, 5 << 20];
    let tags = ["sun", "sea", "Rain", "cat"];
    let cats = ["travel", "Food", "x"];
    let mut rows = Vec::new();
    for i in 0..n {
        let dir = *rng.pick(&dirs);
        let ext = *rng.pick(&exts);
        let base = rng.pick(&names).clone();
        let name = if ext.is_empty() { format!("{base}{i}") } else { format!("{base}{i}.{ext}") };
        let is_dir = rng.below(12) == 0;
        let mut r = row(&format!("{dir}/{name}"), if is_dir { DIR_TYPE } else { ext });
        r.filename = name;
        r.size = *rng.pick(&sizes);
        if rng.below(4) != 0 {
            r.modified = Some(local_time(2023 + rng.below(3) as i32, 1 + rng.below(12) as u32, 1 + rng.below(28) as u32, rng.below(24) as u32, 0, 0));
        }
        if rng.below(3) != 0 {
            r.db_created = Some(local_time(2024, 1 + rng.below(12) as u32, 1 + rng.below(28) as u32, 12, 0, 0));
        }
        for t in tags {
            if rng.below(3) == 0 {
                r.tags.push(t.to_string());
            }
        }
        if rng.below(2) == 0 {
            r.category = Some(rng.pick(&cats).to_string());
        }
        if rng.below(2) == 0 {
            r.description = Some("a description".into());
        }
        rows.push(r);
    }
    rows
}

fn paths_of(view: &ViewIndex, store: &RowStore<Thumbnail>) -> Vec<String> {
    view.rows(store).map(|r| r.path.clone()).collect()
}

fn missing_sort<'r>(_: &'r Thumbnail, _: u8) -> SortValue<'r> {
    SortValue::Missing
}

/// Refreshes `view` for `spec` and returns the listed paths.
fn listed(view: &mut ViewIndex, store: &RowStore<Thumbnail>, spec: &FilterSpec, facts: &Facts) -> Vec<String> {
    let filter = PreparedFilter::new(spec, facts);
    let inputs = ViewInputs { filter_sig: spec.signature(), data_sig: 0, filter: &|r| filter.matches(r), sort_key: &missing_sort };
    view.refresh(store, &inputs);
    view.check_invariants(store, &inputs).unwrap();
    paths_of(view, store)
}

fn facts_for(rows: &[Thumbnail]) -> Facts {
    let mut facts = Facts::default();
    for (i, r) in rows.iter().enumerate() {
        if i % 3 == 0 {
            facts.clip.insert(r.path.clone());
        }
    }
    facts.indexed = Some(rows.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, r)| r.path.clone()).collect());
    facts
}

fn condition_edits() -> Vec<(&'static str, Box<dyn Fn(&mut FilterSpec)>)> {
    vec![
        ("hide videos", Box::new(|s| s.types.videos = false)),
        ("hide folders", Box::new(|s| s.types.dirs = false)),
        ("hide other", Box::new(|s| s.types.other = false)),
        ("min size", Box::new(|s| s.size.min = Some(10_240))),
        ("max size", Box::new(|s| s.size.max = Some(1 << 20))),
        ("modified after", Box::new(|s| s.modified.after = Some("2024-01-01".into()))),
        ("modified before", Box::new(|s| s.modified.before = Some("2024-12-31".into()))),
        ("created after", Box::new(|s| s.created.after = Some("2024-06-01".into()))),
        ("exclude term", Box::new(|s| s.excluded_terms.push("tmp".into()))),
        ("exclude unicode term", Box::new(|s| s.excluded_terms.push("ü".into()))),
        ("exclude folder", Box::new(|s| s.excluded_dirs.push("/a/b".into()))),
        ("search", Box::new(|s| s.query = "1".into())),
        ("name", Box::new(|s| s.name = "a".into())),
        ("ext allow", Box::new(|s| s.ext_allow = vec!["jpg".into(), "png".into(), "mp4".into()])),
        ("ext deny", Box::new(|s| s.ext_deny = vec!["gif".into()])),
        ("indexed", Box::new(|s| s.indexed = Tri::Yes)),
        ("has description", Box::new(|s| s.description = Tri::Yes)),
        ("no clip", Box::new(|s| s.clip = Tri::No)),
        ("online", Box::new(|s| s.offline = Tri::No)),
        ("category", Box::new(|s| s.categories = vec!["Travel".into()])),
        ("tags", Box::new(|s| s.tags = vec!["sun".into(), "rain".into()])),
    ]
}

#[test]
fn each_condition_only_narrows_the_view_and_keeps_the_store_order() {
    let rows = random_rows(1500, 7);
    let facts = facts_for(&rows);
    let mut store = RowStore::with_policy(KeyPolicy::Exact);
    store.bulk_load(rows);
    let stored = store.len();
    let mut view = ViewIndex::new();
    let everything = listed(&mut view, &store, &FilterSpec::none(), &facts);
    assert_eq!(everything, store.iter().map(|r| r.path.clone()).collect::<Vec<_>>());
    for (name, edit) in condition_edits() {
        let mut spec = FilterSpec::none();
        edit(&mut spec);
        let spec = spec.normalized();
        let got = listed(&mut view, &store, &spec, &facts);
        let expected: Vec<String> = store.iter().filter(|r| PreparedFilter::new(&spec, &facts).matches(r)).map(|r| r.path.clone()).collect();
        assert_eq!(got, expected, "{name}");
        assert!(got.len() < everything.len() || name == "indexed" || name == "online", "{name} should hide something in this data set");
        let mut cursor = everything.iter();
        assert!(got.iter().all(|p| cursor.any(|e| e == p)), "{name} reordered the rows");
        assert_eq!(store.len(), stored, "{name} changed the store");
        let back = listed(&mut view, &store, &FilterSpec::none(), &facts);
        assert_eq!(back, everything, "{name} was not undone");
    }
}

#[test]
fn applying_every_condition_then_removing_each_chip_restores_the_original_rows_in_order() {
    for seed in [1u64, 2, 3, 99] {
        let rows = random_rows(1200, seed);
        let facts = facts_for(&rows);
        let mut store = RowStore::with_policy(KeyPolicy::Exact);
        store.bulk_load(rows);
        let mut view = ViewIndex::new();
        let everything = listed(&mut view, &store, &FilterSpec::none(), &facts);
        let mut spec = FilterSpec::none();
        for (_, edit) in condition_edits() {
            edit(&mut spec);
        }
        spec.tags_all = true;
        let mut spec = spec.normalized();
        let narrowed = listed(&mut view, &store, &spec, &facts);
        assert!(narrowed.len() < everything.len());
        let mut rng = Xorshift(seed * 31 + 5);
        let mut previous = narrowed.len();
        while !spec.chips().is_empty() {
            let chips = spec.chips();
            let chip = chips[rng.below(chips.len())].clone();
            spec.remove(&chip.kind);
            let got = listed(&mut view, &store, &spec, &facts);
            let expected: Vec<String> = store.iter().filter(|r| PreparedFilter::new(&spec, &facts).matches(r)).map(|r| r.path.clone()).collect();
            assert_eq!(got, expected, "after removing {:?}", chip.kind);
            assert!(got.len() >= previous, "removing {:?} lost rows", chip.kind);
            previous = got.len();
            assert_eq!(store.len(), 1200);
        }
        assert_eq!(listed(&mut view, &store, &spec, &facts), everything, "seed {seed}");
        assert!(spec.is_empty());
    }
}

#[test]
fn the_view_refreshes_only_when_the_signature_moves() {
    let rows = random_rows(300, 11);
    let mut store = RowStore::with_policy(KeyPolicy::Exact);
    store.bulk_load(rows);
    let mut view = ViewIndex::new();
    let facts = Facts::default();
    let spec = spec_with(|s| s.size.min = Some(10_240));
    let run = |view: &mut ViewIndex, spec: &FilterSpec| {
        let filter = PreparedFilter::new(spec, &facts);
        view.refresh(&store, &ViewInputs { filter_sig: spec.signature(), data_sig: 0, filter: &|r| filter.matches(r), sort_key: &missing_sort })
    };
    assert_eq!(run(&mut view, &spec), Refresh::Rebuilt);
    for _ in 0..20 {
        assert_eq!(run(&mut view, &spec.clone()), Refresh::Unchanged);
    }
    let mut next = spec.clone();
    next.skip_icons = true;
    assert_eq!(run(&mut view, &next), Refresh::Rebuilt);
    assert_eq!(run(&mut view, &spec), Refresh::Rebuilt);
    assert_eq!(view.stats().hidden_by_filter, store.len() - view.len());
}
