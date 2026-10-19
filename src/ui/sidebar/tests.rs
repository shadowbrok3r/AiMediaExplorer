//! Sidebar tests: the saved layout, link to command mapping and headless frames.

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Context, Event, Key, Modifiers, Panel, PointerButton, Pos2, RawInput, Rect, vec2};
use surrealdb::types::RecordId;

use super::data::{PlacesRaw, PresetData};
use super::layout::{self, DEFAULT_WIDTH, FILE_NAME, LAYOUT_VERSION, LayoutError, LayoutStore, LoadNote, MAX_WIDTH, MIN_WIDTH, UiLayout};
use super::links::{self, CollectionEntry, CountEntry, Link, ScanEntry};
use super::*;
use crate::command::{AppCommand, Scope, SmartView, Target};
use crate::ui::test_support::{test_ctx, text_pos, texts};
use crate::utilities::explorer::DriveInfo;
use crate::utilities::types::QuickAccess;

/* ------------------------------------------ saved layout ------------------------------------------- */

fn file_in(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("smart-media").join(FILE_NAME)
}

#[test]
fn the_default_layout_survives_a_json_round_trip() {
    let mut layout = UiLayout::default();
    layout.sidebar_open = false;
    layout.sidebar_width = 301.0;
    layout.bottom_bar_open = false;
    layout.set_section_open("tags", true);
    layout.set_section_open("places", false);
    assert_eq!(UiLayout::parse(&layout.to_json()), Ok(layout));
    assert_eq!(UiLayout::parse(&UiLayout::default().to_json()), Ok(UiLayout::default()));
}

#[test]
fn missing_fields_take_their_defaults() {
    let layout = UiLayout::parse(r#"{"sidebar_open": false}"#).unwrap();
    assert!(!layout.sidebar_open);
    assert_eq!(layout.sidebar_width, DEFAULT_WIDTH);
    assert!(layout.bottom_bar_open && layout.right_rail_open);
    assert!(layout.section_open.is_empty());
    assert_eq!(UiLayout::parse("{}"), Ok(UiLayout::default()));
}

#[test]
fn widths_are_clamped_and_unknown_fields_are_ignored() {
    assert_eq!(UiLayout::parse(r#"{"sidebar_width": 9000}"#).unwrap().sidebar_width, MAX_WIDTH);
    assert_eq!(UiLayout::parse(r#"{"sidebar_width": -4}"#).unwrap().sidebar_width, MIN_WIDTH);
    assert_eq!(layout::clamp_width(f32::NAN), DEFAULT_WIDTH);
    assert_eq!(layout::clamp_width(f32::INFINITY), DEFAULT_WIDTH);
    assert!(UiLayout::parse(r#"{"future_field": [1, 2], "sidebar_open": true}"#).is_ok());
}

#[test]
fn a_section_that_was_never_set_uses_its_default() {
    let mut layout = UiLayout::default();
    for section in SectionId::ALL {
        assert_eq!(layout.is_section_open(section.key(), section.default_open()), section.default_open());
    }
    layout.set_section_open("tags", true);
    assert!(layout.is_section_open("tags", false));
    layout.set_section_open("places", false);
    assert!(!layout.is_section_open("places", true));
}

#[test]
fn only_the_cheap_sections_start_open() {
    let open: Vec<_> = SectionId::ALL.into_iter().filter(|s| s.default_open()).map(SectionId::key).collect();
    assert_eq!(open, vec!["places", "library", "collections"]);
}

#[test]
fn the_section_map_stays_small() {
    let many: std::collections::BTreeMap<String, bool> = (0..500).map(|i| (format!("s{i:03}"), true)).collect();
    let text = serde_json::json!({ "section_open": many }).to_string();
    assert_eq!(UiLayout::parse(&text).unwrap().section_open.len(), 64);
}

#[test]
fn text_that_is_not_a_layout_is_an_error_not_a_panic() {
    for bad in ["", "{", "null", "42", "{\"sidebar_open\": \"yes\"}", "{\"sidebar_width\": \"wide\"}", "\u{0}\u{1}\u{2}"] {
        assert!(matches!(UiLayout::parse(bad), Err(LayoutError::Invalid(_))), "{bad:?}");
    }
    assert_eq!(UiLayout::parse(r#"{"version": 99}"#), Err(LayoutError::TooNew(99)));
}

#[test]
fn a_missing_file_gives_the_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let store = LayoutStore::at(file_in(&dir));
    assert_eq!(store.note, LoadNote::Missing);
    assert_eq!(store.layout, UiLayout::default());
    assert!(!store.is_dirty());
}

#[test]
fn a_corrupt_file_gives_the_defaults_and_the_next_save_repairs_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_in(&dir);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "{ this is not json").unwrap();
    let mut store = LayoutStore::at(path.clone());
    assert!(matches!(store.note, LoadNote::Corrupt(_)), "{:?}", store.note);
    assert_eq!(store.layout, UiLayout::default());
    store.layout.sidebar_open = false;
    assert!(store.save_if_changed());
    let again = LayoutStore::at(path);
    assert_eq!(again.note, LoadNote::Loaded);
    assert!(!again.layout.sidebar_open);
}

#[test]
fn a_binary_or_empty_file_is_also_survivable() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_in(&dir);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    for bytes in [&b""[..], &[0xff, 0xfe, 0x00, 0x80][..]] {
        std::fs::write(&path, bytes).unwrap();
        let store = LayoutStore::at(path.clone());
        assert!(matches!(store.note, LoadNote::Corrupt(_)));
        assert_eq!(store.layout, UiLayout::default());
    }
}

#[test]
fn a_directory_in_place_of_the_file_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_in(&dir);
    std::fs::create_dir_all(&path).unwrap();
    let mut store = LayoutStore::at(path);
    assert!(matches!(store.note, LoadNote::Corrupt(_)));
    store.layout.sidebar_open = false;
    assert!(!store.save_if_changed());
    assert!(store.last_error.is_some());
    assert!(!store.is_dirty(), "a failed save is not retried every frame");
}

#[test]
fn a_newer_file_is_ignored_and_never_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_in(&dir);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let future = format!("{{\"version\": {}, \"sidebar_open\": false}}", LAYOUT_VERSION + 1);
    std::fs::write(&path, &future).unwrap();
    let mut store = LayoutStore::at(path.clone());
    assert_eq!(store.note, LoadNote::TooNew(LAYOUT_VERSION + 1));
    assert!(store.layout.sidebar_open);
    store.layout.bottom_bar_open = false;
    assert!(!store.save_if_changed());
    assert_eq!(std::fs::read_to_string(path).unwrap(), future);
}

#[test]
fn saving_writes_only_when_something_changed_and_leaves_no_temp_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_in(&dir);
    let mut store = LayoutStore::at(path.clone());
    assert!(!store.save_if_changed(), "unchanged layouts write nothing");
    assert!(!path.exists());
    store.layout.sidebar_width = 333.0;
    assert!(store.save_if_changed());
    assert!(!store.save_if_changed(), "the second call has nothing to write");
    assert_eq!(store.saves, 1);
    let names: Vec<_> = std::fs::read_dir(path.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    assert_eq!(names, vec![FILE_NAME.to_string()]);
    assert_eq!(LayoutStore::at(path).layout.sidebar_width, 333.0);
}

#[test]
fn a_memory_store_never_touches_the_disk() {
    let mut store = LayoutStore::memory();
    store.layout.sidebar_open = false;
    assert!(!store.save_if_changed());
    assert!(store.path().is_none());
    assert!(!store.is_dirty());
}

#[test]
fn reset_restores_the_defaults_and_the_next_save_writes_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = LayoutStore::at(file_in(&dir));
    store.layout.sidebar_width = 400.0;
    store.save_if_changed();
    store.reset();
    assert_eq!(store.layout, UiLayout::default());
    assert!(store.save_if_changed());
}

#[test]
fn the_default_path_is_inside_the_app_config_directory() {
    if let Some(path) = layout::default_path() {
        assert!(path.ends_with(format!("{}/{}", crate::ui::titlebar::APP_ID, FILE_NAME)), "{path:?}");
    }
}

/* -------------------------------------------- modes ------------------------------------------------ */

#[test]
fn the_sidebar_collapses_below_900_and_hides_when_closed() {
    assert_eq!(mode_for(1600.0, true), Mode::Full);
    assert_eq!(mode_for(900.0, true), Mode::Full);
    assert_eq!(mode_for(899.0, true), Mode::Rail);
    assert_eq!(mode_for(700.0, true), Mode::Rail);
    assert_eq!(mode_for(1600.0, false), Mode::Hidden);
    assert_eq!(mode_for(700.0, false), Mode::Hidden);
}

/* ---------------------------------------- links and commands --------------------------------------- */

fn quick(label: &str, path: &str) -> QuickAccess {
    QuickAccess { label: label.into(), path: PathBuf::from(path), icon: String::new() }
}

fn drive(root: &str, label: &str) -> DriveInfo {
    DriveInfo { root: root.into(), drive_type: "Disk".into(), label: label.into(), free: 40_000_000_000, total: 500_000_000_000 }
}

fn sample_places() -> PlacesRaw {
    PlacesRaw {
        quick: vec![quick("Pictures", "/home/u/Pictures"), quick("Videos", "/home/u/Videos"), quick("Home", "/home/u")],
        drives: vec![drive("/", "System"), drive("/mnt/games", "")],
        wsl: vec![quick("WSL:Ubuntu /mnt/wsl/disk", "\\\\wsl.localhost\\Ubuntu\\mnt\\wsl\\disk")],
        physical: Vec::new(),
    }
}

fn collection(name: &str, count: u64) -> CollectionEntry {
    CollectionEntry { id: RecordId::new("collections", name.to_lowercase().as_str()), name: name.into(), count }
}

fn counts(names: &[(&str, i64)]) -> Vec<CountEntry> {
    names.iter().map(|(name, count)| CountEntry { name: (*name).into(), count: *count }).collect()
}

fn fill(p: &mut PresetData) {
    p.total = Some(1234);
    p.places = sample_places();
    p.collections = vec![collection("Best Of: 2024", 3), collection("Holidays", 12)];
    p.tags = counts(&[("sunset", 9), ("red car", 4), ("dog", 1)]);
    p.categories = counts(&[("landscape", 20), ("portrait", 5)]);
    p.scans = vec![ScanEntry { scan_id: 42, title: "Pictures".into(), root: "/home/u/Pictures".into(), started: "2026-01-02 10:00".into(), total: Some(321) }];
}

fn all_links() -> Vec<Link> {
    let mut p = PresetData { total: None, collections: vec![], tags: vec![], categories: vec![], scans: vec![], places: PlacesRaw::default() };
    fill(&mut p);
    let recent = vec!["/data/photos".to_string(), "/home/u/Pictures".to_string(), "".to_string()];
    let places = links::places_links(&p.places.quick, &p.places.drives, &p.places.wsl, &recent);
    let mut all: Vec<Link> = places.quick.into_iter().chain(places.drives).chain(places.wsl).chain(places.recent).collect();
    all.extend(links::library_links(p.total, 7));
    all.extend(links::collection_links(&p.collections));
    all.extend(p.tags.iter().map(links::tag_link));
    all.extend(p.categories.iter().map(links::category_link));
    all.extend(links::scan_links(&p.scans));
    all
}

#[test]
fn every_link_emits_the_documented_scope_string() {
    let expected = [
        ("place:/home/u/Pictures", "fs:/home/u/Pictures"),
        ("place:/home/u/Videos", "fs:/home/u/Videos"),
        ("place:/home/u", "fs:/home/u"),
        ("drive:/", "fs:/"),
        ("drive:/mnt/games", "fs:/mnt/games"),
        ("recent:/data/photos", "fs:/data/photos"),
        ("lib:all", "lib:all"),
        ("lib:recent", "recent:all"),
        ("smart:no-clip", "smart:no-clip"),
        ("smart:undescribed", "smart:undescribed"),
        ("smart:untagged", "smart:untagged"),
        ("smart:failed-thumbs", "smart:failed-thumbs"),
        ("smart:duplicates", "smart:duplicates"),
        ("col:Best Of: 2024", "col:Best Of: 2024"),
        ("col:Holidays", "col:Holidays"),
        ("tag:sunset", "tag:sunset"),
        ("tag:red car", "tag:red car"),
        ("tag:dog", "tag:dog"),
        ("cat:landscape", "cat:landscape"),
        ("cat:portrait", "cat:portrait"),
        ("scan:42", "scan:42"),
        ("wsl:\\\\wsl.localhost\\Ubuntu\\mnt\\wsl\\disk", "fs:\\\\wsl.localhost\\Ubuntu\\mnt\\wsl\\disk"),
    ];
    let links = all_links();
    for (key, scope) in expected {
        let link = links.iter().find(|l| l.key == key).unwrap_or_else(|| panic!("no link {key}: {:?}", links.iter().map(|l| &l.key).collect::<Vec<_>>()));
        assert_eq!(link.scope.format(), scope, "{key}");
        let json = serde_json::to_value(link.command(Target::Current, false)).unwrap();
        assert_eq!(json["cmd"], "open_scope", "{key}");
        assert_eq!(json["scope"], scope, "{key}");
        assert_eq!(json["target"], "current", "{key}");
        assert_eq!(Scope::parse(scope).as_ref(), Ok(&link.scope), "{key}");
    }
    assert!(links.iter().all(|l| expected.iter().any(|(k, _)| *k == l.key)), "a link has no documented scope: {:?}", links.iter().map(|l| &l.key).filter(|k| !expected.iter().any(|(e, _)| e == k)).collect::<Vec<_>>());
    assert_eq!(links.len(), expected.len());
}

#[test]
fn link_keys_are_unique() {
    let links = all_links();
    let keys: std::collections::BTreeSet<_> = links.iter().map(|l| l.key.clone()).collect();
    assert_eq!(keys.len(), links.len());
}

#[test]
fn a_click_opens_here_and_ctrl_or_middle_click_opens_a_new_tab() {
    assert_eq!(links::click_target(false, false, false), (Target::Current, false));
    assert_eq!(links::click_target(true, false, false), (Target::NewTab, false));
    assert_eq!(links::click_target(false, true, false), (Target::NewTab, false));
    assert_eq!(links::click_target(false, false, true), (Target::Current, true));
}

#[test]
fn only_folders_scan_recursively() {
    let links = all_links();
    let folder = links.iter().find(|l| l.key == "place:/home/u").unwrap();
    let tag = links.iter().find(|l| l.key == "tag:dog").unwrap();
    assert!(matches!(folder.command(Target::Current, true), AppCommand::OpenScope { recursive: true, .. }));
    assert!(matches!(tag.command(Target::Current, true), AppCommand::OpenScope { recursive: false, .. }));
}

#[test]
fn recent_folders_skip_places_that_are_already_listed() {
    let raw = sample_places();
    let recent: Vec<String> = ["/home/u/Pictures/", "/data/a", "", "  "].into_iter().map(String::from).collect();
    let model = links::places_links(&raw.quick, &raw.drives, &raw.wsl, &recent);
    let keys: Vec<_> = model.recent.iter().map(|l| l.key.as_str()).collect();
    assert_eq!(keys, vec!["recent:/data/a"]);
    let many: Vec<String> = (0..30).map(|i| format!("/r/{i}")).collect();
    assert_eq!(links::places_links(&[], &[], &[], &many).recent.len(), links::MAX_RECENT);
}

#[test]
fn drives_show_the_free_space_and_fall_back_to_the_mount_point() {
    let model = links::places_links(&[], &[drive("/mnt/games", "")], &[], &[]);
    let link = &model.drives[0];
    assert_eq!(link.label, "/mnt/games");
    assert!(link.detail.ends_with("free"), "{}", link.detail);
    assert!(link.tooltip.contains("500 GB"), "{}", link.tooltip);
}

#[test]
fn library_counts_appear_where_they_are_cheap() {
    let links = links::library_links(Some(1234), 7);
    let detail = |key: &str| links.iter().find(|l| l.key == key).unwrap().detail.clone();
    assert_eq!(detail("lib:all"), "1234");
    assert_eq!(detail("smart:failed-thumbs"), "7");
    assert_eq!(detail("smart:no-clip"), "");
    assert_eq!(links::library_links(None, 0).iter().find(|l| l.key == "lib:all").unwrap().detail, "");
    assert_eq!(links.iter().filter(|l| matches!(l.scope, Scope::Smart(_))).count(), SmartView::ALL.len());
}

#[test]
fn tags_show_the_top_dozen_until_filtered_or_expanded() {
    let entries: Vec<CountEntry> = (0..150).map(|i| CountEntry { name: format!("tag{i:03}"), count: 1000 - i }).collect();
    let (top, hidden) = links::top_entries(&entries, "", false);
    assert_eq!((top.len(), hidden), (links::TOP_N, 150 - links::TOP_N as usize));
    assert_eq!(top[0].name, "tag000");
    let (all, hidden) = links::top_entries(&entries, "", true);
    assert_eq!((all.len(), hidden), (links::MAX_SHOWN, 50));
    let (found, hidden) = links::top_entries(&entries, " TAG14 ", false);
    assert_eq!(found.len(), 10);
    assert_eq!(hidden, 0);
    assert_eq!(links::top_entries(&entries, "zzz", false).0.len(), 0);
}

#[test]
fn counts_sort_by_count_then_name() {
    let mut entries = counts(&[("b", 2), ("a", 2), ("c", 9), ("B", 2)]);
    links::sort_counts(&mut entries);
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["c", "a", "B", "b"]);
}

#[test]
fn folder_links_match_the_focused_scope_ignoring_a_trailing_slash() {
    let links = all_links();
    let pictures = links.iter().find(|l| l.key == "place:/home/u/Pictures").unwrap();
    assert!(pictures.is_active(Some(&Scope::Folder("/home/u/Pictures/".into()))));
    assert!(!pictures.is_active(Some(&Scope::Folder("/home/u".into()))));
    assert!(!pictures.is_active(None));
    assert!(links.iter().find(|l| l.key == "lib:all").unwrap().is_active(Some(&Scope::Library)));
}

/* ------------------------------------------ headless frames ---------------------------------------- */

struct Frame {
    shapes: Vec<ClippedShape>,
    actions: Vec<SidebarAction>,
}

impl Frame {
    fn texts(&self) -> Vec<(String, Pos2)> {
        texts(&self.shapes)
    }

    fn has(&self, text: &str) -> bool {
        self.texts().iter().any(|(t, _)| t == text)
    }

    fn pos(&self, text: &str) -> Pos2 {
        text_pos(&self.shapes, text)
    }

    fn commands(&self) -> Vec<serde_json::Value> {
        self.actions.iter().filter_map(|a| if let SidebarAction::Command(c) = a { serde_json::to_value(c).ok() } else { None }).collect()
    }
}

struct Rig {
    ctx: Context,
    state: SidebarState,
    layout: UiLayout,
    size: (f32, f32),
    panel: f32,
    rail: bool,
    db_ready: bool,
    active: Option<Scope>,
    failed: usize,
    recent: Vec<String>,
}

impl Rig {
    fn new(width: f32) -> Self {
        let state = SidebarState::default();
        state.data.preset(fill);
        Self { ctx: test_ctx(), state, layout: UiLayout::default(), size: (width, 700.0), panel: DEFAULT_WIDTH, rail: false, db_ready: true, active: None, failed: 3, recent: vec!["/data/photos".into()] }
    }

    fn frame(&mut self, mut events: Vec<Event>, modifiers: Modifiers) -> Frame {
        events.insert(0, Event::ModifiersChanged(modifiers));
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.size.0, self.size.1))), events, ..Default::default() };
        let mut actions = Vec::new();
        let (state, layout, rail, panel) = (&mut self.state, &mut self.layout, self.rail, self.panel);
        let (db_ready, active, failed, recent) = (self.db_ready, self.active.clone(), self.failed, self.recent.clone());
        let out = self.ctx.run_ui(input, |ui| {
            let width = if rail { RAIL_WIDTH } else { panel };
            Panel::left("test_sidebar").exact_size(width).show(ui, |ui| {
                let input = SidebarInput { db_ready, active: active.as_ref(), failed_thumbs: failed, recent_paths: &recent, now: Instant::now() };
                actions = if rail { show_rail(ui, state, &input) } else { show_full(ui, state, layout, &input) };
            });
        });
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        Frame { shapes, actions }
    }

    fn settle(&mut self) -> Frame {
        (0..3).map(|_| self.frame(Vec::new(), Modifiers::NONE)).last().unwrap()
    }

    fn click(&mut self, at: Pos2, button: PointerButton, modifiers: Modifiers) -> Frame {
        let events = vec![
            Event::PointerMoved(at),
            Event::PointerButton { pos: at, button, pressed: true, modifiers },
            Event::PointerButton { pos: at, button, pressed: false, modifiers },
        ];
        let first = self.frame(events, modifiers);
        let rest = self.settle();
        Frame { shapes: rest.shapes, actions: first.actions.into_iter().chain(rest.actions).collect() }
    }

    fn click_text(&mut self, text: &str, button: PointerButton, modifiers: Modifiers) -> Frame {
        let at = self.settle().pos(text) + vec2(6.0, 6.0);
        self.click(at, button, modifiers)
    }
}

#[test]
fn the_full_sidebar_lists_every_section_with_its_links_and_counts() {
    let mut rig = Rig::new(1200.0);
    let f = rig.settle();
    for text in ["Places", "Library", "Collections", "Tags", "Categories", "Scans"] {
        assert!(f.has(text), "no {text} header: {:?}", f.texts());
    }
    for text in ["Pictures", "Videos", "Home", "Drives", "System", "Recent folders", "photos", "All media", "1234", "Recent", "No CLIP", "Failed thumbnails", "Duplicates", "Best Of: 2024", "Holidays", "12"] {
        assert!(f.has(text), "missing {text:?}: {:?}", f.texts());
    }
    assert!(!f.has("sunset"), "tags start collapsed");
    assert!(!f.has("landscape"), "categories start collapsed");
    assert!(!f.has("Library not connected"));
}

#[test]
fn a_click_on_a_link_opens_that_scope_in_the_current_tab() {
    let mut rig = Rig::new(1200.0);
    let f = rig.click_text("All media", PointerButton::Primary, Modifiers::NONE);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "lib:all", "target": "current"})]);
    let f = rig.click_text("Holidays", PointerButton::Primary, Modifiers::NONE);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "col:Holidays", "target": "current"})]);
}

#[test]
fn ctrl_click_and_middle_click_open_a_new_tab_and_shift_click_scans() {
    let mut rig = Rig::new(1200.0);
    let f = rig.click_text("Pictures", PointerButton::Primary, Modifiers::COMMAND);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "new_tab"})]);
    let f = rig.click_text("Pictures", PointerButton::Middle, Modifiers::NONE);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "new_tab"})]);
    let f = rig.click_text("Videos", PointerButton::Primary, Modifiers::SHIFT);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Videos", "target": "current", "recursive": true})]);
}

#[test]
fn clicking_a_header_collapses_its_section_and_remembers_it() {
    let mut rig = Rig::new(1200.0);
    assert!(rig.settle().has("Videos"));
    rig.click_text("Places", PointerButton::Primary, Modifiers::NONE);
    let f = rig.settle();
    assert!(!f.has("Videos"), "the places are hidden");
    assert!(f.has("Library"), "the other sections stay");
    assert!(!rig.layout.is_section_open("places", true));
    rig.click_text("Places", PointerButton::Primary, Modifiers::NONE);
    assert!(rig.settle().has("Videos"));
    assert!(rig.layout.is_section_open("places", true));
}

#[test]
fn opening_tags_and_categories_lists_them_with_counts() {
    let mut rig = Rig::new(1200.0);
    rig.size.1 = 2000.0;
    rig.click_text("Tags", PointerButton::Primary, Modifiers::NONE);
    rig.click_text("Categories", PointerButton::Primary, Modifiers::NONE);
    let f = rig.settle();
    for text in ["sunset", "red car", "dog", "landscape", "portrait", "20"] {
        assert!(f.has(text), "missing {text:?}: {:?}", f.texts());
    }
    let f = rig.click_text("sunset", PointerButton::Primary, Modifiers::NONE);
    assert_eq!(f.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "tag:sunset", "target": "current"})]);
}

#[test]
fn a_long_tag_list_is_cut_to_the_top_dozen_with_a_filter_box() {
    let mut rig = Rig::new(1200.0);
    rig.size.1 = 2000.0;
    rig.state.data.preset(|p| p.tags = (0..40).map(|i| CountEntry { name: format!("tag{i:02}"), count: 100 - i }).collect());
    rig.layout.set_section_open("tags", true);
    let f = rig.settle();
    assert!(f.has("tag00") && f.has("tag11"));
    assert!(!f.has("tag12"), "only the top dozen show");
    assert!(f.has("Show 28 more"), "{:?}", f.texts());
    rig.click_text("Show 28 more", PointerButton::Primary, Modifiers::NONE);
    assert!(rig.settle().has("tag39"));
    rig.state.tag_filter = "tag3".into();
    let f = rig.settle();
    assert!(f.has("tag30") && !f.has("tag00"));
}

#[test]
fn the_focused_scope_is_highlighted() {
    use eframe::egui::{Color32, Shape};
    fn fills(f: &Frame) -> std::collections::BTreeSet<[u8; 4]> {
        fn walk(shape: &Shape, out: &mut std::collections::BTreeSet<[u8; 4]>) {
            match shape {
                Shape::Rect(r) if r.fill != Color32::TRANSPARENT && r.fill.a() > 0 => {
                    out.insert(r.fill.to_array());
                }
                Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut out = std::collections::BTreeSet::new();
        f.shapes.iter().for_each(|s| walk(&s.shape, &mut out));
        out
    }
    let mut rig = Rig::new(1200.0);
    let plain = fills(&rig.settle());
    rig.active = Some(Scope::Folder("/home/u/Pictures".into()));
    let marked = fills(&rig.settle());
    assert!(!marked.is_subset(&plain), "the active row paints a fill the others do not: {plain:?} vs {marked:?}");
}

#[test]
fn without_a_database_the_library_sections_give_way_to_a_connect_button() {
    let mut rig = Rig::new(1200.0);
    rig.db_ready = false;
    let f = rig.settle();
    assert!(f.has("Places") && f.has("Pictures"), "places do not need the database");
    assert!(f.has("Library not connected") && f.has("Connect"));
    for text in ["Collections", "Tags", "Categories", "Scans", "All media", "Holidays"] {
        assert!(!f.has(text), "{text} should be hidden: {:?}", f.texts());
    }
    let f = rig.click_text("Connect", PointerButton::Primary, Modifiers::NONE);
    assert!(f.actions.iter().any(|a| matches!(a, SidebarAction::Connect)));
}

#[test]
fn the_disconnected_preview_overrides_a_ready_database() {
    let mut rig = Rig::new(1200.0);
    rig.state.force_disconnected = true;
    assert!(rig.settle().has("Library not connected"));
    rig.state.force_disconnected = false;
    assert!(!rig.settle().has("Library not connected"));
}

#[test]
fn collections_offer_rename_and_delete_in_their_context_menu() {
    let mut rig = Rig::new(1200.0);
    let at = rig.settle().pos("Holidays") + vec2(6.0, 6.0);
    rig.frame(vec![Event::PointerMoved(at), Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::NONE }, Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::NONE }], Modifiers::NONE);
    let f = rig.settle();
    for entry in ["Open here", "Open in new tab", "Rename…", "Delete…", "Copy scope"] {
        assert!(f.has(entry), "context menu lacks {entry}: {:?}", f.texts());
    }
    assert!(!f.has("Scan recursively here"), "only folders scan");
}

#[test]
fn choosing_delete_asks_for_confirmation_before_anything_is_removed() {
    let mut rig = Rig::new(1200.0);
    rig.state.confirm_delete = Some("Holidays".into());
    let f = rig.settle();
    assert!(f.has("Delete \"Holidays\"? Files are not touched."), "{:?}", f.texts());
    assert!(f.has("Delete") && f.has("Keep"));
    assert!(f.has("Best Of: 2024"));
    rig.click_text("Keep", PointerButton::Primary, Modifiers::NONE);
    assert!(rig.state.confirm_delete.is_none());
    assert!(rig.settle().has("Holidays"));
}

#[test]
fn a_rename_shows_a_field_and_escape_cancels_it() {
    let mut rig = Rig::new(1200.0);
    let id = rig.state.data.collections()[1].id.clone();
    rig.state.rename = Some(RenameState { id, input: "Holidays".into() });
    let f = rig.settle();
    assert!(!f.has("12"), "the renamed row is replaced by the field");
    let key = |key| vec![Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers: Modifiers::NONE }];
    rig.frame(key(Key::Escape), Modifiers::NONE);
    rig.settle();
    assert!(rig.state.rename.is_none());
}

#[test]
fn the_new_collection_field_appears_from_the_header_and_escape_closes_it() {
    let mut rig = Rig::new(1200.0);
    rig.state.new_collection = Some(String::new());
    rig.settle();
    assert!(rig.state.new_collection.is_some());
    let key = vec![Event::Key { key: Key::Escape, physical_key: Some(Key::Escape), pressed: true, repeat: false, modifiers: Modifiers::NONE }];
    rig.frame(key, Modifiers::NONE);
    rig.settle();
    assert!(rig.state.new_collection.is_none());
}

#[test]
fn the_rail_shows_icons_only_and_a_click_opens_the_flyout() {
    let mut rig = Rig::new(700.0);
    rig.rail = true;
    let f = rig.settle();
    for text in ["Places", "Library", "Collections", "Pictures"] {
        assert!(!f.has(text), "the rail draws no text, saw {text}");
    }
    let flyout = rig.click(Pos2::new(RAIL_WIDTH / 2.0, 22.0), PointerButton::Primary, Modifiers::NONE);
    assert!(flyout.has("Pictures") && flyout.has("Videos"), "the places flyout lists the folders: {:?}", flyout.texts());
    let picked = rig.click_text("Videos", PointerButton::Primary, Modifiers::NONE);
    assert_eq!(picked.commands(), vec![serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Videos", "target": "current"})]);
}

#[test]
fn the_rail_offers_a_connect_icon_when_the_library_is_not_connected() {
    let mut rig = Rig::new(700.0);
    rig.rail = true;
    rig.db_ready = false;
    rig.settle();
    let mut found = false;
    for step in 0..4 {
        let at = Pos2::new(RAIL_WIDTH / 2.0, 22.0 + 30.0 * step as f32);
        found |= rig.click(at, PointerButton::Primary, Modifiers::NONE).actions.iter().any(|a| matches!(a, SidebarAction::Connect));
        let away = Pos2::new(600.0, 600.0);
        rig.click(away, PointerButton::Primary, Modifiers::NONE);
    }
    assert!(found, "one rail icon asks to connect");
    rig.db_ready = true;
    for step in 0..4 {
        let at = Pos2::new(RAIL_WIDTH / 2.0, 22.0 + 30.0 * step as f32);
        assert!(!rig.click(at, PointerButton::Primary, Modifiers::NONE).actions.iter().any(|a| matches!(a, SidebarAction::Connect)), "a connected library shows no connect icon");
        rig.click(Pos2::new(600.0, 600.0), PointerButton::Primary, Modifiers::NONE);
    }
}

#[test]
fn every_section_keeps_its_rows_inside_the_panel_at_the_narrowest_width() {
    for width in [MIN_PANEL, DEFAULT_WIDTH, 420.0] {
        let mut rig = Rig::new(1200.0);
        rig.panel = width;
        rig.layout = UiLayout::default();
        for section in SectionId::ALL {
            rig.layout.set_section_open(section.key(), true);
        }
        let f = rig.settle();
        for (text, at) in f.texts() {
            assert!(at.x >= -0.5 && at.x < width, "{text:?} starts outside the {width}px panel at {at:?}");
        }
    }
}

const MIN_PANEL: f32 = layout::MIN_WIDTH;
