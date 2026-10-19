//! Palette tests: the registry, entry building, ranking and the window's keyboard handling.

use std::collections::BTreeSet;

use eframe::egui::{Context, Event, Key, Modifiers, Pos2, RawInput, Rect, vec2};

use super::entries::{self, DEVTOOLS, build, search_entry};
use super::view::{MAX_RESULTS, hits, move_selection};
use super::*;
use crate::command::{ActionId, Scope, Section, SmartView, Target, Tool};
use crate::ui::shell::ShellAction;
use crate::ui::sidebar::links::{self, CountEntry, PlacesModel};
use crate::ui::test_support::{test_ctx, texts};

fn ready() -> ActionContext {
    ActionContext { selected: 2, focus_file: true, db_ready: true, sidebar_open: true, bottom_bar_open: true, right_rail_open: true }
}

fn find<'a>(all: &'a [Entry], id: &str) -> &'a Entry {
    all.iter().find(|e| e.id == id).unwrap_or_else(|| panic!("no entry {id}: {:?}", all.iter().map(|e| &e.id).collect::<Vec<_>>()))
}

fn command_json(entry: &Entry) -> serde_json::Value {
    match &entry.effect {
        Effect::Command(cmd) => serde_json::to_value(cmd).unwrap(),
        other => panic!("{} is not a command: {other:?}", entry.id),
    }
}

/* ------------------------------------------- registry ---------------------------------------------- */

#[test]
fn registry_ids_and_titles_are_unique_and_filled() {
    let all = registry(&ready());
    let ids: BTreeSet<_> = all.iter().map(|e| e.id.clone()).collect();
    assert_eq!(ids.len(), all.len(), "duplicate ids");
    let titles: BTreeSet<_> = all.iter().map(|e| e.title.clone()).collect();
    assert_eq!(titles.len(), all.len(), "duplicate titles");
    assert!(all.iter().all(|e| !e.title.trim().is_empty() && !e.id.is_empty() && e.kind == Kind::Action));
}

#[test]
fn the_registry_reaches_every_command_a_user_can_send_without_arguments() {
    let all = registry(&ready());
    let names: BTreeSet<&str> = all
        .iter()
        .filter_map(|e| match &e.effect {
            Effect::Command(cmd) => Some(cmd.name()),
            Effect::Shell(_) => None,
        })
        .collect();
    for name in ["open_scope", "open_tool", "run", "show_settings", "clear_selection", "open_devtools", "show_task_center", "open_palette"] {
        assert!(names.contains(name), "no palette action sends {name}: {names:?}");
    }
    assert!(!names.contains("toast") && !names.contains("task_update"), "internal commands stay out of the palette");
}

#[test]
fn every_shell_action_has_a_registry_entry() {
    let all = registry(&ready());
    for action in ShellAction::ALL {
        let entry = find(&all, &format!("shell.{}", action.id()));
        assert_eq!(entry.title, action.label());
        assert!(matches!(entry.effect, Effect::Shell(a) if a == action));
    }
    let ids: BTreeSet<_> = ShellAction::ALL.iter().map(|a| a.id()).collect();
    assert_eq!(ids.len(), ShellAction::ALL.len(), "shell action ids are unique");
}

#[test]
fn every_file_action_runs_on_the_selection_and_needs_one() {
    let with = registry(&ready());
    let without = registry(&ActionContext { selected: 0, ..ready() });
    for action in ActionId::ALL {
        let id = format!("run.{}", serde_json::to_value(action).unwrap().as_str().unwrap());
        let json = command_json(find(&with, &id));
        assert_eq!(json["cmd"], "run");
        assert_eq!(json["on"], "selection");
        assert_eq!(find(&without, &id).enabled.clone().unwrap_err(), "No files selected");
        assert!(find(&with, &id).enabled.is_ok(), "{id}");
    }
    assert!(find(&without, "clear_selection").enabled.is_err());
    assert!(find(&with, "clear_selection").enabled.is_ok());
}

#[test]
fn library_entries_need_a_connected_library() {
    let offline = registry(&ActionContext { db_ready: false, ..ready() });
    for id in ["open_scope.library", "open_scope.recent", "open_scope.smart.no-clip", "shell.new_collection"] {
        assert_eq!(find(&offline, id).enabled, Err("Library not connected".to_string()), "{id}");
    }
    assert!(find(&offline, "shell.connect_library").enabled.is_ok());
    let online = registry(&ready());
    assert!(find(&online, "open_scope.library").enabled.is_ok());
    assert!(find(&online, "shell.connect_library").enabled.is_err(), "no connect button for a connected library");
}

#[test]
fn smart_views_open_in_the_current_tab() {
    let all = registry(&ready());
    for view in SmartView::ALL {
        let json = command_json(find(&all, &format!("open_scope.smart.{}", view.slug())));
        assert_eq!(json["scope"], format!("smart:{}", view.slug()));
        assert_eq!(json["target"], "current");
    }
}

#[test]
fn clip_for_the_previewed_file_needs_a_preview() {
    let none = registry(&ActionContext { focus_file: false, ..ready() });
    assert_eq!(find(&none, "shell.clip_selected").enabled, Err("No file is previewed".to_string()));
    assert!(find(&registry(&ready()), "shell.clip_selected").enabled.is_ok());
}

#[test]
fn tools_and_task_center_open_through_commands() {
    let all = registry(&ready());
    for tool in [Tool::ImageEdit, Tool::Refinements, Tool::Logs] {
        let id = format!("open_tool.{}", format!("{tool:?}").to_lowercase());
        let json = command_json(find(&all, &id));
        assert_eq!(json["cmd"], "open_tool");
        assert_eq!(json["tool"], serde_json::to_value(tool).unwrap());
    }
    assert_eq!(command_json(find(&all, "show_task_center"))["cmd"], "show_task_center");
    assert_eq!(find(&all, "open_palette").shortcut, Some("Ctrl+K"));
    assert_eq!(find(&all, "shell.toggle_sidebar").shortcut, Some("Ctrl+B"));
}

/// Names in backticks on the line of `docs/commands.md` that follows `heading`.
fn documented_names(heading: &str, marker: &str) -> Vec<String> {
    let docs = include_str!("../../../docs/commands.md");
    let from = docs.find(heading).unwrap_or_else(|| panic!("docs lack {heading}"));
    let line = docs[from..].lines().find(|l| l.contains(marker)).unwrap_or_else(|| panic!("docs lack {marker}"));
    line.split('`').skip(1).step_by(2).map(str::to_string).collect()
}

#[test]
fn every_documented_devtools_window_is_in_the_registry() {
    let all = registry(&ready());
    let names = documented_names("### open_devtools", "Opens a debug window");
    assert!(names.len() >= 14, "{names:?}");
    for name in names {
        find(&all, &format!("open_devtools.{name}"));
        let window: crate::command::DevtoolsWindow = serde_json::from_value(serde_json::Value::String(name.clone())).unwrap();
        assert!(DEVTOOLS.contains(&window), "{name}");
    }
}

#[test]
fn every_documented_settings_section_is_in_the_registry() {
    let all = registry(&ready());
    let names: BTreeSet<String> = documented_names("### show_settings", "`section` is optional")
        .into_iter()
        .filter(|n| serde_json::from_value::<Section>(serde_json::Value::String(n.clone())).is_ok())
        .collect();
    assert_eq!(names.len(), 9, "{names:?}");
    for name in names {
        let section: Section = serde_json::from_value(serde_json::Value::String(name.clone())).unwrap();
        let entry = find(&all, &format!("show_settings.{name}"));
        assert_eq!(entry.title, format!("Settings: {}", entries::section_label(section)));
        assert_eq!(command_json(entry)["section"], name);
    }
}

#[test]
fn devtools_labels_split_the_variant_name() {
    assert_eq!(entries::devtools_label(crate::command::DevtoolsWindow::CommandLog), "Command log");
    assert_eq!(entries::devtools_label(crate::command::DevtoolsWindow::Sidebar), "Sidebar");
    assert_eq!(entries::devtools_label(crate::command::DevtoolsWindow::DbMaintenance), "Db maintenance");
}

/* ---------------------------------------------- entries -------------------------------------------- */

fn sample_links() -> (PlacesModel, Vec<links::Link>, Vec<links::Link>, Vec<links::Link>, Vec<links::Link>) {
    let quick = vec![crate::utilities::types::QuickAccess { label: "Pictures".into(), path: "/home/u/Pictures".into(), icon: String::new() }];
    let places = links::places_links(&quick, &[], &[], &["/data/photos".to_string()]);
    let tags = vec![CountEntry { name: "sunset".into(), count: 9 }, CountEntry { name: "red car".into(), count: 4 }];
    let tag_links = tags.iter().map(links::tag_link).collect();
    let cat_links = vec![links::category_link(&CountEntry { name: "landscape".into(), count: 3 })];
    let collections = links::collection_links(&[links::CollectionEntry { id: surrealdb::types::RecordId::new("collections", "holidays"), name: "Holidays".into(), count: 12 }]);
    let scans = links::scan_links(&[links::ScanEntry { scan_id: 42, title: "Pictures".into(), root: "/home/u/Pictures".into(), started: "2026-01-02".into(), total: None }]);
    (places, collections, tag_links, cat_links, scans)
}

#[test]
fn places_collections_tags_categories_and_scans_become_entries_that_open_in_the_current_tab() {
    let (places, collections, tags, categories, scans) = sample_links();
    let all = build(&ready(), &Sources { places: &places, collections: &collections, tags: &tags, categories: &categories, scans: &scans });
    let expect = [
        ("link.place:/home/u/Pictures", Kind::Place, "fs:/home/u/Pictures"),
        ("link.recent:/data/photos", Kind::Recent, "fs:/data/photos"),
        ("link.col:Holidays", Kind::Collection, "col:Holidays"),
        ("link.tag:sunset", Kind::Tag, "tag:sunset"),
        ("link.tag:red car", Kind::Tag, "tag:red car"),
        ("link.cat:landscape", Kind::Category, "cat:landscape"),
        ("link.scan:42", Kind::Scan, "scan:42"),
    ];
    for (id, kind, scope) in expect {
        let entry = find(&all, id);
        assert_eq!(entry.kind, kind, "{id}");
        assert_eq!(entry.detail, scope, "{id}");
        let json = command_json(entry);
        assert_eq!((json["cmd"].as_str(), json["scope"].as_str(), json["target"].as_str()), (Some("open_scope"), Some(scope), Some("current")), "{id}");
        assert!(entry.enabled.is_ok());
    }
}

#[test]
fn library_backed_entries_are_disabled_without_a_library_but_places_are_not() {
    let (places, collections, tags, categories, scans) = sample_links();
    let cx = ActionContext { db_ready: false, ..ready() };
    let all = build(&cx, &Sources { places: &places, collections: &collections, tags: &tags, categories: &categories, scans: &scans });
    assert!(find(&all, "link.place:/home/u/Pictures").enabled.is_ok());
    assert!(find(&all, "link.recent:/data/photos").enabled.is_ok());
    for id in ["link.col:Holidays", "link.tag:sunset", "link.cat:landscape", "link.scan:42"] {
        assert_eq!(find(&all, id).enabled, Err("Library not connected".to_string()), "{id}");
    }
}

#[test]
fn only_the_first_few_hundred_tags_are_offered() {
    let many: Vec<links::Link> = (0..800).map(|i| links::tag_link(&CountEntry { name: format!("t{i}"), count: 1 })).collect();
    let places = PlacesModel::default();
    let all = build(&ready(), &Sources { places: &places, collections: &[], tags: &many, categories: &[], scans: &[] });
    assert_eq!(all.iter().filter(|e| e.kind == Kind::Tag).count(), 300);
}

#[test]
fn the_search_entry_runs_a_semantic_search_in_the_current_tab() {
    assert!(search_entry("", true).is_none() && search_entry("   ", true).is_none());
    let entry = search_entry("  red car ", true).unwrap();
    assert_eq!(entry.title, "Search library for \"red car\"");
    assert_eq!(entry.kind, Kind::Search);
    let json = command_json(&entry);
    assert_eq!(json, serde_json::json!({"cmd": "open_scope", "scope": "q:sem:\"red car\"", "target": "current"}));
    assert_eq!(search_entry("x", false).unwrap().enabled, Err("Library not connected".to_string()));
}

#[test]
fn a_search_with_quotes_survives_the_scope_string() {
    let entry = search_entry("a \"quoted\" word", true).unwrap();
    let Effect::Command(AppCommand::OpenScope { scope, .. }) = &entry.effect else { panic!("{entry:?}") };
    assert_eq!(scope, &Scope::Semantic("a \"quoted\" word".into()));
    assert_eq!(Scope::parse(&scope.format()).as_ref(), Ok(scope));
}

#[test]
fn a_new_tab_effect_changes_only_scopes_and_tools() {
    let scope = Effect::Command(AppCommand::OpenScope { scope: Scope::Library, target: Target::Current, title: Some("T".into()), recursive: true });
    assert!(matches!(scope.in_new_tab(), Effect::Command(AppCommand::OpenScope { target: Target::NewTab, recursive: true, title: Some(ref t), .. }) if t == "T"));
    let tool = Effect::Command(AppCommand::OpenTool { tool: Tool::Logs, path: None, target: Target::Current });
    assert!(matches!(tool.in_new_tab(), Effect::Command(AppCommand::OpenTool { target: Target::NewTab, .. })));
    let other = Effect::Command(AppCommand::ClearSelection);
    assert!(matches!(other.in_new_tab(), Effect::Command(AppCommand::ClearSelection)));
    assert!(matches!(Effect::Shell(ShellAction::Quit).in_new_tab(), Effect::Shell(ShellAction::Quit)));
}

/* ---------------------------------------------- ranking -------------------------------------------- */

#[test]
fn an_empty_query_lists_the_registry_in_order_with_no_search_entry() {
    let all = registry(&ready());
    let found = hits("", &all, None);
    assert_eq!(found.len(), MAX_RESULTS.min(all.len()));
    assert_eq!(found[0].entry.id, all[0].id);
}

#[test]
fn a_query_ranks_matching_entries_and_ends_with_the_search_entry() {
    let all = registry(&ready());
    let search = search_entry("clip", true);
    let found = hits("clip", &all, search.as_ref());
    assert!(found.len() > 2);
    assert_eq!(found.last().unwrap().entry.id, "search.library");
    assert!(found[..found.len() - 1].iter().all(|h| h.entry.title.to_lowercase().contains("clip") || h.entry.keywords.contains("clip") || h.entry.id.contains("clip")), "{:?}", found.iter().map(|h| &h.entry.title).collect::<Vec<_>>());
    let top = found[0].entry.title.clone();
    assert!(top.contains("CLIP"), "{top}");
    assert!(!found[0].positions.is_empty());
}

#[test]
fn a_query_that_matches_nothing_still_offers_the_search() {
    let all = registry(&ready());
    let search = search_entry("zzzzqqqq", true);
    let found = hits("zzzzqqqq", &all, search.as_ref());
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].entry.kind, Kind::Search);
}

#[test]
fn settings_and_devtools_entries_are_found_by_name() {
    let all = registry(&ready());
    let first = |q: &str| hits(q, &all, None).first().map(|h| h.entry.id.clone());
    assert_eq!(first("settings appearance"), Some("show_settings.appearance".to_string()));
    assert_eq!(first("devtools command log"), Some("open_devtools.command_log".to_string()));
    assert_eq!(first("sidebar"), Some("shell.toggle_sidebar".to_string()));
    assert_eq!(first("quit"), Some("shell.quit".to_string()));
}

#[test]
fn the_selection_wraps_and_pages_inside_the_list() {
    assert_eq!(move_selection(0, 0, false, true, false, false), 0);
    assert_eq!(move_selection(0, 5, false, true, false, false), 1);
    assert_eq!(move_selection(4, 5, false, true, false, false), 0, "down from the last row wraps");
    assert_eq!(move_selection(0, 5, true, false, false, false), 4, "up from the first row wraps");
    assert_eq!(move_selection(1, 30, false, false, false, true), 9);
    assert_eq!(move_selection(25, 30, false, false, false, true), 29);
    assert_eq!(move_selection(3, 30, false, false, true, false), 0);
    assert_eq!(move_selection(99, 5, false, false, false, false), 4, "a stale index is pulled back");
}

/* ------------------------------------------ headless window ---------------------------------------- */

fn key(key: Key, modifiers: Modifiers) -> Event {
    Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers }
}

struct Rig {
    ctx: Context,
    state: PaletteState,
    entries: Vec<Entry>,
    db_ready: bool,
}

impl Rig {
    fn new() -> Self {
        Self { ctx: test_ctx(), state: PaletteState::default(), entries: registry(&ready()), db_ready: true }
    }

    fn frame(&mut self, events: Vec<Event>) -> (Option<Effect>, Vec<(String, Pos2)>) {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 700.0))), events, ..Default::default() };
        let mut chosen = None;
        let (state, entries, db_ready) = (&mut self.state, &self.entries, self.db_ready);
        let out = self.ctx.run_ui(input, |ui| chosen = show(ui.ctx(), state, entries, db_ready));
        let drawn = texts(&out.shapes);
        out.drop_without_applying_deltas();
        (chosen, drawn)
    }

    fn settle(&mut self) -> Vec<(String, Pos2)> {
        (0..3).map(|_| self.frame(Vec::new()).1).last().unwrap()
    }
}

#[test]
fn a_closed_palette_draws_nothing_and_returns_nothing() {
    let mut rig = Rig::new();
    let (chosen, drawn) = rig.frame(Vec::new());
    assert!(chosen.is_none() && drawn.is_empty());
}

#[test]
fn the_open_palette_lists_matches_for_the_typed_query() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("tasks".into()));
    let drawn = rig.settle();
    let has = |t: &str| drawn.iter().any(|(s, _)| s == t);
    assert!(has("Show task center"), "{drawn:?}");
    assert!(has("Devtools: Tasks"), "{drawn:?}");
    assert!(!has("Quit"));
    assert!(has("Search library for \"tasks\""));
}

#[test]
fn arrows_move_and_enter_runs_the_selected_entry() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("devtools".into()));
    rig.settle();
    let first = hits("devtools", &rig.entries, None)[0].entry.id.clone();
    let second = hits("devtools", &rig.entries, None)[1].entry.id.clone();
    rig.frame(vec![key(Key::ArrowDown, Modifiers::NONE)]);
    assert_eq!(rig.state.selected, 1);
    let (chosen, _) = rig.frame(vec![key(Key::Enter, Modifiers::NONE)]);
    let effect = chosen.expect("Enter runs the entry");
    let expected = rig.entries.iter().find(|e| e.id == second).unwrap();
    assert_eq!(super::super::devtools::actions::effect_name(&effect), super::super::devtools::actions::effect_name(&expected.effect));
    assert_ne!(first, second);
    assert!(!rig.state.open, "running an entry closes the palette");
}

#[test]
fn up_from_the_first_row_wraps_to_the_last() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("devtools".into()));
    rig.settle();
    let count = hits("devtools", &rig.entries, search_entry("devtools", true).as_ref()).len();
    rig.frame(vec![key(Key::ArrowUp, Modifiers::NONE)]);
    assert_eq!(rig.state.selected, count - 1);
}

#[test]
fn ctrl_enter_opens_the_entry_in_a_new_tab() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("library all media".into()));
    rig.settle();
    let (chosen, _) = rig.frame(vec![key(Key::Enter, Modifiers::COMMAND)]);
    let Some(Effect::Command(AppCommand::OpenScope { scope, target, .. })) = chosen else { panic!("{chosen:?}") };
    assert_eq!((scope, target), (Scope::Library, Target::NewTab));
}

#[test]
fn escape_closes_without_running_anything() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("quit".into()));
    rig.settle();
    let (chosen, _) = rig.frame(vec![key(Key::Escape, Modifiers::NONE)]);
    assert!(chosen.is_none());
    assert!(!rig.state.open);
}

#[test]
fn enter_on_a_disabled_entry_explains_why_and_keeps_the_palette_open() {
    let mut rig = Rig::new();
    rig.entries = registry(&ActionContext { selected: 0, ..ready() });
    rig.state.open_with(Some("copy path selected".into()));
    rig.settle();
    let (chosen, _) = rig.frame(vec![key(Key::Enter, Modifiers::NONE)]);
    assert!(chosen.is_none());
    assert!(rig.state.open);
    assert_eq!(rig.state.notice.as_deref(), Some("Copy path: selected files: No files selected"));
    let drawn = rig.settle();
    assert!(drawn.iter().any(|(t, _)| t.contains("No files selected")), "{drawn:?}");
}

#[test]
fn enter_with_free_text_searches_the_library() {
    let mut rig = Rig::new();
    rig.state.open_with(Some("zzzzqqqq".into()));
    rig.settle();
    let (chosen, _) = rig.frame(vec![key(Key::Enter, Modifiers::NONE)]);
    let Some(Effect::Command(cmd)) = chosen else { panic!("{chosen:?}") };
    assert_eq!(serde_json::to_value(cmd).unwrap(), serde_json::json!({"cmd": "open_scope", "scope": "q:sem:\"zzzzqqqq\"", "target": "current"}));
}

#[test]
fn searching_without_a_library_is_refused_with_a_reason() {
    let mut rig = Rig::new();
    rig.db_ready = false;
    rig.state.open_with(Some("zzzzqqqq".into()));
    rig.settle();
    let (chosen, _) = rig.frame(vec![key(Key::Enter, Modifiers::NONE)]);
    assert!(chosen.is_none());
    assert!(rig.state.notice.as_deref().is_some_and(|n| n.ends_with("Library not connected")), "{:?}", rig.state.notice);
}

#[test]
fn typing_resets_the_selection_to_the_best_match() {
    let mut rig = Rig::new();
    rig.state.open_with(None);
    rig.settle();
    rig.frame(vec![key(Key::ArrowDown, Modifiers::NONE), key(Key::ArrowDown, Modifiers::NONE)]);
    assert!(rig.state.selected > 0);
    rig.frame(vec![Event::Text("q".into())]);
    assert_eq!(rig.state.query, "q");
    assert_eq!(rig.state.selected, 0);
}

#[test]
fn toggling_opens_and_closes() {
    let mut state = PaletteState::default();
    state.toggle();
    assert!(state.open);
    state.query = "x".into();
    state.toggle();
    assert!(!state.open);
    state.open_with(Some("again".into()));
    assert_eq!((state.open, state.query.as_str(), state.selected), (true, "again", 0));
}
