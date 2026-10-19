//! The tab's filter through the real explorer: legacy toggles, former destructive sites, lookups, saved views and Devtools commands.

use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Event, Key, Modifiers, Pos2, RawInput, Rect, vec2};

use super::*;
use crate::database::Thumbnail;
use crate::ui::devtools::filters as devtools;
use crate::ui::file_table::rows::{ChipKind, Condition, Tri};
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

fn file(path: &str, file_type: &str, size: u64) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap());
    t.path = path.into();
    t.file_type = file_type.into();
    t.size = size;
    t.parent_dir = path.rsplit_once('/').map_or(String::new(), |(d, _)| d.to_string());
    t
}

fn dir(path: &str) -> Thumbnail {
    file(path, DIR_TYPE, 0)
}

fn fixture() -> Vec<Thumbnail> {
    vec![
        file("/p/a.jpg", "jpg", 50_000),
        file("/p/b.png", "png", 5_000),
        file("/p/c.mp4", "mp4", 5_000_000),
        file("/p/sub/d.jpg", "jpg", 60_000),
        file("/p/tmp/e.gif", "gif", 70_000),
        file("/p/f.ico", "ico", 900),
        file("/p/g.txt", "txt", 12_000),
        file("/p/h.zip", "<ARCHIVE>", 20_000),
        dir("/p/sub"),
        dir("/p/tmp"),
    ]
}

fn explorer() -> FileExplorer {
    let mut ex = FileExplorer::new(true);
    ex.table.bulk_load(fixture());
    ex.sync_view();
    ex
}

fn listed(ex: &mut FileExplorer) -> Vec<String> {
    ex.sync_view();
    ex.view.rows(&ex.table).map(|r| r.path.clone()).collect()
}

fn all_paths() -> Vec<String> {
    fixture().into_iter().map(|r| r.path).collect()
}

fn without(skip: &[&str]) -> Vec<String> {
    all_paths().into_iter().filter(|p| !skip.contains(&p.as_str())).collect()
}

/* ------------------------------------------ the viewer's own fields ------------------------------------------ */

#[tokio::test]
async fn the_search_text_and_type_toggles_are_part_of_the_spec_both_ways() {
    let mut ex = explorer();
    ex.viewer.filter = "d.jpg".into();
    ex.viewer.types_show_videos = false;
    let spec = ex.filter_spec();
    assert_eq!(spec.query, "d.jpg");
    assert!(!spec.types.videos && spec.types.images);
    assert_eq!(listed(&mut ex), ["/p/sub/d.jpg"]);
    let mut next = FilterSpec::none();
    next.types.images = false;
    ex.set_filter_spec(next);
    assert!(ex.viewer.filter.is_empty() && !ex.viewer.types_show_images && ex.viewer.types_show_videos);
    assert_eq!(listed(&mut ex), without(&["/p/a.jpg", "/p/b.png", "/p/sub/d.jpg", "/p/tmp/e.gif"]), "an .ico file is not an image kind");
}

#[tokio::test]
async fn the_filter_signature_changes_with_every_input_and_only_the_inputs_in_use() {
    let mut ex = explorer();
    let base = ex.viewer.filter_signature();
    assert_eq!(ex.viewer.filter_signature(), base);
    let mut seen = vec![base];
    let mut step = |ex: &mut FileExplorer, seen: &mut Vec<u64>| seen.push(ex.viewer.filter_signature());
    ex.viewer.filter = "fox".into();
    step(&mut ex, &mut seen);
    ex.viewer.types_show_images = false;
    step(&mut ex, &mut seen);
    ex.viewer.types_show_videos = false;
    step(&mut ex, &mut seen);
    ex.viewer.types_show_dirs = false;
    step(&mut ex, &mut seen);
    ex.viewer.mode = ExplorerMode::Database;
    step(&mut ex, &mut seen);
    ex.edit_filters(|s| s.size.min = Some(5));
    step(&mut ex, &mut seen);
    ex.edit_filters(|s| s.excluded_terms.push("x".into()));
    step(&mut ex, &mut seen);
    let mut unique = seen.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), seen.len());

    let quiet = ex.viewer.filter_signature();
    ex.viewer.clip_presence.insert("/p/a.jpg".into());
    assert_eq!(ex.viewer.filter_signature(), quiet, "CLIP presence only matters while a CLIP condition is active");
    ex.edit_filters(|s| s.clip = Tri::Yes);
    let with_clip = ex.viewer.filter_signature();
    ex.viewer.clip_presence.insert("/p/b.png".into());
    assert_ne!(ex.viewer.filter_signature(), with_clip);
}

/* ------------------------------------ the former destructive filter sites ------------------------------------ */

#[tokio::test]
async fn no_filter_edit_touches_the_store_and_removing_a_filter_lists_its_rows_again() {
    let mut ex = explorer();
    let versions = ex.table.versions();
    let all = listed(&mut ex);
    assert_eq!(all, all_paths());
    let untouched = |ex: &FileExplorer| {
        assert_eq!(ex.table.versions(), versions, "the store was written");
        assert_eq!(ex.table.len(), 10);
    };

    // excluded term added and removed
    ex.edit_filters(|s| s.excluded_terms.push("tmp".into()));
    assert_eq!(listed(&mut ex), without(&["/p/tmp/e.gif"]));
    untouched(&ex);
    ex.edit_filters(|s| s.remove(&ChipKind::ExcludedTerm("tmp".into())));
    assert_eq!(listed(&mut ex), all);
    untouched(&ex);

    // size limits
    ex.edit_filters(|s| {
        s.size.min = Some(10_000);
        s.size.max = Some(1_000_000);
    });
    assert_eq!(listed(&mut ex), without(&["/p/b.png", "/p/c.mp4", "/p/f.ico"]));
    untouched(&ex);
    ex.edit_filters(|s| s.remove(&ChipKind::Size));
    assert_eq!(listed(&mut ex), all);

    // skip likely icons, with the probe result of the small image
    ex.viewer.filters.probes.insert("/p/b.png", Probe { exists: true, dims_checked: true, tiny: true });
    ex.edit_filters(|s| s.skip_icons = true);
    assert_eq!(listed(&mut ex), without(&["/p/b.png", "/p/f.ico"]));
    untouched(&ex);
    ex.edit_filters(|s| s.remove(&ChipKind::SkipIcons));
    assert_eq!(listed(&mut ex), all);

    // the whole spec replaced and reset (the old "apply filters to the table")
    ex.set_filter_spec(FilterSpec::default());
    assert_eq!(listed(&mut ex), without(&["/p/b.png", "/p/f.ico"]), "the 10 KiB default hides the two small files");
    untouched(&ex);
    ex.clear_filters();
    assert_eq!(listed(&mut ex), all);
    ex.set_filter_spec(FilterSpec::default());
    ex.reset_filters();
    assert_eq!(ex.filter_spec(), *ex.viewer.filters.base());

    // exclude the selected directories
    ex.clear_filters();
    ex.select_paths(["/p/sub", "/p/tmp"]);
    let dirs: Vec<String> = ex.selected_rows().iter().filter(|r| r.file_type == DIR_TYPE).map(|r| r.path.clone()).collect();
    assert_eq!(dirs, ["/p/sub", "/p/tmp"]);
    ex.edit_filters(|spec| spec.excluded_dirs.extend(dirs.iter().cloned()));
    assert_eq!(listed(&mut ex), without(&["/p/sub", "/p/tmp", "/p/sub/d.jpg", "/p/tmp/e.gif"]));
    untouched(&ex);
    assert!(ex.selected_rows().is_empty(), "rows that are no longer listed leave the selection");
    ex.edit_filters(|s| s.excluded_dirs.clear());
    assert_eq!(listed(&mut ex), all);

    // dates, extensions and a saved view
    ex.edit_filters(|s| s.modified.after = Some("2999-01-01".into()));
    assert_eq!(listed(&mut ex).len(), 2, "only the folders have no date to fail");
    ex.edit_filters(|s| s.modified = Default::default());
    ex.edit_filters(|s| s.ext_allow = vec!["jpg".into()]);
    assert_eq!(listed(&mut ex), ["/p/a.jpg", "/p/sub/d.jpg", "/p/sub", "/p/tmp"]);
    untouched(&ex);
    let mut spec = FilterSpec::none();
    spec.excluded_terms = vec!["sub".into()];
    ex.filter_groups = vec![group_from_spec("No sub", &spec)];
    ex.apply_saved_view("no SUB").unwrap();
    assert_eq!(ex.active_filter_group.as_deref(), Some("No sub"));
    assert_eq!(listed(&mut ex), without(&["/p/sub/d.jpg"]));
    untouched(&ex);
    ex.clear_filters();
    assert_eq!(listed(&mut ex), all, "everything comes back in the original order");
    assert_eq!(ex.active_filter_group, None);
    untouched(&ex);
}

#[tokio::test]
async fn the_excluded_terms_are_mirrored_for_the_scanners() {
    let mut ex = explorer();
    ex.edit_filters(|s| s.excluded_terms = vec!["  Cache ".into(), "tmp".into()]);
    assert_eq!(ex.excluded_terms, ["cache", "tmp"]);
    ex.clear_filters();
    assert!(ex.excluded_terms.is_empty());
}

/* -------------------------------------------- follow the settings -------------------------------------------- */

#[tokio::test]
async fn a_tab_has_no_settings_filter_until_the_settings_are_loaded() {
    let ex = explorer();
    assert!(ex.filter_spec().is_empty(), "the unloaded settings define no filter");
}

#[test]
fn only_the_groups_a_tab_left_alone_follow_the_settings() {
    let mut state = FilterState::with_base(FilterSpec::default());
    state.spec.size.min = Some(1);
    state.spec.query = "keep".into();
    let mut next = FilterSpec::default();
    next.size.min = None;
    next.skip_icons = true;
    next.ext_deny = vec!["tmp".into()];
    state.follow_base(next.clone());
    assert_eq!(state.spec.size.min, Some(1));
    assert!(state.spec.skip_icons && state.spec.ext_deny == ["tmp"]);
    assert_eq!(state.base(), &next);
    state.follow_base(next.clone());
    assert_eq!(state.base(), &next);
}

#[tokio::test]
async fn make_default_writes_the_settings_groups_and_keeps_the_tab_unchanged() {
    let mut ex = explorer();
    ex.edit_filters(|s| {
        s.size.min = Some(2048);
        s.query = "not a setting".into();
    });
    assert!(ex.filters_differ_from_defaults());
    let before = ex.filter_spec();
    let mut written = crate::UiSettings::default();
    before.write_settings(&mut written);
    assert_eq!(written.db_min_size_bytes, Some(2048));
    ex.viewer.filters.follow_base(FilterSpec::from_settings(&written));
    assert!(!ex.filters_differ_from_defaults());
    assert_eq!(ex.filter_spec(), before);
}

/* ------------------------------------------------- lookups --------------------------------------------------- */

#[tokio::test]
async fn clip_presence_by_path_or_hash_drives_the_clip_condition() {
    let mut ex = explorer();
    ex.viewer.clip_presence.insert("/p/a.jpg".into());
    ex.edit_filters(|s| s.clip = Tri::Yes);
    assert_eq!(listed(&mut ex), ["/p/a.jpg", "/p/sub", "/p/tmp"], "folders have no CLIP data to fail");
    let mut hashed = file("/p/hashed.jpg", "jpg", 1);
    hashed.hash = Some("h1".into());
    ex.table.push(hashed);
    ex.viewer.clip_presence_hashes.insert("h1".into());
    assert_eq!(listed(&mut ex), ["/p/a.jpg", "/p/sub", "/p/tmp", "/p/hashed.jpg"]);
    ex.edit_filters(|s| s.clip = Tri::No);
    assert_eq!(listed(&mut ex).len(), 9);
}

#[tokio::test]
async fn indexed_means_known_to_the_library_and_everything_in_library_mode() {
    let mut ex = explorer();
    ex.db_lookup.insert("/p/a.jpg".into(), file("/p/a.jpg", "jpg", 1));
    ex.db_lookup.insert("/p/g.txt".into(), file("/p/g.txt", "txt", 1));
    ex.edit_filters(|s| s.indexed = Tri::Yes);
    assert_eq!(listed(&mut ex), ["/p/a.jpg", "/p/g.txt", "/p/sub", "/p/tmp"]);
    ex.db_lookup.insert("/p/c.mp4".into(), file("/p/c.mp4", "mp4", 1));
    assert_eq!(listed(&mut ex), ["/p/a.jpg", "/p/c.mp4", "/p/g.txt", "/p/sub", "/p/tmp"], "a library lookup that grew is picked up");
    ex.edit_filters(|s| s.indexed = Tri::No);
    assert_eq!(listed(&mut ex).len(), 7);
    ex.viewer.mode = ExplorerMode::Database;
    ex.edit_filters(|s| s.indexed = Tri::Yes);
    assert_eq!(listed(&mut ex), all_paths());
}

fn write_png(path: &std::path::Path, w: u32, h: u32) -> u64 {
    image::RgbImage::from_pixel(w, h, image::Rgb([1, 2, 3])).save(path).unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn settle_probes(ex: &mut FileExplorer) {
    ex.sync_view();
    assert!(ex.viewer.filters.probes.wait_idle(std::time::Duration::from_secs(20)), "probes finished");
    ex.sync_view();
}

#[tokio::test]
async fn icon_sized_images_and_missing_files_are_found_by_background_probes() {
    let dir = tempfile::tempdir().unwrap();
    let icon = dir.path().join("icon.png");
    let photo = dir.path().join("photo.png");
    let icon_size = write_png(&icon, 16, 16);
    let photo_size = write_png(&photo, 200, 150);
    let p = |path: &std::path::Path| path.to_str().unwrap().to_string();
    let gone = dir.path().join("gone.png");
    let mut ex = FileExplorer::new(true);
    ex.table.bulk_load([file(&p(&icon), "png", icon_size), file(&p(&photo), "png", photo_size), file(&p(&gone), "png", 100)]);
    let all = listed(&mut ex);
    assert_eq!(all.len(), 3);

    ex.edit_filters(|s| s.skip_icons = true);
    settle_probes(&mut ex);
    assert_eq!(listed(&mut ex), [p(&photo), p(&gone)], "only the icon-sized image is hidden");
    assert!(ex.viewer.filters.probes.len() >= 3);

    ex.edit_filters(|s| {
        s.skip_icons = false;
        s.offline = Tri::Yes;
    });
    settle_probes(&mut ex);
    assert_eq!(listed(&mut ex), [p(&gone)]);
    ex.edit_filters(|s| s.offline = Tri::No);
    assert_eq!(listed(&mut ex), [p(&icon), p(&photo)]);
    assert_eq!(ex.table.len(), 3);

    let sig = ex.viewer.filter_signature();
    ex.viewer.filters.probes.insert(&p(&photo), Probe { exists: false, dims_checked: false, tiny: false });
    assert_ne!(ex.viewer.filter_signature(), sig, "new probe results refresh the view");
    assert_eq!(listed(&mut ex), [p(&icon)]);
}

#[tokio::test]
async fn probing_follows_rows_that_arrive_later_without_rescanning_old_ones() {
    let dir = tempfile::tempdir().unwrap();
    let mut ex = FileExplorer::new(true);
    ex.edit_filters(|s| s.offline = Tri::Yes);
    let mut paths = Vec::new();
    for i in 0..5 {
        let path = dir.path().join(format!("f{i}.png"));
        paths.push(path.to_str().unwrap().to_string());
        ex.table.push(file(&paths[i], "png", 100));
        settle_probes(&mut ex);
    }
    assert_eq!(listed(&mut ex).len(), 5, "none of them exist");
    assert_eq!(ex.viewer.filters.probes.len(), 5);
    std::fs::write(&paths[0], b"now here").unwrap();
    ex.viewer.filters.probes.clear();
    ex.viewer.filters.scan = None;
    settle_probes(&mut ex);
    assert_eq!(listed(&mut ex).len(), 4);
    ex.table.clear();
    ex.edit_filters(|s| s.offline = Tri::Any);
    settle_probes(&mut ex);
    assert!(ex.viewer.filters.probes.is_empty(), "an emptied tab drops its probe results");
}

/* ----------------------------------------------- commands and JSON -------------------------------------------- */

#[tokio::test]
async fn a_patch_changes_the_tab_and_a_bad_patch_changes_nothing() {
    let mut ex = explorer();
    let result = ex.apply_filter_patch(&serde_json::json!({"size": {"min": 10_000}, "types": {"videos": false}}), true).unwrap();
    assert_eq!(result, "2 active filters");
    assert_eq!(listed(&mut ex), without(&["/p/b.png", "/p/c.mp4", "/p/f.ico"]));
    assert!(!ex.viewer.types_show_videos);
    let before = ex.filter_spec();
    assert!(ex.apply_filter_patch(&serde_json::json!({"size": {"min": "huge"}}), false).is_err());
    assert!(ex.apply_filter_patch(&serde_json::json!({"colour": 1}), false).is_err());
    assert_eq!(ex.filter_spec(), before);
    ex.apply_filter_patch(&serde_json::json!({"types": null, "size": null}), false).unwrap();
    assert_eq!(listed(&mut ex), all_paths());
}

#[tokio::test]
async fn saved_views_apply_by_name_and_report_what_is_loaded() {
    let mut ex = explorer();
    assert!(ex.apply_saved_view("Holidays").unwrap_err().contains("none are loaded"));
    let mut spec = FilterSpec::none();
    spec.tags = vec!["sun".into()];
    ex.filter_groups = vec![group_from_spec("Sunny", &spec)];
    assert!(ex.apply_saved_view("Holidays").unwrap_err().contains("Sunny"));
    assert_eq!(ex.apply_saved_view(" sunny ").unwrap(), "applied view 'Sunny'");
    assert_eq!(ex.filter_spec().tags, ["sun"]);
    assert_eq!(ex.saved_views().len(), 1);
}

#[test]
fn a_saved_view_keeps_the_whole_spec_and_older_views_map_their_columns() {
    let mut spec = FilterSpec::none();
    spec.tags = vec!["a".into(), "b".into()];
    spec.tags_all = true;
    spec.modified.after = Some("2024-01-01".into());
    spec.excluded_terms = vec!["tmp".into()];
    spec.types.videos = false;
    spec.size.max = Some(99);
    let group = group_from_spec("Mine", &spec);
    assert_eq!(spec_from_group(&group), spec);
    assert!((group.include_images, group.include_videos, group.include_dirs) == (true, false, true));
    assert_eq!((group.min_size_bytes, group.max_size_bytes, group.excluded_terms.clone()), (None, Some(99), vec!["tmp".to_string()]));

    let mut old = group.clone();
    old.spec = None;
    let from_columns = spec_from_group(&old);
    assert_eq!(from_columns.tags, Vec::<String>::new(), "the older columns hold no tags");
    assert_eq!((from_columns.types.videos, from_columns.size.max, from_columns.excluded_terms), (false, Some(99), vec!["tmp".to_string()]));
    old.spec = Some("not json".into());
    assert_eq!(spec_from_group(&old).tags.len(), 0, "an unreadable spec falls back to the columns");
}

fn push_all(ctx: &Context, ex: &FileExplorer, commands: Vec<devtools::Command>) {
    for c in commands {
        devtools::push_command(ctx, ex.filter_debug_id(), c);
    }
}

#[tokio::test]
async fn the_devtools_window_can_apply_json_explain_a_row_and_reset() {
    let ctx = test_ctx();
    let mut ex = explorer();
    push_all(&ctx, &ex, vec![devtools::Command::ApplyJson(r#"{"size":{"min":10000},"excluded_terms":["tmp"]}"#.into())]);
    ex.run_filter_commands(&ctx);
    assert_eq!(listed(&mut ex), without(&["/p/b.png", "/p/f.ico", "/p/tmp/e.gif"]));
    assert!(ex.viewer.filters.error().is_none());

    push_all(&ctx, &ex, vec![devtools::Command::ApplyJson("{ nope".into())]);
    ex.run_filter_commands(&ctx);
    assert!(ex.viewer.filters.error().unwrap().starts_with("bad filter"));
    assert_eq!(ex.filter_spec().size.min, Some(10_000), "a bad document changes nothing");

    push_all(&ctx, &ex, vec![devtools::Command::Explain("/p/tmp/e.gif".into())]);
    ex.run_filter_commands(&ctx);
    let report = ex.viewer.filters.explain.clone().unwrap();
    assert!(report.found && !report.listed);
    assert_eq!(report.verdicts.iter().filter(|v| !v.passed).map(|v| v.condition).collect::<Vec<_>>(), [Condition::ExcludedTerms]);
    push_all(&ctx, &ex, vec![devtools::Command::Explain("/nowhere".into())]);
    ex.run_filter_commands(&ctx);
    assert!(!ex.viewer.filters.explain.clone().unwrap().found);

    push_all(&ctx, &ex, vec![devtools::Command::Clear]);
    ex.run_filter_commands(&ctx);
    assert!(ex.filter_spec().is_empty());
    ex.set_filter_spec(FilterSpec::default());
    push_all(&ctx, &ex, vec![devtools::Command::Reset]);
    ex.run_filter_commands(&ctx);
    assert!(ex.filter_spec().is_empty(), "reset goes to the settings, which are not loaded here");
}

#[tokio::test]
async fn the_window_snapshot_counts_rows_conditions_and_changes() {
    let ctx = test_ctx();
    let mut ex = explorer();
    devtools::set_open(&ctx, true);
    ex.edit_filters(|s| {
        s.size.min = Some(10_000);
        s.excluded_terms = vec!["tmp".into()];
    });
    ex.sync_view();
    ex.publish_filter_debug(&ctx);
    let snap = devtools::latest(&ctx).unwrap();
    assert_eq!((snap.rows_total, snap.rows_visible, snap.chips), (10, ex.view.len(), 2));
    assert_eq!(snap.signature, ex.viewer.filter_signature());
    assert!(snap.changes >= 1 && snap.since_change.is_some());
    assert_eq!(snap.report.total, 10);
    let size = snap.report.per_condition.iter().find(|c| c.condition == Condition::Size).unwrap();
    assert_eq!(size.failing, 2, "b.png and f.ico are under the minimum");
    assert!(snap.spec_json.contains("\"excluded_terms\""));
    assert_eq!(snap.indexed, "every row");
}

/* ------------------------------------------------ headless frames ------------------------------------------- */

struct Frames {
    ctx: Context,
    clock: f64,
    shapes: Vec<ClippedShape>,
}

impl Frames {
    fn new() -> Self {
        Self { ctx: test_ctx(), clock: 0.0, shapes: Vec::new() }
    }

    fn run(&mut self, ex: &mut FileExplorer, events: Vec<Event>, draw: impl Fn(&mut FileExplorer, &mut egui::Ui)) {
        self.clock += 0.1;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 900.0))), events, time: Some(self.clock), ..Default::default() };
        let out = self.ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| draw(ex, ui));
        });
        self.shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
    }

    fn drawn(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }
}

#[tokio::test]
async fn the_editor_toggles_a_type_through_a_real_click_without_touching_the_store() {
    let mut ex = explorer();
    let versions = ex.table.versions();
    let mut f = Frames::new();
    let draw = |ex: &mut FileExplorer, ui: &mut egui::Ui| ex.filter_editor(ui, "test");
    f.run(&mut ex, vec![], draw);
    f.run(&mut ex, vec![], draw);
    assert!(f.drawn("Videos") && f.drawn("Skip likely icons") && f.drawn("Make default"));
    let at = text_pos(&f.shapes, "Videos") + vec2(8.0, 6.0);
    f.run(&mut ex, press_release(at), draw);
    f.run(&mut ex, vec![], draw);
    assert!(!ex.viewer.types_show_videos);
    assert_eq!(listed(&mut ex), without(&["/p/c.mp4"]));
    assert_eq!(ex.table.versions(), versions);
    f.run(&mut ex, press_release(at), draw);
    f.run(&mut ex, vec![], draw);
    assert!(ex.viewer.types_show_videos);
    assert_eq!(listed(&mut ex), all_paths());
}

#[tokio::test]
async fn typing_an_excluded_term_and_pressing_enter_adds_it() {
    let mut ex = explorer();
    let mut f = Frames::new();
    let draw = |ex: &mut FileExplorer, ui: &mut egui::Ui| ex.filter_editor(ui, "test");
    f.run(&mut ex, vec![], draw);
    f.run(&mut ex, vec![], draw);
    let label = text_pos(&f.shapes, "Exclude");
    let field = label + vec2(90.0, 8.0);
    f.run(&mut ex, press_release(field), draw);
    f.run(&mut ex, vec![Event::Text("tmp".into())], draw);
    let enter = Event::Key { key: Key::Enter, physical_key: Some(Key::Enter), pressed: true, repeat: false, modifiers: Modifiers::NONE };
    f.run(&mut ex, vec![enter], draw);
    f.run(&mut ex, vec![], draw);
    assert_eq!(ex.filter_spec().excluded_terms, ["tmp"]);
    assert_eq!(listed(&mut ex), without(&["/p/tmp/e.gif"]));
    assert_eq!(ex.table.len(), 10);
}

/* ---------------------------------------------- saved views in kv-mem ---------------------------------------- */

#[tokio::test]
async fn a_saved_view_survives_the_database_round_trip() {
    use crate::database::{define_schema, delete_filter_group_on, list_filter_groups_on, rename_filter_group_on, save_filter_group_on};
    use surrealdb::Surreal;
    use surrealdb::engine::local::Mem;
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    define_schema(&db).await.unwrap();
    let mut spec = FilterSpec::none();
    spec.tags = vec!["sun".into()];
    spec.modified.after = Some("2024-05-05".into());
    spec.ext_allow = vec!["jpg".into()];
    spec.clip = Tri::Yes;
    save_filter_group_on(&db, group_from_spec("Sunny days", &spec)).await.unwrap();
    let mut other = FilterSpec::default();
    other.query = "日本語".into();
    save_filter_group_on(&db, group_from_spec("日本語", &other)).await.unwrap();
    let groups = list_filter_groups_on(&db).await.unwrap();
    assert_eq!(groups.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(), ["Sunny days", "日本語"]);
    assert_eq!(spec_from_group(&groups[0]), spec);
    assert_eq!(spec_from_group(&groups[1]), other.normalized());

    let mut updated = spec.clone();
    updated.tags.push("sea".into());
    save_filter_group_on(&db, group_from_spec("Sunny days", &updated)).await.unwrap();
    let groups = list_filter_groups_on(&db).await.unwrap();
    assert_eq!(groups.len(), 2, "saving under the same name replaces the view");
    assert_eq!(spec_from_group(&groups[0]).tags, ["sun", "sea"]);

    rename_filter_group_on(&db, "Sunny days", "Beach days").await.unwrap();
    let names: Vec<String> = list_filter_groups_on(&db).await.unwrap().into_iter().map(|g| g.name).collect();
    assert_eq!(names, ["Beach days", "日本語"]);
    assert!(rename_filter_group_on(&db, "Beach days", "日本語").await.unwrap_err().to_string().contains("already exists"));
    assert!(rename_filter_group_on(&db, "Nope", "Other").await.unwrap_err().to_string().contains("no saved view"));
    assert!(rename_filter_group_on(&db, "Beach days", "  ").await.is_err());
    rename_filter_group_on(&db, "Beach days", "BEACH DAYS").await.unwrap();
    assert_eq!(list_filter_groups_on(&db).await.unwrap()[0].name, "BEACH DAYS");

    delete_filter_group_on(&db, "beach days").await.unwrap();
    delete_filter_group_on(&db, "日本語").await.unwrap();
    assert!(list_filter_groups_on(&db).await.unwrap().is_empty());
}

#[tokio::test]
async fn views_saved_before_the_spec_column_still_load() {
    use crate::database::{define_schema, list_filter_groups_on};
    use surrealdb::Surreal;
    use surrealdb::engine::local::Mem;
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    define_schema(&db).await.unwrap();
    db.query("CREATE filter_groups:old SET name = 'Old view', include_images = true, include_videos = false, include_dirs = true, skip_icons = true, min_size_bytes = 2048, max_size_bytes = NONE, excluded_terms = ['tmp']")
        .await
        .unwrap()
        .check()
        .unwrap();
    let groups = list_filter_groups_on(&db).await.unwrap();
    assert_eq!(groups.len(), 1);
    assert!(groups[0].spec.is_none());
    let spec = spec_from_group(&groups[0]);
    assert_eq!((spec.types.videos, spec.skip_icons, spec.size.min, spec.excluded_terms.clone()), (false, true, Some(2048), vec!["tmp".to_string()]));
}
