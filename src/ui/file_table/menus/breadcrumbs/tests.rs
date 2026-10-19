use eframe::egui::epaint::ClippedShape;
use eframe::egui::*;

use super::*;
use crate::database::ui_themes::tests::RT;
use crate::ui::test_support::{press_release, test_ctx, text_pos};

#[test]
fn crumbs_accumulate_the_path_up_to_each_segment() {
    let sep = std::path::MAIN_SEPARATOR;
    let c = crumbs("/home/user/Pictures");
    assert_eq!(c.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(), ["home", "user", "Pictures"]);
    assert_eq!(c[0].target, "/home");
    assert_eq!(c[2].target, format!("/home{sep}user{sep}Pictures"));
    let c = crumbs(r"C:\Users\me");
    assert_eq!(c[0].target, "C:");
    assert_eq!(c[1].target, format!("C:{sep}Users"));
    assert!(crumbs("").is_empty());
    assert!(crumbs("/").is_empty());
}

#[test]
fn everything_shows_when_it_fits() {
    assert_eq!(first_visible(&[40.0, 50.0, 60.0], 20.0, 30.0, 230.0), 0);
    assert_eq!(first_visible(&[], 20.0, 30.0, 0.0), 0);
}

#[test]
fn the_leading_crumbs_collapse_first() {
    let w = [40.0, 50.0, 60.0, 70.0];
    let shown = |avail: f32| first_visible(&w, 20.0, 30.0, avail);
    assert_eq!(shown(279.0), 1);
    assert_eq!(shown(200.0), 2);
    assert_eq!(shown(100.0), 3);
}

#[test]
fn the_last_crumb_always_stays() {
    assert_eq!(first_visible(&[40.0, 50.0, 60.0], 20.0, 30.0, 10.0), 2);
    assert_eq!(first_visible(&[500.0], 20.0, 30.0, 10.0), 0);
}

#[test]
fn a_wider_row_never_hides_more() {
    let w = [33.0, 71.0, 44.0, 120.0, 60.0, 90.0];
    let mut last = usize::MAX;
    for avail in (0..900).step_by(7) {
        let first = first_visible(&w, 18.0, 26.0, avail as f32);
        assert!(first <= last, "avail {avail}");
        last = first;
    }
}

fn recents() -> Vec<String> {
    ["/home/u/Pictures", "/home/u/Pictures/2024", "/mnt/disk/Photos", "/home/u/Documents"].into_iter().map(String::from).collect()
}

fn no_children(_: &str) -> Vec<String> {
    Vec::new()
}

#[test]
fn nothing_typed_offers_the_recent_places_in_order() {
    assert_eq!(completions("", &recents(), &no_children, 3), ["/home/u/Pictures", "/home/u/Pictures/2024", "/mnt/disk/Photos"]);
    assert_eq!(completions("  ", &recents(), &no_children, 8).len(), 4);
}

#[test]
fn recent_places_match_by_prefix_before_matching_inside() {
    let got = completions("/home/u/pic", &recents(), &no_children, 8);
    assert_eq!(got, ["/home/u/Pictures", "/home/u/Pictures/2024"], "ignoring case");
    let got = completions("photos", &recents(), &no_children, 8);
    assert_eq!(got, ["/mnt/disk/Photos"], "a name inside a path matches too");
    assert!(completions("/nowhere", &recents(), &no_children, 8).is_empty());
}

#[test]
fn the_typed_text_is_never_offered_back() {
    assert!(completions("/home/u/Pictures", &["/home/u/Pictures".to_string()], &no_children, 8).is_empty());
}

#[test]
fn folders_under_the_typed_folder_come_first() {
    let children = |dir: &str| {
        assert_eq!(dir, "/home/u/");
        vec!["/home/u/Documents".to_string(), "/home/u/Downloads".to_string(), "/home/u/Pictures".to_string()]
    };
    let got = completions("/home/u/D", &recents(), &children, 8);
    assert_eq!(got, ["/home/u/Documents", "/home/u/Downloads"]);
    let got = completions("/home/u/", &recents(), &children, 8);
    assert_eq!(&got[..3], ["/home/u/Documents", "/home/u/Downloads", "/home/u/Pictures"]);
    assert!(got.contains(&"/home/u/Pictures/2024".to_string()), "recent places follow");
    assert_eq!(got.iter().filter(|p| *p == "/home/u/Pictures").count(), 1, "no repeats");
}

#[test]
fn scope_strings_are_offered_for_text_that_is_no_path() {
    assert_eq!(completions("smart:no", &[], &no_children, 8), ["smart:no-clip"]);
    assert_eq!(completions("lib", &[], &no_children, 8), ["lib:all"]);
    assert!(completions("/lib", &[], &no_children, 8).is_empty(), "a path is not completed with scopes");
}

#[test]
fn the_list_stops_at_the_limit() {
    let many: Vec<String> = (0..30).map(|i| format!("/p/{i:02}")).collect();
    assert_eq!(completions("/p", &many, &no_children, SUGGESTIONS).len(), SUGGESTIONS);
}

#[test]
fn path_like_text_is_told_from_scope_strings() {
    for yes in ["/a", "~/a", "C:\\x", "\\\\wsl\\x", "a/b", "relative\\dir"] {
        assert!(is_path_like(yes), "{yes}");
    }
    for no in ["lib:all", "tag:sunset", "fs:/p", "smart:no-clip", "plain"] {
        assert!(!is_path_like(no), "{no}");
    }
    assert_eq!(split_partial("/home/u/Pic"), Some(("/home/u/", "Pic")));
    assert_eq!(split_partial("/"), Some(("/", "")));
    assert_eq!(split_partial("plain"), None);
}

#[test]
fn disk_children_lists_only_folders_in_name_order() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["beta", "Alpha", "gamma"] {
        std::fs::create_dir(dir.path().join(name)).unwrap();
    }
    std::fs::write(dir.path().join("file.txt"), b"x").unwrap();
    let root = format!("{}{}", dir.path().display(), std::path::MAIN_SEPARATOR);
    let names: Vec<String> = disk_children(&root).iter().map(|p| p[root.len()..].to_string()).collect();
    assert_eq!(names, ["Alpha", "beta", "gamma"]);
    assert!(disk_children("/definitely/not/here/").is_empty());
}

/// A headless explorer drawing only the path area.
struct Row {
    ctx: Context,
    ex: FileExplorer,
    width: f32,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl Row {
    fn new(path: &str, width: f32) -> Self {
        let runtime = RT.enter();
        let mut ex = FileExplorer::new(true);
        ex.current_path = path.to_string();
        Self { ctx: test_ctx(), ex, width, _runtime: runtime }
    }

    fn frame(&mut self, events: Vec<Event>) -> Vec<ClippedShape> {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 400.0))), events, ..Default::default() };
        let ex = &mut self.ex;
        let out = self.ctx.run_ui(input, |ui| {
            Panel::top("path_row").exact_size(28.0).show_separator_line(false).show(ui, |ui| {
                ex.path_area(ui);
            });
        });
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        shapes
    }

    fn settle(&mut self) -> Vec<ClippedShape> {
        (0..3).map(|_| self.frame(Vec::new())).last().unwrap()
    }

    fn layout(&mut self) -> CrumbLayout {
        self.settle();
        last_layout(&self.ctx).expect("the row recorded its layout")
    }

    fn click(&mut self, pos: Pos2) -> Vec<ClippedShape> {
        self.frame(press_release(pos));
        self.settle()
    }

    fn type_text(&mut self, text: &str) {
        self.frame(vec![Event::Text(text.to_string())]);
        self.settle();
    }

    fn key(&mut self, key: Key) {
        let event = |pressed| Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers: Modifiers::NONE };
        self.frame(vec![event(true), event(false)]);
        self.settle();
    }
}

const DEEP: &str = "/aaaa/bbbbbbbb/cccc/dddddddd/eeee/ffffffff/gggg";

#[test]
fn a_wide_row_shows_every_crumb_in_order_inside_the_window() {
    let mut row = Row::new(DEEP, 1400.0);
    let l = row.layout();
    assert_eq!(l.hidden, 0);
    assert!(l.overflow.is_none());
    let targets: Vec<String> = l.shown.iter().map(|(_, t)| t.clone()).collect();
    assert_eq!(targets, crumbs(DEEP).into_iter().map(|c| c.target).collect::<Vec<_>>());
    for pair in l.shown.windows(2) {
        assert!(pair[0].0.right() <= pair[1].0.left() + 0.5, "{pair:?}");
    }
    assert!(l.shown.iter().all(|(r, _)| r.left() >= 0.0 && r.right() <= 1400.0));
}

#[test]
fn a_narrow_row_moves_leading_crumbs_into_the_overflow_and_keeps_the_last() {
    let last = crumbs(DEEP).pop().unwrap().target;
    let mut hidden_before = usize::MAX;
    for width in (160..=1400).step_by(20) {
        let mut row = Row::new(DEEP, width as f32);
        let l = row.layout();
        assert!(l.hidden <= hidden_before, "{width}px hides more than a narrower row");
        hidden_before = l.hidden;
        assert_eq!(l.shown.last().map(|(_, t)| t), Some(&last), "{width}px lost the last crumb");
        assert_eq!(l.overflow.is_some(), l.hidden > 0, "{width}px");
        assert_eq!(l.shown.len() + l.hidden, 7, "{width}px");
        let right_end = l.shown.iter().map(|(r, _)| r.right()).fold(0.0_f32, f32::max);
        if l.hidden + 1 < 7 {
            assert!(right_end <= width as f32 + 0.5, "{width}px: crumbs run to {right_end}");
        }
    }
    assert_eq!(hidden_before, 0);
    assert!(Row::new(DEEP, 260.0).layout().hidden > 0, "a 260px row cannot hold seven folders");
}

#[test]
fn clicking_a_crumb_opens_that_folder() {
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a").join("b").join("c");
    std::fs::create_dir_all(&deep).unwrap();
    let deep = deep.display().to_string();
    let mut row = Row::new(&deep, 1400.0);
    let l = row.layout();
    let b = crumbs(&deep).into_iter().find(|c| c.label == "b").unwrap().target;
    let (rect, _) = l.shown.iter().find(|(_, t)| *t == b).expect("the b crumb");
    row.click(rect.center());
    assert_eq!(row.ex.current_path, b);
    assert_eq!(row.ex.scope(), &crate::command::Scope::Folder(b.clone()));
    assert!(row.ex.can_go_back(), "the folder that was shown stays one step back");
    assert!(!row.ex.toolbar.path.editing, "a click on a crumb does not start editing");
}

#[test]
fn a_middle_click_on_a_crumb_asks_for_a_new_tab_and_leaves_the_tab_alone() {
    let _serial = crate::command::bus::testing::serial();
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a").join("b");
    std::fs::create_dir_all(&deep).unwrap();
    let deep = deep.display().to_string();
    let mut row = Row::new(&deep, 1400.0);
    let l = row.layout();
    let a = crumbs(&deep).into_iter().find(|c| c.label == "a").unwrap().target;
    let (rect, _) = l.shown.iter().find(|(_, t)| *t == a).expect("the a crumb");
    let pos = rect.center();
    let press = |pressed| Event::PointerButton { pos, button: PointerButton::Middle, pressed, modifiers: Modifiers::NONE };
    row.frame(vec![Event::PointerMoved(pos), press(true), press(false)]);
    row.settle();
    let asked = crate::command::bus::testing::take_matching(|c| matches!(c, crate::command::AppCommand::OpenScope { scope: crate::command::Scope::Folder(p), target: crate::command::Target::NewTab, .. } if *p == a));
    assert_eq!(asked.len(), 1);
    assert_eq!(row.ex.current_path, deep, "this tab stays where it was");
}

#[test]
fn the_overflow_menu_lists_hidden_folders_and_opens_them() {
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a").join("b").join("c").join("d").join("e");
    std::fs::create_dir_all(&deep).unwrap();
    let deep = deep.display().to_string();
    let mut row = Row::new(&deep, 300.0);
    let l = row.layout();
    assert!(l.hidden > 0);
    let first = crumbs(&deep).remove(0);
    let more = l.overflow.expect("the overflow button").center();
    let open = row.click(more);
    let item = text_pos(&open, &first.label);
    row.click(item + vec2(6.0, 6.0));
    assert_eq!(row.ex.current_path, first.target);
}

#[test]
fn an_empty_path_shows_a_hint_and_no_crumbs() {
    let mut row = Row::new("", 800.0);
    let shapes = row.settle();
    assert_eq!(text_pos(&shapes, "Type a path").y.is_finite(), true);
    assert!(last_layout(&row.ctx).unwrap().shown.is_empty());
}

#[test]
fn a_click_beside_the_crumbs_starts_editing_with_the_path_selected() {
    let mut row = Row::new("/a/b", 1000.0);
    let l = row.layout();
    let right = l.shown.last().unwrap().0.right();
    row.click(pos2(right + 120.0, l.row.center().y));
    assert!(row.ex.toolbar.path.editing);
    assert_eq!(row.ex.nav.input, "/a/b");
}

#[test]
fn typing_a_path_and_pressing_enter_opens_it_and_stops_editing() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().display().to_string();
    let mut row = Row::new("/a/b", 1000.0);
    row.ex.edit_address();
    row.settle();
    row.type_text(&target);
    assert_eq!(row.ex.nav.input, target, "typing replaced the selected text");
    row.key(Key::Enter);
    assert_eq!(row.ex.scope(), &crate::command::Scope::Folder(target.clone()));
    assert!(!row.ex.toolbar.path.editing);
}

#[test]
fn any_scope_string_can_be_typed_too() {
    let mut row = Row::new("/a/b", 1000.0);
    row.ex.edit_address();
    row.settle();
    row.type_text("tag:sunset");
    row.key(Key::Enter);
    assert_eq!(row.ex.scope(), &crate::command::Scope::Tag("sunset".into()));
}

#[test]
fn escape_cancels_editing_and_restores_the_address() {
    let mut row = Row::new("/a/b", 1000.0);
    row.ex.edit_address();
    row.settle();
    row.type_text("garbage");
    row.key(Key::Escape);
    assert!(!row.ex.toolbar.path.editing);
    assert_eq!(row.ex.nav.input, row.ex.scope().format());
    assert_eq!(row.ex.current_path, "/a/b");
}

#[test]
fn tab_completes_the_first_folder_and_the_arrows_choose_another() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["alpha", "alps"] {
        std::fs::create_dir(dir.path().join(name)).unwrap();
    }
    let root = format!("{}{}", dir.path().display(), std::path::MAIN_SEPARATOR);
    let mut row = Row::new("/a/b", 1000.0);
    row.ex.edit_address();
    row.settle();
    row.type_text(&format!("{root}al"));
    assert_eq!(row.ex.toolbar.path.suggestions.len(), 2, "{:?}", row.ex.toolbar.path.suggestions);
    row.key(Key::Tab);
    assert_eq!(row.ex.nav.input, format!("{root}alpha{}", std::path::MAIN_SEPARATOR));
    assert!(row.ex.toolbar.path.editing, "completing does not open anything");
    row.type_text("");
    row.ex.nav.input = format!("{root}al");
    row.settle();
    row.key(Key::ArrowDown);
    row.key(Key::ArrowDown);
    row.key(Key::Tab);
    assert_eq!(row.ex.nav.input, format!("{root}alps{}", std::path::MAIN_SEPARATOR));
}

#[test]
fn suggestions_are_only_computed_when_the_text_changes() {
    let mut row = Row::new("/a/b", 1000.0);
    row.ex.edit_address();
    row.settle();
    let before = row.ex.toolbar.path.computed;
    row.settle();
    row.settle();
    assert_eq!(row.ex.toolbar.path.computed, before, "idle frames do not read the disk");
    row.type_text("x");
    assert_eq!(row.ex.toolbar.path.computed, before + 1);
}
