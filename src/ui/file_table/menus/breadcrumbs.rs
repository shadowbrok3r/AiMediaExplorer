//! Path breadcrumbs row under the explorer toolbar.

use eframe::egui::*;

use super::super::table::ExplorerMode;
use crate::icons::{self, svg};

const ROW_HEIGHT: f32 = 22.0;

/// One path segment: the label shown and the path a click opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crumb {
    pub label: String,
    pub target: String,
}

/// Where the crumbs of the last frame sit, for tests.
#[derive(Clone, Debug)]
pub struct CrumbLayout {
    /// Crumbs moved into the overflow menu.
    pub hidden: usize,
    pub overflow: Option<Rect>,
    pub shown: Vec<(Rect, String)>,
    pub row: Rect,
}

fn layout_id() -> Id {
    Id::new("ame.breadcrumbs.layout")
}

/// The layout recorded by the last [`FileExplorer::breadcrumbs`](super::super::FileExplorer::breadcrumbs).
pub fn last_layout(ctx: &Context) -> Option<CrumbLayout> {
    ctx.data(|d| d.get_temp::<CrumbLayout>(layout_id()))
}

/// The segments of `path`, each with the cumulative path up to it.
pub fn crumbs(path: &str) -> Vec<Crumb> {
    let mut accum = if path.starts_with('/') { String::from("/") } else { String::new() };
    path.split(['\\', '/'])
        .filter(|s| !s.is_empty())
        .map(|part| {
            if !accum.ends_with(std::path::MAIN_SEPARATOR) && !accum.is_empty() {
                accum.push(std::path::MAIN_SEPARATOR);
            }
            accum.push_str(part);
            Crumb { label: part.to_string(), target: accum.clone() }
        })
        .collect()
}

/// Index of the first crumb shown: the trailing crumbs that fit `avail`, at least the last; the rest go to the overflow menu.
pub fn first_visible(widths: &[f32], sep: f32, overflow_w: f32, avail: f32) -> usize {
    let n = widths.len();
    if n == 0 {
        return 0;
    }
    let tail = |from: usize| widths[from..].iter().sum::<f32>() + sep * (n - from - 1) as f32;
    if tail(0) <= avail {
        return 0;
    }
    (1..n).find(|&from| overflow_w + sep + tail(from) <= avail).unwrap_or(n - 1)
}

impl super::super::FileExplorer {
    /// Draws the clickable path under the toolbar; a click opens that folder.
    pub fn breadcrumbs(&mut self, ui: &mut Ui) {
        let crumbs = crumbs(&self.current_path);
        if crumbs.is_empty() {
            return;
        }
        let mut open = None;
        let mut recorded = CrumbLayout { hidden: 0, overflow: None, shown: Vec::new(), row: Rect::ZERO };
        Panel::top("file_explorer_breadcrumbs")
            .exact_size(ROW_HEIGHT)
            .show_separator_line(false)
            .show(ui, |ui| {
                recorded.row = ui.max_rect();
                ui.horizontal_centered(|ui| {
                    ui.add_space(ui.spacing().item_spacing.x);
                    open = crumb_row(ui, &crumbs, &mut recorded);
                });
            });
        ui.ctx().data_mut(|d| d.insert_temp(layout_id(), recorded));
        if let Some(target) = open {
            self.open_crumb(target);
        }
    }

    /// Opens `target` the way a breadcrumb click does.
    pub fn open_crumb(&mut self, target: String) {
        self.push_history(target);
        if self.viewer.mode == ExplorerMode::Database {
            self.db_offset = 0;
            self.db_last_batch_len = 0;
            self.load_database_rows();
        } else {
            self.populate_current_directory();
        }
    }
}

fn crumb_text(label: &str) -> RichText {
    RichText::new(label).underline()
}

fn crumb_width(ui: &Ui, label: &str) -> f32 {
    let galley = WidgetText::from(crumb_text(label)).into_galley(ui, Some(TextWrapMode::Extend), f32::INFINITY, TextStyle::Button);
    galley.size().x + ui.spacing().button_padding.x * 2.0
}

/// Lays the crumbs out left to right inside the remaining width and returns the one clicked.
fn crumb_row(ui: &mut Ui, crumbs: &[Crumb], recorded: &mut CrumbLayout) -> Option<String> {
    let item = ui.spacing().item_spacing.x;
    let caret = icons::IconSize::Sm.px(ui);
    let sep = caret + item * 2.0;
    let overflow_w = ui.spacing().interact_size.x.max(crumb_width(ui, "..."));
    let widths: Vec<f32> = crumbs.iter().map(|c| crumb_width(ui, &c.label)).collect();
    let first = first_visible(&widths, sep, overflow_w, ui.available_width());
    recorded.hidden = first;
    let mut open = None;
    if first > 0 {
        let (response, _) = containers::menu::MenuButton::new("...").ui(ui, |ui| {
            for crumb in &crumbs[..first] {
                if ui.button(&crumb.label).clicked() {
                    open = Some(crumb.target.clone());
                    ui.close();
                }
            }
        });
        recorded.overflow = Some(response.rect);
        svg::CARET_RIGHT.show_tinted(ui, caret, ui.visuals().weak_text_color());
    }
    for (i, crumb) in crumbs.iter().enumerate().skip(first) {
        let response = ui.selectable_label(false, crumb_text(&crumb.label));
        recorded.shown.push((response.rect, crumb.target.clone()));
        if response.clicked() {
            open = Some(crumb.target.clone());
        }
        if i + 1 < crumbs.len() {
            svg::CARET_RIGHT.show_tinted(ui, caret, ui.visuals().weak_text_color());
        }
    }
    open
}

#[cfg(test)]
mod tests {
    use super::*;

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

    use crate::database::ui_themes::tests::RT;
    use crate::ui::file_table::FileExplorer;
    use crate::ui::test_support::{press_release, test_ctx, text_pos};
    use eframe::egui::epaint::ClippedShape;

    /// A headless explorer drawing only the breadcrumb row.
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
            let input = RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 400.0))),
                events,
                ..Default::default()
            };
            let ex = &mut self.ex;
            let out = self.ctx.run_ui(input, |ui| ex.breadcrumbs(ui));
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
        assert_eq!(row.ex.back_stack.last().map(String::as_str), Some(deep.as_str()));
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
    fn an_empty_path_draws_no_row() {
        let mut row = Row::new("", 800.0);
        row.settle();
        assert!(last_layout(&row.ctx).is_none());
    }
}
