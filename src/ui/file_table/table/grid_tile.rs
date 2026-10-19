//! Painting of one grid tile: image or placeholder, selection and focus styling, hover checkbox, badges and caption.

use egui::epaint::text::TextWrapping;
use egui::load::SizedTexture;
use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, CornerRadius, FontId, Image, Rect, Stroke, StrokeKind, TextStyle, Ui, pos2, vec2};

use super::grid_facts::{DIR_COUNT_CAP, Facts};
use super::grid_layout::{cover_uv, duration_text, fit_rect};
use super::view_mode::TileFit;
use crate::Thumbnail;
use crate::icons::{self, Icon, svg};
use crate::ui::file_table::rows::RowKind;
use crate::ui::file_table::thumbs::ThumbView;

/// Edge of the selection checkbox.
pub const CHECK_R: f32 = 9.0;
/// Tiles narrower than this show badges as icons without counts.
const COMPACT_BELOW: f32 = 96.0;
const PILL_H: f32 = 18.0;
const PILL_PAD: f32 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Badge {
    /// No thumbnail could be made for a real file.
    Failed,
    /// The file is missing from the disk.
    Offline,
    /// Member of this many collections.
    Collections(u32),
    /// Not stored in the library.
    NotIndexed,
}

/// Badges of a tile, right to left.
pub fn badges(kind: RowKind, thumb: &ThumbView, facts: &Facts, indexed: bool, virtual_path: bool) -> [Option<Badge>; 4] {
    let media = matches!(kind, RowKind::Image | RowKind::Video);
    [
        (media && !virtual_path && matches!(thumb, ThumbView::Missing)).then_some(Badge::Failed),
        facts.offline.filter(|o| *o).map(|_| Badge::Offline),
        facts.collections.filter(|n| *n > 0).map(Badge::Collections),
        (!indexed && kind != RowKind::Dir).then_some(Badge::NotIndexed),
    ]
}

/// Icon standing in for a tile that has no image.
pub fn placeholder_icon(row: &Thumbnail, kind: RowKind) -> Icon {
    match kind {
        RowKind::Dir => svg::FOLDER,
        RowKind::Archive => svg::FILE_ZIP,
        RowKind::Video => svg::FILE_VIDEO,
        _ => icons::file_icon_svg(&row.filename, false),
    }
}

/// Text of a folder item count.
pub fn dir_count_text(count: u32, compact: bool) -> String {
    let n = if count > DIR_COUNT_CAP { format!("{DIR_COUNT_CAP}+") } else { count.to_string() };
    if compact { n } else if count == 1 { "1 item".to_string() } else { format!("{n} items") }
}

/// Colors of a frame's tiles, read from the theme once.
#[derive(Clone)]
pub struct TileColors {
    pub bg: Color32,
    pub border: Color32,
    pub select_fill: Color32,
    pub select_stroke: Color32,
    pub focus: Color32,
    pub text: Color32,
    pub strong: Color32,
    pub weak: Color32,
    /// Translucent backdrop of badges and checkboxes.
    pub scrim: Color32,
    /// Ink on the scrim.
    pub on_scrim: Color32,
    pub hover_wash: Color32,
    pub accent: Color32,
    pub warn: Color32,
    pub radius: CornerRadius,
    pub small: FontId,
    pub caption_h: f32,
}

impl TileColors {
    pub fn new(ui: &Ui) -> Self {
        let v = ui.visuals();
        let metrics = crate::theme::metrics(ui.ctx());
        let small = TextStyle::Small.resolve(ui.style());
        let caption_h = (ui.text_style_height(&TextStyle::Small) + 6.0).round();
        let sel = v.selection;
        Self {
            bg: v.extreme_bg_color,
            border: v.widgets.noninteractive.bg_stroke.color,
            select_fill: sel.bg_fill.gamma_multiply(0.55),
            select_stroke: sel.stroke.color,
            focus: v.widgets.hovered.fg_stroke.color,
            text: v.text_color(),
            strong: v.strong_text_color(),
            weak: v.weak_text_color(),
            scrim: v.extreme_bg_color.gamma_multiply(0.82),
            on_scrim: v.strong_text_color(),
            hover_wash: v.strong_text_color().gamma_multiply(0.07),
            accent: crate::theme::primary(ui),
            warn: v.warn_fg_color,
            radius: CornerRadius::same(metrics.radius_sm),
            small,
            caption_h,
        }
    }
}

/// What a tile is drawn from.
pub struct TileDraw<'a> {
    pub row: &'a Thumbnail,
    pub kind: RowKind,
    pub thumb: ThumbView,
    pub facts: Facts,
    pub selected: bool,
    pub focused: bool,
    pub hovered: bool,
    /// Checkboxes are shown on every tile because something is selected.
    pub multi: bool,
    pub indexed: bool,
    pub fit: TileFit,
    pub captions: bool,
    pub badges: bool,
    pub time: f64,
    /// Row and column, for the loading shimmer's phase.
    pub cell: (usize, usize),
}

/// Where the parts of a drawn tile ended up.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileParts {
    pub checkbox: Option<Rect>,
}

fn snap(size: f32) -> f32 {
    ((size / 8.0).round() * 8.0).max(8.0)
}

fn paint_icon(ui: &Ui, icon: Icon, rect: Rect, size: f32, tint: Color32) {
    icon.paint(ui, rect, snap(size), tint);
}

fn paint_texture(ui: &Ui, rect: Rect, radius: CornerRadius, id: egui::TextureId, size: egui::Vec2, uv: Rect) {
    Image::from_texture(SizedTexture::new(id, size)).uv(uv).corner_radius(radius).paint_at(ui, rect);
}

fn pill(ui: &Ui, colors: &TileColors, right: f32, y: f32, icon: Option<Icon>, text: Option<&str>, fg: Color32, bg: Color32) -> f32 {
    let galley = text.map(|t| ui.painter().layout_no_wrap(t.to_string(), colors.small.clone(), fg));
    let icon_w = if icon.is_some() { 12.0 } else { 0.0 };
    let gap = if icon.is_some() && galley.is_some() { 3.0 } else { 0.0 };
    let text_w = galley.as_ref().map_or(0.0, |g| g.size().x);
    let width = icon_w + gap + text_w + 2.0 * PILL_PAD;
    let rect = Rect::from_min_size(pos2(right - width, y), vec2(width, PILL_H));
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(PILL_H as u8 / 2), bg);
    let mut x = rect.left() + PILL_PAD;
    if let Some(icon) = icon {
        let slot = Rect::from_min_size(pos2(x, rect.center().y - 6.0), vec2(12.0, 12.0));
        icon.paint(ui, slot, 12.0, fg);
        x += icon_w + gap;
    }
    if let Some(galley) = galley {
        let at = pos2(x, rect.center().y - galley.size().y / 2.0);
        painter.galley(at, galley, fg);
    }
    rect.left() - 4.0
}

fn caption(ui: &Ui, colors: &TileColors, cell: Rect, image: Rect, name: &str, selected: bool) {
    let color = if selected { colors.strong } else { colors.text };
    let width = image.width();
    let mut job = LayoutJob::single_section(name.to_string(), TextFormat { font_id: colors.small.clone(), color, ..Default::default() });
    job.wrap = TextWrapping::truncate_at_width(width - 4.0);
    let galley = ui.painter().layout_job(job);
    let at = pos2(image.center().x - galley.size().x / 2.0, cell.top() + image.height() + (colors.caption_h - galley.size().y) / 2.0);
    ui.painter().galley(at, galley, color);
}

/// Draws one tile into `cell` (image area plus caption) and reports where its checkbox is.
pub fn paint(ui: &Ui, colors: &TileColors, cell: Rect, image: Rect, d: &TileDraw<'_>) -> TileParts {
    let painter = ui.painter();
    let radius = colors.radius;
    let compact = image.width() < COMPACT_BELOW;
    let shimmer = matches!(d.thumb, ThumbView::Pending) && d.kind != RowKind::Dir;
    let base = if shimmer {
        let phase = (d.cell.0 + d.cell.1) as f32 * 0.35;
        let wave = 0.5 + 0.5 * (d.time as f32 * 2.4 - phase).sin();
        colors.bg.lerp_to_gamma(colors.weak, 0.07 + 0.09 * wave)
    } else {
        colors.bg
    };
    painter.rect_filled(image, radius, base);
    match (d.kind, d.thumb) {
        (RowKind::Dir, _) => {
            let size = image.width() * 0.5;
            let center = image.center() - vec2(0.0, if d.facts.dir_items.is_some() { image.height() * 0.06 } else { 0.0 });
            paint_icon(ui, svg::FOLDER, Rect::from_center_size(center, vec2(size, size)), size, colors.accent);
            if let Some(count) = d.facts.dir_items {
                let text = dir_count_text(count, compact);
                let galley = painter.layout_no_wrap(text, colors.small.clone(), colors.weak);
                let at = pos2(image.center().x - galley.size().x / 2.0, image.bottom() - galley.size().y - 6.0);
                painter.galley(at, galley, colors.weak);
            }
        }
        (_, ThumbView::Ready { id, size }) => {
            if d.fit == TileFit::Crop {
                paint_texture(ui, image, radius, id, size, cover_uv(size));
            } else {
                let dest = fit_rect(size, image.shrink(2.0));
                paint_texture(ui, dest, CornerRadius::same(2), id, size, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)));
            }
        }
        _ => {
            let size = (image.width() * 0.4).clamp(16.0, 72.0);
            let tint = if shimmer { colors.weak.gamma_multiply(0.5) } else { colors.weak };
            paint_icon(ui, placeholder_icon(d.row, d.kind), Rect::from_center_size(image.center(), vec2(size, size)), size, tint);
        }
    }
    painter.rect_stroke(image, radius, Stroke::new(1.0, colors.border), StrokeKind::Inside);
    if d.hovered && !d.selected {
        painter.rect_filled(image, radius, colors.hover_wash);
    }
    if d.selected {
        painter.rect_filled(image, radius, colors.select_fill);
        painter.rect_stroke(image, radius, Stroke::new(2.0, colors.select_stroke), StrokeKind::Inside);
    }
    if d.focused {
        let ring = if d.selected { colors.strong } else { colors.focus };
        painter.rect_stroke(image.expand(1.5), CornerRadius::same(radius.nw.saturating_add(1)), Stroke::new(1.5, ring), StrokeKind::Outside);
    }
    let show_check = d.hovered || d.multi || d.selected;
    let mut parts = TileParts::default();
    if show_check {
        let center = image.left_top() + vec2(CHECK_R + 5.0, CHECK_R + 5.0);
        let rect = Rect::from_center_size(center, vec2(CHECK_R * 2.0, CHECK_R * 2.0));
        let (fill, edge) = if d.selected { (colors.select_stroke, colors.bg) } else { (colors.scrim, colors.on_scrim) };
        painter.circle(center, CHECK_R, fill, Stroke::new(1.5, edge));
        if d.selected {
            paint_icon(ui, svg::CHECK, rect.shrink(3.0), CHECK_R * 2.0 - 6.0, colors.bg);
        }
        parts.checkbox = Some(rect.expand(3.0));
    }
    if d.badges {
        let list = badges(d.kind, &d.thumb, &d.facts, d.indexed, d.row.path.contains("://"));
        let mut right = image.right() - 5.0;
        let y = image.top() + 5.0;
        for badge in list.into_iter().flatten() {
            right = match badge {
                Badge::Failed => pill(ui, colors, right, y, Some(svg::WARNING), None, colors.warn, colors.scrim),
                Badge::Offline => pill(ui, colors, right, y, Some(svg::CLOUD_SLASH), None, colors.on_scrim, colors.scrim),
                Badge::Collections(n) => {
                    let text = n.to_string();
                    pill(ui, colors, right, y, Some(svg::AME_LOGICAL_GROUP), (!compact).then_some(text.as_str()), colors.on_scrim, colors.scrim)
                }
                Badge::NotIndexed => {
                    let center = pos2(right - 4.0, y + PILL_H / 2.0);
                    painter.circle_stroke(center, 3.5, Stroke::new(1.5, colors.on_scrim.gamma_multiply(0.7)));
                    right - 12.0
                }
            };
        }
    }
    if d.kind == RowKind::Video {
        let y = image.bottom() - PILL_H - 5.0;
        let mut right = image.right() - 5.0;
        if let Some(secs) = d.facts.duration_secs {
            right = pill(ui, colors, right, y, None, Some(&duration_text(secs)), colors.on_scrim, colors.scrim);
        }
        let _ = right;
        let left = Rect::from_min_size(pos2(image.left() + 5.0, y), vec2(PILL_H + 4.0, PILL_H));
        painter.rect_filled(left, CornerRadius::same(PILL_H as u8 / 2), colors.scrim);
        icons::svg::PLAY.paint(ui, Rect::from_center_size(left.center(), vec2(11.0, 11.0)), 11.0, colors.on_scrim);
    }
    if d.captions && colors.caption_h > 0.0 {
        caption(ui, colors, cell, image, &d.row.filename, d.selected);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media_missing(kind: RowKind) -> [Option<Badge>; 4] {
        badges(kind, &ThumbView::Missing, &Facts::default(), true, false)
    }

    #[test]
    fn a_missing_thumbnail_of_a_real_media_file_is_a_failure() {
        assert_eq!(media_missing(RowKind::Image)[0], Some(Badge::Failed));
        assert_eq!(media_missing(RowKind::Video)[0], Some(Badge::Failed));
        assert_eq!(media_missing(RowKind::Other)[0], None, "other files never had a thumbnail to fail");
        assert_eq!(media_missing(RowKind::Dir)[0], None);
        assert_eq!(badges(RowKind::Image, &ThumbView::Missing, &Facts::default(), true, true)[0], None, "archive entries are made on demand");
        assert_eq!(badges(RowKind::Image, &ThumbView::Pending, &Facts::default(), true, false)[0], None);
    }

    #[test]
    fn facts_become_badges_in_a_fixed_order() {
        let facts = Facts { offline: Some(true), collections: Some(3), ..Facts::default() };
        let list = badges(RowKind::Image, &ThumbView::Missing, &facts, false, false);
        assert_eq!(list, [Some(Badge::Failed), Some(Badge::Offline), Some(Badge::Collections(3)), Some(Badge::NotIndexed)]);
        let none = badges(RowKind::Image, &ThumbView::Pending, &Facts { offline: Some(false), collections: Some(0), ..Facts::default() }, true, false);
        assert_eq!(none, [None; 4]);
    }

    #[test]
    fn folders_are_never_flagged_as_unindexed() {
        assert_eq!(badges(RowKind::Dir, &ThumbView::Pending, &Facts::default(), false, false)[3], None);
        assert_eq!(badges(RowKind::Video, &ThumbView::Pending, &Facts::default(), false, false)[3], Some(Badge::NotIndexed));
    }

    #[test]
    fn folder_counts_read_naturally_and_cap() {
        assert_eq!(dir_count_text(1, false), "1 item");
        assert_eq!(dir_count_text(12, false), "12 items");
        assert_eq!(dir_count_text(12, true), "12");
        assert_eq!(dir_count_text(DIR_COUNT_CAP + 1, false), "999+ items");
        assert_eq!(dir_count_text(0, false), "0 items");
    }

    #[test]
    fn placeholders_follow_the_row_kind() {
        let mut row = Thumbnail::default();
        row.filename = "a.jpg".into();
        assert_eq!(placeholder_icon(&row, RowKind::Dir), svg::FOLDER);
        assert_eq!(placeholder_icon(&row, RowKind::Archive), svg::FILE_ZIP);
        assert_eq!(placeholder_icon(&row, RowKind::Video), svg::FILE_VIDEO);
        assert_eq!(placeholder_icon(&row, RowKind::Image), svg::FILE_JPG);
        row.filename = "notes.txt".into();
        assert_eq!(placeholder_icon(&row, RowKind::Other), svg::FILE_TXT);
    }

    #[test]
    fn icon_sizes_snap_to_eight_pixels_so_the_rasterizer_reuses_them() {
        assert_eq!(snap(64.0), 64.0);
        assert_eq!(snap(67.0), 64.0);
        assert_eq!(snap(70.0), 72.0);
        assert_eq!(snap(1.0), 8.0);
    }
}
