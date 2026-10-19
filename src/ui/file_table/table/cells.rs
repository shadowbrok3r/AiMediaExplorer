//! Cell drawing and sort keys of the details view.

use std::borrow::Cow;

use chrono::Datelike;
use egui::{Button, Color32, CornerRadius, FontId, Image, Label, RichText, TextureOptions, Ui, Widget};
use humansize::DECIMAL;

use super::columns::Col;
use super::details::DetailsEvent;
use super::{FileTableViewer, TabAction};
use crate::Thumbnail;
use crate::icons::svg;
use crate::ui::file_table::rows::{RowKind, SortValue};
use crate::ui::file_table::scope::Badge;
use crate::ui::file_table::thumbs::{ThumbView, Tier};

/// Whether a CLIP embedding is known for the row, by path or by content hash.
pub fn has_clip(viewer: &FileTableViewer, row: &Thumbnail) -> bool {
    viewer.clip_presence.contains(&row.path) || row.hash.as_ref().is_some_and(|h| viewer.clip_presence_hashes.contains(h))
}

/// Sort key of `row` in `column`; the order matches what each column displays.
pub fn sort_value<'r>(viewer: &FileTableViewer, row: &'r Thumbnail, column: u8) -> SortValue<'r> {
    let date = |d: &Option<surrealdb::types::Datetime>| d.as_ref().map_or(SortValue::Missing, |d| SortValue::Int(d.timestamp_millis() as i128));
    let text = |s: &'r Option<String>| s.as_deref().map_or(SortValue::Missing, |s| SortValue::Text(Cow::Borrowed(s)));
    match Col::from_id(column) {
        Some(Col::Name) => SortValue::Text(Cow::Borrowed(&row.filename)),
        Some(Col::Path) => SortValue::Text(Cow::Borrowed(&row.path)),
        Some(Col::Size) => SortValue::Int(row.size as i128),
        Some(Col::Type) => SortValue::Text(Cow::Borrowed(&row.file_type)),
        Some(Col::Clip) => SortValue::Int(if has_clip(viewer, row) { 0 } else { 1 }),
        Some(Col::Modified) => date(&row.modified),
        Some(Col::DbCreated) => date(&row.db_created),
        Some(Col::Category) => text(&row.category),
        Some(Col::Tags) => SortValue::Text(Cow::Owned(row.tags.join(","))),
        Some(Col::Hash) => text(&row.hash),
        Some(Col::Similarity) => SortValue::Float(viewer.similar_scores.get(&row.path).map_or(f32::INFINITY, |s| -s)),
        Some(Col::Thumb) | None => SortValue::Missing,
    }
}

/// Whether `row` is a directory row.
pub fn is_dir(row: &Thumbnail) -> bool {
    row.file_type == crate::ui::file_table::rows::thumbnail::DIR_TYPE
}

pub fn kind_label(kind: RowKind) -> &'static str {
    match kind {
        RowKind::Dir => "folder",
        RowKind::Archive => "archive",
        RowKind::Image => "image",
        RowKind::Video => "video",
        RowKind::Other => "file",
    }
}

fn plain(ui: &mut Ui, text: impl Into<egui::WidgetText>) {
    ui.add(Label::new(text).selectable(false).truncate());
}

fn date_text(d: &Option<surrealdb::types::Datetime>) -> String {
    match d {
        Some(d) => format!("{}/{}/{}", d.month(), d.day(), d.year()),
        None => "-".to_string(),
    }
}

fn color_for_tag(tag: &str) -> Color32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    tag.hash(&mut h);
    let v = h.finish();
    let r = (v & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let b = ((v >> 16) & 0xFF) as u8;
    Color32::from_rgb(r / 2 + 64, g / 2 + 64, b / 2 + 64)
}

/// Name of a file with a badge for whether it is in the library, and whether the indexed file is still on disk.
fn name_cell(ui: &mut Ui, viewer: &FileTableViewer, row: &Thumbnail) {
    if is_dir(row) || row.path.is_empty() {
        return plain(ui, format!(" {}", row.filename));
    }
    let badge = viewer.badges.badge(&row.path);
    let (icon, tint) = match badge {
        Badge::Indexed => (svg::DATABASE, crate::theme::success(ui)),
        Badge::NotIndexed => (svg::CIRCLE_DASHED, ui.visuals().weak_text_color()),
        Badge::Offline => (svg::CLOUD_SLASH, crate::theme::warn(ui)),
    };
    ui.horizontal(|ui| {
        ui.add(icon.image(13.0).tint(tint)).on_hover_text(format!("{}: {}", badge.label(), badge.detail()));
        ui.add(Label::new(format!("{}", row.filename)).selectable(false).truncate());
    });
}

/// Thumbnail column: folder and archive icons, or the texture from the thumbnail service.
fn thumb_cell(ui: &mut Ui, viewer: &FileTableViewer, row: &Thumbnail) {
    ui.vertical_centered_justified(|ui| {
        let size = ui.available_size();
        let mips = TextureOptions::default().with_mipmap_mode(Some(egui::TextureFilter::Linear));
        if row.file_type == "<DIR>" {
            Image::new(egui::include_image!("../../../../assets/Icons/folder.png")).fit_to_exact_size(size).texture_options(mips).ui(ui);
        } else if row.file_type == "<ARCHIVE>" {
            Image::new(egui::include_image!("../../../../assets/Icons/zip.png")).fit_to_exact_size(size).texture_options(mips).ui(ui);
        } else {
            let distance = (ui.max_rect().center().y - ui.clip_rect().center().y).abs();
            let view = if row.path.is_empty() { ThumbView::Missing } else { viewer.thumbs.request(viewer.scope, &row.path, Tier::Thumb, distance) };
            match view {
                ThumbView::Ready { id, size: tex_size } => {
                    Image::from_texture(egui::load::SizedTexture::new(id, tex_size)).fit_to_exact_size(size).ui(ui);
                }
                ThumbView::Pending => {
                    ui.weak("...");
                }
                ThumbView::Missing => {
                    Image::new(egui::include_image!("../../../../assets/Icons/broken_link.png")).fit_to_exact_size(size).texture_options(mips).ui(ui);
                }
            }
        }
    });
}

/// Draws one cell; clicks on category and tag buttons are reported through `events`.
pub fn draw_cell(ui: &mut Ui, viewer: &FileTableViewer, col: Col, row: &Thumbnail, events: &mut Vec<DetailsEvent>) {
    match col {
        Col::Thumb => thumb_cell(ui, viewer, row),
        Col::Name => name_cell(ui, viewer, row),
        Col::Path => {
            ui.add(Label::new(&row.path).selectable(false).truncate()).on_hover_text(&row.path);
        }
        Col::Size => plain(ui, if row.size == 0 { "-".to_string() } else { humansize::format_size(row.size, DECIMAL) }),
        Col::Type => plain(ui, RichText::new(&row.file_type).font(FontId::monospace(11.))),
        Col::Clip => plain(ui, if has_clip(viewer, row) { "Yes" } else { "No" }),
        Col::Modified => plain(ui, date_text(&row.modified)),
        Col::DbCreated => plain(ui, date_text(&row.db_created)),
        Col::Category => {
            ui.vertical_centered_justified(|ui| match row.category.as_ref() {
                Some(cat) => {
                    if ui.button(RichText::new(cat).underline()).clicked() {
                        events.push(DetailsEvent::Tab(TabAction::OpenCategory(cat.clone())));
                    }
                }
                None => {
                    ui.label("-");
                }
            });
        }
        Col::Tags => {
            if row.tags.is_empty() {
                ui.label("-");
            } else {
                ui.horizontal_wrapped(|ui| {
                    for tag in &row.tags {
                        let chip = Button::new(tag.as_str()).fill(color_for_tag(tag)).corner_radius(CornerRadius::same(4));
                        let resp = chip.ui(ui);
                        if resp.hovered() {
                            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                        }
                        if resp.clicked() {
                            events.push(DetailsEvent::Tab(TabAction::OpenTag(tag.clone())));
                        }
                    }
                });
            }
        }
        Col::Hash => plain(ui, row.hash.as_deref().unwrap_or("-")),
        Col::Similarity => match viewer.similar_scores.get(&row.path) {
            Some(score) => plain(ui, format!("{score:.4}")),
            None => plain(ui, "-"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::file_table::table::FileTableViewer;

    fn viewer() -> FileTableViewer {
        let (a, _) = crossbeam::channel::unbounded();
        let (b, _) = crossbeam::channel::unbounded();
        let (c, _) = crossbeam::channel::unbounded();
        FileTableViewer::new(a, b, c)
    }

    fn row(path: &str) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit('/').next().unwrap().into();
        t
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clip_presence_matches_by_path_or_hash() {
        let mut v = viewer();
        let mut r = row("/a/x.jpg");
        assert!(!has_clip(&v, &r));
        v.clip_presence.insert("/a/x.jpg".into());
        assert!(has_clip(&v, &r));
        v.clip_presence.clear();
        r.hash = Some("h1".into());
        assert!(!has_clip(&v, &r));
        v.clip_presence_hashes.insert("h1".into());
        assert!(has_clip(&v, &r));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sort_values_follow_what_each_column_shows() {
        let mut v = viewer();
        let mut a = row("/a/a.jpg");
        a.size = 5;
        a.category = Some("cats".into());
        a.tags = vec!["x".into(), "y".into()];
        let b = row("/a/b.jpg");
        assert_eq!(sort_value(&v, &a, Col::Name.id()), SortValue::Text("a.jpg".into()));
        assert_eq!(sort_value(&v, &a, Col::Size.id()), SortValue::Int(5));
        assert_eq!(sort_value(&v, &a, Col::Category.id()), SortValue::Text("cats".into()));
        assert_eq!(sort_value(&v, &b, Col::Category.id()), SortValue::Missing);
        assert_eq!(sort_value(&v, &a, Col::Tags.id()), SortValue::Text("x,y".into()));
        assert_eq!(sort_value(&v, &b, Col::Modified.id()), SortValue::Missing);
        assert_eq!(sort_value(&v, &a, Col::Thumb.id()), SortValue::Missing);
        assert_eq!(sort_value(&v, &a, 200), SortValue::Missing);
        v.clip_presence.insert(a.path.clone());
        assert_eq!(sort_value(&v, &a, Col::Clip.id()).compare(&sort_value(&v, &b, Col::Clip.id())), std::cmp::Ordering::Less, "rows with a CLIP embedding come first");
        v.similar_scores.insert(a.path.clone(), 0.9);
        v.similar_scores.insert(b.path.clone(), 0.2);
        assert_eq!(sort_value(&v, &a, Col::Similarity.id()).compare(&sort_value(&v, &b, Col::Similarity.id())), std::cmp::Ordering::Less, "higher scores come first");
        let missing = row("/a/none.jpg");
        assert_eq!(sort_value(&v, &b, Col::Similarity.id()).compare(&sort_value(&v, &missing, Col::Similarity.id())), std::cmp::Ordering::Less, "unscored rows come last");
    }

    #[test]
    fn dates_print_month_day_year() {
        assert_eq!(date_text(&None), "-");
        let d: surrealdb::types::Datetime = chrono::DateTime::parse_from_rfc3339("2024-03-07T10:00:00Z").unwrap().with_timezone(&chrono::Utc).into();
        assert_eq!(date_text(&Some(d)), "3/7/2024");
    }

    #[test]
    fn tag_colors_are_stable_per_tag() {
        assert_eq!(color_for_tag("fox"), color_for_tag("fox"));
        assert_ne!(color_for_tag("fox"), color_for_tag("owl"));
    }

    #[test]
    fn kind_labels_cover_every_kind() {
        for k in [RowKind::Dir, RowKind::Archive, RowKind::Image, RowKind::Video, RowKind::Other] {
            assert!(!kind_label(k).is_empty());
        }
    }

    #[test]
    fn directories_are_recognised_by_their_sentinel() {
        let mut d = row("/a/sub");
        d.file_type = "<DIR>".into();
        assert!(is_dir(&d) && !is_dir(&row("/a/f.jpg")));
    }
}
