use super::*;
use crate::database::metadata_edit::MetaPatch;

fn row(path: &str) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap());
    t.path = path.into();
    t.file_type = "jpg".into();
    t
}

fn tags(list: &[&str]) -> Vec<String> {
    list.iter().map(|t| t.to_string()).collect()
}

#[test]
fn a_patch_changes_only_the_fields_it_carries_on_the_row_of_its_path() {
    let mut a = row("/d/a.jpg");
    a.caption = Some("old".into());
    a.category = Some("keep".into());
    let mut b = a.clone();
    b.path = "/d/b.jpg".into();
    let edit = RowEdit::Patch { path: "/d/a.jpg".into(), patch: MetaPatch { caption: Some(Some("new".into())), tags: Some(tags(&["x"])), ..Default::default() } };
    edit.apply_to(&mut a);
    edit.apply_to(&mut b);
    assert_eq!((a.caption.as_deref(), a.category.as_deref(), a.tags.clone()), (Some("new"), Some("keep"), tags(&["x"])));
    assert_eq!(b.caption.as_deref(), Some("old"), "another path is untouched");
    assert!(b.tags.is_empty());
}

#[test]
fn clearing_a_field_in_a_patch_empties_the_row_value() {
    let mut a = row("/d/a.jpg");
    a.description = Some("text".into());
    a.category = Some("c".into());
    RowEdit::Patch { path: a.path.clone(), patch: MetaPatch { description: Some(Some("   ".into())), category: Some(None), ..Default::default() } }.apply_to(&mut a);
    assert_eq!((a.description, a.category), (None, None));
}

#[test]
fn tag_and_category_edits_reach_only_the_rows_they_name() {
    let (mut a, mut b) = (row("/d/a.jpg"), row("/d/b.jpg"));
    let tags_edit = RowEdit::Tags(HashMap::from([("/d/a.jpg".to_string(), tags(&["one", "two"]))]));
    tags_edit.apply_to(&mut a);
    tags_edit.apply_to(&mut b);
    assert_eq!((a.tags.clone(), b.tags.clone()), (tags(&["one", "two"]), Vec::<String>::new()));
    let category = RowEdit::Category { paths: vec!["/d/b.jpg".into()], category: Some("pets".into()) };
    category.apply_to(&mut a);
    category.apply_to(&mut b);
    assert_eq!((a.category, b.category.as_deref()), (None, Some("pets")));
}

#[tokio::test]
async fn an_edit_is_applied_to_the_table_rows_and_the_current_selection_copy() {
    let mut ex = FileExplorer::new(true);
    ex.table.bulk_load(vec![row("/d/a.jpg"), row("/d/b.jpg")]);
    ex.sync_view();
    ex.select_paths(["/d/a.jpg"]);
    ex.current_thumb = ex.table.get_by_path("/d/a.jpg").cloned().unwrap();
    apply_edit(&mut ex, &RowEdit::Category { paths: vec!["/d/a.jpg".into(), "/d/b.jpg".into()], category: Some("trip".into()) });
    assert_eq!(ex.table.get_by_path("/d/a.jpg").unwrap().category.as_deref(), Some("trip"));
    assert_eq!(ex.table.get_by_path("/d/b.jpg").unwrap().category.as_deref(), Some("trip"));
    assert_eq!(ex.current_thumb.category.as_deref(), Some("trip"));
    apply_edit(&mut ex, &RowEdit::Tags(HashMap::from([("/d/zzz.jpg".to_string(), tags(&["ghost"]))])));
    assert!(ex.table.get_by_path("/d/a.jpg").unwrap().tags.is_empty(), "a path the table lacks changes nothing");
}

#[test]
fn only_model_work_on_large_selections_asks_first() {
    let heavy = [BatchAction::Describe { regenerate: false }, BatchAction::Describe { regenerate: true }, BatchAction::GenerateClip];
    let light = [BatchAction::AddTags(tags(&["a"])), BatchAction::RemoveTags(tags(&["a"])), BatchAction::SetCategory(None), BatchAction::Refine, BatchAction::AttachToAssistant];
    for action in &heavy {
        assert!(!needs_confirm(action, CONFIRM_ABOVE), "{action:?} at the threshold runs");
        assert!(needs_confirm(action, CONFIRM_ABOVE + 1), "{action:?} above the threshold asks");
        assert!(!needs_confirm(action, 1));
    }
    for action in &light {
        assert!(!needs_confirm(action, 10_000), "{action:?} never asks");
    }
}

#[test]
fn local_tags_apply_the_change_to_each_row_as_shown() {
    let mut a = row("/d/a.jpg");
    a.tags = tags(&["Cat", "pet"]);
    let b = row("/d/b.jpg");
    let result = local_tags(&[a, b], &TagChange { add: tags(&["trip", "cat"]), remove: tags(&["PET"]) });
    assert_eq!(result["/d/a.jpg"], tags(&["Cat", "trip"]), "the case-insensitive duplicate is not added twice and removal ignores case");
    assert_eq!(result["/d/b.jpg"], tags(&["trip", "cat"]));
}

#[test]
fn counts_and_texture_cost_of_the_preview_report() {
    let report = PreviewReport { source: "thumbnail", texture: Some([200, 100]), image: Some([4000.0, 2000.0]) };
    assert_eq!(report.texture_bytes(), 80_000);
    assert_eq!(PreviewReport::default().texture_bytes(), 0);
}

#[test]
fn opening_the_viewer_starts_from_a_fresh_view() {
    let mut lightbox = Lightbox::default();
    lightbox.view.zoom = 3.0;
    lightbox.view.pan = egui::vec2(10.0, 5.0);
    lightbox.show();
    assert!(lightbox.open);
    assert_eq!((lightbox.view.zoom, lightbox.view.pan), (1.0, egui::Vec2::ZERO));
}

#[test]
fn the_prefs_of_a_new_inspector_come_from_the_store_and_toggle_round_trips() {
    let mut insp = Inspector::new(Arc::new(DbBackend::global()), PrefsStore::default());
    assert!(insp.is_open());
    assert!(!insp.toggle());
    assert!(!insp.is_open());
    assert!(insp.toggle());
    insp.edit_prefs(|p| p.width = 500.0);
    assert_eq!(insp.prefs.prefs.width, 500.0);
}

#[test]
fn notices_are_handed_out_once() {
    let mut insp = Inspector::default();
    insp.notify(ToastLevel::Info, "one");
    insp.notify(ToastLevel::Error, "two");
    let taken = insp.take_notices();
    assert_eq!(taken.len(), 2);
    assert_eq!(taken[1], (ToastLevel::Error, "two".to_string()));
    assert!(insp.take_notices().is_empty());
}
