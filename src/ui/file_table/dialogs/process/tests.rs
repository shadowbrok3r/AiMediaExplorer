use std::time::Duration;

use super::*;
use crate::database::ui_themes::tests::RT;
use crate::tasks::TaskState;
use crate::ui::file_table::menus::db_delete::GLOBAL_DB_TESTS;
use crate::ui::file_table::test_rig::{ROWS, Rig, db, global_db, image_row, path_of};

fn row(path: &str, kind: &str) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
    t.path = path.into();
    t.file_type = kind.into();
    t
}

#[test]
fn only_images_on_disk_are_processable() {
    assert!(processable(&row("/p/a.JPG", "jpg")));
    assert!(!processable(&row("/p/a.mp4", "mp4")));
    assert!(!processable(&row("/p/sub", "<DIR>")));
    assert!(!processable(&row("zip:///p/x.zip!/a.jpg", "jpg")), "an entry inside an archive cannot be read by path");
    assert!(!processable(&row("/p/noext", "")));
}

#[test]
fn counts_tell_how_many_images_a_run_would_work_on() {
    let mut described = row("/p/a.jpg", "jpg");
    described.description = Some("a cat".into());
    let mut captioned = row("/p/b.jpg", "jpg");
    captioned.caption = Some("dog".into());
    let rows = [described, captioned, row("/p/c.jpg", "jpg"), row("/p/d.png", "png"), row("/p/e.mp4", "mp4"), row("/p/sub", "<DIR>")];
    let c = count_rows(rows.iter(), |r| r.path.ends_with("a.jpg") || r.path.ends_with("d.png"));
    assert_eq!(c, Counts { images: 4, described: 2, embedded: 2 });
    assert_eq!((c.describe_todo(false), c.describe_todo(true)), (2, 4));
    assert_eq!((c.clip_todo(false), c.clip_todo(true)), (2, 4));
    assert_eq!(count_rows(std::iter::empty(), |_| false), Counts::default());
}

#[test]
fn images_in_a_folder_are_found_with_or_without_the_subfolders() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("a").join("b");
    std::fs::create_dir_all(&sub).unwrap();
    for p in [dir.path().join("1.jpg"), dir.path().join("note.txt"), dir.path().join("a").join("2.PNG"), sub.join("3.webp"), sub.join("movie.mp4")] {
        std::fs::write(p, b"x").unwrap();
    }
    let flat = images_in(dir.path(), false);
    assert_eq!(flat.len(), 1);
    let deep = images_in(dir.path(), true);
    assert_eq!(deep.len(), 3, "{deep:?}");
    assert!(deep.iter().all(|p| is_image_path(p)));
    assert!(images_in(Path::new("/definitely/not/here"), true).is_empty());
}

#[test]
fn opening_with_a_folder_chooses_the_folder_scope() {
    let mut d = ProcessDialog::default();
    assert_eq!(d.scope, ProcessScope::Selection);
    d.open_for(None);
    assert!(d.open && d.scope == ProcessScope::Selection);
    d.open_for(Some("/p".into()));
    assert_eq!((d.folder.as_str(), d.scope), ("/p", ProcessScope::Folder));
    let o = d.options();
    assert!(o.describe && o.clip && !o.overwrite_descriptions && !o.regenerate_clip, "both operations are on and nothing is overwritten by default");
}

fn process_counts(rig: &mut Rig) -> (Counts, Counts) {
    rig.ex.process_counts()
}

#[test]
fn the_counts_follow_the_selection_and_the_view() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.table.get_mut_by_path(&path_of(0)).unwrap().description = Some("x".into());
    rig.ex.table.get_mut_by_path(&path_of(1)).unwrap().caption = Some("y".into());
    rig.select_images(0..3);
    let (selected, view) = process_counts(&mut rig);
    assert_eq!(selected, Counts { images: 3, described: 2, embedded: 0 });
    assert_eq!(view.images, ROWS, "the folder row is not an image");
    rig.ex.edit_filters(|s| s.name = "img_0".into());
    rig.settle(2);
    let (_, view) = process_counts(&mut rig);
    assert_eq!(view.images, 10, "'this view' counts what the filters list");
    rig.ex.clear_selection();
    assert_eq!(process_counts(&mut rig).0, Counts::default(), "an empty selection counts nothing");
}

#[test]
fn nothing_to_process_and_nothing_chosen_are_refused() {
    let mut rig = Rig::new(1000.0, 700.0);
    assert!(rig.ex.start_process(false).unwrap_err().contains("no images"), "an empty selection is not everything");
    rig.select_images(0..2);
    rig.ex.toolbar.process.describe = false;
    rig.ex.toolbar.process.clip = false;
    assert!(rig.ex.start_process(false).unwrap_err().contains("Choose describe"));
    rig.ex.toolbar.process.clip = true;
    rig.ex.toolbar.process.scope = ProcessScope::Folder;
    rig.ex.toolbar.process.folder = "/definitely/not/here".into();
    assert!(rig.ex.start_process(false).unwrap_err().contains("folder that exists"));
    rig.ex.toolbar.process.scope = ProcessScope::View;
    rig.ex.edit_filters(|s| s.name = "no-such-file".into());
    rig.settle(2);
    assert!(rig.ex.start_process(false).unwrap_err().contains("no images"), "a view with nothing listed has nothing to process");
}

#[test]
fn a_large_run_asks_with_the_exact_counts_before_anything_starts() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.table.bulk_load((100..140).map(image_row));
    rig.ex.toolbar.process.scope = ProcessScope::View;
    let before = crate::tasks::with(|r| r.tasks().len());
    let waiting = rig.ex.start_process(false).unwrap();
    assert_eq!(waiting, "waiting for confirmation");
    assert_eq!(rig.ex.toolbar.process.confirming, Some((ROWS + 40, 2)));
    assert_eq!(crate::tasks::with(|r| r.tasks().len()), before, "nothing started yet");
    rig.ex.toolbar.process.open = true;
    rig.settle(3);
    assert!(rig.has_text("Start on 52 images"), "the confirmation names the exact count");
    assert!(rig.has_text("Back"));
    let back = rig.text_pos("Back").unwrap();
    rig.click(back + vec2(10.0, 6.0));
    assert_eq!(rig.ex.toolbar.process.confirming, None);
}

#[test]
fn the_dialog_draws_the_scopes_with_counts_and_the_work_to_do() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.table.get_mut_by_path(&path_of(0)).unwrap().description = Some("x".into());
    rig.select_images(0..2);
    rig.ex.toolbar.process.open_for(None);
    rig.settle(3);
    for text in ["Process files", "Selection (2 images)", &format!("This view ({ROWS} images)"), "Folder", "Describe with the vision model", "Generate CLIP embeddings", "Start", "Close"] {
        assert!(rig.has_text(text), "{text} is not drawn");
    }
    assert!(rig.has_text("Show each image in the preview while it is described"), "the follow option of the old scan menu lives here now");
    assert!(rig.has_text("Describe: 1 of 2 images (1 already described)"), "the describe line counts what is skipped");
    assert!(rig.has_text("CLIP: 2 of 2 images"));
    rig.ex.toolbar.process.overwrite_descriptions = true;
    rig.settle(2);
    assert!(rig.has_text("Describe: 2 of 2 images"), "replacing descriptions covers every image");
}

#[test]
fn the_folder_scope_counts_its_images_in_the_background() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["a.jpg", "b.png", "c.txt"] {
        std::fs::write(dir.path().join(name), b"x").unwrap();
    }
    let mut rig = Rig::new(1000.0, 700.0);
    rig.ex.toolbar.process.open_for(Some(dir.path().display().to_string()));
    for _ in 0..200 {
        rig.settle(1);
        if rig.has_text("2 images found") {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(rig.has_text("2 images found"));
    assert_eq!(rig.ex.toolbar.process.folder_images(), Some(2));
    rig.ex.toolbar.process.recursive = false;
    assert_eq!(rig.ex.toolbar.process.folder_images(), None, "a changed option invalidates the count");
}

fn task_state(title: &str) -> Option<TaskState> {
    crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == title).map(|t| t.state))
}

fn wait_state(title: &str, want: TaskState) {
    for _ in 0..500 {
        if task_state(title) == Some(want) {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("{title}: task is {:?}, wanted {want:?}", task_state(title));
}

#[test]
fn describe_runs_as_a_tracked_task_that_reports_progress_and_completes() {
    let _runtime = RT.enter();
    let (tx, rx) = crossbeam::channel::unbounded();
    let paths: Vec<String> = (0..4).map(|i| format!("/definitely/not/here/{i}.jpg")).collect();
    spawn_describe(Source::Paths(paths), true, String::new(), tx);
    wait_state("Describe 4 images", TaskState::Completed);
    let t = crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == "Describe 4 images").cloned().unwrap());
    assert_eq!((t.done, t.total), (4, 4));
    assert!(rx.try_recv().is_err(), "files that are gone produce no update");
}

#[test]
fn a_folder_without_images_completes_with_nothing_to_do() {
    let _runtime = RT.enter();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.txt"), b"x").unwrap();
    let (tx, _rx) = crossbeam::channel::unbounded();
    let title = format!("Describe images in {}", dir.path().display());
    spawn_describe(Source::Folder { path: dir.path().display().to_string(), recursive: true }, true, String::new(), tx);
    wait_state(&title, TaskState::Completed);
    let t = crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == title).cloned().unwrap());
    assert_eq!((t.done, t.total), (0, 0));
}

#[test]
fn clip_skips_images_that_already_have_an_embedding_and_never_loads_the_model_for_nothing() {
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    db(async {
        crate::database::DB.query("CREATE clip_embeddings:procskip SET path = '/proc-skip/a.jpg', embedding = $e").bind(("e", vec![0.1f32; 4])).await.unwrap().check().unwrap();
    });
    let _runtime = RT.enter();
    spawn_clip(Source::Paths(vec!["/proc-skip/a.jpg".into()]), false);
    wait_state("Embed 1 image", TaskState::Completed);
    let t = crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == "Embed 1 image").cloned().unwrap());
    assert_eq!((t.state, t.total), (TaskState::Completed, 0));
    db(async {
        crate::database::DB.query("DELETE clip_embeddings WHERE path = '/proc-skip/a.jpg'").await.unwrap();
    });
}

#[test]
fn the_known_description_and_embedding_lists_come_from_the_library() {
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let (described, embedded) = db(async {
        crate::database::DB
            .query(
                r#"
                CREATE thumbnails:pd1 SET path = '/proc-known/1.jpg', filename = '1.jpg', description = 'a cat';
                CREATE thumbnails:pd2 SET path = '/proc-known/2.jpg', filename = '2.jpg', description = '';
                CREATE thumbnails:pd3 SET path = '/proc-known/3.jpg', filename = '3.jpg';
                CREATE clip_embeddings:pk1 SET path = '/proc-known/1.jpg', embedding = $e;
                "#,
            )
            .bind(("e", vec![0.1f32; 4]))
            .await
            .unwrap()
            .check()
            .unwrap();
        let paths: Vec<String> = (1..=3).map(|i| format!("/proc-known/{i}.jpg")).collect();
        let result = (described_paths(&paths).await, embedded_paths(&paths).await);
        crate::database::DB.query("DELETE thumbnails WHERE path CONTAINS '/proc-known/'; DELETE clip_embeddings WHERE path CONTAINS '/proc-known/'").await.unwrap();
        result
    });
    assert_eq!(described, HashSet::from(["/proc-known/1.jpg".to_string()]), "an empty description does not count");
    assert_eq!(embedded, HashSet::from(["/proc-known/1.jpg".to_string()]));
}
