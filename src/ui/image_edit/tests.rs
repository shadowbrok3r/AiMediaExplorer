//! The Image Edit tab: model rules, saving, the job state machine against a scripted backend, and headless frames.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui::{CentralPanel, Context, Pos2, RawInput, Rect, vec2};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::backend::fake::FakeBackend;
use super::model::*;
use super::view::contained;
use super::*;
use crate::ui::test_support::{test_ctx, texts};

fn png_bytes(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba(color));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn source_file(dir: &Path, name: &str) -> String {
    let path = dir.join(name);
    image::RgbImage::from_pixel(40, 30, image::Rgb([200, 40, 40])).save(&path).unwrap();
    path.display().to_string()
}

fn panel(backend: Arc<FakeBackend>) -> ImageEditPanel {
    ImageEditPanel::with_backend(backend, DbSource::Global, crate::ui::file_table::thumbs::service())
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
}

async fn settle(panel: &mut ImageEditPanel, ctx: &Context, done: impl Fn(&ImageEditPanel) -> bool) {
    for _ in 0..400 {
        panel.pump(ctx);
        if done(panel) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the panel did not reach the expected state; phase {:?}", panel.phase());
}

fn finished(p: &ImageEditPanel) -> bool {
    !p.phase().is_busy()
}

#[test]
fn errors_are_sorted_by_what_the_user_can_do() {
    for text in ["CUDA_ERROR_OUT_OF_MEMORY", "out of memory", "cuda error: named symbol not found", "failed to allocate 3 GiB", "OOM while denoising"] {
        assert_eq!(classify(text), ErrorKind::OutOfMemory, "{text}");
    }
    assert_eq!(classify("No such file or directory"), ErrorKind::Source);
    assert_eq!(classify("model load failed: repo not found"), ErrorKind::ModelLoad);
    assert_eq!(classify("something else"), ErrorKind::Other);
    for kind in [ErrorKind::OutOfMemory, ErrorKind::ModelLoad, ErrorKind::Source, ErrorKind::Other] {
        assert!(EditError::new(kind, "x").advice().len() > 20);
    }
    assert!(EditError::classified("out of memory").may_retry_lower_precision());
    assert!(!EditError::classified("repo not found while loading").may_retry_lower_precision());
}

#[test]
fn the_phases_describe_themselves() {
    assert!(Phase::LoadingModel.is_busy() && Phase::Cancelling.is_busy());
    assert!(!Phase::Idle.is_busy() && !Phase::Done.is_busy() && !Phase::Cancelled.is_busy());
    let running = Phase::Running { done: 3, total: 12, detail: "Denoising".into() };
    assert_eq!(running.fraction(), Some(0.25));
    assert_eq!(running.label(), "Denoising (3 of 12)");
    assert_eq!(Phase::Running { done: 0, total: 0, detail: String::new() }.fraction(), None);
    assert_eq!(Phase::Running { done: 2, total: 9, detail: String::new() }.label(), "Editing, step 2 of 9");
    assert_eq!(Phase::Failed(EditError::new(ErrorKind::Other, "boom")).label(), "Failed: boom");
}

#[test]
fn an_edit_needs_a_prompt_and_presets_fill_or_extend_it() {
    let mut params = Params::default();
    assert_eq!(params.problem(), Some("Describe the change first"));
    params.prompt = "  ".into();
    assert!(params.problem().is_some());
    params.prompt = "Make it brighter.".into();
    assert!(params.problem().is_none());
    assert_eq!(params.with_preset("Sharpen the details", false), "Sharpen the details");
    assert_eq!(params.with_preset("Sharpen the details", true), "Make it brighter. Sharpen the details");
    params.prompt.clear();
    assert_eq!(params.with_preset("Sharpen the details", true), "Sharpen the details", "nothing to extend");
    let labels: std::collections::HashSet<&str> = PRESETS.iter().map(|(l, _)| *l).collect();
    assert_eq!(labels.len(), PRESETS.len(), "preset labels are unique");
    assert!(PRESETS.iter().all(|(l, t)| !l.is_empty() && t.len() > 10));
}

#[test]
fn output_names_never_collide_with_existing_files() {
    let taken = ["/p/a-edited.png", "/p/a-edited-2.png"];
    let exists = |p: &Path| taken.contains(&p.to_str().unwrap());
    assert_eq!(unique_output_path("/p/a.jpg", "png", exists).to_str().unwrap(), "/p/a-edited-3.png");
    assert_eq!(unique_output_path("/p/b.jpg", "png", exists).to_str().unwrap(), "/p/b-edited.png");
    assert_eq!(unique_output_path("/p/no_ext", "png", |_| false).to_str().unwrap(), "/p/no_ext-edited.png");
    assert_eq!(file_name("/p/q/a.jpg"), "a.jpg");
    assert_eq!(fit_dims(4000, 3000, 2000), (2000, 1500));
    assert_eq!(fit_dims(800, 600, 2000), (800, 600));
}

#[test]
fn a_picture_is_contained_in_its_area_at_its_own_ratio() {
    let area = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 200.0));
    let wide = contained(area, vec2(200.0, 50.0));
    assert_eq!((wide.width(), wide.height()), (400.0, 100.0));
    let tall = contained(area, vec2(50.0, 200.0));
    assert_eq!((tall.width(), tall.height()), (50.0, 200.0));
    assert_eq!(tall.center(), area.center());
    assert_eq!(contained(area, vec2(0.0, 0.0)), area);
}

#[test]
fn results_are_written_in_the_format_of_the_target_and_next_to_the_original() {
    let dir = tempfile::tempdir().unwrap();
    let png = png_bytes(16, 8, [10, 200, 30, 255]);
    assert_eq!(save::encode_for_path(&png, Path::new("a.png")).unwrap(), png, "png passes through");
    let jpg = save::encode_for_path(&png, Path::new("a.JPG")).unwrap();
    assert_eq!(image::guess_format(&jpg).unwrap(), image::ImageFormat::Jpeg);
    assert_eq!(image::load_from_memory(&jpg).unwrap().width(), 16);
    assert!(save::encode_for_path(&png, Path::new("a.xyz")).unwrap_err().contains(".xyz"));

    let source = source_file(dir.path(), "photo.jpg");
    let first = save::save_next_to(&png, &source).unwrap();
    let second = save::save_next_to(&png, &source).unwrap();
    assert!(first.ends_with("photo-edited.png") && second.ends_with("photo-edited-2.png"), "{first} {second}");
    assert!(Path::new(&source).exists(), "the original is untouched");
    let nested = dir.path().join("new/folder/out.png");
    save::write_result(&png, &nested).unwrap();
    assert!(nested.exists(), "missing folders are created");
}

#[test]
fn only_the_source_file_can_be_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let other = source_file(dir.path(), "other.jpg");
    let png = png_bytes(10, 10, [0, 0, 255, 255]);
    assert!(save::overwrite_original(&png, &source, &other).unwrap_err().contains("only the file"));
    let before = std::fs::read(&other).unwrap();
    assert_eq!(std::fs::read(&other).unwrap(), before);
    save::overwrite_original(&png, &source, &source).unwrap();
    let written = std::fs::read(&source).unwrap();
    assert_eq!(image::guess_format(&written).unwrap(), image::ImageFormat::Jpeg, "the original keeps its format");
    assert_eq!(image::load_from_memory(&written).unwrap().width(), 10);
}

#[tokio::test]
async fn a_saved_result_can_join_the_library_and_a_collection() {
    let db = mem_db().await;
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let saved = save::save_next_to(&png_bytes(8, 8, [1, 2, 3, 255]), &source).unwrap();
    let source_db = DbSource::Handle(db.clone());
    let text = save::add_to_library(&source_db, &saved, Some("Edits")).await.unwrap();
    assert!(text.contains("Added photo-edited.png to the library") && text.contains("\"Edits\" (new collection)"), "{text}");
    let again = save::add_to_library(&source_db, &saved, Some("Edits")).await.unwrap();
    assert!(again.contains("already in the library") && !again.contains("new collection"), "{again}");
    assert_eq!(save::collection_names(&source_db).await, vec!["Edits".to_string()]);
    let in_collection: Vec<String> = db.query("SELECT VALUE path FROM collection_items").await.unwrap().take(0).unwrap();
    assert_eq!(in_collection, vec![saved.clone()]);
    let entries = crate::database::journal::recent_with(&db, 5).await.unwrap();
    assert_eq!(entries.len(), 1, "the membership is journaled once");
    let none = save::add_to_library(&source_db, &source, None).await.unwrap();
    assert!(none.contains("Added photo.jpg"));
}

#[tokio::test]
async fn an_edit_loads_the_model_once_runs_and_keeps_the_result() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(24, 18, [20, 120, 220, 255]));
    let mut p = panel(backend.clone());
    let ctx = test_ctx();
    p.open_with_path(&source);
    p.params.prompt = "Make the sky orange".into();
    p.params.steps = 7;
    p.start().unwrap();
    assert_eq!(p.phase(), &Phase::LoadingModel);
    assert!(p.cancel_handle().is_some(), "the job can be cancelled from the task list");
    settle(&mut p, &ctx, finished).await;
    assert_eq!(p.phase(), &Phase::Done);
    assert_eq!((backend.loads.load(Ordering::SeqCst), backend.edits.load(Ordering::SeqCst)), (1, 1));
    let seen = backend.seen.lock().unwrap()[0].clone();
    assert_eq!((seen.params.prompt.as_str(), seen.params.steps, seen.source.to_str().unwrap()), ("Make the sky orange", 7, source.as_str()));
    assert_eq!(p.history().len(), 1);
    let e = &p.history()[0];
    assert_eq!((e.full, e.source.as_str(), e.prompt.as_str(), e.saved.clone()), ([24, 18], source.as_str(), "Make the sky orange", None));
    assert_eq!(p.shown_entry().unwrap().id, e.id, "the newest result for the source is shown");

    p.start().unwrap();
    assert_eq!(p.phase(), &Phase::Running { done: 0, total: 7, detail: "Starting".into() }, "a loaded model starts at once");
    settle(&mut p, &ctx, |p| finished(p) && p.history().len() == 2).await;
    assert_eq!(backend.loads.load(Ordering::SeqCst), 1, "the model was not loaded again");
}

#[tokio::test]
async fn an_edit_cannot_start_without_a_source_a_prompt_or_while_busy() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    let mut p = panel(backend.clone());
    assert_eq!(p.blocked(), Some("Choose an image first"));
    assert!(p.start().is_err());
    p.open_with_path(&source);
    assert_eq!(p.blocked(), Some("Describe the change first"));
    p.params.prompt = "x".into();
    assert_eq!(p.blocked(), None);
    p.start().unwrap();
    assert_eq!(p.blocked(), Some("An edit is already running"));
    assert!(p.start().is_err());
    let ctx = test_ctx();
    settle(&mut p, &ctx, finished).await;
    assert_eq!(backend.edits.load(Ordering::SeqCst), 1, "only one edit ran");
}

#[tokio::test]
async fn a_cancelled_edit_leaves_no_result() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    let mut p = ImageEditPanel::with_backend(backend.clone(), DbSource::Global, crate::ui::file_table::thumbs::service());
    backend.loaded.store(true, Ordering::SeqCst);
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.params.steps = 3;
    let ctx = test_ctx();
    p.start().unwrap();
    p.cancel();
    assert_eq!(p.phase(), &Phase::Cancelling);
    p.cancel_handle().unwrap().call();
    settle(&mut p, &ctx, finished).await;
    assert_eq!(p.phase(), &Phase::Cancelled);
    assert!(p.history().is_empty(), "the partial picture is not kept");
    assert!(p.start().is_ok(), "a new edit can start after a cancel");
    settle(&mut p, &ctx, finished).await;
}

#[tokio::test]
async fn a_model_that_will_not_load_says_what_to_do() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    *backend.load_error.lock().unwrap() = Some("repo not found".into());
    let mut p = panel(backend.clone());
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.start().unwrap();
    let ctx = test_ctx();
    settle(&mut p, &ctx, finished).await;
    let Phase::Failed(e) = p.phase().clone() else { panic!("expected a failure, got {:?}", p.phase()) };
    assert_eq!((e.kind, e.message.as_str()), (ErrorKind::ModelLoad, "repo not found"));
    assert!(e.advice().contains("model repo"));
    assert_eq!(backend.edits.load(Ordering::SeqCst), 0, "no edit ran without a model");
    *backend.load_error.lock().unwrap() = None;
    p.phase = Phase::Idle;
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    assert_eq!(p.phase(), &Phase::Done);
}

#[tokio::test]
async fn running_out_of_memory_offers_a_lower_precision_retry_that_works() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(8, 8, [9, 9, 9, 255]));
    backend.edit_errors.lock().unwrap().push("CUDA_ERROR_OUT_OF_MEMORY".into());
    let mut p = panel(backend.clone());
    backend.loaded.store(true, Ordering::SeqCst);
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.params.steps = 40;
    assert!(p.params.prefer_full_precision);
    p.start().unwrap();
    let ctx = test_ctx();
    settle(&mut p, &ctx, finished).await;
    let Phase::Failed(e) = p.phase().clone() else { panic!("expected a failure") };
    assert!(e.may_retry_lower_precision());
    p.retry_lower_precision().unwrap();
    assert!(!p.params.prefer_full_precision);
    assert_eq!(p.params.steps, 20, "the retry shortens the run");
    settle(&mut p, &ctx, finished).await;
    assert_eq!(p.phase(), &Phase::Done);
    let seen = backend.seen.lock().unwrap().clone();
    assert_eq!((seen.len(), seen[1].params.prefer_full_precision, seen[1].params.steps), (2, false, 20));
}

#[tokio::test]
async fn a_missing_source_fails_at_once_with_a_source_error() {
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    let mut p = panel(backend.clone());
    p.open_with_path("/no/such/dir/photo.jpg");
    p.params.prompt = "x".into();
    p.start().unwrap();
    let ctx = test_ctx();
    settle(&mut p, &ctx, finished).await;
    let Phase::Failed(e) = p.phase().clone() else { panic!("expected a failure") };
    assert_eq!(e.kind, ErrorKind::Source);
    assert_eq!(backend.loads.load(Ordering::SeqCst), 0, "the model is not loaded for a file that is not there");
}

#[tokio::test]
async fn step_previews_show_while_the_edit_runs_and_go_when_it_ends() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(12, 12, [200, 100, 0, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let mut p = panel(backend);
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.params.preview_every_n = 1;
    p.start().unwrap();
    let ctx = test_ctx();
    let mut saw_preview = false;
    for _ in 0..400 {
        p.pump(&ctx);
        saw_preview |= p.live.is_some();
        if finished(&p) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert!(saw_preview, "a live preview appeared");
    assert!(p.live.is_none() && p.phase() == &Phase::Done, "the preview is replaced by the result");
}

#[tokio::test]
async fn the_history_is_capped_and_a_result_can_be_picked_or_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (source_file(dir.path(), "a.jpg"), source_file(dir.path(), "b.jpg"));
    let backend = FakeBackend::new(png_bytes(8, 8, [5, 5, 5, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let mut p = panel(backend);
    let ctx = test_ctx();
    p.params.prompt = "x".into();
    for i in 0..(HISTORY_MAX + 2) {
        p.open_with_path(if i % 2 == 0 { &a } else { &b });
        p.start().unwrap();
        settle(&mut p, &ctx, finished).await;
    }
    assert_eq!(p.history().len(), HISTORY_MAX, "the oldest results are dropped");
    let first_id = p.history()[0].id;
    p.show_entry(first_id);
    assert_eq!(p.shown_entry().unwrap().id, first_id);
    assert_eq!(p.current_path.as_deref(), Some(p.history()[0].source.as_str()), "the source of that result comes back with it");
    p.remove_entry(first_id);
    assert_eq!(p.history().len(), HISTORY_MAX - 1);
    assert!(p.shown_entry().map_or(true, |e| e.id != first_id));
}

#[tokio::test]
async fn saving_and_replacing_go_through_explicit_steps() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let original = std::fs::read(&source).unwrap();
    let backend = FakeBackend::new(png_bytes(20, 20, [250, 0, 0, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let mut p = panel(backend);
    let ctx = test_ctx();
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    let id = p.history()[0].id;

    p.save_next_to_original(id);
    settle(&mut p, &ctx, |p| p.history()[0].saved.is_some()).await;
    let saved = p.history()[0].saved.clone().unwrap();
    assert!(saved.ends_with("photo-edited.png") && Path::new(&saved).exists());
    assert_eq!(std::fs::read(&source).unwrap(), original, "saving a copy leaves the original alone");

    p.ask_replace(id);
    assert!(p.replace_target().is_some_and(|(_, s)| s == source), "the question names the file");
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(std::fs::read(&source).unwrap(), original, "asking writes nothing");
    p.cancel_replace();
    assert!(p.replace_target().is_none());
    assert_eq!(std::fs::read(&source).unwrap(), original, "declining writes nothing");

    p.ask_replace(id);
    p.confirm_replace();
    settle(&mut p, &ctx, |p| p.notice.as_ref().is_some_and(|(_, t)| t.starts_with("Saved") && t.ends_with("photo.jpg"))).await;
    assert_ne!(std::fs::read(&source).unwrap(), original, "only the confirmed replace wrote the original");
}

#[tokio::test]
async fn adding_a_result_to_the_library_saves_it_first_and_files_it() {
    let db = mem_db().await;
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(20, 20, [0, 250, 0, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let mut p = ImageEditPanel::with_backend(backend, DbSource::Handle(db.clone()), crate::ui::file_table::thumbs::service());
    let ctx = test_ctx();
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    let id = p.history()[0].id;
    p.new_collection = "Edited photos".into();
    p.add_to_library(id);
    settle(&mut p, &ctx, |p| p.history()[0].in_library).await;
    let saved = p.history()[0].saved.clone().expect("the result was written first");
    let rows: Vec<String> = db.query("SELECT VALUE path FROM thumbnails").await.unwrap().take(0).unwrap();
    assert_eq!(rows, vec![saved.clone()]);
    let members: Vec<String> = db.query("SELECT VALUE path FROM collection_items").await.unwrap().take(0).unwrap();
    assert_eq!(members, vec![saved]);
    assert!(p.notice.as_ref().is_some_and(|(ok, t)| *ok && t.contains("Edited photos")));
}

#[tokio::test]
async fn the_gguf_and_model_buttons_report_back() {
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    let mut p = panel(backend.clone());
    let ctx = test_ctx();
    p.load_model();
    settle(&mut p, &ctx, |p| !p.model_busy).await;
    assert!(backend.is_loaded() && p.notice.as_ref().is_some_and(|(ok, _)| *ok));
    p.unload_model();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!backend.is_loaded());
    *backend.load_error.lock().unwrap() = Some("download failed".into());
    p.load_model();
    settle(&mut p, &ctx, |p| !p.model_busy).await;
    assert_eq!(p.model_error.as_ref().map(|e| e.message.as_str()), Some("download failed"));
    p.sender().send(Msg::Gguf(Ok("/cache/model.gguf".into()))).unwrap();
    p.pump(&ctx);
    assert_eq!(p.settings.gguf_path.as_deref(), Some("/cache/model.gguf"));
    p.sender().send(Msg::Gguf(Err("no network".into()))).unwrap();
    p.pump(&ctx);
    assert_eq!(p.gguf.status.as_deref(), Some("Error: no network"));
}

fn frame(ctx: &Context, panel: &mut ImageEditPanel, width: f32) -> Vec<(String, Pos2)> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 1000.0))), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| panel.ui(ui));
        });
        out.textures_delta.clear();
        last = texts(&out.shapes);
    }
    last
}

fn has(drawn: &[(String, Pos2)], needle: &str) -> bool {
    drawn.iter().any(|(t, _)| t.contains(needle))
}

struct NoProducer;

impl crate::ui::file_table::thumbs::service::Producer for NoProducer {
    fn request(&self, _path: &Path) -> bool {
        true
    }
}

/// A thumbnail service that already holds a picture for `path`.
fn service_with(path: &str) -> Arc<crate::ui::file_table::thumbs::ThumbService> {
    use crate::ui::file_table::thumbs::service::ServiceConfig;
    use crate::ui::file_table::thumbs::store::{Blob, BlobFormat, MemStore};
    let img: image::RgbImage = image::ImageBuffer::from_fn(120, 80, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, 90]));
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg).unwrap();
    let store = MemStore::new();
    store.insert(path, Blob { format: BlobFormat::Jpeg, w: 120, h: 80, bytes });
    crate::ui::file_table::thumbs::ThumbService::new(store, tokio::runtime::Handle::current(), ServiceConfig::default(), Arc::new(NoProducer))
}

#[tokio::test]
async fn the_tab_draws_every_section_at_three_widths() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(30, 20, [30, 90, 200, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let ctx = test_ctx();
    for width in [700.0, 1000.0, 1600.0] {
        let mut p = ImageEditPanel::with_backend(backend.clone(), DbSource::Global, service_with(&source));
        let before = frame(&ctx, &mut p, width);
        assert!(has(&before, "No image selected") && has(&before, "Choose image..."), "{width}: an empty tab explains itself");
        p.open_with_path(&source);
        p.params.prompt = "Make it pop".into();
        for _ in 0..100 {
            frame(&ctx, &mut p, width);
            if p.source_view().is_ready() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(p.source_view().is_ready(), "{width}: the original has a picture");
        let idle = frame(&ctx, &mut p, width);
        for expected in ["Image Edit", "photo.jpg", "Prompt", "Enhance light", "Run edit", "Compare", "Side by side", "Swipe", "Flip", "Model settings", "Original", "No result yet"] {
            assert!(has(&idle, expected), "{width}: missing {expected:?} in {idle:?}");
        }
        p.start().unwrap();
        settle(&mut p, &ctx, finished).await;
        let done = frame(&ctx, &mut p, width);
        for expected in ["Result", "Save as new file...", "Save next to original", "Replace original...", "Add to library", "Collection", "History (1)", "30 x 20 px", "Make it pop"] {
            assert!(has(&done, expected), "{width}: missing {expected:?} in {done:?}");
        }
        assert!(done.iter().all(|(_, pos)| pos.x < width), "{width}: nothing starts beyond the right edge");
        p.compare = CompareMode::Swipe;
        let swipe = frame(&ctx, &mut p, width);
        assert!(has(&swipe, "Original") && has(&swipe, "Result"), "{width}: both sides of the swipe are labelled");
        p.compare = CompareMode::Toggle;
        let flip = frame(&ctx, &mut p, width);
        assert!(has(&flip, "Result (click to flip)"), "{width}: {flip:?}");
        p.flipped = true;
        let flipped = frame(&ctx, &mut p, width);
        assert!(has(&flipped, "Original (click to flip)"), "{width}: {flipped:?}");
    }
}

#[tokio::test]
async fn the_tab_shows_progress_cancel_and_the_failure_with_its_advice() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(8, 8, [0, 0, 0, 255]));
    let ctx = test_ctx();
    let mut p = panel(backend);
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.phase = Phase::Running { done: 4, total: 10, detail: "Denoising".into() };
    let running = frame(&ctx, &mut p, 1000.0);
    assert!(has(&running, "Cancel") && has(&running, "Denoising (4 of 10)"), "{running:?}");
    assert!(!has(&running, "Run edit"), "the run button gives way to cancel");
    p.phase = Phase::Failed(EditError::classified("CUDA_ERROR_OUT_OF_MEMORY"));
    let failed = frame(&ctx, &mut p, 1000.0);
    for expected in ["The edit failed: CUDA_ERROR_OUT_OF_MEMORY", "ran out of memory", "Try again", "Retry at lower precision"] {
        assert!(has(&failed, expected), "missing {expected:?} in {failed:?}");
    }
    p.phase = Phase::Failed(EditError::new(ErrorKind::ModelLoad, "repo not found"));
    let load = frame(&ctx, &mut p, 1000.0);
    assert!(has(&load, "could not be loaded") && !has(&load, "Retry at lower precision"));
    p.phase = Phase::Idle;
    p.params.prompt.clear();
    let blocked = frame(&ctx, &mut p, 1000.0);
    assert!(has(&blocked, "Run edit"));
    assert_eq!(p.blocked(), Some("Describe the change first"));
}

#[tokio::test]
async fn the_replace_question_names_the_file_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let backend = FakeBackend::new(png_bytes(16, 16, [1, 1, 1, 255]));
    backend.loaded.store(true, Ordering::SeqCst);
    let ctx = test_ctx();
    let mut p = panel(backend);
    p.open_with_path(&source);
    p.params.prompt = "x".into();
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    let id = p.history()[0].id;
    p.ask_replace(id);
    let drawn = frame(&ctx, &mut p, 1000.0);
    for expected in ["Replace the original?", "This overwrites photo.jpg with the edited image.", "Replace photo.jpg", "Keep the original"] {
        assert!(has(&drawn, expected), "missing {expected:?} in {drawn:?}");
    }
}

#[test]
fn the_test_backend_grays_inverts_or_warms_by_prompt() {
    use image::GenericImageView;
    let px = |img: &image::DynamicImage| img.get_pixel(0, 0).0;
    let base = || image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(4, 4, image::Rgba([100, 150, 200, 255])));
    let gray = px(&debug::apply_effect(base(), "Make it BLACK AND WHITE", 0.7));
    assert!(gray[0] == gray[1] && gray[1] == gray[2], "{gray:?}");
    assert_eq!(px(&debug::apply_effect(base(), "invert the colors", 0.7)), [155, 105, 55, 255]);
    let warm = px(&debug::apply_effect(base(), "anything else", 1.0));
    assert!(warm[0] > 100 && warm[2] < 200 && warm[1] == 150, "{warm:?}");
    assert_eq!(px(&debug::apply_effect(base(), "anything else", 0.0)), [100, 150, 200, 255], "no strength, no change");
}

#[tokio::test]
async fn the_test_backend_runs_a_real_edit_end_to_end_and_can_fail_like_a_gpu() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_file(dir.path(), "photo.jpg");
    let ctx = test_ctx();
    let mut p = ImageEditPanel::with_backend(Arc::new(debug::DebugBackend::default()), DbSource::Global, crate::ui::file_table::thumbs::service());
    p.open_with_path(&source);
    p.params.prompt = "Make it black and white".into();
    p.params.steps = 2;
    p.params.preview_every_n = 1;
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    assert_eq!(p.phase(), &Phase::Done);
    assert_eq!(p.history()[0].full, [40, 30], "the result keeps the size of a small source");
    p.params.prompt = format!("{} please", debug::OOM_PROMPT);
    p.start().unwrap();
    settle(&mut p, &ctx, finished).await;
    let Phase::Failed(e) = p.phase().clone() else { panic!("expected a failure") };
    assert_eq!(e.kind, ErrorKind::OutOfMemory);
    assert!(p.history().len() == 1, "a failed edit adds nothing");
    p.use_debug_backend(false);
    assert!(!p.uses_debug_backend());
    p.use_debug_backend(true);
    assert!(p.uses_debug_backend());
}

#[test]
fn the_environment_variable_has_a_stable_name() {
    assert_eq!(debug::ENV, "SMART_MEDIA_IMAGE_EDIT_BACKEND");
}
