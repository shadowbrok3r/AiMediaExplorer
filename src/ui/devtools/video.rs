//! Devtools window for the video subsystem: backends, probe, thumbnail generation, decode slots and a preview player.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, ComboBox, Context, DragValue, Grid, Id, RichText, ScrollArea, TextEdit, TextureHandle, TextureOptions, Ui, Window};
use humansize::{BINARY, format_size};

use crate::icons::{self, svg};
use crate::utilities::thumbs::{EncodedThumb, VideoThumbReport, generate_video_thumb_with};
use crate::video::{self, BackendChoice, Cancel, PosterOptions, ProbeOptions, VideoInfo};

const HISTORY: usize = 40;

fn open_id() -> Id {
    Id::new("ame_devtools_video_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// Result of one thumbnail run.
#[derive(Clone)]
pub struct ThumbOutcome {
    pub id: u64,
    pub thumb: EncodedThumb,
    pub report: VideoThumbReport,
    pub total: Duration,
}

#[derive(Default)]
pub struct LabState {
    pub path: String,
    pub backend: BackendChoice,
    pub busy: Option<&'static str>,
    pub probe: Option<Result<VideoInfo, String>>,
    pub probe_time: Option<Duration>,
    pub thumb: Option<Result<ThumbOutcome, String>>,
    pub history: Vec<String>,
    pub picked: Option<String>,
    runs: u64,
    cancel: Option<Cancel>,
}

/// Background runner behind the window; tests drive it without a UI.
#[derive(Clone, Default)]
pub struct Lab {
    state: Arc<Mutex<LabState>>,
    ctx: Arc<Mutex<Option<Context>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Lab {
    pub fn state(&self) -> MutexGuard<'_, LabState> {
        lock(&self.state)
    }

    fn repaint(&self) {
        if let Some(ctx) = lock(&self.ctx).as_ref() {
            ctx.request_repaint();
        }
    }

    pub fn set_path(&self, path: impl Into<String>) {
        self.state().path = path.into();
    }

    pub fn is_busy(&self) -> bool {
        self.state().busy.is_some()
    }

    /// Waits until no job runs; false when `timeout` passed first.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let end = Instant::now() + timeout;
        while self.is_busy() {
            if Instant::now() >= end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// Raises the cancel flag of the running job.
    pub fn cancel(&self) {
        if let Some(c) = &self.state().cancel {
            c.cancel();
        }
    }

    fn begin(&self, what: &'static str) -> Option<(PathBuf, BackendChoice, Cancel)> {
        let mut st = self.state();
        if st.busy.is_some() || st.path.trim().is_empty() {
            return None;
        }
        let cancel = Cancel::new();
        st.busy = Some(what);
        st.cancel = Some(cancel.clone());
        Some((PathBuf::from(st.path.trim()), st.backend, cancel))
    }

    fn finish(&self, line: String) {
        let mut st = self.state();
        st.busy = None;
        st.cancel = None;
        st.history.push(line);
        let extra = st.history.len().saturating_sub(HISTORY);
        st.history.drain(..extra);
        drop(st);
        self.repaint();
    }

    /// Runs the probe on a worker thread.
    pub fn start_probe(&self) -> bool {
        let Some((path, backend, cancel)) = self.begin("probing") else { return false };
        let lab = self.clone();
        std::thread::spawn(move || {
            let started = Instant::now();
            let result = video::probe_with(&path, &ProbeOptions { backend, cancel, ..Default::default() });
            let elapsed = started.elapsed();
            let line = match &result {
                Ok(info) => format!("probe ok in {} ms: {}", elapsed.as_millis(), info.summary()),
                Err(e) => format!("probe failed in {} ms: {e}", elapsed.as_millis()),
            };
            let mut st = lab.state();
            st.probe = Some(result.map_err(|e| e.to_string()));
            st.probe_time = Some(elapsed);
            drop(st);
            lab.finish(line);
        });
        true
    }

    /// Runs thumbnail generation through the same entry point the scanner uses.
    pub fn start_thumb(&self) -> bool {
        let Some((path, backend, cancel)) = self.begin("generating thumbnail") else { return false };
        let lab = self.clone();
        std::thread::spawn(move || {
            let started = Instant::now();
            let opts = PosterOptions { max_dim: crate::utilities::thumbs::THUMB_MAX_DIM, backend, cancel, ..Default::default() };
            let result = generate_video_thumb_with(&path, &opts);
            let total = started.elapsed();
            let (line, outcome) = {
                let mut st = lab.state();
                st.runs += 1;
                let id = st.runs;
                match result {
                    Ok((thumb, report)) => {
                        let kind = report.placeholder.map_or("frame".to_string(), |k| format!("{k:?} placeholder"));
                        (format!("thumbnail ok in {} ms: {kind}, {}x{}, {} bytes", total.as_millis(), thumb.width, thumb.height, thumb.bytes.len()), Ok(ThumbOutcome { id, thumb, report, total }))
                    }
                    Err(e) => (format!("thumbnail failed in {} ms: {e}", total.as_millis()), Err(e)),
                }
            };
            lab.state().thumb = Some(outcome);
            lab.finish(line);
        });
        true
    }

    fn browse(&self) {
        let lab = self.clone();
        std::thread::spawn(move || {
            let exts: Vec<&str> = crate::utilities::types::VIDEO_EXTS.iter().copied().collect();
            let picked = rfd::FileDialog::new().add_filter("Video", &exts).pick_file();
            if let Some(p) = picked {
                lab.state().picked = Some(p.display().to_string());
                lab.repaint();
            }
        });
    }
}

static LAB: LazyLock<Lab> = LazyLock::new(Lab::default);

fn backend_label(b: BackendChoice) -> &'static str {
    match b {
        BackendChoice::Auto => "Auto",
        BackendChoice::Native => "In-process ffmpeg",
        BackendChoice::Cli => "ffmpeg command",
        BackendChoice::Disabled => "None (placeholder)",
    }
}

fn texture_of(ctx: &Context, thumb: &EncodedThumb, id: u64) -> Option<TextureHandle> {
    let img = image::load_from_memory(&thumb.bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let color = ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    Some(ctx.load_texture(format!("video_lab_thumb_{id}"), color, TextureOptions::LINEAR))
}

#[derive(Default)]
struct UiState {
    texture: Option<(u64, TextureHandle)>,
    #[cfg(ame_ffmpeg)]
    view: crate::video::view::PlayerView,
}

static UI: LazyLock<Mutex<UiState>> = LazyLock::new(|| Mutex::new(UiState::default()));

/// Draws the window when it is open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    *lock(&LAB.ctx) = Some(ctx.clone());
    Window::new("Devtools: Video").open(&mut open).default_width(560.0).default_height(680.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| body(ui, &LAB));
    });
    set_open(ctx, open);
}

/// Window contents for `lab`.
pub fn body(ui: &mut Ui, lab: &Lab) {
    *lock(&lab.ctx) = Some(ui.ctx().clone());
    if let Some(p) = lab.state().picked.take() {
        lab.set_path(p);
    }
    if lab.is_busy() {
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }
    backend_section(ui);
    ui.separator();
    input_section(ui, lab);
    ui.separator();
    probe_section(ui, lab);
    ui.separator();
    thumb_section(ui, lab);
    #[cfg(ame_ffmpeg)]
    {
        ui.separator();
        player_section(ui, lab);
    }
    ui.separator();
    history_section(ui, lab);
}

fn backend_section(ui: &mut Ui) {
    icons::heading(ui, svg::VIDEO_CAMERA, "Backends");
    ui.label(video::backend_summary());
    let gate = video::gate::global();
    ui.horizontal(|ui| {
        ui.label("Concurrent decodes");
        let mut limit = gate.limit();
        if ui.add(DragValue::new(&mut limit).range(1..=16)).changed() {
            gate.set_limit(limit);
        }
        ui.monospace(format!("running {}  waiting {}", gate.in_flight(), gate.waiting()));
    });
    let c = video::gate::counters().snapshot();
    let avg = if c.ok > 0 { c.total_micros as f64 / c.ok as f64 / 1000.0 } else { 0.0 };
    ui.monospace(format!("frames {}  placeholders {}  failed {}  timeouts {}  blank retries {}", c.ok, c.placeholders, c.failed, c.timeouts, c.blank_retries));
    ui.monospace(format!("avg {avg:.1} ms  last {:.1} ms", c.last_micros as f64 / 1000.0));
}

fn input_section(ui: &mut Ui, lab: &Lab) {
    icons::heading(ui, svg::FILE_VIDEO, "File");
    let busy = lab.is_busy();
    ui.horizontal(|ui| {
        let mut path = lab.state().path.clone();
        if ui.add(TextEdit::singleline(&mut path).desired_width(ui.available_width() - 90.0).hint_text("Path to a video file")).changed() {
            lab.set_path(path);
        }
        if icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked() {
            lab.browse();
        }
    });
    ui.horizontal(|ui| {
        let mut backend = lab.state().backend;
        ComboBox::from_id_salt("video_lab_backend").selected_text(backend_label(backend)).show_ui(ui, |ui| {
            for b in [BackendChoice::Auto, BackendChoice::Native, BackendChoice::Cli, BackendChoice::Disabled] {
                ui.selectable_value(&mut backend, b, backend_label(b));
            }
        });
        lab.state().backend = backend;
        let has_path = !lab.state().path.trim().is_empty();
        if ui.add_enabled(!busy && has_path, egui::Button::new("Probe")).clicked() {
            lab.start_probe();
        }
        if ui.add_enabled(!busy && has_path, egui::Button::new("Thumbnail")).clicked() {
            lab.start_thumb();
        }
        if ui.add_enabled(busy, egui::Button::new("Cancel")).clicked() {
            lab.cancel();
        }
        if let Some(what) = lab.state().busy {
            ui.spinner();
            ui.label(what);
        }
    });
}

fn row(ui: &mut Ui, label: &str, value: impl Into<String>) {
    ui.label(label);
    ui.monospace(value.into());
    ui.end_row();
}

fn opt<T: ToString>(v: Option<T>) -> String {
    v.map_or_else(|| "-".to_string(), |v| v.to_string())
}

fn probe_section(ui: &mut Ui, lab: &Lab) {
    icons::heading(ui, svg::INFO, "Probe");
    let (result, time) = {
        let st = lab.state();
        (st.probe.clone(), st.probe_time)
    };
    match result {
        None => {
            ui.weak("No probe yet");
        }
        Some(Err(e)) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        Some(Ok(info)) => {
            Grid::new("video_lab_probe").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
                row(ui, "Time", time.map_or("-".into(), |t| format!("{} ms", t.as_millis())));
                row(ui, "Container", opt(info.container.clone()));
                row(ui, "Size", opt(info.size_bytes.map(|b| format_size(b, BINARY))));
                row(ui, "Duration", opt(info.duration_secs.map(|d| format!("{d:.3} s"))));
                row(ui, "Bitrate", opt(info.bitrate.map(|b| format!("{} kb/s", b / 1000))));
                row(ui, "Created", opt(info.creation_time.clone()));
                row(ui, "Video", if info.has_video { opt(info.video_codec.clone()) } else { "none".into() });
                let (dw, dh) = info.display_size();
                row(ui, "Coded size", format!("{}x{}", info.width, info.height));
                row(ui, "Display size", format!("{dw}x{dh}"));
                row(ui, "Pixel aspect", info.sar.map_or("square".into(), |(n, d)| format!("{n}:{d}")));
                row(ui, "Rotation", format!("{} deg clockwise", info.rotation));
                row(ui, "Frame rate", opt(info.fps.map(|f| format!("{f:.3}"))));
                row(ui, "Pixel format", opt(info.pixel_format.clone()));
                row(ui, "Video bitrate", opt(info.video_bitrate.map(|b| format!("{} kb/s", b / 1000))));
                row(ui, "Subtitle tracks", info.subtitle_tracks.to_string());
                for a in &info.audio {
                    row(
                        ui,
                        &format!("Audio #{}", a.index),
                        format!("{} {} ch {} Hz {}", opt(a.codec.clone()), opt(a.channels), opt(a.sample_rate), a.language.clone().unwrap_or_default()),
                    );
                }
            });
        }
    }
}

fn thumb_section(ui: &mut Ui, lab: &Lab) {
    icons::heading(ui, svg::IMAGE, "Thumbnail");
    let result = lab.state().thumb.clone();
    match result {
        None => {
            ui.weak("No thumbnail yet");
        }
        Some(Err(e)) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        Some(Ok(out)) => {
            let ctx = ui.ctx().clone();
            let mut ui_state = lock(&UI);
            if ui_state.texture.as_ref().map(|(id, _)| *id) != Some(out.id) {
                ui_state.texture = texture_of(&ctx, &out.thumb, out.id).map(|t| (out.id, t));
            }
            if let Some((_, tex)) = &ui_state.texture {
                ui.image((tex.id(), tex.size_vec2()));
            }
            drop(ui_state);
            let r = &out.report;
            Grid::new("video_lab_thumb").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
                row(ui, "Total time", format!("{} ms", out.total.as_millis()));
                row(ui, "Decode time", format!("{} ms", r.elapsed.as_millis()));
                row(ui, "Encoded", format!("{:?} {}x{} {} bytes", out.thumb.format, out.thumb.width, out.thumb.height, out.thumb.bytes.len()));
                row(ui, "Backend", r.backend.map_or("none".into(), |b| b.label().to_string()));
                row(ui, "Placeholder", r.placeholder.map_or("no".into(), |k| format!("{k:?}")));
                row(ui, "Attempts", r.attempts.to_string());
                row(ui, "Frame position", opt(r.position_secs.map(|p| format!("{p:.2} s"))));
                row(ui, "Blank frame", if r.blank { "yes, best of the attempts" } else { "no" });
                row(ui, "Source size", format!("{}x{} rotated {}", r.source.0, r.source.1, r.rotation));
            });
        }
    }
}

#[cfg(ame_ffmpeg)]
fn player_section(ui: &mut Ui, lab: &Lab) {
    icons::heading(ui, svg::PLAY, "Preview player");
    let path = lab.state().path.trim().to_string();
    if path.is_empty() {
        ui.weak("Enter a path to play it");
        return;
    }
    let mut ui_state = lock(&UI);
    let max = egui::vec2(ui.available_width().min(520.0), 320.0);
    ui_state.view.ui(ui, Path::new(&path), max, |ui| {
        ui.weak("Press play to decode");
    });
    if let Some(s) = ui_state.view.status() {
        ui.monospace(format!("decoded {}  shown {}  dropped {}  size {}x{}", s.decoded, s.sent, s.dropped, s.size.0, s.size.1));
    }
}

fn history_section(ui: &mut Ui, lab: &Lab) {
    icons::heading(ui, svg::LIST, "Log");
    let lines = lab.state().history.clone();
    if lines.is_empty() {
        ui.weak("Nothing run yet");
    }
    for line in lines.iter().rev() {
        ui.label(RichText::new(line).monospace().color(Color32::GRAY));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::fixtures;

    fn frame(ctx: &Context) {
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(900.0, 1200.0))), ..Default::default() };
        ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |_| {});
            show(ui.ctx());
        })
        .drop_without_applying_deltas();
    }

    #[test]
    fn window_draws_and_toggles() {
        let ctx = crate::ui::test_support::test_ctx();
        frame(&ctx);
        assert!(!is_open(&ctx));
        toggle(&ctx);
        frame(&ctx);
        frame(&ctx);
        assert!(is_open(&ctx));
        toggle(&ctx);
        assert!(!is_open(&ctx));
    }

    #[test]
    fn lab_rejects_empty_paths_and_reports_missing_files() {
        let lab = Lab::default();
        assert!(!lab.start_probe());
        lab.set_path("/definitely/not/here.mp4");
        assert!(lab.start_probe());
        assert!(lab.wait_idle(Duration::from_secs(20)));
        assert!(matches!(lab.state().probe, Some(Err(_))));
        assert!(lab.start_thumb());
        assert!(lab.wait_idle(Duration::from_secs(20)));
        assert!(matches!(lab.state().thumb, Some(Err(_))));
        assert_eq!(lab.state().history.len(), 2);
    }

    #[test]
    fn lab_probes_and_thumbnails_a_generated_clip_and_the_window_shows_it() {
        if !fixtures::available() {
            return fixtures::skip("lab_probes_and_thumbnails_a_generated_clip");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fixtures::h264_mp4(dir.path(), "lab.mp4", 3) else { return fixtures::skip("lab clip") };
        let lab = Lab::default();
        lab.set_path(clip.display().to_string());
        assert!(lab.start_probe());
        assert!(!lab.start_thumb(), "a second job must be refused while one runs");
        assert!(lab.wait_idle(Duration::from_secs(30)));
        assert!(lab.start_thumb());
        assert!(lab.wait_idle(Duration::from_secs(30)));
        {
            let st = lab.state();
            let info = st.probe.clone().unwrap().unwrap();
            assert!(info.has_video && info.audio.len() == 1);
            let out = st.thumb.clone().unwrap().unwrap();
            assert!(out.thumb.width <= 256 && out.thumb.height <= 256);
            assert!(out.report.placeholder.is_none());
        }
        let ctx = crate::ui::test_support::test_ctx();
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(900.0, 1400.0))), ..Default::default() };
        let mut drawn = Vec::new();
        for _ in 0..2 {
            let out = ctx.run_ui(input.clone(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| body(ui, &lab));
            });
            drawn = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
        }
        let has = |needle: &str| drawn.iter().any(|(t, _)| t.contains(needle));
        assert!(has("Container") && has("Display size") && has("Backend"), "{drawn:?}");
    }
}
