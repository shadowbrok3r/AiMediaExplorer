//! Preview player widget: the current frame, play and pause, a seek slider and the elapsed time.

use super::player::{Player, PlayerOptions, PlayerStatus};
use crate::icons::{self, svg};
use egui::{Slider, TextureHandle, TextureOptions, Ui, Vec2};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Redraw interval that keeps an idle player marked as shown.
const KEEP_ALIVE: Duration = Duration::from_millis(1000);

/// Shared handle to a player view kept in egui memory.
pub type Slot = std::sync::Arc<std::sync::Mutex<PlayerView>>;

/// The player view stored under `key`, created on first use.
pub fn slot(ctx: &egui::Context, key: impl std::hash::Hash + std::fmt::Debug) -> Slot {
    let id = egui::Id::new("ame_video_view").with(key);
    ctx.data_mut(|d| d.get_temp_mut_or_insert_with(id, Slot::default).clone())
}

/// UI state for one preview player.
#[derive(Default)]
pub struct PlayerView {
    path: Option<PathBuf>,
    player: Option<Player>,
    texture: Option<TextureHandle>,
    scrub: Option<f32>,
    resume: f64,
}

/// Elapsed or total time as `m:ss`.
pub fn clock(secs: f64) -> String {
    let total = if secs.is_finite() { secs.max(0.0) as u64 } else { 0 };
    format!("{}:{:02}", total / 60, total % 60)
}

/// Size that fits `tex` inside `max` without enlarging it.
pub fn fit(tex: Vec2, max: Vec2) -> Vec2 {
    if tex.x <= 0.0 || tex.y <= 0.0 {
        return Vec2::ZERO;
    }
    let scale = (max.x / tex.x).min(max.y / tex.y).min(1.0).max(0.0);
    tex * scale
}

impl PlayerView {
    /// Stops playback and forgets the current file.
    pub fn reset(&mut self) {
        self.player = None;
        self.texture = None;
        self.scrub = None;
        self.path = None;
        self.resume = 0.0;
    }

    pub fn is_active(&self) -> bool {
        self.player.is_some()
    }

    pub fn status(&self) -> Option<PlayerStatus> {
        self.player.as_ref().map(Player::status)
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Size in pixels of the frame currently shown.
    pub fn frame_size(&self) -> Option<[usize; 2]> {
        self.texture.as_ref().map(TextureHandle::size)
    }

    /// Reports whether the decode thread has exited, usable after the view lets go of the player.
    #[cfg(test)]
    pub(crate) fn watcher(&self) -> Option<impl Fn() -> bool + Send + 'static> {
        self.player.as_ref().map(Player::watcher)
    }

    /// Draws the video area and controls for `path`; `still` draws the picture shown before playback starts.
    pub fn ui(&mut self, ui: &mut Ui, path: &Path, max_size: Vec2, still: impl FnOnce(&mut Ui)) {
        if self.path.as_deref() != Some(path) {
            self.reset();
            self.path = Some(path.to_path_buf());
        }
        if self.player.as_ref().is_some_and(|p| p.status().finished && p.status().error.is_none()) {
            self.player = None;
        }
        if let Some(player) = &self.player {
            player.touch();
            let status = player.status();
            if status.ended {
                self.resume = 0.0;
            } else if status.ready {
                self.resume = status.position_secs;
            }
            ui.ctx().request_repaint_after(KEEP_ALIVE);
            if let Some(frame) = player.latest_frame() {
                match &mut self.texture {
                    Some(tex) => tex.set(frame.image, TextureOptions::LINEAR),
                    None => self.texture = Some(ui.ctx().load_texture(format!("video_preview:{}", path.display()), frame.image, TextureOptions::LINEAR)),
                }
            }
        }
        match &self.texture {
            Some(tex) if self.player.is_some() => {
                ui.image((tex.id(), fit(tex.size_vec2(), max_size)));
            }
            _ => still(ui),
        }
        self.controls(ui, path);
    }

    fn controls(&mut self, ui: &mut Ui, path: &Path) {
        let status = self.player.as_ref().map(Player::status);
        ui.horizontal(|ui| {
            let playing = status.as_ref().is_some_and(|s| s.playing);
            let (icon, tip) = if playing { (svg::PAUSE, "Pause") } else { (svg::PLAY, "Play") };
            if icons::icon_button(ui, icon, tip).clicked() {
                self.toggle(ui, path, playing);
            }
            let Some(status) = &status else { return };
            if let Some(error) = &status.error {
                ui.colored_label(ui.visuals().error_fg_color, format!("Playback failed: {error}"));
                return;
            }
            let position = self.scrub.map_or(status.position_secs as f32, |s| s);
            ui.monospace(match status.duration_secs {
                Some(d) => format!("{} / {}", clock(f64::from(position)), clock(d)),
                None => clock(f64::from(position)),
            });
            if let Some(duration) = status.duration_secs.filter(|d| *d > 0.0) {
                let mut value = position;
                let response = ui.add(Slider::new(&mut value, 0.0..=duration as f32).show_value(false));
                if response.dragged() || response.changed() {
                    self.scrub = Some(value);
                    if let Some(player) = &self.player {
                        player.seek(f64::from(value));
                    }
                }
                if response.drag_stopped() || (!response.dragged() && !response.has_focus()) {
                    self.scrub = None;
                }
            }
        });
    }

    fn toggle(&mut self, ui: &Ui, path: &Path, playing: bool) {
        if self.player.as_ref().is_some_and(|p| p.status().finished) {
            self.player = None;
        }
        match &self.player {
            Some(player) if playing => player.pause(),
            Some(player) => player.play(),
            None => self.start(ui.ctx(), path),
        }
    }

    /// Starts playing `path` from the remembered position.
    pub fn start(&mut self, ctx: &egui::Context, path: &Path) {
        self.start_with(path, PlayerOptions { ctx: Some(ctx.clone()), ..PlayerOptions::default() });
    }

    pub fn start_with(&mut self, path: &Path, opts: PlayerOptions) {
        let player = Player::spawn(path.to_path_buf(), opts);
        if self.resume > 0.0 {
            player.seek(self.resume);
        }
        self.player = Some(player);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::video::fixtures as fx;

    fn frame(ctx: &egui::Context, view: &mut PlayerView, path: &Path, press_play: bool) -> Vec<String> {
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                view.ui(ui, path, egui::vec2(400.0, 300.0), |ui| {
                    ui.label("still picture");
                });
                if press_play {
                    view.toggle(ui, path, false);
                }
            });
        });
        let texts = crate::ui::test_support::texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
        out.drop_without_applying_deltas();
        texts
    }

    #[test]
    fn the_view_shows_the_still_then_plays_and_stops_when_the_file_changes() {
        if !fx::available() {
            return fx::skip("the_view_shows_the_still");
        }
        let dir = tempfile::tempdir().unwrap();
        let (Some(a), Some(b)) = (fx::h264_mp4(dir.path(), "a.mp4", 3), fx::h264_mp4(dir.path(), "b.mp4", 3)) else { return fx::skip("view clips") };
        let ctx = crate::ui::test_support::test_ctx();
        let mut view = PlayerView::default();
        assert!(frame(&ctx, &mut view, &a, false).iter().any(|t| t == "still picture"));
        assert!(!view.is_active() && view.path() == Some(a.as_path()));

        frame(&ctx, &mut view, &a, true);
        assert!(view.is_active());
        let end = std::time::Instant::now() + Duration::from_secs(15);
        while view.texture.is_none() && std::time::Instant::now() < end {
            frame(&ctx, &mut view, &a, false);
            std::thread::sleep(Duration::from_millis(10));
        }
        let tex = view.texture.as_ref().expect("a decoded frame reached the texture");
        assert_eq!(tex.size(), [320, 180]);
        let texts = frame(&ctx, &mut view, &a, false);
        assert!(!texts.iter().any(|t| t == "still picture"), "the video replaces the still: {texts:?}");
        assert!(texts.iter().any(|t| t.contains(" / 0:03")), "elapsed and total time are shown: {texts:?}");

        let done = view.player.as_ref().unwrap().watcher();
        assert!(!done());
        assert!(frame(&ctx, &mut view, &b, false).iter().any(|t| t == "still picture"));
        assert!(!view.is_active() && view.texture.is_none() && view.path() == Some(b.as_path()));
        let end = std::time::Instant::now() + Duration::from_secs(3);
        while !done() && std::time::Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(done(), "changing the file must stop the old decode thread");
    }

    #[test]
    fn clock_formats_minutes_and_seconds() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(65.9), "1:05");
        assert_eq!(clock(3600.0), "60:00");
        assert_eq!(clock(-5.0), "0:00");
        assert_eq!(clock(f64::NAN), "0:00");
    }

    #[test]
    fn fit_shrinks_but_never_enlarges() {
        assert_eq!(fit(Vec2::new(1280.0, 720.0), Vec2::new(640.0, 640.0)), Vec2::new(640.0, 360.0));
        assert_eq!(fit(Vec2::new(100.0, 50.0), Vec2::new(640.0, 640.0)), Vec2::new(100.0, 50.0));
        assert_eq!(fit(Vec2::new(0.0, 50.0), Vec2::new(640.0, 640.0)), Vec2::ZERO);
        assert_eq!(fit(Vec2::new(100.0, 50.0), Vec2::new(-1.0, 640.0)), Vec2::ZERO);
    }
}
