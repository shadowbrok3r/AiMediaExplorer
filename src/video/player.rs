//! Muted preview playback: a decode thread paced by timestamps that hands scaled frames to the UI over a bounded channel.

use super::info;
use super::native::{self, Scaler};
use super::{Cancel, Deadline, VideoError, check_file};
use crossbeam::channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, bounded, unbounded};
use egui::{Color32, ColorImage};
use ffmpeg_next::software::scaling;
use ffmpeg_next::{decoder, frame};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const FRAME_QUEUE: usize = 2;
const IDLE_POLL: Duration = Duration::from_millis(30);
const SLEEP_SLICE: Duration = Duration::from_millis(15);
const SEEK_SETTLE_FRAMES: u32 = 4000;

#[derive(Clone)]
pub struct PlayerOptions {
    /// Longest side of the decoded frames.
    pub max_dim: u32,
    pub autoplay: bool,
    /// Decoder threads; 0 lets ffmpeg choose.
    pub threads: usize,
    /// Repaint target woken whenever a frame is ready.
    pub ctx: Option<egui::Context>,
    /// The decode thread ends when the UI has not touched the player for this long.
    pub idle_timeout: Duration,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self { max_dim: 1280, autoplay: true, threads: 2, ctx: None, idle_timeout: Duration::from_secs(3) }
    }
}

/// One decoded, upright, scaled frame.
pub struct Frame {
    pub image: ColorImage,
    pub pts_secs: f64,
}

enum Command {
    Play,
    Pause,
    Seek(f64),
}

struct Shared {
    cancel: Cancel,
    epoch: Instant,
    playing: AtomicBool,
    ended: AtomicBool,
    ready: AtomicBool,
    finished: AtomicBool,
    position_bits: AtomicU64,
    duration_bits: AtomicU64,
    width: AtomicU32,
    height: AtomicU32,
    last_seen_ms: AtomicU64,
    decoded: AtomicU64,
    sent: AtomicU64,
    dropped: AtomicU64,
    seeks: AtomicU64,
    error: Mutex<Option<String>>,
}

impl Shared {
    fn new() -> Self {
        Self {
            cancel: Cancel::new(),
            epoch: Instant::now(),
            playing: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            ready: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            position_bits: AtomicU64::new(0f64.to_bits()),
            duration_bits: AtomicU64::new(f64::NAN.to_bits()),
            width: AtomicU32::new(0),
            height: AtomicU32::new(0),
            last_seen_ms: AtomicU64::new(0),
            decoded: AtomicU64::new(0),
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            seeks: AtomicU64::new(0),
            error: Mutex::new(None),
        }
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn touch(&self) {
        self.last_seen_ms.store(self.now_ms(), Ordering::Relaxed);
    }

    fn stale(&self, timeout: Duration) -> bool {
        self.now_ms().saturating_sub(self.last_seen_ms.load(Ordering::Relaxed)) > timeout.as_millis() as u64
    }

    fn should_stop(&self, timeout: Duration) -> bool {
        self.cancel.is_cancelled() || self.stale(timeout)
    }

    fn set_error(&self, message: String) {
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
    }
}

/// Snapshot of a player for drawing controls.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerStatus {
    pub playing: bool,
    pub ended: bool,
    pub ready: bool,
    pub finished: bool,
    pub position_secs: f64,
    pub duration_secs: Option<f64>,
    pub size: (u32, u32),
    pub error: Option<String>,
    pub decoded: u64,
    pub sent: u64,
    pub dropped: u64,
}

/// Handle to a playback thread; dropping it stops the thread.
pub struct Player {
    shared: Arc<Shared>,
    cmd_tx: Sender<Command>,
    frame_rx: Receiver<Frame>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Player {
    /// Starts decoding `path` on a new thread.
    pub fn spawn(path: PathBuf, opts: PlayerOptions) -> Self {
        let shared = Arc::new(Shared::new());
        shared.touch();
        let (cmd_tx, cmd_rx) = unbounded();
        let (frame_tx, frame_rx) = bounded(FRAME_QUEUE);
        let thread_shared = shared.clone();
        let handle = std::thread::Builder::new().name("video-preview".into()).spawn(move || {
            let outcome = crate::utilities::guard::guarded(|| run(&path, &thread_shared, &cmd_rx, &frame_tx, &opts), |msg| VideoError::Backend(format!("playback panicked: {msg}")));
            if let Err(e) = outcome {
                if !matches!(e, VideoError::Cancelled) {
                    log::debug!("[video] playback ended with error: {e}");
                    thread_shared.set_error(e.to_string());
                }
            }
            thread_shared.playing.store(false, Ordering::SeqCst);
            thread_shared.finished.store(true, Ordering::SeqCst);
            if let Some(ctx) = &opts.ctx {
                ctx.request_repaint();
            }
        });
        let handle = match handle {
            Ok(h) => Some(h),
            Err(e) => {
                shared.set_error(format!("could not start playback thread: {e}"));
                shared.finished.store(true, Ordering::SeqCst);
                None
            }
        };
        Self { shared, cmd_tx, frame_rx, handle: Mutex::new(handle) }
    }

    pub fn play(&self) {
        let _ = self.cmd_tx.send(Command::Play);
    }

    pub fn pause(&self) {
        let _ = self.cmd_tx.send(Command::Pause);
    }

    pub fn seek(&self, secs: f64) {
        let _ = self.cmd_tx.send(Command::Seek(secs));
    }

    /// Tells the thread the UI is still showing it.
    pub fn touch(&self) {
        self.shared.touch();
    }

    /// Asks the thread to end without waiting for it.
    pub fn stop(&self) {
        self.shared.cancel.cancel();
    }

    /// Newest queued frame; older ones are discarded.
    pub fn latest_frame(&self) -> Option<Frame> {
        let mut latest = None;
        while let Ok(f) = self.frame_rx.try_recv() {
            latest = Some(f);
        }
        latest
    }

    pub fn status(&self) -> PlayerStatus {
        let s = &self.shared;
        let duration = f64::from_bits(s.duration_bits.load(Ordering::Relaxed));
        PlayerStatus {
            playing: s.playing.load(Ordering::SeqCst),
            ended: s.ended.load(Ordering::SeqCst),
            ready: s.ready.load(Ordering::SeqCst),
            finished: s.finished.load(Ordering::SeqCst),
            position_secs: f64::from_bits(s.position_bits.load(Ordering::Relaxed)),
            duration_secs: duration.is_finite().then_some(duration),
            size: (s.width.load(Ordering::Relaxed), s.height.load(Ordering::Relaxed)),
            error: s.error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            decoded: s.decoded.load(Ordering::Relaxed),
            sent: s.sent.load(Ordering::Relaxed),
            dropped: s.dropped.load(Ordering::Relaxed),
        }
    }

    /// Waits for the thread to exit; false when it is still running after `timeout`.
    pub fn join(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.shared.finished.load(Ordering::SeqCst) {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        if let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
        true
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.shared.cancel.cancel();
    }
}

#[cfg(test)]
impl Player {
    /// Reports whether the decode thread has exited, usable after the player is dropped.
    pub(crate) fn watcher(&self) -> impl Fn() -> bool + Send + 'static {
        let shared = self.shared.clone();
        move || shared.finished.load(Ordering::SeqCst)
    }
}

/// Playback control state owned by the decode thread.
struct Control {
    playing: bool,
    ended: bool,
    seek: Option<f64>,
    show_one: bool,
    anchor: Option<(Instant, f64)>,
}

impl Control {
    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::Play => {
                if self.ended {
                    self.seek = Some(0.0);
                }
                self.playing = true;
                self.ended = false;
                self.anchor = None;
            }
            Command::Pause => {
                self.playing = false;
                self.anchor = None;
            }
            Command::Seek(t) => {
                self.seek = Some(t);
                self.show_one = true;
                self.ended = false;
                self.anchor = None;
            }
        }
    }
}

struct Source {
    ictx: ffmpeg_next::format::context::Input,
    decoder: decoder::Video,
    index: usize,
    time_base: f64,
    start_secs: f64,
}

/// Stream facts the playback loop needs.
struct Lite {
    sar: Option<(u32, u32)>,
    rotation: u32,
    fps: Option<f64>,
    duration_secs: Option<f64>,
}

fn open_source(path: &Path, threads: usize, deadline: &Deadline) -> Result<(Source, Lite), VideoError> {
    let size = check_file(path)?;
    let ictx = native::open(path, deadline)?;
    let info = native::info_from_context(&ictx, size);
    let index = info.video_index.ok_or(VideoError::NoVideoStream)?;
    let decoder = native::open_decoder(&ictx, index, threads, deadline)?;
    let stream = ictx.stream(index).ok_or(VideoError::NoVideoStream)?;
    let tb = stream.time_base();
    let time_base = if tb.denominator() > 0 { f64::from(tb.numerator()) / f64::from(tb.denominator()) } else { 0.0 };
    let start = stream.start_time();
    let start_secs = if start == i64::MIN { 0.0 } else { start as f64 * time_base };
    let lite = Lite { sar: info.sar, rotation: info.rotation, fps: info.fps, duration_secs: info.duration_secs };
    Ok((Source { ictx, decoder, index, time_base, start_secs }, lite))
}

fn frame_secs(f: &frame::Video, src: &Source) -> Option<f64> {
    f.timestamp().map(|ts| ts as f64 * src.time_base - src.start_secs)
}

fn seek_source(src: &mut Source, secs: f64, frame_dur: f64, decoded: &mut frame::Video, deadline: &Deadline) -> Result<bool, VideoError> {
    let target = (secs.max(0.0) + src.start_secs).max(0.0);
    let micros = (target * 1_000_000.0) as i64;
    if src.ictx.seek(micros, ..micros).is_err() {
        return Ok(false);
    }
    src.decoder.flush();
    for _ in 0..SEEK_SETTLE_FRAMES {
        if !native::next_frame(&mut src.ictx, &mut src.decoder, src.index, decoded, deadline)? {
            return Ok(false);
        }
        if frame_secs(decoded, src).is_none_or(|pts| pts >= secs - frame_dur * 0.5) {
            return Ok(true);
        }
    }
    Ok(true)
}

/// Packs RGB24 rows into an image, rotated clockwise by `rotation` degrees.
pub fn color_image_from_rgb(data: &[u8], stride: usize, w: usize, h: usize, rotation: u32) -> ColorImage {
    let (dw, dh) = if rotation % 180 == 90 { (h, w) } else { (w, h) };
    let mut pixels = vec![Color32::BLACK; dw * dh];
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * 3];
        for (x, px) in row.chunks_exact(3).enumerate() {
            let (dx, dy) = match rotation % 360 {
                90 => (h - 1 - y, x),
                180 => (w - 1 - x, h - 1 - y),
                270 => (y, w - 1 - x),
                _ => (x, y),
            };
            pixels[dy * dw + dx] = Color32::from_rgb(px[0], px[1], px[2]);
        }
    }
    ColorImage::new([dw, dh], pixels)
}

fn convert(decoded: &frame::Video, lite: &Lite, scaler: &mut Scaler, max_dim: u32) -> Result<ColorImage, VideoError> {
    let sar = native::frame_sar(decoded, lite.sar);
    let geo = info::thumb_geometry(decoded.width(), decoded.height(), sar, lite.rotation, max_dim).ok_or_else(|| VideoError::Corrupt("frame has no size".into()))?;
    let out = scaler.scale(decoded, geo.scaled_w, geo.scaled_h, scaling::Flags::FAST_BILINEAR)?;
    if out.stride(0) < geo.scaled_w as usize * 3 || out.data(0).len() < out.stride(0) * (geo.scaled_h as usize - 1) + geo.scaled_w as usize * 3 {
        return Err(VideoError::Corrupt("scaled frame is smaller than expected".into()));
    }
    Ok(color_image_from_rgb(out.data(0), out.stride(0), geo.scaled_w as usize, geo.scaled_h as usize, geo.rotation))
}

fn run(path: &Path, shared: &Shared, cmd_rx: &Receiver<Command>, frame_tx: &Sender<Frame>, opts: &PlayerOptions) -> Result<(), VideoError> {
    let deadline = Deadline::unlimited(shared.cancel.clone());
    let (mut src, lite) = open_source(path, opts.threads, &deadline)?;
    let fps = lite.fps.unwrap_or(30.0).clamp(1.0, 240.0);
    let frame_dur = 1.0 / fps;
    let duration = lite.duration_secs;
    if let Some(d) = duration {
        shared.duration_bits.store(d.to_bits(), Ordering::Relaxed);
    }
    let mut scaler = Scaler::default();
    let mut decoded = frame::Video::empty();
    let mut ctl = Control { playing: opts.autoplay, ended: false, seek: None, show_one: true, anchor: None };
    let mut have_frame = false;
    let mut last_pts = 0.0f64;
    shared.ready.store(true, Ordering::SeqCst);
    loop {
        if shared.should_stop(opts.idle_timeout) {
            return Ok(());
        }
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => ctl.apply(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(()),
            }
        }
        if let Some(t) = ctl.seek.take() {
            let t = duration.map_or(t.max(0.0), |d| t.clamp(0.0, (d - frame_dur * 1.5).max(0.0)));
            shared.seeks.fetch_add(1, Ordering::Relaxed);
            have_frame = seek_source(&mut src, t, frame_dur, &mut decoded, &deadline)?;
            if !have_frame && t <= 0.0 {
                (src, _) = open_source(path, opts.threads, &deadline)?;
                have_frame = false;
            }
            ctl.show_one = true;
            last_pts = t;
        }
        shared.playing.store(ctl.playing && !ctl.ended, Ordering::SeqCst);
        if (!ctl.playing || ctl.ended) && !ctl.show_one {
            match cmd_rx.recv_timeout(IDLE_POLL) {
                Ok(cmd) => ctl.apply(cmd),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            continue;
        }
        if !have_frame {
            if !native::next_frame(&mut src.ictx, &mut src.decoder, src.index, &mut decoded, &deadline)? {
                ctl.ended = true;
                ctl.playing = false;
                ctl.show_one = false;
                shared.ended.store(true, Ordering::SeqCst);
                shared.playing.store(false, Ordering::SeqCst);
                shared.position_bits.store(duration.unwrap_or(last_pts).to_bits(), Ordering::Relaxed);
                wake(opts);
                continue;
            }
            shared.decoded.fetch_add(1, Ordering::Relaxed);
        }
        have_frame = false;
        shared.ended.store(false, Ordering::SeqCst);
        let pts = frame_secs(&decoded, &src).unwrap_or(last_pts + frame_dur);
        last_pts = pts;
        if ctl.playing {
            match pace(&mut ctl, pts, frame_dur, shared, cmd_rx, opts) {
                Pace::Show => {}
                Pace::Drop => {
                    if ctl.seek.is_none() {
                        shared.dropped.fetch_add(1, Ordering::Relaxed);
                    }
                    continue;
                }
                Pace::Stop => return Ok(()),
            }
        }
        let image = convert(&decoded, &lite, &mut scaler, opts.max_dim)?;
        if shared.width.load(Ordering::Relaxed) == 0 {
            shared.width.store(image.size[0] as u32, Ordering::Relaxed);
            shared.height.store(image.size[1] as u32, Ordering::Relaxed);
        }
        if !send_frame(Frame { image, pts_secs: pts }, frame_tx, shared, opts) {
            return Ok(());
        }
        shared.position_bits.store(pts.to_bits(), Ordering::Relaxed);
        shared.sent.fetch_add(1, Ordering::Relaxed);
        ctl.show_one = false;
        wake(opts);
    }
}

fn wake(opts: &PlayerOptions) {
    if let Some(ctx) = &opts.ctx {
        ctx.request_repaint();
    }
}

enum Pace {
    Show,
    Drop,
    Stop,
}

/// Sleeps until the frame is due and decides whether it is still worth showing.
fn pace(ctl: &mut Control, pts: f64, frame_dur: f64, shared: &Shared, cmd_rx: &Receiver<Command>, opts: &PlayerOptions) -> Pace {
    let (t0, p0) = *ctl.anchor.get_or_insert((Instant::now(), pts));
    let due = t0 + Duration::from_secs_f64((pts - p0).max(0.0));
    loop {
        let now = Instant::now();
        if now >= due {
            let late = now - due;
            if late > Duration::from_secs(1) {
                ctl.anchor = Some((now, pts));
                return Pace::Show;
            }
            return if late <= Duration::from_secs_f64((frame_dur * 2.0).max(0.08)) { Pace::Show } else { Pace::Drop };
        }
        if shared.should_stop(opts.idle_timeout) {
            return Pace::Stop;
        }
        if let Ok(cmd) = cmd_rx.try_recv() {
            ctl.apply(cmd);
            return if ctl.seek.is_some() { Pace::Drop } else { Pace::Show };
        }
        std::thread::sleep((due - now).min(SLEEP_SLICE));
    }
}

/// Queues a frame, waiting while the UI lags; false when playback should end.
fn send_frame(mut frame: Frame, tx: &Sender<Frame>, shared: &Shared, opts: &PlayerOptions) -> bool {
    loop {
        match tx.send_timeout(frame, Duration::from_millis(40)) {
            Ok(()) => return true,
            Err(SendTimeoutError::Disconnected(_)) => return false,
            Err(SendTimeoutError::Timeout(back)) => {
                if shared.should_stop(opts.idle_timeout) {
                    return false;
                }
                frame = back;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize, stride: usize) -> Vec<u8> {
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                data[y * stride + x * 3..y * stride + x * 3 + 3].copy_from_slice(&[x as u8, y as u8, 7]);
            }
        }
        data
    }

    #[test]
    fn rows_are_unpadded_and_unrotated() {
        let data = gradient(4, 2, 16);
        let img = color_image_from_rgb(&data, 16, 4, 2, 0);
        assert_eq!(img.size, [4, 2]);
        assert_eq!(img.pixels[0], Color32::from_rgb(0, 0, 7));
        assert_eq!(img.pixels[5], Color32::from_rgb(1, 1, 7));
    }

    #[test]
    fn rotations_move_the_corner_pixel_like_the_image_crate() {
        let (w, h) = (4usize, 2usize);
        let data = gradient(w, h, w * 3);
        let rgb = image::RgbImage::from_raw(w as u32, h as u32, data.clone()).unwrap();
        for rotation in [90u32, 180, 270] {
            let ours = color_image_from_rgb(&data, w * 3, w, h, rotation);
            let reference = crate::video::quality::rotate_upright(rgb.clone(), rotation);
            assert_eq!(ours.size, [reference.width() as usize, reference.height() as usize], "{rotation}");
            for (i, px) in ours.pixels.iter().enumerate() {
                let r = reference.get_pixel((i % ours.size[0]) as u32, (i / ours.size[0]) as u32).0;
                assert_eq!([px.r(), px.g(), px.b()], r, "rotation {rotation} pixel {i}");
            }
        }
    }

    use crate::video::fixtures as fx;

    fn wait_for<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> Option<T> {
        let end = Instant::now() + timeout;
        while Instant::now() < end {
            if let Some(v) = f() {
                return Some(v);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    }

    fn options() -> PlayerOptions {
        PlayerOptions { idle_timeout: Duration::from_secs(60), ..Default::default() }
    }

    #[test]
    fn playback_is_paced_pausable_seekable_and_restartable() {
        if !fx::available() {
            return fx::skip("playback_is_paced");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fx::h264_mp4(dir.path(), "play.mp4", 3) else { return fx::skip("playback clip") };
        let player = Player::spawn(clip, options());
        let first = wait_for(Duration::from_secs(10), || player.latest_frame()).expect("a first frame");
        assert_eq!(first.image.size, [320, 180]);
        let status = player.status();
        assert!(status.ready && status.playing && !status.ended);
        assert!(status.duration_secs.is_some_and(|d| (d - 3.0).abs() < 0.2), "{status:?}");

        let (before, started) = (player.status().sent, Instant::now());
        while started.elapsed() < Duration::from_millis(1000) {
            let _ = player.latest_frame();
            std::thread::sleep(Duration::from_millis(5));
        }
        let (after, wall) = (player.status().sent, started.elapsed().as_secs_f64());
        let fps = (after - before) as f64 / wall;
        assert!((15.0..=32.0).contains(&fps), "frames are delivered at the clip rate, got {fps}");
        let pos = player.status().position_secs;
        assert!((0.5..=2.5).contains(&pos), "position {pos}");

        player.pause();
        std::thread::sleep(Duration::from_millis(250));
        assert!(!player.status().playing);
        let frozen = player.status().position_secs;
        std::thread::sleep(Duration::from_millis(300));
        assert!((player.status().position_secs - frozen).abs() < 0.05, "paused playback must not advance");
        while player.latest_frame().is_some() {}

        player.seek(2.0);
        let frame = wait_for(Duration::from_secs(5), || player.latest_frame().filter(|f| (f.pts_secs - 2.0).abs() < 0.1)).expect("a frame at the seek target");
        assert!((frame.pts_secs - 2.0).abs() < 0.1, "{}", frame.pts_secs);
        assert!(!player.status().playing, "seeking while paused stays paused");

        player.play();
        let ended = wait_for(Duration::from_secs(6), || {
            let _ = player.latest_frame();
            player.status().ended.then_some(())
        });
        assert!(ended.is_some(), "{:?}", player.status());
        assert!(player.status().position_secs >= 2.5);

        player.play();
        let restarted = wait_for(Duration::from_secs(5), || {
            let _ = player.latest_frame();
            let s = player.status();
            (!s.ended && s.position_secs < 1.0 && s.playing).then_some(s.position_secs)
        });
        assert!(restarted.is_some(), "play after the end restarts: {:?}", player.status());
        player.stop();
        assert!(player.join(Duration::from_secs(5)));
    }

    #[test]
    fn the_thread_ends_when_dropped_and_when_the_ui_stops_touching_it() {
        if !fx::available() {
            return fx::skip("the_thread_ends");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fx::h264_mp4(dir.path(), "drop.mp4", 3) else { return fx::skip("drop clip") };
        let player = Player::spawn(clip.clone(), options());
        wait_for(Duration::from_secs(10), || player.latest_frame()).expect("a frame");
        let done = player.watcher();
        assert!(!done());
        drop(player);
        assert!(wait_for(Duration::from_secs(3), || done().then_some(())).is_some(), "dropping must end the thread");

        let idle = Player::spawn(clip, PlayerOptions { idle_timeout: Duration::from_millis(300), ..Default::default() });
        assert!(idle.join(Duration::from_secs(5)), "an untouched player must end itself");
        assert!(idle.status().error.is_none());
    }

    #[test]
    fn a_paused_player_stays_alive_while_touched() {
        if !fx::available() {
            return fx::skip("a_paused_player_stays_alive");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fx::h264_mp4(dir.path(), "paused.mp4", 3) else { return fx::skip("paused clip") };
        let player = Player::spawn(clip, PlayerOptions { autoplay: false, idle_timeout: Duration::from_millis(400), ..Default::default() });
        let first = wait_for(Duration::from_secs(10), || player.latest_frame()).expect("a still frame while paused");
        assert!(first.pts_secs < 0.2);
        for _ in 0..12 {
            player.touch();
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(!player.status().finished && !player.status().playing);
        assert_eq!(player.status().sent, 1, "a paused player shows one frame");
    }

    #[test]
    fn rotated_clips_play_upright_and_bad_files_report_errors() {
        if !fx::available() {
            return fx::skip("rotated_clips_play_upright");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fx::rotated(dir.path(), "rot.mp4", 90) else { return fx::skip("rotated clip") };
        let player = Player::spawn(clip, options());
        let frame = wait_for(Duration::from_secs(10), || player.latest_frame()).expect("a frame");
        assert_eq!(frame.image.size, [180, 320]);

        let junk = fx::garbage(dir.path(), "junk.mp4", 50_000);
        let bad = Player::spawn(junk, options());
        assert!(bad.join(Duration::from_secs(10)));
        let status = bad.status();
        assert!(status.error.is_some() && status.finished && status.sent == 0, "{status:?}");
        let missing = Player::spawn(dir.path().join("none.mp4"), options());
        assert!(missing.join(Duration::from_secs(5)));
        assert!(missing.status().error.is_some());
        if let Some(audio) = fx::audio_only(dir.path(), "tone.mp4") {
            let p = Player::spawn(audio, options());
            assert!(p.join(Duration::from_secs(5)));
            assert!(p.status().error.is_some_and(|e| e.contains("no video")));
        }
    }

    #[test]
    fn seeking_past_the_end_and_before_the_start_is_clamped() {
        if !fx::available() {
            return fx::skip("seeking_past_the_end");
        }
        let dir = tempfile::tempdir().unwrap();
        let Some(clip) = fx::h264_mp4(dir.path(), "clamp.mp4", 3) else { return fx::skip("clamp clip") };
        let player = Player::spawn(clip, PlayerOptions { autoplay: false, ..options() });
        wait_for(Duration::from_secs(10), || player.latest_frame()).expect("first frame");
        player.seek(1000.0);
        let late = wait_for(Duration::from_secs(5), || player.latest_frame()).expect("a frame near the end");
        assert!(late.pts_secs > 2.5, "{}", late.pts_secs);
        player.seek(-5.0);
        let early = wait_for(Duration::from_secs(5), || player.latest_frame()).expect("a frame at the start");
        assert!(early.pts_secs < 0.2, "{}", early.pts_secs);
        assert!(player.status().error.is_none());
    }

    #[test]
    fn control_play_after_end_restarts_and_seek_shows_a_frame() {
        let mut ctl = Control { playing: false, ended: true, seek: None, show_one: false, anchor: Some((Instant::now(), 1.0)) };
        ctl.apply(Command::Play);
        assert!(ctl.playing && !ctl.ended && ctl.seek == Some(0.0) && ctl.anchor.is_none());
        ctl.apply(Command::Pause);
        assert!(!ctl.playing);
        ctl.apply(Command::Seek(3.5));
        assert!(ctl.show_one && ctl.seek == Some(3.5));
    }
}
