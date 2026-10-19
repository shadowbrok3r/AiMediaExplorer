//! In-process ffmpeg backend: probing and poster decoding with ffmpeg-next.

use super::info::{self, AudioTrack, VideoInfo};
use super::poster::{Candidate, PosterOptions, Selector};
use super::quality;
use super::{Backend, Deadline, Poster, VideoError};
use ffmpeg_next as ff;
use ff::format::context::Input;
use ff::format::stream::Disposition;
use ff::software::scaling;
use ff::{Error as FfError, Packet, codec, decoder, format, frame, media};
use image::RgbImage;
use std::path::Path;
use std::sync::Once;
use std::time::Instant;

const AV_TIME_BASE: f64 = 1_000_000.0;
const MAX_PACKETS_PER_ATTEMPT: u32 = 20_000;
const MAX_BAD_PACKETS: u32 = 400;
const MAX_CORRUPT_FRAMES: u32 = 6;

/// Initialises ffmpeg once and silences its stderr logging.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = ff::init();
        ff::util::log::set_level(ff::util::log::Level::Fatal);
    });
}

/// Maps an ffmpeg error to the typed error, preferring the reason the deadline gives.
pub fn map_error(e: FfError, deadline: &Deadline) -> VideoError {
    if let Some(reason) = deadline.reason() {
        return reason;
    }
    match e {
        FfError::Exit => VideoError::Timeout,
        FfError::DecoderNotFound => VideoError::Corrupt("unsupported codec".into()),
        FfError::Other { errno } if errno == ff::error::ENOENT => VideoError::Missing("no such file".into()),
        other => VideoError::Corrupt(other.to_string()),
    }
}

/// Opens a media file whose blocking reads end at the deadline.
pub fn open(path: &Path, deadline: &Deadline) -> Result<Input, VideoError> {
    init();
    deadline.check()?;
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let d = deadline.clone();
    format::input_with_interrupt(&abs, move || d.reason().is_some()).map_err(|e| map_error(e, deadline))
}

struct ParamView {
    width: u32,
    height: u32,
    format: i32,
    bit_rate: i64,
    sample_rate: i32,
    channels: i32,
}

fn param_view(p: &codec::Parameters) -> ParamView {
    // SAFETY: the pointer comes from a live Parameters value and is only read.
    let raw = unsafe { &*p.as_ptr() };
    ParamView { width: raw.width.max(0) as u32, height: raw.height.max(0) as u32, format: raw.format, bit_rate: raw.bit_rate, sample_rate: raw.sample_rate, channels: raw.ch_layout.nb_channels }
}

fn pixel_format_name(fmt: i32) -> Option<String> {
    if fmt < 0 || fmt >= ff::ffi::AVPixelFormat::AV_PIX_FMT_NB as i32 {
        return None;
    }
    // SAFETY: the value is inside the enum's valid range and the name pointer is static.
    unsafe {
        let name = ff::ffi::av_get_pix_fmt_name(std::mem::transmute::<i32, ff::ffi::AVPixelFormat>(fmt));
        (!name.is_null()).then(|| std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

fn stream_sar(stream: &ff::Stream) -> Option<(u32, u32)> {
    // SAFETY: the stream pointer is valid while the input is open and is only read.
    let sar = unsafe { (*stream.as_ptr()).sample_aspect_ratio };
    info::normalize_sar(i64::from(sar.num), i64::from(sar.den)).or_else(|| {
        let raw = unsafe { (*stream.parameters().as_ptr()).sample_aspect_ratio };
        info::normalize_sar(i64::from(raw.num), i64::from(raw.den))
    })
}

fn stream_rotation(stream: &ff::Stream) -> u32 {
    let from_matrix = stream.side_data().find(|sd| sd.kind() == ff::packet::side_data::Type::DisplayMatrix).and_then(|sd| info::display_matrix_from_bytes(sd.data())).and_then(|m| info::rotation_from_display_matrix(&m)).map(info::clockwise_rotation);
    from_matrix.or_else(|| stream.metadata().get("rotate").and_then(info::rotation_from_tag)).unwrap_or(0)
}

fn language(stream: &ff::Stream) -> Option<String> {
    stream.metadata().get("language").filter(|l| *l != "und").map(str::to_string)
}

/// Index of the stream used as the video: not cover art, preferring the default and the largest.
pub fn pick_video(ictx: &Input) -> Option<usize> {
    ictx.streams()
        .filter(|s| s.parameters().medium() == media::Type::Video && !s.disposition().contains(Disposition::ATTACHED_PIC))
        .max_by_key(|s| {
            let p = param_view(&s.parameters());
            (s.disposition().contains(Disposition::DEFAULT), u64::from(p.width) * u64::from(p.height))
        })
        .map(|s| s.index())
}

/// Stream facts of an open input.
pub fn info_from_context(ictx: &Input, size_bytes: u64) -> VideoInfo {
    let mut out = VideoInfo { container: Some(ictx.format().name().to_string()), size_bytes: Some(size_bytes), ..VideoInfo::default() };
    let duration = ictx.duration();
    if duration > 0 {
        out.duration_secs = Some(duration as f64 / AV_TIME_BASE);
    }
    if ictx.bit_rate() > 0 {
        out.bitrate = Some(ictx.bit_rate() as u64);
    }
    out.creation_time = ictx.metadata().get("creation_time").map(str::to_string);
    let video = pick_video(ictx);
    for stream in ictx.streams() {
        let params = stream.parameters();
        let view = param_view(&params);
        match params.medium() {
            media::Type::Video if Some(stream.index()) == video => {
                out.has_video = true;
                out.video_index = video;
                out.video_codec = Some(params.id().name().to_string());
                out.pixel_format = pixel_format_name(view.format);
                (out.width, out.height) = (view.width, view.height);
                out.sar = stream_sar(&stream);
                out.rotation = stream_rotation(&stream);
                let rate = stream.avg_frame_rate();
                let rate = if rate.denominator() > 0 && rate.numerator() > 0 { rate } else { stream.rate() };
                out.fps = (rate.denominator() > 0 && rate.numerator() > 0).then(|| f64::from(rate.numerator()) / f64::from(rate.denominator()));
                out.video_bitrate = (view.bit_rate > 0).then_some(view.bit_rate as u64);
                if out.creation_time.is_none() {
                    out.creation_time = stream.metadata().get("creation_time").map(str::to_string);
                }
            }
            media::Type::Audio => out.audio.push(AudioTrack {
                index: stream.index(),
                codec: Some(params.id().name().to_string()),
                channels: (view.channels > 0).then_some(view.channels as u32),
                sample_rate: (view.sample_rate > 0).then_some(view.sample_rate as u32),
                bitrate: (view.bit_rate > 0).then_some(view.bit_rate as u64),
                language: language(&stream),
            }),
            media::Type::Subtitle => out.subtitle_tracks += 1,
            _ => {}
        }
    }
    out
}

/// Probes a file in-process.
pub fn probe(path: &Path, size_bytes: u64, deadline: &Deadline) -> Result<VideoInfo, VideoError> {
    let ictx = open(path, deadline)?;
    Ok(info_from_context(&ictx, size_bytes))
}

/// Opens the video decoder of a stream; `threads` of 0 lets ffmpeg choose.
pub fn open_decoder(ictx: &Input, index: usize, threads: usize, deadline: &Deadline) -> Result<decoder::Video, VideoError> {
    let stream = ictx.stream(index).ok_or(VideoError::NoVideoStream)?;
    let mut ctx = codec::Context::from_parameters(stream.parameters()).map_err(|e| map_error(e, deadline))?;
    ctx.set_threading(codec::threading::Config::count(threads));
    ctx.decoder().video().map_err(|e| map_error(e, deadline))
}

/// Reusable swscale context and output frame for a stream of decoded frames.
#[derive(Default)]
pub struct Scaler {
    key: Option<(ff::format::Pixel, u32, u32, u32, u32)>,
    ctx: Option<scaling::Context>,
    out: Option<frame::Video>,
}

impl Scaler {
    /// Scales a decoded frame to a packed RGB image of `scaled_w` x `scaled_h`.
    pub fn to_rgb(&mut self, src: &frame::Video, scaled_w: u32, scaled_h: u32, flags: scaling::Flags) -> Result<RgbImage, VideoError> {
        let out = self.scale(src, scaled_w, scaled_h, flags)?;
        rgb_from_frame(out, scaled_w, scaled_h)
    }

    /// Scales a decoded frame into the reused RGB24 output frame.
    pub fn scale(&mut self, src: &frame::Video, scaled_w: u32, scaled_h: u32, flags: scaling::Flags) -> Result<&frame::Video, VideoError> {
        let (fmt, w, h) = (src.format(), src.width(), src.height());
        if fmt == ff::format::Pixel::None || w == 0 || h == 0 || scaled_w == 0 || scaled_h == 0 {
            return Err(VideoError::Corrupt("frame has no usable format or size".into()));
        }
        let key = (fmt, w, h, scaled_w, scaled_h);
        if self.key != Some(key) {
            let mut ctx = scaling::Context::get(fmt, w, h, ff::format::Pixel::RGB24, scaled_w, scaled_h, flags).map_err(|e| VideoError::Corrupt(format!("scaler: {e}")))?;
            apply_colorspace(&mut ctx, src);
            self.ctx = Some(ctx);
            self.out = Some(frame::Video::empty());
            self.key = Some(key);
        }
        let (Some(ctx), Some(out)) = (self.ctx.as_mut(), self.out.as_mut()) else {
            return Err(VideoError::Backend("scaler not initialised".into()));
        };
        ctx.run(src, out).map_err(|e| VideoError::Corrupt(format!("scale: {e}")))?;
        Ok(out)
    }
}

fn apply_colorspace(ctx: &mut scaling::Context, src: &frame::Video) {
    use ff::color::{Range, Space};
    let space = match src.color_space() {
        Space::BT709 => Some(ff::ffi::SWS_CS_ITU709),
        Space::BT470BG | Space::SMPTE170M => Some(ff::ffi::SWS_CS_ITU601),
        Space::BT2020NCL | Space::BT2020CL => Some(ff::ffi::SWS_CS_BT2020),
        Space::SMPTE240M => Some(ff::ffi::SWS_CS_SMPTE240M),
        _ if src.height() >= 720 => Some(ff::ffi::SWS_CS_ITU709),
        _ => None,
    };
    let Some(space) = space else { return };
    let full_range = i32::from(src.color_range() == Range::JPEG);
    // SAFETY: the scaling context is live and the coefficient tables are static ffmpeg data.
    unsafe {
        let src_table = ff::ffi::sws_getCoefficients(space as i32);
        let dst_table = ff::ffi::sws_getCoefficients(ff::ffi::SWS_CS_DEFAULT as i32);
        ff::ffi::sws_setColorspaceDetails(ctx.as_mut_ptr(), src_table, full_range, dst_table, 1, 0, 1 << 16, 1 << 16);
    }
}

/// Copies the visible rows of a packed RGB24 frame into an image.
pub fn rgb_from_frame(out: &frame::Video, w: u32, h: u32) -> Result<RgbImage, VideoError> {
    let stride = out.stride(0);
    let data = out.data(0);
    let row = w as usize * 3;
    if stride < row || data.len() < stride * (h as usize - 1) + row {
        return Err(VideoError::Corrupt("scaled frame is smaller than expected".into()));
    }
    let mut rgb = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        rgb.extend_from_slice(&data[y * stride..y * stride + row]);
    }
    RgbImage::from_raw(w, h, rgb).ok_or_else(|| VideoError::Corrupt("rgb size mismatch".into()))
}

/// Pixel aspect of a decoded frame, falling back to the stream's.
pub fn frame_sar(f: &frame::Video, fallback: Option<(u32, u32)>) -> Option<(u32, u32)> {
    let sar = f.aspect_ratio();
    info::normalize_sar(i64::from(sar.numerator()), i64::from(sar.denominator())).or(fallback)
}

/// Decodes forward from the current position to the first usable frame.
pub(super) fn next_frame(ictx: &mut Input, decoder: &mut decoder::Video, index: usize, out: &mut frame::Video, deadline: &Deadline) -> Result<bool, VideoError> {
    let (mut packets, mut bad, mut corrupt, mut eof_sent) = (0u32, 0u32, 0u32, false);
    loop {
        deadline.check()?;
        match decoder.receive_frame(out) {
            Ok(()) => {
                corrupt += 1;
                if out.is_corrupt() && corrupt <= MAX_CORRUPT_FRAMES {
                    continue;
                }
                return Ok(true);
            }
            Err(FfError::Eof) => return Ok(false),
            Err(FfError::Other { errno }) if errno == ff::error::EAGAIN => {}
            Err(_) => {
                bad += 1;
                if bad > MAX_BAD_PACKETS {
                    return Err(VideoError::Corrupt("too many undecodable packets".into()));
                }
            }
        }
        if eof_sent {
            return Ok(false);
        }
        let mut packet = Packet::empty();
        match packet.read(ictx) {
            Ok(()) => {
                if packet.stream() == index {
                    packets += 1;
                    if packets > MAX_PACKETS_PER_ATTEMPT {
                        return Err(VideoError::Corrupt("no decodable frame found".into()));
                    }
                    if decoder.send_packet(&packet).is_err() {
                        bad += 1;
                    }
                }
            }
            Err(FfError::Eof) => {
                let _ = decoder.send_eof();
                eof_sent = true;
            }
            Err(FfError::InvalidData) => {
                bad += 1;
                if bad > MAX_BAD_PACKETS {
                    return Err(VideoError::Corrupt("too many damaged packets".into()));
                }
            }
            Err(e) => return Err(map_error(e, deadline)),
        }
    }
}

/// Container start time in microseconds, 0 when unknown.
fn start_micros(ictx: &Input) -> i64 {
    // SAFETY: the context pointer is valid while the input is open and is only read.
    let start = unsafe { (*ictx.as_ptr()).start_time };
    if start == i64::MIN { 0 } else { start }
}

/// Decodes a poster in-process: seek, decode, scale, rotate, skip blank frames.
pub fn poster(path: &Path, size_bytes: u64, opts: &PosterOptions, deadline: &Deadline) -> Result<Poster, VideoError> {
    let started = Instant::now();
    let mut ictx = open(path, deadline)?;
    let info = info_from_context(&ictx, size_bytes);
    let index = info.video_index.ok_or(VideoError::NoVideoStream)?;
    let mut decoder = open_decoder(&ictx, index, 1, deadline)?;
    let start_us = start_micros(&ictx);
    let duration_us = info.duration_secs.map(|d| (d * AV_TIME_BASE) as i64).filter(|d| *d > 0);
    let mut selector = Selector::new(opts.attempts);
    let (mut scaler, mut decoded) = (Scaler::default(), frame::Video::empty());
    let mut last_error = None;
    for (i, fraction) in info::attempt_fractions(opts.at_fraction, opts.attempts as usize).into_iter().enumerate() {
        deadline.check()?;
        let mut position = None;
        match duration_us {
            Some(duration) => {
                let target = info::seek_target_micros(duration, fraction);
                if target > 0 || i > 0 {
                    if ictx.seek(start_us + target, ..start_us + target).is_err() {
                        if i > 0 {
                            continue;
                        }
                    } else {
                        decoder.flush();
                        position = Some(target as f64 / AV_TIME_BASE);
                    }
                }
            }
            None if i > 0 => break,
            None => {}
        }
        match next_frame(&mut ictx, &mut decoder, index, &mut decoded, deadline) {
            Ok(true) => match frame_to_candidate(&decoded, &info, opts.max_dim, &mut scaler, position) {
                Ok(candidate) => {
                    if selector.offer(candidate) {
                        break;
                    }
                }
                Err(e) => last_error = Some(e),
            },
            Ok(false) => last_error = Some(VideoError::Corrupt("no frame decoded".into())),
            Err(e @ (VideoError::Timeout | VideoError::Cancelled)) => return Err(e),
            Err(e) => last_error = Some(e),
        }
    }
    if selector.attempts() == 0 {
        let mut fresh = open(path, deadline)?;
        let mut decoder = open_decoder(&fresh, index, 1, deadline)?;
        match next_frame(&mut fresh, &mut decoder, index, &mut decoded, deadline) {
            Ok(true) => match frame_to_candidate(&decoded, &info, opts.max_dim, &mut scaler, Some(0.0)) {
                Ok(candidate) => {
                    selector.offer(candidate);
                }
                Err(e) => last_error = Some(e),
            },
            Ok(false) => {}
            Err(e @ (VideoError::Timeout | VideoError::Cancelled)) => return Err(e),
            Err(e) => last_error = Some(e),
        }
    }
    selector.finish(&info, Backend::Native, started.elapsed()).ok_or_else(|| last_error.unwrap_or_else(|| VideoError::Corrupt("no frame decoded".into())))
}

fn frame_to_candidate(decoded: &frame::Video, info: &VideoInfo, max_dim: u32, scaler: &mut Scaler, position: Option<f64>) -> Result<Candidate, VideoError> {
    let sar = frame_sar(decoded, info.sar);
    let geo = info::thumb_geometry(decoded.width(), decoded.height(), sar, info.rotation, max_dim).ok_or_else(|| VideoError::Corrupt("frame has no size".into()))?;
    let image = scaler.to_rgb(decoded, geo.scaled_w, geo.scaled_h, scaling::Flags::AREA)?;
    Ok(Candidate { image: quality::rotate_upright(image, geo.rotation), position_secs: position })
}
