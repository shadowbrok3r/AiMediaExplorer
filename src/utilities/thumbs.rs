use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::io::Cursor;
use std::path::Path;

use crate::LogicalGroup;
use crate::utilities::guard::guarded;
use crate::utilities::types::is_raw;

pub const THUMB_MAX_DIM: u32 = 256;
pub const THUMB_JPEG_QUALITY: u8 = 85;
const PNG_DATA_URL_PREFIX: &str = "data:image/png;base64,";

// Base64 payload of a thumbnail string, with any `data:<mime>;base64,` prefix removed.
pub fn thumb_payload(thumb: &str) -> &str {
    thumb
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .map(|(_, payload)| payload)
        .unwrap_or(thumb)
}

fn has_transparency(img: &DynamicImage) -> bool {
    img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255)
}

// Opaque images become raw-base64 JPEG; images with real transparency stay PNG data URLs.
fn encode_thumb(img: &DynamicImage) -> Result<String, String> {
    let mut buf = Vec::new();
    if has_transparency(img) {
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png).map_err(|e| format!("encode png failed: {e}"))?;
        return Ok(format!("{PNG_DATA_URL_PREFIX}{}", BASE64.encode(&buf)));
    }
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, THUMB_JPEG_QUALITY);
    let encoded = match img.color() {
        image::ColorType::L8 | image::ColorType::L16 => encoder.encode_image(&img.to_luma8()),
        _ => encoder.encode_image(&img.to_rgb8()),
    };
    encoded.map_err(|e| format!("encode jpeg failed: {e}"))?;
    Ok(BASE64.encode(&buf))
}

pub fn generate_image_thumb_data(path: &Path) -> Result<String, String> {
    guarded(|| generate_image_thumb_data_inner(path), |msg| format!("thumbnail decode panicked: {msg}"))
}

// Blocking-pool variant for async callers.
pub async fn generate_image_thumb_data_blocking(path: std::path::PathBuf) -> Result<String, String> {
    tokio::task::spawn_blocking(move || generate_image_thumb_data(&path))
        .await
        .unwrap_or_else(|e| Err(format!("thumbnail task failed: {e}")))
}

fn generate_image_thumb_data_inner(path: &Path) -> Result<String, String> {
    log::debug!("[thumb] generating image thumb: {}", path.display());
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if is_raw(&ext) {
        return match crate::utilities::raw_decode::decode_raw_image(path, THUMB_MAX_DIM) {
            Ok(img) => encode_thumb(&img),
            Err(e) => {
                log::warn!("[thumb] raw decode failed {}: {e:#}", path.display());
                #[cfg(windows)]
                {
                    if let Ok(data) = generate_shell_thumb_data(path) {
                        return Ok(data);
                    }
                }
                Err(format!("raw: {e:#}"))
            }
        };
    }

    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            log::warn!("[thumb] open image failed {}: {}", path.display(), e);
            #[cfg(windows)]
            {
                match generate_shell_thumb_data(path) {
                    Ok(data) => {
                        log::debug!("[thumb] windows shell fallback success for image: {}", path.display());
                        return Ok(data);
                    }
                    Err(fe) => {
                        log::warn!("[thumb] windows shell fallback failed {}: {}", path.display(), fe);
                    }
                }
            }
            return Err(e.to_string());
        }
    };
    let thumb = img.thumbnail(THUMB_MAX_DIM, THUMB_MAX_DIM);
    let data = encode_thumb(&thumb).inspect_err(|e| log::warn!("[thumb] {}: {e}", path.display()))?;
    log::debug!("[thumb] image thumb success: {} ({} chars)", path.display(), data.len());
    Ok(data)
}

#[cfg(windows)]
pub fn generate_video_thumb_data(path: &Path) -> Result<String, String> {
    generate_shell_thumb_data(path)
}

/// On non-Windows platforms, video thumbnail generation is not supported via shell.
/// Returns an error indicating the feature is unavailable.
#[cfg(not(windows))]
pub fn generate_video_thumb_data(path: &Path) -> Result<String, String> {
    Err(format!("Video thumbnail generation not supported on this platform for: {}", path.display()))
}

// Windows Shell generic thumbnail generator to cover formats not handled by the image crate (e.g., RAW like .arw)
#[cfg(windows)]
fn generate_shell_thumb_data(path: &Path) -> Result<String, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::{
            Foundation::SIZE,
            System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx, IBindCtx},
            UI::Shell::{IShellItem, IShellItemImageFactory, SHCreateItemFromParsingName, SIIGBF},
        },
        core::{Interface, PCWSTR},
    };
    // Initialize COM only once per thread (STA). Repeated CoInitializeEx can produce warnings.
    thread_local! {
        static COM_INIT: std::cell::Cell<bool> = std::cell::Cell::new(false);
    }
    let mut init_err: Option<String> = None;
    COM_INIT.with(|cell| {
        if !cell.get() {
            unsafe {
                match CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok() {
                    Ok(_) => cell.set(true),
                    Err(e) => {
                        // Ignore RPC_E_CHANGED_MODE and proceed (already initialized in MTA or different model)
                        init_err = Some(format!("CoInitializeEx: {e}"));
                    }
                }
            }
        }
    });
    if let Some(e) = init_err.take() { log::trace!("[thumb] COM init issue (continuing): {e}"); }

    unsafe {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let shell_item: IShellItem = SHCreateItemFromParsingName(PCWSTR(wide.as_ptr()), None::<&IBindCtx>)
            .map_err(|e| format!("SHCreateItemFromParsingName: {e}"))?;
        let factory: IShellItemImageFactory = shell_item
            .cast()
            .map_err(|e| format!("cast IShellItemImageFactory: {e}"))?;
        let hbmp: windows::Win32::Graphics::Gdi::HBITMAP = factory
            .GetImage(SIZE { cx: 256, cy: 256 }, SIIGBF(0))
            .map_err(|e| format!("GetImage: {e}"))?;
        let data = hbitmap_to_png_data_url(hbmp)?;
        log::debug!("[thumb] shell thumb success: {}", path.display());
        Ok(data)
    }
}

#[cfg(windows)]
unsafe fn hbitmap_to_png_data_url(
    hbmp: windows::Win32::Graphics::Gdi::HBITMAP,
) -> Result<String, String> {
    use windows::Win32::Graphics::Gdi::*;
    let mut bmp = BITMAP::default();
    if unsafe {
        GetObjectW(
            HGDIOBJ(hbmp.0),
            std::mem::size_of::<BITMAP>() as i32,
            Some(&mut bmp as *mut _ as *mut _),
        )
    } == 0
    {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("GetObjectW failed".into());
    }
    let width = bmp.bmWidth as i32;
    let height = bmp.bmHeight as i32;
    let mut bi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: Default::default(),
    };
    let stride = (width * 4) as usize;
    let mut buffer = vec![0u8; (stride * height as usize) as usize];
    let hdc: HDC = unsafe { CreateCompatibleDC(None) };
    if hdc.0.is_null() {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("CreateCompatibleDC failed".into());
    }
    let _old = unsafe { SelectObject(hdc, HGDIOBJ(hbmp.0)) };
    let got = unsafe {
        GetDIBits(
            hdc,
            hbmp,
            0,
            height as u32,
            Some(buffer.as_mut_ptr() as *mut _),
            &mut bi as *mut _,
            DIB_RGB_COLORS,
        )
    };
    let _ = unsafe { DeleteDC(hdc) };
    let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
    if got == 0 {
        return Err("GetDIBits failed".into());
    }
    for px in buffer.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    let img = image::RgbaImage::from_raw(width as u32, height as u32, buffer)
        .ok_or("rgba from raw failed")?;
    let mut png = Vec::new();
    DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let b64 = BASE64.encode(&png);
    Ok(format!("data:image/png;base64,{}", b64))
}

// Helper: convert a FoundFile (basic scan result) into a minimal Thumbnail row for persistence
pub fn file_to_thumbnail(f: &crate::utilities::types::FoundFile) -> Option<crate::Thumbnail> {
    use chrono::Utc;
    // Store the actual file extension (lowercase) in file_type, not a coarse kind.
    let ft_string = f
        .path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| String::new());
    let modified = f.modified.map(|m| m.with_timezone(&chrono::Utc));
    let parent_dir = f.path.parent().map(|p| p.to_string_lossy().to_string().clone()).unwrap_or_default();
    Some(crate::Thumbnail {
        id: crate::Thumbnail::new(&f.path.file_name().unwrap_or_default().display().to_string()).id,
        db_created: Some(Utc::now().into()),
        path: f.path.display().to_string(),
        filename: f.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
        file_type: ft_string,
        size: f.size.unwrap_or(0),
        description: None,
        caption: None,
        tags: Vec::new(),
        category: None,
        thumbnail_b64: f.thumb_data.clone(), 
        modified: if let Some(date) = modified { Some(date.into()) } else { Some(Utc::now().into()) },
        hash: None,
        parent_dir,
        logical_group: LogicalGroup::default().id,
    })
}

// Windows Shell: return PNG bytes for any shell-supported preview (e.g., RAW like .arw)
#[cfg(windows)]
pub fn generate_shell_png_bytes(path: &Path, size: i32) -> Result<Vec<u8>, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::{
            Foundation::SIZE,
            System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx, IBindCtx},
            UI::Shell::{IShellItem, IShellItemImageFactory, SHCreateItemFromParsingName, SIIGBF},
        },
        core::{Interface, PCWSTR},
    };

    // Initialize COM only once per thread to avoid repeated init warnings
    thread_local! {
        static COM_INIT: std::cell::Cell<bool> = std::cell::Cell::new(false);
    }
    COM_INIT.with(|cell| {
        if !cell.get() {
            unsafe {
                if CoInitializeEx(None, COINIT_APARTMENTTHREADED).is_ok() {
                    cell.set(true);
                } else {
                    // Ignore and continue; shell calls may still succeed depending on apartment
                }
            }
        }
    });
    unsafe {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let shell_item: IShellItem = SHCreateItemFromParsingName(PCWSTR(wide.as_ptr()), None::<&IBindCtx>)
            .map_err(|e| format!("SHCreateItemFromParsingName: {e}"))?;
        let factory: IShellItemImageFactory = shell_item
            .cast()
            .map_err(|e| format!("cast IShellItemImageFactory: {e}"))?;
        let hbmp: windows::Win32::Graphics::Gdi::HBITMAP = factory
            .GetImage(SIZE { cx: size, cy: size }, SIIGBF(0))
            .map_err(|e| format!("GetImage: {e}"))?;
        let png = hbitmap_to_png_bytes(hbmp)?;
        Ok(png)
    }
}

#[cfg(windows)]
fn hbitmap_to_png_bytes(
    hbmp: windows::Win32::Graphics::Gdi::HBITMAP,
) -> Result<Vec<u8>, String> {
    use windows::Win32::Graphics::Gdi::*;
    let mut bmp = BITMAP::default();
    if unsafe {
        GetObjectW(
            HGDIOBJ(hbmp.0),
            std::mem::size_of::<BITMAP>() as i32,
            Some(&mut bmp as *mut _ as *mut _),
        )
    } == 0
    {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("GetObjectW failed".into());
    }
    let width = bmp.bmWidth as i32;
    let height = bmp.bmHeight as i32;
    let mut bi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: Default::default(),
    };
    let stride = (width * 4) as usize;
    let mut buffer = vec![0u8; (stride * height as usize) as usize];
    let hdc: HDC = unsafe { CreateCompatibleDC(None) };
    if hdc.0.is_null() {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("CreateCompatibleDC failed".into());
    }
    let _old = unsafe { SelectObject(hdc, HGDIOBJ(hbmp.0)) };
    let got = unsafe {
        GetDIBits(
            hdc,
            hbmp,
            0,
            height as u32,
            Some(buffer.as_mut_ptr() as *mut _),
            &mut bi as *mut _,
            DIB_RGB_COLORS,
        )
    };
    let _ = unsafe { DeleteDC(hdc) };
    let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
    if got == 0 {
        return Err("GetDIBits failed".into());
    }
    for px in buffer.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    let img = image::RgbaImage::from_raw(width as u32, height as u32, buffer)
        .ok_or("rgba from raw failed")?;
    let mut png = Vec::new();
    DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(png)
}

/// Generate a higher-quality preview PNG bytes for preview pane (not the small grid thumb).
/// Strategy:
/// - For RAW: use raw_decode at a larger max dimension (e.g., 1600px)
/// - Else: try image crate, resize to fit max_dim if larger, else use shell bitmap at larger size as fallback.
pub fn generate_image_preview_png(path: &Path, max_dim: u32) -> Result<Vec<u8>, String> {
    guarded(|| generate_image_preview_png_inner(path, max_dim), |msg| format!("preview decode panicked: {msg}"))
}

// Blocking-pool variant for async callers.
pub async fn generate_image_preview_png_blocking(path: std::path::PathBuf, max_dim: u32) -> Result<Vec<u8>, String> {
    tokio::task::spawn_blocking(move || generate_image_preview_png(&path, max_dim))
        .await
        .unwrap_or_else(|e| Err(format!("preview task failed: {e}")))
}

const PREVIEW_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(30);
const PREVIEW_RESULT_HOLD: std::time::Duration = std::time::Duration::from_secs(5);
const PREVIEW_GATE_MAX_ENTRIES: usize = 1024;

#[derive(Clone, Copy)]
enum PreviewGateState {
    InFlight,
    FailedAt(std::time::Instant),
    DoneAt(std::time::Instant),
}

static PREVIEW_GATE: once_cell::sync::Lazy<std::sync::Mutex<std::collections::HashMap<String, PreviewGateState>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

fn preview_state_blocks(state: PreviewGateState, now: std::time::Instant) -> bool {
    match state {
        PreviewGateState::InFlight => true,
        PreviewGateState::FailedAt(at) => now.saturating_duration_since(at) < PREVIEW_RETRY_AFTER,
        PreviewGateState::DoneAt(at) => now.saturating_duration_since(at) < PREVIEW_RESULT_HOLD,
    }
}

fn begin_preview_at(key: &str, now: std::time::Instant) -> bool {
    let Ok(mut gate) = PREVIEW_GATE.lock() else { return true };
    if gate.get(key).is_some_and(|s| preview_state_blocks(*s, now)) {
        return false;
    }
    if gate.len() >= PREVIEW_GATE_MAX_ENTRIES {
        gate.retain(|_, s| preview_state_blocks(*s, now));
    }
    gate.insert(key.to_string(), PreviewGateState::InFlight);
    true
}

fn end_preview_at(key: &str, succeeded: bool, now: std::time::Instant) {
    if let Ok(mut gate) = PREVIEW_GATE.lock() {
        gate.insert(key.to_string(), if succeeded { PreviewGateState::DoneAt(now) } else { PreviewGateState::FailedAt(now) });
    }
}

// True when the caller should start generating the preview for `key`; false while one is in flight, just finished or recently failed.
pub fn begin_preview(key: &str) -> bool {
    begin_preview_at(key, std::time::Instant::now())
}

// Records the outcome so the pane does not respawn before the result is stored, nor retry a failure every frame.
pub fn end_preview(key: &str, succeeded: bool) {
    end_preview_at(key, succeeded, std::time::Instant::now())
}

fn generate_image_preview_png_inner(path: &Path, max_dim: u32) -> Result<Vec<u8>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if is_raw(&ext) {
        return match crate::utilities::raw_decode::decode_raw_to_png_bytes(path, max_dim) {
            Ok(png) => Ok(png),
            Err(e) => {
                #[cfg(windows)]
                {
                    if let Ok(png) = generate_shell_png_bytes(path, max_dim as i32) {
                        return Ok(png);
                    }
                }
                Err(format!("raw: {e:#}"))
            }
        };
    }

    match image::open(path) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            let (nw, nh) = if w.max(h) > max_dim {
                if w >= h { (max_dim, ((h as f32) * (max_dim as f32) / (w as f32)).round() as u32) }
                else { (((w as f32) * (max_dim as f32) / (h as f32)).round() as u32, max_dim) }
            } else { (w, h) };
            let img2 = if nw != w || nh != h { img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3) } else { img };
            let mut out = Vec::new();
            img2.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(out)
        }
        Err(e) => {
            #[cfg(windows)]
            {
                // Fallback: larger shell bitmap for preview
                let _ = &e;
                return generate_shell_png_bytes(path, max_dim as i32);
            }
            #[cfg(not(windows))]
            {
                Err(format!("preview decode failed: {e}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb, Rgba};

    fn decode_payload(thumb: &str) -> DynamicImage {
        let bytes = BASE64.decode(thumb_payload(thumb)).expect("base64 payload");
        image::load_from_memory(&bytes).expect("decodable thumbnail")
    }

    fn opaque_rgba_png(dir: &Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let p = dir.join(name);
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| Rgba([(x % 251) as u8, (y % 241) as u8, 90, 255]));
        img.save(&p).unwrap();
        p
    }

    #[test]
    fn payload_strips_any_data_url_prefix() {
        assert_eq!(thumb_payload("data:image/png;base64,QUJD"), "QUJD");
        assert_eq!(thumb_payload("data:image/jpeg;base64,QUJD"), "QUJD");
        assert_eq!(thumb_payload("QUJD"), "QUJD");
    }

    #[test]
    fn opaque_jpeg_source_yields_raw_base64_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.jpg");
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(800, 400, |x, y| Rgb([(x / 4) as u8, (y / 2) as u8, 30]));
        img.save(&p).unwrap();
        let t = generate_image_thumb_data(&p).unwrap();
        assert!(!t.starts_with("data:"), "jpeg thumbnails carry no data url prefix");
        assert_eq!(&BASE64.decode(&t).unwrap()[..3], &[0xFF, 0xD8, 0xFF]);
        assert_eq!(decode_payload(&t).dimensions(), (256, 128));
    }

    #[test]
    fn opaque_rgba_png_source_becomes_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let p = opaque_rgba_png(dir.path(), "a.png", 600, 300);
        let t = generate_image_thumb_data(&p).unwrap();
        assert_eq!(&BASE64.decode(&t).unwrap()[..3], &[0xFF, 0xD8, 0xFF]);
        assert_eq!(decode_payload(&t).dimensions(), (256, 128));
    }

    #[test]
    fn transparent_png_source_stays_png_data_url() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.png");
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(300, 300, |x, _| Rgba([200, 10, 10, if x < 150 { 0 } else { 255 }]));
        img.save(&p).unwrap();
        let t = generate_image_thumb_data(&p).unwrap();
        assert!(t.starts_with(PNG_DATA_URL_PREFIX));
        assert!(decode_payload(&t).color().has_alpha());
    }

    #[test]
    fn gray_source_yields_gray_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("g.png");
        let img: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(500, 500, |x, _| image::Luma([(x / 2) as u8]));
        img.save(&p).unwrap();
        let t = generate_image_thumb_data(&p).unwrap();
        assert_eq!(decode_payload(&t).color(), image::ColorType::L8);
    }

    #[test]
    fn raw_file_with_embedded_jpeg_gets_a_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let mut jpeg = Vec::new();
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(640, 480, Rgb([20, 120, 220]));
        img.write_to(&mut Cursor::new(&mut jpeg), ImageFormat::Jpeg).unwrap();
        let mut bytes = vec![0u8; 3000];
        bytes.extend_from_slice(&jpeg);
        let p = dir.path().join("shot.cr3");
        std::fs::write(&p, &bytes).unwrap();
        let t = generate_image_thumb_data(&p).unwrap();
        assert_eq!(decode_payload(&t).dimensions(), (256, 192));
    }

    #[test]
    fn undecodable_raw_reports_a_raw_error_instead_of_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("broken.arw");
        std::fs::write(&p, [0x49u8, 0x49, 0x2A, 0x00, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let err = generate_image_thumb_data(&p).unwrap_err();
        assert!(err.starts_with("raw: "), "{err}");
        let err = generate_image_preview_png(&p, 1600).unwrap_err();
        assert!(err.starts_with("raw: "), "{err}");
    }

    #[test]
    fn preview_gate_blocks_duplicates_and_remembers_outcomes() {
        use std::time::{Duration, Instant};
        let t0 = Instant::now();
        let ok_key = "preview::/gate/ok.png,1600";
        assert!(begin_preview_at(ok_key, t0));
        assert!(!begin_preview_at(ok_key, t0), "second request while in flight");
        end_preview_at(ok_key, true, t0);
        assert!(!begin_preview_at(ok_key, t0 + Duration::from_secs(1)), "held while the result is being stored");
        assert!(begin_preview_at(ok_key, t0 + Duration::from_secs(6)), "allowed again if the cache dropped it");
        let bad_key = "preview::/gate/bad.png,1600";
        assert!(begin_preview_at(bad_key, t0));
        end_preview_at(bad_key, false, t0);
        assert!(!begin_preview_at(bad_key, t0 + Duration::from_secs(10)), "failure is negative-cached");
        assert!(begin_preview_at(bad_key, t0 + Duration::from_secs(31)), "failure expires");
        assert!(begin_preview_at("preview::/gate/other.png,1600", t0), "other keys are independent");
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(generate_image_thumb_data(Path::new("/nonexistent/none.jpg")).is_err());
    }

    #[test]
    fn real_arw_thumbnail_and_preview() {
        let Some(arw) = std::env::var_os("SMART_MEDIA_TEST_ARW").map(std::path::PathBuf::from).filter(|p| p.exists()) else { return };
        let t = generate_image_thumb_data(&arw).unwrap();
        let (w, h) = decode_payload(&t).dimensions();
        assert!(w.max(h) <= THUMB_MAX_DIM && w.max(h) > 200, "{w}x{h}");
        let png = generate_image_preview_png(&arw, 1600).unwrap();
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    #[ignore]
    fn bench_thumb_encode() {
        use image::{ExtendedColorType, ImageEncoder};
        let Some(dir) = std::env::var_os("SMART_MEDIA_BENCH_DIR") else { return };
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let reps = 30u32;
        let t_ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);
        for f in files {
            let Ok(img) = image::open(&f) else { continue };
            let thumb = img.thumbnail(THUMB_MAX_DIM, THUMB_MAX_DIM);
            let t = std::time::Instant::now();
            let mut png_len = 0;
            for _ in 0..reps {
                let mut buf = Vec::new();
                thumb.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png).unwrap();
                png_len = BASE64.encode(&buf).len();
            }
            let png_ms = t_ms(t);
            let t = std::time::Instant::now();
            let mut jpeg_len = 0;
            for _ in 0..reps {
                jpeg_len = encode_thumb(&thumb).unwrap().len();
            }
            let current_ms = t_ms(t);
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let rgb = thumb.to_rgb8();
                let mut buf = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, THUMB_JPEG_QUALITY)
                    .write_image(rgb.as_raw(), rgb.width(), rgb.height(), ExtendedColorType::Rgb8)
                    .unwrap();
                let _ = BASE64.encode(&buf);
            }
            let fast_ms = t_ms(t);
            println!("bench encode {:<30} png {png_ms:>6.2} ms ({png_len} ch) | jpeg encode_image {current_ms:>6.2} ms ({jpeg_len} ch) | jpeg write_image {fast_ms:>6.2} ms", f.file_name().unwrap().to_string_lossy());
        }
    }

    #[test]
    #[ignore]
    fn bench_thumb_pipeline() {
        let Some(dir) = std::env::var_os("SMART_MEDIA_BENCH_DIR") else { return };
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let (mut total_ms, mut total_bytes, mut n) = (0.0f64, 0usize, 0usize);
        for f in files {
            let t = std::time::Instant::now();
            let res = std::panic::catch_unwind(|| generate_image_thumb_data(&f));
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            match res {
                Ok(Ok(s)) => { println!("bench {name:<34} {ms:>9.1} ms {:>8} chars", s.len()); total_ms += ms; total_bytes += s.len(); n += 1; }
                Ok(Err(e)) => println!("bench {name:<34} {ms:>9.1} ms ERR {e}"),
                Err(_) => println!("bench {name:<34} {ms:>9.1} ms PANIC"),
            }
        }
        println!("bench total {n} ok: {total_ms:.1} ms, avg {:.1} ms, avg {} chars", total_ms / n.max(1) as f64, total_bytes / n.max(1));
    }
}