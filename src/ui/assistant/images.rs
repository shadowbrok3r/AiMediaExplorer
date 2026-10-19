//! Pictures for the gateway: only when the profile allows it, downscaled to fit 1024 pixels as JPEG, never above the inline size limit.

use zeroclaw_client::InlineImage;

/// Longest side of a picture sent to the gateway.
pub const MAX_SIDE: u32 = 1024;
/// Files larger than this are not even read.
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// The pictures that can be sent and what happened to the rest.
#[derive(Debug, Default)]
pub struct Prepared {
    pub images: Vec<InlineImage>,
    /// File names that were not sent as pictures.
    pub skipped: Vec<String>,
    /// Why some or all were not sent.
    pub notice: Option<String>,
}

fn name_of(path: &str) -> String {
    std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(path).to_string()
}

fn downscale(bytes: &[u8]) -> Option<Vec<u8>> {
    let img = image::load_from_memory(bytes).ok()?;
    let small = if img.width().max(img.height()) > MAX_SIDE { img.thumbnail(MAX_SIDE, MAX_SIDE) } else { img };
    let rgb = small.to_rgb8();
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&rgb).ok()?;
    Some(out)
}

/// Prepares `paths` for a message. With `allow` off nothing is sent and every name is listed as skipped.
pub fn prepare(paths: &[String], allow: bool) -> Prepared {
    let mut out = Prepared::default();
    if paths.is_empty() {
        return out;
    }
    if !allow {
        out.skipped = paths.iter().map(|p| name_of(p)).collect();
        out.notice = Some("Pictures are not sent: image upload is off for this gateway (Settings > Assistant > Privacy). The file names go in the message instead.".to_string());
        return out;
    }
    for p in paths {
        let name = name_of(p);
        let readable = std::fs::metadata(p).map(|m| m.is_file() && m.len() <= MAX_FILE_BYTES).unwrap_or(false);
        let bytes = if readable { std::fs::read(p).ok() } else { None };
        let image = bytes.as_deref().and_then(|b| downscale(b).and_then(|jpeg| InlineImage::from_bytes("picture.jpg", &jpeg).ok()).or_else(|| InlineImage::from_bytes(&name, b).ok()));
        match image {
            Some(i) => out.images.push(i),
            None => out.skipped.push(name),
        }
    }
    if !out.skipped.is_empty() {
        out.notice = Some(format!("Not sent as pictures (unreadable, too large or not an image): {}.", out.skipped.join(", ")));
    }
    out
}

/// A line naming files that went along without their pixels.
pub fn skipped_line(names: &[String]) -> String {
    if names.is_empty() { String::new() } else { format!("\n\n[Attached files, not sent as pictures: {}]", names.join(", ")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ame-assistant-img-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn with_images_off_nothing_is_sent_and_the_names_are_kept() {
        let p = prepare(&["/a/one.jpg".to_string(), "/b/two.png".to_string()], false);
        assert!(p.images.is_empty());
        assert_eq!(p.skipped, ["one.jpg", "two.png"]);
        assert!(p.notice.unwrap().contains("off for this gateway"));
        assert!(prepare(&[], false).notice.is_none());
    }

    #[test]
    fn a_large_picture_is_downscaled_and_sent_as_jpeg() {
        let d = dir("large");
        let path = d.join("big.png");
        std::fs::write(&path, png(3000, 1500)).unwrap();
        let p = prepare(&[path.display().to_string()], true);
        assert_eq!(p.images.len(), 1);
        assert_eq!(p.images[0].mime(), "image/jpeg");
        assert!(p.skipped.is_empty() && p.notice.is_none());
        let marker = p.images[0].marker();
        assert!(marker.starts_with("[IMAGE:data:image/jpeg;base64,"));
        assert!(p.images[0].raw_len() < 200_000, "a flat 3000 px picture shrinks well below its size");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn files_that_are_not_pictures_are_listed_as_skipped() {
        let d = dir("skip");
        let txt = d.join("notes.txt");
        std::fs::write(&txt, "hello").unwrap();
        let ok = d.join("ok.png");
        std::fs::write(&ok, png(10, 10)).unwrap();
        let p = prepare(&[txt.display().to_string(), ok.display().to_string(), d.join("missing.jpg").display().to_string()], true);
        assert_eq!(p.images.len(), 1);
        assert_eq!(p.skipped, ["notes.txt", "missing.jpg"]);
        assert!(p.notice.unwrap().contains("notes.txt"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn skipped_names_become_one_line() {
        assert_eq!(skipped_line(&[]), "");
        assert!(skipped_line(&["a.jpg".into(), "b.png".into()]).contains("a.jpg, b.png"));
    }
}
