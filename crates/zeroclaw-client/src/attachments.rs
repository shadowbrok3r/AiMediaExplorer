use std::path::Path;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use crate::error::{Error, Result};

/// Largest raw image accepted for an inline marker.
pub const MAX_INLINE_IMAGE_BYTES: usize = 8 * 1024 * 1024;

const OPEN: &str = "[IMAGE:data:";

/// An image sent inside message text as an `[IMAGE:data:<mime>;base64,...]` marker (PNG, JPEG, GIF, WebP).
#[derive(Clone, PartialEq, Eq)]
pub struct InlineImage {
    mime: &'static str,
    base64: String,
    raw_len: usize,
}

impl std::fmt::Debug for InlineImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineImage").field("mime", &self.mime).field("raw_len", &self.raw_len).finish()
    }
}

impl InlineImage {
    /// MIME comes from magic bytes first, then from `name_hint`'s extension.
    pub fn from_bytes(name_hint: &str, bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(Error::Attachment(format!("{name_hint} is empty")));
        }
        if bytes.len() > MAX_INLINE_IMAGE_BYTES {
            return Err(Error::Attachment(format!(
                "{name_hint} is {} MiB; the inline limit is {} MiB",
                bytes.len().div_ceil(1_048_576),
                MAX_INLINE_IMAGE_BYTES / 1_048_576
            )));
        }
        let mime = image_mime(name_hint, bytes)
            .ok_or_else(|| Error::Attachment(format!("{name_hint} is not a PNG, JPEG, GIF or WebP image")))?;
        Ok(Self { mime, base64: STANDARD.encode(bytes), raw_len: bytes.len() })
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::Attachment(format!("cannot read {}: {e}", path.display())))?;
        Self::from_bytes(&path.to_string_lossy(), &bytes)
    }

    pub fn mime(&self) -> &'static str {
        self.mime
    }

    pub fn raw_len(&self) -> usize {
        self.raw_len
    }

    /// The marker text to splice into a message or prompt.
    pub fn marker(&self) -> String {
        format!("{OPEN}{};base64,{}]", self.mime, self.base64)
    }

    /// Bytes the marker adds to a message.
    pub fn marker_len(&self) -> usize {
        OPEN.len() + self.mime.len() + ";base64,".len() + self.base64.len() + 1
    }
}

/// The image MIME the provider accepts for these bytes, or `None`.
pub fn image_mime(name_hint: &str, bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    let ext = name_hint.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Message body with `images` appended as markers after the text; an image-only message is the markers alone.
pub fn compose_message(text: &str, images: &[InlineImage]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(images.len() + 1);
    if !text.trim().is_empty() {
        parts.push(text.to_string());
    }
    parts.extend(images.iter().map(InlineImage::marker));
    parts.join(" ")
}

/// An image reference found in message text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageRef {
    /// `[IMAGE:data:<mime>;base64,...]`; the payload is not kept.
    Inline { mime: String, base64_len: usize },
    /// `[IMAGE:<path>]`: a file on the gateway, fetched through the session media endpoint.
    Path(String),
    /// An unterminated or malformed marker.
    Broken,
}

/// Text without `[IMAGE:...]` markers plus the markers in order; an unterminated marker drops the rest of the text.
pub fn split_images(text: &str) -> (String, Vec<ImageRef>) {
    const TAG: &str = "[IMAGE:";
    let mut parts: Vec<&str> = Vec::new();
    let mut refs = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(TAG) {
        parts.push(&rest[..i]);
        let body = &rest[i + TAG.len()..];
        let Some(end) = body.find(']') else {
            refs.push(ImageRef::Broken);
            rest = "";
            break;
        };
        let inner = &body[..end];
        refs.push(match inner.strip_prefix("data:") {
            Some(data) => match data.split_once(";base64,") {
                Some((mime, b64)) => ImageRef::Inline { mime: mime.to_string(), base64_len: b64.len() },
                None => ImageRef::Broken,
            },
            None if inner.is_empty() => ImageRef::Broken,
            None => ImageRef::Path(inner.to_string()),
        });
        rest = &body[end + 1..];
    }
    parts.push(rest);
    let text = parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ");
    (text, refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\nrest";

    #[test]
    fn mime_prefers_magic_bytes_over_the_name() {
        assert_eq!(image_mime("photo.txt", PNG), Some("image/png"));
        assert_eq!(image_mime("x.png", &[0xff, 0xd8, 0xff, 0x00]), Some("image/jpeg"));
        assert_eq!(image_mime("a.gif", b"GIF89a....."), Some("image/gif"));
        assert_eq!(image_mime("a.bin", b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some("image/webp"));
    }

    #[test]
    fn mime_falls_back_to_the_extension_then_gives_up() {
        assert_eq!(image_mime("a.jpeg", b"unknown-magic"), Some("image/jpeg"));
        assert_eq!(image_mime("notes.pdf", b"%PDF-1.7"), None);
    }

    #[test]
    fn marker_has_the_documented_shape() {
        let img = InlineImage::from_bytes("a.png", PNG).unwrap();
        let marker = img.marker();
        assert!(marker.starts_with("[IMAGE:data:image/png;base64,"), "{marker}");
        assert!(marker.ends_with(']'));
        assert_eq!(marker.len(), img.marker_len());
        assert_eq!(img.mime(), "image/png");
        assert_eq!(img.raw_len(), PNG.len());
        assert!(!format!("{img:?}").contains("base64"));
    }

    #[test]
    fn oversized_empty_and_unknown_images_are_refused() {
        let big = vec![0xff, 0xd8, 0xff, 0x00].into_iter().chain(std::iter::repeat_n(0u8, MAX_INLINE_IMAGE_BYTES)).collect::<Vec<_>>();
        assert!(matches!(InlineImage::from_bytes("a.jpg", &big), Err(Error::Attachment(_))));
        assert!(InlineImage::from_bytes("a.png", b"").is_err());
        assert!(InlineImage::from_bytes("doc.pdf", b"%PDF").is_err());
        assert!(InlineImage::read("/nonexistent/definitely/missing.png").is_err());
    }

    #[test]
    fn compose_puts_markers_after_text_and_handles_image_only() {
        let img = InlineImage::from_bytes("a.png", PNG).unwrap();
        let both = compose_message("look", std::slice::from_ref(&img));
        assert_eq!(both, format!("look {}", img.marker()));
        assert_eq!(compose_message("   ", std::slice::from_ref(&img)), img.marker());
        assert_eq!(compose_message("hi", &[]), "hi");
        let two = compose_message("x", &[img.clone(), img.clone()]);
        assert_eq!(two.matches("[IMAGE:data:").count(), 2);
    }

    #[test]
    fn split_images_strips_markers_and_classifies_them() {
        let (text, refs) = split_images("look [IMAGE:data:image/png;base64,AAAA] please [IMAGE:/tmp/out.png]");
        assert_eq!(text, "look please");
        assert_eq!(
            refs,
            vec![
                ImageRef::Inline { mime: "image/png".into(), base64_len: 4 },
                ImageRef::Path("/tmp/out.png".into())
            ]
        );
    }

    #[test]
    fn split_images_does_not_leak_a_broken_marker() {
        let (text, refs) = split_images("cut [IMAGE:data:image/png;base64,AAAA");
        assert_eq!(text, "cut");
        assert_eq!(refs, vec![ImageRef::Broken]);
        let (text, refs) = split_images("odd [IMAGE:data:nonsense] [IMAGE:]");
        assert_eq!(text, "odd");
        assert_eq!(refs, vec![ImageRef::Broken, ImageRef::Broken]);
        assert_eq!(split_images("[note] [IMAGES] plain").0, "[note] [IMAGES] plain");
    }
}
