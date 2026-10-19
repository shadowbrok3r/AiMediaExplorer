//! Tool that returns a downscaled picture of a file as image content.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use image::GenericImageView;
use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::safe_text;
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};

const DEFAULT_PX: u32 = 768;
const MIN_PX: u32 = 64;
const MAX_PX: u32 = 1024;
const MAX_SOURCE_BYTES: u64 = 256 * 1024 * 1024;
const JPEG_QUALITY: u8 = 85;

pub fn entries() -> Vec<ToolEntry> {
    vec![entry("library_get_preview", ToolClass::Read, Duration::from_secs(60), "Look at an image", "Return a downscaled JPEG of an image file inside the allowed folders (up to 1024 pixels on the long side) so the picture itself can be inspected.", library_get_preview)]
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PreviewArgs {
    /// Absolute path of an image inside the library, a watched folder or the Pictures folder.
    pub path: String,
    /// Longest side in pixels, 64 to 1024; default 768.
    pub max_px: Option<u32>,
}

/// JPEG bytes of `img` scaled to fit `max_px`.
pub fn encode_jpeg(img: &image::DynamicImage, max_px: u32) -> Result<(Vec<u8>, u32, u32), ToolError> {
    let (w, h) = img.dimensions();
    let scaled = if w.max(h) > max_px { img.resize(max_px, max_px, image::imageops::FilterType::Triangle) } else { img.clone() };
    let (sw, sh) = scaled.dimensions();
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY).encode_image(&scaled.to_rgb8()).map_err(ToolError::failed)?;
    Ok((bytes, sw, sh))
}

async fn library_get_preview(ctx: Arc<ToolCtx>, _call: CallInfo, args: PreviewArgs) -> Result<ToolOutput, ToolError> {
    ctx.policy.refresh(&ctx.db).await;
    let real = ctx.policy.check_file(args.path.trim())?;
    let ext = real.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    if !(crate::is_image(&ext) || crate::is_raw(&ext)) {
        return Err(ToolError::invalid("the file is not a supported image type"));
    }
    let size = std::fs::metadata(&real).map_err(ToolError::failed)?.len();
    if size > MAX_SOURCE_BYTES {
        return Err(ToolError::invalid(format!("the file is {} MiB; previews are limited to {} MiB", size >> 20, MAX_SOURCE_BYTES >> 20)));
    }
    let max_px = args.max_px.unwrap_or(DEFAULT_PX).clamp(MIN_PX, MAX_PX);
    let source = real.clone();
    let (bytes, w, h) = tokio::task::spawn_blocking(move || {
        let img = crate::utilities::thumbs::generate_image_preview_image(&source, max_px).map_err(ToolError::Failed)?;
        encode_jpeg(&img, max_px)
    })
    .await
    .map_err(ToolError::failed)??;
    let name = real.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let data = json!({"path": args.path.trim(), "width": w, "height": h, "jpeg_bytes": bytes.len(), "source_bytes": size});
    Ok(ToolOutput::new(format!("preview of {} at {w}x{h}", safe_text(&name, 80)), data).with_image("image/jpeg", base64::engine::general_purpose::STANDARD.encode(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_images_are_scaled_to_fit_and_small_ones_are_kept() {
        let big = image::DynamicImage::new_rgb8(2000, 1000);
        let (bytes, w, h) = encode_jpeg(&big, 512).unwrap();
        assert_eq!((w, h), (512, 256));
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "a JPEG starts with FFD8");
        let small = image::DynamicImage::new_rgb8(100, 50);
        let (_, w, h) = encode_jpeg(&small, 512).unwrap();
        assert_eq!((w, h), (100, 50));
    }
}
