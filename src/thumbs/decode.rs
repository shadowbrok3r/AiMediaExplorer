//! Worker-side decoding: stored blobs and source files into egui pixel buffers.

use super::store::Blob;
use crate::utilities::guard::guarded;
use egui::ColorImage;
use image::DynamicImage;
use std::path::Path;

/// Largest side accepted from a stored blob; anything bigger is not a thumbnail.
const MAX_BLOB_SIDE: u32 = 2048;

#[derive(Debug)]
pub struct Decoded {
    pub image: ColorImage,
    pub cost: usize,
}

/// Bytes a thumbnail texture is charged against the budget, including its mip chain.
pub fn thumb_cost(w: usize, h: usize) -> usize {
    w * h * 4 * 4 / 3
}

pub fn preview_cost(w: usize, h: usize) -> usize {
    w * h * 4
}

pub fn color_image(img: DynamicImage) -> ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    match img {
        DynamicImage::ImageRgb8(i) => ColorImage::from_rgb(size, i.as_raw()),
        DynamicImage::ImageLuma8(i) => ColorImage::from_gray(size, i.as_raw()),
        other => ColorImage::from_rgba_unmultiplied(size, other.into_rgba8().as_raw()),
    }
}

pub fn decode_blob(blob: &Blob) -> Result<Decoded, String> {
    guarded(
        || {
            let img = image::load_from_memory_with_format(&blob.bytes, blob.format.image_format()).map_err(|e| format!("thumbnail decode: {e}"))?;
            let (w, h) = (img.width(), img.height());
            if w == 0 || h == 0 || w > MAX_BLOB_SIDE || h > MAX_BLOB_SIDE {
                return Err(format!("thumbnail has unusable size {w}x{h}"));
            }
            Ok(Decoded { cost: thumb_cost(w as usize, h as usize), image: color_image(img) })
        },
        |msg| format!("thumbnail decode panicked: {msg}"),
    )
}

pub fn decode_preview(path: &Path, max_dim: u32) -> Result<Decoded, String> {
    let img = crate::utilities::thumbs::generate_image_preview_image(path, max_dim)?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Err("preview has no pixels".into());
    }
    Ok(Decoded { cost: preview_cost(w, h), image: color_image(img) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::store::BlobFormat;
    use image::{ImageBuffer, Rgb, Rgba};
    use std::io::Cursor;

    fn encode(img: DynamicImage, format: image::ImageFormat) -> Vec<u8> {
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), format).unwrap();
        out
    }

    #[test]
    fn jpeg_blob_decodes_to_an_opaque_color_image() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(64, 32, Rgb([200, 30, 30]));
        let blob = Blob { format: BlobFormat::Jpeg, w: 64, h: 32, bytes: encode(DynamicImage::ImageRgb8(img), image::ImageFormat::Jpeg) };
        let d = decode_blob(&blob).unwrap();
        assert_eq!(d.image.size, [64, 32]);
        assert_eq!(d.cost, thumb_cost(64, 32));
        let px = d.image.pixels[0];
        assert!(px.r() > 150 && px.g() < 80 && px.a() == 255, "{px:?}");
    }

    #[test]
    fn png_blob_keeps_transparency() {
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(8, 8, |x, _| Rgba([10, 200, 10, if x < 4 { 0 } else { 255 }]));
        let blob = Blob { format: BlobFormat::Png, w: 8, h: 8, bytes: encode(DynamicImage::ImageRgba8(img), image::ImageFormat::Png) };
        let d = decode_blob(&blob).unwrap();
        assert_eq!(d.image.pixels[0].a(), 0);
        assert_eq!(d.image.pixels[7].a(), 255);
    }

    #[test]
    fn gray_jpeg_becomes_gray_pixels() {
        let img: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(16, 16, image::Luma([100]));
        let blob = Blob { format: BlobFormat::Jpeg, w: 16, h: 16, bytes: encode(DynamicImage::ImageLuma8(img), image::ImageFormat::Jpeg) };
        let d = decode_blob(&blob).unwrap();
        let px = d.image.pixels[0];
        assert!(px.r().abs_diff(px.g()) < 3 && px.g().abs_diff(px.b()) < 3);
    }

    #[test]
    fn garbage_and_oversized_blobs_are_errors_not_panics() {
        let junk = Blob { format: BlobFormat::Jpeg, w: 1, h: 1, bytes: vec![0xFF, 0xD8, 0xFF, 1, 2, 3] };
        assert!(decode_blob(&junk).is_err());
        let big: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(MAX_BLOB_SIDE + 8, 4, Rgb([1, 2, 3]));
        let blob = Blob { format: BlobFormat::Png, w: 0, h: 0, bytes: encode(DynamicImage::ImageRgb8(big), image::ImageFormat::Png) };
        assert!(decode_blob(&blob).unwrap_err().contains("unusable size"));
    }

    #[test]
    fn preview_decode_scales_to_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wide.png");
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(400, 200, Rgb([5, 6, 7]));
        img.save(&p).unwrap();
        let d = decode_preview(&p, 100).unwrap();
        assert_eq!(d.image.size, [100, 50]);
        assert_eq!(d.cost, preview_cost(100, 50));
        assert!(decode_preview(&dir.path().join("missing.png"), 100).is_err());
    }

    #[test]
    fn cost_includes_the_mip_chain() {
        assert_eq!(thumb_cost(256, 171), 256 * 171 * 4 * 4 / 3);
        assert!(thumb_cost(256, 256) > preview_cost(256, 256));
    }
}
