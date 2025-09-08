use std::path::Path;

pub fn decode_raw_to_png_bytes(path: &Path, max_dim: u32) -> anyhow::Result<Vec<u8>, anyhow::Error> {
    use std::fs::File;
    use std::io::BufReader;
    use image::{imageops::FilterType, DynamicImage, ImageBuffer, RgbImage};

    // Safety/IO: open file buffered
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Decode and process to 8-bit RGB using rawkit
    let raw = rawkit::RawImage::decode(&mut reader)?;
    // Optional: you could tweak white balance / transform here if desired.
    let rgb = raw.process_8bit();
    // rawkit image fields (assumed): width, height, data (RGB bytes)
    if rgb.data.len() < (rgb.width as usize) * (rgb.height as usize) * 3 {
        return Err(anyhow::anyhow!("rawkit produced unexpected buffer size"));
    }
    let mut img: RgbImage = ImageBuffer::from_raw(rgb.width as u32, rgb.height as u32, rgb.data).unwrap_or_default();

    // Resize to max_dim while preserving aspect ratio, if larger
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = if w.max(h) > max_dim {
        if w >= h {
            (max_dim, ((h as f32) * (max_dim as f32) / (w as f32)).round() as u32)
        } else {
            (((w as f32) * (max_dim as f32) / (h as f32)).round() as u32, max_dim)
        }
    } else {
        (w, h)
    };
    if nw != w || nh != h {
        let dyn_img = DynamicImage::ImageRgb8(img);
        img = dyn_img.resize_exact(nw, nh, FilterType::Lanczos3).to_rgb8();
    }

    // Encode to PNG bytes
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    Ok(out)
}

