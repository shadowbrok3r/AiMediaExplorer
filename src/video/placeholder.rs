//! Generated stand-in thumbnails for media files that have no frame to show.

use image::{Rgb, RgbImage};

pub const PLACEHOLDER_W: u32 = 256;
pub const PLACEHOLDER_H: u32 = 144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaceholderKind {
    /// Play triangle for video files without a decodable frame.
    Video,
    /// Level bars for files that only carry audio.
    Audio,
}

/// Dark tile with a centred symbol for the media kind.
pub fn placeholder(kind: PlaceholderKind) -> RgbImage {
    let (w, h) = (PLACEHOLDER_W, PLACEHOLDER_H);
    let mut img = RgbImage::from_fn(w, h, |_, y| {
        let t = y as f32 / h as f32;
        let base = 38.0 + 18.0 * t;
        Rgb([base as u8, (base + 4.0) as u8, (base + 12.0) as u8])
    });
    let ink = Rgb([196, 204, 220]);
    let (cx, cy) = (w as i32 / 2, h as i32 / 2);
    match kind {
        PlaceholderKind::Video => draw_play(&mut img, cx, cy, 30, ink),
        PlaceholderKind::Audio => draw_bars(&mut img, cx, cy, ink),
    }
    img
}

fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn draw_play(img: &mut RgbImage, cx: i32, cy: i32, r: i32, c: Rgb<u8>) {
    for dy in -r..=r {
        let width = (r - dy.abs()) * 4 / 3;
        for dx in 0..=width {
            put(img, cx - r / 3 + dx, cy + dy, c);
        }
    }
}

fn draw_bars(img: &mut RgbImage, cx: i32, cy: i32, c: Rgb<u8>) {
    for (i, half) in [14, 28, 40, 24, 12].into_iter().enumerate() {
        let x0 = cx - 40 + i as i32 * 16;
        for dx in 0..10 {
            for dy in -half..=half {
                put(img, x0 + dx, cy + dy, c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::quality::{frame_stats, is_blank};

    #[test]
    fn placeholders_have_the_poster_size_and_visible_symbols() {
        for kind in [PlaceholderKind::Video, PlaceholderKind::Audio] {
            let img = placeholder(kind);
            assert_eq!(img.dimensions(), (PLACEHOLDER_W, PLACEHOLDER_H));
            assert!(!is_blank(&frame_stats(&img)), "{kind:?}");
        }
    }

    #[test]
    fn kinds_differ_and_are_deterministic() {
        assert_ne!(placeholder(PlaceholderKind::Video), placeholder(PlaceholderKind::Audio));
        assert_eq!(placeholder(PlaceholderKind::Video), placeholder(PlaceholderKind::Video));
    }

    #[test]
    fn symbol_sits_in_the_centre() {
        let img = placeholder(PlaceholderKind::Video);
        let centre = img.get_pixel(PLACEHOLDER_W / 2, PLACEHOLDER_H / 2);
        assert!(centre.0[0] > 150, "{centre:?}");
        assert!(img.get_pixel(4, 4).0[0] < 80);
    }
}
