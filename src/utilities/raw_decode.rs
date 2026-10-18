use crate::utilities::guard::guarded;
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader, RgbImage, imageops::FilterType};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

const PROBE_BYTES: u64 = 256 << 10;
const MAX_PREVIEW_BYTES: u64 = 64 << 20;
const SCAN_WINDOW_BYTES: u64 = 8 << 20;
const MAX_DECODE_ATTEMPTS: usize = 3;

pub fn is_raw_ext(ext: &str) -> bool {
    crate::utilities::types::is_raw(ext)
}

pub fn decode_raw_to_png_bytes(path: &Path, max_dim: u32) -> anyhow::Result<Vec<u8>> {
    let img = decode_raw_image(path, max_dim)?;
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)?;
    Ok(out)
}

// Decodes the embedded JPEG preview (full rawkit decode for ARW only when none exists); never panics.
pub fn decode_raw_image(path: &Path, max_dim: u32) -> anyhow::Result<DynamicImage> {
    guarded(
        || decode_raw_image_inner(path, max_dim),
        |msg| anyhow::anyhow!("raw decoder panicked on {}: {msg}", path.display()),
    )
}

fn decode_raw_image_inner(path: &Path, max_dim: u32) -> anyhow::Result<DynamicImage> {
    if let Some((img, orientation)) = embedded_preview(path, max_dim) {
        return Ok(fit(apply_orientation(img, orientation), max_dim));
    }
    let is_arw = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw"));
    if !is_arw {
        anyhow::bail!("no embedded JPEG preview and no full decoder for this RAW format: {}", path.display());
    }
    Ok(fit(rawkit_decode(path)?, max_dim))
}

fn fit(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w.max(h) <= max_dim {
        return img;
    }
    let filter = if max_dim > 512 { FilterType::Lanczos3 } else { FilterType::Triangle };
    img.resize(max_dim, max_dim, filter)
}

fn rawkit_decode(path: &Path) -> anyhow::Result<DynamicImage> {
    guarded(
        || {
            let mut reader = BufReader::new(File::open(path)?);
            let raw = rawkit::RawImage::decode(&mut reader)?;
            let rgb = raw.process_8bit();
            let (w, h) = (rgb.width as u32, rgb.height as u32);
            let img = RgbImage::from_raw(w, h, rgb.data).ok_or_else(|| anyhow::anyhow!("rawkit produced unexpected buffer size"))?;
            Ok(DynamicImage::ImageRgb8(img))
        },
        |msg| anyhow::anyhow!("rawkit panicked on {}: {msg} (unsupported ARW variant, e.g. 14-bit compressed)", path.display()),
    )
}

fn apply_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

#[derive(Clone, Copy)]
enum Endian {
    Le,
    Be,
}

struct Entry {
    tag: u16,
    ty: u16,
    count: u32,
    field: [u8; 4],
}

struct Tiff<R: Read + Seek> {
    r: R,
    e: Endian,
    len: u64,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    offset: u64,
    len: u64,
}

impl<R: Read + Seek> Tiff<R> {
    fn open(mut r: R) -> Option<Self> {
        let len = r.seek(SeekFrom::End(0)).ok()?;
        r.seek(SeekFrom::Start(0)).ok()?;
        let mut h = [0u8; 8];
        r.read_exact(&mut h).ok()?;
        let e = match &h[0..2] {
            b"II" => Endian::Le,
            b"MM" => Endian::Be,
            _ => return None,
        };
        let me = Tiff { r, e, len };
        // 42 = TIFF, 0x4F52 and 0x5352 = ORF, 0x55 = RW2
        if !matches!(me.u16(&h[2..4]), 42 | 0x4F52 | 0x5352 | 0x55) {
            return None;
        }
        Some(me)
    }

    fn u16(&self, b: &[u8]) -> u16 {
        let a = [b[0], b[1]];
        match self.e {
            Endian::Le => u16::from_le_bytes(a),
            Endian::Be => u16::from_be_bytes(a),
        }
    }

    fn u32(&self, b: &[u8]) -> u32 {
        let a = [b[0], b[1], b[2], b[3]];
        match self.e {
            Endian::Le => u32::from_le_bytes(a),
            Endian::Be => u32::from_be_bytes(a),
        }
    }

    fn first_ifd(&mut self) -> Option<u32> {
        self.r.seek(SeekFrom::Start(4)).ok()?;
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).ok()?;
        Some(self.u32(&b))
    }

    fn read_ifd(&mut self, off: u32) -> Option<(Vec<Entry>, u32)> {
        if off == 0 || u64::from(off) + 2 > self.len {
            return None;
        }
        self.r.seek(SeekFrom::Start(off.into())).ok()?;
        let mut n = [0u8; 2];
        self.r.read_exact(&mut n).ok()?;
        let n = self.u16(&n) as usize;
        if n == 0 || n > 512 {
            return None;
        }
        let mut buf = vec![0u8; n * 12 + 4];
        self.r.read_exact(&mut buf).ok()?;
        let entries = (0..n)
            .map(|i| {
                let c = &buf[i * 12..i * 12 + 12];
                Entry { tag: self.u16(&c[0..2]), ty: self.u16(&c[2..4]), count: self.u32(&c[4..8]), field: [c[8], c[9], c[10], c[11]] }
            })
            .collect();
        Some((entries, self.u32(&buf[n * 12..n * 12 + 4])))
    }

    fn scalar(&self, en: &Entry) -> Option<u32> {
        match en.ty {
            3 => Some(u32::from(self.u16(&en.field[0..2]))),
            4 => Some(self.u32(&en.field)),
            _ => None,
        }
    }

    fn longs(&mut self, en: &Entry) -> Option<Vec<u32>> {
        if !matches!(en.ty, 4 | 13) || en.count == 0 || en.count > 16 {
            return None;
        }
        if en.count == 1 {
            return Some(vec![self.u32(&en.field)]);
        }
        self.r.seek(SeekFrom::Start(self.u32(&en.field).into())).ok()?;
        let mut b = vec![0u8; en.count as usize * 4];
        self.r.read_exact(&mut b).ok()?;
        Some(b.chunks_exact(4).map(|c| self.u32(c)).collect())
    }
}

// JPEG byte ranges referenced from IFD0, its chain and SubIFDs, plus the IFD0 orientation.
fn tiff_candidates(path: &Path) -> Option<(Vec<Candidate>, u16)> {
    let mut t = Tiff::open(BufReader::new(File::open(path).ok()?))?;
    let mut queue: Vec<u32> = vec![t.first_ifd()?];
    let mut seen: Vec<u32> = Vec::new();
    let mut out: Vec<Candidate> = Vec::new();
    let (mut orientation, mut first) = (1u16, true);
    while let Some(off) = queue.pop() {
        if seen.contains(&off) || seen.len() > 32 {
            continue;
        }
        seen.push(off);
        let Some((entries, next)) = t.read_ifd(off) else { continue };
        let (mut jif, mut jif_len) = (None, None);
        let (mut compression, mut photometric) = (0u32, 0u32);
        let (mut strip_off, mut strip_len, mut strip_cnt) = (None, None, 0u32);
        for en in &entries {
            match en.tag {
                0x0103 => compression = t.scalar(en).unwrap_or(0),
                0x0106 => photometric = t.scalar(en).unwrap_or(0),
                0x0111 => {
                    strip_off = t.scalar(en);
                    strip_cnt = en.count;
                }
                0x0117 => strip_len = t.scalar(en),
                0x0112 if first => orientation = t.scalar(en).unwrap_or(1) as u16,
                0x0201 => jif = t.scalar(en),
                0x0202 => jif_len = t.scalar(en),
                0x014A => {
                    if let Some(v) = t.longs(en) {
                        queue.extend(v);
                    }
                }
                _ => {}
            }
        }
        first = false;
        if let (Some(o), Some(l)) = (jif, jif_len) {
            out.push(Candidate { offset: o.into(), len: l.into() });
        }
        // Single-strip JPEG IFD that is not CFA or linear-raw data.
        if matches!(compression, 6 | 7) && !matches!(photometric, 32803 | 34892) && strip_cnt == 1 {
            if let (Some(o), Some(l)) = (strip_off, strip_len) {
                out.push(Candidate { offset: o.into(), len: l.into() });
            }
        }
        if next != 0 {
            queue.push(next);
        }
    }
    Some((out, orientation))
}

fn raf_candidate(path: &Path) -> Option<Candidate> {
    let mut f = File::open(path).ok()?;
    let mut h = [0u8; 92];
    f.read_exact(&mut h).ok()?;
    if &h[0..15] != b"FUJIFILMCCD-RAW" {
        return None;
    }
    Some(Candidate {
        offset: u32::from_be_bytes([h[84], h[85], h[86], h[87]]).into(),
        len: u32::from_be_bytes([h[88], h[89], h[90], h[91]]).into(),
    })
}

fn scan_candidates(path: &Path) -> Vec<Candidate> {
    let mut buf = Vec::new();
    let Ok(f) = File::open(path) else { return Vec::new() };
    if f.take(SCAN_WINDOW_BYTES).read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= buf.len() {
        match buf[i..].iter().position(|&b| b == 0xFF) {
            Some(p) => i += p,
            None => break,
        }
        if i + 3 <= buf.len() && buf[i + 1] == 0xD8 && buf[i + 2] == 0xFF {
            out.push(Candidate { offset: i as u64, len: (buf.len() - i) as u64 });
        }
        i += 1;
    }
    out
}

fn read_range(path: &Path, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = Vec::new();
    f.take(len).read_to_end(&mut b)?;
    Ok(b)
}

fn probe_dimensions(path: &Path, c: &Candidate) -> Option<(u32, u32)> {
    let head = read_range(path, c.offset, c.len.min(PROBE_BYTES)).ok()?;
    if head.len() < 4 || head[0..3] != [0xFF, 0xD8, 0xFF] {
        return None;
    }
    ImageReader::with_format(Cursor::new(head), ImageFormat::Jpeg).into_dimensions().ok()
}

// Smallest preview whose long side covers `target`, then the remaining ones from largest to smallest.
fn preference_order(mut probed: Vec<(Candidate, u32, u32)>, target: u32) -> Vec<Candidate> {
    probed.sort_by_key(|(_, w, h)| {
        let long = (*w).max(*h);
        if long >= target { (0u8, long) } else { (1u8, u32::MAX - long) }
    });
    probed.into_iter().map(|(c, _, _)| c).collect()
}

fn embedded_preview(path: &Path, target: u32) -> Option<(DynamicImage, u16)> {
    let (mut cands, orientation) = tiff_candidates(path).unwrap_or((Vec::new(), 1));
    if let Some(c) = raf_candidate(path) {
        cands.push(c);
    }
    if cands.is_empty() {
        cands = scan_candidates(path);
    }
    let probed: Vec<(Candidate, u32, u32)> =
        cands.into_iter().filter_map(|c| probe_dimensions(path, &c).map(|(w, h)| (c, w, h))).collect();
    for c in preference_order(probed, target).into_iter().take(MAX_DECODE_ATTEMPTS) {
        let Ok(bytes) = read_range(path, c.offset, c.len.min(MAX_PREVIEW_BYTES)) else { continue };
        match image::load_from_memory_with_format(&bytes, ImageFormat::Jpeg) {
            Ok(img) => return Some((img, orientation)),
            Err(e) => log::warn!("[raw] embedded preview undecodable {}: {e}", path.display()),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn jpeg(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
        let img: RgbImage = ImageBuffer::from_pixel(w, h, Rgb(color));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Jpeg).unwrap();
        out
    }

    struct Ifd {
        entries: Vec<(u16, u16, u32, [u8; 4])>,
    }

    fn put16(le: bool, v: u16) -> [u8; 2] {
        if le { v.to_le_bytes() } else { v.to_be_bytes() }
    }

    fn put32(le: bool, v: u32) -> [u8; 4] {
        if le { v.to_le_bytes() } else { v.to_be_bytes() }
    }

    // IFD0 with Orientation and JPEGInterchangeFormat (small preview), plus a SubIFD holding a JPEG strip.
    fn synthetic_tiff(le: bool, orientation: u16, ifd0_jpeg: Option<&[u8]>, sub_jpeg: Option<&[u8]>) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        let ifd0_entries = 1 + if ifd0_jpeg.is_some() { 2 } else { 0 } + if sub_jpeg.is_some() { 1 } else { 0 };
        let ifd0_size = 2 + ifd0_entries * 12 + 4;
        let sub_size = 2 + 5 * 12 + 4;
        let ifd0_off = 8u32;
        let sub_off = ifd0_off + ifd0_size as u32;
        let mut cursor = sub_off + if sub_jpeg.is_some() { sub_size as u32 } else { 0 };
        let mut ifd0 = Ifd { entries: Vec::new() };
        ifd0.entries.push((0x0112, 3, 1, { let mut f = [0u8; 4]; f[..2].copy_from_slice(&put16(le, orientation)); f }));
        let mut blobs: Vec<(u32, &[u8])> = Vec::new();
        if let Some(j) = ifd0_jpeg {
            ifd0.entries.push((0x0201, 4, 1, put32(le, cursor)));
            ifd0.entries.push((0x0202, 4, 1, put32(le, j.len() as u32)));
            blobs.push((cursor, j));
            cursor += j.len() as u32;
        }
        let mut sub = Ifd { entries: Vec::new() };
        if let Some(j) = sub_jpeg {
            ifd0.entries.push((0x014A, 4, 1, put32(le, sub_off)));
            sub.entries.push((0x0103, 3, 1, { let mut f = [0u8; 4]; f[..2].copy_from_slice(&put16(le, 7)); f }));
            sub.entries.push((0x0106, 3, 1, { let mut f = [0u8; 4]; f[..2].copy_from_slice(&put16(le, 6)); f }));
            sub.entries.push((0x0111, 4, 1, put32(le, cursor)));
            sub.entries.push((0x0117, 4, 1, put32(le, j.len() as u32)));
            sub.entries.push((0x0115, 3, 1, { let mut f = [0u8; 4]; f[..2].copy_from_slice(&put16(le, 3)); f }));
            blobs.push((cursor, j));
        }
        data.extend_from_slice(if le { b"II" } else { b"MM" });
        data.extend_from_slice(&put16(le, 42));
        data.extend_from_slice(&put32(le, ifd0_off));
        let write_ifd = |data: &mut Vec<u8>, ifd: &mut Ifd| {
            ifd.entries.sort_by_key(|e| e.0);
            data.extend_from_slice(&put16(le, ifd.entries.len() as u16));
            for (tag, ty, count, field) in &ifd.entries {
                data.extend_from_slice(&put16(le, *tag));
                data.extend_from_slice(&put16(le, *ty));
                data.extend_from_slice(&put32(le, *count));
                data.extend_from_slice(field);
            }
            data.extend_from_slice(&put32(le, 0));
        };
        write_ifd(&mut data, &mut ifd0);
        if sub_jpeg.is_some() {
            write_ifd(&mut data, &mut sub);
        }
        for (off, blob) in blobs {
            assert_eq!(data.len() as u32, off);
            data.extend_from_slice(blob);
        }
        data
    }

    fn write_tmp(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        (dir, p)
    }

    #[test]
    fn preview_in_ifd0_little_endian() {
        let (_d, p) = write_tmp("a.nef", &synthetic_tiff(true, 1, Some(&jpeg(64, 32, [200, 10, 10])), None));
        let img = decode_raw_image(&p, 256).unwrap();
        assert_eq!(img.dimensions(), (64, 32));
    }

    #[test]
    fn preview_in_ifd0_big_endian() {
        let (_d, p) = write_tmp("a.nef", &synthetic_tiff(false, 1, Some(&jpeg(48, 24, [10, 200, 10])), None));
        assert_eq!(decode_raw_image(&p, 256).unwrap().dimensions(), (48, 24));
    }

    #[test]
    fn orientation_rotates_preview() {
        let (_d, p) = write_tmp("a.dng", &synthetic_tiff(true, 6, Some(&jpeg(64, 32, [1, 2, 3])), None));
        assert_eq!(decode_raw_image(&p, 256).unwrap().dimensions(), (32, 64));
    }

    #[test]
    fn long_side_is_capped() {
        let (_d, p) = write_tmp("a.nef", &synthetic_tiff(true, 1, Some(&jpeg(512, 256, [9, 9, 9])), None));
        let img = decode_raw_image(&p, 128).unwrap();
        assert_eq!(img.dimensions(), (128, 64));
    }

    #[test]
    fn smallest_covering_preview_is_used() {
        let tiff = synthetic_tiff(true, 1, Some(&jpeg(40, 20, [255, 0, 0])), Some(&jpeg(160, 80, [0, 0, 255])));
        let (_d, p) = write_tmp("a.nef", &tiff);
        let small = decode_raw_image(&p, 32).unwrap();
        assert_eq!(small.dimensions(), (32, 16));
        let px = small.to_rgb8().get_pixel(4, 4).0;
        assert!(px[0] > 200 && px[2] < 60, "expected the red 40x20 preview, got {px:?}");
        let big = decode_raw_image(&p, 120).unwrap();
        assert_eq!(big.dimensions(), (120, 60));
        let px = big.to_rgb8().get_pixel(4, 4).0;
        assert!(px[2] > 200 && px[0] < 60, "expected the blue 160x80 preview, got {px:?}");
    }

    #[test]
    fn largest_preview_when_none_covers_target() {
        let tiff = synthetic_tiff(true, 1, Some(&jpeg(40, 20, [255, 0, 0])), Some(&jpeg(160, 80, [0, 0, 255])));
        let (_d, p) = write_tmp("a.nef", &tiff);
        assert_eq!(decode_raw_image(&p, 1600).unwrap().dimensions(), (160, 80));
    }

    #[test]
    fn jpeg_found_by_signature_scan_without_tiff() {
        let mut bytes = vec![0u8; 5000];
        bytes.extend_from_slice(&jpeg(30, 30, [5, 50, 5]));
        let (_d, p) = write_tmp("a.cr3", &bytes);
        assert_eq!(decode_raw_image(&p, 256).unwrap().dimensions(), (30, 30));
    }

    #[test]
    fn no_preview_non_arw_is_an_error() {
        let (_d, p) = write_tmp("a.nef", &synthetic_tiff(true, 1, None, None));
        let err = decode_raw_image(&p, 256).unwrap_err().to_string();
        assert!(err.contains("no embedded JPEG preview"), "{err}");
    }

    #[test]
    fn arw_without_preview_never_panics() {
        let (_d, p) = write_tmp("a.arw", &synthetic_tiff(true, 1, None, None));
        assert!(decode_raw_image(&p, 256).is_err());
        assert!(decode_raw_to_png_bytes(&p, 256).is_err());
    }

    #[test]
    fn garbage_and_missing_files_are_errors() {
        let (_d, p) = write_tmp("a.dng", &[7u8; 4096]);
        assert!(decode_raw_image(&p, 256).is_err());
        assert!(decode_raw_image(Path::new("/nonexistent/none.arw"), 256).is_err());
        let (_d2, empty) = write_tmp("e.arw", &[]);
        assert!(decode_raw_image(&empty, 256).is_err());
    }

    #[test]
    fn png_bytes_roundtrip() {
        let (_d, p) = write_tmp("a.nef", &synthetic_tiff(true, 1, Some(&jpeg(64, 32, [90, 90, 90])), None));
        let png = decode_raw_to_png_bytes(&p, 256).unwrap();
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
    }

    fn real_arw() -> Option<std::path::PathBuf> {
        std::env::var_os("SMART_MEDIA_TEST_ARW").map(std::path::PathBuf::from).filter(|p| p.exists())
    }

    #[test]
    fn real_arw_uses_embedded_preview() {
        let Some(arw) = real_arw() else { return };
        let t = std::time::Instant::now();
        let img = decode_raw_image(&arw, 256).unwrap();
        eprintln!("real ARW thumbnail {:?} in {:?}", img.dimensions(), t.elapsed());
        assert!(img.dimensions().0.max(img.dimensions().1) <= 256);
        let big = decode_raw_image(&arw, 1600).unwrap();
        assert!(big.dimensions().0.max(big.dimensions().1) > 256);
    }

    // Copy of the real ARW with every JPEG signature and preview pointer destroyed so only rawkit remains.
    #[test]
    fn real_arw_without_preview_reports_rawkit_panic() {
        let Some(arw) = real_arw() else { return };
        let mut bytes = std::fs::read(&arw).unwrap();
        let mut i = 0;
        while i + 3 <= bytes.len() {
            if bytes[i..i + 3] == [0xFF, 0xD8, 0xFF] {
                bytes[i] = 0x00;
            }
            i += 1;
        }
        let (_d, p) = write_tmp("no_preview.arw", &bytes);
        let err = decode_raw_image(&p, 256).unwrap_err().to_string();
        eprintln!("rawkit path error: {err}");
        assert!(err.contains("rawkit panicked") || err.contains("rawkit") || err.contains("unsupported"), "{err}");
    }
}
