use std::collections::HashSet;
use std::path::Path;

use egui::{Image, Vec2};

use super::glyphs::{drive_icon, file_icon_svg, folder_shortcut_svg};
use super::pack::{ALL, BORROWED, IconPack, PACK};
use super::svg;
use crate::utilities::explorer::DriveType;

const ICON_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/icons");

fn stems(dir: &str) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(Path::new(ICON_DIR).join(dir))
        .expect("icon dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "svg"))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
        .collect();
    out.sort();
    out
}

fn rasterise(name: &str, bytes: &[u8], px: u32) -> resvg::tiny_skia::Pixmap {
    let text = std::str::from_utf8(bytes).unwrap_or_else(|e| panic!("{name}: not UTF-8: {e}"));
    assert!(text.trim_start().starts_with("<svg"), "{name}: not an <svg> root");
    assert!(!text.contains("currentColor"), "{name}: currentColor cannot be tinted");
    let tree = resvg::usvg::Tree::from_data(bytes, &resvg::usvg::Options::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(tree.size().width(), 256.0, "{name}: not on the 256 grid");
    assert_eq!(tree.size().height(), 256.0, "{name}: not on the 256 grid");
    let mut pixmap = resvg::tiny_skia::Pixmap::new(px, px).expect("non-zero pixmap");
    let scale = px as f32 / 256.0;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    pixmap
}

fn coverage(pixmap: &resvg::tiny_skia::Pixmap) -> f32 {
    let lit = pixmap.pixels().iter().filter(|p| p.alpha() > 127).count();
    lit as f32 / pixmap.pixels().len() as f32
}

#[test]
fn names_sorted_unique_kebab_case() {
    assert!(ALL.windows(2).all(|w| w[0].name() < w[1].name()));
    let uris: HashSet<_> = ALL.iter().map(|i| i.source()).map(|s| format!("{s:?}")).collect();
    assert_eq!(uris.len(), ALL.len());
    assert!(ALL.len() >= 200, "pack shrank to {}", ALL.len());
    for icon in ALL {
        let name = icon.name();
        assert!(name.starts_with(|c: char| c.is_ascii_lowercase()), "{name}");
        assert!(
            name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{name}"
        );
        assert!(!name.ends_with('-') && !name.contains("--"), "{name}");
    }
}

#[test]
fn every_svg_file_has_a_const_and_back() {
    let files = stems("regular");
    let names: Vec<_> = ALL.iter().map(|i| i.name().to_owned()).collect();
    assert_eq!(files, names);
}

#[test]
fn find_round_trips() {
    for icon in ALL {
        assert_eq!(PACK.find(icon.name()), Some(*icon));
    }
    assert_eq!(PACK.find("cpu"), Some(svg::CPU));
    assert_eq!(PACK.find("ame-logo"), Some(svg::AME_LOGO));
    assert_eq!(PACK.find("no-such-icon"), None);
}

#[test]
fn from_glyph_round_trips_phosphor_icons() {
    for icon in ALL.iter().filter(|i| !i.is_custom()) {
        let found = PACK.from_glyph(icon.glyph()).expect(icon.name());
        assert_eq!(found.glyph(), icon.glyph(), "{}", icon.name());
        assert!(!found.is_custom());
    }
    assert_eq!(PACK.from_glyph(egui_phosphor::regular::CPU), Some(svg::CPU));
    assert_eq!(
        PACK.from_glyph(svg::AME_LOGO.glyph()).map(|i| i.is_custom()),
        Some(false)
    );
    assert_eq!(PACK.from_glyph("x"), None);
}

#[test]
fn custom_icons_borrow_a_phosphor_glyph() {
    let custom: Vec<_> = ALL.iter().filter(|i| i.name().starts_with("ame-")).collect();
    assert!(custom.len() >= 25, "only {} custom icons", custom.len());
    for icon in &custom {
        assert!(BORROWED.contains(&icon.name()), "{}", icon.name());
        assert!(icon.is_custom());
        let svg = std::str::from_utf8(icon.svg()).unwrap();
        assert!(svg.contains("data-glyph=\""), "{}: missing data-glyph", icon.name());
    }
    assert_eq!(BORROWED.len(), custom.len(), "non-ame icon declares data-glyph");
    for icon in ALL.iter().filter(|i| !i.name().starts_with("ame-")) {
        assert!(!icon.is_custom(), "{}", icon.name());
    }
}

#[test]
fn custom_icons_follow_drawing_conventions() {
    for icon in ALL.iter().filter(|i| i.is_custom()) {
        let svg = std::str::from_utf8(icon.svg()).unwrap();
        for needle in [
            "viewBox=\"0 0 256 256\"",
            "stroke=\"#ffffff\"",
            "stroke-width=\"16\"",
            "stroke-linecap=\"round\"",
            "stroke-linejoin=\"round\"",
        ] {
            assert!(svg.contains(needle), "{}: missing {needle}", icon.name());
        }
        assert!(!svg.contains("<image") && !svg.contains("<text"), "{}", icon.name());
        assert!(!svg.contains("Gradient") && !svg.contains("<filter"), "{}", icon.name());
        if let Some(duo) = icon.duo_svg() {
            let duo = std::str::from_utf8(duo).unwrap();
            assert!(duo.contains("fill=\"#ffffff\""), "{}: duo layer is not white", icon.name());
        }
    }
}

#[test]
fn gallery_draws_without_panicking() {
    let ctx = egui::Context::default();
    super::install(&ctx);
    super::gallery::set_open(&ctx, true);
    assert!(super::gallery::is_open(&ctx));
    for _ in 0..2 {
        let mut output = ctx.run_ui(Default::default(), |ui| {
            super::gallery::show_window(ui.ctx());
            super::gallery::show(ui);
        });
        output.textures_delta.clear();
    }
    super::gallery::toggle(&ctx);
    assert!(!super::gallery::is_open(&ctx));
}

#[test]
fn glyphs_are_single_phosphor_chars() {
    for icon in ALL {
        let mut chars = icon.glyph().chars();
        let c = chars
            .next()
            .unwrap_or_else(|| panic!("{} has an empty glyph", icon.name()));
        assert!(chars.next().is_none(), "{} glyph is not one char", icon.name());
        assert!(
            ('\u{E000}'..='\u{F8FF}').contains(&c),
            "{} glyph is outside the PUA",
            icon.name()
        );
    }
}

#[test]
fn uris_end_in_svg() {
    for icon in ALL {
        let uri = format!("{:?}", icon.source());
        assert!(uri.contains("bytes://ame-icons/") && uri.contains(".svg"), "{uri}");
        if let Some(duo) = icon.duo_source() {
            let duo_uri = format!("{duo:?}");
            assert!(duo_uri.contains("bytes://ame-icons/duo/") && duo_uri != uri, "{duo_uri}");
        }
    }
}

#[test]
fn every_layer_parses_and_renders_white() {
    for icon in ALL {
        let small = rasterise(icon.name(), icon.svg(), 16);
        assert!(small.pixels().iter().any(|p| p.alpha() > 0), "{}: empty at 16 px", icon.name());
        let front = rasterise(icon.name(), icon.svg(), 48);
        let cov = coverage(&front);
        assert!(
            (0.003..0.75).contains(&cov),
            "{}: front coverage {cov}",
            icon.name()
        );
        for p in front.pixels().iter().filter(|p| p.alpha() == 255) {
            assert_eq!(
                (p.red(), p.green(), p.blue()),
                (255, 255, 255),
                "{}: non-white ink",
                icon.name()
            );
        }
        if let Some(duo) = icon.duo_svg() {
            assert!(
                coverage(&rasterise(icon.name(), duo, 48)) > 0.003,
                "{}: empty duo layer",
                icon.name()
            );
        }
    }
}

#[test]
fn duo_layers_pair_with_regular_icons() {
    let regular: HashSet<_> = stems("regular").into_iter().collect();
    let duo = stems("duo");
    assert!(!duo.is_empty());
    for name in &duo {
        assert!(regular.contains(name), "duo/{name}.svg has no regular/{name}.svg");
        assert!(PACK.find(name).is_some_and(|i| i.has_duo()), "{name}: duo not wired");
    }
    assert_eq!(ALL.iter().filter(|i| i.has_duo()).count(), duo.len());
    for hero in [svg::AME_LOGO, svg::AME_SCAN, svg::AME_ASSISTANT, svg::AME_THUMBNAIL_GRID] {
        assert!(hero.has_duo(), "{}", hero.name());
    }
}

#[test]
fn loads_through_egui_extras_svg_loader() {
    let ctx = egui::Context::default();
    super::install(&ctx);
    for icon in [svg::CPU, svg::HARD_DRIVE, svg::AME_LOGO, svg::AME_SCAN] {
        let duo = icon.duo_source().expect("duo layer");
        for source in [icon.source(), duo] {
            let image = Image::new(source).fit_to_exact_size(Vec2::splat(24.0));
            match image.load_for_size(&ctx, Vec2::splat(100.0)) {
                Ok(egui::load::TexturePoll::Ready { texture }) => {
                    let raster = ctx.tex_manager().read().meta(texture.id).map(|m| m.size);
                    assert_eq!(raster, Some([24, 24]), "{}", icon.name());
                }
                Ok(_) => panic!("{}: still loading", icon.name()),
                Err(e) => panic!("{}: {e}", icon.name()),
            }
        }
    }
}

#[test]
fn file_icons_follow_app_media_types() {
    assert_eq!(file_icon_svg("a", true), svg::FOLDER);
    assert_eq!(file_icon_svg("shot.JPG", false), svg::FILE_JPG);
    assert_eq!(file_icon_svg("shot.png", false), svg::FILE_PNG);
    assert_eq!(file_icon_svg("shot.webp", false), svg::FILE_IMAGE);
    assert_eq!(file_icon_svg("DSC0001.ARW", false), svg::AME_RAW_PHOTO);
    assert_eq!(file_icon_svg("clip.mkv", false), svg::FILE_VIDEO);
    assert_eq!(file_icon_svg("pics.7z", false), svg::FILE_ZIP);
    assert_eq!(file_icon_svg("backup.surql", false), svg::FILE_SQL);
    assert_eq!(file_icon_svg("noext", false), svg::FILE);
    assert_eq!(super::file_icon("shot.png", false), svg::FILE_PNG.glyph());
}

#[test]
fn places_and_drives_map_to_icons() {
    assert_eq!(folder_shortcut_svg("Pictures"), svg::IMAGE);
    assert_eq!(folder_shortcut_svg("Home"), svg::HOUSE);
    assert_eq!(folder_shortcut_svg("Unknown"), svg::FOLDER);
    assert_eq!(drive_icon(&DriveType::Usb), svg::USB);
    assert_eq!(drive_icon(&DriveType::HardDrive), svg::HARD_DRIVE);
}
