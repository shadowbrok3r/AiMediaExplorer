//! Bundled Inter UI faces, the `medium` / `semibold` weight families, the app font stack and the user text scale.

mod sfnt;

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock, OnceLock};

use eframe::egui::{Context, FontData, FontDefinitions, FontFamily, FontId, Id, RichText};

/// `font_data` key of Inter Regular, the first face of `Proportional`.
pub const INTER: &str = "Inter";
/// `font_data` key of Inter Medium.
pub const INTER_MEDIUM: &str = "Inter-Medium";
/// `font_data` key of Inter SemiBold.
pub const INTER_SEMIBOLD: &str = "Inter-SemiBold";
/// Name of the medium-weight font family.
pub const MEDIUM: &str = "medium";
/// Name of the semibold-weight font family.
pub const SEMIBOLD: &str = "semibold";
/// `font_data` key of the face placed first in `Monospace`.
pub const UBUNTU_SANS_MONO: &str = "UbuntuSansMono";
/// `font_data` key of the Nerd Font face placed second in `Monospace`.
pub const UBUNTU_MONO_NERD: &str = "UbuntuMonoNerdFont";

const INTER_REGULAR_TTF: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");
const INTER_MEDIUM_TTF: &[u8] = include_bytes!("../../assets/fonts/Inter-Medium.ttf");
const INTER_SEMIBOLD_TTF: &[u8] = include_bytes!("../../assets/fonts/Inter-SemiBold.ttf");
const MONASPACE_REGULAR_OTF: &[u8] = include_bytes!("../../assets/fonts/MonaspaceNeon-Regular.otf");
const MONASPACE_BOLD_OTF: &[u8] = include_bytes!("../../assets/fonts/MonaspaceNeon-Bold.otf");
const UBUNTU_SANS_MONO_OTF: &[u8] = include_bytes!("../../assets/fonts/UbuntuSansMono-Regular.otf");
const UBUNTU_MONO_NERD_TTF: &[u8] =
    include_bytes!("../../assets/fonts/UbuntuMonoNerdFont-Regular.ttf");

/// The medium-weight family.
pub(crate) static MEDIUM_FAMILY: LazyLock<FontFamily> =
    LazyLock::new(|| FontFamily::Name(MEDIUM.into()));
/// The semibold-weight family.
pub(crate) static SEMIBOLD_FAMILY: LazyLock<FontFamily> =
    LazyLock::new(|| FontFamily::Name(SEMIBOLD.into()));

static REGULAR_FACE: OnceLock<Cow<'static, [u8]>> = OnceLock::new();
static MEDIUM_FACE: OnceLock<Cow<'static, [u8]>> = OnceLock::new();
static SEMIBOLD_FACE: OnceLock<Cow<'static, [u8]>> = OnceLock::new();
static NERD_FACE: OnceLock<Cow<'static, [u8]>> = OnceLock::new();

/// `raw` with its BMP Private Use Area cmap entries removed, cached in `cell` for the process.
pub fn without_private_use(
    cell: &'static OnceLock<Cow<'static, [u8]>>,
    raw: &'static [u8],
) -> &'static [u8] {
    cell.get_or_init(|| match sfnt::without_private_use(raw) {
        Some(stripped) => Cow::Owned(stripped),
        None => {
            log::warn!("fonts: cmap rewrite failed; icon glyphs may resolve to another face");
            Cow::Borrowed(raw)
        }
    })
}

/// Puts Inter first in `Proportional` and adds the weight families; call after any icon face is registered.
pub fn install(fonts: &mut FontDefinitions) {
    let faces = [
        (INTER, without_private_use(&REGULAR_FACE, INTER_REGULAR_TTF)),
        (
            INTER_MEDIUM,
            without_private_use(&MEDIUM_FACE, INTER_MEDIUM_TTF),
        ),
        (
            INTER_SEMIBOLD,
            without_private_use(&SEMIBOLD_FACE, INTER_SEMIBOLD_TTF),
        ),
    ];
    for (key, bytes) in faces {
        fonts
            .font_data
            .insert(key.to_owned(), Arc::new(FontData::from_static(bytes)));
    }

    let proportional = fonts.families.entry(FontFamily::Proportional).or_default();
    proportional.retain(|k| k != INTER);
    proportional.insert(0, INTER.to_owned());
    let fallback = proportional.clone();

    for (family, face) in [(MEDIUM, INTER_MEDIUM), (SEMIBOLD, INTER_SEMIBOLD)] {
        let stack = std::iter::once(face.to_owned())
            .chain(fallback.iter().filter(|k| *k != face).cloned())
            .collect();
        fonts
            .families
            .insert(FontFamily::Name(family.into()), stack);
    }
}

/// Registers icon font faces; Inter installs after this so icon codepoints resolve to the icon face.
fn install_icon_faces(fonts: &mut FontDefinitions) {
    crate::icons::install_fonts(fonts);
}

/// The app's font stack at `scale`: Inter UI faces, the Ubuntu monospace stack and the Monaspace families.
pub fn definitions(scale: TextScale) -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    let faces: [(&str, &'static [u8]); 5] = [
        ("Monaspace", MONASPACE_REGULAR_OTF),
        (UBUNTU_SANS_MONO, UBUNTU_SANS_MONO_OTF),
        (
            UBUNTU_MONO_NERD,
            without_private_use(&NERD_FACE, UBUNTU_MONO_NERD_TTF),
        ),
        ("Regular", MONASPACE_REGULAR_OTF),
        ("Bold", MONASPACE_BOLD_OTF),
    ];
    for (key, bytes) in faces {
        fonts
            .font_data
            .insert(key.to_owned(), Arc::new(FontData::from_static(bytes)));
    }

    let monospace = fonts.families.entry(FontFamily::Monospace).or_default();
    monospace.insert(0, UBUNTU_SANS_MONO.to_owned());
    monospace.insert(1, UBUNTU_MONO_NERD.to_owned());

    for name in ["Regular", "Bold"] {
        fonts
            .families
            .insert(FontFamily::Name(name.into()), vec![name.to_owned()]);
    }

    install_icon_faces(&mut fonts);
    install(&mut fonts);
    apply_text_scale(&mut fonts, scale);
    fonts
}

/// Installs the app font stack at the scale recorded on `ctx`.
pub fn install_for_context(ctx: &Context) {
    ctx.set_fonts(definitions(text_scale(ctx)));
}

/// Inter Medium at `size`.
pub fn medium(size: f32) -> FontId {
    FontId::new(size, MEDIUM_FAMILY.clone())
}

/// Inter SemiBold at `size`.
pub fn semibold(size: f32) -> FontId {
    FontId::new(size, SEMIBOLD_FAMILY.clone())
}

/// The monospace stack at `size`.
pub fn mono(size: f32) -> FontId {
    FontId::monospace(size)
}

/// Weight shorthands for [`RichText`]; the size still comes from the text style.
pub trait RichTextExt {
    fn medium(self) -> RichText;
    fn semibold(self) -> RichText;
}

impl RichTextExt for RichText {
    fn medium(self) -> RichText {
        self.family(MEDIUM_FAMILY.clone())
    }

    fn semibold(self) -> RichText {
        self.family(SEMIBOLD_FAMILY.clone())
    }
}

/// User text size, applied as a glyph scale on the UI faces rather than as a zoom factor.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum TextScale {
    Small = 90,
    #[default]
    Normal = 100,
    Large = 115,
    XLarge = 130,
}

impl TextScale {
    pub const ALL: [Self; 4] = [Self::Small, Self::Normal, Self::Large, Self::XLarge];

    pub fn percent(self) -> u16 {
        self as u16
    }

    pub fn factor(self) -> f32 {
        f32::from(self.percent()) / 100.0
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Small => "Small (90%)",
            Self::Normal => "Normal (100%)",
            Self::Large => "Large (115%)",
            Self::XLarge => "Extra large (130%)",
        }
    }
}

/// Multiplies the glyph scale of every UI text face by `scale`; expects freshly built definitions.
pub fn apply_text_scale(fonts: &mut FontDefinitions, scale: TextScale) {
    if scale == TextScale::Normal {
        return;
    }
    let families = [
        FontFamily::Proportional,
        FontFamily::Monospace,
        FontFamily::Name(MEDIUM.into()),
        FontFamily::Name(SEMIBOLD.into()),
    ];
    let faces: BTreeSet<String> = families
        .iter()
        .filter_map(|family| fonts.families.get(family))
        .flatten()
        .cloned()
        .collect();
    for key in faces {
        if let Some(data) = fonts.font_data.get_mut(&key) {
            Arc::make_mut(data).tweak.scale *= scale.factor();
        }
    }
}

fn text_scale_id() -> Id {
    Id::unique("smart_media.fonts.text_scale")
}

/// The text scale last installed on `ctx`.
pub fn text_scale(ctx: &Context) -> TextScale {
    ctx.data(|d| d.get_temp::<TextScale>(text_scale_id()))
        .unwrap_or_default()
}

/// Records `scale` on `ctx` and installs `build(scale)` when it differs from the recorded scale.
pub fn set_text_scale(
    ctx: &Context,
    scale: TextScale,
    build: impl FnOnce(TextScale) -> FontDefinitions,
) {
    if text_scale(ctx) == scale {
        return;
    }
    ctx.data_mut(|d| d.insert_temp(text_scale_id(), scale));
    ctx.set_fonts(build(scale));
    ctx.request_repaint();
}

/// [`set_text_scale`] with the app font stack.
pub fn apply_scale(ctx: &Context, scale: TextScale) {
    set_text_scale(ctx, scale, definitions);
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::epaint::text::{Fonts, TextOptions};

    const PUA: std::ops::RangeInclusive<u32> = 0xE000..=0xF8FF;
    /// Powerline arrow, mapped by the Nerd Font and by Inter's raw cmap.
    const POWERLINE: char = '\u{E0B0}';

    fn app_fonts() -> FontDefinitions {
        definitions(TextScale::Normal)
    }

    fn with_single_face(mut fonts: FontDefinitions, family: &str, face: &str) -> FontDefinitions {
        fonts
            .families
            .insert(FontFamily::Name(family.into()), vec![face.to_owned()]);
        fonts
    }

    #[test]
    fn every_inter_face_drops_its_private_use_mappings_and_keeps_the_rest() {
        for raw in [INTER_REGULAR_TTF, INTER_MEDIUM_TTF, INTER_SEMIBOLD_TTF] {
            let before = sfnt::mapped_codepoints(raw).expect("Inter has a format 12 cmap");
            let stripped = sfnt::without_private_use(raw).expect("Inter's cmap rewrites in place");
            let after = sfnt::mapped_codepoints(&stripped).expect("rewritten cmap parses");

            assert!(before.iter().any(|c| PUA.contains(c)));
            assert!(after.iter().all(|c| !PUA.contains(c)));
            let expected: Vec<u32> = before.into_iter().filter(|c| !PUA.contains(c)).collect();
            assert_eq!(after, expected);
            assert_eq!(stripped.len(), raw.len());
        }
    }

    #[test]
    fn the_nerd_font_drops_its_private_use_mappings() {
        let raw = UBUNTU_MONO_NERD_TTF;
        let stripped = without_private_use(&NERD_FACE, raw);
        let before = sfnt::mapped_codepoints(raw).expect("Nerd Font has a format 12 cmap");
        let after = sfnt::mapped_codepoints(stripped).expect("rewritten cmap parses");
        assert!(before.iter().any(|c| PUA.contains(c)));
        assert!(after.iter().all(|c| !PUA.contains(c)));
        assert_eq!(stripped.len(), raw.len());
    }

    #[test]
    fn private_use_codepoints_fall_through_to_the_icon_face_in_every_ui_family() {
        let defs = with_single_face(app_fonts(), "icon-only", "phosphor");
        let ui_families = [
            FontFamily::Proportional,
            FontFamily::Name(MEDIUM.into()),
            FontFamily::Name(SEMIBOLD.into()),
        ];
        let glyph = egui_phosphor::regular::GEAR.chars().next().expect("glyph");
        let mut fonts = Fonts::new(TextOptions::default(), defs);
        let mut view = fonts.with_pixels_per_point(1.0);
        let icon = view.glyph_width(
            &FontId::new(14.0, FontFamily::Name("icon-only".into())),
            glyph,
        );
        assert!(icon > 0.0, "the icon face lacks the glyph");
        for family in ui_families {
            let width = view.glyph_width(&FontId::new(14.0, family.clone()), glyph);
            assert_eq!(width, icon, "{glyph:?} in {family:?} is not the icon face's");
        }
    }

    #[test]
    fn latin_text_resolves_to_the_inter_weights() {
        let defs = with_single_face(app_fonts(), "inter-only", INTER);
        let defs = with_single_face(defs, "semibold-only", INTER_SEMIBOLD);
        let mut fonts = Fonts::new(TextOptions::default(), defs);
        let mut view = fonts.with_pixels_per_point(1.0);
        for ch in ['a', 'W', '0'] {
            let regular = view.glyph_width(
                &FontId::new(14.0, FontFamily::Name("inter-only".into())),
                ch,
            );
            let bold = view.glyph_width(
                &FontId::new(14.0, FontFamily::Name("semibold-only".into())),
                ch,
            );
            assert_eq!(view.glyph_width(&FontId::proportional(14.0), ch), regular);
            assert_eq!(view.glyph_width(&semibold(14.0), ch), bold);
        }
    }

    #[test]
    fn installing_twice_keeps_one_inter_entry() {
        let mut fonts = app_fonts();
        install(&mut fonts);
        let proportional = &fonts.families[&FontFamily::Proportional];
        assert_eq!(proportional.iter().filter(|k| *k == INTER).count(), 1);
        assert_eq!(proportional.first().map(String::as_str), Some(INTER));
        let medium = &fonts.families[&FontFamily::Name(MEDIUM.into())];
        assert_eq!(medium.iter().filter(|k| *k == INTER_MEDIUM).count(), 1);
    }

    #[test]
    fn the_monospace_stack_and_named_families_survive() {
        let fonts = app_fonts();
        let mono = &fonts.families[&FontFamily::Monospace];
        assert_eq!(mono[0], UBUNTU_SANS_MONO);
        let nerd = mono.iter().position(|k| k == UBUNTU_MONO_NERD).expect("nerd face kept");
        let icons = mono.iter().position(|k| k == "phosphor").expect("icon face registered");
        assert!(icons < nerd, "the icon face precedes the stripped nerd face: {mono:?}");
        for name in ["Regular", "Bold"] {
            let family = &fonts.families[&FontFamily::Name(name.into())];
            assert_eq!(family, &vec![name.to_owned(), "phosphor".to_owned()]);
            assert!(fonts.font_data.contains_key(name));
        }
        let proportional = &fonts.families[&FontFamily::Proportional];
        assert!(
            proportional
                .iter()
                .any(|k| k.to_lowercase().contains("emoji")),
            "emoji fallback faces are kept: {proportional:?}"
        );
    }

    #[test]
    fn text_scale_reaches_ui_faces_and_skips_dedicated_families() {
        let fonts = definitions(TextScale::Large);
        let large = TextScale::Large.factor();
        for key in [
            INTER,
            INTER_MEDIUM,
            INTER_SEMIBOLD,
            UBUNTU_SANS_MONO,
            UBUNTU_MONO_NERD,
        ] {
            assert_eq!(fonts.font_data[key].tweak.scale, large, "{key}");
        }
        assert_eq!(fonts.font_data["Bold"].tweak.scale, 1.0);
        assert_eq!(fonts.font_data["Regular"].tweak.scale, 1.0);
    }

    #[test]
    fn text_scale_is_recorded_and_rebuilds_only_on_change() {
        let ctx = Context::default();
        let mut builds = 0;
        set_text_scale(&ctx, TextScale::Normal, |_| {
            builds += 1;
            FontDefinitions::default()
        });
        assert_eq!(builds, 0);
        set_text_scale(&ctx, TextScale::XLarge, |_| {
            builds += 1;
            FontDefinitions::default()
        });
        assert_eq!(builds, 1);
        assert_eq!(text_scale(&ctx), TextScale::XLarge);
    }
}
