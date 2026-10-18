//! Command-line switches for the theme: `--theme-dump`, `--theme <path.json>` and `--no-frost`.

use std::path::PathBuf;
use std::sync::OnceLock;

use eframe::egui::{Color32, Context, FontFamily};
use serde_json::{Value, json};

use super::{
    colors, fonts, glass, metrics,
    metrics::Metrics,
    persist,
    presets::{apply_saved_theme, installed_preset},
    tokens::Tokens,
};

/// Theme-related command-line options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Print the resolved tokens and metrics as JSON and exit without opening a window.
    pub theme_dump: bool,
    /// Theme JSON file applied at startup in place of the saved theme.
    pub theme_path: Option<PathBuf>,
    /// Switch the glass backdrop-blur pass off for this run.
    pub no_frost: bool,
    /// Open the Theme window at startup.
    pub theme_panel: bool,
}

impl Options {
    /// Parses the options out of `args`, skipping arguments it does not know.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut out = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--theme-dump" => out.theme_dump = true,
                "--no-frost" => out.no_frost = true,
                "--theme-panel" => out.theme_panel = true,
                "--theme" => {
                    let path = args.next().ok_or("--theme needs a path to a JSON file")?;
                    out.theme_path = Some(PathBuf::from(path));
                }
                other => {
                    if let Some(path) = other.strip_prefix("--theme=") {
                        out.theme_path = Some(PathBuf::from(path));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The options of the running process: its arguments, with [`THEME_ENV`] standing in for a missing `--theme`.
    pub fn from_env() -> Result<Self, String> {
        let env = std::env::var_os(THEME_ENV).map(PathBuf::from);
        Self::parse_with_env(std::env::args().skip(1), env)
    }

    /// [`Self::parse`] with `env_theme` filling in a theme path the arguments did not give.
    pub fn parse_with_env(
        args: impl IntoIterator<Item = String>,
        env_theme: Option<PathBuf>,
    ) -> Result<Self, String> {
        let mut out = Self::parse(args)?;
        if out.theme_path.is_none() {
            out.theme_path = env_theme.filter(|p| !p.as_os_str().is_empty());
        }
        Ok(out)
    }
}

/// Environment variable naming a theme file, the same as `--theme`.
pub const THEME_ENV: &str = "SMART_MEDIA_THEME";

static OPTIONS: OnceLock<Options> = OnceLock::new();

/// Records the process options and applies `--no-frost`.
pub fn install(options: Options) {
    if options.no_frost {
        glass::disable("--no-frost on the command line");
    }
    let _ = OPTIONS.set(options);
}

/// The recorded process options, or the defaults before [`install`].
pub fn options() -> Options {
    OPTIONS.get().cloned().unwrap_or_default()
}

fn hex(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn hex_rgba(c: Color32) -> String {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
}

fn vec2(v: eframe::egui::Vec2) -> Value {
    json!([v.x, v.y])
}

/// The metrics as a JSON object.
pub fn metrics_json(m: &Metrics) -> Value {
    json!({
        "control_h": m.control_h,
        "button_h": m.button_h,
        "pad": vec2(m.pad),
        "gap": vec2(m.gap),
        "window_margin": m.window_margin,
        "menu_margin": m.menu_margin,
        "radius_sm": m.radius_sm,
        "radius_md": m.radius_md,
        "radius_lg": m.radius_lg,
        "combo_w": m.combo_w,
        "slider_w": m.slider_w,
        "tooltip_w": m.tooltip_w,
        "icon_sm": m.icon_sm,
        "icon_md": m.icon_md,
        "icon_lg": m.icon_lg,
        "row_h": m.row_h,
        "header_h": m.header_h,
        "bar_h": m.bar_h,
        "footer_h": m.footer_h,
    })
}

/// The tokens as a JSON object of `#rrggbb` colors and radii.
pub fn tokens_json(t: &Tokens) -> Value {
    let mut map = serde_json::Map::new();
    for (name, color) in t.colors() {
        map.insert(name.to_owned(), json!(hex(color)));
    }
    map.insert("radius_sm".into(), json!(t.radius_sm));
    map.insert("radius_md".into(), json!(t.radius_md));
    map.insert("radius_lg".into(), json!(t.radius_lg));
    Value::Object(map)
}

/// The resolved theme of `ctx` as a JSON document.
pub fn resolved_json(ctx: &Context) -> Value {
    let style = ctx.global_style();
    let glass_params = glass::params(ctx);
    let text_styles: serde_json::Map<String, Value> = style
        .text_styles
        .iter()
        .map(|(name, font)| {
            let family = match &font.family {
                FontFamily::Proportional => "Proportional".to_owned(),
                FontFamily::Monospace => "Monospace".to_owned(),
                FontFamily::Name(name) => name.to_string(),
            };
            (
                name.to_string(),
                json!({ "size": font.size, "family": family }),
            )
        })
        .collect();
    let families = ctx.fonts(|f| {
        f.definitions()
            .families
            .iter()
            .map(|(family, faces)| (format!("{family:?}"), json!(faces)))
            .collect::<serde_json::Map<_, _>>()
    });
    json!({
        "source": persist::source(ctx).label(),
        "preset": installed_preset(ctx).as_str(),
        "density": metrics::density(ctx).label(),
        "text_scale": fonts::text_scale(ctx).label(),
        "glass": {
            "enabled": glass_params.enabled,
            "blur_radius": glass_params.blur_radius,
            "tint": hex_rgba(glass_params.tint),
            "corner_radius": glass_params.corner_radius,
            "presence": glass_params.presence,
        },
        "metrics": metrics_json(&Metrics::get(ctx)),
        "tokens": tokens_json(&Tokens::get(ctx)),
        "semantic": {
            "success": hex(colors::success_ctx(ctx)),
            "accent_secondary": hex(colors::accent_secondary_ctx(ctx)),
        },
        "text_styles": text_styles,
        "font_families": families,
        "interact_size": vec2(style.spacing.interact_size),
        "item_spacing": vec2(style.spacing.item_spacing),
        "button_padding": vec2(style.spacing.button_padding),
        "font_hinting": style.visuals.text_options.font_hinting,
    })
}

/// The pretty-printed JSON `--theme-dump` prints for `options`, resolved without a window.
pub fn dump(options: &Options) -> anyhow::Result<String> {
    let ctx = Context::default();
    fonts::install_for_context(&ctx);
    let (theme, source) = persist::resolve(None, options.theme_path.as_deref());
    apply_saved_theme(&ctx, theme);
    persist::set_source(&ctx, source);
    ctx.run_ui(Default::default(), |_| {})
        .drop_without_applying_deltas();
    Ok(serde_json::to_string_pretty(&resolved_json(&ctx))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        Options::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn flags_are_recognised_and_unknown_arguments_pass() {
        let opts = parse(&[
            "--theme-dump",
            "--no-frost",
            "--theme-panel",
            "--theme",
            "a.json",
            "--other",
        ])
        .unwrap();
        assert!(opts.theme_dump && opts.no_frost && opts.theme_panel);
        assert_eq!(opts.theme_path, Some(PathBuf::from("a.json")));
        let eq = parse(&["--theme=b.json"]).unwrap();
        assert_eq!(eq.theme_path, Some(PathBuf::from("b.json")));
        assert_eq!(parse(&[]).unwrap(), Options::default());
    }

    #[test]
    fn the_environment_theme_fills_in_only_a_missing_flag() {
        let env = Some(PathBuf::from("env.json"));
        let from_env = Options::parse_with_env(Vec::new(), env.clone()).unwrap();
        assert_eq!(from_env.theme_path, env);
        let flag = Options::parse_with_env(vec!["--theme".to_owned(), "flag.json".to_owned()], env)
            .unwrap();
        assert_eq!(flag.theme_path, Some(PathBuf::from("flag.json")));
        let blank = Options::parse_with_env(Vec::new(), Some(PathBuf::new())).unwrap();
        assert_eq!(blank.theme_path, None);
    }

    #[test]
    fn a_theme_flag_without_a_path_is_an_error() {
        assert!(parse(&["--theme"]).is_err());
    }

    #[test]
    fn the_dump_reports_the_mastertech_numbers() {
        let text = dump(&Options::default()).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["preset"], "Mastertech");
        assert_eq!(v["density"], "Compact");
        assert_eq!(v["text_scale"], "Small (90%)");
        assert_eq!(v["metrics"]["control_h"], 22.0);
        assert_eq!(v["metrics"]["radius_sm"], 6);
        assert_eq!(v["tokens"]["radius_sm"], 3);
        assert_eq!(v["tokens"]["radius_md"], 4);
        assert_eq!(v["tokens"]["radius_lg"], 5);
        assert_eq!(v["text_styles"]["Body"]["size"], 12.5);
        assert_eq!(v["text_styles"]["Small"]["size"], 10.5);
        assert_eq!(v["text_styles"]["Monospace"]["size"], 12.0);
        assert_eq!(v["text_styles"]["Heading"]["size"], 15.0);
        assert_eq!(v["text_styles"]["Heading"]["family"], "semibold");
        assert_eq!(v["glass"]["blur_radius"], 32.0);
        assert_eq!(v["tokens"]["base"], "#08080a");
        assert!(v["tokens"]["primary"].as_str().unwrap().starts_with('#'));
        assert_eq!(v["font_hinting"], true);
        assert!(v["font_families"]["Proportional"][0] == "Inter");
    }

    #[test]
    fn the_dump_honours_a_theme_file() {
        let dir = std::env::temp_dir().join(format!("smart_media_dump_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("studio.json");
        let ctx = Context::default();
        super::super::presets::apply_preset(&ctx, super::super::presets::PresetStyles::MtechStudio);
        let theme = super::super::saved::SavedTheme::current(&ctx);
        std::fs::write(&path, super::super::saved::theme_to_json(&theme).unwrap()).unwrap();

        let opts = Options {
            theme_path: Some(path),
            ..Options::default()
        };
        let v: Value = serde_json::from_str(&dump(&opts).unwrap()).unwrap();
        assert_eq!(v["preset"], "MTech Studio");
        assert_eq!(v["tokens"]["primary"], "#ff2a6d");
        std::fs::remove_dir_all(&dir).ok();
    }
}
