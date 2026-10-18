//! App configuration of egui-theme-kit: font stack, command-line options and the kit config.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use egui_theme_kit::{FontConfig, KitConfig, cli, startup};

/// `font_data` key of the face placed first in `Monospace`.
pub const UBUNTU_SANS_MONO: &str = "UbuntuSansMono";
/// `font_data` key of the Nerd Font face placed second in `Monospace`.
pub const UBUNTU_MONO_NERD: &str = "UbuntuMonoNerdFont";

/// Theme file variable of earlier releases; `EGUI_THEME_KIT_THEME` wins when both are set.
pub const LEGACY_THEME_ENV: &str = "SMART_MEDIA_THEME";
/// Frost kill-switch variable of earlier releases; honoured alongside `EGUI_THEME_KIT_NO_FROST`.
pub const LEGACY_NO_FROST_ENV: &str = "SMART_MEDIA_NO_FROST";

static MONASPACE_REGULAR_OTF: &[u8] = include_bytes!("../../assets/fonts/MonaspaceNeon-Regular.otf");
static MONASPACE_BOLD_OTF: &[u8] = include_bytes!("../../assets/fonts/MonaspaceNeon-Bold.otf");
static UBUNTU_SANS_MONO_OTF: &[u8] = include_bytes!("../../assets/fonts/UbuntuSansMono-Regular.otf");
static UBUNTU_MONO_NERD_TTF: &[u8] = include_bytes!("../../assets/fonts/UbuntuMonoNerdFont-Regular.ttf");

/// The app font stack: Ubuntu monospace, Monaspace "Regular" and "Bold" families and the Phosphor icon font.
pub fn font_config() -> FontConfig {
    FontConfig::new()
        .face(UBUNTU_SANS_MONO, UBUNTU_SANS_MONO_OTF)
        .face_stripped(UBUNTU_MONO_NERD, UBUNTU_MONO_NERD_TTF)
        .face("Regular", MONASPACE_REGULAR_OTF)
        .face("Bold", MONASPACE_BOLD_OTF)
        .monospace_first(UBUNTU_SANS_MONO)
        .monospace_first(UBUNTU_MONO_NERD)
        .family("Regular", ["Regular"])
        .family("Bold", ["Bold"])
        .installer(crate::icons::install_fonts)
}

/// The kit config the app starts with: its fonts and `store` as the startup theme source.
pub(super) fn kit_config(store: Arc<dyn startup::ThemeStore>) -> KitConfig {
    KitConfig::new().parse_cli(false).fonts(font_config()).store(store)
}

/// The theme switches of the running process; the old `SMART_MEDIA_*` variables stand in for the new ones.
pub fn cli_options() -> Result<cli::Options, String> {
    cli_options_from(std::env::args().skip(1), |name| std::env::var_os(name))
}

fn cli_options_from(
    args: impl IntoIterator<Item = String>,
    env: impl Fn(&str) -> Option<OsString>,
) -> Result<cli::Options, String> {
    let theme = env(cli::THEME_ENV).or_else(|| env(LEGACY_THEME_ENV)).map(PathBuf::from);
    let mut options = cli::Options::parse_with_env(args, theme)?;
    options.no_frost |= env(LEGACY_NO_FROST_ENV).is_some_and(|value| enabled(&value));
    Ok(options)
}

/// Whether an on/off variable is switched on: anything but empty, `0`, `false`, `off` or `no`.
fn enabled(value: &OsString) -> bool {
    let text = value.to_string_lossy();
    let text = text.trim().to_ascii_lowercase();
    !matches!(text.as_str(), "" | "0" | "false" | "off" | "no")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn parse(args: &[&str], env: &[(&str, &str)]) -> cli::Options {
        let vars: HashMap<&str, OsString> = env.iter().map(|(k, v)| (*k, OsString::from(v))).collect();
        cli_options_from(args.iter().map(|a| a.to_string()), |name| vars.get(name).cloned()).unwrap()
    }

    #[test]
    fn the_new_theme_variable_wins_and_the_old_one_stands_in() {
        assert_eq!(parse(&[], &[("SMART_MEDIA_THEME", "old.json")]).theme_path, Some("old.json".into()));
        let both = [("SMART_MEDIA_THEME", "old.json"), ("EGUI_THEME_KIT_THEME", "new.json")];
        assert_eq!(parse(&[], &both).theme_path, Some("new.json".into()));
        assert_eq!(parse(&["--theme", "flag.json"], &both).theme_path, Some("flag.json".into()));
        assert_eq!(parse(&[], &[]).theme_path, None);
    }

    #[test]
    fn the_old_frost_variable_switches_frosting_off_unless_it_says_no() {
        assert!(parse(&[], &[("SMART_MEDIA_NO_FROST", "1")]).no_frost);
        assert!(parse(&[], &[("SMART_MEDIA_NO_FROST", "yes")]).no_frost);
        for off in ["", "0", "false", "OFF", "no"] {
            assert!(!parse(&[], &[("SMART_MEDIA_NO_FROST", off)]).no_frost, "{off:?}");
        }
        assert!(parse(&["--no-frost"], &[]).no_frost);
        assert!(!parse(&[], &[]).no_frost);
    }

    #[test]
    fn the_title_bar_flag_passes_the_theme_parser() {
        let options = parse(&["--system-titlebar", "--theme-panel"], &[("EGUI_THEME_KIT_SYSTEM_TITLEBAR", "1")]);
        assert!(options.theme_panel && !options.theme_dump && options.theme_path.is_none());
    }

    #[test]
    fn the_theme_flags_still_parse() {
        let options = parse(&["--theme-dump", "--theme-panel", "--theme=a.json"], &[]);
        assert!(options.theme_dump && options.theme_panel);
        assert_eq!(options.theme_path, Some("a.json".into()));
    }
}
