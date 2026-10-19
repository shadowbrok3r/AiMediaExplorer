//! Phosphor font glyphs for inline text, plus file-type and place-to-icon mappings.

pub use egui_icon_pack::{icon, icon_colored, icon_sized, install_fonts};
pub use egui_phosphor::regular as p;

use super::pack::{self as svg, Icon};
use crate::utilities::explorer::DriveType;

pub const CARET_LEFT: &str = p::CARET_LEFT;
pub const CARET_RIGHT: &str = p::CARET_RIGHT;
pub const CARET_DOWN: &str = p::CARET_DOWN;

pub const REFRESH: &str = p::ARROW_CLOCKWISE;

pub const STOP: &str = p::STOP;
pub const PLAY: &str = p::PLAY;
pub const PAUSE: &str = p::PAUSE;

pub const INFO: &str = p::INFO;

pub const CHECK: &str = p::CHECK;
pub const PLUS: &str = p::PLUS;
pub const MINUS: &str = p::MINUS;
pub const EDIT: &str = p::PENCIL_SIMPLE;
pub const COPY: &str = p::COPY;
pub const CLIPBOARD: &str = p::CLIPBOARD;
pub const TRASH: &str = p::TRASH;
pub const EXPORT: &str = p::EXPORT;
pub const OPEN: &str = p::ARROW_SQUARE_OUT;
pub const LINK: &str = p::LINK;
pub const TAG: &str = p::TAG;
pub const STAR: &str = p::STAR;
pub const FLAG: &str = p::FLAG;
pub const SEARCH: &str = p::MAGNIFYING_GLASS;
pub const EYE: &str = p::EYE;
pub const GEAR: &str = p::GEAR;
pub const SLIDERS: &str = p::SLIDERS_HORIZONTAL;
pub const LIST: &str = p::LIST;
pub const TABLE: &str = p::TABLE;
pub const CHAT: &str = p::CHAT;

pub const FOLDER: &str = p::FOLDER;
pub const FOLDER_OPEN: &str = p::FOLDER_OPEN;
pub const FOLDER_PLUS: &str = p::FOLDER_PLUS;
pub const FILE: &str = p::FILE;
pub const FILE_TEXT: &str = p::FILE_TEXT;
pub const IMAGE: &str = p::IMAGE;
pub const ARCHIVE: &str = p::ARCHIVE;
pub const CAMERA: &str = p::CAMERA;

pub const DATABASE: &str = p::DATABASE;
pub const HARD_DRIVE: &str = p::HARD_DRIVE;
pub const CPU: &str = p::CPU;
pub const MEMORY: &str = p::MEMORY;
pub const GPU: &str = p::GRAPHICS_CARD;
pub const GLOBE: &str = p::GLOBE;
pub const NETWORK: &str = p::NETWORK;

pub const SPARKLE: &str = p::SPARKLE;
pub const ROBOT: &str = p::ROBOT;
pub const BRAIN: &str = p::BRAIN;
pub const MAGIC_WAND: &str = p::MAGIC_WAND;
pub const LIGHTBULB: &str = p::LIGHTBULB;
pub const CUBE: &str = p::CUBE;

pub const WINDOWS_LOGO: &str = p::WINDOWS_LOGO;
pub const LINUX_LOGO: &str = p::LINUX_LOGO;

/// Raw camera formats shown with the aperture icon.
const RAW_EXTS: &[&str] = &[
    "arw", "cr2", "cr3", "nef", "nrw", "dng", "orf", "rw2", "raf", "pef", "srw", "raw", "3fr", "erf",
    "kdc", "mrw", "x3f",
];

/// Font glyph of a quick-access place by label.
pub fn folder_shortcut_icon(label: &str) -> &'static str {
    folder_shortcut_svg(label).glyph()
}

/// SVG icon of a quick-access place by label.
pub fn folder_shortcut_svg(label: &str) -> Icon {
    match label {
        "Pictures" => svg::IMAGE,
        "Videos" => svg::VIDEO_CAMERA,
        "Desktop" => svg::DESKTOP,
        "Downloads" => svg::DOWNLOAD_SIMPLE,
        "Documents" => svg::FILE_TEXT,
        "Music" => svg::MUSIC_NOTES,
        "Home" => svg::HOUSE,
        _ => svg::FOLDER,
    }
}

/// Font glyph of a file by name.
pub fn file_icon(filename: &str, is_directory: bool) -> &'static str {
    file_icon_svg(filename, is_directory).glyph()
}

/// SVG icon of a file by name.
pub fn file_icon_svg(filename: &str, is_directory: bool) -> Icon {
    if is_directory {
        return svg::FOLDER;
    }
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    if RAW_EXTS.contains(&ext.as_str()) {
        return svg::AME_RAW_PHOTO;
    }
    match ext.as_str() {
        "jpg" | "jpeg" => svg::FILE_JPG,
        "png" => svg::FILE_PNG,
        "svg" => svg::FILE_SVG,
        "gif" | "bmp" | "tif" | "tiff" | "webp" | "heic" | "heif" | "avif" | "ico" | "psd" | "ai" => {
            svg::FILE_IMAGE
        }
        "mp4" | "mov" | "avi" | "mkv" | "webm" | "wmv" | "m4v" | "flv" | "mpeg" | "mpg" | "3gp" => {
            svg::FILE_VIDEO
        }
        "mp3" | "wav" | "flac" | "ogg" | "m4a" => svg::FILE_AUDIO,
        "zip" | "7z" | "rar" | "tar" | "gz" | "bz" | "bz2" | "xz" | "tgz" | "tbz" | "tbz2"
        | "txz" => svg::FILE_ZIP,
        "iso" | "img" => svg::DISC,
        "exe" | "msi" | "dll" | "so" | "dylib" => svg::FILE_ARCHIVE,
        "sh" | "bash" | "zsh" | "bat" | "cmd" | "ps1" | "psm1" | "psd1" => svg::FILE_CODE,
        "js" | "ts" | "mjs" | "jsx" | "tsx" => svg::FILE_JS,
        "py" | "pyw" => svg::FILE_PY,
        "rs" => svg::FILE_RS,
        "c" | "cpp" | "h" | "hpp" => svg::FILE_CPP,
        "cs" | "go" | "java" | "kt" => svg::FILE_CODE,
        "html" | "htm" => svg::FILE_HTML,
        "css" | "scss" | "sass" => svg::FILE_CSS,
        "txt" | "log" => svg::FILE_TXT,
        "md" | "rst" => svg::FILE_MD,
        "doc" | "docx" => svg::FILE_DOC,
        "pdf" => svg::FILE_PDF,
        "xls" | "xlsx" => svg::FILE_XLS,
        "csv" => svg::FILE_CSV,
        "ppt" | "pptx" => svg::FILE_PPT,
        "json" | "yaml" | "yml" | "toml" | "xml" | "ini" | "cfg" | "conf" | "config" => {
            svg::FILE_TEXT
        }
        "db" | "sqlite" | "mdb" => svg::DATABASE,
        "sql" | "surql" => svg::FILE_SQL,
        _ => svg::FILE,
    }
}

/// SVG icon of a drive class.
pub fn drive_icon(kind: &DriveType) -> Icon {
    match kind {
        DriveType::Cloud => svg::CLOUD,
        DriveType::Memory => svg::FLOPPY_DISK,
        DriveType::HardDrive => svg::HARD_DRIVE,
        DriveType::Dvd => svg::DISC,
        DriveType::Usb => svg::USB,
    }
}
