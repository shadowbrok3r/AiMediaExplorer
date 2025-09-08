use crate::utilities::types::QuickAccess;
use directories::UserDirs;
use std::path::{Path, PathBuf};

pub fn quick_access() -> Vec<QuickAccess> {
    let mut v = Vec::new();
    if let Some(ud) = UserDirs::new() {
        if let Some(p) = ud.picture_dir() {
            v.push(QuickAccess {
                label: "Pictures".into(),
                path: p.to_path_buf(),
                icon: "🖼".to_string(),
            });
        }
        if let Some(p) = ud.video_dir() {
            v.push(QuickAccess {
                label: "Videos".into(),
                path: p.to_path_buf(),
                icon: "📹".to_string(),
            });
        }
        if let Some(p) = ud.desktop_dir() {
            v.push(QuickAccess {
                label: "Desktop".into(),
                path: p.to_path_buf(),
                icon: "🖥".to_string(),
            });
        }
        if let Some(p) = ud.download_dir() {
            v.push(QuickAccess {
                label: "Downloads".into(),
                path: p.to_path_buf(),
                icon: "⮋".to_string(),
            });
        }
        if let Some(p) = ud.document_dir() {
            v.push(QuickAccess {
                label: "Documents".into(),
                path: p.to_path_buf(),
                icon: "🖹".to_string(),
            });
        }
        v.push(QuickAccess {
            label: "Home".into(),
            path: ud.home_dir().to_path_buf(),
            icon: "🏠".to_string(),
        });
    }
    v
}

pub fn default_pictures_root() -> Option<PathBuf> {
    UserDirs::new().and_then(|ud| ud.picture_dir().map(|p| p.to_path_buf()))
}

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
#[cfg(windows)]
use windows::{
    Win32::Storage::FileSystem::{
        GetDiskFreeSpaceExW, GetDriveTypeW, GetLogicalDrives, GetVolumeInformationW,
    },
    core::PCWSTR,
};

#[derive(Clone, Debug)]
pub struct DriveInfo {
    pub root: String,
    pub drive_type: String,
    pub label: String,
    pub free: u64,
    pub total: u64,
}

pub enum DriveType {
    Cloud,
    Memory,
    HardDrive,
    Dvd,
    Usb,
}

impl DriveType {
    pub fn as_str(&self) -> &str {
        match self {
            DriveType::Cloud => "☁",
            DriveType::Memory => "💾",
            DriveType::HardDrive => "🖴",
            DriveType::Dvd => "📀",
            DriveType::Usb => "📷",
        }
    }

    pub fn from_u32(drive_type: u32) -> Self {
        match drive_type {
            2 => DriveType::Usb,
            3 => DriveType::HardDrive,
            4 => DriveType::Cloud,
            5 => DriveType::Dvd,
            6 => DriveType::Memory,
            _ => DriveType::HardDrive,
        }
    }
}

#[cfg(windows)]
pub fn list_drive_infos() -> Vec<DriveInfo> {
    unsafe {
        let mut out = Vec::new();
        let mask = GetLogicalDrives();
        for i in 0..26u32 {
            if (mask & (1 << i)) != 0 {
                let letter = (b'A' + i as u8) as char;
                let root = format!("{}:\\", letter);
                if !Path::new(&root).exists() {
                    continue;
                }

                let wide: Vec<u16> = std::ffi::OsStr::new(&root)
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();
                // Volume label
                let mut vol_name: [u16; 260] = [0; 260];
                let mut serial: u32 = 0;
                let mut max_comp: u32 = 0;
                let mut fs_flags: u32 = 0;
                let mut fs_name: [u16; 260] = [0; 260];
                let _ = GetVolumeInformationW(
                    PCWSTR(wide.as_ptr()),
                    Some(&mut vol_name),
                    Some(&mut serial),
                    Some(&mut max_comp),
                    Some(&mut fs_flags),
                    Some(&mut fs_name),
                );
                let nul = vol_name
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(vol_name.len());
                let label = String::from_utf16_lossy(&vol_name[..nul])
                    .trim()
                    .to_string();

                // Free/Total
                let mut free_avail: u64 = 0;
                let mut total: u64 = 0;
                let mut total_free: u64 = 0;
                let _ = GetDiskFreeSpaceExW(
                    PCWSTR(wide.as_ptr()),
                    Some(&mut free_avail),
                    Some(&mut total),
                    Some(&mut total_free),
                );

                let wide: Vec<u16> = std::ffi::OsStr::new(&root)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
                let kind = GetDriveTypeW(PCWSTR(wide.as_ptr()));
                if total > 0 {
                    out.push(DriveInfo {
                        root,
                        label,
                        free: total_free,
                        total,
                        drive_type: DriveType::from_u32(kind).as_str().to_string()
                    });
                }
            }
        }
        out
    }
}
#[cfg(not(windows))]
pub fn list_drive_infos() -> Vec<DriveInfo> {
    Vec::new()
}
