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

// --- WSL quick access helpers (Windows only) ---
#[cfg(windows)]
pub fn list_wsl_distros() -> Vec<String> {
    // First try UNC root (some systems can't list \\wsl.localhost root)
    let root = std::path::Path::new("\\\\wsl.localhost\\");
    let mut out: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for dent in rd.flatten() {
            if let Ok(ft) = dent.file_type() {
                if ft.is_dir() {
                    if let Some(name) = dent.file_name().to_str() { out.push(name.to_string()); }
                }
            }
        }
    }
    if out.is_empty() {
        // Fallback: query via wsl.exe (fast, reliable)
        use std::process::Command;
        if let Ok(outp) = Command::new("wsl").args(["-l", "-q"]).output() {
            if outp.status.success() {
                let s = String::from_utf8_lossy(&outp.stdout);
                for line in s.lines() {
                    let name = line.trim();
                    if !name.is_empty() { out.push(name.to_string()); }
                }
            }
        }
    }
    out.sort();
    out
}

#[cfg(not(windows))]
pub fn list_wsl_distros() -> Vec<String> { Vec::new() }

#[cfg(windows)]
pub fn wsl_dynamic_mounts(distro: &str) -> Vec<QuickAccess> {
    // Build list including common homes and discovered mounts under mnt/wsl and mnt/*
    let mut items: Vec<QuickAccess> = Vec::new();
    let base = format!("\\\\wsl.localhost\\{}", distro);
    let push = |items: &mut Vec<QuickAccess>, label: String, sub: &str, icon: &str| items.push(QuickAccess { label, path: PathBuf::from(format!("{}\\{}", base, sub)), icon: icon.to_string() });
    // Always include home and root if they exist
    if std::path::Path::new(&format!("{}\\home", base)).exists() { push(&mut items, format!("WSL:{distro} /home"), "home", "🐧"); }
    if std::path::Path::new(&format!("{}\\root", base)).exists() { push(&mut items, format!("WSL:{distro} /root"), "root", "👑"); }
    // Preferred: mnt\\wsl (contains mounted drives)
    let wsl_mount_str = format!("{}\\mnt\\wsl", base);
    let wsl_mount = std::path::Path::new(&wsl_mount_str);
    if let Ok(rd) = std::fs::read_dir(wsl_mount) {
        for dent in rd.flatten() {
            if dent.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                if let Some(name) = dent.file_name().to_str() {
                    let label = format!("WSL:{distro} /mnt/wsl/{}", name);
                    push(&mut items, label, &format!("mnt\\wsl\\{}", name), "�");
                }
            }
        }
    } else {
        // Fallback: enumerate mnt/* top-level letters (c,d,...) as typical Windows drives
    let mnt_str = format!("{}\\mnt", base);
    let mnt = std::path::Path::new(&mnt_str);
        if let Ok(rd) = std::fs::read_dir(mnt) {
            for dent in rd.flatten() {
                if dent.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    if let Some(name) = dent.file_name().to_str() {
                        let label = format!("WSL:{distro} /mnt/{}", name);
                        push(&mut items, label, &format!("mnt\\{}", name), "🖴");
                    }
                }
            }
        }
    }
    items
}

#[cfg(not(windows))]
pub fn wsl_dynamic_mounts(_distro: &str) -> Vec<QuickAccess> { Vec::new() }
