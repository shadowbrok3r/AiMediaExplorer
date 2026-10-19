use crate::utilities::types::QuickAccess;
use directories::UserDirs;
use std::path::Path;
#[cfg(windows)]
use std::path::PathBuf;

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
#[cfg(target_os = "linux")]
pub fn list_drive_infos() -> Vec<DriveInfo> {
    let Ok(text) = std::fs::read_to_string("/proc/mounts") else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::<String>::new();

    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(source) = parts.next() else { continue };
        let Some(mount) = parts.next() else { continue };
        let Some(fstype) = parts.next() else { continue };

        let mount = unescape_mount_path(mount);
        if !is_useful_linux_mount(source, &mount, fstype) {
            continue;
        }
        if !seen.insert(mount.clone()) {
            continue;
        }

        let Some((free, total)) = linux_disk_usage(&mount) else { continue };
        if total == 0 {
            continue;
        }

        let label = linux_volume_label(source, &mount);
        let drive_type = linux_drive_type(source, &mount, fstype)
            .as_str()
            .to_string();

        out.push(DriveInfo {
            root: mount,
            drive_type,
            label,
            free,
            total,
        });
    }

    out.sort_by(|a, b| a.root.cmp(&b.root));
    out
}

#[cfg(all(not(windows), not(target_os = "linux")))]
pub fn list_drive_infos() -> Vec<DriveInfo> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn unescape_mount_path(s: &str) -> String {
    // /proc/mounts escapes space as \040, tab as \011, newline as \012, backslash as \\
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let code: String = chars.by_ref().take(3).collect();
            if let Ok(n) = u8::from_str_radix(&code, 8) {
                out.push(n as char);
            } else {
                out.push('\\');
                out.push_str(&code);
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn is_useful_linux_mount(source: &str, mount: &str, fstype: &str) -> bool {
    const SKIP_FS: &[&str] = &[
        "proc", "sysfs", "cgroup", "cgroup2", "tmpfs", "devtmpfs", "devpts",
        "securityfs", "pstore", "bpf", "autofs", "configfs", "mqueue",
        "tracefs", "debugfs", "hugetlbfs", "fusectl", "efivarfs", "overlay",
        "squashfs", "nsfs", "binfmt_misc", "rpc_pipefs", "fuse.gvfsd-fuse",
        "fuse.portal", "fuse.kio-fuse",
    ];
    if SKIP_FS.iter().any(|s| *s == fstype) {
        return false;
    }
    if fstype.starts_with("fuse.") && !source.starts_with("/dev/") {
        // Skip AppImage / portal FUSE mounts that are not block devices.
        return false;
    }
    if source.contains("/loop") || source.starts_with("/dev/loop") {
        return false;
    }

    if mount == "/" {
        return true;
    }
    if mount.starts_with("/mnt/")
        || mount.starts_with("/media/")
        || mount.starts_with("/run/media/")
    {
        return true;
    }
    // Other real block-device mounts (e.g. /home on a separate disk).
    source.starts_with("/dev/") && !mount.starts_with("/boot")
}

#[cfg(target_os = "linux")]
fn linux_drive_type(source: &str, mount: &str, fstype: &str) -> DriveType {
    if matches!(fstype, "iso9660" | "udf") {
        return DriveType::Dvd;
    }
    if mount.starts_with("/media/") || mount.starts_with("/run/media/") {
        return DriveType::Usb;
    }
    if source.starts_with("//") || fstype.contains("nfs") || fstype.contains("cifs") || fstype.contains("smb") {
        return DriveType::Cloud;
    }
    DriveType::HardDrive
}

#[cfg(target_os = "linux")]
fn linux_volume_label(source: &str, mount: &str) -> String {
    if mount == "/" {
        return "Root".to_string();
    }
    if let Some(label) = linux_label_for_device(source) {
        return label;
    }
    Path::new(mount)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(mount)
        .to_string()
}

#[cfg(target_os = "linux")]
fn linux_label_for_device(source: &str) -> Option<String> {
    let source = std::fs::canonicalize(source).ok()?;
    let by_label = Path::new("/dev/disk/by-label");
    let entries = std::fs::read_dir(by_label).ok()?;
    for dent in entries.flatten() {
        let path = dent.path();
        let Ok(target) = std::fs::canonicalize(&path) else { continue };
        if target == source {
            let name = dent.file_name().into_string().ok()?;
            return Some(decode_udev_name(&name));
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn decode_udev_name(name: &str) -> String {
    // udev encodes spaces/specials as \xNN in by-label names.
    let mut out = String::with_capacity(name.len());
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() && bytes[i + 1] == b'x' {
            if let Ok(n) = u8::from_str_radix(std::str::from_utf8(&bytes[i + 2..i + 4]).unwrap_or(""), 16) {
                out.push(n as char);
                i += 4;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[cfg(target_os = "linux")]
fn linux_disk_usage(path: &str) -> Option<(u64, u64)> {
    use std::ffi::CString;
    let c_path = CString::new(path).ok()?;
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut s) != 0 {
            return None;
        }
        let frsize = s.f_frsize as u64;
        let total = s.f_blocks as u64 * frsize;
        let free = s.f_bavail as u64 * frsize;
        Some((free, total))
    }
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
                    if let Some(name) = dent.file_name().to_str() {
                        if !name.contains("docker") && !name.contains("podman") {
                            out.push(name.to_string()); 
                        }
                    }
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
                    // Show each mounted physical partition as a drive
                    push(&mut items, label, &format!("mnt\\wsl\\{}", name), "🖴");
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

#[derive(Clone, Debug)]
pub struct PhysicalDrive {
    pub caption: String,
    pub device_id: String,
    pub model: String,
    pub partitions: u32,
    pub size: u64,
}

#[cfg(windows)]
pub fn list_physical_drives() -> Vec<PhysicalDrive> {
    use std::process::Command;
    
    let mut drives = Vec::new();
    
    // Use wmic to list physical drives
    if let Ok(output) = Command::new("wmic")
        .args(["diskdrive", "list", "brief"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            
            // Skip header line
            for line in lines.iter().skip(1) {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                
                // Parse the line - format is usually:
                // Caption       DeviceID            Model         Partitions  Size
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    // Try to extract device ID (should contain \\.\PHYSICALDRIVE)
                    if let Some(device_pos) = parts.iter().position(|&p| p.contains("PHYSICALDRIVE")) {
                        let _device_id = parts[device_pos].to_string();
                        
                        // Extract other fields - this is a bit tricky due to spacing
                        // Let's use a more robust approach with the full line
                        if let Some(device_start) = line.find("\\\\") {
                            if let Some(device_end) = line[device_start..].find(' ') {
                                let device_id = line[device_start..device_start + device_end].to_string();
                                
                                // Extract size (should be the last number)
                                if let Some(size_str) = parts.last() {
                                    if let Ok(size) = size_str.parse::<u64>() {
                                        // Extract partitions (second to last number)
                                        let partitions = if parts.len() > 1 {
                                            parts[parts.len() - 2].parse::<u32>().unwrap_or(0)
                                        } else {
                                            0
                                        };
                                        
                                        // Extract caption and model (everything before device_id and after)
                                        let caption = if device_start > 0 {
                                            line[..device_start].trim().to_string()
                                        } else {
                                            "Unknown Drive".to_string()
                                        };
                                        
                                        // Model is between device_id and partitions
                                        let after_device = &line[device_start + device_id.len()..];
                                        let model_part = after_device.trim().split_whitespace().collect::<Vec<_>>();
                                        let model = if model_part.len() >= 3 {
                                            model_part[..model_part.len() - 2].join(" ")
                                        } else {
                                            caption.clone()
                                        };
                                        
                                        drives.push(PhysicalDrive {
                                            caption,
                                            device_id,
                                            model,
                                            partitions,
                                            size,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    drives
}

#[cfg(not(windows))]
pub fn list_physical_drives() -> Vec<PhysicalDrive> { Vec::new() }

#[cfg(windows)]
pub fn mount_wsl_drive(device_id: &str, partition: u32) -> Result<String, String> {
    use std::process::Command;
    log::info!("wsl --mount {device_id} --partition {partition}");

    let output = Command::new("wsl")
        .args(["--mount", device_id, "--partition", &partition.to_string()])
        .output()
        .map_err(|e| format!("Failed to execute wsl command: {}", e))?;
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("WSL mount failed: {}", stderr))
    }
}

#[cfg(not(windows))]
pub fn mount_wsl_drive(_device_id: &str, _partition: u32) -> Result<String, String> {
    Err("WSL mounting is only available on Windows".to_string())
}

#[cfg(windows)]
pub fn unmount_wsl_drive(device_id: &str) -> Result<String, String> {
    use std::process::Command;
    
    let output = Command::new("wsl")
        .args(["--unmount", device_id])
        .output()
        .map_err(|e| format!("Failed to execute wsl command: {}", e))?;
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("WSL unmount failed: {}", stderr))
    }
}

#[cfg(not(windows))]
pub fn unmount_wsl_drive(_device_id: &str) -> Result<String, String> {
    Err("WSL unmounting is only available on Windows".to_string())
}

#[cfg(all(test, target_os = "linux"))]
mod linux_drive_tests {
    use super::*;

    #[test]
    fn lists_root_and_real_mounts() {
        let drives = list_drive_infos();
        assert!(
            drives.iter().any(|d| d.root == "/"),
            "expected root mount, got {drives:?}"
        );
        assert!(
            drives.iter().all(|d| d.total > 0),
            "expected positive totals, got {drives:?}"
        );
        for d in &drives {
            assert!(!d.root.is_empty());
            println!("{} {} free={} total={} label={}", d.drive_type, d.root, d.free, d.total, d.label);
        }
    }
}
