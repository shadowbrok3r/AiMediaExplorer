
#[cfg(windows)]
use windows::{
    core::Interface, 
    Win32::{ 
        Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, IDXGIFactory6, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE},
        Foundation::FILETIME, 
        System::{
            ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS}, 
            Threading::{GetCurrentProcess, GetProcessTimes}
        }
    }
};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use std::time::{Duration, Instant};

// --- WSL/UNC path normalization helpers ---
/// Normalize common WSL UNC inputs to a canonical \\\\wsl.localhost\\<distro>\\... form.
/// No-op on non-Windows or for virtual schemes like zip://
#[cfg(windows)]
pub fn normalize_wsl_unc(input: &str) -> String {
    // Skip virtual schemes
    if input.starts_with("zip://") || input.starts_with("tar://") || input.starts_with("7z://") { return input.to_string(); }
    let s = input.trim().replace('/', "\\");
    let slow = s.to_ascii_lowercase();
    // Accept variations: \\wsl$\\, \wsl$\\, wsl$\\, /wsl$/, same for wsl.localhost
    // Map all to \\wsl.localhost\\
    let mut rest = None;
    if slow.starts_with("\\\\wsl$\\") { rest = Some(s[6..].to_string()); } // strip "\\wsl$\\"
    else if slow.starts_with("\\wsl$\\") { rest = Some(s[5..].to_string()); } // add missing backslash later
    else if slow.starts_with("wsl$\\") { rest = Some(s[5..].to_string()); }
    else if slow.starts_with("/wsl$/") { rest = Some(s[6..].to_string()); }
    else if slow.starts_with("\\\\wsl.localhost\\") { return s; }
    else if slow.starts_with("\\wsl.localhost\\") { return format!("\\\\{}", &s[1..]); }
    else if slow.starts_with("wsl.localhost\\") { rest = Some(s[14..].to_string()); }
    else if slow.starts_with("/wsl.localhost/") { rest = Some(s[15..].to_string()); }

    if let Some(r) = rest {
        let r = r.trim_start_matches('\\');
        return format!("\\\\wsl.localhost\\{}", r);
    }
    s
}

#[cfg(not(windows))]
pub fn normalize_wsl_unc(input: &str) -> String { input.to_string() }

#[cfg(windows)]
pub fn normalize_wsl_unc_pathbuf(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::PathBuf;
    let s = p.to_string_lossy().to_string();
    PathBuf::from(normalize_wsl_unc(&s))
}

#[cfg(not(windows))]
pub fn normalize_wsl_unc_pathbuf(p: &std::path::Path) -> std::path::PathBuf { p.to_path_buf() }

#[cfg(windows)]
pub fn gpu_mem_mb() -> Option<(f32, f32)> {
    // Prefer PDH performance counters when available as they reflect system VRAM usage on NVIDIA drivers.
    if let Some(v) = gpu_mem_mb_pdh() { return Some(v); }
    // Fallback to DXGI budgets
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1::<IDXGIFactory1>().ok()?;
        if let Ok(factory6) = factory.cast::<IDXGIFactory6>() {
            if let Ok(adapter_hp) = factory6.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) {
                if let Some(v) = query_vram_from_adapter(adapter_hp) { return Some(v); }
            }
        }
        if let Ok(adapter0) = factory.EnumAdapters1(0) { return query_vram_from_adapter(adapter0); }
        None
    }
}

#[cfg(windows)]
unsafe fn query_vram_from_adapter(adapter1: windows::Win32::Graphics::Dxgi::IDXGIAdapter1) -> Option<(f32, f32)> {
    use windows::Win32::Graphics::Dxgi::{IDXGIAdapter3, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO};
    // Use dynamic usage+budget for both local and non-local segments if available
    if let Ok(adapter3) = adapter1.cast::<IDXGIAdapter3>() {
        let mut info_local: DXGI_QUERY_VIDEO_MEMORY_INFO = unsafe { std::mem::zeroed() };
        let mut info_nonlocal: DXGI_QUERY_VIDEO_MEMORY_INFO = unsafe { std::mem::zeroed() };
        let mut ok = false;
        let mut used = 0u64;
        let mut total = 0u64;
        if unsafe { adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info_local) }.is_ok() {
            used = used.saturating_add(info_local.CurrentUsage);
            total = total.saturating_add(info_local.Budget);
            ok = true;
        }
        if unsafe { adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut info_nonlocal) }.is_ok() {
            used = used.saturating_add(info_nonlocal.CurrentUsage);
            total = total.saturating_add(info_nonlocal.Budget);
            ok = true;
        }
        if ok && total > 0 {
            let used_mb = (used as f64 / (1024.0 * 1024.0)) as f32;
            let total_mb = (total as f64 / (1024.0 * 1024.0)) as f32;
            return Some((used_mb, total_mb));
        }
    }
    // Fallback to dedicated VRAM from description
    if let Ok(desc) = unsafe { adapter1.GetDesc1() } {
        let total_mb = (desc.DedicatedVideoMemory as f64 / (1024.0 * 1024.0)) as f32;
        if total_mb > 0.0 { return Some((0.0, total_mb)); }
    }
    None
}

// --- Smoothing / normalization helpers ---
static SMOOTHED_CPU01: Lazy<Mutex<Option<f32>>> = Lazy::new(|| Mutex::new(None));
static CPU_LAST_UPDATE: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));
const CPU_MIN_PERIOD: Duration = Duration::from_millis(500);

pub fn smoothed_cpu01() -> f32 {
    // Rate-limit updates to avoid recomputing on every frame/input event
    let now = Instant::now();
    {
        let mut last = CPU_LAST_UPDATE.lock().unwrap();
        if let Some(prev) = *last {
            if now.duration_since(prev) < CPU_MIN_PERIOD {
                return SMOOTHED_CPU01.lock().unwrap().unwrap_or(0.0);
            }
        }
        *last = Some(now);
    }

    let sample_pct = sample_process_cpu_percent(); // 0..100
    let mut guard = SMOOTHED_CPU01.lock().unwrap();
    let alpha: f32 = 0.07;
    let smoothed = match sample_pct {
        Some(pct) => {
            let current = (pct / 100.0).clamp(0.0, 1.0);
            match *guard { Some(prev) => prev + alpha * (current - prev), None => current }
        }
        None => guard.unwrap_or(0.0),
    };
    *guard = Some(smoothed);
    smoothed
}

// Smoothed, normalized RAM usage for progress bars (0..1). Uses process working set
// normalized by a rolling max observed during this session.
static SMOOTHED_RAM01: Lazy<Mutex<Option<f32>>> = Lazy::new(|| Mutex::new(None));
static RAM_LAST_UPDATE: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));
const RAM_MIN_PERIOD: Duration = Duration::from_millis(500);

pub fn smoothed_ram01() -> f32 {
    // Rate-limit updates to avoid recomputing on every frame
    let now = Instant::now();
    {
        let mut last = RAM_LAST_UPDATE.lock().unwrap();
        if let Some(prev) = *last {
            if now.duration_since(prev) < RAM_MIN_PERIOD {
                return SMOOTHED_RAM01.lock().unwrap().unwrap_or(0.0);
            }
        }
        *last = Some(now);
    }

    // Use system memory ratio for a changing bar
    let (used, total) = system_mem_mb().unwrap_or((0.0, 0.0));
    let ratio = if total > 0.0 { (used / total).clamp(0.0, 1.0) } else { 0.0 };
    let mut g = SMOOTHED_RAM01.lock().unwrap();
    let alpha: f32 = 0.18;
    let smoothed = match *g { Some(prev) => prev + alpha * (ratio - prev), None => ratio };
    *g = Some(smoothed);
    smoothed
}

// Smoothed, normalized VRAM usage for progress bars (0..1).
static SMOOTHED_VRAM01: Lazy<Mutex<Option<f32>>> = Lazy::new(|| Mutex::new(None));
static VRAM_LAST_UPDATE: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));
const VRAM_MIN_PERIOD: Duration = Duration::from_millis(500);

pub fn smoothed_vram01() -> f32 {
    // Rate-limit updates
    let now = Instant::now();
    {
        let mut last = VRAM_LAST_UPDATE.lock().unwrap();
        if let Some(prev) = *last {
            if now.duration_since(prev) < VRAM_MIN_PERIOD {
                return SMOOTHED_VRAM01.lock().unwrap().unwrap_or(0.0);
            }
        }
        *last = Some(now);
    }
    let ratio = match gpu_mem_mb() { Some((used, total)) if total > 0.0 => (used / total).clamp(0.0, 1.0), _ => 0.0 };
    let mut g = SMOOTHED_VRAM01.lock().unwrap();
    // VRAM can spike; keep smoothing moderate
    let alpha: f32 = 0.16;
    let smoothed = match *g { Some(prev) => prev + alpha * (ratio - prev), None => ratio };
    *g = Some(smoothed);
    smoothed
}


// --- System metrics helpers ---
#[cfg(windows)]
pub fn process_mem_mb() -> Option<f32> {
    unsafe {
        let handle = GetCurrentProcess();
        let mut counters = PROCESS_MEMORY_COUNTERS::default();
        if K32GetProcessMemoryInfo(handle, &mut counters, std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32).as_bool() {
            let mb = counters.WorkingSetSize as f32 / (1024.0 * 1024.0);
            Some(mb)
        } else { None }
    }
}

// System-wide memory usage (used, total) in MiB
#[cfg(windows)]
pub fn system_mem_mb() -> Option<(f32, f32)> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut stat = MEMORYSTATUSEX::default();
        stat.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut stat).is_ok() {
            let total = stat.ullTotalPhys as f32 / (1024.0 * 1024.0);
            let avail = stat.ullAvailPhys as f32 / (1024.0 * 1024.0);
            let used = (total - avail).max(0.0);
            Some((used, total))
        } else { None }
    }
}

#[cfg(not(windows))]
pub fn system_mem_mb() -> Option<(f32, f32)> { None }

#[cfg(not(windows))]
pub fn gpu_mem_mb() -> Option<(f32, f32)> { None }

// Try to read VRAM usage via PDH performance counters (works well on NVIDIA drivers)
#[cfg(windows)]
fn gpu_mem_mb_pdh() -> Option<(f32, f32)> {
    use windows::{core::PCWSTR, Win32::System::Performance::{PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterValue, PdhOpenQueryW, PDH_FMT_COUNTERVALUE, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY}};
    unsafe {
        let mut query: PDH_HQUERY = std::mem::zeroed();
        if PdhOpenQueryW(None, 0, &mut query) != 0 { return None; }

        // Use _Total instance to aggregate across adapters
        let path_usage = widestring::U16CString::from_str(r"\GPU Adapter Memory(_Total)\Dedicated Usage").ok()?;
        let path_limit = widestring::U16CString::from_str(r"\GPU Adapter Memory(_Total)\Dedicated Limit").ok()?;
        let mut h_usage: PDH_HCOUNTER = std::mem::zeroed();
        let mut h_limit: PDH_HCOUNTER = std::mem::zeroed();
        if PdhAddEnglishCounterW(query, PCWSTR(path_usage.as_ptr()), 0, &mut h_usage) != 0 { PdhCloseQuery(query); return None; }
        if PdhAddEnglishCounterW(query, PCWSTR(path_limit.as_ptr()), 0, &mut h_limit) != 0 { PdhCloseQuery(query); return None; }
        if PdhCollectQueryData(query) != 0 { PdhCloseQuery(query); return None; }

        let mut usage_val: PDH_FMT_COUNTERVALUE = std::mem::zeroed();
        let mut limit_val: PDH_FMT_COUNTERVALUE = std::mem::zeroed();
        if PdhGetFormattedCounterValue(h_usage, PDH_FMT_LARGE, None, &mut usage_val) != 0 { PdhCloseQuery(query); return None; }
        if PdhGetFormattedCounterValue(h_limit, PDH_FMT_LARGE, None, &mut limit_val) != 0 { PdhCloseQuery(query); return None; }

        PdhCloseQuery(query);

        let used_bytes = usage_val.Anonymous.largeValue as f64;
        let total_bytes = limit_val.Anonymous.largeValue as f64;
        let used_mb = (used_bytes / (1024.0 * 1024.0)) as f32;
        let total_mb = (total_bytes / (1024.0 * 1024.0)) as f32;
        if total_mb > 0.0 { Some((used_mb, total_mb)) } else { None }
    }
}

#[cfg(not(windows))]
#[allow(dead_code)]
fn gpu_mem_mb_pdh() -> Option<(f32, f32)> { None }

#[cfg(not(windows))]
fn sample_process_cpu_percent() -> Option<f32> { None }
#[cfg(not(windows))]
#[allow(dead_code)]
fn process_mem_mb() -> Option<f32> { None }

#[cfg(windows)]
fn filetime_to_u64(ft: &FILETIME) -> u64 { ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64) }

#[cfg(windows)]
static CPU_SNAPSHOT: Lazy<Mutex<Option<(u64, std::time::Instant)>>> = Lazy::new(|| Mutex::new(None));

#[cfg(windows)]
fn sample_process_cpu_percent() -> Option<f32> {
    unsafe {
        let handle = GetCurrentProcess();
        let mut creation = FILETIME::default();
        let mut exit = FILETIME::default();
        let mut kernel = FILETIME::default();
        let mut user = FILETIME::default();
        if GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user).is_err() { return None; }
        let proc_time_100ns = filetime_to_u64(&kernel) + filetime_to_u64(&user);
        let now = std::time::Instant::now();
        let mut guard = CPU_SNAPSHOT.lock().ok()?;
        if let Some((prev_100ns, prev_t)) = *guard {
            let dt = now.duration_since(prev_t).as_secs_f64();
            if dt <= 0.0 { *guard = Some((proc_time_100ns, now)); return None; }
            let dproc = (proc_time_100ns.saturating_sub(prev_100ns)) as f64 / 10_000_000.0; // seconds
            let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
            let pct = ((dproc / dt) / cores * 100.0) as f32;
            *guard = Some((proc_time_100ns, now));
            Some(pct)
        } else {
            *guard = Some((proc_time_100ns, now));
            None
        }
    }
}
