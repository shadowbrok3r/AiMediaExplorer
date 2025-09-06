use std::path::PathBuf;

use eframe::egui::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;
#[cfg(windows)]
use windows::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
#[cfg(windows)]
use windows::Win32::Foundation::FILETIME;
use crate::ui::status::{self, GlobalStatusIndicator};
use humansize::{format_size, DECIMAL};

use crate::list_drive_infos;

// We assume SmartMediaApp has (or will get) a boolean `ai_initializing` and `ai_ready` flags plus `open_settings_modal`.
// If they don't exist yet, they need to be added to `SmartMediaApp` (app.rs). For now we optimistically reference via super::MainPage's parent.

impl super::MainPage {
    pub fn navbar(&mut self, ctx: &Context) {
        TopBottomPanel::top("MainPageTopPanel")
        .exact_height(25.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {

                ui.menu_button("File", |ui| {
                    ui.menu_button("AI", |ui| {
                        // Enable AI Search: spawn async global init if not already in progress
                        if ui.button("Enable AI Search").clicked() {
                            status::JOY_STATUS.set_state(status::StatusState::Initializing, "Loading vision model");
                            tokio::spawn(async move {
                                crate::ai::init_global_ai_engine_async().await;
                                status::JOY_STATUS.set_state(status::StatusState::Idle, "Ready");
                            });
                        }
                        if ui.button("Enable Auto Indexing").clicked() {
                            // Toggle auto indexing flag inside global engine (atomic set true)
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::JOY_STATUS.set_state(status::StatusState::Running, "Auto descriptions enabled");
                        }
                        if ui.button("Enable Auto CLIP").clicked() {
                            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
                            status::CLIP_STATUS.set_state(status::StatusState::Running, "Auto CLIP backfill");
                            tokio::spawn(async move {
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                log::info!("[CLIP] Auto CLIP enabled via navbar; backfill added {added}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, "Idle");
                                Ok::<(), anyhow::Error>(())
                            });
                        }
                        if ui.button("Generate CLIP (Missing)").clicked() {
                            status::CLIP_STATUS.set_state(status::StatusState::Running, "Generating missing");
                            tokio::spawn(async move {
                                let count_before = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                let remaining = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Manual missing generation added {added}; remaining {remaining} (was {count_before})");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, format!("Remaining {remaining}"));
                                Ok::<(), anyhow::Error>(())
                            });
                        }
                        if ui.button("Generate CLIP (Selected)").clicked() {
                            let path = self.file_explorer.current_thumb.path.clone();
                            if !path.is_empty() {
                                status::CLIP_STATUS.set_state(status::StatusState::Running, "Selected path");
                                tokio::spawn(async move {
                                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await?;
                                    log::info!("[CLIP] Selected generation for {path} -> added {added}");
                                    status::CLIP_STATUS.set_state(status::StatusState::Idle, "Idle");
                                    Ok::<(), anyhow::Error>(())
                                });
                            } else {
                                log::warn!("[CLIP] No current preview path set for generation");
                            }
                        }
                        if ui.button("CLIP Missing Count").clicked() {
                            tokio::spawn(async move {
                                let missing = crate::ai::GLOBAL_AI_ENGINE.clip_missing_count().await;
                                log::info!("[CLIP] Missing embeddings: {missing}");
                                status::CLIP_STATUS.set_state(status::StatusState::Idle, format!("Missing {missing}"));
                            });
                        }
                    });
                    if ui.button("Preferences").clicked() {
                        crate::app::OPEN_SETTINGS_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });

                ui.menu_button("Quick Access", |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.add_space(5.);
                        ui.heading("User Directories");
                        ui.add_space(5.);
                        for access in crate::quick_access().iter() {
                            ui.separator();
                            if Button::new(&access.icon).min_size(vec2(20., 20.)).right_text(&access.label).ui(ui).on_hover_text(&access.label).clicked() {
                                self.file_explorer.set_path(access.path.to_string_lossy());
                            }
                        }
                        ui.add_space(5.);
                        ui.heading("Drives");
                        ui.add_space(5.);
                        for drive in list_drive_infos() {
                            ui.separator();
                            let root = drive.root.clone();
                            let display = format!("{} - {}", drive.drive_type, drive.root);
                            let path = PathBuf::from(root.clone());
                            let free = format_size(drive.free, DECIMAL);
                            let total = format_size(drive.total, DECIMAL);

                            let response = Button::new(display)
                            .right_text(&drive.label)
                            .ui(ui).on_hover_text(format!("{free} free of {total}"));


                            if response.clicked() {
                                self.file_explorer.set_path(path.to_string_lossy());
                            }
                        }
                    });
                });
                
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.checkbox(&mut self.open_log_window, "View Logs");
                    ui.separator();
                    // System metrics (CPU, RAM, VRAM)
                    let cpu01 = smoothed_cpu01(); // normalized 0..1 with EMA smoothing
                    log::debug!("CPU: {:.2}%", cpu01 * 100.0);
                    ProgressBar::new(cpu01)
                    .animate(false)
                    .desired_width(100.)
                    .desired_height(1.)
                    .fill(ui.style().visuals.error_fg_color)
                    .show_percentage()
                    .ui(ui);
                    
                    ui.label("CPU %");
                    ui.separator(); 
                    if let Some(ram_mb) = process_mem_mb() { 
                        ctx.request_repaint();
                        ui.label(format!("RAM: {:.0} MiB", ram_mb)); 
                        ui.separator(); 
                    }
                    if let Some((v_used, v_total)) = gpu_mem_mb() { 
                        ui.label(format!("VRAM: {:.0}/{:.0} MiB", v_used, v_total)); 
                        ui.separator(); 
                    }
                    status::status_bar_inline(ui);
                });
            });
        });
    }
}

// --- System metrics helpers ---
#[cfg(windows)]
fn process_mem_mb() -> Option<f32> {
    unsafe {
        let handle = GetCurrentProcess();
        let mut counters = PROCESS_MEMORY_COUNTERS::default();
        if K32GetProcessMemoryInfo(handle, &mut counters, std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32).as_bool() {
            let mb = counters.WorkingSetSize as f32 / (1024.0 * 1024.0);
            Some(mb)
        } else { None }
    }
}

#[cfg(not(windows))]
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

#[cfg(not(windows))]
fn sample_process_cpu_percent() -> Option<f32> { None }

// Placeholder: GPU memory query (requires DXGI; return None for now)
fn gpu_mem_mb() -> Option<(f32, f32)> { None }

// --- Smoothing / normalization helpers ---
static SMOOTHED_CPU01: Lazy<Mutex<Option<f32>>> = Lazy::new(|| Mutex::new(None));

fn smoothed_cpu01() -> f32 {
    let sample_pct = sample_process_cpu_percent(); // 0..100
    let mut guard = SMOOTHED_CPU01.lock().unwrap();
    let alpha: f32 = 0.15; // smoothing factor (lower = smoother)
    match sample_pct {
        Some(pct) => {
            let current = (pct / 100.0).clamp(0.0, 1.0);
            let smoothed = match *guard {
                Some(prev) => prev + alpha * (current - prev),
                None => current,
            };
            *guard = Some(smoothed);
            smoothed
        }
        None => guard.unwrap_or(0.0),
    }
}