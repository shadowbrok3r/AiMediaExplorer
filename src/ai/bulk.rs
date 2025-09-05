use crate::utilities::types::{FoundFile, IMAGE_EXTS};
use crate::ai::AISearchEngine;

/// Spawn an async task to bulk-generate (vision) descriptions for all image rows.
/// Updates `progress` (done,total) and flips `bulk_flag` while running.
/// On failure or empty queue, sets an error message via `err_sig`.
pub fn spawn_bulk_generate(
    engine_opt: Option<AISearchEngine>,
    rows: Vec<FoundFile>,
    prompt_template: String,
    progress: crossbeam::channel::Sender<(usize, usize)>,
    bulk_flag: crossbeam::channel::Sender<bool>,
    err_sig: crossbeam::channel::Sender<Option<String>>,
    overwrite: bool,
) {
    let _ = bulk_flag.try_send(true);
    tokio::spawn(async move {
        // initialize progress (0 done, total)
        let _ = progress.try_send((0, rows.len()));
        if let Some(engine) = engine_opt {
            // Allow enabling per-iteration memory logging with env var to diagnose RSS growth.
            let log_mem = std::env::var("SM_LOG_MEM").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
            for (idx, f) in rows.iter().enumerate() {
                // Update active vision path so UI can follow during bulk
                {
                    let mut act = engine.active_vision_path.lock().await;
                    *act = Some(f.path.display().to_string());
                }
                if log_mem { super::log_memory_usage(Some(&format!("before {}", f.path.display()))); }
                // Skip if description exists and we are not overwriting
                if !overwrite {
                    if let Some(meta) = engine.get_file_metadata(&f.path.display().to_string()).await {
                        if meta.description.is_some() {
                            let _ = progress.try_send((idx + 1, rows.len()));
                            if log_mem { super::log_memory_usage(Some(&format!("skip {}", f.path.display()))); }
                            continue;
                        }
                    }
                }
                let ext = f
                    .path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                if !IMAGE_EXTS.iter().any(|e| *e == ext) {
                    let _ = progress.try_send((idx + 1, rows.len()));
                    continue;
                }
                #[cfg(feature = "joycaption")]
                {
                    if crate::ai::joycaption_adapter::is_enabled() {
                        if let Ok(bytes) = tokio::fs::read(&f.path).await {
                            let mut interim = String::new();
                            let _ = crate::ai::joycaption_adapter::stream_describe_bytes_with_callback(
                                bytes,
                                &prompt_template,
                                |frag| {
                                    interim.push_str(frag);
                                },
                            )
                            .await;
                            if let Some(val) =
                                crate::ai::joycaption_adapter::extract_json_vision(&interim)
                            {
                                if let Ok(vd) = serde_json::from_value::<crate::ai::generate::VisionDescription>(val) {
                                    let _ = engine
                                        .apply_vision_description(
                                            &f.path.display().to_string(),
                                            &vd,
                                        )
                                        .await;
                                }
                            } else {
                                log::warn!("[bulk] No VisionDescription JSON produced for {} - skipping", f.path.display());
                            }
                        }
                    } else {
                        if let Some(vd) = engine.generate_vision_description(&f.path).await {
                            let _ = engine
                                .apply_vision_description(
                                    &f.path.display().to_string(),
                                    &vd,
                                )
                                .await;
                        } else {
                            log::warn!("[bulk] VisionDescription generation returned None for {}", f.path.display());
                        }
                    }
                }
                #[cfg(not(feature = "joycaption"))]
                {
                    if let Some(vd) = engine.generate_vision_description(&f.path).await {
                        let p_str = f.path.display().to_string();
                        // Apply (which attempts persistence). Add small retry loop for DB reliability.
                        let mut attempts = 0usize;
                        let mut last_err: Option<anyhow::Error> = None;
                        loop {
                            attempts += 1;
                            match engine.apply_vision_description(&p_str, &vd).await {
                                Ok(_) => { log::info!("[bulk] persisted vision metadata for {} (attempt {})", p_str, attempts); break; }
                                Err(e) => {
                                    log::warn!("[bulk] persist attempt {} failed for {}: {}", attempts, p_str, e);
                                    last_err = Some(e);
                                    if attempts >= 3 { break; }
                                    // backoff (50ms, 100ms)
                                    let delay_ms = 50 * attempts as u64;
                                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                                }
                            }
                        }
                        if let Some(e) = last_err { if attempts >= 3 { log::error!("[bulk] giving up persisting {} after {} attempts: {}", p_str, attempts, e); } }
                    } else {
                        log::warn!("[bulk] VisionDescription generation returned None for {}", f.path.display());
                    }
                }
                if log_mem { super::log_memory_usage(Some(&format!("after {}", f.path.display()))); }
                let _ = progress.try_send((idx + 1, rows.len()));
            }
        } else {
            let _ = err_sig.try_send(Some("AI engine not initialized".into()));
        }
        let _ = bulk_flag.try_send(false);
    });
}
