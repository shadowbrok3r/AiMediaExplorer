#[test]
fn qwen_image_edit_e2e_internal() {
    use std::path::Path;
    use candle_core::DType;
    use simplelog::{Config, LevelFilter, WriteLogger};

    let _ = WriteLogger::init(
        LevelFilter::Trace,
        Config::default(),
        std::fs::File::create("output.log").expect("create output.log"),
    );

    // Allow overriding image path via env for local runs; default to the requested path
    let img_path = std::env::var("QWEN_EDIT_IMAGE").unwrap_or_else(|_| {
        r"E:\\Backups\\MiscBackups\\Galleries\\JPEG Graphics file\\FILE1190.JPG".to_string()
    });
    let img = Path::new(&img_path);

    // Skip early if the image does not exist on this machine; log a warning so devs can see it
    if !img.exists() {
        log::warn!("[qwen-image-edit-test] Image path does not exist on this machine: {}. Set QWEN_EDIT_IMAGE to a valid path to run e2e.", img.display());
        return;
    }

    // Prefer F16 on CUDA hardware
    let prefer = if candle_core::Device::new_cuda(0).is_ok() { DType::F16 } else { DType::F32 };

    // Load pipeline strictly from HF (non-GGUF)
    let pipe = crate::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(
        "Qwen/Qwen-Image-Edit",
        prefer,
    ).expect("load_from_hf failed");
    pipe.info();

    // Build options
    let mut opts = crate::ai::qwen_image_edit::EditOptions::default();
    opts.prompt = "remove the text watermark".to_string();

    // Run edit
    let png = pipe.run_edit(img.as_ref(), &opts).expect("run_edit failed");
    assert!(!png.is_empty(), "Empty PNG output");

    // Write output file in workspace root for quick inspection
    std::fs::write("qwen_edit_test_output.png", &png).expect("write output png");

    log::info!("[qwen-image-edit-test] Completed successfully. Output: qwen_edit_test_output.png ({} bytes)", png.len());
}
