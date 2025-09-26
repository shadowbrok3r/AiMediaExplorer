fn main() -> anyhow::Result<()> {
    simplelog::WriteLogger::init(
        log::LevelFilter::Trace,
        simplelog::Config::default(),
        std::fs::File::create("output.log").unwrap(),
    )
    .ok();

    // Accept image path via CLI arg; optional second arg is a GGUF path for the transformer.
    let mut args = std::env::args();
    let _exe = args.next();
    let img_path = args.next().unwrap_or_else(|| {
        r"E:\\Backups\\MiscBackups\\Galleries\\JPEG Graphics file\\FILE1190.JPG".to_string()
    });
    let transformer_gguf: Option<std::path::PathBuf> = args.next().map(std::path::PathBuf::from);
    let img = std::path::Path::new(&img_path);
    if !img.exists() {
        log::warn!(
            "[qwen-image-edit-e2e] Image path not found: {}. Pass an image path as the first argument.",
            img.display()
        );
        return Ok(());
    }

    let prefer = if candle_core::Device::new_cuda(0).is_ok() {  candle_core::DType::F32 } else { candle_core::DType::F32 };
    let pipe = if let Some(gguf) = transformer_gguf.clone() {
        log::info!("[qwen-image-edit-e2e] Using GGUF transformer: {}", gguf.display());
        smart_media::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf_with_overrides(
            "Qwen/Qwen-Image-Edit",
            prefer,
            Some(gguf),
            None,
            None,
        )?
    } else {
        log::info!("[qwen-image-edit-e2e] No GGUF provided; attempting GGUF fallback from safetensors if possible");
        smart_media::ai::qwen_image_edit::model::QwenImageEditPipeline::load_from_hf(
            "Qwen/Qwen-Image-Edit",
            prefer,
        )?
    };
    pipe.info();

    let mut opts = smart_media::ai::qwen_image_edit::EditOptions::default();
    opts.prompt = "remove the text watermark".to_string();
    // Faster validation: fewer steps by default (fixed)
    opts.num_inference_steps = 12;

    let png = pipe.run_edit(img, &opts)?;
    std::fs::write("qwen_edit_test_output.png", &png)?;
    log::info!(
        "[qwen-image-edit-e2e] Completed. Output: qwen_edit_test_output.png ({} bytes)",
        png.len()
    );

    Ok(())
}
