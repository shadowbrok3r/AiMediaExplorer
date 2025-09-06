#![allow(unused)]
pub mod clip;
pub mod clip_image_processor;
pub mod config;
pub mod constants;
pub mod conversation;
pub mod llama;
pub mod model;
pub mod utils;

pub fn load_image(
    img: &image::DynamicImage,
    processor: &clip_image_processor::CLIPImageProcessor,
    llava_config: &config::LLaVAConfig,
    dtype: candle_core::DType,
) -> anyhow::Result<((u32, u32), candle_core::Tensor)> {
    let img_tensor = crate::ai::candle_llava::utils::process_image(img, processor, llava_config)?;
    Ok(((img.width(), img.height()), img_tensor.to_dtype(dtype)?))
}