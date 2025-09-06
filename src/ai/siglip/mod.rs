use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
pub mod embed;
// pub use embed::*;

pub struct SiglipEngine {
    model: siglip::Model,
    tokenizer: Tokenizer,
    device: Device,
    config: siglip::Config,
    image_size: usize,
}

impl SiglipEngine {
    pub fn new(model_key: &str) -> Result<Self> {
        let hf_repo = map_model_key_to_repo(model_key);
        let device = candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu);
        let api = Api::new()?;
        let mdl = api.model(hf_repo.to_string());
        let model_file = mdl.get("model.safetensors")?;
        let config_file = mdl.get("config.json")?;
        let tokenizer = load_tokenizer(hf_repo, None)?;
        let config: siglip::Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let image_size = config.vision_config.image_size;
        // Load weights mmap
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&model_file),
                DType::F32,
                &device,
            )?
        };
        let model = siglip::Model::new(&config, vb)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            image_size,
        })
    }

    fn load_image_tensor<T: AsRef<std::path::Path>>(&self, path: T) -> Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (self.image_size, self.image_size);
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
        let img = img.to_rgb8();
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?;
        Ok(img.to_device(&self.device)?)
    }
}

fn load_tokenizer(hf_repo: &str, override_path: Option<&str>) -> Result<Tokenizer> {
    let path = if let Some(p) = override_path {
        std::path::PathBuf::from(p)
    } else {
        let api = Api::new()?;
        api.model(hf_repo.to_string()).get("tokenizer.json")?
    };
    Ok(Tokenizer::from_file(path).map_err(anyhow::Error::msg)?)
}

fn map_model_key_to_repo(model_key: &str) -> &'static str {
    match model_key {
        "siglip-base-patch16-224" => "google/siglip-base-patch16-224",
        "siglip2-base-patch16-224" => "google/siglip2-base-patch16-224",
        "siglip2-base-patch16-256" => "google/siglip2-base-patch16-256",
        "siglip2-base-patch16-384" => "google/siglip2-base-patch16-384",
        "siglip2-base-patch16-512" => "google/siglip2-base-patch16-512",
        "siglip2-large-patch16-256" => "google/siglip2-large-patch16-256",
        "siglip2-large-patch16-384" => "google/siglip2-large-patch16-384",
        "siglip2-large-patch16-512" => "google/siglip2-large-patch16-512",
        _ => "google/siglip-base-patch16-224",
    }
}
