use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct SiglipEngine {
    model: siglip::Model,
    tokenizer: Tokenizer,
    device: Device,
    config: siglip::Config,
    image_size: usize,
}

fn load_tokenizer(hf_repo: &str, override_path: Option<&str>) -> Result<Tokenizer> {
    let path = if let Some(p) = override_path { std::path::PathBuf::from(p) } else {
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
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&model_file), DType::F32, &device)? };
        let model = siglip::Model::new(&config, vb)?;
        Ok(Self { model, tokenizer, device, config, image_size })
    }

    fn load_image_tensor<T: AsRef<std::path::Path>>(&self, path: T) -> Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (self.image_size, self.image_size);
        let img = img.resize_to_fill(width as u32, height as u32, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?;
        Ok(img.to_device(&self.device)?)
    }

    pub fn embed_image_path(&self, path: &str) -> Result<Vec<f32>> {
        let img = self.load_image_tensor(path)?;
        let imgs = Tensor::stack(&[img], 0)?; // [1,3,H,W]
        let feats = self.model.get_image_features(&imgs)?;
        let v = feats.flatten_all()?.to_vec1::<f32>()?;
        Ok(l2_normalize(v))
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
        let max_len = self.config.text_config.max_position_embeddings;
        let pad_id = self.config.text_config.pad_token_id as i64;
        if ids.len() < max_len {
            ids.extend(std::iter::repeat(pad_id).take(max_len - ids.len()));
        } else if ids.len() > max_len {
            ids.truncate(max_len);
        }
        let ids = Tensor::from_vec(ids, (1, max_len), &self.device)?; // [1, L]
        let feats = self.model.get_text_features(&ids)?;
        let v = feats.flatten_all()?.to_vec1::<f32>()?;
        Ok(l2_normalize(v))
    }

    pub fn logits_image_vs_texts(&self, image_path: &str, texts: &[String]) -> Result<Vec<f32>> {
        // Helper for testing using real data with the official forward API (no dummy inputs)
        let img = self.load_image_tensor(image_path)?;
        let images = Tensor::stack(&[img], 0)?; 
        // Tokenize and pad multiple sequences
        let pad_id = self.config.text_config.pad_token_id as i64;
        let max_len = self.config.text_config.max_position_embeddings;
        let mut tokens: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self.tokenizer.encode(t.clone(), true).map_err(anyhow::Error::msg)?;
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            if ids.len() < max_len { ids.extend(std::iter::repeat(pad_id).take(max_len - ids.len())); }
            else if ids.len() > max_len { ids.truncate(max_len); }
            tokens.push(ids);
        }
        let input_ids = Tensor::new(tokens, &self.device)?; // [T, L]
        // The model expects [B, L] shaped text ids. Stack into batch of T and reuse the single image by broadcasting.
        // To match example semantics (scores per text for the image), forward on (images, input_ids) and read logits_per_image.
        let (_logits_per_text, logits_per_image) = self.model.forward(&images, &input_ids)?;
        let v = logits_per_image.flatten_all()?.to_vec1::<f32>()?;
        Ok(v)
    }
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }
    v
}
