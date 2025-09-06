use crate::ai::{candle_llava::{load_image, utils::tokenizer_image_token}, joycap::extract_json_vision};
use candle_core::{DType, Tensor, IndexOp};
use std::path::Path;
use anyhow::Result;

impl super::JoyCaptionModel {
    pub fn run_generation_from_image(&self, prompt: &str, img: image::DynamicImage) -> Result<String> {
        let requested_dtype_str = self.llava_config.torch_dtype.clone();
        let mut effective_dtype = match requested_dtype_str.as_str() {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => DType::F32,
        };
        let device = candle_examples::device(if cfg!(feature="cpu") { true } else { false })?;
        log::error!("DEVICE: {device:?}");
        let is_cpu = device.is_cpu();
        if is_cpu {
            if matches!(effective_dtype, DType::BF16 | DType::F16) {
                log::info!("[dtype] CPU detected: falling back from {:?} to F32 for compatibility", effective_dtype);
                effective_dtype = DType::F32;
            }
        }
        let dtype = effective_dtype;
        let ((w, h), image_tensor) = load_image(&img, &self.processor, &self.llava_config, dtype)?;
        log::info!("[joycaption.gen] start size={}x{} prompt_len={}", w, h, prompt.len());
        let img_tensor = image_tensor.to_device(&self.device)?;
        log::error!("img_tensor: {:?}", img_tensor.dtype());
        let tokens = tokenizer_image_token(
            prompt,
            &self.tokenizer,
            self.llava_config.image_token_index as i64,
            &self.llava_config,
        )?;
        log::info!("[joycaption.gen] temperature={:.3}", self.temperature);
        let input_embeds = self.llava.prepare_inputs_labels_for_multimodal(&tokens, &[img_tensor], &[(w,h)])?;
        log::info!("[joycaption.gen] prepare_inputs_labels_for_multimodal");
        use candle_transformers::generation::{Sampling, LogitsProcessor};
        // Currently candle's LogitsProcessor::from_sampling supports temperature / argmax. top_p/top_k not wired yet here.
        let temperature = f64::from(self.temperature);
        let sampling = if temperature <= 0.0 { Sampling::ArgMax } else { Sampling::All { temperature } };
        
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);
        let mut cache = self.cache.clone();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut idx_pos = 0usize;
        let mut embeds = input_embeds.clone();
        const LOG_INTERVAL: usize = 8;
        log::info!("[joycaption.gen] running steps: {}", self.max_new_tokens);
        for step in 0..self.max_new_tokens {
            let (_, total_len, _) = embeds.dims3()?;
            let (context_size, context_index) = if cache.use_kv_cache && step > 0 { (1, idx_pos) } else { (total_len, 0) };
            let input = embeds.i((.., total_len - context_size.., ..))?;
            let logits = self.llava.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let (_, input_len, _) = input.dims3()?; idx_pos += input_len;
            let next_token = logits_processor.sample(&logits)?;
            if next_token as usize == self.eos_token_id { log::info!("[joycaption.gen] eos step={} total_tokens={}", step, token_ids.len()); break; }
            let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &self.device)?;
            let next_embeds = self.llava.llama.embed(&next_token_tensor)?.unsqueeze(0)?;
            embeds = Tensor::cat(&[embeds, next_embeds], 1)?;
            token_ids.push(next_token);
            if step % LOG_INTERVAL == 0 { log::info!("[joycaption.gen] step={} generated_tokens={} last_token={}", step, token_ids.len(), next_token); }
        }
        let text = if token_ids.is_empty() { String::new() } else { self.tokenizer.decode(&token_ids, true).unwrap_or_default() };
        log::info!("[joycaption.gen] done chars={} tokens={}", text.len(), token_ids.len());
        Ok(text)
    }

    pub fn describe_image(&self, image_path: &Path) -> Result<crate::ai::VisionDescription> {
        let instruction = "Analyze the image and produce concise JSON with keys: description (detailed multi-sentence), caption (short), tags (array of lowercase single-word nouns), category (single general category). Return ONLY JSON.";
        let (prompt, _query) = self.build_prompt(instruction);
        let raw = self.run_generation_from_image(&prompt, image::ImageReader::open(image_path)?.decode()?)?;
        let vd = extract_json_vision(&raw)
            .and_then(|v| serde_json::from_value::<crate::ai::VisionDescription>(v).ok())
            .unwrap_or_else(|| crate::ai::VisionDescription { description: raw.trim().to_string(), caption: raw.split('.').next().unwrap_or(&raw).trim().to_string(), tags: Vec::new(), category: "general".into() });
        Ok(vd)
    }
}