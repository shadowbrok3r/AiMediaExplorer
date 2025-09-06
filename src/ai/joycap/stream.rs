use crate::ai::candle_llava::load_image;
use crate::ai::candle_llava::utils::tokenizer_image_token;
use candle_core::{DType, IndexOp, Tensor};

impl super::JoyCaptionModel {
    pub fn stream_generate_bytes(
        &self,
        prompt: &str,
        bytes: &[u8],
        on_token: impl FnMut(&str),
    ) -> anyhow::Result<String, anyhow::Error> {
        let img = image::load_from_memory(bytes)?;
        self.stream_generate_from_image(prompt, img, on_token)
    }

    pub fn stream_generate_from_image(
        &self,
        prompt: &str,
        img: image::DynamicImage,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<String, anyhow::Error> {
        use crate::ui::status::{
            GlobalStatusIndicator, JOY_STATUS, StatusState, VISION_MAX_TOKENS, VISION_TOKENS,
        };
        let requested_dtype_str = self.llava_config.torch_dtype.clone();
        let mut effective_dtype = match requested_dtype_str.as_str() {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => DType::F32,
        };
        let device = candle_examples::device(if cfg!(feature = "cpu") { true } else { false })?;
        log::error!("DEVICE: {device:?}");
        let is_cpu = device.is_cpu();
        if is_cpu {
            if matches!(effective_dtype, DType::BF16 | DType::F16) {
                log::info!(
                    "[dtype] CPU detected: falling back from {:?} to F32 for compatibility",
                    effective_dtype
                );
                effective_dtype = DType::F32;
            }
        }
        let dtype = effective_dtype;
        let ((w, h), image_tensor) = load_image(&img, &self.processor, &self.llava_config, dtype)?;
        log::info!(
            "[joycaption.gen] start size={}x{} prompt_len={}",
            w,
            h,
            prompt.len()
        );
        let img_tensor = image_tensor.to_device(&self.device)?;
        log::error!("img_tensor: {:?}", img_tensor.dtype());
        let tokens = tokenizer_image_token(
            prompt,
            &self.tokenizer,
            self.llava_config.image_token_index as i64,
            &self.llava_config,
        )?;
        log::info!("[joycaption.gen] temperature={:.3}", self.temperature);
        JOY_STATUS.set_state(StatusState::Running, format!("Generating ({w}x{h})"));
        JOY_STATUS.set_progress(0, self.max_new_tokens as u64);
        VISION_TOKENS.store(0, std::sync::atomic::Ordering::Relaxed);
        VISION_MAX_TOKENS.store(self.max_new_tokens, std::sync::atomic::Ordering::Relaxed);
        let input_embeds =
            self.llava
                .prepare_inputs_labels_for_multimodal(&tokens, &[img_tensor], &[(w, h)])?;
        use candle_transformers::generation::{LogitsProcessor, Sampling};
        let temperature = f64::from(self.temperature);
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);
        let mut cache = self.cache.clone();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut idx_pos = 0usize;
        let mut embeds = input_embeds.clone();
        let mut last_decoded_len = 0usize;
        const STREAM_LOG_INTERVAL: usize = 8;
        for step in 0..self.max_new_tokens {
            let (_, total_len, _) = embeds.dims3()?;
            let (context_size, context_index) = if cache.use_kv_cache && step > 0 {
                (1, idx_pos)
            } else {
                (total_len, 0)
            };
            let input = embeds.i((.., total_len - context_size.., ..))?;
            let logits = self.llava.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let (_, input_len, _) = input.dims3()?;
            idx_pos += input_len;
            let next_token = logits_processor.sample(&logits)?;
            if next_token as usize == self.eos_token_id {
                log::debug!(
                    "[joycaption.stream] eos step={} total_tokens={}",
                    step,
                    token_ids.len()
                );
                break;
            }
            let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &self.device)?;
            let next_embeds = self.llava.llama.embed(&next_token_tensor)?.unsqueeze(0)?;
            embeds = Tensor::cat(&[embeds, next_embeds], 1)?;
            token_ids.push(next_token);
            // Progress update (token based)
            let produced = token_ids.len() as u64;
            VISION_TOKENS.store(produced as usize, std::sync::atomic::Ordering::Relaxed);
            JOY_STATUS.set_progress(produced, self.max_new_tokens as u64);
            // Attempt ultra-incremental decode: decode just the last token id alone to guess its text.
            // Some tokenizers may require full context to merge bytes properly; fallback to full decode diff if needed.
            let mut emitted_this_step = false;
            if let Ok(last_piece) = self.tokenizer.decode(&[next_token], true) {
                if !last_piece.is_empty() {
                    on_token(&last_piece);
                    emitted_this_step = true;
                }
            }
            if !emitted_this_step {
                // Fallback: full decode diff (previous behavior)
                if let Ok(full_so_far) = self.tokenizer.decode(&token_ids, true) {
                    if full_so_far.len() > last_decoded_len {
                        let new_part = &full_so_far[last_decoded_len..];
                        if !new_part.is_empty() {
                            on_token(new_part);
                        }
                        last_decoded_len = full_so_far.len();
                    }
                }
            } else {
                // Maintain last_decoded_len by full length occasionally to keep diff logic consistent
                if step % STREAM_LOG_INTERVAL == 0 {
                    if let Ok(full_so_far) = self.tokenizer.decode(&token_ids, true) {
                        last_decoded_len = full_so_far.len();
                    }
                }
            }
            if step % STREAM_LOG_INTERVAL == 0 {
                log::info!(
                    "[joycaption.stream] step={} generated_tokens={} last_token={}",
                    step,
                    token_ids.len(),
                    next_token
                );
            }
        }
        let text = if token_ids.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode(&token_ids, true).unwrap_or_default()
        };
        log::info!(
            "[joycaption.stream] done chars={} tokens={}\nText: {}",
            text.len(),
            token_ids.len(),
            text
        );
        JOY_STATUS.set_state(
            StatusState::Idle,
            format!("Generated {} tokens", token_ids.len()),
        );
        Ok(text)
    }
}
