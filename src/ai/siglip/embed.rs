use candle_core::Tensor;

impl super::SiglipEngine {
    pub fn embed_image_path(&self, path: &str) -> anyhow::Result<Vec<f32>, anyhow::Error> {
        log::info!("Embedding image from siglip");
        let img = self.load_image_tensor(path)?;
        let imgs = Tensor::stack(&[img], 0)?; // [1,3,H,W]
        let feats = self.model.get_image_features(&imgs)?;
        let v = feats.flatten_all()?.to_vec1::<f32>()?;
        Ok(l2_normalize(v))
    }

    pub fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>, anyhow::Error> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
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

    pub fn logits_image_vs_texts(
        &self,
        image_path: &str,
        texts: &[String],
    ) -> anyhow::Result<Vec<f32>, anyhow::Error> {
        // Helper for testing using real data with the official forward API (no dummy inputs)
        let img = self.load_image_tensor(image_path)?;
        let images = Tensor::stack(&[img], 0)?;
        // Tokenize and pad multiple sequences
        let pad_id = self.config.text_config.pad_token_id as i64;
        let max_len = self.config.text_config.max_position_embeddings;
        let mut tokens: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(t.clone(), true)
                .map_err(anyhow::Error::msg)?;
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            if ids.len() < max_len {
                ids.extend(std::iter::repeat(pad_id).take(max_len - ids.len()));
            } else if ids.len() > max_len {
                ids.truncate(max_len);
            }
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

pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
