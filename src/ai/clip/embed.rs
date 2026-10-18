const CLIP_INPUT_SIZE: u32 = 224;

impl super::ClipEngine {
    pub fn embed_image_path(&mut self, path: &str) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            super::ClipBackend::FastEmbed { image_model, .. } => {
                let img = image::ImageReader::open(path)?.decode()?;
                // Squash to the model input size before fastembed preprocessing.
                let img = img.resize_exact(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, image::imageops::FilterType::CatmullRom);
                let out = image_model.embed_images(vec![img])?;
                Ok(crate::ai::clip::l2_normalize(out.into_iter().next().unwrap()))
            }
            super::ClipBackend::Siglip(sig) => Ok(sig.embed_image_path(path)?),
        }
    }

    pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            super::ClipBackend::FastEmbed { text_model, .. } => {
                let out = text_model.embed(vec![text.to_string()], None)?;
                Ok(crate::ai::clip::l2_normalize(out.into_iter().next().unwrap()))
            }
            super::ClipBackend::Siglip(sig) => Ok(sig.embed_text(text)?),
        }
    }
    
    pub fn zero_shot_tags(&mut self, _image_vec: &[f32], _top_k: usize) -> Vec<String> { Vec::new() }

    pub fn zero_shot_category(&mut self, _image_vec: &[f32]) -> Option<String> { None }

}