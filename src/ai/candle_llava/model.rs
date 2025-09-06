use super::clip::clip_vit_large_patch14_336;
use super::llama::Cache;
use super::llama::Llama;
use super::utils::get_anyres_image_grid_shape;
use candle_core::{bail, Device, IndexOp, Result, Tensor, DType};
use candle_nn::Module;
use candle_nn::{seq, Activation, Sequential, VarBuilder};
use candle_transformers::models::clip::vision_model::ClipVisionConfig;
use candle_transformers::models::with_tracing::linear;
use regex::Regex;

use super::clip::ClipVisionTransformerWithHiddenStates;
use super::config::LLaVAConfig;

fn mlp_gelu_match(mm_projector_type: &str) -> Option<usize> {
    let mlp_gelu_regex = Regex::new(r"^mlp(\d+)x_gelu$").unwrap();

    if let Some(captures) = mlp_gelu_regex.captures(mm_projector_type) {
        if let Some(match_str) = captures.get(1) {
            let match_str = match_str.as_str();
            match_str.parse::<usize>().ok()
        } else {
            None
        }
    } else {
        None
    }
}

fn unpad_image(tensor: &Tensor, original_size: &(u32, u32)) -> Result<Tensor> {
    assert_eq!(tensor.dims().len(), 3);
    let (original_width, original_height) = *original_size;
    let tensor_dims = tensor.dims();
    let current_height = tensor_dims[1];
    let current_width = tensor_dims[2];
    let original_aspect_ratio = (original_width as f32) / (original_height as f32);
    let current_aspect_ratio = (current_width as f32) / (current_height as f32);
    if original_aspect_ratio > current_aspect_ratio {
        let scale_factor = (current_width as f32) / (original_width as f32);
        let new_height = (original_height as f32 * scale_factor).floor() as usize;
        let padding = (current_height - new_height) / 2;
        tensor.i((.., padding..current_width - padding, ..))
    } else {
        let scale_factor = (current_height as f32) / (original_height as f32);
        let new_width = (original_width as f32 * scale_factor).floor() as usize;
        let padding = (current_width - new_width) / 2;
        tensor.i((.., .., padding..current_width - padding))
    }
}

pub struct IdentityMap {}

impl Module for IdentityMap {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

pub struct MMProjector {
    pub modules: Sequential,
}

impl MMProjector {
    pub fn load(vb: &VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        log::info!(
            "[mm_projector.load] type={} mm_hidden_size={} llama_hidden_size={} name_or_path={}",
            config.mm_projector_type, config.mm_hidden_size, config.hidden_size, config._name_or_path
        );
        if config.mm_projector_type == "linear" {
            let vb_prefix = if config._name_or_path.contains("hf") {
                "multi_modal_projector.linear_1"
            } else {
                "model.mm_projector.0"
            };
            let linear = linear(config.mm_hidden_size, config.hidden_size, vb.pp(vb_prefix))?;
            let modules = seq().add(linear);
            Ok(Self { modules })
        } else if let Some(mlp_depth) = mlp_gelu_match(&config.mm_projector_type) {
            log::info!(
                "[mm_projector.load] mlp_gelu_depth={} ({} layers of linear+gelu incl first)",
                mlp_depth, mlp_depth
            );
            let modules = if config._name_or_path.contains("hf") {
                let mut modules = seq().add(linear(
                    config.mm_hidden_size,
                    config.hidden_size,
                    vb.pp("multi_modal_projector.linear_1"),
                )?);
                for i in 1..mlp_depth {
                    modules = modules.add(Activation::Gelu).add(linear(
                        config.hidden_size,
                        config.hidden_size,
                        vb.pp(format!("multi_modal_projector.linear_{}", i + 1)),
                    )?);
                }
                modules
            } else {
                let mut modules = seq().add(linear(
                    config.mm_hidden_size,
                    config.hidden_size,
                    vb.pp("model.mm_projector.0"),
                )?);
                for i in 1..mlp_depth {
                    modules = modules.add(Activation::Gelu).add(linear(
                        config.hidden_size,
                        config.hidden_size,
                        vb.pp(format!("model.mm_projector.{}", i * 2)),
                    )?);
                }
                modules
            };
            Ok(Self { modules })
        } else if config.mm_projector_type == "identity" {
            log::info!("[mm_projector.load] identity projector - passing vision features through");
            Ok(Self {
                modules: seq().add(IdentityMap {}),
            })
        } else {
            bail!(
                "Unsupported MM projector type: {}",
                config.mm_projector_type
            )
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.modules.forward(x)
    }
}

pub struct ClipVisionTower {
    model: ClipVisionTransformerWithHiddenStates,
    select_layer: isize,
    select_feature_method: String,
    pub config: ClipVisionConfig,
}

impl ClipVisionTower {
    pub fn new(
        vb: VarBuilder,
        select_layer: isize,
        select_feature_method: &str,
        config: &Option<ClipVisionConfig>,
    ) -> Result<Self> {
        // to simulate hidden_state of python version clip
        let _config = if config.is_none() {
            clip_vit_large_patch14_336()
        } else {
            config.clone().unwrap()
        };
        let select_layer = match select_layer {
            -1 | -2 => select_layer,
            _ => bail!("Unsupported select layer: {}", select_layer),
        };
        let model = ClipVisionTransformerWithHiddenStates::new(vb, &_config)?;
        Ok(Self {
            model,
            select_layer,
            select_feature_method: select_feature_method.to_string(),
            config: _config,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        log::info!("ClipVisionTower::forward");
        let result = self.model.output_hidden_states(x)?;
        let index = result.len() as isize + self.select_layer;
        let result = result[index as usize].clone();
        if self.select_feature_method == "cls_patch" {
            Ok(result)
        } else {
            result.i((.., 1..))
        }
    }

    pub fn num_patches_per_side(&self) -> usize {
        self.config.image_size / self.config.patch_size
    }
}

pub struct LLaVA {
    pub clip_vision_tower: ClipVisionTower,
    pub image_newline: Tensor,
    pub mm_projector: MMProjector,
    pub llama: Llama,
    config: LLaVAConfig,
    device: Device,
}

impl LLaVA {
    pub fn load(
        vb: VarBuilder,
        config: &LLaVAConfig,
        clip_vision_config: Option<ClipVisionConfig>,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let llama_config = config.to_llama_config();
        let mm_projector = MMProjector::load(&vb, config)?;
        log::info!("config._name_or_path: {}", config._name_or_path);
        let (clip_vision_tower, image_newline, llama) = if config._name_or_path.contains("hf") {
            let clip = ClipVisionTower::new(
                vb.pp("vision_tower.vision_model"),
                config.mm_vision_select_layer,
                &config.mm_vision_select_feature,
                &clip_vision_config,
            )?;
            let target_dtype = match config.torch_dtype.as_str() { "bfloat16" => DType::BF16, "float16" => DType::F16, _ => DType::F32 };
            let image_newline = match vb.get(&[config.hidden_size], "image_newline") {
                Ok(t) => {
                    log::info!("[llava.load] loaded image_newline tensor");
                    t.to_device(&device)?.to_dtype(target_dtype)?
                }
                Err(e) => {
                    log::info!(
                        "[llava.load] WARNING missing image_newline tensor ({}). Creating fallback vector (random normal).",
                        e
                    );
                    Tensor::randn(0f32, 0.02f32, &[config.hidden_size], &device)?.to_dtype(target_dtype)?
                }
            };
            let llama = Llama::load(vb.pp("language_model"), &llama_config)?;
            (clip, image_newline, llama)
        } else {
            let clip = ClipVisionTower::new(
                vb.pp("model.vision_tower.vision_tower.vision_model"),
                config.mm_vision_select_layer,
                &config.mm_vision_select_feature,
                &clip_vision_config,
            )?;
            let target_dtype = match config.torch_dtype.as_str() { "bfloat16" => DType::BF16, "float16" => DType::F16, _ => DType::F32 };
            let image_newline = match vb.get(&[config.hidden_size], "model.image_newline") {
                Ok(t) => {
                    log::info!("[llava.load] loaded model.image_newline tensor");
                    t.to_device(&device)?.to_dtype(target_dtype)?
                }
                Err(e) => {
                    log::info!(
                        "[llava.load] WARNING missing model.image_newline tensor ({}). Creating fallback vector (random normal).",
                        e
                    );
                    Tensor::randn(0f32, 0.02f32, &[config.hidden_size], &device)?.to_dtype(target_dtype)?
                }
            };
            let llama = Llama::load(vb, &llama_config)?;
            (clip, image_newline, llama)
        };
        
        // CPU may not support matmul in BF16/F16 for all kernels: upcast critical tensors if needed.
        let need_upcast = matches!(device, candle_core::Device::Cpu)
            && matches!(image_newline.dtype(), DType::BF16 | DType::F16);
        let image_newline = if need_upcast {
            log::info!("[llava.load] upcasting image_newline {:?} -> F32 for CPU matmul", image_newline.dtype());
            image_newline.to_dtype(DType::F32)?
        } else { image_newline };

        Ok(Self {
            clip_vision_tower,
            image_newline,
            mm_projector,
            llama,
            config: (*config).clone(),
            device,
        })
    }

    pub fn encode_images(&self, x: &Tensor) -> Result<Tensor> {
        log::info!("self.clip_vision_tower.forward(x)");
        let image_features = self.clip_vision_tower.forward(x)?;
        log::info!("self.mm_projector.forward(&image_features)");
        let image_features = self.mm_projector.forward(&image_features)?;
        let target_dtype = match self.config.torch_dtype.as_str() {
            "bfloat16" => DType::BF16,
            "float16" => DType::F16,
            _ => DType::F32,
        };
        let image_features = if image_features.dtype() != target_dtype {
            image_features.to_dtype(target_dtype)?
        } else {
            image_features
        };
        log::info!("target_dtype: {target_dtype:?}");
        Ok(image_features)
    }

    // currently only for single image, 4 dim tensor
    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        input_ids: &Tensor,
        images: &[Tensor],
        image_sizes: &[(u32, u32)],
    ) -> Result<Tensor> {
        log::info!("Torch DType: {:?}", self.config.torch_dtype);
        let target_dtype = match self.config.torch_dtype.as_str() {
            "bfloat16" => DType::BF16,
            "float16" => DType::F16,
            _ => DType::F32,
        };
        log::info!("TARGET Torch DType: {:?}", target_dtype);
        //TODO: process of multiple images/ new line
        // 576: 336(input size)/14(patch size)=24 24*24+1(class)=577 577-1=576
        let concat_images = Tensor::cat(images, 0)?;
        log::info!("concat_images DType: {:?}", concat_images.dtype());
        let image_features_together = self.encode_images(&concat_images)?;
        log::info!(
            "[prepare_inputs] image_features_together dtype={:?}",
            image_features_together.dtype()
        );
        let split_sizes = images
            .iter()
            .map(|x| x.shape().dims()[0])
            .collect::<Vec<usize>>();
        // can be replaced by split
        let mut index_pos = 0;
        let mut image_features = Vec::new();
        for split_size in split_sizes.iter() {
            image_features.push(image_features_together.i(index_pos..index_pos + (*split_size))?);
            index_pos += *split_size;
        }
        let mm_patch_merge_type = &self.config.mm_patch_merge_type;
        let image_aspect_ratio = &self.config.image_aspect_ratio;

        let image_features = if mm_patch_merge_type == "flat" {
            image_features
                .iter()
                .map(|x| x.flatten(0, 1).unwrap())
                .collect::<Vec<Tensor>>()
        } else if mm_patch_merge_type.starts_with("spatial") {
            let mut new_image_features = Vec::new();
            for (image_idx, image_feature) in image_features.iter().enumerate() {
                let new_image_feature = if image_feature.dims()[0] > 1 {
                    let base_image_feature = image_feature.get(0).unwrap();
                    let patch_image_feature = image_feature.i(1..).unwrap();
                    let height = self.clip_vision_tower.num_patches_per_side();
                    let width = height;
                    assert_eq!(height * width, base_image_feature.dims()[0]);
                    let image_size = image_sizes[image_idx];
                    let new_image_feature = if image_aspect_ratio == "anyres" {
                        let (num_patch_width, num_patch_height) = get_anyres_image_grid_shape(
                            image_size,
                            &self.config.image_grid_pinpoints,
                            self.clip_vision_tower.config.image_size as u32,
                        );
                        patch_image_feature.reshape((
                            num_patch_height as usize,
                            num_patch_width as usize,
                            height,
                            width,
                            (),
                        ))?
                    } else {
                        todo!("not implemented in original python LLaVA yet")
                    };
                    let new_image_feature = if mm_patch_merge_type.contains("unpad") {
                        let new_image_feature = new_image_feature
                            .permute((4, 0, 2, 1, 3))?
                            .flatten(1, 2)?
                            .flatten(2, 3)?;
                        let new_image_feature = unpad_image(&new_image_feature, &image_size)?;
                        let new_image_feature_dims = new_image_feature.dims();
                        let mut image_new_line = self
                            .image_newline
                            .reshape((self.config.hidden_size, 1, 1))?
                            .broadcast_as((
                                new_image_feature_dims[0],
                                new_image_feature_dims[1],
                                1,
                            ))?;
                        let mut new_image_feature = new_image_feature;
                        if new_image_feature.dtype() != target_dtype {
                            log::info!(
                                "[prepare_inputs][vision-unpad] casting patch tensor {:?} -> {:?}",
                                new_image_feature.dtype(), target_dtype
                            );
                            new_image_feature = new_image_feature.to_dtype(target_dtype)?;
                        }
                        if image_new_line.dtype() != target_dtype {
                            log::info!(
                                "[prepare_inputs][vision-unpad] casting image_new_line {:?} -> {:?}",
                                image_new_line.dtype(), target_dtype
                            );
                            image_new_line = image_new_line.to_dtype(target_dtype)?;
                        }
                        log::info!(
                            "[prepare_inputs][vision-unpad] cat dtypes feature={:?} newline={:?}",
                            new_image_feature.dtype(), image_new_line.dtype()
                        );
                        let new_image_feature = Tensor::cat(&[new_image_feature, image_new_line], 2)?;
                        new_image_feature.flatten(1, 2)?.transpose(0, 1)?
                    } else {
                        new_image_feature.permute((0, 2, 1, 3, 4))?.flatten(0, 3)?
                    };
                    let mut base_image_feature = base_image_feature;
                    let mut new_image_feature = new_image_feature;
                    if base_image_feature.dtype() != target_dtype {
                        log::info!(
                            "[prepare_inputs][vision] casting base feature {:?} -> {:?}",
                            base_image_feature.dtype(), target_dtype
                        );
                        base_image_feature = base_image_feature.to_dtype(target_dtype)?;
                    }
                    if new_image_feature.dtype() != target_dtype {
                        log::info!(
                            "[prepare_inputs][vision] casting new feature {:?} -> {:?}",
                            new_image_feature.dtype(), target_dtype
                        );
                        new_image_feature = new_image_feature.to_dtype(target_dtype)?;
                    }
                    log::info!(
                        "[prepare_inputs][vision] cat base/new dtypes base={:?} new={:?}",
                        base_image_feature.dtype(), new_image_feature.dtype()
                    );
                    Tensor::cat(&[base_image_feature, new_image_feature], 0)?
                } else {
                    let mut new_image_feature = image_feature.get(0).unwrap();
                    if mm_patch_merge_type.contains("unpad") {
                        let mut newline = self.image_newline.clone().unsqueeze(0).unwrap();
                        if new_image_feature.dtype() != target_dtype {
                            log::info!(
                                "[prepare_inputs][vision-single] casting feature {:?} -> {:?}",
                                new_image_feature.dtype(), target_dtype
                            );
                            new_image_feature = new_image_feature.to_dtype(target_dtype)?;
                        }
                        if newline.dtype() != target_dtype {
                            newline = newline.to_dtype(target_dtype)?;
                        }
                        log::info!(
                            "[prepare_inputs][vision-single] cat dtypes feature={:?} newline={:?}",
                            new_image_feature.dtype(), newline.dtype()
                        );
                        Tensor::cat(&[new_image_feature, newline], 0).unwrap()
                    } else {
                        new_image_feature
                    }
                };
                new_image_features.push(new_image_feature);
            }
            new_image_features
        } else {
            bail!("Unexpected mm_patch_merge_type: {mm_patch_merge_type}")
        };
        // DType harmonization (redundant safeguard).
        let image_features: Vec<Tensor> = image_features
            .into_iter()
            .map(|t| if t.dtype() != target_dtype { t.to_dtype(target_dtype).unwrap() } else { t })
            .collect();
        #[allow(unused)]
        {
            let mut counts = std::collections::HashMap::new();
            for t in &image_features { *counts.entry(format!("{:?}", t.dtype())).or_insert(0usize) += 1; }
            log::info!("[prepare_inputs] image feature dtypes: {:?}", counts);
        }
        // can easily be replaced by nonzero if it is implemented in candle
        let input_ids_vec = input_ids.squeeze(0)?.to_vec1::<i64>()?;
        let mut image_indices = {
            let mut image_indices = vec![0_i64];
            image_indices.extend(
                input_ids_vec
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if *x == self.config.image_token_index as i64 {
                            Some(i as i64)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<i64>>(),
            );
            image_indices
        };
        if image_indices.len() == 1 {
            //no image, only [0],
            return self.llama.embed(input_ids);
        }

        let input_ids_noim = input_ids_vec
            .iter()
            .filter_map(|x| {
                if *x != self.config.image_token_index as i64 {
                    Some(*x)
                } else {
                    None
                }
            })
            .collect::<Vec<i64>>();
        let input_ids_noim_len = input_ids_noim.len();
        image_indices.push((input_ids_noim_len) as i64);
        let input_ids_noim = Tensor::from_vec(input_ids_noim, input_ids_noim_len, &self.device)?;
    let cur_input_embeds = self.llama.embed(&input_ids_noim)?;
        // Ensure embedding dtype matches model target dtype for later concatenations.
        let target_dtype = match self.config.torch_dtype.as_str() {
            "bfloat16" => DType::BF16,
            "float16" => DType::F16,
            _ => DType::F32,
        };
        let cur_input_embeds = if cur_input_embeds.dtype() != target_dtype {
            cur_input_embeds.to_dtype(target_dtype)?
        } else {
            cur_input_embeds
        };
        // can be replace by split if it is implemented in candle
        let input_embed_no_ims = {
            let mut input_embeds = Vec::new();
            for i in 0..image_indices.len() - 1 {
                let start = (image_indices[i]) as usize;
                let end = image_indices[i + 1] as usize;
                input_embeds.push(cur_input_embeds.i((start..end, ..))?)
            }
            input_embeds
        };

        let mut cur_new_input_embeds = Vec::new();
        for (i, image_feature) in image_features.iter().enumerate() {
            cur_new_input_embeds.push(input_embed_no_ims[i].clone());
            cur_new_input_embeds.push(image_feature.clone());
        }
        cur_new_input_embeds.push(input_embed_no_ims[image_features.len()].clone());
        // Harmonize dtype for all pieces before concatenation.
        let cur_new_input_embeds: Vec<Tensor> = cur_new_input_embeds
            .into_iter()
            .map(|t| if t.dtype() != target_dtype { t.to_dtype(target_dtype).unwrap() } else { t })
            .collect();
        #[allow(unused)]
        {
            let mut counts = std::collections::HashMap::new();
            for t in &cur_new_input_embeds { *counts.entry(format!("{:?}", t.dtype())).or_insert(0usize) += 1; }
            log::info!("[prepare_inputs] concat segment dtypes: {:?}", counts);
        }
        let new_input_embeds = Tensor::cat(&cur_new_input_embeds, 0)?;
        let new_input_embeds = if new_input_embeds.dtype() != target_dtype {
            new_input_embeds.to_dtype(target_dtype)?
        } else { new_input_embeds };
        //trancate
        let new_input_embeds =
            if let Some(tokenizer_model_max_length) = self.config.tokenizer_model_max_length {
                let (new_input_embeds_length, _) = new_input_embeds.shape().dims2()?;
                if new_input_embeds_length > tokenizer_model_max_length {
                    new_input_embeds.i((..tokenizer_model_max_length, ..))?
                } else {
                    new_input_embeds
                }
            } else {
                new_input_embeds
            };
        new_input_embeds.unsqueeze(0)
    }

    pub fn forward(
        &self,
        input_embeds: &Tensor,
        position_id: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.llama
            .forward_input_embed(input_embeds, position_id, cache)
    }
}
