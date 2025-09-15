use anyhow::Result;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub fn pick_device_cuda0_or_cpu() -> Device {
    candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
}

pub fn hf_model(hf_repo: &str) -> Result<hf_hub::api::sync::ApiRepo> {
    let api = Api::new()?;
    Ok(api.model(hf_repo.to_string()))
}

pub fn hf_get_file(repo: &hf_hub::api::sync::ApiRepo, filename: &str) -> Result<std::path::PathBuf> {
    Ok(repo.get(filename)?)
}

pub fn load_tokenizer(hf_repo: &str, override_path: Option<&str>) -> Result<Tokenizer> {
    let path = if let Some(p) = override_path { std::path::PathBuf::from(p) } else { hf_model(hf_repo)?.get("tokenizer.json")? };
    Ok(Tokenizer::from_file(path).map_err(anyhow::Error::msg)?)
}

pub unsafe fn with_mmap_varbuilder_single<R, F>(model_file: &std::path::PathBuf, dtype: DType, device: &Device, f: F) -> Result<R>
where
    F: FnOnce(VarBuilder) -> Result<R>,
{
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(std::slice::from_ref(model_file), dtype, device)? };
    f(vb)
}

pub unsafe fn with_mmap_varbuilder_multi<R, F>(files: &[std::path::PathBuf], dtype: DType, device: &Device, f: F) -> Result<R>
where
    F: FnOnce(VarBuilder) -> Result<R>,
{
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
    f(vb)
}
