//! AI models section: the vision folder, CLIP and reranker choices, loading and freeing models.

use std::path::Path;

use eframe::egui::{ComboBox, DragValue, Ui};

use super::draft::Draft;
use super::engines::{DEFAULT_CLIP, EngineAction, Keep};
use super::sections::name;
use super::widgets::{Focus, caption, group, path_field, toggle};
use crate::command::{AppCommand, Target, Tool};
use crate::status::DeviceKind;
use crate::ui::dialogs;

const SLOT_VISION: &str = "settings-vision-folder";

/// The CLIP and SigLIP models the engine can load: key and caption.
pub const CLIP_MODELS: [(&str, &str); 10] = [
    ("unicom-vit-b32", "Unicom ViT-B/32 (fastembed)"),
    ("clip-vit-b32", "OpenAI CLIP ViT-B/32 (fastembed)"),
    ("siglip-base-patch16-224", "SigLIP v1 base 224 (experimental)"),
    ("siglip2-base-patch16-224", "SigLIP v2 base 224 (experimental)"),
    ("siglip2-base-patch16-256", "SigLIP v2 base 256 (experimental)"),
    ("siglip2-base-patch16-384", "SigLIP v2 base 384 (experimental)"),
    ("siglip2-base-patch16-512", "SigLIP v2 base 512 (experimental)"),
    ("siglip2-large-patch16-256", "SigLIP v2 large 256 (experimental)"),
    ("siglip2-large-patch16-384", "SigLIP v2 large 384 (experimental)"),
    ("siglip2-large-patch16-512", "SigLIP v2 large 512 (experimental)"),
];

/// Repository the reranker uses when none is set.
pub const DEFAULT_RERANKER: &str = "jinaai/jina-reranker-m0";

/// What a folder looks like as a model folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FolderCheck {
    Missing,
    NotAFolder,
    /// No weights found.
    Empty,
    /// A config and this many weight files.
    Looks(usize),
}

impl FolderCheck {
    pub fn text(&self) -> String {
        match self {
            Self::Missing => "The folder does not exist".to_string(),
            Self::NotAFolder => "This is a file, not a folder".to_string(),
            Self::Empty => "No model files (config.json and .safetensors or .gguf) found in this folder".to_string(),
            Self::Looks(n) => format!("Looks like a model folder ({n} weight files)"),
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Looks(_))
    }
}

/// Looks for a config file and weight files in `dir`.
pub fn check_model_folder(dir: &Path) -> FolderCheck {
    if !dir.exists() {
        return FolderCheck::Missing;
    }
    if !dir.is_dir() {
        return FolderCheck::NotAFolder;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return FolderCheck::Empty };
    let (mut config, mut weights) = (false, 0usize);
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().to_lowercase();
        config |= file == "config.json";
        weights += usize::from(file.ends_with(".safetensors") || file.ends_with(".gguf"));
    }
    if config && weights > 0 { FolderCheck::Looks(weights) } else { FolderCheck::Empty }
}

/// What the model section asked for this frame.
#[derive(Default)]
pub struct Requests {
    pub engine: Option<EngineAction>,
}

/// State of the AI models section.
#[derive(Default)]
pub struct Models {
    checked: Option<(String, FolderCheck)>,
}

impl Models {
    fn folder_check(&mut self, text: &str) -> Option<&FolderCheck> {
        if text.trim().is_empty() {
            return None;
        }
        if self.checked.as_ref().map(|(t, _)| t.as_str() != text).unwrap_or(true) {
            self.checked = Some((text.to_string(), check_model_folder(Path::new(text.trim()))));
        }
        self.checked.as_ref().map(|(_, c)| c)
    }

    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, draft: &mut Draft) -> Requests {
        let mut requests = Requests::default();
        if let Some(dir) = dialogs::take_picked(SLOT_VISION) {
            draft.value.joycaption_model_dir = Some(dir.display().to_string());
        }
        let s = &mut draft.value;

        group(ui, "Vision");
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::VISION_DIR);
            let mut text = s.joycaption_model_dir.clone().unwrap_or_default();
            let response = path_field(ui, &mut text, "Folder with the vision model", "Choose the vision model folder", SLOT_VISION, true);
            if response.changed() {
                s.joycaption_model_dir = if text.trim().is_empty() { None } else { Some(text.trim().to_string()) };
            }
        });
        match self.folder_check(s.joycaption_model_dir.as_deref().unwrap_or("")).cloned() {
            Some(check) if check.is_ok() => {
                ui.label(check.text());
            }
            Some(check) => {
                ui.colored_label(ui.visuals().warn_fg_color, check.text());
            }
            None => {
                ui.weak("No folder chosen. Describing files needs one.");
            }
        }
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::DEVICE);
            ui.label(device_text());
        });

        group(ui, "CLIP and SigLIP");
        ui.horizontal(|ui| {
            caption(ui, focus, name::CLIP_MODEL);
            let mut selected = s.clip_model.clone().unwrap_or_else(|| DEFAULT_CLIP.to_string());
            let shown = CLIP_MODELS.iter().find(|(key, _)| *key == selected).map(|(_, label)| *label).unwrap_or(selected.as_str()).to_string();
            ComboBox::from_id_salt("settings-clip-model").selected_text(shown).show_ui(ui, |ui| {
                for (i, (key, label)) in CLIP_MODELS.iter().enumerate() {
                    if i == 2 {
                        ui.separator();
                    }
                    ui.selectable_value(&mut selected, key.to_string(), *label);
                }
            });
            if Some(&selected) != s.clip_model.as_ref() {
                s.clip_model = Some(selected);
            }
        });
        toggle(ui, focus, name::CLIP_BLEND, "Embed the image together with its text metadata", &mut s.clip_augment_with_text);
        toggle(ui, focus, name::CLIP_OVERWRITE, "Automatic generation replaces embeddings that exist already", &mut s.clip_overwrite_embeddings);

        group(ui, "Reranker");
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::RERANKER);
            let mut text = s.reranker_model.clone().unwrap_or_else(|| DEFAULT_RERANKER.to_string());
            if ui.text_edit_singleline(&mut text).changed() {
                s.reranker_model = if text.trim().is_empty() { None } else { Some(text.trim().to_string()) };
            }
            if ui.button("Reset").on_hover_text(format!("Back to {DEFAULT_RERANKER}")).clicked() {
                s.reranker_model = Some(DEFAULT_RERANKER.to_string());
            }
        });
        let jina = ui.collapsing(name::JINA, |ui| {
            ui.horizontal(|ui| {
                ui.label("Max tokens (query and document)");
                let mut tokens = s.jina_max_length.unwrap_or(1024);
                if ui.add(DragValue::new(&mut tokens).range(128..=8192)).changed() {
                    s.jina_max_length = Some(tokens);
                }
            });
            for (label, id, slot) in [("Query type", "settings-jina-query", 0), ("Document type", "settings-jina-doc", 1)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    let target = if slot == 0 { &mut s.jina_query_type } else { &mut s.jina_doc_type };
                    let mut kind = target.clone().unwrap_or_else(|| "text".to_string());
                    ComboBox::from_id_salt(id).selected_text(kind.as_str()).show_ui(ui, |ui| {
                        ui.selectable_value(&mut kind, "text".to_string(), "text");
                        ui.selectable_value(&mut kind, "image".to_string(), "image");
                    });
                    if Some(&kind) != target.as_ref() {
                        *target = Some(kind);
                    }
                });
            }
            let mut multimodal = s.jina_enable_multimodal.unwrap_or(false);
            if ui.checkbox(&mut multimodal, "Enable multimodal reranking").changed() {
                s.jina_enable_multimodal = Some(multimodal);
            }
        });
        focus.mark(ui, name::JINA, &jina.header_response);

        group(ui, name::IMAGE_EDIT);
        ui.horizontal_wrapped(|ui| {
            let response = ui.label("Repository, GGUF file and precision are chosen in the Image Edit tool.");
            focus.mark(ui, name::IMAGE_EDIT, &response);
            if ui.button("Open Image Edit").clicked() {
                crate::command::send(AppCommand::OpenTool { tool: Tool::ImageEdit, path: None, target: Target::Current });
            }
        });

        group(ui, name::LOAD_NOW);
        ui.horizontal_wrapped(|ui| {
            ui.label("Applies pending changes first.");
            for (label, action) in [
                ("Load vision", EngineAction::LoadVision),
                ("Reload vision", EngineAction::ReloadVision),
                ("Reload CLIP", EngineAction::ReloadClip),
                ("Reload reranker", EngineAction::ReloadReranker),
                ("Generate missing CLIP embeddings", EngineAction::BackfillClip),
            ] {
                if ui.button(label).clicked() {
                    requests.engine = Some(action);
                }
            }
        });
        group(ui, name::MEMORY);
        ui.horizontal_wrapped(|ui| {
            let response = ui.label("Free memory:");
            focus.mark(ui, name::MEMORY, &response);
            for keep in [Keep::Clip, Keep::Vision, Keep::Reranker] {
                if ui.button(keep.label()).on_hover_text("Unloads the other models now; they load again when used").clicked() {
                    requests.engine = Some(EngineAction::Unload(Some(keep)));
                }
            }
            if ui.button("Unload all").clicked() {
                requests.engine = Some(EngineAction::Unload(None));
            }
        });
        requests
    }
}

fn device_text() -> String {
    let status = crate::status::snapshot("VISION");
    match status.and_then(|s| s.device) {
        Some(DeviceKind::GPU) => "GPU (CUDA)".to_string(),
        Some(DeviceKind::CPU) => "CPU".to_string(),
        None => "Chosen when a model loads: CUDA when available, otherwise CPU".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_folder_needs_a_config_and_weights() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(check_model_folder(&dir.path().join("nope")), FolderCheck::Missing);
        assert_eq!(check_model_folder(dir.path()), FolderCheck::Empty);
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert_eq!(check_model_folder(dir.path()), FolderCheck::Empty, "a config alone is not a model");
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), "x").unwrap();
        std::fs::write(dir.path().join("model-00002-of-00002.SAFETENSORS"), "x").unwrap();
        assert_eq!(check_model_folder(dir.path()), FolderCheck::Looks(2));
        assert!(check_model_folder(dir.path()).is_ok());
        let file = dir.path().join("config.json");
        assert_eq!(check_model_folder(&file), FolderCheck::NotAFolder);
    }

    #[test]
    fn gguf_weights_count_too() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("q4.gguf"), "x").unwrap();
        assert_eq!(check_model_folder(dir.path()), FolderCheck::Looks(1));
    }

    #[test]
    fn the_clip_list_has_the_default_and_unique_keys() {
        assert!(CLIP_MODELS.iter().any(|(k, _)| *k == DEFAULT_CLIP));
        let keys: std::collections::BTreeSet<_> = CLIP_MODELS.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys.len(), CLIP_MODELS.len());
    }

    #[test]
    fn the_folder_check_is_cached_per_text() {
        let mut models = Models::default();
        assert!(models.folder_check("  ").is_none());
        assert_eq!(models.folder_check("/does/not/exist"), Some(&FolderCheck::Missing));
        assert_eq!(models.checked.as_ref().map(|(t, _)| t.as_str()), Some("/does/not/exist"));
    }
}
