//! The sections of the Settings page and the index of setting labels the search box matches.

use crate::command::Section;
use crate::icons::{Icon, svg};

/// Sections in the order the list shows them.
pub const ALL: [Section; 9] = [
    Section::General,
    Section::Library,
    Section::AiModels,
    Section::Automation,
    Section::Assistant,
    Section::MetadataTools,
    Section::Shortcuts,
    Section::Appearance,
    Section::Advanced,
];

pub fn label(section: Section) -> &'static str {
    match section {
        Section::General => "General",
        Section::Library => "Library",
        Section::AiModels => "AI models",
        Section::Automation => "Automation",
        Section::Assistant => "Assistant and providers",
        Section::MetadataTools => "Metadata tools",
        Section::Shortcuts => "Shortcuts",
        Section::Appearance => "Appearance",
        Section::Advanced => "Advanced",
    }
}

pub fn icon(section: Section) -> Icon {
    match section {
        Section::General => svg::SLIDERS,
        Section::Library => svg::AME_DATABASE,
        Section::AiModels => svg::AME_MODEL,
        Section::Automation => svg::LIGHTNING,
        Section::Assistant => svg::AME_ASSISTANT,
        Section::MetadataTools => svg::TAG,
        Section::Shortcuts => svg::LIST_BULLETS,
        Section::Appearance => svg::PALETTE,
        Section::Advanced => svg::WRENCH,
    }
}

/// Sections whose body scrolls itself.
pub fn scrolls_itself(section: Section) -> bool {
    section == Section::Appearance
}

/// Labels of the settings; each section draws them as its row captions.
pub mod name {
    pub const RESTORE_TABS: &str = "Restore tabs on launch";
    pub const RESTORE_SCOPE: &str = "Restore the last scope on launch";
    pub const DEFAULT_VIEW: &str = "Default view";
    pub const THUMB_SIZE: &str = "Thumbnail size";
    pub const THUMB_FIT: &str = "Thumbnail fit";
    pub const CAPTIONS: &str = "Show captions";
    pub const BADGES: &str = "Show badges";
    pub const CONFIRMATIONS: &str = "Confirmations";

    pub const DB_LOCATION: &str = "Library location";
    pub const BACKUP: &str = "Backup and restore";
    pub const REDACT_KEYS: &str = "Redact API keys";
    pub const WATCHED: &str = "Watched locations";
    pub const SKIP_ICONS: &str = "Skip likely icons in scans";
    pub const MIN_SIZE: &str = "Smallest file";
    pub const MAX_SIZE: &str = "Largest file";
    pub const EXCLUDED_DIRS: &str = "Excluded directories";
    pub const EXCLUDED_EXTS: &str = "Excluded extensions";
    pub const DATE_RANGE: &str = "Modified date range";
    pub const BATCH: &str = "Scan batch size";
    pub const AUTO_SAVE: &str = "Save scanned files to the library";

    pub const VISION_DIR: &str = "Vision model folder";
    pub const DEVICE: &str = "Device";
    pub const CLIP_MODEL: &str = "CLIP / SigLIP model";
    pub const CLIP_BLEND: &str = "Blend image and text embeddings";
    pub const CLIP_OVERWRITE: &str = "Overwrite existing CLIP embeddings";
    pub const RERANKER: &str = "Reranker repository";
    pub const JINA: &str = "Jina M0 options";
    pub const IMAGE_EDIT: &str = "Image edit model";
    pub const MEMORY: &str = "Memory";
    pub const LOAD_NOW: &str = "Load models now";

    pub const AUTO_INDEX: &str = "Auto indexing";
    pub const AUTO_CLIP: &str = "Auto CLIP embeddings";
    pub const OVERWRITE_DESC: &str = "Overwrite existing descriptions";
    pub const PROMPT: &str = "Prompt template";

    pub const ZEROCLAW: &str = "ZeroClaw";
    pub const PROVIDER: &str = "Provider";
    pub const API_KEY: &str = "API key";
    pub const BASE_URL: &str = "Base URL";
    pub const MODEL: &str = "Model";
    pub const ORGANIZATION: &str = "Organization";
    pub const TEST: &str = "Test connection";
    pub const RECENT_MODELS: &str = "Recent models";

    pub const RENAME_CATEGORY: &str = "Rename or merge a category";
    pub const RENAME_TAG: &str = "Rename or merge a tag";
    pub const DELETE_TAG: &str = "Delete a tag";
    pub const PRUNE_TAGS: &str = "Limit tags per item";

    pub const BINDINGS: &str = "Actions and bindings";

    pub const THEME: &str = "Theme";
    pub const EGUI_OPTIONS: &str = "Egui options";

    pub const LOG_LEVEL: &str = "Log level";
    pub const PERF_HUD: &str = "Performance HUD";
    pub const RAW_JSON: &str = "Raw settings JSON";
    pub const RESET: &str = "Reset settings";
    pub const MCP: &str = "MCP tools server";
}

/// One searchable setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub section: Section,
    pub label: &'static str,
    /// Extra words the search also matches.
    pub words: &'static str,
}

const fn e(section: Section, label: &'static str, words: &'static str) -> Entry {
    Entry { section, label, words }
}

/// Every setting the page draws.
pub const ENTRIES: &[Entry] = &[
    e(Section::General, name::RESTORE_TABS, "startup open"),
    e(Section::General, name::RESTORE_SCOPE, "startup folder"),
    e(Section::General, name::DEFAULT_VIEW, "grid details list"),
    e(Section::General, name::THUMB_SIZE, "tile zoom small large"),
    e(Section::General, name::THUMB_FIT, "crop square"),
    e(Section::General, name::CAPTIONS, "names under tiles"),
    e(Section::General, name::BADGES, "indexed offline"),
    e(Section::General, name::CONFIRMATIONS, "ask warn delete move import restore"),
    e(Section::Library, name::DB_LOCATION, "database folder path move copy choose"),
    e(Section::Library, name::BACKUP, "export import snapshot restore surql"),
    e(Section::Library, name::REDACT_KEYS, "secrets password export"),
    e(Section::Library, name::WATCHED, "folders scan add"),
    e(Section::Library, name::SKIP_ICONS, "assets small"),
    e(Section::Library, name::MIN_SIZE, "minimum bytes kib"),
    e(Section::Library, name::MAX_SIZE, "maximum bytes"),
    e(Section::Library, name::EXCLUDED_DIRS, "skip ignore folders"),
    e(Section::Library, name::EXCLUDED_EXTS, "skip ignore file types"),
    e(Section::Library, name::DATE_RANGE, "after before date"),
    e(Section::Library, name::BATCH, "found batch performance"),
    e(Section::Library, name::AUTO_SAVE, "database library group"),
    e(Section::AiModels, name::VISION_DIR, "joycaption llava model folder"),
    e(Section::AiModels, name::DEVICE, "gpu cpu cuda"),
    e(Section::AiModels, name::CLIP_MODEL, "siglip clip embedding"),
    e(Section::AiModels, name::CLIP_BLEND, "text image"),
    e(Section::AiModels, name::CLIP_OVERWRITE, "embeddings regenerate"),
    e(Section::AiModels, name::RERANKER, "jina huggingface"),
    e(Section::AiModels, name::JINA, "tokens multimodal query document"),
    e(Section::AiModels, name::IMAGE_EDIT, "qwen gguf precision"),
    e(Section::AiModels, name::MEMORY, "unload vram ram free"),
    e(Section::AiModels, name::LOAD_NOW, "reload engine"),
    e(Section::Automation, name::AUTO_INDEX, "describe background"),
    e(Section::Automation, name::AUTO_CLIP, "embeddings background"),
    e(Section::Automation, name::OVERWRITE_DESC, "descriptions replace"),
    e(Section::Automation, name::PROMPT, "template reset instructions"),
    e(Section::Assistant, name::ZEROCLAW, "gateway agent pairing"),
    e(Section::Assistant, name::PROVIDER, "openai grok gemini groq openrouter custom"),
    e(Section::Assistant, name::API_KEY, "token secret"),
    e(Section::Assistant, name::BASE_URL, "endpoint host"),
    e(Section::Assistant, name::MODEL, "default"),
    e(Section::Assistant, name::ORGANIZATION, "openai"),
    e(Section::Assistant, name::TEST, "check connection models"),
    e(Section::Assistant, name::RECENT_MODELS, "history last used"),
    e(Section::MetadataTools, name::RENAME_CATEGORY, "merge category"),
    e(Section::MetadataTools, name::RENAME_TAG, "merge tag"),
    e(Section::MetadataTools, name::DELETE_TAG, "remove tag"),
    e(Section::MetadataTools, name::PRUNE_TAGS, "limit truncate tags"),
    e(Section::Shortcuts, name::BINDINGS, "keyboard keys hotkeys commands palette"),
    e(Section::Appearance, name::THEME, "colors density text size presets"),
    e(Section::Appearance, name::EGUI_OPTIONS, "visuals fonts animation"),
    e(Section::Advanced, name::LOG_LEVEL, "debug trace logging"),
    e(Section::Advanced, name::PERF_HUD, "frame time overlay"),
    e(Section::Advanced, name::RAW_JSON, "settings record ui_prefs"),
    e(Section::Advanced, name::RESET, "defaults clear"),
    e(Section::Advanced, name::MCP, "tools server devtools"),
];

/// True when every word of `query` is in the label, the extra words or the section name.
fn entry_matches(entry: &Entry, words: &[String]) -> bool {
    let hay = format!("{} {} {}", entry.label, entry.words, label(entry.section)).to_lowercase();
    words.iter().all(|w| hay.contains(w.as_str()))
}

fn split_query(query: &str) -> Vec<String> {
    query.split_whitespace().map(str::to_lowercase).collect()
}

/// The settings matching `query`, in page order; an empty query matches nothing.
pub fn search(query: &str) -> Vec<&'static Entry> {
    let words = split_query(query);
    if words.is_empty() {
        return Vec::new();
    }
    ENTRIES.iter().filter(|e| entry_matches(e, &words)).collect()
}

/// Number of settings of `section` matching `query`.
pub fn match_count(section: Section, query: &str) -> usize {
    search(query).iter().filter(|e| e.section == section).count()
}

/// Entries of `section`.
pub fn entries_of(section: Section) -> impl Iterator<Item = &'static Entry> {
    ENTRIES.iter().filter(move |e| e.section == section)
}
