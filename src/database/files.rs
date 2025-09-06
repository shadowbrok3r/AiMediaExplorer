use surrealdb::RecordId;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileMetadata {
    pub id: Option<String>,
    pub path: String,
    pub filename: String,
    pub file_type: String,
    pub size: u64,
    pub modified: Option<chrono::DateTime<chrono::Local>>,
    pub created: Option<chrono::DateTime<chrono::Local>>,
    pub thumbnail_path: Option<String>,
    // In-memory/base64 thumbnail data (data URL) when available. This lets AI search results
    // render thumbnails even if the underlying FoundFile list isn't currently visible.
    pub thumb_b64: Option<String>,
    // BLAKE3 hex hash of file contents to detect if content changed and re-embedding is needed.
    pub hash: Option<String>,
    // AI-powered metadata
    pub description: Option<String>,   // AI-generated description (multi-sentence)
    pub caption: Option<String>,       // Short caption/alt text
    pub tags: Vec<String>,             // AI-extracted tags
    pub category: Option<String>,      // Single high-level AI category
    pub embedding: Option<Vec<f32>>,   // AI embedding vector
    pub similarity_score: Option<f32>, // For search ranking
    // Dedicated CLIP embedding (fastembed) kept separate from legacy semantic `embedding` above.
    pub clip_embedding: Option<Vec<f32>>,
    pub clip_similarity_score: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Thumbnail {
    pub id: Option<RecordId>,
    pub db_created: surrealdb::sql::Datetime,
    pub path: String,
    pub filename: String,
    pub file_type: String,
    pub size: u64,
    pub description: Option<String>,
    pub caption: Option<String>,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub thumbnail_b64: Option<String>,
    pub modified: Option<surrealdb::sql::Datetime>,
    pub hash: Option<String>,
}

// Lightweight projection for semantic document debug (id + first chars)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DebugDocumentSnippet {
    pub id: String,
    pub title: Option<String>,
    pub preview: String,
    pub len: usize,
}

// Separate table row for CLIP (and future model-specific) embeddings.
// This avoids bloating the core thumbnails row and lets us regenerate embeddings independently.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ClipEmbeddingRow {
    pub id: Option<RecordId>,
    pub thumb_ref: Option<RecordId>,
    pub path: String,
    pub hash: Option<String>,
    pub embedding: Vec<f32>,
    pub created: Option<surrealdb::sql::Datetime>,
    pub updated: Option<surrealdb::sql::Datetime>,
}