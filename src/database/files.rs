use surrealdb::RecordId;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Thumbnail {
    pub id: Option<RecordId>,
    pub db_created: Option<surrealdb::sql::Datetime>,
    pub path: String,
    pub filename: String,
    pub file_type: String,
    pub size: u64,
    pub description: Option<String>,
    pub caption: Option<String>,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub thumbnail_b64: Option<String>,
    pub modified: Option<surrealdb::sql::Datetime>,
    pub hash: Option<String>,
    // In-memory fields used for UI/AI operations (not always persisted)
    // Base64 data URL of thumbnail for quick UI rendering
    pub thumb_b64: Option<String>,
    // For search ranking in-memory
    pub similarity_score: Option<f32>,
    // CLIP embedding (in-memory only; persisted separately in clip_embeddings)
    pub clip_embedding: Option<Vec<f32>>,
    pub clip_similarity_score: Option<f32>,
}

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
    // Convenience fields for in-memory/UI (optional) and scoring
    pub similarity_score: Option<f32>,
    pub clip_embedding: Option<Vec<f32>>, // duplicate of embedding for API parity
    pub clip_similarity_score: Option<f32>,
}
