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
    pub parent_dir: String,
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
    pub similarity_score: Option<f32>,
    pub clip_similarity_score: Option<f32>,
}
