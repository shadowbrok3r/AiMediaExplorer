//! Helpers the assistant panel calls for its slash commands and MCP menu.

use serde::{Deserialize, Serialize};

use super::gpu::JobSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaSearchHit {
    pub path: String,
    /// Cosine similarity, higher is closer.
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaDescribeResponse {
    pub description: String,
    pub caption: String,
    pub tags: Vec<String>,
    pub category: String,
}

/// Echoes `message`.
pub async fn ping_tool(message: String) -> Result<String, String> {
    Ok(message)
}

/// CLIP text search over the whole library through the shared GPU queue.
pub async fn media_search_tool(query: String, top_k: Option<u32>) -> Result<Vec<MediaSearchHit>, String> {
    let top_k = top_k.unwrap_or(32).clamp(1, 200) as usize;
    let hub = super::server::hub();
    let ctx = hub.ctx.clone();
    let db = ctx.db.get().map_err(|e| e.message())?;
    let cancel = tokio_util::sync::CancellationToken::new();
    let (vector, _) = ctx.gpu.run(JobSpec::new("clip_text", "assistant search", "assistant"), &cancel, |_| async { ctx.clip.embed_text(&query).await.map_err(|e| format!("{e:#}")) }).await.map_err(|e| e.to_string())?;
    let hits = crate::database::ClipEmbeddingRow::find_similar_paths_in(db, &vector, top_k, (top_k * 2).clamp(64, 1000), 0).await.map_err(|e| e.to_string())?;
    Ok(hits.into_iter().map(|h| MediaSearchHit { score: h.similarity(), path: h.path }).collect())
}

/// Describes an image with the local vision model and saves the result on its library row.
pub async fn media_describe_tool(path: String) -> Result<MediaDescribeResponse, String> {
    let hub = super::server::hub();
    let ctx = hub.ctx.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let (vd, _) = ctx
        .gpu
        .run(JobSpec::new("describe", "assistant describe", "assistant"), &cancel, |_| async {
            crate::ai::GLOBAL_AI_ENGINE.ensure_file_metadata_entry(&path).await.map_err(|e| e.to_string())?;
            let vd = ctx.vision.describe(std::path::Path::new(&path)).await.map_err(|e| format!("{e:#}"))?;
            crate::ai::GLOBAL_AI_ENGINE.apply_vision_description(&path, &vd).await.map_err(|e| e.to_string())?;
            Ok(vd)
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(MediaDescribeResponse { description: vd.description, caption: vd.caption, tags: vd.tags, category: vd.category })
}
