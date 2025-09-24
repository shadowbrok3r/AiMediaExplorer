use rmcp::{
    Json, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

// -------------------- Schemas --------------------

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PingRequest { pub message: String }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PingResponse { pub message: String }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct MediaSearchRequest { pub query: String, pub top_k: Option<u32> }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct MediaSearchHit { pub path: String, pub score: f32 }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct MediaSearchResponse { pub results: Vec<MediaSearchHit> }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct MediaDescribeRequest { pub path: String }

#[derive(Debug, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct MediaDescribeResponse {
    pub description: String,
    pub caption: String,
    pub tags: Vec<String>,
    pub category: String,
}

// -------------------- Server --------------------

#[derive(Clone)]
pub struct MCPServer {
    pub tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl rmcp::ServerHandler for MCPServer {}

#[tool_router(router = tool_router)]
impl MCPServer {
    pub fn new() -> Self { Self { tool_router: Self::tool_router() } }

    /// Echo input for connectivity checks
    #[tool(name = "util.ping", description = "Echo input for connectivity checks")]
    pub async fn ping(&self, params: Parameters<PingRequest>) -> Result<Json<PingResponse>, String> {
        Ok(Json(PingResponse { message: params.0.message }))
    }

    /// Semantic search over media by text; returns similar items (paths and distances)
    #[tool(name = "media.search", description = "Semantic search over media by text")]
    pub async fn media_search(&self, params: Parameters<MediaSearchRequest>) -> Result<Json<MediaSearchResponse>, String> {
        let top_k = params.0.top_k.unwrap_or(32).max(1) as usize;
        // Ensure CLIP engine and embed query
        crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await.map_err(|e| e.to_string())?;
        let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
        let engine = guard.as_mut().ok_or_else(|| "CLIP engine unavailable".to_string())?;
        let q = engine.embed_text(&params.0.query).map_err(|e| e.to_string())?;
        drop(guard);

        let ef = (top_k * 4).clamp(32, 256);
        let hits = crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, top_k, ef)
            .await
            .map_err(|e| e.to_string())?;
        let results = hits
            .into_iter()
            .map(|h| MediaSearchHit { path: h.path, score: h.dist })
            .collect();
        Ok(Json(MediaSearchResponse { results }))
    }

    /// Generate a JSON description for an image and persist it to the database
    #[tool(name = "media.describe", description = "Describe an image and persist metadata")]
    pub async fn media_describe(&self, params: Parameters<MediaDescribeRequest>) -> Result<Json<MediaDescribeResponse>, String> {
        let path = params.0.path;
        crate::ai::GLOBAL_AI_ENGINE
            .ensure_file_metadata_entry(&path)
            .await
            .map_err(|e| e.to_string())?;

        let pb = std::path::PathBuf::from(&path);
        let vd = crate::ai::GLOBAL_AI_ENGINE
            .generate_vision_description(&pb)
            .await
            .ok_or_else(|| "Unable to generate description".to_string())?;

        crate::ai::GLOBAL_AI_ENGINE
            .apply_vision_description(&path, &vd)
            .await
            .map_err(|e| e.to_string())?;

        Ok(Json(MediaDescribeResponse {
            description: vd.description,
            caption: vd.caption,
            tags: vd.tags,
            category: vd.category,
        }))
    }
}

// -------------------- Convenience helpers --------------------

/// Construct a server instance (router available via `.tool_router`).
pub fn new_server() -> MCPServer { MCPServer::new() }

/// List all tool schemas (useful for UI introspection)
pub fn list_tools(server: &MCPServer) -> Vec<serde_json::Value> {
    server
        .tool_router
        .list_all()
        .into_iter()
        .map(|t| serde_json::json!({
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
            "output_schema": t.output_schema
        }))
        .collect()
}

/// Optionally start an MCP stdio server in the background (spawned task)
pub async fn serve_stdio_background() -> anyhow::Result<()> {
    use rmcp::transport::stdio;
    let server = MCPServer::new();
    // Log available tools for quick verification
    let tools = list_tools(&server);
    log::info!(
        "[MCP] starting stdio server with {} tools: {}",
        tools.len(),
        tools.iter().map(|t| t["name"].as_str().unwrap_or("?")).collect::<Vec<_>>().join(", ")
    );
    let service = server.serve(stdio()).await?;
    tokio::spawn(async move { let _ = service.waiting().await; });
    Ok(())
}

// -------------------- UI-friendly helpers --------------------

/// Convenience: Call the ping tool and return the echoed message.
pub async fn ping_tool(message: String) -> Result<String, String> {
    let server = MCPServer::new();
    let params = rmcp::handler::server::wrapper::Parameters(PingRequest { message });
    let res = server.ping(params).await?;
    Ok(res.0.message)
}

/// Convenience: Call the media.search tool and return results.
pub async fn media_search_tool(query: String, top_k: Option<u32>) -> Result<Vec<MediaSearchHit>, String> {
    let server = MCPServer::new();
    let params = rmcp::handler::server::wrapper::Parameters(MediaSearchRequest { query, top_k });
    let res = server.media_search(params).await?;
    Ok(res.0.results)
}

/// Convenience: Call the media.describe tool and return the response.
pub async fn media_describe_tool(path: String) -> Result<MediaDescribeResponse, String> {
    let server = MCPServer::new();
    let params = rmcp::handler::server::wrapper::Parameters(MediaDescribeRequest { path });
    let res = server.media_describe(params).await?;
    Ok(res.0)
}

