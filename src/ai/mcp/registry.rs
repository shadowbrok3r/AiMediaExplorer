//! The list of tools, their schemas and the one function that runs a call with a time limit and a log entry.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmcp::model::{Tool, ToolAnnotations};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::activity::{ActivityEntry, ActivityLog, ServerStats, args_summary};
use super::ctx::{CallInfo, ToolCtx};
use super::error::{ToolError, ToolOutput};

/// Largest serialized result a tool may return.
pub const MAX_RESULT_BYTES: usize = 512 * 1024;

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send>>;
pub type ToolFn = Arc<dyn Fn(Arc<ToolCtx>, CallInfo, Value) -> ToolFuture + Send + Sync>;

/// What a tool does to the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolClass {
    /// Reads the library.
    Read,
    /// Asks the UI to show something.
    Ui,
    /// Runs a local model through the GPU queue.
    Gpu,
    /// Stores a proposal for the user to review.
    Propose,
}

impl ToolClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Ui => "ui",
            Self::Gpu => "gpu",
            Self::Propose => "propose",
        }
    }
}

#[derive(Clone)]
pub struct ToolEntry {
    pub name: &'static str,
    pub class: ToolClass,
    pub timeout: Duration,
    pub tool: Tool,
    run: ToolFn,
}

/// A tool whose arguments are the type `A`; its schema comes from `A`.
pub fn entry<A, Fut>(
    name: &'static str,
    class: ToolClass,
    timeout: Duration,
    title: &'static str,
    description: &'static str,
    handler: fn(Arc<ToolCtx>, CallInfo, A) -> Fut,
) -> ToolEntry
where
    A: DeserializeOwned + rmcp::schemars::JsonSchema + 'static,
    Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
{
    let annotations = ToolAnnotations::with_title(title)
        .read_only(matches!(class, ToolClass::Read))
        .destructive(false)
        .idempotent(matches!(class, ToolClass::Read))
        .open_world(false);
    let mut tool = Tool::new(name, description, Arc::new(serde_json::Map::new())).with_input_schema::<A>().with_title(title).annotate(annotations);
    Arc::make_mut(&mut tool.input_schema).entry("properties").or_insert_with(|| Value::Object(Default::default()));
    let run: ToolFn = Arc::new(move |ctx, call, args| {
        let parsed = serde_json::from_value::<A>(args).map_err(|e| ToolError::Invalid(format!("bad arguments: {e}")));
        match parsed {
            Ok(args) => Box::pin(handler(ctx, call, args)),
            Err(e) => Box::pin(std::future::ready(Err(e))),
        }
    });
    ToolEntry { name, class, timeout, tool, run }
}

/// One tool as shown to people: name, class, text and schema.
#[derive(Clone, Debug, Serialize)]
pub struct CatalogRow {
    pub name: String,
    pub class: ToolClass,
    pub title: String,
    pub description: String,
    pub timeout_secs: u64,
    pub input_schema: Value,
}

pub struct ToolRegistry {
    entries: Vec<ToolEntry>,
    activity: Arc<ActivityLog>,
    stats: Arc<ServerStats>,
}

impl ToolRegistry {
    pub fn new(entries: Vec<ToolEntry>, activity: Arc<ActivityLog>, stats: Arc<ServerStats>) -> Self {
        Self { entries, activity, stats }
    }

    pub fn get(&self, name: &str) -> Option<&ToolEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.entries.iter().map(|e| e.tool.clone()).collect()
    }

    pub fn catalog(&self) -> Vec<CatalogRow> {
        self.entries
            .iter()
            .map(|e| CatalogRow {
                name: e.name.to_string(),
                class: e.class,
                title: e.tool.title.clone().unwrap_or_default(),
                description: e.tool.description.as_ref().map(|d| d.to_string()).unwrap_or_default(),
                timeout_secs: e.timeout.as_secs(),
                input_schema: Value::Object(e.tool.input_schema.as_ref().clone()),
            })
            .collect()
    }

    pub fn activity(&self) -> &Arc<ActivityLog> {
        &self.activity
    }

    /// Runs the tool `name`; every call ends with an activity entry, whatever happened.
    pub async fn call(&self, ctx: Arc<ToolCtx>, call: CallInfo, name: &str, args: Value) -> Result<ToolOutput, ToolError> {
        let started = Instant::now();
        let args = if args.is_null() { Value::Object(Default::default()) } else { args };
        let summary = args_summary(&args);
        let client = call.client.clone();
        ServerStats::bump(&self.stats.tool_calls);
        let clock = ctx.clock.clone();
        let outcome = self.run(ctx, call, name, args).await;
        let bytes = outcome.as_ref().map(|o| o.text.len() + o.data.to_string().len()).unwrap_or(0);
        let error = outcome.as_ref().err().map(|e| e.kind().to_string());
        self.activity.push(ActivityEntry {
            at_ms: clock.unix_ms(),
            client,
            tool: name.to_string(),
            args: summary,
            ok: outcome.is_ok(),
            error_kind: error,
            duration_ms: started.elapsed().as_millis() as u64,
            result_bytes: bytes,
        });
        outcome
    }

    async fn run(&self, ctx: Arc<ToolCtx>, call: CallInfo, name: &str, args: Value) -> Result<ToolOutput, ToolError> {
        let entry = self.get(name).ok_or_else(|| ToolError::NotFound(format!("unknown tool '{name}'; list the tools first")))?;
        let cancel = call.cancel.clone();
        let work = (entry.run)(ctx, call, args);
        let output = tokio::select! {
            r = tokio::time::timeout(entry.timeout, work) => r.map_err(|_| ToolError::Timeout(format!("{name} took longer than {}s", entry.timeout.as_secs())))??,
            _ = cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        let size = output.text.len() + output.data.to_string().len() + output.images.iter().map(|i| i.data_base64.len()).sum::<usize>();
        if size > MAX_RESULT_BYTES {
            return Err(ToolError::failed(format!("the result is {} KiB, over the {} KiB limit; ask for fewer items", size / 1024, MAX_RESULT_BYTES / 1024)));
        }
        Ok(output)
    }
}
