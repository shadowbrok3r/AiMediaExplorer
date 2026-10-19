//! Devtools window for the MCP server: status, token, clients, tools with a tester, GPU queue, proposals and logs.

use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use eframe::egui::{self, Button, CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::ai::mcp::config::McpConfig;
use crate::ai::mcp::proposals::{Counts, Proposal, Status};
use crate::ai::mcp::{Hub, ServerState, auth, hub};

const REFRESH: Duration = Duration::from_millis(500);
const LOG_ROWS: usize = 40;
const CLIENT_WINDOW_MS: u64 = 5 * 60 * 1000;

fn open_id() -> Id {
    Id::unique("ame_devtools_mcp_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// One finished tester call.
#[derive(Clone, Debug, Default)]
struct TesterResult {
    tool: String,
    millis: u128,
    ok: bool,
    text: String,
    data: String,
    images: Vec<(String, Vec<u8>)>,
}

struct Panel {
    ctx: Option<Context>,
    tool: usize,
    args: String,
    args_for: Option<usize>,
    running: Option<CancellationToken>,
    result: Option<TesterResult>,
    show_token: bool,
    confirm_rotate: bool,
    draft: Option<McpConfig>,
    hosts_text: String,
    folders_text: String,
    settings_note: Option<(bool, String)>,
    proposals: Vec<Proposal>,
    proposal_counts: Counts,
    proposals_note: Option<(bool, String)>,
    proposals_error: Option<String>,
    proposals_loading: bool,
    proposals_at: Option<std::time::Instant>,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            ctx: None,
            tool: 0,
            args: "{}".to_string(),
            args_for: None,
            running: None,
            result: None,
            show_token: false,
            confirm_rotate: false,
            draft: None,
            hosts_text: String::new(),
            folders_text: String::new(),
            settings_note: None,
            proposals: Vec::new(),
            proposal_counts: Counts::default(),
            proposals_note: None,
            proposals_error: None,
            proposals_loading: false,
            proposals_at: None,
        }
    }
}

static PANEL: LazyLock<Arc<Mutex<Panel>>> = LazyLock::new(|| Arc::new(Mutex::new(Panel::default())));

fn with<R>(f: impl FnOnce(&mut Panel) -> R) -> R {
    f(&mut PANEL.lock().unwrap_or_else(|e| e.into_inner()))
}

fn repaint() {
    if let Some(ctx) = with(|p| p.ctx.clone()) {
        ctx.request_repaint();
    }
}

/// Entries of a comma or line separated list, trimmed and without blanks.
pub fn parse_list(text: &str) -> Vec<String> {
    text.split([',', '\n']).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// Seconds, minutes or hours since `at_ms`, as short text.
pub fn ago(now_ms: u64, at_ms: u64) -> String {
    let secs = now_ms.saturating_sub(at_ms) / 1000;
    match secs {
        0..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m ago", secs / 60),
        _ => format!("{}h ago", secs / 3600),
    }
}

fn resolve<'a>(root: &'a Value, schema: &'a Value) -> &'a Value {
    match schema.get("$ref").and_then(Value::as_str).and_then(|r| r.strip_prefix("#/")) {
        Some(path) => path.split('/').try_fold(root, |v, key| v.get(key)).unwrap_or(schema),
        None => schema,
    }
}

fn placeholder(root: &Value, schema: &Value) -> Value {
    let schema = resolve(root, schema);
    if let Some(first) = schema.get("enum").and_then(Value::as_array).and_then(|e| e.first()) {
        return first.clone();
    }
    if let Some(first) = schema.get("oneOf").and_then(Value::as_array).and_then(|e| e.first()).and_then(|v| v.get("const")) {
        return first.clone();
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("integer") | Some("number") => json!(0),
        Some("boolean") => json!(false),
        Some("array") => json!([]),
        Some("object") => json!({}),
        _ => json!(""),
    }
}

/// Arguments for a tool with its required fields filled with placeholders.
pub fn skeleton(schema: &Value) -> Value {
    let mut out = serde_json::Map::new();
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for name in required {
            if let Some(prop) = props.get(name) {
                out.insert(name.to_string(), placeholder(schema, prop));
            }
        }
    }
    Value::Object(out)
}

fn refresh_proposals(hub: Arc<Hub>) {
    if with(|p| std::mem::replace(&mut p.proposals_loading, true)) {
        return;
    }
    tokio::spawn(async move {
        let listed = hub.ctx.proposals.list(Some(Status::Pending), None, 0, 50).await;
        let counts = hub.ctx.proposals.counts().await;
        with(|p| {
            match (listed, counts) {
                (Ok((rows, _)), Ok(c)) => {
                    p.proposals = rows;
                    p.proposal_counts = c;
                    p.proposals_error = None;
                }
                (Err(e), _) | (_, Err(e)) => p.proposals_error = Some(e.to_string()),
            }
            p.proposals_loading = false;
            p.proposals_at = Some(std::time::Instant::now());
        });
        repaint();
    });
}

fn decide(hub: Arc<Hub>, id: String, accept: bool) {
    tokio::spawn(async move {
        let outcome = if accept { hub.ctx.proposals.accept(&id).await } else { hub.ctx.proposals.reject(&id).await };
        let note = match outcome {
            Ok(p) => (true, format!("Proposal {} is {}", p.id, p.status.as_str())),
            Err(e) => {
                log::warn!("[MCP] proposal {id} not decided: {e}");
                (false, e.to_string())
            }
        };
        with(|p| {
            p.proposals_note = Some(note);
            p.proposals_at = None;
        });
        repaint();
    });
}

fn run_tool(hub: Arc<Hub>, name: String, args_text: String) {
    let args: Value = match serde_json::from_str(&args_text) {
        Ok(v) => v,
        Err(e) => {
            with(|p| p.result = Some(TesterResult { tool: name, ok: false, text: format!("The arguments are not valid JSON: {e}"), ..Default::default() }));
            return;
        }
    };
    let cancel = CancellationToken::new();
    with(|p| p.running = Some(cancel.clone()));
    tokio::spawn(async move {
        let started = std::time::Instant::now();
        let outcome = hub.call_local(&name, args, cancel).await;
        let millis = started.elapsed().as_millis();
        let result = match outcome {
            Ok(out) => {
                use base64::Engine as _;
                let images = out.images.iter().filter_map(|i| base64::engine::general_purpose::STANDARD.decode(&i.data_base64).ok().map(|b| (i.mime.clone(), b))).collect();
                TesterResult { tool: name, millis, ok: true, text: out.text, data: serde_json::to_string_pretty(&out.data).unwrap_or_default(), images }
            }
            Err(e) => TesterResult { tool: name, millis, ok: false, text: e.to_string(), ..Default::default() },
        };
        with(|p| {
            p.result = Some(result);
            p.running = None;
        });
        repaint();
    });
}

fn state_label(state: &ServerState) -> (String, bool) {
    match state {
        ServerState::Stopped => ("Stopped".to_string(), false),
        ServerState::Starting => ("Starting".to_string(), false),
        ServerState::Running { addr, .. } => (format!("Running on {addr}"), true),
        ServerState::Failed(e) => (format!("Failed: {e}"), false),
    }
}

fn status_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let state = hub.state();
    let (label, running) = state_label(&state);
    ui.horizontal(|ui| {
        let text = RichText::new(&label).strong();
        ui.label(if matches!(state, ServerState::Failed(_)) { text.color(ui.visuals().error_fg_color) } else { text });
        if ui.add_enabled(!running && state != ServerState::Starting, Button::new("Start")).clicked() {
            crate::command::send(crate::command::AppCommand::Mcp { op: crate::ai::mcp::command::McpOp::Start, id: None, tool: None, args: None });
        }
        if ui.add_enabled(running, Button::new("Stop")).clicked() {
            crate::command::send(crate::command::AppCommand::Mcp { op: crate::ai::mcp::command::McpOp::Stop, id: None, tool: None, args: None });
        }
    });
    if let Some(warning) = hub.warning() {
        ui.label(RichText::new(warning).color(ui.visuals().warn_fg_color));
    }
    let stats = hub.stats.snapshot();
    ui.label(format!(
        "{} requests, {} tool calls, {} turned away (auth {}, host {}, rate {}), {} GPU jobs",
        stats.requests,
        stats.tool_calls,
        stats.rejected_auth + stats.rejected_host + stats.rate_limited,
        stats.rejected_auth,
        stats.rejected_host,
        stats.rate_limited,
        hub.ctx.gpu.depth()
    ));
    ui.horizontal(|ui| {
        ui.label("URL");
        ui.monospace(hub.url());
        if ui.button("Copy URL").clicked() {
            ui.ctx().copy_text(hub.url());
        }
        if ui.button("Copy gateway config snippet").on_hover_text("ZeroClaw gateway configuration with a placeholder for the token").clicked() {
            ui.ctx().copy_text(hub.gateway_snippet());
        }
    });
}

fn token_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let (show, confirm) = with(|p| (p.show_token, p.confirm_rotate));
    let token = hub.token();
    ui.horizontal(|ui| {
        ui.label("Token");
        match &token {
            Ok(t) => {
                ui.monospace(if show { t.clone() } else { auth::mask(t) });
                if ui.button(if show { "Hide" } else { "Show" }).clicked() {
                    with(|p| p.show_token = !show);
                }
                if ui.button("Copy token").clicked() {
                    ui.ctx().copy_text(t.clone());
                }
            }
            Err(e) => {
                ui.label(RichText::new(e).color(ui.visuals().error_fg_color));
            }
        }
        if confirm {
            if ui.button("Confirm new token").clicked() {
                match hub.rotate_token() {
                    Ok(_) => with(|p| p.settings_note = Some((true, "New token written; update the clients".to_string()))),
                    Err(e) => with(|p| p.settings_note = Some((false, e))),
                }
                with(|p| p.confirm_rotate = false);
            }
            if ui.button("Keep").clicked() {
                with(|p| p.confirm_rotate = false);
            }
        } else if ui.button("Rotate token").clicked() {
            with(|p| p.confirm_rotate = true);
        }
    });
    if let Some(dir) = hub.dir() {
        ui.label(RichText::new(format!("Stored in {}", auth::token_path(dir).display())).weak());
    }
}

fn settings_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let (mut draft, mut hosts, mut folders, note) = with(|p| {
        if p.draft.is_none() {
            let cfg = hub.config();
            p.hosts_text = cfg.allowed_hosts.join(", ");
            p.folders_text = cfg.watched_folders.join("\n");
            p.draft = Some(cfg);
        }
        (p.draft.clone().unwrap_or_default(), p.hosts_text.clone(), p.folders_text.clone(), p.settings_note.clone())
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut draft.enabled, "Start with the application");
        ui.label("Port");
        ui.add(egui::DragValue::new(&mut draft.port).range(0..=65535));
    });
    ui.checkbox(&mut draft.allow_tailnet, "Allow tailnet access (binds a specific interface; needs allowed hosts)");
    ui.add_enabled_ui(draft.allow_tailnet, |ui| {
        ui.horizontal(|ui| {
            ui.label("Interface address");
            ui.add(TextEdit::singleline(&mut draft.bind_addr).desired_width(140.0).hint_text("100.x.y.z"));
        });
        ui.horizontal(|ui| {
            ui.label("Allowed hosts");
            ui.add(TextEdit::singleline(&mut hosts).desired_width(320.0).hint_text("desk.tail.ts.net, 100.x.y.z"));
        });
    });
    ui.label("Watched folders (one per line); tools may read these besides the library and the Pictures and Videos folders");
    ui.add(TextEdit::multiline(&mut folders).desired_rows(2).desired_width(f32::INFINITY));
    let mut candidate = draft.clone();
    candidate.allowed_hosts = parse_list(&hosts);
    candidate.watched_folders = parse_list(&folders);
    let problem = candidate.binding().err();
    if let Some(p) = &problem {
        ui.label(RichText::new(p.to_string()).color(ui.visuals().error_fg_color));
    }
    let (mut save, mut revert) = (false, false);
    ui.horizontal(|ui| {
        save = ui.add_enabled(problem.is_none(), Button::new("Save settings")).on_hover_text("Saved settings apply the next time the server starts").clicked();
        revert = ui.button("Revert").clicked();
        if let Some((ok, text)) = &note {
            ui.label(RichText::new(text).color(if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color }));
        }
    });
    let saved = save.then(|| match hub.set_config(candidate.clone(), true) {
        Ok(()) => (true, "Settings saved".to_string()),
        Err(e) => (false, e),
    });
    with(|p| {
        p.draft = if revert { None } else { Some(draft) };
        p.hosts_text = hosts;
        p.folders_text = folders;
        if saved.is_some() {
            p.settings_note = saved;
        }
    });
}

fn clients_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let now = crate::ai::mcp::clock::Clock::unix_ms(&crate::ai::mcp::clock::SystemClock);
    let clients = hub.clients();
    ui.label(format!("{} connected client(s), counted when they sent a request in the last {} minutes", clients.len(), CLIENT_WINDOW_MS / 60_000));
    if !clients.is_empty() {
        Grid::new("ame_devtools_mcp_clients").striped(true).show(ui, |ui| {
            for head in ["Session", "From", "Agent", "Requests", "Last seen"] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for c in &clients {
                ui.monospace(crate::ai::mcp::activity::clip(&c.session, 12));
                ui.label(&c.remote);
                ui.label(&c.user_agent);
                ui.label(c.requests.to_string());
                ui.label(ago(now, c.last_seen_ms));
                ui.end_row();
            }
        });
    }
    CollapsingHeader::new("Connection log").show(ui, |ui| {
        for e in hub.conns.events(LOG_ROWS) {
            ui.label(format!("{}  {:?}  {}  {}", ago(now, e.at_ms), e.kind, e.remote, e.detail));
        }
    });
}

fn tools_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let catalog = hub.registry.catalog();
    CollapsingHeader::new(format!("Tools ({})", catalog.len())).show(ui, |ui| {
        for row in &catalog {
            CollapsingHeader::new(format!("{}  [{}]", row.name, row.class.label())).id_salt(("ame_mcp_tool", &row.name)).show(ui, |ui| {
                ui.label(&row.description);
                ui.label(RichText::new(format!("Timeout {} s", row.timeout_secs)).weak());
                ui.monospace(serde_json::to_string_pretty(&row.input_schema).unwrap_or_default());
            });
        }
    });
    CollapsingHeader::new("Tool tester").default_open(true).show(ui, |ui| {
        let (mut index, running) = with(|p| (p.tool.min(catalog.len().saturating_sub(1)), p.running.is_some()));
        let selected = catalog.get(index).map(|r| r.name.clone()).unwrap_or_default();
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("ame_mcp_tester_tool").selected_text(&selected).show_ui(ui, |ui| {
                for (i, row) in catalog.iter().enumerate() {
                    ui.selectable_value(&mut index, i, &row.name);
                }
            });
            if ui.add_enabled(!running, Button::new("Run")).clicked() {
                let args = with(|p| p.args.clone());
                run_tool(hub.clone(), selected.clone(), args);
            }
            if ui.add_enabled(running, Button::new("Cancel")).clicked() {
                with(|p| {
                    if let Some(c) = &p.running {
                        c.cancel();
                    }
                });
            }
            if running {
                ui.spinner();
            }
        });
        with(|p| {
            if p.args_for != Some(index) {
                p.tool = index;
                p.args_for = Some(index);
                p.args = catalog.get(index).map(|r| serde_json::to_string_pretty(&skeleton(&r.input_schema)).unwrap_or_else(|_| "{}".to_string())).unwrap_or_else(|| "{}".to_string());
            }
        });
        with(|p| {
            ui.add(TextEdit::multiline(&mut p.args).code_editor().desired_rows(4).desired_width(f32::INFINITY));
        });
        if let Some(r) = with(|p| p.result.clone()) {
            let color = if r.ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
            ui.label(RichText::new(format!("{}: {} in {} ms", r.tool, if r.ok { "ok" } else { "failed" }, r.millis)).color(color).strong());
            ui.label(&r.text);
            for (i, (mime, bytes)) in r.images.iter().enumerate() {
                ui.label(format!("Image {} ({mime}, {} KiB)", i + 1, bytes.len() / 1024));
                ui.add(egui::Image::from_bytes(format!("bytes://mcp-tester-{}-{}-{i}", r.tool, r.millis), bytes.clone()).max_width(240.0));
            }
            if !r.data.is_empty() {
                ScrollArea::vertical().id_salt("ame_mcp_tester_result").max_height(220.0).show(ui, |ui| {
                    ui.monospace(&r.data);
                });
            }
        }
    });
}

fn gpu_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let status = hub.ctx.gpu.status();
    ui.label(format!(
        "{} running, {} waiting of {} places; queue wait limit {} s, job limit {} s; {} done, {} failed, {} cancelled",
        usize::from(status.running.is_some()),
        status.queued.len(),
        status.max_queued,
        status.queue_timeout_secs,
        status.job_timeout_secs,
        status.completed,
        status.failed,
        status.cancelled
    ));
    for job in status.running.iter().chain(status.queued.iter()) {
        ui.horizontal(|ui| {
            ui.label(format!("#{} {:?} {} ({}), waited {} ms, ran {} ms", job.id, job.state, job.label, job.client, job.queued_ms, job.ran_ms));
            if ui.button("Cancel").clicked() {
                hub.ctx.gpu.cancel(job.id);
            }
        });
    }
    CollapsingHeader::new("Recent jobs").show(ui, |ui| {
        for job in &status.recent {
            ui.label(format!("#{} {:?} {} ({}) waited {} ms, ran {} ms{}", job.id, job.state, job.label, job.client, job.queued_ms, job.ran_ms, job.error.as_deref().map(|e| format!(", {e}")).unwrap_or_default()));
        }
    });
}

fn proposals_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let (due, loading) = with(|p| (p.proposals_at.is_none_or(|t| t.elapsed() > Duration::from_secs(5)), p.proposals_loading));
    if due && !loading && crate::database::is_connected() {
        refresh_proposals(hub.clone());
    }
    let (proposals, counts, note, error) = with(|p| (p.proposals.clone(), p.proposal_counts, p.proposals_note.clone(), p.proposals_error.clone()));
    ui.horizontal(|ui| {
        ui.label(format!("{} pending, {} accepted, {} rejected, {} failed", counts.pending, counts.accepted, counts.rejected, counts.failed));
        if ui.button("Refresh").clicked() {
            with(|p| p.proposals_at = None);
        }
        if let Some((ok, text)) = &note {
            ui.label(RichText::new(text).color(if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color }));
        }
        if let Some(text) = &error {
            ui.label(RichText::new(text).color(ui.visuals().error_fg_color));
        }
    });
    for p in &proposals {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.monospace(&p.id);
                ui.label(&p.path);
                if ui.button("Accept").on_hover_text("Writes the changes to the library").clicked() {
                    decide(hub.clone(), p.id.clone(), true);
                }
                if ui.button("Reject").clicked() {
                    decide(hub.clone(), p.id.clone(), false);
                }
            });
            ui.label(format!("{}  from {}", serde_json::to_string(&p.changes).unwrap_or_default(), p.source));
            if let Some(reason) = &p.reason {
                ui.label(RichText::new(reason).weak());
            }
        });
    }
}

fn activity_section(ui: &mut Ui, hub: &Arc<Hub>) {
    let now = crate::ai::mcp::clock::Clock::unix_ms(&crate::ai::mcp::clock::SystemClock);
    ui.label(format!("{} tool calls since the app started", hub.activity.total()));
    CollapsingHeader::new("Tool calls").default_open(true).show(ui, |ui| {
        ScrollArea::vertical().id_salt("ame_mcp_activity").max_height(160.0).show(ui, |ui| {
            for e in hub.activity.recent(LOG_ROWS) {
                let line = format!("{}  {}  {}  {} ms  {}  {}", ago(now, e.at_ms), e.client, e.tool, e.duration_ms, e.error_kind.clone().unwrap_or_else(|| "ok".to_string()), e.args);
                ui.label(if e.ok { RichText::new(line) } else { RichText::new(line).color(ui.visuals().error_fg_color) });
            }
        });
    });
    CollapsingHeader::new("Command results").show(ui, |ui| {
        for (at, text) in hub.notes() {
            ui.label(format!("{}  {}", ago(now, at), text));
        }
    });
    CollapsingHeader::new("What the UI reports").show(ui, |ui| {
        let snapshot = crate::ai::mcp::snapshot::current(&hub.ctx.snapshot);
        ui.monospace(serde_json::to_string_pretty(&snapshot).unwrap_or_default());
    });
}

fn body(ctx: &Context, ui: &mut Ui) {
    with(|p| p.ctx = Some(ctx.clone()));
    let hub = hub().clone();
    ctx.request_repaint_after(REFRESH);
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        status_section(ui, &hub);
        token_section(ui, &hub);
        CollapsingHeader::new("Settings").show(ui, |ui| settings_section(ui, &hub));
        CollapsingHeader::new("Clients").default_open(true).show(ui, |ui| clients_section(ui, &hub));
        tools_section(ui, &hub);
        CollapsingHeader::new("GPU queue").default_open(true).show(ui, |ui| gpu_section(ui, &hub));
        CollapsingHeader::new("Metadata proposals").default_open(true).show(ui, |ui| proposals_section(ui, &hub));
        CollapsingHeader::new("Activity").default_open(true).show(ui, |ui| activity_section(ui, &hub));
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: MCP server").open(&mut open).default_width(720.0).default_height(560.0).show(ctx, |ui| body(ctx, ui));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_split_on_commas_and_lines() {
        assert_eq!(parse_list("a, b\n c ,, \n"), vec!["a", "b", "c"]);
        assert!(parse_list("  ").is_empty());
    }

    #[test]
    fn ages_are_short_text() {
        assert_eq!(ago(10_000, 9_500), "0s ago");
        assert_eq!(ago(10_000, 7_000), "3s ago");
        assert_eq!(ago(100_000, 10_000), "1m ago");
        assert_eq!(ago(10_000_000, 0), "2h ago");
        assert_eq!(ago(5, 10), "0s ago");
    }

    #[test]
    fn a_skeleton_fills_required_fields_only() {
        let schema = json!({
            "type": "object",
            "required": ["path", "n", "on", "kind", "flag"],
            "properties": {
                "path": {"type": "string"},
                "n": {"type": "integer"},
                "optional": {"type": "string"},
                "flag": {"type": "boolean"},
                "on": {"$ref": "#/$defs/On"},
                "kind": {"enum": ["a", "b"]}
            },
            "$defs": {"On": {"oneOf": [{"const": "selection"}, {"const": "focus"}]}}
        });
        assert_eq!(skeleton(&schema), json!({"path": "", "n": 0, "on": "selection", "kind": "a", "flag": false}));
        assert_eq!(skeleton(&json!({"type": "object", "properties": {}})), json!({}));
    }

    #[test]
    fn every_tool_gets_a_skeleton_that_is_an_object() {
        let hub = hub();
        for row in hub.registry.catalog() {
            assert!(skeleton(&row.input_schema).is_object(), "{}", row.name);
        }
    }

    #[test]
    fn the_window_draws_its_sections() {
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        let mut seen = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(Default::default(), |ui| show(ui.ctx()));
            seen = crate::ui::test_support::texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
            out.drop_without_applying_deltas();
        }
        for expected in ["Tool tester", "GPU queue", "Metadata proposals", "Copy gateway config snippet", "Start", "Stop"] {
            assert!(seen.iter().any(|t| t.contains(expected)), "{expected} not drawn: {seen:?}");
        }
    }

    #[test]
    fn the_state_label_names_the_address() {
        assert_eq!(state_label(&ServerState::Stopped), ("Stopped".to_string(), false));
        let addr = "127.0.0.1:9871".parse().unwrap();
        assert_eq!(state_label(&ServerState::Running { addr, started_ms: 0 }), ("Running on 127.0.0.1:9871".to_string(), true));
        assert!(state_label(&ServerState::Failed("busy".into())).0.contains("busy"));
    }
}
