use std::process::Command;
use std::time::Duration;

use serde_json::{Value, json};
use zeroclaw_client::mock::{MockGateway, MockOptions, Turn};
use zeroclaw_client::replay::Recording;

const TOKEN: &str = "ZCTOKEN-secret-0123456789";

struct Run {
    code: i32,
    out: String,
    err: String,
}

async fn zcprobe(args: &[&str], env: &[(&str, &str)]) -> Run {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let env: Vec<(String, String)> = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    tokio::task::spawn_blocking(move || {
        let output = Command::new(env!("CARGO_BIN_EXE_zcprobe"))
            .args(&args)
            .env_clear()
            .envs(env)
            .output()
            .expect("run zcprobe");
        Run {
            code: output.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&output.stdout).into_owned(),
            err: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    })
    .await
    .unwrap()
}

async fn mock() -> MockGateway {
    MockGateway::start_with(MockOptions { token: Some(TOKEN.into()), ..MockOptions::default() }).await
}

fn no_token(run: &Run) {
    assert!(!run.out.contains(TOKEN) && !run.err.contains(TOKEN), "token leaked:\nstdout: {}\nstderr: {}", run.out, run.err);
}

fn base(mock: &MockGateway) -> Vec<String> {
    vec!["--url".into(), mock.url(), "--token".into(), TOKEN.into(), "--agent".into(), "default".into()]
}

async fn run(mock: &MockGateway, extra: &[&str]) -> Run {
    let mut args = base(mock);
    args.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let r = zcprobe(&refs, &[]).await;
    no_token(&r);
    r
}

#[tokio::test]
async fn help_and_usage_errors() {
    let r = zcprobe(&["--help"], &[]).await;
    assert_eq!(r.code, 0);
    assert!(r.out.contains("EXIT CODES") && r.out.contains("frames"));

    let r = zcprobe(&["health"], &[]).await;
    assert_eq!(r.code, 2);
    assert!(r.err.contains("no gateway address"), "{}", r.err);

    let r = zcprobe(&["--url", "h", "bogus"], &[]).await;
    assert_eq!(r.code, 2);
    assert!(r.err.contains("unknown command"));

    let r = zcprobe(&[], &[]).await;
    assert_eq!(r.code, 2);

    let r = zcprobe(&["--url", "h", "chat", "hi"], &[]).await;
    assert_eq!(r.code, 2, "chat needs an agent");
    assert!(r.err.contains("no agent"));
}

#[tokio::test]
async fn health_reports_auth_and_agents() {
    let mock = mock().await;
    let r = run(&mock, &["health"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("status: ok"));
    assert!(r.out.contains("auth: accepted"));
    assert!(r.out.contains("agents: default"));
    assert!(r.out.contains("agent default: known"));
}

#[tokio::test]
async fn health_reads_the_environment_and_json_output() {
    let mock = mock().await;
    let url = mock.url();
    let r = zcprobe(&["health", "--json"], &[("ZC_URL", &url), ("ZC_TOKEN", TOKEN), ("ZC_AGENT", "default")]).await;
    no_token(&r);
    assert_eq!(r.code, 0, "{}", r.err);
    let v: Value = serde_json::from_str(r.out.trim()).unwrap();
    assert_eq!(v["auth"], "accepted");
    assert_eq!(v["agent_known"], true);
    assert_eq!(v["agents"], json!(["default"]));
}

#[tokio::test]
async fn token_file_works_and_the_flag_is_optional() {
    let mock = mock().await;
    let path = std::env::temp_dir().join(format!("zcprobe-token-{}", std::process::id()));
    std::fs::write(&path, format!("{TOKEN}\n")).unwrap();
    let r = zcprobe(&["--url", &mock.url(), "--token-file", path.to_str().unwrap(), "health"], &[]).await;
    let _ = std::fs::remove_file(&path);
    no_token(&r);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("auth: accepted"));
}

#[tokio::test]
async fn health_exit_codes_for_bad_token_unknown_agent_and_unreachable() {
    let mock = mock().await;
    let r = zcprobe(&["--url", &mock.url(), "--token", "wrong", "health"], &[]).await;
    assert_eq!(r.code, 3);
    assert!(r.out.contains("auth: rejected (HTTP 401)"));

    let r = run(&mock, &["--agent", "ghost", "health"]).await;
    assert_eq!(r.code, 1);
    assert!(r.out.contains("agent ghost: unknown"));

    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let r = zcprobe(&["--url", &format!("127.0.0.1:{port}"), "--token", TOKEN, "health"], &[]).await;
    no_token(&r);
    assert_eq!(r.code, 4, "{}", r.err);
    assert!(r.err.contains("cannot reach the gateway"));
}

#[tokio::test]
async fn sessions_lists_rows_and_json_lines() {
    let mock = mock().await;
    mock.add_session("alpha", "default", 4, "2026-09-11T10:00:00+00:00");
    mock.add_session("beta", "default", 0, "2026-09-12T10:00:00+00:00");
    mock.add_session("gamma", "other", 3, "2026-09-12T11:00:00+00:00");
    let r = run(&mock, &["sessions"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("alpha") && !r.out.contains("beta") && !r.out.contains("gamma"), "{}", r.out);
    let r = run(&mock, &["sessions", "--all", "--json"]).await;
    let ids: Vec<String> = r.out.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["session_id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids, ["alpha", "beta"]);
}

#[tokio::test]
async fn history_prints_the_newest_messages() {
    let mock = mock().await;
    let rows: Vec<_> = (0..6).map(|i| json!({"role":"user","content":format!("message {i}")})).collect();
    mock.set_messages("gw_s1", rows);
    let r = run(&mock, &["history", "s1", "--limit", "2"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out.lines().count(), 2);
    assert!(r.out.contains("message 4") && r.out.contains("message 5"));
    assert!(r.err.contains("4 older messages not shown"));
    let r = run(&mock, &["history", "s1", "--json"]).await;
    assert_eq!(r.out.lines().count(), 6);
    let missing = run(&mock, &["history"]).await;
    assert_eq!(missing.code, 2);
}

#[tokio::test]
async fn complete_prints_the_reply_and_maps_errors() {
    let mock = mock().await;
    let r = run(&mock, &["complete", "say", "hello"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out.trim(), "echo: say hello");
    let r = run(&mock, &["complete", "x", "--json", "--system", "brief", "--temperature", "0.1"]).await;
    let v: Value = serde_json::from_str(r.out.trim()).unwrap();
    assert_eq!(v["model"], "mock-model");
    let r = run(&mock, &["complete", "x", "--hint", "slow"]).await;
    assert_eq!(r.code, 1);
    assert!(r.err.contains("no [[model_routes]] entry has hint `slow`"), "{}", r.err);
}

#[tokio::test]
async fn chat_streams_text_to_stdout_and_events_to_stderr() {
    let mock = mock().await;
    mock.script_turn(
        Turn::new()
            .thinking("pondering")
            .tool_call("c1", "aimedia__library_stats", json!({}))
            .tool_result("c1", "aimedia__library_stats", "1042 items")
            .chunk("Hello ")
            .chunk("world")
            .done("Hello world"),
    );
    let r = run(&mock, &["chat", "hi", "there", "--thinking"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out, "Hello world\n");
    assert!(r.err.contains("[thinking] pondering"));
    assert!(r.err.contains("[tool_call] aimedia__library_stats"));
    assert!(r.err.contains("[tool_result] aimedia__library_stats 1042 items"));
    assert!(r.err.contains("[done] in=100 out=20 ctx=10% model=mock-model"), "{}", r.err);
    assert_eq!(mock.received_of_type("message")[0]["content"], "hi there");
}

#[tokio::test]
async fn chat_json_mode_prints_one_event_per_line() {
    let mock = mock().await;
    mock.script_turn(Turn::new().chunk("a").done("a"));
    let r = run(&mock, &["chat", "go", "--json"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    let events: Vec<Value> = r.out.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let kinds: Vec<_> = events.iter().map(|e| e["event"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["connected", "turn_started", "token", "done"]);
    assert_eq!(events[3]["full_response"], "a");
}

#[tokio::test]
async fn chat_denies_approvals_unless_told_otherwise() {
    let mock = mock().await;
    for _ in 0..2 {
        mock.script_turn(Turn::new().approval("r1", "aimedia__image_edit_run", "edit a.jpg").chunk("ok").done("ok"));
    }
    let r = run(&mock, &["chat", "edit"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.err.contains("[approval] aimedia__image_edit_run edit a.jpg -> denied"));
    let r = run(&mock, &["chat", "edit", "--auto-approve"]).await;
    assert!(r.err.contains("-> approved"));
    let decisions: Vec<_> = mock.received_of_type("approval_response").iter().map(|f| f["decision"].as_str().unwrap().to_string()).collect();
    assert_eq!(decisions, ["deny", "approve"]);
}

#[tokio::test]
async fn chat_exit_codes_for_failed_turns_and_credentials() {
    let mock = mock().await;
    mock.script_turn(Turn::new().chunk("par").error("PROVIDER_ERROR", "upstream returned 502"));
    let r = run(&mock, &["chat", "go"]).await;
    assert_eq!(r.code, 1);
    assert!(r.err.contains("[error] PROVIDER_ERROR upstream returned 502"));

    let r = zcprobe(&["--url", &mock.url(), "--token", "wrong", "--agent", "default", "chat", "go"], &[]).await;
    assert_eq!(r.code, 3, "{}", r.err);

    let r = run(&mock, &["--agent", "ghost", "chat", "go"]).await;
    assert_eq!(r.code, 1, "{}", r.err);
    assert!(r.err.contains("Unknown agent"));
}

#[tokio::test]
async fn chat_gives_up_after_the_timeout() {
    let mock = mock().await;
    mock.script_turn(Turn::new().chunk("slow").delay_ms(30_000));
    let started = std::time::Instant::now();
    let r = run(&mock, &["chat", "go", "--timeout", "1"]).await;
    assert_eq!(r.code, 1);
    assert!(r.err.contains("timed out"));
    assert!(started.elapsed() < Duration::from_secs(8));
}

#[tokio::test]
async fn chat_sends_images_as_inline_markers() {
    let mock = mock().await;
    let path = std::env::temp_dir().join(format!("zcprobe-img-{}.png", std::process::id()));
    std::fs::write(&path, b"\x89PNG\r\n\x1a\nrest").unwrap();
    let r = run(&mock, &["chat", "describe", "--image", path.to_str().unwrap()]).await;
    let _ = std::fs::remove_file(&path);
    assert_eq!(r.code, 0, "{}", r.err);
    let sent = mock.received_of_type("message")[0]["content"].as_str().unwrap().to_string();
    assert!(sent.starts_with("describe [IMAGE:data:image/png;base64,"), "{sent}");

    let r = run(&mock, &["chat", "describe", "--image", "/nonexistent/x.png"]).await;
    assert_eq!(r.code, 2, "{}", r.err);
}

#[tokio::test]
async fn frames_dumps_replayable_json_lines() {
    let mock = mock().await;
    mock.script_turn(Turn::new().thinking("hm").chunk("a").done("a"));
    let r = run(&mock, &["frames", "go"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    let rec = Recording::parse(&r.out).expect("output is a valid recording");
    let types: Vec<_> = rec.entries.iter().map(|e| e.frame["type"].as_str().unwrap().to_string()).collect();
    assert_eq!(types.first().map(String::as_str), Some("connect"));
    assert!(types.contains(&"session_start".into()) && types.contains(&"message".into()));
    assert_eq!(types.last().map(String::as_str), Some("done"));
    assert!(rec.entries.windows(2).all(|w| w[0].t_ms <= w[1].t_ms));
}

#[tokio::test]
async fn frames_without_a_prompt_listens_for_the_given_seconds() {
    let mock = mock().await;
    let started = std::time::Instant::now();
    let r = run(&mock, &["frames", "--seconds", "1"]).await;
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(started.elapsed() >= Duration::from_millis(900));
    let rec = Recording::parse(&r.out).unwrap();
    assert!(rec.entries.iter().any(|e| e.frame["type"] == "session_start"));
}

#[tokio::test]
async fn the_token_is_scrubbed_even_when_the_server_echoes_it() {
    let mock = mock().await;
    mock.override_rest("GET", "/api/sessions", 502, json!({"error": format!("upstream rejected {TOKEN}")}), Duration::ZERO);
    let r = run(&mock, &["sessions"]).await;
    assert_eq!(r.code, 1);
    assert!(r.err.contains("upstream rejected"), "{}", r.err);
    assert!(r.err.contains("***"), "{}", r.err);
}
