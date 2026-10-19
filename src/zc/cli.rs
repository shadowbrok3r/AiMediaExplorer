//! `--zc-status`: probes a gateway from the command line and exits, without opening the window or the database.

use super::health::{self, Health, HealthKind};
use super::profile::ZcProfile;
use super::secrets;

pub const USAGE: &str = "--zc-status [--zc-url URL] [--zc-agent ALIAS] [--zc-token-file PATH]  (env ZC_URL, ZC_AGENT, ZC_TOKEN_FILE)";

/// The values after `--zc-status`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StatusArgs {
    pub url: Option<String>,
    pub agent: Option<String>,
    pub token_file: Option<String>,
}

/// Reads the flags; `None` when `--zc-status` is absent.
pub fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Option<Result<StatusArgs, String>> {
    if !args.iter().any(|a| a == "--zc-status") {
        return None;
    }
    let mut out = StatusArgs::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f, Some(v.to_string())),
            None => (arg.as_str(), None),
        };
        let slot = match flag {
            "--zc-url" => &mut out.url,
            "--zc-agent" => &mut out.agent,
            "--zc-token-file" => &mut out.token_file,
            _ => continue,
        };
        match inline.or_else(|| it.next().cloned()) {
            Some(v) => *slot = Some(v),
            None => return Some(Err(format!("{flag} needs a value"))),
        }
    }
    out.url = out.url.or_else(|| env("ZC_URL"));
    out.agent = out.agent.or_else(|| env("ZC_AGENT"));
    out.token_file = out.token_file.or_else(|| env("ZC_TOKEN_FILE"));
    Some(Ok(out))
}

/// Exit code of a probe: 0 online, 2 not set up, 3 token refused, 4 unreachable, 1 anything else.
pub fn exit_code(health: &Health) -> i32 {
    match health.kind() {
        HealthKind::Online => 0,
        HealthKind::Unconfigured => 2,
        HealthKind::Unauthorized => 3,
        HealthKind::Offline => 4,
        HealthKind::Checking | HealthKind::UnknownAgent => 1,
    }
}

/// Runs the probe described by `args` and returns the printed line and the exit code.
pub async fn probe_args(args: &StatusArgs) -> (String, i32) {
    let profile = ZcProfile { host: args.url.clone().unwrap_or_default(), agent: args.agent.clone().unwrap_or_default(), ..Default::default() };
    let token = match args.token_file.as_deref() {
        Some(file) => match secrets::read_token_file(std::path::Path::new(file)) {
            Ok(t) => Some(t),
            Err(e) => return (serde_json::json!({ "state": "unconfigured", "detail": e }).to_string(), 2),
        },
        None => None,
    };
    let found = health::probe(&profile, token.as_deref()).await;
    (found.to_json().to_string(), exit_code(&found))
}

/// Handles `--zc-status` when it is among `args`; the exit code to end the process with.
pub fn run(args: &[String]) -> Option<i32> {
    let parsed = parse(args, &|k| std::env::var(k).ok())?;
    let parsed = match parsed {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}\nusage: {USAGE}");
            return Some(2);
        }
    };
    let outcome = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
                rt.block_on(probe_args(&parsed))
            })
            .join()
    });
    match outcome {
        Ok((line, code)) => {
            println!("{line}");
            Some(code)
        }
        Err(_) => {
            eprintln!("the probe thread panicked");
            Some(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_client::mock::MockGateway;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn the_flag_is_required_and_other_arguments_are_ignored() {
        assert!(parse(&args(&["--db-info"]), &no_env).is_none());
        let parsed = parse(&args(&["--zc-status", "--zc-url", "h:1", "--zc-agent=media", "--theme", "x"]), &no_env).unwrap().unwrap();
        assert_eq!(parsed, StatusArgs { url: Some("h:1".into()), agent: Some("media".into()), token_file: None });
    }

    #[test]
    fn the_environment_fills_in_what_the_flags_leave_out() {
        let env = |k: &str| (k == "ZC_URL").then(|| "env-host".to_string());
        let parsed = parse(&args(&["--zc-status"]), &env).unwrap().unwrap();
        assert_eq!(parsed.url.as_deref(), Some("env-host"));
        assert!(parse(&args(&["--zc-status", "--zc-url"]), &no_env).unwrap().is_err());
    }

    #[test]
    fn exit_codes_follow_the_probe_taxonomy() {
        assert_eq!(exit_code(&Health::Online { agents: vec![], require_pairing: false, latency_ms: 1 }), 0);
        assert_eq!(exit_code(&Health::Unconfigured("x".into())), 2);
        assert_eq!(exit_code(&Health::Unauthorized), 3);
        assert_eq!(exit_code(&Health::Offline("x".into())), 4);
    }

    #[tokio::test]
    async fn a_probe_against_the_mock_prints_the_state_and_a_zero_exit() {
        let mock = MockGateway::start().await;
        let dir = std::env::temp_dir().join(format!("ame-zc-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("token");
        std::fs::write(&file, mock.token().unwrap()).unwrap();
        let (line, code) = probe_args(&StatusArgs { url: Some(mock.url()), agent: Some("default".into()), token_file: Some(file.display().to_string()) }).await;
        assert_eq!(code, 0, "{line}");
        assert!(line.contains("\"online\""));
        assert!(!line.contains(mock.token().unwrap()));
        std::fs::write(&file, "wrong").unwrap();
        let (line, code) = probe_args(&StatusArgs { url: Some(mock.url()), agent: Some("default".into()), token_file: Some(file.display().to_string()) }).await;
        assert_eq!(code, 3, "{line}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_missing_token_file_is_a_usage_failure_not_a_panic() {
        let (line, code) = probe_args(&StatusArgs { url: Some("127.0.0.1:1".into()), agent: Some("a".into()), token_file: Some("/nonexistent/zc-token".into()) }).await;
        assert_eq!(code, 2);
        assert!(line.contains("cannot read"));
    }
}
