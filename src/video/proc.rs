//! Runs ffmpeg and ffprobe as child processes with a deadline, cancellation and capped output.

use super::{Deadline, VideoError};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

const STDERR_KEEP: usize = 4096;
const POLL: Duration = Duration::from_millis(5);

pub struct Output {
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub success: bool,
}

/// Command line for a tool without a console window and without stdin.
pub fn command(tool: &Path) -> Command {
    let mut cmd = Command::new(tool);
    cmd.stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

/// Input argument that ffmpeg tools never mistake for an option or a protocol URL.
pub fn input_arg(path: &Path) -> String {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    format!("file:{}", abs.display())
}

fn read_capped(mut src: impl Read, cap: usize) -> (Vec<u8>, bool) {
    let mut out = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        match src.read(&mut buf) {
            Ok(0) | Err(_) => return (out, false),
            Ok(n) => {
                if out.len() + n > cap {
                    out.extend_from_slice(&buf[..cap.saturating_sub(out.len())]);
                    return (out, true);
                }
                out.extend_from_slice(&buf[..n]);
            }
        }
    }
}

fn tail_text(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(STDERR_KEEP);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

/// Runs `cmd` to completion, killing it when the deadline passes, the job is cancelled or stdout exceeds `max_stdout`.
pub fn run_capture(mut cmd: Command, deadline: &Deadline, max_stdout: usize) -> Result<Output, VideoError> {
    deadline.check()?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| VideoError::Backend(format!("spawn failed: {e}")))?;
    let stdout = child.stdout.take().map(|s| std::thread::spawn(move || read_capped(s, max_stdout)));
    let stderr = child.stderr.take().map(|s| std::thread::spawn(move || read_capped(s, 64 * 1024).0));
    let mut failure = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                failure = Some(VideoError::Backend(format!("wait failed: {e}")));
                break None;
            }
        }
        if let Some(reason) = deadline.reason() {
            failure = Some(reason);
            break None;
        }
        std::thread::sleep(POLL);
    };
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let (stdout, truncated) = stdout.map(|t| t.join().unwrap_or_default()).unwrap_or_default();
    let stderr = stderr.map(|t| t.join().unwrap_or_default()).unwrap_or_default();
    if let Some(err) = failure {
        return Err(err);
    }
    if truncated {
        return Err(VideoError::Backend("tool output exceeded the size limit".into()));
    }
    Ok(Output { stdout, stderr: tail_text(&stderr), success: status.is_some_and(|s| s.success()) })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::video::Cancel;
    use std::time::Instant;

    fn sh(script: &str) -> Command {
        let mut cmd = command(Path::new("/bin/sh"));
        cmd.env("PATH", "/usr/bin:/bin").arg("-c").arg(script);
        cmd
    }

    fn deadline(ms: u64) -> Deadline {
        Deadline::new(Duration::from_millis(ms), Cancel::new())
    }

    #[test]
    fn captures_output_and_status() {
        let out = run_capture(sh("printf hello; printf oops >&2; exit 3"), &deadline(5000), 1024).unwrap();
        assert_eq!(out.stdout, b"hello");
        assert_eq!(out.stderr, "oops");
        assert!(!out.success);
        assert!(run_capture(sh("true"), &deadline(5000), 1024).unwrap().success);
    }

    #[test]
    fn a_hung_child_is_killed_at_the_deadline() {
        let started = Instant::now();
        let err = run_capture(sh("sleep 30"), &deadline(150), 1024).err();
        assert_eq!(err, Some(VideoError::Timeout));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn cancellation_kills_the_child() {
        let cancel = Cancel::new();
        let d = Deadline::new(Duration::from_secs(60), cancel.clone());
        let flag = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            flag.cancel();
        });
        let started = Instant::now();
        assert_eq!(run_capture(sh("sleep 30"), &d, 1024).err(), Some(VideoError::Cancelled));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn oversized_output_is_an_error_and_stops_the_child() {
        let started = Instant::now();
        let err = run_capture(sh("yes"), &deadline(10_000), 4096).err();
        assert!(matches!(err, Some(VideoError::Backend(_))), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn missing_tool_is_a_backend_error() {
        let err = run_capture(command(Path::new("/nonexistent/tool")), &deadline(1000), 16).err();
        assert!(matches!(err, Some(VideoError::Backend(_))));
    }

    #[test]
    fn input_arg_is_absolute_and_prefixed() {
        assert_eq!(input_arg(Path::new("/a/-b:c.mp4")), "file:/a/-b:c.mp4");
        assert!(input_arg(Path::new("rel.mp4")).starts_with("file:/"));
    }
}
