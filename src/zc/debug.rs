//! Debug tap of the chat socket: the last frames in both directions and an optional JSONL recording.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};

use zeroclaw_client::{Direction, RawFrame};

/// Frames kept for the Devtools window.
pub const FRAME_CAP: usize = 500;

/// A recording file being written.
pub struct Recorder {
    path: PathBuf,
    file: std::io::BufWriter<std::fs::File>,
    pub lines: u64,
}

impl Recorder {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self { path: path.to_path_buf(), file: std::io::BufWriter::new(file), lines: 0 })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn write(&mut self, line: &str) {
        if writeln!(self.file, "{line}").is_ok() {
            self.lines += 1;
            let _ = self.file.flush();
        }
    }
}

/// The frames of the connection and what is recorded.
#[derive(Default)]
pub struct FrameLog {
    frames: VecDeque<RawFrame>,
    /// Frames seen since the log was created or cleared.
    pub total_in: u64,
    pub total_out: u64,
    pub recorder: Option<Recorder>,
    /// Bumped on every change, so a window can tell when to redraw.
    pub revision: u64,
}

impl FrameLog {
    /// Adds a frame; `redact` removes secrets from the text of the recorded line.
    pub fn push(&mut self, frame: RawFrame, redact: impl Fn(&str) -> String) {
        match frame.dir {
            Direction::In => self.total_in += 1,
            Direction::Out => self.total_out += 1,
        }
        if let Some(rec) = &mut self.recorder {
            let line = zeroclaw_client::replay::line(frame.t_ms, frame.dir, &frame.frame);
            rec.write(&redact(&line));
        }
        self.frames.push_back(frame);
        while self.frames.len() > FRAME_CAP {
            self.frames.pop_front();
        }
        self.revision += 1;
    }

    pub fn frames(&self) -> impl Iterator<Item = &RawFrame> {
        self.frames.iter()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.total_in = 0;
        self.total_out = 0;
        self.revision += 1;
    }

    /// Starts writing every new frame to `path`.
    pub fn start_recording(&mut self, path: &Path) -> std::io::Result<()> {
        self.recorder = Some(Recorder::create(path)?);
        self.revision += 1;
        Ok(())
    }

    pub fn stop_recording(&mut self) -> Option<(PathBuf, u64)> {
        self.revision += 1;
        self.recorder.take().map(|r| (r.path.clone(), r.lines))
    }

    pub fn recording(&self) -> Option<&Recorder> {
        self.recorder.as_ref()
    }
}

/// The `type` of a frame, or `?` when it has none.
pub fn frame_type(frame: &serde_json::Value) -> String {
    frame.get("type").and_then(|t| t.as_str()).unwrap_or("?").to_string()
}

/// True when `frame` matches the filter: `dir` limits the direction, `text` is a case-insensitive substring of its type or JSON.
pub fn matches(frame: &RawFrame, dir: Option<Direction>, text: &str) -> bool {
    if dir.is_some_and(|d| d != frame.dir) {
        return false;
    }
    let needle = text.trim().to_lowercase();
    needle.is_empty() || frame_type(&frame.frame).to_lowercase().contains(&needle) || frame.frame.to_string().to_lowercase().contains(&needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frame(t: u64, dir: Direction, v: serde_json::Value) -> RawFrame {
        RawFrame { t_ms: t, dir, frame: v }
    }

    #[test]
    fn the_log_keeps_only_the_newest_frames_and_counts_every_one() {
        let mut log = FrameLog::default();
        for i in 0..(FRAME_CAP as u64 + 25) {
            log.push(frame(i, Direction::In, json!({"type":"chunk","content":i})), |s| s.to_string());
        }
        log.push(frame(9999, Direction::Out, json!({"type":"message","content":"hi"})), |s| s.to_string());
        assert_eq!(log.len(), FRAME_CAP);
        assert_eq!(log.total_in, FRAME_CAP as u64 + 25);
        assert_eq!(log.total_out, 1);
        assert_eq!(log.frames().next().unwrap().t_ms, 26);
        log.clear();
        assert!(log.is_empty() && log.total_in == 0);
    }

    #[test]
    fn filters_match_direction_type_and_content() {
        let a = frame(0, Direction::In, json!({"type":"tool_call","name":"aimedia__library_stats"}));
        let b = frame(1, Direction::Out, json!({"type":"message","content":"find sunsets"}));
        assert!(matches(&a, None, ""));
        assert!(matches(&a, None, "TOOL_call"));
        assert!(matches(&a, None, "library_stats"));
        assert!(!matches(&a, Some(Direction::Out), ""));
        assert!(matches(&b, Some(Direction::Out), "sunset"));
        assert!(!matches(&b, None, "tool"));
        assert_eq!(frame_type(&json!({"x":1})), "?");
    }

    #[test]
    fn a_recording_writes_redacted_json_lines_a_replay_can_read() {
        let dir = std::env::temp_dir().join(format!("ame-zc-debug-{}", std::process::id()));
        let path = dir.join("rec.jsonl");
        let mut log = FrameLog::default();
        log.start_recording(&path).unwrap();
        log.push(frame(0, Direction::In, json!({"type":"session_start","session_id":"s","note":"SECRET-TOKEN"})), |s| s.replace("SECRET-TOKEN", "***"));
        log.push(frame(5, Direction::Out, json!({"type":"message","content":"hi"})), |s| s.to_string());
        let (written, lines) = log.stop_recording().unwrap();
        assert_eq!(lines, 2);
        let text = std::fs::read_to_string(&written).unwrap();
        assert!(!text.contains("SECRET-TOKEN") && text.contains("***"));
        let rec = zeroclaw_client::replay::Recording::parse(&text).unwrap();
        assert_eq!(rec.entries.len(), 2);
        assert!(log.recording().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
