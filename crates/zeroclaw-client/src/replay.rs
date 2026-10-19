use std::time::Duration;

use serde_json::{Value, json};

use crate::event::{Direction, RawFrame};

/// One recorded frame: when it crossed the socket, which way, and its JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub t_ms: u64,
    pub dir: Direction,
    pub frame: Value,
}

/// A recorded frame sequence as JSON lines `{"t_ms":0,"dir":"in","frame":{...}}`; `in` is gateway to client.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recording {
    pub entries: Vec<Entry>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("line {line}: {message}")]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl Recording {
    /// Parses JSON lines; blank lines and lines starting with `#` are skipped.
    pub fn parse(text: &str) -> Result<Recording, ParseError> {
        let mut entries = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let err = |message: &str| ParseError { line: i + 1, message: message.to_string() };
            let v: Value = serde_json::from_str(line).map_err(|e| err(&e.to_string()))?;
            let dir = match v.get("dir").and_then(Value::as_str) {
                Some("in") => Direction::In,
                Some("out") => Direction::Out,
                _ => return Err(err("`dir` must be \"in\" or \"out\"")),
            };
            let frame = v.get("frame").cloned().ok_or_else(|| err("missing `frame`"))?;
            let t_ms = v.get("t_ms").and_then(Value::as_u64).unwrap_or(0);
            entries.push(Entry { t_ms, dir, frame });
        }
        Ok(Recording { entries })
    }

    pub fn from_raw(frames: &[RawFrame]) -> Recording {
        Recording {
            entries: frames.iter().map(|f| Entry { t_ms: f.t_ms, dir: f.dir, frame: f.frame.clone() }).collect(),
        }
    }

    pub fn to_jsonl(&self) -> String {
        self.entries.iter().map(|e| line(e.t_ms, e.dir, &e.frame)).collect::<Vec<_>>().join("\n")
    }

    /// Frames the gateway sent, in order.
    pub fn server_frames(&self) -> impl Iterator<Item = &Value> {
        self.entries.iter().filter(|e| e.dir == Direction::In).map(|e| &e.frame)
    }

    /// Delay before each entry, scaled by `speed` (2.0 plays twice as fast, 0 removes every delay).
    pub fn delays(&self, speed: f64) -> Vec<Duration> {
        let mut prev = 0u64;
        self.entries
            .iter()
            .map(|e| {
                let dt = e.t_ms.saturating_sub(prev);
                prev = e.t_ms.max(prev);
                if speed <= 0.0 { Duration::ZERO } else { Duration::from_secs_f64(dt as f64 / 1000.0 / speed) }
            })
            .collect()
    }
}

/// One JSONL line for a frame.
pub fn line(t_ms: u64, dir: Direction, frame: &Value) -> String {
    let dir = match dir {
        Direction::In => "in",
        Direction::Out => "out",
    };
    json!({ "t_ms": t_ms, "dir": dir, "frame": frame }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# comment
{"t_ms":0,"dir":"in","frame":{"type":"session_start","session_id":"s"}}
{"t_ms":5,"dir":"out","frame":{"type":"message","content":"hi"}}

{"t_ms":105,"dir":"in","frame":{"type":"done","full_response":"ok"}}
"#;

    #[test]
    fn parses_skipping_blank_and_comment_lines() {
        let r = Recording::parse(SAMPLE).unwrap();
        assert_eq!(r.entries.len(), 3);
        assert_eq!(r.entries[1].dir, Direction::Out);
        assert_eq!(r.server_frames().count(), 2);
    }

    #[test]
    fn round_trips_through_jsonl() {
        let r = Recording::parse(SAMPLE).unwrap();
        assert_eq!(Recording::parse(&r.to_jsonl()).unwrap(), r);
    }

    #[test]
    fn reports_the_failing_line() {
        let err = Recording::parse("{\"dir\":\"in\",\"frame\":{}}\nnot json").unwrap_err();
        assert_eq!(err.line, 2);
        assert_eq!(Recording::parse("{\"dir\":\"sideways\",\"frame\":{}}").unwrap_err().line, 1);
        assert_eq!(Recording::parse("{\"dir\":\"in\"}").unwrap_err().message, "missing `frame`");
    }

    #[test]
    fn delays_are_gaps_scaled_by_speed() {
        let r = Recording::parse(SAMPLE).unwrap();
        let ms = |s: f64| r.delays(s).iter().map(|d| d.as_millis()).collect::<Vec<_>>();
        assert_eq!(ms(1.0), [0, 5, 100]);
        assert_eq!(ms(10.0), [0, 0, 10]);
        assert_eq!(ms(0.0), [0, 0, 0]);
    }

    #[test]
    fn builds_from_raw_frames() {
        let raw = [RawFrame { t_ms: 7, dir: Direction::In, frame: json!({"type":"chunk"}) }];
        let r = Recording::from_raw(&raw);
        assert_eq!(r.entries[0].t_ms, 7);
        assert!(r.to_jsonl().contains("\"dir\":\"in\""));
    }
}
