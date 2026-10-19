//! What the job runner remembers about finished and running jobs: steps with their raw request and reply, usage totals, and display helpers.

use std::sync::LazyLock;

use regex::Regex;

use super::{JobId, JobKind, JobOutcome};

/// Rough tokens of `bytes` of text; the gateway reports none for `/api/complete`.
pub fn est_tokens(bytes: u64) -> u64 {
    bytes.div_ceil(4)
}

/// `text` cut to `max` bytes at a character boundary, marked when cut.
pub fn clip_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[cut: {} more bytes]", &text[..end], text.len() - end)
}

static PATHS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?:[A-Za-z]:\\[^\s"'<>|,;]+|/[^\s"'<>|,;/]+(?:/[^\s"'<>|,;/]+)+)"#).expect("the path pattern"));

/// `text` with file paths replaced by `<path>`.
pub fn mask_paths(text: &str) -> String {
    PATHS.replace_all(text, "<path>").into_owned()
}

/// One request of a job and what came back.
#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    pub index: usize,
    pub label: String,
    pub system: Option<String>,
    pub prompt: String,
    pub response: Option<String>,
    pub model: String,
    pub error: Option<String>,
    /// Requests sent for this step, retries included.
    pub attempts: u32,
    /// System and prompt bytes, the amount the gateway's input cap counts.
    pub prompt_bytes: usize,
    pub response_bytes: usize,
    pub duration_ms: u64,
    /// How the JSON of the reply was read, or why it was not.
    pub json: Option<String>,
}

impl StepRecord {
    pub fn est_tokens_in(&self) -> u64 {
        est_tokens(self.prompt_bytes as u64)
    }

    pub fn est_tokens_out(&self) -> u64 {
        est_tokens(self.response_bytes as u64)
    }
}

/// A job as the Devtools window lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct JobRecord {
    pub id: JobId,
    pub kind: JobKind,
    pub title: String,
    pub hint: String,
    pub queued_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub steps_total: usize,
    pub steps: Vec<StepRecord>,
    pub outcome: Option<JobOutcome>,
}

impl JobRecord {
    pub fn prompt_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.prompt_bytes).sum()
    }

    pub fn response_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.response_bytes).sum()
    }

    pub fn duration_ms(&self) -> Option<u64> {
        Some(self.finished_ms?.saturating_sub(self.started_ms?))
    }

    pub fn attempts(&self) -> u32 {
        self.steps.iter().map(|s| s.attempts).sum()
    }

    pub fn outcome_label(&self) -> String {
        match &self.outcome {
            None => "running".to_string(),
            Some(o) => o.label(),
        }
    }
}

/// Sums over every job since the application started.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub jobs: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// Requests sent, retries included.
    pub calls: u64,
    pub retries: u64,
    pub prompt_bytes: u64,
    pub response_bytes: u64,
    pub duration_ms: u64,
}

impl Totals {
    pub fn est_tokens_in(&self) -> u64 {
        est_tokens(self.prompt_bytes)
    }

    pub fn est_tokens_out(&self) -> u64 {
        est_tokens(self.response_bytes)
    }

    pub(super) fn add_job(&mut self, record: &JobRecord) {
        self.jobs += 1;
        match &record.outcome {
            Some(JobOutcome::Completed) => self.completed += 1,
            Some(JobOutcome::Cancelled) => self.cancelled += 1,
            Some(JobOutcome::Failed(_)) => self.failed += 1,
            None => {}
        }
        for step in &record.steps {
            self.calls += u64::from(step.attempts);
            self.retries += u64::from(step.attempts.saturating_sub(1));
            self.prompt_bytes += step.prompt_bytes as u64 * u64::from(step.attempts.max(1));
            self.response_bytes += step.response_bytes as u64;
        }
        self.duration_ms += record.duration_ms().unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_masked_and_plain_text_is_left_alone() {
        let text = "Photo /home/u/Pictures/Trip/beach.jpg and C:\\Users\\me\\a.png, see \"/x/y/z.mp4\"; rate 1/2 and /api only.";
        let masked = mask_paths(text);
        assert_eq!(masked, "Photo <path> and <path>, see \"<path>\"; rate 1/2 and /api only.");
        assert_eq!(mask_paths("no paths here"), "no paths here");
    }

    #[test]
    fn clipping_keeps_character_boundaries_and_says_what_it_cut() {
        assert_eq!(clip_text("short", 10), "short");
        let clipped = clip_text("aeioux\u{e9}\u{e9}\u{e9}", 8);
        assert!(clipped.starts_with("aeioux") && clipped.contains("[cut: "), "{clipped}");
        assert!(clip_text(&"x".repeat(100), 10).contains("[cut: 90 more bytes]"));
    }

    #[test]
    fn token_estimates_round_up_per_four_bytes() {
        assert_eq!((est_tokens(0), est_tokens(1), est_tokens(4), est_tokens(5)), (0, 1, 1, 2));
    }
}
