//! Merges the small answer and reasoning deltas of a stream so the interface gets about 30 updates a second.

use std::time::{Duration, Instant};

/// Longest a delta waits before it is handed on.
pub const PERIOD: Duration = Duration::from_millis(33);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    Text,
    Thinking,
}

/// Deltas of one kind for one turn and segment, joined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub kind: RunKind,
    pub turn: u64,
    pub segment: u32,
    pub text: String,
}

#[derive(Debug)]
pub struct Coalescer {
    period: Duration,
    run: Option<Run>,
    since: Option<Instant>,
}

impl Default for Coalescer {
    fn default() -> Self {
        Self::new(PERIOD)
    }
}

impl Coalescer {
    pub fn new(period: Duration) -> Self {
        Self { period, run: None, since: None }
    }

    /// Adds a delta. A delta that does not continue the buffered run pushes that run out first, which keeps the order of reasoning and answer text.
    pub fn push(&mut self, kind: RunKind, turn: u64, segment: u32, text: &str, now: Instant) -> Option<Run> {
        let continues = self.run.as_ref().is_some_and(|r| r.kind == kind && r.turn == turn && r.segment == segment);
        let pushed = if continues { None } else { self.flush() };
        match &mut self.run {
            Some(run) => run.text.push_str(text),
            None => {
                self.run = Some(Run { kind, turn, segment, text: text.to_string() });
                self.since = Some(now);
            }
        }
        pushed
    }

    /// True when the buffered run has waited a full period.
    pub fn due(&self, now: Instant) -> bool {
        self.since.is_some_and(|at| now.saturating_duration_since(at) >= self.period)
    }

    /// Time until the buffered run is due, or `None` when nothing is buffered.
    pub fn wait(&self, now: Instant) -> Option<Duration> {
        self.since.map(|at| self.period.saturating_sub(now.saturating_duration_since(at)))
    }

    pub fn is_empty(&self) -> bool {
        self.run.is_none()
    }

    /// Takes whatever is buffered.
    pub fn flush(&mut self) -> Option<Run> {
        self.since = None;
        self.run.take().filter(|r| !r.text.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(ms: u64, base: Instant) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn deltas_of_one_run_join_and_wait_for_the_period() {
        let t0 = Instant::now();
        let mut c = Coalescer::new(Duration::from_millis(33));
        assert!(c.push(RunKind::Text, 0, 0, "Hel", t0).is_none());
        assert!(c.push(RunKind::Text, 0, 0, "lo ", at(5, t0)).is_none());
        assert!(c.push(RunKind::Text, 0, 0, "world", at(10, t0)).is_none());
        assert!(!c.due(at(20, t0)));
        assert!(c.due(at(33, t0)));
        assert_eq!(c.wait(at(20, t0)), Some(Duration::from_millis(13)));
        let run = c.flush().unwrap();
        assert_eq!((run.kind, run.turn, run.segment, run.text.as_str()), (RunKind::Text, 0, 0, "Hello world"));
        assert!(c.is_empty() && !c.due(at(100, t0)) && c.wait(at(100, t0)).is_none());
    }

    #[test]
    fn a_different_kind_pushes_the_earlier_run_out_first() {
        let t0 = Instant::now();
        let mut c = Coalescer::default();
        c.push(RunKind::Thinking, 0, 0, "hmm", t0);
        let out = c.push(RunKind::Text, 0, 0, "answer", at(1, t0)).expect("the reasoning run is pushed out");
        assert_eq!((out.kind, out.text.as_str()), (RunKind::Thinking, "hmm"));
        assert_eq!(c.flush().unwrap().text, "answer");
    }

    #[test]
    fn a_new_turn_or_segment_starts_a_new_run() {
        let t0 = Instant::now();
        let mut c = Coalescer::default();
        c.push(RunKind::Text, 0, 0, "a", t0);
        assert_eq!(c.push(RunKind::Text, 0, 1, "b", t0).unwrap().text, "a");
        assert_eq!(c.push(RunKind::Text, 1, 1, "c", t0).unwrap().text, "b");
        assert_eq!(c.flush().unwrap().text, "c");
    }

    #[test]
    fn a_flush_with_nothing_buffered_gives_nothing() {
        let mut c = Coalescer::default();
        assert!(c.flush().is_none());
        c.push(RunKind::Text, 0, 0, "", Instant::now());
        assert!(c.flush().is_none(), "an empty run is not worth sending");
    }

    #[test]
    fn a_thousand_small_deltas_become_a_few_runs() {
        let t0 = Instant::now();
        let mut c = Coalescer::default();
        let mut runs = 0;
        for i in 0..1000u64 {
            let now = at(i, t0);
            if let Some(_) = c.push(RunKind::Text, 0, 0, "x", now) {
                runs += 1;
            }
            if c.due(now) && c.flush().is_some() {
                runs += 1;
            }
        }
        if c.flush().is_some() {
            runs += 1;
        }
        assert!(runs <= 1000 / 30 + 2, "{runs} runs for one second of deltas");
    }
}
