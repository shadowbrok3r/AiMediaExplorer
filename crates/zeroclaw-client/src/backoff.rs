use std::time::Duration;

use crate::config::ReconnectPolicy;

/// Exponential backoff with downward jitter and a deterministic random source.
#[derive(Debug, Clone)]
pub struct Backoff {
    policy: ReconnectPolicy,
    attempt: u32,
    rng: u64,
}

impl Backoff {
    pub fn new(policy: ReconnectPolicy, seed: u64) -> Self {
        Self { policy, attempt: 0, rng: seed | 1 }
    }

    /// Seeds from the wall clock.
    pub fn from_clock(policy: ReconnectPolicy) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self::new(policy, seed)
    }

    /// Failed attempts counted since the last reset.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
    }

    /// Delay before the next attempt, or `None` when reconnecting is disabled or the attempts are used up.
    pub fn next_delay(&mut self) -> Option<Duration> {
        if !self.policy.enabled {
            return None;
        }
        if self.policy.max_attempts.is_some_and(|max| self.attempt >= max) {
            return None;
        }
        let exp = self.attempt.min(63) as i32;
        self.attempt = self.attempt.saturating_add(1);
        let base = self.policy.initial.as_secs_f64() * self.policy.multiplier.max(1.0).powi(exp);
        let capped = base.min(self.policy.max.as_secs_f64());
        let jitter = self.policy.jitter.clamp(0.0, 1.0);
        let factor = 1.0 - jitter * self.unit();
        Some(Duration::from_secs_f64((capped * factor).max(0.0)))
    }

    /// Uniform value in `[0, 1)` from xorshift64*.
    fn unit(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ReconnectPolicy {
        ReconnectPolicy {
            enabled: true,
            initial: Duration::from_millis(100),
            max: Duration::from_millis(1000),
            multiplier: 2.0,
            jitter: 0.0,
            max_attempts: None,
        }
    }

    #[test]
    fn schedule_doubles_and_caps() {
        let mut b = Backoff::new(policy(), 1);
        let ms: Vec<u128> = (0..6).map(|_| b.next_delay().unwrap().as_millis()).collect();
        assert_eq!(ms, [100, 200, 400, 800, 1000, 1000]);
        assert_eq!(b.attempt(), 6);
    }

    #[test]
    fn reset_restarts_the_schedule() {
        let mut b = Backoff::new(policy(), 1);
        b.next_delay();
        b.next_delay();
        b.reset();
        assert_eq!(b.next_delay().unwrap().as_millis(), 100);
    }

    #[test]
    fn max_attempts_ends_the_schedule() {
        let mut b = Backoff::new(ReconnectPolicy { max_attempts: Some(2), ..policy() }, 1);
        assert!(b.next_delay().is_some());
        assert!(b.next_delay().is_some());
        assert!(b.next_delay().is_none());
        assert!(b.next_delay().is_none());
    }

    #[test]
    fn disabled_policy_never_retries() {
        let mut b = Backoff::new(ReconnectPolicy { enabled: false, ..policy() }, 1);
        assert!(b.next_delay().is_none());
    }

    #[test]
    fn jitter_stays_within_bounds_and_is_deterministic() {
        let p = ReconnectPolicy { jitter: 0.5, ..policy() };
        let mut a = Backoff::new(p.clone(), 42);
        let mut b = Backoff::new(p, 42);
        let mut distinct = std::collections::BTreeSet::new();
        for i in 0..20 {
            let da = a.next_delay().unwrap();
            assert_eq!(da, b.next_delay().unwrap());
            let cap = (100.0 * 2f64.powi(i)).min(1000.0);
            let ms = da.as_secs_f64() * 1000.0;
            assert!(ms <= cap + 1e-9 && ms >= cap * 0.5 - 1e-9, "attempt {i}: {ms} outside [{}, {cap}]", cap * 0.5);
            distinct.insert(da.as_micros());
        }
        assert!(distinct.len() > 5, "jitter should vary the delays");
    }

    #[test]
    fn huge_attempt_counts_do_not_overflow() {
        let mut b = Backoff::new(policy(), 1);
        for _ in 0..200 {
            assert!(b.next_delay().unwrap() <= Duration::from_millis(1000));
        }
    }
}
