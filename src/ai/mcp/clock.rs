//! Time source of the MCP server, replaceable in tests.

use std::time::Instant;

pub trait Clock: Send + Sync + 'static {
    /// Monotonic time for rate limits and timeouts.
    fn now(&self) -> Instant;
    /// Wall-clock milliseconds since the Unix epoch for records and logs.
    fn unix_ms(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn unix_ms(&self) -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
    }
}

/// A clock that only moves when told to.
#[cfg(test)]
pub struct FakeClock {
    base: Instant,
    elapsed_ms: std::sync::atomic::AtomicU64,
    unix_base_ms: u64,
}

#[cfg(test)]
impl FakeClock {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { base: Instant::now(), elapsed_ms: 0.into(), unix_base_ms: 1_760_000_000_000 })
    }

    pub fn advance(&self, by: std::time::Duration) {
        self.elapsed_ms.fetch_add(by.as_millis() as u64, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.base + std::time::Duration::from_millis(self.elapsed_ms.load(std::sync::atomic::Ordering::SeqCst))
    }

    fn unix_ms(&self) -> u64 {
        self.unix_base_ms + self.elapsed_ms.load(std::sync::atomic::Ordering::SeqCst)
    }
}
