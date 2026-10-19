//! Process meters read from the operating system; every reading is `None` where the platform gives none.

use std::time::Duration;

/// Resident set size in bytes.
pub fn rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_statm_rss(&std::fs::read_to_string("/proc/self/statm").ok()?, page_size())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Resident memory split into heap and stack pages (anonymous) and pages of mapped files, in bytes.
pub fn rss_split() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        Some((parse_status_kib(&text, "RssAnon:")?, parse_status_kib(&text, "RssFile:")?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Highest resident set size so far in bytes.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_status_kib(&std::fs::read_to_string("/proc/self/status").ok()?, "VmHWM:")
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// CPU time the process has used, user plus system, in seconds.
pub fn cpu_seconds() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        parse_stat_cpu_ticks(&std::fs::read_to_string("/proc/self/stat").ok()?).map(|ticks| ticks as f64 / clock_ticks() as f64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Time since the process was started.
pub fn process_age() -> Option<Duration> {
    #[cfg(target_os = "linux")]
    {
        let started = parse_stat_start_ticks(&std::fs::read_to_string("/proc/self/stat").ok()?)?;
        let uptime = std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse::<f64>().ok()?;
        let age = uptime - started as f64 / clock_ticks() as f64;
        (age >= 0.0).then(|| Duration::from_secs_f64(age))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Bytes the allocator hands out right now.
pub fn heap_in_use() -> Option<u64> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallinfo2 only reads allocator statistics.
        let info = unsafe { libc::mallinfo2() };
        Some(info.uordblks as u64 + info.hblkhd as u64)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size > 0 { size as u64 } else { 4096 }
}

#[cfg(target_os = "linux")]
fn clock_ticks() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks > 0 { ticks as u64 } else { 100 }
}

/// Resident pages of `/proc/self/statm` as bytes.
pub fn parse_statm_rss(text: &str, page: u64) -> Option<u64> {
    text.split_whitespace().nth(1)?.parse::<u64>().ok().map(|pages| pages * page)
}

/// A `Name:   123 kB` line of `/proc/self/status` as bytes.
pub fn parse_status_kib(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(key))?;
    line[key.len()..].split_whitespace().next()?.parse::<u64>().ok().map(|kib| kib * 1024)
}

// The fields of `/proc/self/stat` after the command name, which may hold spaces and parentheses.
fn stat_fields(text: &str) -> Option<Vec<&str>> {
    let after = &text[text.rfind(')')? + 1..];
    Some(after.split_whitespace().collect())
}

/// User plus system CPU ticks of `/proc/self/stat`.
pub fn parse_stat_cpu_ticks(text: &str) -> Option<u64> {
    let fields = stat_fields(text)?;
    let user: u64 = fields.get(11)?.parse().ok()?;
    let system: u64 = fields.get(12)?.parse().ok()?;
    Some(user + system)
}

/// Start time of the process in clock ticks after boot.
pub fn parse_stat_start_ticks(text: &str) -> Option<u64> {
    stat_fields(text)?.get(19)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAT: &str = "4242 (smart (media) x) S 1 4242 4242 0 -1 4194560 1000 0 0 0 150 50 0 0 20 0 30 0 987654 123456789 5000 18446744073709551615";

    #[test]
    fn cpu_ticks_are_user_plus_system_even_with_parentheses_in_the_name() {
        assert_eq!(parse_stat_cpu_ticks(STAT), Some(200));
    }

    #[test]
    fn the_start_tick_is_the_twenty_second_field() {
        assert_eq!(parse_stat_start_ticks(STAT), Some(987654));
    }

    #[test]
    fn garbage_gives_none() {
        assert_eq!(parse_stat_cpu_ticks("nonsense"), None);
        assert_eq!(parse_stat_start_ticks("1 (x) S 1"), None);
        assert_eq!(parse_statm_rss("1", 4096), None);
        assert_eq!(parse_status_kib("VmRSS: abc kB", "VmRSS:"), None);
    }

    #[test]
    fn statm_and_status_readings_are_bytes() {
        assert_eq!(parse_statm_rss("100 25 10 1 0 5 0", 4096), Some(25 * 4096));
        assert_eq!(parse_status_kib("Name: x\nVmHWM:\t  2048 kB\nVmRSS: 1 kB", "VmHWM:"), Some(2048 * 1024));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_reports_plausible_meters() {
        let (anon, file) = rss_split().unwrap();
        assert!(anon > 0 && file > 0 && anon + file <= rss_bytes().unwrap() * 2);
        assert!(rss_bytes().unwrap() > 1024 * 1024);
        assert!(peak_rss_bytes().unwrap() >= rss_bytes().unwrap() / 2);
        assert!(cpu_seconds().unwrap() >= 0.0);
        assert!(process_age().unwrap() < Duration::from_secs(3600 * 24));
    }
}
