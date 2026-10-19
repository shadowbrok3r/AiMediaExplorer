/// Percent-encode a path or query component; only unreserved characters pass through.
pub fn component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Inverse of [`component`]; malformed escapes are kept verbatim.
pub fn decode_component(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2]))
        {
            out.push(h << 4 | l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Seconds since the epoch for a `YYYY-MM-DDTHH:MM:SS...` stamp; any fraction and offset suffix is ignored (UTC assumed).
pub fn epoch_utc(stamp: &str) -> Option<i64> {
    let b = stamp.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' {
        return None;
    }
    let num = |a: usize, z: usize| stamp.get(a..z)?.parse::<i64>().ok();
    let (y, m, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (hh, mm, ss) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    Some(days_from_civil(y, m, d) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// `YYYY-MM-DDTHH:MM:SS+00:00` for epoch seconds, the stamp format the gateway uses.
pub fn rfc3339_utc(epoch: i64) -> String {
    let days = epoch.div_euclid(86_400);
    let secs = epoch.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00", secs / 3600, secs % 3600 / 60, secs % 60)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_encodes_everything_but_unreserved() {
        assert_eq!(component("abc-._~123"), "abc-._~123");
        assert_eq!(component("a b/c+d"), "a%20b%2Fc%2Bd");
        assert_eq!(component("ü"), "%C3%BC");
        assert_eq!(component(""), "");
    }

    #[test]
    fn component_round_trips() {
        for s in ["", "plain", "a b/c+d?e=f&g#h", "ünï/cødé 日本", "100%", "%41"] {
            assert_eq!(decode_component(&component(s)), s, "{s}");
        }
    }

    #[test]
    fn decode_keeps_malformed_escapes() {
        assert_eq!(decode_component("%zz%4"), "%zz%4");
        assert_eq!(decode_component("a%2"), "a%2");
        assert_eq!(decode_component("%41%42"), "AB");
    }

    #[test]
    fn rfc3339_round_trips_with_epoch() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00+00:00");
        assert_eq!(rfc3339_utc(951_868_800), "2000-03-01T00:00:00+00:00");
        assert_eq!(rfc3339_utc(1_789_163_509), "2026-09-11T21:51:49+00:00");
        for epoch in [0, 59, 86_399, 86_400, 951_782_400, 1_709_251_199, 1_789_268_293, 4_102_444_800] {
            assert_eq!(epoch_utc(&rfc3339_utc(epoch)), Some(epoch), "{epoch}");
        }
    }

    #[test]
    fn epoch_matches_known_instants() {
        assert_eq!(epoch_utc("1970-01-01T00:00:00+00:00"), Some(0));
        assert_eq!(epoch_utc("2000-03-01T00:00:00Z"), Some(951_868_800));
        assert_eq!(epoch_utc("2026-09-11T21:51:49.008980236+00:00"), Some(1_789_163_509));
        assert_eq!(epoch_utc("2026-09-11"), None);
        assert_eq!(epoch_utc("2026-13-11T00:00:00Z"), None);
        assert_eq!(epoch_utc("garbage-garbage-garbage"), None);
    }
}
