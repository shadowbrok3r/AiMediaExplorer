//! Width planning for the status bar: items shrink to compact forms, then collapse into an overflow menu, lowest priority first.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Full,
    Compact,
}

/// One bar item's measured widths; entries are in display order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub priority: u8,
    pub full: f32,
    pub compact: Option<f32>,
}

/// Which items are shown at which width, and whether the rest moved to the overflow menu.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub modes: Vec<Option<Mode>>,
    pub overflow: bool,
}

fn plan_width(entries: &[Entry], modes: &[Option<Mode>], gap: f32, overflow_w: f32) -> f32 {
    let mut total = 0.0;
    let mut shown = 0;
    for (e, m) in entries.iter().zip(modes) {
        let Some(mode) = m else { continue };
        total += match mode {
            Mode::Full => e.full,
            Mode::Compact => e.compact.unwrap_or(e.full),
        };
        shown += 1;
    }
    if shown > 1 {
        total += gap * (shown - 1) as f32;
    }
    if modes.iter().any(Option::is_none) {
        total += overflow_w + if shown > 0 { gap } else { 0.0 };
    }
    total
}

/// Shrinks items to compact form lowest priority first, then collapses the lowest priority items into the overflow menu.
pub fn plan(entries: &[Entry], gap: f32, overflow_w: f32, avail: f32) -> Plan {
    let mut modes: Vec<Option<Mode>> = vec![Some(Mode::Full); entries.len()];
    while plan_width(entries, &modes, gap, overflow_w) > avail {
        let next = entries
            .iter()
            .enumerate()
            .filter(|(i, e)| modes[*i] == Some(Mode::Full) && e.compact.is_some_and(|c| c < e.full))
            .min_by_key(|(_, e)| e.priority)
            .map(|(i, _)| i);
        match next {
            Some(i) => modes[i] = Some(Mode::Compact),
            None => break,
        }
    }
    while plan_width(entries, &modes, gap, overflow_w) > avail {
        let next = entries
            .iter()
            .enumerate()
            .filter(|(i, _)| modes[*i].is_some())
            .min_by_key(|(_, e)| e.priority)
            .map(|(i, _)| i);
        match next {
            Some(i) => modes[i] = None,
            None => break,
        }
    }
    let overflow = modes.iter().any(Option::is_none);
    Plan { modes, overflow }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<Entry> {
        vec![
            Entry { priority: 100, full: 150.0, compact: Some(50.0) },
            Entry { priority: 80, full: 70.0, compact: Some(30.0) },
            Entry { priority: 60, full: 130.0, compact: Some(60.0) },
            Entry { priority: 40, full: 160.0, compact: Some(60.0) },
            Entry { priority: 30, full: 160.0, compact: Some(60.0) },
        ]
    }

    #[test]
    fn everything_stays_full_when_it_fits() {
        let p = plan(&entries(), 8.0, 28.0, 2000.0);
        assert!(p.modes.iter().all(|m| *m == Some(Mode::Full)));
        assert!(!p.overflow);
    }

    #[test]
    fn meters_shrink_before_anything_collapses_lowest_priority_first() {
        let es = entries();
        let full: f32 = es.iter().map(|e| e.full).sum::<f32>() + 8.0 * 4.0;
        let p = plan(&es, 8.0, 28.0, full - 10.0);
        assert_eq!(p.modes[4], Some(Mode::Compact), "lowest priority shrinks first");
        assert_eq!(p.modes[0], Some(Mode::Full), "highest priority untouched");
        assert!(!p.overflow);
    }

    #[test]
    fn the_lowest_priority_items_collapse_into_the_overflow_once_all_are_compact() {
        let es = entries();
        let compact_total: f32 = es.iter().map(|e| e.compact.unwrap()).sum::<f32>() + 8.0 * 4.0;
        let p = plan(&es, 8.0, 28.0, compact_total - 1.0);
        assert!(p.overflow);
        assert_eq!(p.modes[4], None);
        assert_eq!(p.modes[0], Some(Mode::Compact));
    }

    #[test]
    fn a_plan_never_exceeds_the_available_width_while_anything_can_still_collapse() {
        let es = entries();
        for avail in (80..=900).step_by(7) {
            let avail = avail as f32;
            let p = plan(&es, 8.0, 28.0, avail);
            let width = plan_width(&es, &p.modes, 8.0, 28.0);
            let shown = p.modes.iter().filter(|m| m.is_some()).count();
            assert!(width <= avail || shown == 0, "avail {avail}: width {width} with {shown} shown");
        }
    }

    #[test]
    fn wider_space_never_shows_fewer_items() {
        let es = entries();
        let mut last_shown = 0;
        for avail in (0..=1000).step_by(5) {
            let p = plan(&es, 8.0, 28.0, avail as f32);
            let shown = p.modes.iter().filter(|m| m.is_some()).count();
            assert!(shown >= last_shown, "avail {avail} showed {shown} after {last_shown}");
            last_shown = shown;
        }
    }
}
