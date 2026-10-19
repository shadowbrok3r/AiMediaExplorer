//! Splitting rows into requests that stay under the gateway's input cap.

use std::ops::Range;

use zeroclaw_client::types::COMPLETE_INPUT_CAP;

/// Most rows one request carries unless the caller asks for fewer.
pub const MAX_ROWS: usize = 40;

/// Which rows go into which request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchPlan {
    /// Consecutive row ranges, in order.
    pub batches: Vec<Range<usize>>,
    /// Rows that cannot fit a request even alone.
    pub oversize: Vec<usize>,
}

impl BatchPlan {
    /// Rows that are in some batch.
    pub fn rows(&self) -> usize {
        self.batches.iter().map(|r| r.len()).sum()
    }
}

/// The row count per request that splits `total` rows evenly without passing `max`.
pub fn even_target(total: usize, max: usize) -> usize {
    let max = max.max(1);
    if total == 0 {
        return max;
    }
    total.div_ceil(total.div_ceil(max))
}

/// Groups rows of `sizes` bytes into even requests of at most `max_rows` rows and `cap` bytes with `fixed` bytes of instructions; a row too big alone goes to `oversize`.
pub fn plan_batches(sizes: &[usize], fixed: usize, cap: usize, max_rows: usize) -> BatchPlan {
    let per = even_target(sizes.len(), max_rows);
    let room = cap.saturating_sub(fixed);
    let mut plan = BatchPlan::default();
    let mut start = 0;
    let mut bytes = 0;
    for (i, &size) in sizes.iter().enumerate() {
        if size > room {
            if start < i {
                plan.batches.push(start..i);
            }
            plan.oversize.push(i);
            start = i + 1;
            bytes = 0;
            continue;
        }
        if i > start && (i - start >= per || bytes + size > room) {
            plan.batches.push(start..i);
            start = i;
            bytes = 0;
        }
        bytes += size;
    }
    if start < sizes.len() {
        plan.batches.push(start..sizes.len());
    }
    plan
}

/// [`plan_batches`] for the gateway's real cap.
pub fn plan_for_gateway(sizes: &[usize], fixed: usize) -> BatchPlan {
    plan_batches(sizes, fixed, COMPLETE_INPUT_CAP, MAX_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(plan: &BatchPlan) -> Vec<usize> {
        plan.batches.iter().map(|r| r.len()).collect()
    }

    #[test]
    fn counts_spread_evenly_under_the_row_limit() {
        assert_eq!(lens(&plan_batches(&[10; 30], 100, 64 * 1024, 40)), [30]);
        assert_eq!(lens(&plan_batches(&[10; 40], 100, 64 * 1024, 40)), [40]);
        assert_eq!(lens(&plan_batches(&[10; 41], 100, 64 * 1024, 40)), [21, 20]);
        assert_eq!(lens(&plan_batches(&[10; 100], 100, 64 * 1024, 40)), [34, 34, 32]);
        assert_eq!(lens(&plan_batches(&[10; 80], 100, 64 * 1024, 40)), [40, 40]);
        assert_eq!(lens(&plan_batches(&[10; 1], 100, 64 * 1024, 40)), [1]);
        assert!(plan_batches(&[], 100, 64 * 1024, 40).batches.is_empty());
    }

    #[test]
    fn the_byte_cap_splits_batches_before_the_row_limit() {
        let sizes = vec![1000; 50];
        let plan = plan_batches(&sizes, 2000, 10_000, 40);
        assert_eq!(plan.rows(), 50);
        assert!(plan.oversize.is_empty());
        for range in &plan.batches {
            let bytes: usize = sizes[range.clone()].iter().sum();
            assert!(2000 + bytes <= 10_000, "{range:?} takes {bytes}");
        }
        assert_eq!(plan.batches.iter().map(|r| r.len()).max(), Some(8));
    }

    #[test]
    fn every_row_lands_exactly_once_and_in_order() {
        let sizes: Vec<usize> = (0..137).map(|i| 200 + (i * 37) % 900).collect();
        let plan = plan_batches(&sizes, 1500, 9000, 25);
        let mut next = 0;
        for range in &plan.batches {
            assert_eq!(range.start, next);
            assert!(range.len() <= 25);
            assert!(1500 + sizes[range.clone()].iter().sum::<usize>() <= 9000);
            next = range.end;
        }
        assert_eq!(next, 137);
    }

    #[test]
    fn a_row_too_big_for_any_request_is_set_aside() {
        let sizes = [100, 100, 9_000, 100, 100];
        let plan = plan_batches(&sizes, 500, 5_000, 40);
        assert_eq!(plan.oversize, [2]);
        assert_eq!(plan.batches, [0..2, 3..5]);
        let alone = plan_batches(&[6_000], 500, 5_000, 40);
        assert_eq!((alone.batches.len(), alone.oversize.clone()), (0, vec![0]));
        let none_fit = plan_batches(&[10, 10], 5_000, 5_000, 40);
        assert_eq!(none_fit.oversize, [0, 1]);
    }

    #[test]
    fn the_gateway_cap_is_the_default() {
        let plan = plan_for_gateway(&vec![2_000; 100], 3_000);
        assert!(plan.oversize.is_empty());
        for range in &plan.batches {
            assert!(3_000 + 2_000 * range.len() <= COMPLETE_INPUT_CAP);
            assert!(range.len() <= MAX_ROWS);
        }
        assert_eq!(plan.rows(), 100);
        assert_eq!(even_target(0, 40), 40);
        assert_eq!(even_target(7, 0), 1);
    }
}
