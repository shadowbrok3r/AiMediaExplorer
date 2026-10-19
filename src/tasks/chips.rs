//! Aggregation of running tasks into the activity chips of the status bar.

use std::time::Instant;

use super::model::{Task, TaskKind};

/// One chip: all visible active tasks of a kind.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivityChip {
    pub kind: TaskKind,
    pub count: usize,
    pub done: u64,
    pub total: u64,
    pub fraction: Option<f32>,
    /// True when every task of the chip waits for a worker.
    pub queued: bool,
    /// Text of the full chip, for example `Describe 12/340`.
    pub label: String,
    /// Text of the narrow chip, next to the kind's icon.
    pub short: String,
}

fn percent(fraction: f32) -> String {
    format!("{:.0}%", fraction * 100.0)
}

fn full_label(kind: TaskKind, count: usize, done: u64, total: u64, fraction: Option<f32>) -> String {
    let name = match (kind, count) {
        (TaskKind::Scan, n) if n > 1 => format!("Scans ({n})"),
        (TaskKind::ModelLoad, n) if n > 1 => format!("Loading models ({n})"),
        (TaskKind::ModelLoad, _) => "Loading model".to_string(),
        (_, n) if n > 1 => format!("{} ({n})", kind.label()),
        _ => kind.label().to_string(),
    };
    match (fraction, kind.counts_items()) {
        (Some(_), true) => format!("{name} {done}/{total}"),
        (Some(f), false) => format!("{name} {}", percent(f)),
        (None, _) => name,
    }
}

fn short_label(kind: TaskKind, count: usize, done: u64, total: u64, fraction: Option<f32>) -> String {
    match (fraction, kind.counts_items()) {
        (Some(_), true) => format!("{done}/{total}"),
        (Some(f), false) => percent(f),
        (None, _) if count > 1 => count.to_string(),
        (None, _) => String::new(),
    }
}

/// Chips for the active tasks old enough to show, in [`TaskKind::ALL`] order.
pub fn activity_chips(tasks: &[Task], now: Instant) -> Vec<ActivityChip> {
    TaskKind::ALL
        .into_iter()
        .filter_map(|kind| {
            let of_kind: Vec<&Task> = tasks.iter().filter(|t| t.kind == kind && t.is_visible(now)).collect();
            if of_kind.is_empty() {
                return None;
            }
            let count = of_kind.len();
            let (done, total) = of_kind.iter().filter(|t| t.total > 0).fold((0, 0), |(d, t), task| (d + task.done.min(task.total), t + task.total));
            let fraction = (total > 0).then(|| (done as f32 / total as f32).clamp(0.0, 1.0));
            Some(ActivityChip {
                kind,
                count,
                done,
                total,
                fraction,
                queued: of_kind.iter().all(|t| t.state == super::model::TaskState::Queued),
                label: full_label(kind, count, done, total, fraction),
                short: short_label(kind, count, done, total, fraction),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::registry::TaskRegistry;
    use crate::tasks::model::TaskSpec;
    use std::time::Duration;

    fn tasks_of(specs: Vec<(TaskSpec, u64)>) -> (Vec<Task>, Instant) {
        let mut r = TaskRegistry::new();
        let now = Instant::now();
        for (spec, done) in specs {
            let total = spec.total;
            let id = r.begin_at(spec, now);
            r.set_progress(id, done, total);
        }
        (r.tasks().to_vec(), now)
    }

    #[test]
    fn counted_kinds_show_items_and_others_a_percentage() {
        let (tasks, now) = tasks_of(vec![
            (TaskSpec::new(TaskKind::Describe, "d").total(340), 12),
            (TaskSpec::new(TaskKind::Scan, "s").total(200), 84),
        ]);
        let chips = activity_chips(&tasks, now);
        let labels: Vec<_> = chips.iter().map(|c| (c.kind, c.label.as_str(), c.short.as_str())).collect();
        assert_eq!(labels, vec![(TaskKind::Scan, "Scan 42%", "42%"), (TaskKind::Describe, "Describe 12/340", "12/340")]);
    }

    #[test]
    fn several_tasks_of_one_kind_merge_into_one_chip() {
        let (tasks, now) = tasks_of(vec![
            (TaskSpec::new(TaskKind::Scan, "a").total(100), 50),
            (TaskSpec::new(TaskKind::Scan, "b").total(100), 100),
        ]);
        let chips = activity_chips(&tasks, now);
        assert_eq!(chips.len(), 1);
        assert_eq!((chips[0].count, chips[0].label.as_str()), (2, "Scans (2) 75%"));
    }

    #[test]
    fn tasks_without_a_total_show_just_their_name() {
        let (tasks, now) = tasks_of(vec![(TaskSpec::new(TaskKind::Refine, "r"), 0), (TaskSpec::new(TaskKind::ModelLoad, "m"), 0), (TaskSpec::new(TaskKind::ModelLoad, "n"), 0)]);
        let chips = activity_chips(&tasks, now);
        assert_eq!(chips.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(), vec!["Refine", "Loading models (2)"]);
        assert_eq!(chips[1].short, "2");
        assert_eq!(chips[0].short, "");
    }

    #[test]
    fn young_and_finished_tasks_make_no_chip() {
        let (mut tasks, now) = tasks_of(vec![
            (TaskSpec::new(TaskKind::Database, "db").visible_after(Duration::from_secs(1)), 0),
            (TaskSpec::new(TaskKind::Clip, "c"), 0),
        ]);
        assert_eq!(activity_chips(&tasks, now).iter().map(|c| c.kind).collect::<Vec<_>>(), vec![TaskKind::Clip]);
        let later = now + Duration::from_secs(2);
        assert_eq!(activity_chips(&tasks, later).len(), 2);
        tasks[1].state = crate::tasks::model::TaskState::Completed;
        assert_eq!(activity_chips(&tasks, later).len(), 1);
    }

    #[test]
    fn the_chip_fraction_is_the_summed_share_of_tasks_with_a_total() {
        let (tasks, now) = tasks_of(vec![(TaskSpec::new(TaskKind::Backup, "copy").total(10), 3), (TaskSpec::new(TaskKind::Backup, "move"), 0)]);
        let chip = &activity_chips(&tasks, now)[0];
        assert_eq!((chip.done, chip.total), (3, 10));
        assert_eq!(chip.fraction, Some(0.3));
        assert_eq!(chip.label, "Copy (2) 3/10");
    }
}
