//! Randomized operation sequences checked against plain `Vec` models of the store, the view and the selection.

use super::key::{KeyPolicy, PathKey};
use super::selection::{Selection, Step, ViewRef};
use super::store::RowStore;
use super::testing::{TestRow, sort_key};
use super::view::{RefreshCounts, RowKind, SortKey, SortSpec, ViewIndex, ViewInputs};

/// Small deterministic generator so failures reproduce from the seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

const UNIVERSE: usize = 48;

fn path(i: usize) -> String {
    format!("/u/{i:02}.jpg")
}

fn random_row(rng: &mut Rng) -> TestRow {
    let kinds = [RowKind::Image, RowKind::Video, RowKind::Dir, RowKind::Archive, RowKind::Other];
    TestRow::new(&path(rng.below(UNIVERSE)))
        .size(rng.below(12) as u64)
        .name(&format!("n{}", rng.below(6)))
        .kind(kinds[rng.below(kinds.len())])
        .tag(if rng.chance(30) { None } else { Some(rng.below(5) as i64) })
}

/// The store as a plain list: replace on a repeated path, append otherwise.
#[derive(Default)]
struct ModelStore {
    rows: Vec<TestRow>,
}

impl ModelStore {
    fn upsert(&mut self, row: TestRow) {
        if row.path.is_empty() {
            return;
        }
        match self.rows.iter().position(|r| r.path == row.path) {
            Some(i) => self.rows[i] = row,
            None => self.rows.push(row),
        }
    }
}

fn random_sort(rng: &mut Rng) -> SortSpec {
    let n = rng.below(4);
    SortSpec::from_keys((0..n).map(|_| SortKey { column: rng.below(5) as u8, descending: rng.chance(50) }).collect())
}

fn expected_order(model: &[TestRow], spec: &SortSpec, keep: impl Fn(&TestRow) -> bool) -> Vec<String> {
    let mut visible: Vec<(usize, &TestRow)> = model.iter().enumerate().filter(|(_, r)| keep(r)).collect();
    visible.sort_by(|(ia, a), (ib, b)| {
        for key in spec.keys() {
            let ord = sort_key(a, key.column).compare(&sort_key(b, key.column));
            let ord = if key.descending { ord.reverse() } else { ord };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        ia.cmp(ib)
    });
    visible.into_iter().map(|(_, r)| r.path.clone()).collect()
}

fn run_store_and_view(seed: u64, steps: usize) -> RefreshCounts {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut store: RowStore<TestRow> = RowStore::with_policy(KeyPolicy::Exact);
    let mut model = ModelStore::default();
    let mut view = ViewIndex::new();
    let mut modulus = 0usize;
    let mut sig = 0u64;
    for step in 0..steps {
        match rng.below(14) {
            0..=4 => {
                let row = random_row(&mut rng);
                model.upsert(row.clone());
                store.upsert(row);
            }
            5 => {
                let p = path(rng.below(UNIVERSE));
                model.rows.retain(|r| r.path != p);
                store.remove_path(&p);
            }
            6 => {
                let m = 2 + rng.below(4);
                let r = rng.below(m);
                model.rows.retain(|row| row.size as usize % m != r);
                store.retain(|row| row.size as usize % m != r);
            }
            7 if rng.chance(20) => {
                model.rows.clear();
                store.clear();
            }
            8 => {
                let rows: Vec<TestRow> = (0..rng.below(30)).map(|_| random_row(&mut rng)).collect();
                for row in &rows {
                    model.upsert(row.clone());
                }
                store.bulk_load(rows);
            }
            9 if !model.rows.is_empty() => {
                let i = rng.below(model.rows.len());
                let size = rng.below(12) as u64;
                model.rows[i].size = size;
                store.get_mut(i).unwrap().size = size;
            }
            10 if !model.rows.is_empty() => {
                let i = rng.below(model.rows.len());
                let target = path(rng.below(UNIVERSE));
                if model.rows[i].path != target {
                    let mut row = model.rows.remove(i);
                    row.path = target.clone();
                    model.upsert(row);
                }
                store.get_mut(i).unwrap().path = target;
            }
            11 => {
                model.rows.iter_mut().for_each(|r| r.size += 1);
                store.for_each_mut(|r| r.size += 1);
            }
            12 => {
                view.set_sort(random_sort(&mut rng));
            }
            13 => {
                modulus = rng.below(4);
                sig += 1;
            }
            _ => {}
        }
        store.check_invariants().unwrap_or_else(|e| panic!("seed {seed} step {step}: {e}"));
        assert_eq!(store.len(), model.rows.len(), "seed {seed} step {step}");
        for (i, row) in model.rows.iter().enumerate() {
            assert_eq!(store.get(i), Some(row), "seed {seed} step {step} row {i}");
            assert_eq!(store.position(&row.path), Some(i));
        }
        if rng.chance(55) {
            let keep = move |r: &TestRow| modulus == 0 || (r.size as usize) % (modulus + 1) != 0;
            let inputs = ViewInputs { filter_sig: sig, data_sig: 0, filter: &keep, sort_key: &sort_key };
            view.refresh(&store, &inputs);
            let want = expected_order(&model.rows, view.sort(), keep);
            let got: Vec<String> = view.rows(&store).map(|r| r.path.clone()).collect();
            assert_eq!(got, want, "seed {seed} step {step}: view order");
            view.check_invariants(&store, &inputs).unwrap_or_else(|e| panic!("seed {seed} step {step}: {e}"));
            let stats = view.stats();
            assert_eq!(stats.total, model.rows.len());
            assert_eq!(stats.visible, want.len());
            assert_eq!(stats.hidden_by_filter, model.rows.len() - want.len());
            let size: u64 = model.rows.iter().filter(|r| keep(r) && r.kind != RowKind::Dir).map(|r| r.size).sum();
            assert_eq!(stats.total_size, size, "seed {seed} step {step}");
            let count = |k: RowKind| model.rows.iter().filter(|r| keep(r) && r.kind == k).count();
            assert_eq!((stats.images, stats.videos, stats.dirs, stats.archives, stats.other), (count(RowKind::Image), count(RowKind::Video), count(RowKind::Dir), count(RowKind::Archive), count(RowKind::Other)));
        }
    }
    view.counts()
}

fn total(counts: impl Iterator<Item = RefreshCounts>) -> RefreshCounts {
    counts.fold(RefreshCounts::default(), |a, c| RefreshCounts { unchanged: a.unchanged + c.unchanged, appended: a.appended + c.appended, rebuilt: a.rebuilt + c.rebuilt })
}

#[test]
fn random_store_and_view_operations_match_the_model() {
    let counts = total((1..=300u64).map(|seed| run_store_and_view(seed, 160)));
    assert!(counts.appended > 500 && counts.rebuilt > 500 && counts.unchanged > 100, "the runs must exercise every refresh path: {counts:?}");
}

#[test]
fn long_random_runs_keep_incremental_and_full_refreshes_identical() {
    let counts = total((1000..1006u64).map(|seed| run_store_and_view(seed, 1500)));
    assert!(counts.appended > 200 && counts.rebuilt > 200, "{counts:?}");
}

/// Selection state as plain data: ordered paths, anchor and focus.
#[derive(Clone, Debug, Default, PartialEq)]
struct ModelSel {
    order: Vec<String>,
    anchor: Option<String>,
    focus: Option<String>,
}

impl ModelSel {
    fn pos(view: &[String], p: &Option<String>) -> Option<usize> {
        p.as_ref().and_then(|p| view.iter().position(|v| v == p))
    }

    fn click(&mut self, p: &str) {
        *self = ModelSel { order: vec![p.to_string()], anchor: Some(p.to_string()), focus: Some(p.to_string()) };
    }

    fn toggle(&mut self, p: &str) {
        match self.order.iter().position(|s| s == p) {
            Some(i) => {
                self.order.remove(i);
            }
            None => self.order.push(p.to_string()),
        }
        self.anchor = Some(p.to_string());
        self.focus = Some(p.to_string());
    }

    fn range(&mut self, view: &[String], target: &str, additive: bool) {
        let Some(t) = view.iter().position(|v| v == target) else { return };
        let (a, anchor) = match Self::pos(view, &self.anchor) {
            Some(a) => (a, self.anchor.clone()),
            None => match Self::pos(view, &self.focus) {
                Some(f) => (f, self.focus.clone()),
                None => (t, Some(target.to_string())),
            },
        };
        let span: Vec<String> = view[a.min(t)..=a.max(t)].to_vec();
        if additive {
            for s in span {
                if !self.order.contains(&s) {
                    self.order.push(s);
                }
            }
        } else {
            self.order = span;
        }
        self.anchor = anchor;
        self.focus = Some(target.to_string());
    }

    fn all(&mut self, view: &[String]) {
        let first = view.first().cloned();
        let keep = |p: &Option<String>| p.clone().filter(|p| view.contains(p));
        self.anchor = keep(&self.anchor).or(first.clone());
        self.focus = keep(&self.focus).or(first);
        self.order = view.to_vec();
    }

    fn invert(&mut self, view: &[String]) {
        let keep = |p: &Option<String>| p.clone().filter(|p| view.contains(p));
        self.anchor = keep(&self.anchor);
        self.focus = keep(&self.focus);
        self.order = view.iter().filter(|v| !self.order.contains(v)).cloned().collect();
    }

    fn prune(&mut self, view: &[String]) {
        self.order.retain(|p| view.contains(p));
        let keep = |p: &Option<String>| p.clone().filter(|p| view.contains(p));
        self.anchor = keep(&self.anchor);
        self.focus = keep(&self.focus);
    }

    fn step(&mut self, view: &[String], step: Step, extend: bool) {
        if view.is_empty() {
            return;
        }
        let last = view.len() - 1;
        let target = match (step, Self::pos(view, &self.focus)) {
            (Step::Home, _) => 0,
            (Step::End, _) => last,
            (Step::Down(_), None) => 0,
            (Step::Up(_), None) => last,
            (Step::Down(n), Some(c)) => (c + n).min(last),
            (Step::Up(n), Some(c)) => c.saturating_sub(n),
        };
        let p = view[target].clone();
        if extend {
            self.range(view, &p, false);
        } else {
            self.click(&p);
        }
    }
}

fn model_of(sel: &Selection) -> ModelSel {
    ModelSel {
        order: sel.iter().map(|k| k.as_str().to_string()).collect(),
        anchor: sel.anchor().map(|k| k.as_str().to_string()),
        focus: sel.focus().map(|k| k.as_str().to_string()),
    }
}

fn run_selection(seed: u64, steps: usize) {
    let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
    let mut store: RowStore<TestRow> = RowStore::with_policy(KeyPolicy::Exact);
    store.bulk_load((0..UNIVERSE).map(|i| TestRow::new(&path(i)).size(rng.below(10) as u64).name(&format!("n{}", rng.below(5)))));
    let mut view = ViewIndex::new();
    let mut sel = Selection::new();
    let mut model = ModelSel::default();
    let mut hide = 0usize;
    let mut sig = 0u64;
    for step in 0..steps {
        let keep = move |r: &TestRow| hide == 0 || (r.size as usize) % (hide + 1) != 0;
        let inputs = ViewInputs { filter_sig: sig, data_sig: 0, filter: &keep, sort_key: &sort_key };
        view.refresh(&store, &inputs);
        let names: Vec<String> = view.rows(&store).map(|r| r.path.clone()).collect();
        let pick = |rng: &mut Rng| -> Option<String> { if names.is_empty() { None } else { Some(names[rng.below(names.len())].clone()) } };
        let key_of = |p: &str| PathKey::with_policy(p, KeyPolicy::Exact);
        let v = ViewRef::new(&store, &view);
        match rng.below(12) {
            0 | 1 => {
                if let Some(p) = pick(&mut rng) {
                    sel.click(key_of(&p));
                    model.click(&p);
                }
            }
            2 | 3 => {
                if let Some(p) = pick(&mut rng) {
                    sel.toggle(key_of(&p));
                    model.toggle(&p);
                }
            }
            4 | 5 => {
                if let Some(p) = pick(&mut rng) {
                    let additive = rng.chance(40);
                    sel.select_range(&v, &key_of(&p), additive);
                    model.range(&names, &p, additive);
                }
            }
            6 => {
                sel.select_all(&v);
                model.all(&names);
            }
            7 => {
                sel.invert(&v);
                model.invert(&names);
            }
            8 => {
                let n = 1 + rng.below(9);
                let (s, extend) = (if rng.chance(50) { Step::Down(n) } else { Step::Up(n) }, rng.chance(40));
                sel.step_focus(&v, s, extend);
                model.step(&names, s, extend);
            }
            9 => {
                let s = if rng.chance(50) { Step::Home } else { Step::End };
                let extend = rng.chance(40);
                sel.step_focus(&v, s, extend);
                model.step(&names, s, extend);
            }
            10 => {
                sel.clear();
                model.order.clear();
            }
            _ => {
                hide = rng.below(4);
                sig += 1;
            }
        }
        let keep = move |r: &TestRow| hide == 0 || (r.size as usize) % (hide + 1) != 0;
        let inputs = ViewInputs { filter_sig: sig, data_sig: 0, filter: &keep, sort_key: &sort_key };
        view.refresh(&store, &inputs);
        let names: Vec<String> = view.rows(&store).map(|r| r.path.clone()).collect();
        if rng.chance(50) {
            let v = ViewRef::new(&store, &view);
            sel.prune(&v);
            model.prune(&names);
            assert!(sel.iter().all(|k| v.is_visible(k)), "seed {seed} step {step}: pruned selection holds a hidden row");
        }
        assert_eq!(model_of(&sel), model, "seed {seed} step {step}");
        let v = ViewRef::new(&store, &view);
        let rows = sel.rows(&v);
        let visible_selected = model.order.iter().filter(|p| names.contains(p)).count();
        assert_eq!(rows.len(), visible_selected, "seed {seed} step {step}: rows() returns the visible selected rows");
        if sel.is_empty() {
            assert!(rows.is_empty() && sel.store_indices(&v).is_empty(), "seed {seed} step {step}: an empty selection selected rows");
        }
        let positions: Vec<usize> = rows.iter().map(|r| names.iter().position(|n| *n == r.path).unwrap()).collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "seed {seed} step {step}: selected rows are in view order");
    }
}

#[test]
fn random_selection_operations_match_the_model() {
    for seed in 1..=200u64 {
        run_selection(seed, 200);
    }
}
