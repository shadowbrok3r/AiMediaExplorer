//! UI-free row model: row store, view index, selection and path keys.

pub mod filter;
pub mod filter_match;
pub mod filter_probe;
pub mod key;
pub mod selection;
pub mod store;
pub mod thumbnail;
pub mod view;

#[cfg(test)]
mod filter_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
pub mod testing;

pub use filter::{Chip, ChipKind, DEFAULT_MIN_SIZE, DateRange, FilterSpec, SizeRange, Tri, TypeSet};
pub use filter_match::{Condition, ConditionCount, FactSource, HiddenReport, NoFacts, PreparedFilter, Verdict};
pub use filter_probe::{Probe, ProbeCache, ProbeJob};
pub use key::{KeyPolicy, PathKey, Scheme};
pub use selection::{Selection, SelectionStats, Step, ViewRef};
pub use store::{Keyed, LoadSummary, MemoryEstimate, RowMut, RowStore, Upsert, Versions};
pub use view::{HIDDEN, Refresh, RefreshCounts, RowFacts, RowKind, SortKey, SortKeyFn, SortSpec, SortValue, ViewIndex, ViewInputs, ViewStats};
