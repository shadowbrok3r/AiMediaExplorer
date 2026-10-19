//! Allocation counters kept per thread by the counting allocator of the `alloc-stats` feature.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Counter slots; the interface thread owns slot 0, every other thread shares the rest.
const SLOTS: usize = 64;

#[repr(align(64))]
struct Slot {
    allocs: AtomicU64,
    bytes: AtomicU64,
}

static SHARDS: [Slot; SLOTS] = [const { Slot { allocs: AtomicU64::new(0), bytes: AtomicU64::new(0) } }; SLOTS];
static NEXT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MINE: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn my_slot() -> usize {
    MINE.with(|mine| {
        let slot = mine.get();
        if slot != usize::MAX {
            return slot;
        }
        let slot = 1 + NEXT.fetch_add(1, Ordering::Relaxed) % (SLOTS - 1);
        mine.set(slot);
        slot
    })
}

/// Allocations and bytes handed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub allocs: u64,
    pub bytes: u64,
}

impl Counts {
    pub fn since(&self, earlier: &Counts) -> Counts {
        Counts { allocs: self.allocs.saturating_sub(earlier.allocs), bytes: self.bytes.saturating_sub(earlier.bytes) }
    }
}

/// Whether the build counts allocations.
pub const fn enabled() -> bool {
    cfg!(feature = "alloc-stats")
}

/// Makes the calling thread the interface thread whose allocations [`ui_thread`] reports.
pub fn mark_ui_thread() {
    MINE.with(|mine| mine.set(0));
}

fn read(slot: &Slot) -> Counts {
    Counts { allocs: slot.allocs.load(Ordering::Relaxed), bytes: slot.bytes.load(Ordering::Relaxed) }
}

/// Allocations of the interface thread so far.
pub fn ui_thread() -> Counts {
    read(&SHARDS[0])
}

/// Allocations of every thread so far.
pub fn total() -> Counts {
    SHARDS.iter().map(read).fold(Counts::default(), |a, b| Counts { allocs: a.allocs + b.allocs, bytes: a.bytes + b.bytes })
}

/// Counts one allocation of `bytes` on the calling thread.
pub fn note(bytes: usize) {
    let slot = &SHARDS[my_slot()];
    slot.allocs.fetch_add(1, Ordering::Relaxed);
    slot.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Global allocator that counts and forwards to the system allocator.
#[cfg(feature = "alloc-stats")]
pub struct Counting;

#[cfg(feature = "alloc-stats")]
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        note(layout.size());
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        note(layout.size());
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_lands_in_the_slot_of_its_thread_and_in_the_total() {
        let before = total();
        std::thread::spawn(|| {
            note(100);
            note(28);
        })
        .join()
        .unwrap();
        let delta = total().since(&before);
        assert!(delta.allocs >= 2 && delta.bytes >= 128, "{delta:?}");
    }

    #[test]
    fn the_interface_thread_has_its_own_counter() {
        let ui_before = ui_thread();
        let handle = std::thread::spawn(|| {
            mark_ui_thread();
            note(10);
            note(10);
            note(10);
            ui_thread()
        });
        let seen = handle.join().unwrap();
        assert!(seen.since(&ui_before).allocs >= 3);
        let others_before = total().since(&ui_thread());
        std::thread::spawn(|| note(1_000_000)).join().unwrap();
        let others_after = total().since(&ui_thread());
        assert!(others_after.bytes >= others_before.bytes + 1_000_000, "other threads stay out of the interface counter");
    }

    #[test]
    fn since_never_goes_negative() {
        let a = Counts { allocs: 5, bytes: 50 };
        let b = Counts { allocs: 9, bytes: 90 };
        assert_eq!(a.since(&b), Counts::default());
        assert_eq!(b.since(&a), Counts { allocs: 4, bytes: 40 });
    }

    #[test]
    fn the_build_reports_whether_it_counts() {
        assert_eq!(enabled(), cfg!(feature = "alloc-stats"));
    }
}
