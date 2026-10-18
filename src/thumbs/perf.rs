//! Test-only counting allocator and checks that the steady-state request path allocates nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Allocator that counts, on the calling thread only, while a [`measure`] is active.
pub struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn count(bytes: usize) {
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
            let _ = BYTES.try_with(|b| b.set(b.get() + bytes as u64));
        }
    });
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub allocs: u64,
    pub bytes: u64,
}

/// Runs `f` and returns what it allocated on this thread.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, Counts) {
    ALLOCS.with(|a| a.set(0));
    BYTES.with(|b| b.set(0));
    ON.with(|on| on.set(true));
    let result = f();
    ON.with(|on| on.set(false));
    (result, Counts { allocs: ALLOCS.with(Cell::get), bytes: BYTES.with(Cell::get) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::core::{Core, CoreConfig, DecodeResult, Lookup};
    use super::super::key::Tier;

    #[test]
    fn the_counter_sees_allocations_on_this_thread_only() {
        let (_, c) = measure(|| {
            let v: Vec<u8> = Vec::with_capacity(1000);
            std::hint::black_box(v)
        });
        assert!(c.allocs >= 1 && c.bytes >= 1000);
        let (_, none) = measure(|| 1 + 1);
        assert_eq!(none, Counts::default());
        let (_, other) = measure(|| std::thread::spawn(|| std::hint::black_box(vec![0u8; 4_000_000]).len()).join().unwrap());
        assert!(other.bytes < 1_000_000, "allocations on other threads are not counted");
    }

    fn warm_core(n: usize) -> (Core<u32>, Vec<String>) {
        let mut core = Core::<u32>::new(CoreConfig { claim_budget: 1000, ..CoreConfig::default() });
        core.begin_frame(1);
        let paths: Vec<String> = (0..n).map(|i| format!("/steady/img_{i:04}.jpg")).collect();
        for p in &paths {
            core.request(1, p, Tier::Thumb, 0.0);
        }
        let mut actions = Vec::new();
        core.plan(&mut actions);
        let (batch, keys) = actions
            .iter()
            .find_map(|a| match a {
                super::super::core::Action::Fetch { batch, keys } => Some((*batch, keys.clone())),
                _ => None,
            })
            .unwrap();
        let blob = std::sync::Arc::new(super::super::store::Blob { format: super::super::store::BlobFormat::Jpeg, w: 4, h: 4, bytes: vec![0xFF, 0xD8, 0xFF] });
        core.on_fetch_done(batch, Ok((keys.iter().map(|k| (*k, blob.clone())).collect(), Vec::new())));
        actions.clear();
        core.plan(&mut actions);
        for a in actions.drain(..) {
            if let super::super::core::Action::Decode(job) = a {
                core.complete_decode(job.key, DecodeResult::Image { cost: 1000 }, |_| Some(1));
            }
        }
        (core, paths)
    }

    #[test]
    fn the_cached_request_path_in_the_core_allocates_nothing() {
        let (mut core, paths) = warm_core(12);
        assert!(paths.iter().all(|p| matches!(core.request(1, p, Tier::Thumb, 0.0), Lookup::Ready(_))));
        let mut actions: Vec<super::super::core::Action> = Vec::with_capacity(8);
        let ((), counts) = measure(|| {
            for frame in 0..500u64 {
                core.begin_frame(frame);
                for p in &paths {
                    let _ = core.request(1, p, Tier::Thumb, 1.0);
                }
                core.plan(&mut actions);
            }
        });
        assert_eq!(counts, Counts::default(), "steady-state frames must not allocate");
        assert!(actions.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_steady_frame_through_the_service_allocates_nothing_for_requests() {
        use super::super::service::{ServiceConfig, ThumbService, ThumbView};
        use super::super::store::{Blob, BlobFormat, MemStore};
        struct Idle;
        impl super::super::service::Producer for Idle {
            fn request(&self, _: &std::path::Path) -> bool {
                true
            }
        }
        let store = MemStore::new();
        let img: image::RgbImage = image::ImageBuffer::from_pixel(32, 32, image::Rgb([5, 50, 90]));
        let mut bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg).unwrap();
        let paths: Vec<String> = (0..24).map(|i| format!("/svc/img_{i}.jpg")).collect();
        for p in &paths {
            store.insert(p, Blob { format: BlobFormat::Jpeg, w: 32, h: 32, bytes: bytes.clone() });
        }
        let svc = ThumbService::new(store, tokio::runtime::Handle::current(), ServiceConfig::default(), std::sync::Arc::new(Idle));
        let ctx = egui::Context::default();
        for _ in 0..400 {
            for p in &paths {
                let _ = svc.request(1, p, Tier::Thumb, 0.0);
            }
            svc.poll_now(&ctx);
            if paths.iter().all(|p| svc.texture_for(p, Tier::Thumb).is_ready()) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(paths.iter().all(|p| svc.texture_for(p, Tier::Thumb).is_ready()));
        let ((), requests) = measure(|| {
            for _ in 0..200 {
                for p in &paths {
                    assert!(matches!(svc.request(1, p, Tier::Thumb, 3.0), ThumbView::Ready { .. }));
                }
            }
        });
        assert_eq!(requests, Counts::default(), "drawing already-cached cells must not allocate");
        let ((), polls) = measure(|| {
            for _ in 0..100 {
                svc.poll_now(&ctx);
            }
        });
        println!("steady poll: {} allocations, {} bytes over 100 polls", polls.allocs, polls.bytes);
        assert!(polls.allocs <= 100 * 4, "poll should stay near allocation free, got {polls:?}");
    }
}
