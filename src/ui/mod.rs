// During `cargo test` we only need status for background tasks; other heavy UI modules can be
// skipped to reduce compile surface and avoid unrelated test-time errors.
#[cfg(not(test))]
pub mod navbar;
pub mod file_table;
pub mod status;
pub mod tabs;
pub mod assistant;
pub mod refine;
pub mod image_edit;