pub mod utilities;
pub mod database;
pub mod ai;
pub mod app;
pub mod ui; // status module remains available under cfg(test)
pub mod receive;

// Re-exports to mirror main.rs so tests using the lib crate can access items like `crate::Thumbnail`.
pub use utilities::{explorer::*, scan::*, thumbs::*, types::*};
pub use database::*;
