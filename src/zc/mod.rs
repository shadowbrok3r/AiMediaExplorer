//! ZeroClaw glue: gateway profile and token, health, the chat runtime, the connection card and the `zc` command.

pub mod card;
mod card_tools;
pub mod cli;
pub mod coalesce;
pub mod command;
pub mod debug;
pub mod events;
pub mod health;
pub mod import;
pub mod jobs;
pub mod profile;
pub mod runtime;
pub mod secrets;
pub mod session;
pub mod tools;
pub mod widgets;

#[cfg(test)]
pub(crate) mod runtime_tests;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};

use eframe::egui::{Context, Ui};

pub use events::{ChatEv, Link, LinkState, Pending, Tagged};
pub use health::Health;
pub use profile::{BackendPref, ZcProfile};
pub use runtime::ZcRuntime;

static RUNTIME: OnceLock<Arc<ZcRuntime>> = OnceLock::new();

/// Runs the runtime's tasks when no application runtime is current (and always under test, where each test owns its own).
static BACKGROUND: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).thread_name("zc-runtime").enable_all().build().expect("the ZeroClaw background runtime")
});

fn handle() -> tokio::runtime::Handle {
    if cfg!(test) {
        return BACKGROUND.handle().clone();
    }
    tokio::runtime::Handle::try_current().unwrap_or_else(|_| BACKGROUND.handle().clone())
}

/// The runtime of the application.
pub fn runtime() -> &'static Arc<ZcRuntime> {
    RUNTIME.get_or_init(|| {
        let rt = ZcRuntime::new(handle(), token_store());
        rt.set_publish(true);
        rt
    })
}

/// The real token file, or a private folder under test so no test reads a person's token.
fn token_store() -> Option<secrets::TokenStore> {
    if cfg!(test) {
        return Some(secrets::TokenStore::at(std::env::temp_dir().join(format!("ame-zc-global-token-{}", std::process::id()))));
    }
    secrets::TokenStore::standard()
}

/// Makes `profile` the live one of `rt` and stores it in the settings.
pub fn save_profile_with(rt: &ZcRuntime, profile: ZcProfile) {
    rt.set_profile(profile.clone());
    crate::settings::update("zc:profile", move |s| s.zeroclaw = Some(profile));
}

/// Makes `profile` the live one and stores it in the settings.
pub fn save_profile(profile: ZcProfile) {
    save_profile_with(runtime(), profile);
}

/// Registers the ZEROCLAW engine dot and creates the runtime.
pub fn init() {
    let rt = runtime();
    health::publish(&rt.health(), &rt.profile());
}

static SETTINGS_SEEN: AtomicU64 = AtomicU64::new(u64::MAX);
static TOOLS_STARTED: AtomicBool = AtomicBool::new(false);

/// Per-frame work: follows the settings store, runs the health sweep and starts the tools server once the profile is known.
pub fn tick(ctx: &Context) {
    let rt = runtime();
    rt.attach_ctx(ctx);
    sync_settings(rt);
    rt.tick();
}

fn sync_settings(rt: &Arc<ZcRuntime>) {
    let store = crate::settings::store();
    let generation = store.generation();
    if !store.is_loaded() || SETTINGS_SEEN.swap(generation, Ordering::AcqRel) == generation {
        return;
    }
    let Some(stored) = store.read(|s| s.zeroclaw.clone()) else { return };
    rt.set_profile(stored.clone());
    if stored.tools_enabled && !TOOLS_STARTED.swap(true, Ordering::AcqRel) {
        tools::ensure_running(&stored);
    }
}

/// Draws the ZeroClaw connection card of Settings > Assistant.
pub fn connection_ui(ui: &mut Ui) {
    card::show(ui, runtime(), card::Mode::Settings);
}

/// Draws the ZeroClaw step of the first-run wizard; true once the step is complete or skipped.
pub fn onboarding_ui(ui: &mut Ui) -> bool {
    card::show(ui, runtime(), card::Mode::Wizard)
}
