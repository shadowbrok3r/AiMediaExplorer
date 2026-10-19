//! Own process: with `ring` and `aws-lc-rs` both compiled in, rustls panics until a provider is installed.

use std::panic::catch_unwind;

use zeroclaw_client::install_crypto_provider;
use zeroclaw_client::tls::crypto_provider_installed;

#[test]
fn with_two_providers_rustls_panics_until_one_is_installed() {
    assert!(!crypto_provider_installed());
    let before = catch_unwind(|| {
        let _ = rustls::ClientConfig::builder();
    });
    assert!(before.is_err(), "rustls should not be able to choose between ring and aws-lc-rs on its own");

    install_crypto_provider();
    assert!(crypto_provider_installed());
    let _ = rustls::ClientConfig::builder();

    install_crypto_provider();
    install_crypto_provider();
    assert!(crypto_provider_installed());
}
