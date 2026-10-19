//! Own process: building a client installs the crypto provider before any TLS use.

use zeroclaw_client::tls::crypto_provider_installed;
use zeroclaw_client::{Config, RestClient};

#[test]
fn building_a_client_installs_the_provider_itself() {
    assert!(!crypto_provider_installed());
    let rest = RestClient::new(&Config::new("https://gateway.invalid", "t"));
    assert!(rest.is_ok(), "{rest:?}");
    assert!(crypto_provider_installed());
}
