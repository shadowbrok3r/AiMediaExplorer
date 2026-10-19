use std::sync::Arc;
use std::sync::Once;

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;
use rustls::{ClientConfig, RootCertStore};

use crate::error::{Error, Result};

static INSTALL: Once = Once::new();

/// Installs the process-wide rustls provider (ring) when none is set; idempotent and thread-safe.
pub fn install_crypto_provider() {
    INSTALL.call_once(|| {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            let _ = rustls::crypto::ring::default_provider().install_default();
        }
    });
}

/// True once a process-wide provider exists.
pub fn crypto_provider_installed() -> bool {
    rustls::crypto::CryptoProvider::get_default().is_some()
}

/// Parses every certificate in a PEM bundle.
pub fn parse_pem_certs(pem: &str) -> Result<Vec<CertificateDer<'static>>> {
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(pem.as_bytes())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Error::Config(format!("invalid root certificate PEM: {e}")))?;
    if certs.is_empty() {
        return Err(Error::Config("root certificate PEM contains no certificate".into()));
    }
    Ok(certs)
}

/// Rustls client config with webpki roots plus an optional extra root, built on the installed provider.
pub(crate) fn client_config(extra_root_pem: Option<&str>) -> Result<Arc<ClientConfig>> {
    install_crypto_provider();
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .ok_or_else(|| Error::Config("no rustls crypto provider is installed".into()))?;
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Some(pem) = extra_root_pem {
        for cert in parse_pem_certs(pem)? {
            roots
                .add(cert)
                .map_err(|e| Error::Config(format!("rejected root certificate: {e}")))?;
        }
    }
    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Config(format!("tls versions: {e}")))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent_and_thread_safe() {
        let handles: Vec<_> = (0..8).map(|_| std::thread::spawn(install_crypto_provider)).collect();
        for h in handles {
            h.join().unwrap();
        }
        install_crypto_provider();
        assert!(crypto_provider_installed());
        let first = rustls::crypto::CryptoProvider::get_default().cloned().unwrap();
        install_crypto_provider();
        let second = rustls::crypto::CryptoProvider::get_default().cloned().unwrap();
        assert!(Arc::ptr_eq(&first, &second), "a second call must not replace the provider");
    }

    #[test]
    fn client_config_builds_without_panicking() {
        client_config(None).unwrap();
    }

    #[test]
    fn extra_root_pem_is_validated() {
        assert!(client_config(Some("not a certificate")).is_err());
        assert!(parse_pem_certs("").is_err());
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let pem = cert.cert.pem();
        assert_eq!(parse_pem_certs(&pem).unwrap().len(), 1);
        client_config(Some(&pem)).unwrap();
    }
}
