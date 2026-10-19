mod common;

use std::net::SocketAddr;
use std::sync::Arc;

use common::*;
use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;
use zeroclaw_client::mock::{MockGateway, MockOptions};
use zeroclaw_client::{ChatSession, ConnectOptions, Error, RestClient};

/// Accepts TCP, completes the TLS handshake and hands the stream to axum; failed handshakes are skipped.
struct TlsListener {
    inner: TcpListener,
    acceptor: TlsAcceptor,
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((tcp, addr)) = self.inner.accept().await else { continue };
            if let Ok(tls) = self.acceptor.accept(tcp).await {
                return (tls, addr);
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

async fn tls_mock() -> (MockGateway, String) {
    let cert = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string(), "localhost".to_string()]).unwrap();
    let pem = cert.cert.pem();
    let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let server = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.der().clone()], key)
        .unwrap();
    let listener = TlsListener {
        inner: TcpListener::bind("127.0.0.1:0").await.unwrap(),
        acceptor: TlsAcceptor::from(Arc::new(server)),
    };
    (MockGateway::start_on(MockOptions::default(), listener, true).await, pem)
}

#[tokio::test]
async fn chat_and_rest_work_over_tls_with_an_extra_root() {
    let (mock, pem) = tls_mock().await;
    assert!(mock.url().starts_with("https://127.0.0.1:"));
    let mut cfg = mock.config();
    cfg.extra_root_pem = Some(pem);

    let health = RestClient::new(&cfg).unwrap().health().await.unwrap();
    assert_eq!(health.status, "ok");

    let (session, mut events) = connect(cfg.clone()).await;
    session.send_text("over tls").await.unwrap();
    assert_eq!(text_of(&until_done(&mut events).await), "echo: over tls");
    assert_eq!(RestClient::new(&cfg).unwrap().sessions().await.unwrap().len(), 1);
}

#[tokio::test]
async fn untrusted_certificates_are_refused_for_both_transports() {
    let (mock, _pem) = tls_mock().await;
    let cfg = mock.config();
    let err = RestClient::new(&cfg).unwrap().health().await.unwrap_err();
    assert!(matches!(err, Error::Connect(_)), "{err:?}");

    let err = ChatSession::connect(cfg, ConnectOptions::default()).await.err().unwrap();
    assert!(matches!(err, Error::Connect(_)), "{err:?}");
    assert!(err.is_retryable());
    assert!(mock.connections().is_empty(), "no upgrade request may reach the server over an untrusted channel");
}

#[tokio::test]
async fn a_certificate_for_another_host_is_refused() {
    let cert = rcgen::generate_simple_self_signed(vec!["other.example".to_string()]).unwrap();
    let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let server = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.der().clone()], key)
        .unwrap();
    let listener = TlsListener {
        inner: TcpListener::bind("127.0.0.1:0").await.unwrap(),
        acceptor: TlsAcceptor::from(Arc::new(server)),
    };
    let mock = MockGateway::start_on(MockOptions::default(), listener, true).await;
    let mut cfg = mock.config();
    cfg.extra_root_pem = Some(cert.cert.pem());
    assert!(RestClient::new(&cfg).unwrap().health().await.is_err(), "trusting the root is not enough when the name does not match");
    assert!(ChatSession::connect(cfg, ConnectOptions::default()).await.is_err());
}
