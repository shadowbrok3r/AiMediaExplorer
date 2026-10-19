#![doc = include_str!("../README.md")]

pub mod attachments;
pub mod backoff;
pub mod chat;
pub mod config;
pub mod error;
pub mod event;
pub mod frames;
pub mod history;
pub mod rest;
pub mod text;
pub mod tls;
pub mod replay;
pub mod types;

#[cfg(feature = "mock")]
pub mod mock;
#[doc(hidden)]
pub mod probe;

pub use attachments::{ImageRef, InlineImage, compose_message, split_images};
pub use chat::{ChatSession, ConnState, ConnectOptions, EventStream};
pub use config::{Config, DEFAULT_PORT, Endpoint, ReconnectPolicy, Token, WsAuth};
pub use error::{Error, ExitClass, Result};
pub use event::{ApprovalRequest, Direction, DisconnectReason, Event, RawFrame, Translator, Usage};
pub use frames::{ApprovalDecision, ClientFrame, ServerFrame};
pub use history::{HistoryItem, HistoryKind, HistoryPager, StoredMessage, strip_date_stamp, unwrap_stored};
pub use rest::{AuthState, Probe, RestClient};
pub use tls::install_crypto_provider;
pub use types::{
    AbortOutcome, CompleteRequest, CompleteResponse, ConfigPatch, Health, MemoryEntry, MemoryQuery, MemoryStore,
    PatchOp, SessionInfo, SessionRow, TurnState, group_sessions,
};
