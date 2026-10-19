use std::fmt;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("unauthorized (HTTP 401): the gateway refused the token")]
    Unauthorized,
    #[error("forbidden (HTTP 403)")]
    Forbidden,
    #[error("gateway rejected the request (HTTP {status}): {message}")]
    Rejected { status: u16, message: String },
    #[error("cannot reach the gateway: {0}")]
    Connect(String),
    #[error("timed out: {0}")]
    Timeout(&'static str),
    #[error("gateway resumed a different session: requested {requested}, received {received}")]
    SessionMismatch { requested: String, received: String },
    #[error("gateway error {code}: {message}")]
    Gateway { code: String, message: String },
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("the chat session is closed")]
    Closed,
    #[error("the chat socket is not connected")]
    NotConnected,
    #[error("message is empty")]
    EmptyMessage,
    #[error("message of {size} bytes exceeds the {limit} byte limit")]
    TooLarge { size: usize, limit: usize },
    #[error("invalid attachment: {0}")]
    Attachment(String),
    #[error("HTTP {status} from {path}: {message}")]
    Http { status: u16, path: String, message: String },
    #[error("response from {path} exceeds {limit} bytes")]
    ResponseTooLarge { path: String, limit: usize },
    #[error("invalid response from {path}: {message}")]
    Decode { path: String, message: String },
}

impl Error {
    /// True when retrying the same request later can succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Connect(_) | Error::Timeout(_) | Error::NotConnected => true,
            Error::Rejected { status, .. } | Error::Http { status, .. } => {
                matches!(*status, 408 | 425 | 429) || *status >= 500
            }
            _ => false,
        }
    }

    /// True for a refused or missing credential.
    pub fn is_auth(&self) -> bool {
        match self {
            Error::Unauthorized | Error::Forbidden => true,
            Error::Rejected { status, .. } | Error::Http { status, .. } => matches!(*status, 401 | 403),
            _ => false,
        }
    }

    /// Process exit code class used by `zcprobe`.
    pub fn exit_class(&self) -> ExitClass {
        if self.is_auth() {
            ExitClass::Auth
        } else if matches!(self, Error::Connect(_) | Error::Timeout(_) | Error::NotConnected) {
            ExitClass::Unreachable
        } else if matches!(self, Error::Config(_) | Error::EmptyMessage | Error::Attachment(_)) {
            ExitClass::Usage
        } else {
            ExitClass::Failure
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitClass {
    Usage,
    Failure,
    Auth,
    Unreachable,
}

impl ExitClass {
    pub fn code(self) -> u8 {
        match self {
            ExitClass::Failure => 1,
            ExitClass::Usage => 2,
            ExitClass::Auth => 3,
            ExitClass::Unreachable => 4,
        }
    }
}

impl fmt::Display for ExitClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ExitClass::Usage => "usage",
            ExitClass::Failure => "failure",
            ExitClass::Auth => "auth",
            ExitClass::Unreachable => "unreachable",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_and_auth_classification() {
        assert!(Error::Connect("x".into()).is_retryable());
        assert!(Error::Timeout("connect").is_retryable());
        assert!(Error::Http { status: 503, path: "/p".into(), message: String::new() }.is_retryable());
        assert!(Error::Http { status: 429, path: "/p".into(), message: String::new() }.is_retryable());
        assert!(!Error::Http { status: 404, path: "/p".into(), message: String::new() }.is_retryable());
        assert!(!Error::Unauthorized.is_retryable());
        assert!(Error::Unauthorized.is_auth());
        assert!(Error::Forbidden.is_auth());
        assert!(Error::Http { status: 401, path: "/p".into(), message: String::new() }.is_auth());
        assert!(!Error::Closed.is_auth());
    }

    #[test]
    fn exit_classes_map_to_distinct_codes() {
        assert_eq!(Error::Unauthorized.exit_class().code(), 3);
        assert_eq!(Error::Connect("x".into()).exit_class().code(), 4);
        assert_eq!(Error::Config("x".into()).exit_class().code(), 2);
        assert_eq!(Error::Closed.exit_class().code(), 1);
    }
}
