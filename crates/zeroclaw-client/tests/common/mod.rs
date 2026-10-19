#![allow(dead_code)]

use std::time::Duration;

use zeroclaw_client::event::event_kind;
use zeroclaw_client::{ChatSession, ConnectOptions, Config, Event, EventStream};

pub const WAIT: Duration = Duration::from_secs(10);

/// Reads events until `stop` accepts one (inclusive) or the stream ends; panics on timeout.
pub async fn collect_until(events: &mut EventStream, stop: impl Fn(&Event) -> bool) -> Vec<Event> {
    let mut out = Vec::new();
    loop {
        let next = tokio::time::timeout(WAIT, events.next()).await.unwrap_or_else(|_| {
            panic!("timed out waiting for an event; got so far: {:?}", kinds(&out));
        });
        let Some(event) = next else { return out };
        let done = stop(&event);
        out.push(event);
        if done {
            return out;
        }
    }
}

pub async fn until_done(events: &mut EventStream) -> Vec<Event> {
    collect_until(events, |e| matches!(e, Event::Done { .. } | Event::Aborted { .. })).await
}

pub fn kinds(events: &[Event]) -> Vec<&'static str> {
    events.iter().map(event_kind).collect()
}

/// Kinds without the session-level events, for asserting a turn's shape.
pub fn turn_kinds(events: &[Event]) -> Vec<&'static str> {
    events
        .iter()
        .filter(|e| !matches!(e, Event::Connected { .. } | Event::Raw(_)))
        .map(event_kind)
        .collect()
}

pub fn text_of(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|e| if let Event::Token { text, .. } = e { Some(text.as_str()) } else { None })
        .collect()
}

/// Connects and swallows the first `Connected` event.
pub async fn connect(cfg: Config) -> (ChatSession, EventStream) {
    connect_with(cfg, ConnectOptions::default()).await
}

pub async fn connect_with(cfg: Config, opts: ConnectOptions) -> (ChatSession, EventStream) {
    let (session, mut events) = ChatSession::connect(cfg, opts).await.expect("connect");
    loop {
        let event = tokio::time::timeout(WAIT, events.next()).await.expect("connected event").expect("stream open");
        match event {
            Event::Raw(_) => {}
            Event::Connected { .. } => return (session, events),
            other => panic!("first event was {other:?}"),
        }
    }
}

pub async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + WAIT;
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
