//! How the local and direct backends report into the panel: the same tagged events the gateway runtime sends.

use crossbeam::channel::Sender;
use zeroclaw_client::Usage;

use crate::zc::events::{ChatEv, Tagged};

/// Sends the events of one turn.
#[derive(Clone)]
pub struct Emitter {
    tx: Sender<Tagged>,
    session: String,
    epoch: u64,
    turn: u64,
}

impl Emitter {
    pub fn new(tx: Sender<Tagged>, session: &str, epoch: u64, turn: u64) -> Self {
        Self { tx, session: session.to_string(), epoch, turn }
    }

    fn send(&self, ev: ChatEv) {
        let _ = self.tx.try_send(Tagged { session: self.session.clone(), epoch: self.epoch, ev });
        crate::command::request_repaint();
    }

    pub fn text(&self, text: &str) {
        self.send(ChatEv::Text { turn: self.turn, segment: 0, text: text.to_string() });
    }

    pub fn done(&self, full: &str) {
        self.send(ChatEv::Done { turn: self.turn, full_response: full.to_string(), usage: Usage::default() });
    }

    pub fn error(&self, message: impl Into<String>) {
        self.send(ChatEv::Error { turn: self.turn, code: Some("PROVIDER_ERROR".into()), message: message.into(), ends_turn: true });
    }

    pub fn aborted(&self) {
        self.send(ChatEv::Aborted { turn: self.turn });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_carry_the_session_epoch_and_turn() {
        let (tx, rx) = crossbeam::channel::unbounded();
        let e = Emitter::new(tx, "s", 4, 9);
        e.text("a");
        e.done("a");
        e.error("boom");
        e.aborted();
        let got: Vec<Tagged> = rx.try_iter().collect();
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|t| t.session == "s" && t.epoch == 4));
        assert!(matches!(&got[0].ev, ChatEv::Text { turn: 9, segment: 0, text } if text == "a"));
        assert!(matches!(&got[2].ev, ChatEv::Error { ends_turn: true, .. }));
    }
}
