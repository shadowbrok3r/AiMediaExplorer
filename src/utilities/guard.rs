use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

// Runs `f`; a panic becomes `Err(on_panic(message))`.
pub fn guarded<T, E>(f: impl FnOnce() -> Result<T, E>, on_panic: impl FnOnce(String) -> E) -> Result<T, E> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(on_panic(panic_message(payload.as_ref()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_becomes_error() {
        let r: Result<u32, String> = guarded(|| -> Result<u32, String> { panic!("boom {}", 7) }, |m| format!("caught: {m}"));
        assert_eq!(r.unwrap_err(), "caught: boom 7");
    }

    #[test]
    fn str_payload_and_ok_passthrough() {
        let r: Result<u32, String> = guarded(|| -> Result<u32, String> { panic!("static") }, |m| m);
        assert_eq!(r.unwrap_err(), "static");
        let ok: Result<u32, String> = guarded(|| Ok(3), |m| m);
        assert_eq!(ok.unwrap(), 3);
    }
}
