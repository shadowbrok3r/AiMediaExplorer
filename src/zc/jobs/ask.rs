//! Natural-language filters: the request that turns a sentence into a filter, and the strict check of the reply.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use chrono::{Datelike, NaiveDate};
use serde_json::Value;

use super::{JobKind, JobSpec, Quality, Shape, Step, StepReply, ZcJobs, run_one};
use crate::ai::mcp::ctx::DbSource;
use crate::ai::refine::zeroclaw::{Vocabulary, library_vocabulary};
use crate::ui::file_table::rows::{Chip, FilterSpec};

/// Longest sentence sent.
pub const MAX_SENTENCE: usize = 400;

/// The filter a sentence stands for, ready to show and to apply.
#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    pub sentence: String,
    /// The JSON the model returned, as applied with `set_filter` and `replace`.
    pub patch: Value,
    /// The conditions the tab would have afterwards.
    pub filter: FilterSpec,
    pub chips: Vec<Chip>,
    pub model: String,
    pub quality: Quality,
}

/// The model's reply as a filter, or why it is not one. Anything the filter parser refuses is an error; nothing is dropped or guessed.
pub fn validate(patch: &Value) -> Result<FilterSpec, String> {
    let Some(object) = patch.as_object() else { return Err("the reply is not a JSON object".to_string()) };
    if object.is_empty() {
        return Err("the reply sets no conditions".to_string());
    }
    let filter = FilterSpec::default().merged(patch, true)?;
    if filter == FilterSpec::default() {
        return Err("the reply sets no conditions".to_string());
    }
    Ok(filter)
}

/// The preview of a reply to `sentence`.
pub fn read_reply(sentence: &str, reply: &StepReply) -> Result<Preview, String> {
    let found = match (&reply.json, &reply.json_error) {
        (Some(found), _) => found,
        (None, Some(e)) => return Err(format!("{e}: {}", super::record::clip_text(reply.text.trim(), 160))),
        (None, None) => return Err("the reply held no filter".to_string()),
    };
    let filter = validate(&found.value)?;
    let chips = filter.chips();
    Ok(Preview { sentence: sentence.to_string(), patch: found.value.clone(), filter, chips, model: reply.model.clone(), quality: found.quality })
}

fn first_of_month(day: NaiveDate) -> NaiveDate {
    day.with_day(1).unwrap_or(day)
}

/// The instructions: the exact fields of a filter, what they mean and examples; the library's own tags and categories help the model pick existing ones.
pub fn system_prompt(today: NaiveDate, vocab: &Vocabulary) -> String {
    let schema = serde_json::to_string_pretty(&FilterSpec::none()).unwrap_or_default();
    let month = first_of_month(today);
    let mut text = format!(
        "You turn a sentence about files in a personal media library into a filter.\n\
Reply with ONLY one JSON object, with no prose and no code fences, that sets the conditions the sentence asks for. Leave out every field the sentence does not mention.\n\
Today is {today}.\n\
These are the only fields you may use, shown with the value that means no condition:\n{schema}\n\
Meaning of the fields:\n\
- types: images, videos, dirs, archives, other. false hides that kind of file, so {{\"types\":{{\"videos\":false}}}} hides videos.\n\
- size: min and max in bytes (1 KiB = 1024, 1 MiB = 1048576, 1 GiB = 1073741824).\n\
- modified and created: after and before as YYYY-MM-DD, both inclusive. modified is the date of the file, created the date it was added to the library.\n\
- query: text that must appear in the file name, type, path, tags, category or description. name: text that must appear in the file name.\n\
- excluded_terms: lowercase text; files whose path contains it are hidden. excluded_dirs: folders that are hidden.\n\
- ext_allow: only files with these extensions (lowercase, no dot). ext_deny: hide these extensions.\n\
- indexed, description, clip, offline: \"any\", \"yes\" or \"no\". description \"yes\" means the file has a description, clip \"yes\" that it has a CLIP embedding.\n\
- categories: files of any of these categories. tags: files with any of these tags, or with all of them when tags_all is true.\n\
- skip_icons: hide icons and tiny pictures.\n\
Without a size condition files under 10 KiB are hidden; to include them set {{\"size\":{{\"min\":null}}}}.\n\
Never invent a field. If the sentence asks for something no field can express, use query or name for its main words.\n"
    );
    if !vocab.tags.is_empty() {
        text.push_str(&format!("Tags of the library, most used first: {}\n", vocab.tags.join(", ")));
    }
    if !vocab.categories.is_empty() {
        text.push_str(&format!("Categories of the library, most used first: {}\n", vocab.categories.join(", ")));
    }
    if !vocab.is_empty() {
        text.push_str("Use these exact tags and categories when one fits the sentence.\n");
    }
    text.push_str(&examples(month));
    text
}

/// The sentences and replies shown to the model; each reply must pass [`validate`].
pub fn example_pairs(month_start: NaiveDate) -> Vec<(String, String)> {
    vec![
        ("videos larger than 1 GB from 2023".to_string(), r#"{"types":{"images":false,"dirs":false,"archives":false,"other":false},"size":{"min":1073741824},"modified":{"after":"2023-01-01","before":"2023-12-31"}}"#.to_string()),
        ("pictures tagged sunset or beach that have no description".to_string(), r#"{"types":{"videos":false,"archives":false,"other":false},"tags":["sunset","beach"],"description":"no"}"#.to_string()),
        ("everything added to the library this month".to_string(), format!(r#"{{"created":{{"after":"{month_start}"}}}}"#)),
        ("jpg or png files that are not indexed, but not from the downloads folder".to_string(), r#"{"ext_allow":["jpg","png"],"indexed":"no","excluded_terms":["downloads"]}"#.to_string()),
    ]
}

fn examples(month_start: NaiveDate) -> String {
    let mut text = String::from("Examples:\n");
    for (sentence, reply) in example_pairs(month_start) {
        text.push_str(&format!("Sentence: {sentence}\nReply: {reply}\n"));
    }
    text
}

/// The request for `sentence`.
pub fn request(sentence: &str, today: NaiveDate, vocab: &Vocabulary, hint: Option<String>) -> Result<JobSpec, String> {
    let sentence = sentence.trim();
    if sentence.is_empty() {
        return Err("Describe the files to look for first".to_string());
    }
    if sentence.chars().count() > MAX_SENTENCE {
        return Err(format!("The sentence is longer than {MAX_SENTENCE} characters"));
    }
    let title = format!("Ask: {}", super::record::clip_text(sentence, 48).replace('\n', " "));
    let step = Step::new("filter", format!("Sentence: {sentence}\nReply:")).system(system_prompt(today, vocab)).expect(Shape::Object).temperature(0.0);
    let mut spec = JobSpec::new(JobKind::Ask, title).step(step);
    if let Some(hint) = hint {
        spec = spec.hint(hint);
    }
    Ok(spec)
}

/// Asks the gateway what filter `sentence` means; `cancel` stops the wait and the job.
pub async fn ask(jobs: &Arc<ZcJobs>, db: &DbSource, sentence: &str, hint: Option<String>, cancel: &AtomicBool) -> Result<Preview, String> {
    ask_on(jobs, db, sentence, hint, cancel, chrono::Local::now().date_naive()).await
}

/// [`ask`] with the date of today given.
pub async fn ask_on(jobs: &Arc<ZcJobs>, db: &DbSource, sentence: &str, hint: Option<String>, cancel: &AtomicBool, today: NaiveDate) -> Result<Preview, String> {
    let vocab = match db.get() {
        Ok(db) => library_vocabulary(db).await,
        Err(_) => Vocabulary::default(),
    };
    let spec = request(sentence, today, &vocab, hint)?;
    let reply = run_one(jobs, spec, cancel).await?;
    read_reply(sentence.trim(), &reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn day() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, 19).unwrap()
    }

    #[test]
    fn every_example_of_the_prompt_passes_the_strict_parser() {
        for (sentence, reply) in example_pairs(first_of_month(day())) {
            let value: Value = serde_json::from_str(&reply).unwrap_or_else(|e| panic!("{sentence}: {e}"));
            let filter = validate(&value).unwrap_or_else(|e| panic!("{sentence}: {e}"));
            assert!(!filter.chips().is_empty(), "{sentence}");
        }
    }

    #[test]
    fn the_prompt_lists_the_exact_fields_the_parser_accepts() {
        let prompt = system_prompt(day(), &Vocabulary { tags: vec!["sunset".into(), "beach".into()], categories: vec!["landscape".into()] });
        let schema = serde_json::to_value(FilterSpec::none()).unwrap();
        for field in schema.as_object().unwrap().keys() {
            assert!(prompt.contains(&format!("\"{field}\"")), "the schema in the prompt names {field}");
        }
        for expected in ["Today is 2026-10-19", "Tags of the library, most used first: sunset, beach", "Categories of the library, most used first: landscape", "\"after\":\"2026-10-01\"", "Sentence: videos larger than 1 GB"] {
            assert!(prompt.contains(expected), "missing {expected:?}");
        }
        assert!(!system_prompt(day(), &Vocabulary::default()).contains("Tags of the library"));
    }

    #[test]
    fn replies_are_checked_strictly_and_changes_nothing_on_a_failure() {
        struct Case {
            name: &'static str,
            reply: Value,
            err: Option<&'static str>,
        }
        let cases = [
            Case { name: "a tag filter", reply: json!({"tags": ["sunset"]}), err: None },
            Case { name: "dates and size", reply: json!({"modified": {"after": "2024-01-01"}, "size": {"min": null, "max": 5242880}}), err: None },
            Case { name: "an unknown field", reply: json!({"colour": "red"}), err: Some("unknown field")},
            Case { name: "an unknown nested field", reply: json!({"size": {"minimum": 3}}), err: Some("unknown field") },
            Case { name: "a malformed date", reply: json!({"modified": {"after": "last summer"}}), err: Some("not a date") },
            Case { name: "a wrong type", reply: json!({"tags": "sunset"}), err: Some("bad filter") },
            Case { name: "a bad tri-state", reply: json!({"clip": "maybe"}), err: Some("bad filter") },
            Case { name: "an empty object", reply: json!({}), err: Some("no conditions") },
            Case { name: "an object that only repeats the defaults", reply: json!({"types": {"images": true}}), err: Some("no conditions") },
            Case { name: "an array", reply: json!([{"tags": ["a"]}]), err: Some("not a JSON object") },
            Case { name: "a string", reply: json!("tags"), err: Some("not a JSON object") },
            Case { name: "negative size", reply: json!({"size": {"min": -1}}), err: Some("bad filter") },
        ];
        for case in cases {
            let got = validate(&case.reply);
            match case.err {
                None => assert!(got.is_ok(), "{}: {got:?}", case.name),
                Some(part) => assert!(got.as_ref().is_err_and(|e| e.contains(part)), "{}: {got:?}", case.name),
            }
        }
    }

    #[test]
    fn a_valid_reply_becomes_a_preview_with_the_chips_it_would_apply() {
        let reply = StepReply {
            index: 0,
            label: "filter".into(),
            text: String::new(),
            model: "mock-model".into(),
            json: Some(super::super::Extracted { value: json!({"tags": ["sunset", "beach"], "description": "no", "types": {"videos": false}}), quality: Quality::Clean }),
            json_error: None,
            prompt_bytes: 1,
            duration: std::time::Duration::ZERO,
            attempts: 1,
        };
        let preview = read_reply("sunset photos without a description", &reply).unwrap();
        let labels: Vec<&str> = preview.chips.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"Tag: sunset, beach") && labels.contains(&"No description") && labels.contains(&"Hiding: Videos"), "{labels:?}");
        assert!(labels.contains(&"Size >= 10 KiB"), "the default size limit is part of the result: {labels:?}");
        assert_eq!(preview.filter.tags, ["sunset", "beach"]);
        assert_eq!(preview.sentence, "sunset photos without a description");
    }

    #[test]
    fn a_reply_without_json_says_what_came_back() {
        let reply = StepReply { index: 0, label: String::new(), text: "I cannot do that, sorry.".into(), model: String::new(), json: None, json_error: Some(super::super::ExtractError::NoJson), prompt_bytes: 0, duration: std::time::Duration::ZERO, attempts: 1 };
        let err = read_reply("x", &reply).unwrap_err();
        assert!(err.contains("no JSON") && err.contains("I cannot do that"), "{err}");
    }

    #[test]
    fn the_request_checks_the_sentence_and_asks_for_one_object() {
        let spec = request("  all my sunsets  ", day(), &Vocabulary::default(), Some("slow".into())).unwrap();
        assert_eq!((spec.kind, spec.hint.as_deref(), spec.steps.len()), (JobKind::Ask, Some("slow"), 1));
        let step = &spec.steps[0];
        assert_eq!((step.expect, step.temperature, step.prompt.as_str()), (Some(Shape::Object), 0.0, "Sentence: all my sunsets\nReply:"));
        assert!(step.input_len() < zeroclaw_client::types::COMPLETE_INPUT_CAP);
        assert!(request("   ", day(), &Vocabulary::default(), None).is_err());
        assert!(request(&"x".repeat(MAX_SENTENCE + 1), day(), &Vocabulary::default(), None).is_err());
    }
}
