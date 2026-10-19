//! The job service against the loopback mock gateway: success, malformed and partial JSON, a refused token, transient errors, cancel, batching and the queue.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::runtime::Handle;
use zeroclaw_client::mock::{MockGateway, MockOptions};

use super::*;
use crate::tasks::TaskState;
use crate::zc::health::Health;
use crate::zc::profile::ZcProfile;
use crate::zc::runtime::ZcRuntime;
use crate::zc::runtime_tests::{runtime_for, temp_dir, until};
use crate::zc::secrets::TokenStore;

async fn rig(name: &str) -> (MockGateway, Arc<ZcRuntime>, Arc<ZcJobs>) {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", name);
    until("online", || rt.health().is_online().then_some(())).await;
    let jobs = ZcJobs::new(rt.clone(), JobsConfig::quick());
    (mock, rt, jobs)
}

fn complete_calls(mock: &MockGateway) -> Vec<serde_json::Value> {
    mock.requests().into_iter().filter(|r| r.path == "/api/complete").map(|r| r.body).collect()
}

async fn finish(handle: &mut JobHandle) -> Vec<JobEvent> {
    let mut events = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(8), handle.next()).await {
            Ok(Some(event)) => {
                let last = matches!(event, JobEvent::Finished(_));
                events.push(event);
                if last {
                    return events;
                }
            }
            other => panic!("the job did not finish: {other:?} after {events:?}"),
        }
    }
}

fn outcome(events: &[JobEvent]) -> JobOutcome {
    match events.last() {
        Some(JobEvent::Finished(o)) => o.clone(),
        other => panic!("no outcome: {other:?}"),
    }
}

fn replies(events: &[JobEvent]) -> Vec<&StepReply> {
    events.iter().filter_map(|e| if let JobEvent::Reply(r) = e { Some(r) } else { None }).collect()
}

fn task_state(title: &str) -> Option<(TaskState, u64, u64)> {
    crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == title).map(|t| (t.state, t.done, t.total)))
}

fn one_step(title: &str, prompt: &str) -> JobSpec {
    JobSpec::new(JobKind::Manual, title).step(Step::new("only", prompt))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_runs_as_a_tracked_task_and_reports_its_reply_and_usage() {
    let (mock, _rt, jobs) = rig("jobs-ok").await;
    mock.script_complete(200, json!({"response": "pong", "model": "mock-model", "hint": "quick"}), Duration::from_millis(250));
    let mut handle = jobs.submit(one_step("jobs-ok task", "ping")).unwrap();
    let running = until("the task", || task_state("jobs-ok task")).await;
    assert_eq!((running.0.is_active(), running.2), (true, 1), "the job is a task while it runs");
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Completed);
    let got = replies(&events);
    assert_eq!((got.len(), got[0].text.as_str(), got[0].model.as_str(), got[0].attempts), (1, "pong", "mock-model", 1));
    assert!(events.iter().any(|e| matches!(e, JobEvent::Started { steps: 1 })));
    let calls = complete_calls(&mock);
    assert_eq!(calls.len(), 1);
    assert_eq!((calls[0]["hint"].as_str(), calls[0]["prompt"].as_str(), calls[0]["temperature"].as_f64()), (Some("quick"), Some("ping"), Some(0.2)));
    let snap = jobs.snapshot();
    let record = &snap.history[0];
    assert_eq!((record.title.as_str(), record.hint.as_str(), record.outcome.clone()), ("jobs-ok task", "quick", Some(JobOutcome::Completed)));
    assert_eq!((record.steps[0].prompt_bytes, record.steps[0].response_bytes, record.steps[0].attempts), (4, 4, 1));
    assert_eq!((snap.totals.jobs, snap.totals.completed, snap.totals.calls, snap.totals.prompt_bytes, snap.totals.response_bytes), (1, 1, 1, 4, 4));
    assert_eq!((snap.totals.est_tokens_in(), snap.totals.est_tokens_out()), (1, 1));
    assert!(jobs.is_idle() && snap.running.is_none() && snap.queue.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn replies_with_json_are_read_even_when_fenced_or_wrapped_in_prose() {
    let (mock, _rt, jobs) = rig("jobs-json").await;
    mock.script_complete_reply("```json\n[{\"id\":1,\"new_category\":\"beach\"}]\n```");
    mock.script_complete_reply("Sure! Here you go: [{\"id\":2},] bye");
    let spec = JobSpec::new(JobKind::Refine, "jobs-json task").steps(["a", "b"].map(|p| Step::new(p, p).expect(Shape::Rows)));
    let mut handle = jobs.submit(spec).unwrap();
    let events = finish(&mut handle).await;
    let got = replies(&events);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].json.as_ref().map(|j| (j.value.clone(), j.quality)), Some((json!([{"id":1,"new_category":"beach"}]), Quality::Clean)));
    assert_eq!(got[1].json.as_ref().map(|j| (j.value.clone(), j.quality)), Some((json!([{"id":2}]), Quality::Repaired)));
    let progress: Vec<(usize, usize)> = events.iter().filter_map(|e| if let JobEvent::Progress { done, total } = e { Some((*done, *total)) } else { None }).collect();
    assert_eq!(progress, [(0, 2), (1, 2), (2, 2)]);
    assert_eq!(task_state("jobs-json task"), Some((TaskState::Completed, 2, 2)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_without_json_completes_the_job_and_says_why_it_cannot_be_read() {
    let (mock, _rt, jobs) = rig("jobs-malformed").await;
    mock.script_complete_reply("I am sorry, I cannot produce that.");
    let spec = JobSpec::new(JobKind::Refine, "jobs-malformed task").step(Step::new("a", "a").expect(Shape::Rows));
    let mut handle = jobs.submit(spec).unwrap();
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Completed);
    let got = replies(&events);
    assert!(got[0].json.is_none());
    assert_eq!(got[0].json_error, Some(ExtractError::NoJson));
    assert_eq!(got[0].text, "I am sorry, I cannot produce that.", "the raw reply is still handed over");
    let record = jobs.snapshot().history[0].clone();
    assert_eq!(record.steps[0].json.as_deref(), Some("the reply held no JSON of the expected kind"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cut_off_reply_keeps_its_complete_leading_rows() {
    let (mock, _rt, jobs) = rig("jobs-partial").await;
    mock.script_complete_reply(r#"[{"id":1,"new_tags":["a"]},{"id":2,"new_tags":["b"]},{"id":3,"new_ta"#);
    let spec = JobSpec::new(JobKind::Refine, "jobs-partial task").step(Step::new("a", "a").expect(Shape::Rows));
    let mut handle = jobs.submit(spec).unwrap();
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Completed);
    let found = replies(&events)[0].json.clone().unwrap();
    assert_eq!(found.quality, Quality::Salvaged);
    assert_eq!(found.value, json!([{"id":1,"new_tags":["a"]},{"id":2,"new_tags":["b"]}]));
    assert_eq!(jobs.snapshot().history[0].steps[0].json.as_deref(), Some("salvaged"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_token_fails_the_job_once_without_retries_and_marks_the_gateway() {
    let (mock, rt, jobs) = rig("jobs-401").await;
    mock.script_complete(401, json!({"error": "Unauthorized"}), Duration::ZERO);
    let mut handle = jobs.submit(one_step("jobs-401 task", "ping")).unwrap();
    let events = finish(&mut handle).await;
    let JobOutcome::Failed(error) = outcome(&events) else { panic!("{events:?}") };
    assert_eq!((error.kind, error.attempts, error.needs_repair()), (ErrorKind::Auth, 1, true));
    assert!(error.message.contains("Pair again"), "{}", error.message);
    assert_eq!(complete_calls(&mock).len(), 1, "a refused token is never retried");
    assert!(events.iter().all(|e| !matches!(e, JobEvent::Retrying { .. })));
    assert_eq!(rt.health(), Health::Unauthorized, "the Re-pair attention of health applies");
    assert!(matches!(task_state("jobs-401 task"), Some((TaskState::Failed, _, _))));
    match jobs.submit(one_step("jobs-401 again", "ping")) {
        Err(SubmitError::Unavailable { repair: true, reason }) => assert!(reason.contains("refused the token"), "{reason}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(complete_calls(&mock).len(), 1, "nothing was sent after the refusal");
}

#[tokio::test(flavor = "multi_thread")]
async fn jobs_are_refused_while_the_gateway_is_not_online() {
    let mock = MockGateway::start_with(MockOptions { token: Some("good".into()), ..Default::default() }).await;
    let rt = ZcRuntime::new(Handle::current(), Some(TokenStore::at(temp_dir("jobs-off"))));
    let jobs = ZcJobs::new(rt.clone(), JobsConfig::quick());
    assert!(matches!(jobs.availability(), Availability::Unavailable { repair: false, .. }));
    assert!(jobs.availability().reason().unwrap().contains("not set up"));
    rt.save_token("wrong").unwrap();
    rt.set_profile(ZcProfile { host: mock.url(), agent: "default".into(), ..Default::default() });
    until("refused", || (rt.health() == Health::Unauthorized).then_some(())).await;
    assert!(matches!(jobs.submit(one_step("jobs-off task", "x")), Err(SubmitError::Unavailable { repair: true, .. })));
    assert!(complete_calls(&mock).is_empty());
    assert_eq!(task_state("jobs-off task"), None, "a refused submission leaves no task");
    rt.save_token("good").unwrap();
    until("online", || rt.health().is_online().then_some(())).await;
    assert!(jobs.availability().is_ready());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_error_is_retried_with_backoff_until_it_succeeds() {
    let (mock, _rt, jobs) = rig("jobs-5xx").await;
    mock.script_complete(503, json!({"error": "busy"}), Duration::ZERO);
    mock.script_complete(502, json!({"error": "bad gateway"}), Duration::ZERO);
    mock.script_complete_reply("finally");
    let mut handle = jobs.submit(one_step("jobs-5xx task", "ping")).unwrap();
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Completed);
    let retries: Vec<(u32, Duration)> = events.iter().filter_map(|e| if let JobEvent::Retrying { attempt, wait, .. } = e { Some((*attempt, *wait)) } else { None }).collect();
    assert_eq!(retries, [(2, Duration::from_millis(5)), (3, Duration::from_millis(10))], "the wait doubles");
    assert_eq!(replies(&events)[0].attempts, 3);
    assert_eq!(complete_calls(&mock).len(), 3);
    let totals = jobs.snapshot().totals;
    assert_eq!((totals.calls, totals.retries), (3, 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_that_keeps_failing_ends_the_job_after_the_attempt_limit() {
    let (mock, _rt, jobs) = rig("jobs-5xx-fail").await;
    for _ in 0..5 {
        mock.script_complete(500, json!({"error": "boom"}), Duration::ZERO);
    }
    let mut handle = jobs.submit(one_step("jobs-5xx-fail task", "ping")).unwrap();
    let events = finish(&mut handle).await;
    let JobOutcome::Failed(error) = outcome(&events) else { panic!("{events:?}") };
    assert_eq!((error.kind, error.attempts), (ErrorKind::Other, 3));
    assert!(error.message.contains("boom"), "{}", error.message);
    assert_eq!(complete_calls(&mock).len(), 3);
    assert!(matches!(task_state("jobs-5xx-fail task"), Some((TaskState::Failed, _, _))));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_error_that_repeating_cannot_fix_is_not_retried() {
    let (mock, _rt, jobs) = rig("jobs-400").await;
    let mut handle = jobs.submit(JobSpec::new(JobKind::Manual, "jobs-400 task").hint("no-such-route").step(Step::new("a", "ping"))).unwrap();
    let events = finish(&mut handle).await;
    let JobOutcome::Failed(error) = outcome(&events) else { panic!("{events:?}") };
    assert_eq!((error.kind, error.attempts), (ErrorKind::Rejected, 1));
    assert!(error.message.contains("no-such-route"), "the hint reached the gateway: {}", error.message);
    assert_eq!(complete_calls(&mock).len(), 1);
    assert_eq!(complete_calls(&mock)[0]["hint"], "no-such-route");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unreachable_gateway_ends_the_job_and_turns_health_offline() {
    let (mock, rt, jobs) = rig("jobs-down").await;
    drop(mock);
    let handle = jobs.submit(one_step("jobs-down task", "ping")).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), handle.wait()).await.unwrap();
    let JobOutcome::Failed(error) = result.outcome else { panic!("{:?}", result.outcome) };
    assert_eq!((error.kind, error.attempts), (ErrorKind::Unreachable, 3));
    assert!(matches!(rt.health(), Health::Offline(_)), "{:?}", rt.health());
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelling_between_requests_stops_the_job_and_keeps_what_it_made() {
    let (mock, _rt, jobs) = rig("jobs-cancel").await;
    mock.script_complete_reply("first");
    mock.script_complete(200, json!({"response": "second", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    mock.script_complete_reply("third");
    let spec = JobSpec::new(JobKind::Refine, "jobs-cancel task").steps(["a", "b", "c"].map(|p| Step::new(p, p)));
    let mut handle = jobs.submit(spec).unwrap();
    let mut seen = Vec::new();
    let started = Instant::now();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(8), handle.next()).await.unwrap().unwrap();
        if matches!(event, JobEvent::Reply(_)) {
            tokio::time::sleep(Duration::from_millis(150)).await;
            handle.cancel();
        }
        let done = matches!(event, JobEvent::Finished(_));
        seen.push(event);
        if done {
            break;
        }
    }
    assert!(started.elapsed() < Duration::from_secs(5), "the 30 s request was abandoned");
    assert_eq!(outcome(&seen), JobOutcome::Cancelled);
    assert_eq!(replies(&seen).len(), 1);
    assert_eq!(complete_calls(&mock).len(), 2, "the third request was never sent");
    assert_eq!(task_state("jobs-cancel task").map(|t| t.0), Some(TaskState::Cancelled));
    let record = jobs.snapshot().history[0].clone();
    assert_eq!((record.steps.len(), record.steps[0].response.as_deref(), record.steps[1].error.as_deref()), (2, Some("first"), Some("cancelled")));
    assert_eq!(jobs.snapshot().totals.cancelled, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelling_from_the_task_center_stops_the_running_job() {
    let (mock, _rt, jobs) = rig("jobs-task-cancel").await;
    mock.script_complete(200, json!({"response": "late", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    let mut handle = jobs.submit(one_step("jobs-task-cancel task", "ping")).unwrap();
    until("the request", || (!complete_calls(&mock).is_empty()).then_some(())).await;
    let id = crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.title == "jobs-task-cancel task").map(|t| t.id)).unwrap();
    crate::tasks::with(|r| r.request_cancel(id));
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Cancelled);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_during_the_backoff_wait_ends_the_job_without_another_request() {
    let (mock, rt, _) = rig("jobs-backoff-cancel").await;
    let slow = JobsConfig { backoff_initial: Duration::from_secs(30), backoff_max: Duration::from_secs(30), ..JobsConfig::quick() };
    let jobs = ZcJobs::new(rt, slow);
    mock.script_complete(503, json!({"error": "busy"}), Duration::ZERO);
    let mut handle = jobs.submit(one_step("jobs-backoff-cancel task", "ping")).unwrap();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), handle.next()).await.unwrap().unwrap();
        if matches!(event, JobEvent::Retrying { .. }) {
            break;
        }
    }
    handle.cancel();
    assert_eq!(outcome(&finish(&mut handle).await), JobOutcome::Cancelled);
    assert_eq!(complete_calls(&mock).len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn jobs_run_one_at_a_time_and_a_queued_job_can_be_cancelled_at_once() {
    let (mock, _rt, jobs) = rig("jobs-queue").await;
    mock.script_complete(200, json!({"response": "slow", "model": "m", "hint": "quick"}), Duration::from_millis(400));
    mock.script_complete_reply("next");
    let mut first = jobs.submit(one_step("jobs-queue first", "1")).unwrap();
    until("running", || jobs.snapshot().running.map(|_| ())).await;
    let mut second = jobs.submit(one_step("jobs-queue second", "2")).unwrap();
    let mut third = jobs.submit(one_step("jobs-queue third", "3")).unwrap();
    let snap = jobs.snapshot();
    assert_eq!(snap.queue.iter().map(|q| q.title.as_str()).collect::<Vec<_>>(), ["jobs-queue second", "jobs-queue third"]);
    assert!(matches!(second.poll()[0], JobEvent::Queued { position: 2 }));
    assert!(matches!(third.poll()[0], JobEvent::Queued { position: 3 }));
    assert!(third.id() > second.id() && second.id() > first.id());
    third.cancel();
    assert_eq!(outcome(&finish(&mut third).await), JobOutcome::Cancelled);
    assert_eq!(jobs.snapshot().queue.len(), 1);
    assert_eq!(task_state("jobs-queue third").map(|t| t.0), Some(TaskState::Cancelled));
    let started = Instant::now();
    assert_eq!(outcome(&finish(&mut first).await), JobOutcome::Completed);
    let first_done = started.elapsed();
    assert_eq!(outcome(&finish(&mut second).await), JobOutcome::Completed);
    let calls = complete_calls(&mock);
    assert_eq!(calls.iter().map(|c| c["prompt"].as_str().unwrap()).collect::<Vec<_>>(), ["1", "2"], "the cancelled job never reached the gateway and the others ran in order");
    assert!(first_done >= Duration::from_millis(200));
    assert!(!jobs.cancel(first.id()), "a finished job cannot be cancelled");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_queue_has_a_depth_limit() {
    let (mock, rt, _) = rig("jobs-depth").await;
    let jobs = ZcJobs::new(rt, JobsConfig { queue_limit: 2, ..JobsConfig::quick() });
    mock.script_complete(200, json!({"response": "slow", "model": "m", "hint": "quick"}), Duration::from_secs(30));
    let _running = jobs.submit(one_step("jobs-depth 1", "1")).unwrap();
    until("running", || jobs.snapshot().running.map(|_| ())).await;
    let _a = jobs.submit(one_step("jobs-depth 2", "2")).unwrap();
    let _b = jobs.submit(one_step("jobs-depth 3", "3")).unwrap();
    assert_eq!(jobs.submit(one_step("jobs-depth 4", "4")).unwrap_err(), SubmitError::QueueFull { limit: 2 });
    assert_eq!(task_state("jobs-depth 4"), None);
    assert_eq!(jobs.cancel_all(), 3);
    until("idle", || jobs.is_idle().then_some(())).await;
    assert_eq!(jobs.snapshot().totals.cancelled, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_over_the_input_cap_is_refused_before_it_is_sent() {
    let (mock, _rt, jobs) = rig("jobs-oversize").await;
    let big = Step::new("big", "x".repeat(zeroclaw_client::types::COMPLETE_INPUT_CAP)).system("system text");
    let err = jobs.submit(JobSpec::new(JobKind::Refine, "jobs-oversize task").step(Step::new("ok", "fine")).step(big)).unwrap_err();
    assert!(matches!(&err, SubmitError::Invalid(why) if why.contains("request 2 of 2")), "{err:?}");
    assert_eq!(jobs.submit(JobSpec::new(JobKind::Manual, "empty")).unwrap_err(), SubmitError::NoSteps);
    assert!(complete_calls(&mock).is_empty());
    assert!(jobs.is_idle());
}

#[tokio::test(flavor = "multi_thread")]
async fn batched_requests_all_fit_the_cap_and_run_in_order() {
    let (mock, _rt, jobs) = rig("jobs-batches").await;
    let sizes = vec![1_500usize; 90];
    let plan = batch::plan_for_gateway(&sizes, 2_000);
    assert!(plan.batches.len() >= 3 && plan.oversize.is_empty());
    let steps: Vec<Step> = plan
        .batches
        .iter()
        .enumerate()
        .map(|(i, range)| Step::new(format!("batch {}", i + 1), format!("rows {}..{} {}", range.start, range.end, "x".repeat(sizes[range.clone()].iter().sum::<usize>()))).system("s".repeat(2_000)))
        .collect();
    let count = steps.len();
    for i in 0..count {
        mock.script_complete_reply(&format!("reply {i}"));
    }
    let mut handle = jobs.submit(JobSpec::new(JobKind::Refine, "jobs-batches task").steps(steps)).unwrap();
    let events = finish(&mut handle).await;
    assert_eq!(outcome(&events), JobOutcome::Completed);
    let texts: Vec<String> = replies(&events).iter().map(|r| r.text.clone()).collect();
    assert_eq!(texts, (0..count).map(|i| format!("reply {i}")).collect::<Vec<_>>());
    for call in complete_calls(&mock) {
        let size = call["prompt"].as_str().unwrap().len() + call["system"].as_str().unwrap().len();
        assert!(size <= zeroclaw_client::types::COMPLETE_INPUT_CAP, "{size}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_log_never_holds_the_token_and_clips_long_text() {
    let (mock, rt, _) = rig("jobs-redact").await;
    let token = mock.token().unwrap().to_string();
    let jobs = ZcJobs::new(rt, JobsConfig { text_cap: 64, ..JobsConfig::quick() });
    mock.script_complete_reply(&format!("echoed {token} {}", "y".repeat(200)));
    let mut handle = jobs.submit(JobSpec::new(JobKind::Manual, "jobs-redact task").step(Step::new("a", format!("my token is {token}")).system(format!("sys {token}")))).unwrap();
    let events = finish(&mut handle).await;
    assert!(!replies(&events)[0].text.contains(&token));
    let record = jobs.snapshot().history[0].clone();
    let step = &record.steps[0];
    for text in [step.prompt.as_str(), step.system.as_deref().unwrap(), step.response.as_deref().unwrap()] {
        assert!(!text.contains(&token), "{text}");
    }
    assert!(step.prompt.contains("***"));
    assert!(step.response.as_deref().unwrap().contains("[cut: "), "the stored reply is clipped to the cap");
    assert!(step.response_bytes > 200, "the byte count is of the whole reply");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_history_is_bounded_and_can_be_cleared() {
    let (mock, rt, _) = rig("jobs-history").await;
    let jobs = ZcJobs::new(rt, JobsConfig { history: 3, ..JobsConfig::quick() });
    for i in 0..5 {
        mock.script_complete_reply(&format!("r{i}"));
        let mut h = jobs.submit(one_step(&format!("jobs-history {i}"), "p")).unwrap();
        finish(&mut h).await;
    }
    let snap = jobs.snapshot();
    assert_eq!(snap.history.iter().map(|r| r.title.as_str()).collect::<Vec<_>>(), ["jobs-history 4", "jobs-history 3", "jobs-history 2"], "newest first");
    assert_eq!(snap.totals.jobs, 5, "the totals keep counting");
    jobs.clear_history();
    assert!(jobs.snapshot().history.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_profile_hint_is_the_default_for_every_job() {
    let (mock, rt, jobs) = rig("jobs-hint").await;
    assert_eq!(jobs.default_hint(), "quick");
    let mut profile = rt.profile();
    profile.complete_hint = "  ".into();
    rt.set_profile(profile);
    assert_eq!(jobs.default_hint(), "quick", "a blank hint reads as the default");
    mock.script_complete_reply("a");
    finish(&mut jobs.submit(one_step("jobs-hint a", "p")).unwrap()).await;
    assert_eq!(complete_calls(&mock)[0]["hint"], "quick");
    assert_eq!(jobs.snapshot().history[0].hint, "quick");
}

/// Serves a mock gateway on loopback for manual runs (`ZC_MOCK_DIR=<dir>`): each `reply-*.txt` in the folder becomes the answer of the next `/api/complete` call and is removed; the requests are logged to `requests.log` until `stop` appears.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "serves a loopback mock gateway for manual runs"]
async fn manual_jobs_gateway() {
    let dir = std::path::PathBuf::from(std::env::var("ZC_MOCK_DIR").expect("set ZC_MOCK_DIR to a scratch folder"));
    std::fs::create_dir_all(&dir).unwrap();
    let mock = MockGateway::start().await;
    std::fs::write(dir.join("token"), mock.token().unwrap()).unwrap();
    std::fs::write(dir.join("url"), mock.url()).unwrap();
    let mut logged = 0;
    for _ in 0..30_000 {
        if dir.join("stop").exists() {
            break;
        }
        let mut replies: Vec<std::path::PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("reply-") && n.ends_with(".txt"))).collect();
        replies.sort();
        for path in replies {
            mock.script_complete_reply(&std::fs::read_to_string(&path).unwrap());
            std::fs::remove_file(path).unwrap();
        }
        let requests = mock.requests();
        for r in requests.iter().skip(logged).filter(|r| r.path == "/api/complete") {
            let line = format!("{}\n", serde_json::json!({"hint": r.body["hint"], "system_bytes": r.body["system"].as_str().map(str::len), "prompt_bytes": r.body["prompt"].as_str().map(str::len), "prompt": r.body["prompt"]}));
            std::fs::OpenOptions::new().create(true).append(true).open(dir.join("requests.log")).and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes())).unwrap();
        }
        logged = requests.len();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[test]
fn backoff_doubles_up_to_the_limit() {
    let cfg = JobsConfig::default();
    let delays: Vec<u64> = (1..=6).map(|a| cfg.delay(a).as_millis() as u64).collect();
    assert_eq!(delays, [800, 1600, 3200, 6400, 10_000, 10_000]);
}

#[test]
fn error_and_outcome_labels_read_as_sentences() {
    assert_eq!(JobOutcome::Completed.label(), "completed");
    assert!(JobOutcome::Failed(JobError { kind: ErrorKind::Other, message: "boom".into(), attempts: 2 }).label().starts_with("failed: boom"));
    assert_eq!(SubmitError::QueueFull { limit: 8 }.to_string(), "8 ZeroClaw jobs are already waiting");
    assert_eq!(JobKind::Refine.task_kind(), crate::tasks::TaskKind::Refine);
    assert_eq!(JobKind::Ask.task_kind(), crate::tasks::TaskKind::Assistant);
}
