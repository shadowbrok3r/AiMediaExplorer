use super::cloud::reply_to_proposal;
use super::*;
use crate::Thumbnail;

fn row(path: &str, category: Option<&str>, tags: &[&str], caption: Option<&str>) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
    t.path = path.to_string();
    t.category = category.map(str::to_string);
    t.tags = tags.iter().map(|s| s.to_string()).collect();
    t.caption = caption.map(str::to_string);
    t
}

fn strings(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn neighbours() -> Candidates {
    let rows = vec![
        row("/n/1.jpg", Some("beach"), &["sea", "sand", "sun"], Some("A sunny beach")),
        row("/n/2.jpg", Some("beach"), &["sea", "sand"], Some("A sunny beach")),
        row("/n/3.jpg", Some("beach"), &["sea", "boat"], Some("Boats in the bay")),
        row("/n/4.jpg", Some("harbour"), &["sea", "boat"], None),
    ];
    Candidates::from_rows(rows.iter())
}

#[test]
fn the_heuristic_orders_tags_by_votes_and_keeps_twelve() {
    let mut h = HeuristicReranker;
    assert_eq!(h.rerank_tags("", &strings(&["Sea", "sea", "Boat", "boat", "boat", "sun"])), strings(&["boat", "sea", "sun"]));
    let many: Vec<String> = (0..30).map(|i| format!("t{i:02}")).collect();
    assert_eq!(h.rerank_tags("", &many).len(), 12);
    assert!(h.rerank_tags("", &[]).is_empty());
}

#[test]
fn the_heuristic_picks_the_majority_category_and_the_most_common_text() {
    let mut h = HeuristicReranker;
    assert_eq!(h.rerank_category("", &strings(&["Beach", "beach ", "harbour"])), Some("beach".to_string()));
    assert_eq!(h.rerank_category("", &[]), None);
    assert_eq!(h.rerank_caption("", &strings(&["a", "b", "b", " "])), Some("b".to_string()));
    assert_eq!(h.rerank_description("", &strings(&["  ", ""])), None);
}

#[test]
fn neighbours_are_counted_with_their_metadata() {
    let c = neighbours();
    assert_eq!((c.neighbours, c.categories.len(), c.captions.len()), (4, 4, 3));
    assert_eq!(c.tags.iter().filter(|t| *t == "sea").count(), 4);
}

#[test]
fn confidence_is_the_mean_support_of_tags_and_category() {
    let c = neighbours();
    let fields = Fields { category: Some("beach".into()), tags: strings(&["sea", "boat"]), ..Default::default() };
    // category 3 of 4, "sea" 4 of 4 neighbours, "boat" 2 of 4
    let expected = (0.75 + 1.0 + 0.5) / 3.0;
    assert!((confidence(&c, &fields).unwrap() - expected).abs() < 1e-6);
    let caption_only = Fields { caption: Some("A sunny beach".into()), ..Default::default() };
    assert!((confidence(&c, &caption_only).unwrap() - 2.0 / 3.0).abs() < 1e-6, "a caption counts when it is all that is proposed");
    assert_eq!(confidence(&c, &Fields::default()), None);
    assert_eq!(confidence(&Candidates::default(), &fields).unwrap(), 0.0, "no neighbours means no support");
}

#[test]
fn a_proposal_leaves_out_what_the_file_already_has() {
    let current = row("/a/x.jpg", Some("Beach"), &["Sea", "sand"], Some("a sunny beach"));
    let mut h = HeuristicReranker;
    let p = build_proposal(&current, &neighbours(), &mut h).expect("the tags differ");
    assert_eq!(p.new_category, None, "the same category ignoring case is not proposed");
    assert_eq!(p.new_caption, None);
    assert_eq!(p.generator, "Heuristic");
    assert_eq!(p.new_tags, strings(&["sea", "boat", "sand", "sun"]));
    assert!(p.confidence.unwrap() > 0.0);
    assert_eq!(p.current_tags, strings(&["Sea", "sand"]));

    let same_tags = row("/a/y.jpg", Some("beach"), &["sea", "boat", "sand", "sun"], Some("A sunny beach"));
    assert!(build_proposal(&same_tags, &neighbours(), &mut h).is_none(), "nothing new, no proposal");
}

struct Silent;

impl Reranker for Silent {
    fn name(&self) -> &'static str {
        "Silent"
    }
    fn rerank_tags(&mut self, _: &str, _: &[String]) -> Vec<String> {
        Vec::new()
    }
    fn rerank_category(&mut self, _: &str, _: &[String]) -> Option<String> {
        None
    }
    fn rerank_caption(&mut self, _: &str, _: &[String]) -> Option<String> {
        None
    }
    fn rerank_description(&mut self, _: &str, _: &[String]) -> Option<String> {
        None
    }
}

#[test]
fn a_reranker_with_nothing_to_say_falls_back_to_the_heuristic_and_says_so() {
    let p = build_proposal(&row("/a/x.jpg", None, &[], None), &neighbours(), &mut Silent).expect("the fallback proposes");
    assert_eq!(p.generator, "Heuristic (fallback)");
    assert_eq!(p.new_category.as_deref(), Some("beach"));
    let mut named = HeuristicReranker;
    assert_eq!(build_proposal(&row("/a/x.jpg", None, &[], None), &neighbours(), &mut named).unwrap().generator, "Heuristic");
}

#[test]
fn the_default_reranker_is_the_heuristic() {
    assert_eq!(RerankerType::default(), RerankerType::Heuristic);
}

#[test]
fn model_replies_are_read_from_plain_fenced_and_wrapped_json() {
    let plain = r#"{"new_category":" Travel ","new_tags":["a"," b ",""],"new_caption":null,"new_description":"  "}"#;
    assert_eq!(parse_cloud_reply(plain), Some(CloudReply { category: Some("Travel".into()), tags: strings(&["a", "b"]), caption: None, description: None }));
    let fenced = "Here you go:\n```json\n{\"new_caption\":\"A dog\"}\n```\nDone.";
    assert_eq!(parse_cloud_reply(fenced).unwrap().caption.as_deref(), Some("A dog"));
    let wrapped = "Sure! {\"new_tags\":[\"x\"]} hope that helps";
    assert_eq!(parse_cloud_reply(wrapped).unwrap().tags, strings(&["x"]));
    assert_eq!(parse_cloud_reply("no json here"), None);
    assert_eq!(parse_cloud_reply("[1,2]"), None, "an array is not a reply");
}

#[test]
fn a_cloud_reply_proposes_only_what_differs() {
    let current = row("/a/x.jpg", Some("travel"), &["Sun", "sea"], Some("A beach"));
    let reply = CloudReply { category: Some("Travel".into()), tags: strings(&["sea", "sun"]), caption: Some("A sunny beach".into()), description: None };
    let p = reply_to_proposal(&current, reply, "cloud:test").expect("the caption differs");
    assert_eq!((p.new_category, p.new_tags.len(), p.new_caption.as_deref()), (None, 0, Some("A sunny beach")));
    assert_eq!(p.generator, "cloud:test");
    assert!(reply_to_proposal(&current, CloudReply::default(), "cloud:test").is_none());
}

#[test]
fn the_cloud_prompt_names_the_file_and_asks_for_json() {
    let prompt = cloud_prompt(&row("/a/x.jpg", Some("travel"), &["sun", "sea"], Some("A beach")));
    for part in ["/a/x.jpg", "travel", "sun, sea", "A beach", "new_tags"] {
        assert!(prompt.contains(part), "{part}");
    }
}

#[test]
fn the_provider_config_takes_the_model_of_the_panel_or_a_default() {
    let named = provider_config("openrouter", " vendor/model ");
    assert_eq!((named.model.as_str(), named.base_url.as_deref()), ("vendor/model", Some("https://openrouter.ai/api/v1")));
    assert!(!provider_config("openai", "").model.is_empty());
    assert_eq!(provider_config("groq", "m").provider, "groq");
}

#[test]
fn jina_scores_order_the_candidates_best_first() {
    assert_eq!(jina::rank(strings(&["a", "b", "c"]), vec![0.1, 0.9, 0.5]), strings(&["b", "c", "a"]));
}

#[test]
fn a_run_stops_when_asked_and_reports_progress() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let seen = AtomicU64::new(0);
    let progress = |done: u64, _total: u64| seen.store(done, Ordering::SeqCst);
    let stop = || true;
    let ctl = RunCtl { should_stop: &stop, progress: &progress };
    assert!((ctl.should_stop)());
    (ctl.progress)(3, 9);
    assert_eq!(seen.load(Ordering::SeqCst), 3);
}
