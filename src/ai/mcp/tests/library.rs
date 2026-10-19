use surrealdb::types::SurrealValue;
use serde_json::json;

use crate::ai::mcp::test_support::*;

fn names(out: &crate::ai::mcp::error::ToolOutput) -> Vec<String> {
    Fixture::paths(out).iter().map(|p| p.rsplit('/').next().unwrap().to_string()).collect()
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

#[tokio::test]
async fn search_matches_words_in_every_field_ignoring_case() {
    let f = fixture().await;
    let all_beach = f.call("library_search", json!({"query": "BEACH"})).await.unwrap();
    assert_eq!(sorted(names(&all_beach)), vec!["beach1.png", "beach2.png", "beach3.png"]);
    assert_eq!(all_beach.data["total"], 3);
    let by_tag = f.call("library_search", json!({"query": "dusk"})).await.unwrap();
    assert_eq!(names(&by_tag), vec!["beach2.png"]);
    let by_description = f.call("library_search", json!({"query": "wide sandy"})).await.unwrap();
    assert_eq!(names(&by_description), vec!["beach1.png"]);
    let by_category = f.call("library_search", json!({"query": "landscape"})).await.unwrap();
    assert_eq!(sorted(names(&by_category)), vec!["beach1.png", "beach2.png"]);
    let both_words = f.call("library_search", json!({"query": "beach sea"})).await.unwrap();
    assert_eq!(names(&both_words), vec!["beach1.png"]);
    let nothing = f.call("library_search", json!({"query": "submarine"})).await.unwrap();
    assert_eq!(nothing.data["total"], 0);
    assert!(nothing.data["next_offset"].is_null());
}

#[tokio::test]
async fn search_input_is_bound_never_spliced_into_the_query() {
    let f = fixture().await;
    for attack in ["\"; DELETE thumbnails; \"", "' OR true OR '", "$t0", "beach\\\"; REMOVE TABLE thumbnails; --"] {
        let out = f.call("library_search", json!({"query": attack, "tags": [attack], "category": attack, "folder": attack, "file_type": attack})).await.unwrap();
        assert_eq!(out.data["total"], 0, "{attack}");
    }
    assert_eq!(row_count(&f.db, "thumbnails").await, 5);
}

#[tokio::test]
async fn search_filters_combine() {
    let f = fixture().await;
    let tagged = f.call("library_search", json!({"tags": ["beach"]})).await.unwrap();
    assert_eq!(sorted(names(&tagged)), vec!["beach1.png", "beach2.png"]);
    let two_tags = f.call("library_search", json!({"tags": ["beach", "sea"]})).await.unwrap();
    assert_eq!(names(&two_tags), vec!["beach1.png"]);
    let category = f.call("library_search", json!({"category": "LANDSCAPE"})).await.unwrap();
    assert_eq!(category.data["total"], 2);
    let ext = f.call("library_search", json!({"file_type": ".PNG"})).await.unwrap();
    assert_eq!(ext.data["total"], 5);
    let jpg = f.call("library_search", json!({"file_type": "jpg"})).await.unwrap();
    assert_eq!(jpg.data["total"], 0);
    let sub = f.call("library_search", json!({"folder": f.path("photos/sub")})).await.unwrap();
    assert_eq!(names(&sub), vec!["beach3.png"]);
    let partial_folder = f.call("library_search", json!({"folder": f.path("photos/su")})).await.unwrap();
    assert_eq!(partial_folder.data["total"], 0, "a folder filter needs a whole folder name");
    let photos = f.call("library_search", json!({"folder": format!("{}/", f.path("photos"))})).await.unwrap();
    assert_eq!(photos.data["total"], 5);
    let holidays = f.call("library_search", json!({"collection": "holidays"})).await.unwrap();
    assert_eq!(sorted(names(&holidays)), vec!["beach1.png", "beach2.png"]);
    let combined = f.call("library_search", json!({"collection": "Holidays", "query": "dusk", "undescribed": true})).await.unwrap();
    assert_eq!(names(&combined), vec!["beach2.png"]);
    let undescribed = f.call("library_search", json!({"undescribed": true})).await.unwrap();
    assert_eq!(sorted(names(&undescribed)), vec!["beach2.png", "beach3.png"]);
    let untagged = f.call("library_search", json!({"untagged": true})).await.unwrap();
    assert_eq!(names(&untagged), vec!["beach3.png"]);
}

#[tokio::test]
async fn search_errors_are_specific() {
    let f = fixture().await;
    let missing = f.call("library_search", json!({"collection": "Nope"})).await.unwrap_err();
    assert_eq!(missing.kind(), "not_found");
    assert!(missing.message().contains("list_collections"));
    let long = f.call("library_search", json!({"query": "a b c d e f g"})).await.unwrap_err();
    assert_eq!(long.kind(), "invalid_arguments");
    let bad = f.call("library_search", json!({"limit": "ten"})).await.unwrap_err();
    assert_eq!(bad.kind(), "invalid_arguments");
    assert!(bad.message().contains("bad arguments"));
}

#[tokio::test]
async fn search_pages_and_sorts() {
    let f = fixture().await;
    let first = f.call("library_search", json!({"limit": 2})).await.unwrap();
    assert_eq!(first.data["returned"], 2);
    assert_eq!(first.data["next_offset"], 2);
    assert_eq!(first.data["total"], 5);
    let last = f.call("library_search", json!({"limit": 2, "offset": 4})).await.unwrap();
    assert_eq!(last.data["returned"], 1);
    assert!(last.data["next_offset"].is_null());
    let by_name = f.call("library_search", json!({"sort": "filename", "descending": true})).await.unwrap();
    assert_eq!(names(&by_name), vec!["forest1.png", "city1.png", "beach3.png", "beach2.png", "beach1.png"]);
    let by_size = f.call("library_search", json!({"sort": "size", "descending": true, "limit": 1})).await.unwrap();
    assert_eq!(names(&by_size), vec!["city1.png"]);
    let clamped = f.call("library_search", json!({"limit": 0})).await.unwrap();
    assert_eq!(clamped.data["limit"], 1);
    let huge = f.call("library_search", json!({"limit": 100000})).await.unwrap();
    assert_eq!(huge.data["limit"], 100);
}

#[tokio::test]
async fn search_items_carry_the_metadata_fields() {
    let f = fixture().await;
    let out = f.call("library_search", json!({"query": "wide sandy"})).await.unwrap();
    let item = &out.data["items"][0];
    assert_eq!(item["caption"], "Sunny beach");
    assert_eq!(item["category"], "landscape");
    assert_eq!(item["tags"], json!(["beach", "sea", "summer"]));
    assert_eq!(item["file_type"], "png");
    assert_eq!(item["filename"], "beach1.png");
    assert!(item["folder"].as_str().unwrap().ends_with("photos"));
    assert!(item.get("thumbnail_b64").is_none());
}

#[tokio::test]
async fn semantic_search_ranks_by_similarity_and_pages() {
    let f = fixture().await;
    let out = f.call("library_semantic_search", json!({"query": "a sunny beach holiday", "top_k": 2})).await.unwrap();
    assert_eq!(names(&out), vec!["beach1.png", "beach2.png"]);
    let sims: Vec<f64> = out.data["items"].as_array().unwrap().iter().map(|i| i["similarity"].as_f64().unwrap()).collect();
    assert!(sims[0] > sims[1] && sims[1] > 0.95 && sims[0] <= 1.0, "{sims:?}");
    assert_eq!(out.data["items"][0]["rank"], 1);
    assert!(out.data["items"][0]["distance"].as_f64().unwrap() < 0.01);
    assert_eq!(out.data["items"][0]["caption"], "Sunny beach");
    assert!(out.data["gpu"]["job_id"].as_u64().unwrap() >= 1);
    let next = f.call("library_semantic_search", json!({"query": "beach", "top_k": 2, "offset": 2})).await.unwrap();
    assert_eq!(sorted(names(&next)), vec!["city1.png", "forest1.png"]);
    assert_eq!(next.data["items"][0]["rank"], 3);
    assert!(next.data["items"][0]["similarity"].as_f64().unwrap().abs() < 0.01, "unrelated topics are orthogonal");
    let strict = f.call("library_semantic_search", json!({"query": "beach", "min_similarity": 0.9})).await.unwrap();
    assert_eq!(strict.data["returned"], 2);
    assert_eq!(f.clip.text_calls.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[tokio::test]
async fn semantic_search_returns_more_than_64_results() {
    let f = fixture().await;
    let embeddings: Vec<SeedBulk> = (0..150).map(|i| SeedBulk { path: format!("/bulk/{i:03}.jpg"), embedding: dense_vector(i) }).collect();
    f.db.query("INSERT INTO clip_embeddings $rows").bind(("rows", embeddings)).await.unwrap().check().unwrap();
    let out = f.call("library_semantic_search", json!({"query": "something else entirely", "top_k": 120})).await.unwrap();
    assert_eq!(out.data["returned"], 120, "the old fixed limit of 64 is gone");
    let deep = f.call("library_semantic_search", json!({"query": "something else entirely", "top_k": 50, "offset": 100})).await.unwrap();
    assert_eq!(deep.data["returned"], 50);
    let limit = f.call("library_semantic_search", json!({"query": "x", "top_k": 200, "offset": 801})).await.unwrap_err();
    assert_eq!(limit.kind(), "invalid_arguments");
}

fn dense_vector(seed: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..DIMS).map(|j| ((seed as f32 + 1.0) * 13.7 + j as f32 * 0.37 + (seed * j) as f32 * 0.001).sin()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

#[derive(Clone, Debug, SurrealValue)]
struct SeedBulk {
    path: String,
    embedding: Vec<f32>,
}

#[tokio::test]
async fn semantic_search_validates_the_query_and_reports_model_failures() {
    let f = fixture().await;
    assert_eq!(f.call("library_semantic_search", json!({"query": "   "})).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.call("library_semantic_search", json!({"query": "x".repeat(513)})).await.unwrap_err().kind(), "invalid_arguments");
    f.clip.fail.store(true, std::sync::atomic::Ordering::SeqCst);
    let err = f.call("library_semantic_search", json!({"query": "beach"})).await.unwrap_err();
    assert_eq!(err.kind(), "failed");
    assert!(err.message().contains("broken"));
    assert_eq!(f.ctx.gpu.status().failed, 1);
}

#[tokio::test]
async fn similar_to_uses_stored_embeddings_and_leaves_out_the_item_itself() {
    let f = fixture().await;
    let out = f.call("library_similar_to", json!({"path": f.path("photos/beach1.png"), "top_k": 3})).await.unwrap();
    let found = Fixture::paths(&out);
    assert_eq!(found.len(), 3);
    assert!(!found.contains(&f.path("photos/beach1.png")));
    assert_eq!(found[0], f.path("photos/beach2.png"));
    assert_eq!(f.clip.image_calls.load(std::sync::atomic::Ordering::SeqCst), 0, "a stored embedding needs no model");
}

#[tokio::test]
async fn similar_to_embeds_a_file_without_a_stored_embedding() {
    let f = fixture().await;
    let out = f.call("library_similar_to", json!({"path": f.path("photos/sub/beach3.png"), "top_k": 2})).await.unwrap();
    assert_eq!(f.clip.image_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(sorted(names(&out)), vec!["beach1.png", "beach2.png"]);
}

#[tokio::test]
async fn similar_to_refuses_foreign_and_unknown_paths() {
    let f = fixture().await;
    let outside = f.call("library_similar_to", json!({"path": f.path("outside_library/secret.png")})).await.unwrap_err();
    assert_eq!(outside.kind(), "denied");
    let missing = f.call("library_similar_to", json!({"path": f.path("photos/missing.png")})).await.unwrap_err();
    assert_eq!(missing.kind(), "not_found");
    assert!(missing.message().contains("clip_embed"));
    assert_eq!(f.call("library_similar_to", json!({"path": "  "})).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.call("library_similar_to", json!({"path": "photos/beach1.png"})).await.unwrap_err().kind(), "invalid_arguments");
}

#[tokio::test]
async fn get_item_reports_collections_and_embedding_status() {
    let f = fixture().await;
    let out = f.call("library_get_item", json!({"path": f.path("photos/beach1.png")})).await.unwrap();
    assert_eq!(out.data["collections"], json!(["Holidays"]));
    assert_eq!(out.data["has_description"], true);
    assert_eq!(out.data["clip_embedding"], json!({"stored": true, "dimensions": 1024}));
    assert_eq!(out.data["description"], "A wide sandy beach with blue sea");
    assert!(out.images.is_empty());
    let bare = f.call("library_get_item", json!({"path": f.path("photos/sub/beach3.png")})).await.unwrap();
    assert_eq!(bare.data["clip_embedding"], json!({"stored": false, "dimensions": null}));
    assert_eq!(bare.data["has_description"], false);
    assert_eq!(bare.data["collections"], json!([]));
    let missing = f.call("library_get_item", json!({"path": f.path("photos/none.png")})).await.unwrap_err();
    assert_eq!(missing.kind(), "not_found");
}

#[tokio::test]
async fn get_item_attaches_the_stored_thumbnail_on_request() {
    let f = fixture().await;
    let mut png = Vec::new();
    image::DynamicImage::new_rgb8(16, 16).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let blob = crate::database::thumb_blobs::ThumbBlobIn::from_image_bytes(f.path("photos/beach1.png"), &png).unwrap();
    crate::database::thumb_blobs::put_with(&f.db, vec![blob]).await.unwrap();
    let with = f.call("library_get_item", json!({"path": f.path("photos/beach1.png"), "include_thumbnail": true})).await.unwrap();
    assert_eq!(with.images.len(), 1);
    assert!(with.images[0].mime.starts_with("image/"));
    assert!(!with.images[0].data_base64.is_empty());
    let none = f.call("library_get_item", json!({"path": f.path("photos/forest1.png"), "include_thumbnail": true})).await.unwrap();
    assert!(none.images.is_empty());
    assert_eq!(none.data["thumbnail"], "none stored");
}

#[tokio::test]
async fn stats_count_the_library() {
    let f = fixture().await;
    let out = f.call("library_stats", json!({})).await.unwrap();
    let d = &out.data;
    assert_eq!((d["items"].as_u64(), d["described"].as_u64(), d["tagged"].as_u64(), d["categorized"].as_u64()), (Some(5), Some(3), Some(4), Some(4)));
    assert_eq!((d["with_clip_embedding"].as_u64(), d["without_clip_embedding"].as_u64(), d["collections"].as_u64()), (Some(4), Some(1), Some(2)));
    assert_eq!(d["distinct_categories"], 3);
    assert_eq!(d["top_categories"][0], json!({"name": "landscape", "count": 2}));
    assert_eq!(d["top_tags"][0], json!({"name": "beach", "count": 2}));
    assert_eq!(d["pending_proposals"], 0);
    assert!(d["folders"].as_array().unwrap().iter().any(|p| p.as_str().unwrap().ends_with("photos")));
    assert!(out.text.starts_with("5 items"));
}

#[tokio::test]
async fn collections_and_their_items_are_listed() {
    let f = fixture().await;
    let list = f.call("list_collections", json!({})).await.unwrap();
    assert_eq!(list.data["items"], json!([{"name": "Empty", "count": 0, "note": null}, {"name": "Holidays", "count": 2, "note": null}]));
    let filtered = f.call("list_collections", json!({"contains": "HOL"})).await.unwrap();
    assert_eq!(filtered.data["total"], 1);
    let items = f.call("collection_items", json!({"name": "holidays"})).await.unwrap();
    assert_eq!(names(&items), vec!["beach1.png", "beach2.png"]);
    assert_eq!(items.data["collection"], "Holidays");
    assert_eq!(items.data["total"], 2);
    assert_eq!(items.data["items"][0]["caption"], "Sunny beach");
    let page = f.call("collection_items", json!({"name": "Holidays", "limit": 1, "offset": 1})).await.unwrap();
    assert_eq!(names(&page), vec!["beach2.png"]);
    let empty = f.call("collection_items", json!({"name": "Empty"})).await.unwrap();
    assert_eq!(empty.data["total"], 0);
    assert_eq!(f.call("collection_items", json!({"name": "Nope"})).await.unwrap_err().kind(), "not_found");
}

#[tokio::test]
async fn tags_and_categories_come_with_counts() {
    let f = fixture().await;
    let tags = f.call("list_tags", json!({})).await.unwrap();
    assert_eq!(tags.data["items"][0], json!({"name": "beach", "count": 2}));
    assert_eq!(tags.data["total"], 8);
    let named = f.call("list_tags", json!({"sort": "name", "limit": 2})).await.unwrap();
    assert_eq!(named.data["items"], json!([{"name": "beach", "count": 2}, {"name": "city", "count": 1}]));
    assert_eq!(named.data["next_offset"], 2);
    let some = f.call("list_tags", json!({"contains": "SU"})).await.unwrap();
    assert_eq!(some.data["items"], json!([{"name": "summer", "count": 1}]));
    let categories = f.call("list_categories", json!({})).await.unwrap();
    assert_eq!(categories.data["items"][0], json!({"name": "landscape", "count": 2}));
    assert_eq!(categories.data["total"], 3);
    let filtered = f.call("list_categories", json!({"contains": "urb"})).await.unwrap();
    assert_eq!(filtered.data["items"], json!([{"name": "urban", "count": 1}]));
}

#[tokio::test]
async fn results_never_exceed_the_limits_for_hostile_text() {
    let f = fixture().await;
    let nasty = format!("evil\u{7}\n{}", "A".repeat(10_000));
    f.db.query("UPDATE thumbnails SET description = $d, caption = $d WHERE filename = 'city1.png'").bind(("d", nasty)).await.unwrap().check().unwrap();
    let out = f.call("library_search", json!({"query": "evil"})).await.unwrap();
    let item = &out.data["items"][0];
    let description = item["description"].as_str().unwrap();
    assert!(description.chars().count() <= 600 && !description.contains('\u{7}') && !description.contains('\n'));
    assert!(item["caption"].as_str().unwrap().chars().count() <= 400);
    let detail = f.call("library_get_item", json!({"path": f.path("photos/city1.png")})).await.unwrap();
    assert!(detail.data["description"].as_str().unwrap().chars().count() <= 4000);
}
