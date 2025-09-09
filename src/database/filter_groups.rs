use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterGroup {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<surrealdb::RecordId>,
    pub name: String,
    pub include_images: bool,
    pub include_videos: bool,
    pub include_dirs: bool,
    pub skip_icons: bool,
    pub min_size_bytes: Option<u64>,
    pub max_size_bytes: Option<u64>,
    pub excluded_terms: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<surrealdb::sql::Datetime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated: Option<surrealdb::sql::Datetime>,
}

fn slugify_name(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() { s.push(ch.to_ascii_lowercase()); }
        else if ch.is_whitespace() || ch == '-' { s.push('_'); }
        else { s.push('_'); }
    }
    // trim duplicate underscores
    let mut out = String::with_capacity(s.len());
    let mut prev = '\0';
    for ch in s.chars() {
        if ch == '_' && prev == '_' { continue; }
        out.push(ch);
        prev = ch;
    }
    out.trim_matches('_').to_string()
}

pub async fn save_filter_group(mut g: FilterGroup) -> anyhow::Result<FilterGroup, anyhow::Error> {
    let key = slugify_name(&g.name);
    let rid = surrealdb::RecordId::from_table_key(super::FILTER_GROUPS, key);
    g.id = Some(rid.clone());
    super::DB
        .upsert::<Option<FilterGroup>>(rid)
        .content::<FilterGroup>(g.clone())
        .await?;
    Ok(g)
}

pub async fn list_filter_groups() -> anyhow::Result<Vec<FilterGroup>, anyhow::Error> {
    let res: Vec<FilterGroup> = super::DB.select(super::FILTER_GROUPS).await?;
    Ok(res)
}

pub async fn delete_filter_group_by_name(name: &str) -> anyhow::Result<(), anyhow::Error> {
    let key = slugify_name(name);
    let rid = surrealdb::RecordId::from_table_key(super::FILTER_GROUPS, key);
    let _deleted: Option<FilterGroup> = super::DB.delete(rid).await?;
    Ok(())
}
