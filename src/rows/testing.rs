//! A small row type for exercising the store, the view and the selection without the app's data model.

use std::borrow::Cow;

use super::store::Keyed;
use super::view::{RowFacts, RowKind, SortValue};

#[derive(Clone, Debug, PartialEq)]
pub struct TestRow {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub kind: RowKind,
    pub tag: Option<i64>,
}

impl TestRow {
    pub fn new(path: &str) -> Self {
        Self { path: path.to_string(), name: path.rsplit('/').next().unwrap_or(path).to_string(), size: 0, kind: RowKind::Image, tag: None }
    }

    pub fn size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn kind(mut self, kind: RowKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn tag(mut self, tag: Option<i64>) -> Self {
        self.tag = tag;
        self
    }
}

impl Keyed for TestRow {
    fn key_path(&self) -> &str {
        &self.path
    }
}

impl RowFacts for TestRow {
    fn kind(&self) -> RowKind {
        self.kind
    }

    fn size(&self) -> u64 {
        self.size
    }
}

/// Columns: 0 name, 1 size, 2 tag, 3 kind, 4 path.
pub fn sort_key(row: &TestRow, column: u8) -> SortValue<'_> {
    match column {
        0 => SortValue::Text(Cow::Borrowed(&row.name)),
        1 => SortValue::Int(row.size as i128),
        2 => row.tag.map_or(SortValue::Missing, |t| SortValue::Int(t as i128)),
        3 => SortValue::Int(row.kind as i128),
        4 => SortValue::Text(Cow::Borrowed(&row.path)),
        _ => SortValue::Missing,
    }
}
