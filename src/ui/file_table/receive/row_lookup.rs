impl crate::ui::file_table::FileExplorer {
    // Row position in `table` for `path`; repairs `table_index` when it no longer matches the table.
    pub(super) fn table_pos(&mut self, path: &str) -> Option<usize> {
        let valid = |this: &Self, idx: usize| this.table.get(idx).is_some_and(|r| r.path == path);
        match self.table_index.get(path).copied() {
            Some(idx) if valid(self, idx) => return Some(idx),
            Some(_) => {}
            None if self.table_index.len() == self.table.len() => return None,
            None => {}
        }
        self.rebuild_table_index();
        self.table_index.get(path).copied().filter(|&idx| valid(self, idx))
    }

    pub(super) fn rebuild_table_index(&mut self) {
        self.table_index.clear();
        for (i, r) in self.table.iter().enumerate() {
            self.table_index.insert(r.path.clone(), i);
        }
    }
}
