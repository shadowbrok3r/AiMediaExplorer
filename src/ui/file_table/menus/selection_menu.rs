use crate::database::Thumbnail;

/// Splits selected rows into the file rows and the paths of the directory rows; no rows split into nothing.
pub fn split_selection<'a>(selected: &[&'a Thumbnail]) -> (Vec<&'a Thumbnail>, Vec<String>) {
    let files = selected.iter().copied().filter(|r| r.file_type != "<DIR>").collect();
    let dirs = selected.iter().filter(|r| r.file_type == "<DIR>").map(|r| r.path.clone()).collect();
    (files, dirs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, kind: &str) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit('/').next().unwrap_or(path).into();
        t.file_type = kind.into();
        t
    }

    #[test]
    fn no_selected_rows_split_into_nothing() {
        let (files, dirs) = split_selection(&[]);
        assert!(files.is_empty() && dirs.is_empty());
    }

    #[test]
    fn files_and_directories_are_split() {
        let rows = vec![row("/a/1.jpg", "jpg"), row("/a/sub", "<DIR>"), row("/a/2.jpg", "jpg")];
        let refs: Vec<&Thumbnail> = rows.iter().collect();
        let (files, dirs) = split_selection(&refs);
        let paths: Vec<_> = files.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(paths, vec!["/a/1.jpg", "/a/2.jpg"]);
        assert_eq!(dirs, vec!["/a/sub".to_string()]);
    }
}
