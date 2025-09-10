use crate::Thumbnail;
use egui_data_table::viewer::{DecodeErrorBehavior, RowCodec};

/* --------------------------------------------- Codec ------------------------------------------ */

pub struct ThumbCodec;

impl RowCodec<Thumbnail> for ThumbCodec {
    type DeserializeError = &'static str;

    fn encode_column(&mut self, row: &Thumbnail, column: usize, dst: &mut String) {
        match column {
            0 => { /* thumbnail placeholder */ }
            1 => dst.push_str(&row.filename),
            2 => dst.push_str(&row.path),
            3 => dst.push_str(row.category.as_deref().unwrap_or("")),
            4 => dst.push_str(&row.tags.join(",")),
            5 => dst.push_str(
                &row.modified
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
            ),
            6 => dst.push_str(&row.size.to_string()),
            7 => dst.push_str(&row.file_type),
            _ => {}
        }
    }

    fn decode_column(
        &mut self,
        src: &str,
        column: usize,
        row: &mut Thumbnail,
    ) -> Result<(), DecodeErrorBehavior> {
        match column {
            0 => { /* thumbnail placeholder */ }
            1 => row.filename = src.to_string(),
            2 => row.path = src.to_string(),
            3 => {
                row.category = if src.is_empty() {
                    None
                } else {
                    Some(src.to_string())
                }
            }
            4 => {
                row.tags = src
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            5 => { /* skip modified parse */ }
            6 => row.size = src.parse().unwrap_or_default(),
            7 => row.file_type = src.to_string(),
            _ => {}
        }
        Ok(())
    }

    fn create_empty_decoded_row(&mut self) -> Thumbnail {
        Thumbnail::default()
    }
}
