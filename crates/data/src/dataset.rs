use arrow_array::Array;
use model_core::Tokenizer;
use std::path::Path;

#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    Parquet(parquet::errors::ParquetError),
    MissingTextColumn,
    WrongColumnType,
}
impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::Io(e) => write!(f, "io error: {e}"),
            DatasetError::Parquet(e) => write!(f, "parquet error: {e}"),
            DatasetError::MissingTextColumn => write!(f, "missing 'text' column"),
            DatasetError::WrongColumnType => write!(f, "'text' is not string"),
        }
    }
}
impl std::error::Error for DatasetError {}
impl From<std::io::Error> for DatasetError {
    fn from(e: std::io::Error) -> Self {
        DatasetError::Io(e)
    }
}
impl From<parquet::errors::ParquetError> for DatasetError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        DatasetError::Parquet(e)
    }
}

impl From<arrow_schema::ArrowError> for DatasetError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        DatasetError::Io(std::io::Error::other(e))
    }
}

/// Read all text stories from a Parquet file incrementally (row-group at a time).
///
/// # Errors
///
/// Returns `DatasetError` on IO failure, Parquet/Arrow parse errors, or a missing
/// / wrong-type `"text"` column.
pub fn read_texts_from_parquet(path: &Path) -> Result<Vec<String>, DatasetError> {
    let file = std::fs::File::open(path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "text")
        .ok_or_else(|| {
            let available: Vec<_> = schema.fields().iter().map(|f| f.name().to_string()).collect();
            eprintln!("Error: Parquet file missing 'text' column. Available columns: {:?}", available);
            DatasetError::MissingTextColumn
        })?;
    let mut texts = Vec::new();
    let reader = builder.build()?;
    for batch in reader {
        let batch = batch?;
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or(DatasetError::WrongColumnType)?;
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                texts.push(arr.value(i).to_string());
            }
        }
    }
    Ok(texts)
}

pub struct TinyStoriesDataset {
    chunks: Vec<Vec<u32>>,
}

impl TinyStoriesDataset {
    pub fn from_texts(texts: Vec<String>, tokenizer: &impl Tokenizer, seq_len: usize) -> Self {
        assert!(seq_len > 0, "seq_len must be > 0");
        let mut all_ids: Vec<u32> = Vec::new();
        for text in texts {
            all_ids.extend(tokenizer.encode(&text));
            all_ids.push(0); // EOS
        }
        let chunk_size = seq_len + 1;
        let total_tokens = all_ids.len();
        let chunks: Vec<Vec<u32>> = all_ids
            .chunks(chunk_size)
            .filter(|c| c.len() == chunk_size)
            .map(std::vec::Vec::from)
            .collect();
        let kept_tokens = chunks.len() * chunk_size;
        let dropped_tokens = total_tokens - kept_tokens;
        if dropped_tokens > 0 {
            eprintln!("Warning: dropped {dropped_tokens} tokens ({:.1}% of data) due to incomplete final chunk",
                100.0 * dropped_tokens as f64 / total_tokens as f64);
        }
        Self { chunks }
    }

    #[must_use]
    pub fn get_pair(&self, index: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        self.chunks.get(index).map(|chunk| {
            let input = chunk[..chunk.len() - 1].to_vec();
            let target = chunk[1..].to_vec();
            (input, target)
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::CharTokenizer;

    #[test]
    fn batch_shapes_ok() {
        let tok = CharTokenizer::from_corpus(["abc", "bcd", "cde", "def"].iter().copied());
        let texts: Vec<String> = ["abc", "bcd", "cde", "def"]
            .iter()
            .copied()
            .map(String::from)
            .collect();
        let ds = TinyStoriesDataset::from_texts(texts, &tok, 4);
        assert_eq!(ds.len(), 3);
        let (inp, tgt) = ds.get_pair(0).unwrap();
        assert_eq!(inp.len(), 4);
        assert_eq!(tgt.len(), 4);
    }
}
