use serde::{Deserialize, Serialize};
use std::fmt;

/// Newtype for vocabulary size, derived from observed corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct VocabSize(pub usize);

impl fmt::Display for VocabSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Port (trait) for tokenizers.
///
/// Lives in `model-core` so both `data` (training prep) and `infer`
/// (text generation) can depend on it without cross-crate cycles.
pub trait Tokenizer: Send + Sync {
    /// Encode a string into token ids.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode token ids back into a string.
    fn decode(&self, ids: &[u32]) -> String;

    /// Vocabulary size observed by this tokenizer.
    fn vocab_size(&self) -> VocabSize;
}
