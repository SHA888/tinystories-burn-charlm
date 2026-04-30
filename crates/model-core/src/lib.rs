pub mod config;
pub mod model;
pub mod types;

pub use config::{ConfigError, ModelConfig};
pub use model::{CharLm, CharLmConfig};
pub use types::{DropoutProb, EmbedDim, HeadCount, LayerCount, SeqLen, VocabSize};

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
