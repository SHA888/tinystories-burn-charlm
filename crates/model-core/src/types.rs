use serde::{Deserialize, Serialize};
use std::fmt;

/// Number of tokens in the vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct VocabSize(pub usize);

impl fmt::Display for VocabSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maximum sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct SeqLen(pub usize);

/// Token and positional embedding dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct EmbedDim(pub usize);

/// Number of transformer layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct LayerCount(pub usize);

/// Number of attention heads per layer.
/// Must divide `EmbedDim` evenly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct HeadCount(pub usize);

/// Dropout probability in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DropoutProb(pub f64);
