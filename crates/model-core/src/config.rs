use crate::types::{DropoutProb, EmbedDim, HeadCount, LayerCount, SeqLen, VocabSize};
use burn::config::Config;
use std::fmt;

/// Errors that can occur when constructing or validating a `ModelConfig`.
#[derive(Debug, PartialEq, Eq)]
pub enum ConfigError {
    InvariantViolation {
        msg: String,
    },
    ParseError {
        field: String,
        value: String,
        reason: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvariantViolation { msg } => {
                write!(f, "config invariant violated: {msg}")
            }
            ConfigError::ParseError {
                field,
                value,
                reason,
            } => {
                write!(f, "failed to parse field `{field}` = `{value}`: {reason}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Hyper-parameter configuration for the character-level language model.
///
/// Burn's `Config` derive generates a `new(vocab_size, seq_len, ...)` constructor.
/// Call `validate()` to enforce invariants (e.g. `EmbedDim % HeadCount == 0`).
#[derive(Config, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: VocabSize,
    pub seq_len: SeqLen,
    pub embed_dim: EmbedDim,
    pub layer_count: LayerCount,
    pub head_count: HeadCount,
    pub d_ff: usize,
    pub dropout: DropoutProb,
}

impl ModelConfig {
    /// Validate invariants and return a typed error if any fail.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::InvariantViolation` if `embed_dim % head_count != 0`
    /// or if any hyperparameter is zero.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size.0 == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "vocab_size must be > 0".into(),
            });
        }
        if self.seq_len.0 == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "seq_len must be > 0".into(),
            });
        }
        if self.embed_dim.0 == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "embed_dim must be > 0".into(),
            });
        }
        if self.layer_count.0 == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "layer_count must be > 0".into(),
            });
        }
        if self.head_count.0 == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "head_count must be > 0".into(),
            });
        }
        if self.embed_dim.0 % self.head_count.0 != 0 {
            return Err(ConfigError::InvariantViolation {
                msg: format!(
                    "embed_dim ({}) must be divisible by head_count ({})",
                    self.embed_dim.0, self.head_count.0
                ),
            });
        }
        if !(0.0..=1.0).contains(&self.dropout.0) {
            return Err(ConfigError::InvariantViolation {
                msg: format!("dropout ({}) must be in [0.0, 1.0]", self.dropout.0),
            });
        }
        if self.d_ff == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "d_ff must be > 0".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelConfig {
        ModelConfig::new(
            VocabSize(128),
            SeqLen(256),
            EmbedDim(64),
            LayerCount(2),
            HeadCount(4),
            128,
            DropoutProb(0.1),
        )
    }

    #[test]
    fn valid_passes() {
        let c = valid_config();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn embed_not_divisible_by_head_count() {
        let mut c = valid_config();
        c.embed_dim = EmbedDim(63);
        assert_eq!(
            c.validate(),
            Err(ConfigError::InvariantViolation {
                msg: "embed_dim (63) must be divisible by head_count (4)".into(),
            })
        );
    }

    #[test]
    fn zero_vocab_fails() {
        let mut c = valid_config();
        c.vocab_size = VocabSize(0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn dropout_out_of_range_fails() {
        let mut c = valid_config();
        c.dropout = DropoutProb(1.5);
        assert!(c.validate().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let c = valid_config();
        c.validate().unwrap();
        let json = serde_json::to_string(&c).unwrap();
        let decoded: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c.vocab_size, decoded.vocab_size);
        assert_eq!(c.seq_len, decoded.seq_len);
        assert_eq!(c.embed_dim, decoded.embed_dim);
        assert_eq!(c.layer_count, decoded.layer_count);
        assert_eq!(c.head_count, decoded.head_count);
        assert_eq!(c.d_ff, decoded.d_ff);
        assert!((c.dropout.0 - decoded.dropout.0).abs() < f64::EPSILON);
    }
}
