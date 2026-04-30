use burn::config::Config;
use std::fmt;

/// Errors that can occur when constructing or validating a `ModelConfig`.
#[derive(Debug, PartialEq, Eq)]
pub enum ConfigError {
    InvariantViolation { msg: String },
    ParseError {
        field: String,
        value: String,
        reason: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvariantViolation { msg } => write!(f, "config invariant violated: {msg}"),
            ConfigError::ParseError { field, value, reason } => {
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
    pub vocab_size: usize,
    pub seq_len: usize,
    pub embed_dim: usize,
    pub layer_count: usize,
    pub head_count: usize,
    pub dropout: f64,
}

impl ModelConfig {
    /// Validate invariants and return a typed error if any fail.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::InvariantViolation` if `embed_dim % head_count != 0`
    /// or if any hyperparameter is zero.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "vocab_size must be > 0".into(),
            });
        }
        if self.seq_len == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "seq_len must be > 0".into(),
            });
        }
        if self.embed_dim == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "embed_dim must be > 0".into(),
            });
        }
        if self.layer_count == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "layer_count must be > 0".into(),
            });
        }
        if self.head_count == 0 {
            return Err(ConfigError::InvariantViolation {
                msg: "head_count must be > 0".into(),
            });
        }
        if self.embed_dim % self.head_count != 0 {
            return Err(ConfigError::InvariantViolation {
                msg: format!(
                    "embed_dim ({}) must be divisible by head_count ({})",
                    self.embed_dim, self.head_count
                ),
            });
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(ConfigError::InvariantViolation {
                msg: format!("dropout ({}) must be in [0.0, 1.0]", self.dropout),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelConfig {
        ModelConfig::new(128, 256, 64, 2, 4, 0.1)
    }

    #[test]
    fn valid_passes() {
        let c = valid_config();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn embed_not_divisible_by_head_count() {
        let mut c = valid_config();
        c.embed_dim = 63;
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
        c.vocab_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn dropout_out_of_range_fails() {
        let mut c = valid_config();
        c.dropout = 1.5;
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
        assert!((c.dropout - decoded.dropout).abs() < f64::EPSILON);
    }
}
