use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

/// Configuration for the character-level language model.
#[derive(Config, Debug, PartialEq)]
pub struct CharLmConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl CharLmConfig {
    /// Build a [`CharLm`] module from this config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CharLm<B> {
        let token_embed = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_embed = EmbeddingConfig::new(self.seq_len, self.d_model).init(device);

        let transformer = TransformerEncoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_layers,
            self.n_heads,
        )
        .with_dropout(self.dropout)
        .with_norm_first(true)
        .init(device);

        let norm = LayerNormConfig::new(self.d_model).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        CharLm {
            token_embed,
            pos_embed,
            transformer,
            norm,
            dropout,
        }
    }
}

/// Causal character-level language model.
///
/// Architecture (GPT-style decoder-only transformer):
/// 1. Token + positional embeddings (learned)
/// 2. `TransformerEncoder` with causal attention mask and pre-norm
/// 3. Final layer norm
/// 4. LM head via weight-tied token embedding projection
#[derive(Module, Debug)]
pub struct CharLm<B: Backend> {
    pub token_embed: Embedding<B>,
    pub pos_embed: Embedding<B>,
    pub transformer: TransformerEncoder<B>,
    pub norm: LayerNorm<B>,
    pub dropout: Dropout,
}

impl<B: Backend> CharLm<B> {
    /// Forward pass: input token IDs -> logits.
    ///
    /// # Shapes
    /// - `input_ids`: `[batch_size, seq_len]` (Int)
    /// - returns: `[batch_size, seq_len, vocab_size]`
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = input_ids.dims();
        let device = input_ids.device();

        let token_emb = self.token_embed.forward(input_ids);

        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq as i64, &device)
            .reshape([1, seq])
            .repeat_dim(0, batch);
        let pos_emb = self.pos_embed.forward(pos_ids);

        let x = self.dropout.forward(token_emb + pos_emb);

        let mask = generate_autoregressive_mask(batch, seq, &device);
        let encoder_input = TransformerEncoderInput::new(x).mask_attn(mask);

        let hidden = self.transformer.forward(encoder_input);
        let hidden = self.norm.forward(hidden);

        // Weight-tied LM head: project via transposed token-embedding matrix.
        let weight = self.token_embed.weight.val(); // [vocab_size, d_model]
        let vocab_size = weight.dims()[0];
        let weight_t = weight.clone().transpose(); // [d_model, vocab_size]

        let [batch, seq, d_model] = hidden.dims();
        let hidden_2d = hidden.reshape([batch * seq, d_model]);
        let logits_2d = hidden_2d.matmul(weight_t);
        let logits = logits_2d.reshape([batch, seq, vocab_size]);

        logits
    }
}

/// Compute cross-entropy loss for language-modeling logits.
///
/// # Shapes
/// - `logits`: `[batch_size, seq_len, vocab_size]`
/// - `targets`: `[batch_size, seq_len]` (Int)
/// - returns: scalar loss tensor `[1]`
///
/// # Errors
///
/// Panics on shape mismatch or if targets contain out-of-range class indices.
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    ignore_index: Option<usize>,
) -> Tensor<B, 1> {
    use burn::nn::loss::CrossEntropyLossConfig;
    let [batch, seq, vocab_size] = logits.dims();
    let logits_2d = logits.reshape([batch * seq, vocab_size]);
    let targets_1d = targets.reshape([batch * seq]);

    let config = CrossEntropyLossConfig::new()
        .with_pad_tokens(ignore_index.map(|i| vec![i]));
    let loss = config.init(&logits_2d.device());
    loss.forward(logits_2d, targets_1d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> CharLmConfig {
        CharLmConfig::new(8, 4, 16, 1, 2, 32, 0.0)
    }

    #[test]
    fn char_lm_config_new() {
        let c = small_config();
        assert_eq!(c.vocab_size, 8);
        assert_eq!(c.seq_len, 4);
        assert_eq!(c.d_model, 16);
        assert_eq!(c.n_layers, 1);
        assert_eq!(c.n_heads, 2);
        assert_eq!(c.d_ff, 32);
        assert_eq!(c.dropout, 0.0);
    }
}
