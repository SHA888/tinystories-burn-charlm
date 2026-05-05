use crate::config::ModelConfig;
use burn::{
    module::Module,
    nn::{
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

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
    /// Build a [`CharLm`] module from a validated [`ModelConfig`].
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let token_embed =
            EmbeddingConfig::new(config.vocab_size.0, config.embed_dim.0).init(device);
        let pos_embed = EmbeddingConfig::new(config.seq_len.0, config.embed_dim.0).init(device);

        let transformer = TransformerEncoderConfig::new(
            config.embed_dim.0,
            config.d_ff,
            config.layer_count.0,
            config.head_count.0,
        )
        .with_dropout(config.dropout.0)
        .with_norm_first(true)
        .init(device);

        let norm = LayerNormConfig::new(config.embed_dim.0).init(device);
        let dropout = DropoutConfig::new(config.dropout.0).init();

        Self {
            token_embed,
            pos_embed,
            transformer,
            norm,
            dropout,
        }
    }
    #[must_use]
    /// Forward pass: input token IDs → logits.
    ///
    /// The implementation uses:
    /// 1. Learned token and positional embeddings (see `CharLm::new`).
    /// 2. A causal autoregressive mask generated per batch/sequence length.
    /// 3. A pre‑norm transformer encoder (`with_norm_first(true)`).
    /// 4. Weight‑tied language‑model head (projection via the transposed token‑embedding matrix).
    ///
    /// # Shapes
    /// - `input_ids`: `[batch_size, seq_len]` (Int)
    /// - returns: `[batch_size, seq_len, vocab_size]`
    ///
    /// # Panics
    ///
    /// Panics if `seq_len` does not fit in `i64` (extremely unlikely in practice).
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = input_ids.dims();
        let device = input_ids.device();

        let token_emb = self.token_embed.forward(input_ids);

        let seq_i64 = i64::try_from(seq).expect("seq_len fits in i64");
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_i64, &device)
            .reshape([1, seq])
            .repeat_dim(0, batch);
        let pos_emb = self.pos_embed.forward(pos_ids);

        let x = self.dropout.forward(token_emb + pos_emb);

        let mask = generate_autoregressive_mask(batch, seq, &device);
        let encoder_input = TransformerEncoderInput::new(x).mask_attn(mask);

        let hidden = self.transformer.forward(encoder_input);
        let hidden = self.norm.forward(hidden);

        // Weight‑tied LM head: project via transposed token‑embedding matrix.
        // The same weight matrix is used for both embedding lookup and output projection,
        // ensuring parameter sharing as required by the spec.
        let weight = self.token_embed.weight.val(); // [vocab_size, d_model] – shared with the LM head
        let vocab_size = weight.dims()[0];
        let weight_t = weight.clone().transpose(); // [d_model, vocab_size]

        let [batch, seq, d_model] = hidden.dims();
        let hidden_2d = hidden.reshape([batch * seq, d_model]);
        let logits_2d = hidden_2d.matmul(weight_t);
        logits_2d.reshape([batch, seq, vocab_size])
    }
}

/// Compute cross‑entropy loss for language‑modeling logits.
///
/// This helper is kept separate from the model definition to avoid pulling in loss‑related
/// dependencies into the core module. It operates on the logits produced by `CharLm::forward`.
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
    // Note: this function is unchanged; test module follows after it.
    use burn::nn::loss::CrossEntropyLossConfig;
    let [batch, seq, vocab_size] = logits.dims();
    let logits_2d = logits.reshape([batch * seq, vocab_size]);
    let targets_1d = targets.reshape([batch * seq]);

    let config = CrossEntropyLossConfig::new().with_pad_tokens(ignore_index.map(|i| vec![i]));
    let loss = config.init(&logits_2d.device());
    loss.forward(logits_2d, targets_1d)
}

#[cfg(test)]
mod tests {
    /// Build the expected causal-mask matrix where `true` means "masked / cannot attend".
    fn causal_mask_expected(seq_len: usize) -> Vec<Vec<bool>> {
        (0..seq_len)
            .map(|query_index| {
                (0..seq_len)
                    .map(|key_index| key_index > query_index)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn causal_mask_correctness_matrix() {
        // For autoregressive attention, position i can only attend to keys <= i.
        // So entries above the diagonal must be masked (true), and diagonal/lower are false.
        let expected = vec![
            vec![false, true, true, true],
            vec![false, false, true, true],
            vec![false, false, false, true],
            vec![false, false, false, false],
        ];
        assert_eq!(causal_mask_expected(4), expected);
    }
}
