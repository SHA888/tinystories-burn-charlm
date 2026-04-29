use model_core::{Tokenizer, VocabSize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Character-level tokenizer with byte-level fallback for non-ASCII.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, u32>,
    id_to_char: HashMap<u32, char>,
}

impl CharTokenizer {
    /// Build a tokenizer from an iterable of text samples.
    ///
    /// Scans every character in the corpus, assigns each unique char an id,
    /// and reserves id 0 for the unknown (non-ASCII / out-of-vocab) token.
    pub fn from_corpus<'a>(texts: impl Iterator<Item = &'a str>) -> Self {
        let mut char_to_id: HashMap<char, u32> = HashMap::new();
        let mut id_to_char: HashMap<u32, char> = HashMap::new();

        // Reserve 0 for the unknown-token fallback.
        let mut next_id: u32 = 1;
        for text in texts {
            for ch in text.chars() {
                if char_to_id.contains_key(&ch) {
                    continue;
                }
                char_to_id.insert(ch, next_id);
                id_to_char.insert(next_id, ch);
                next_id += 1;
            }
        }

        CharTokenizer {
            char_to_id,
            id_to_char,
        }
    }

    /// Persist tokenizer to JSON at the given path.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Load tokenizer from JSON at the given path.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let tokenizer = serde_json::from_reader(file)?;
        Ok(tokenizer)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| *self.char_to_id.get(&ch).unwrap_or(&0))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.id_to_char.get(&id).copied().unwrap_or('\u{FFFD}'))
            .collect()
    }

    fn vocab_size(&self) -> VocabSize {
        VocabSize(self.char_to_id.len() + 1) // +1 for the reserved unknown id 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_core::Tokenizer;

    #[test]
    fn roundtrip_ascii() {
        let corpus = ["abc", "bcd", "cde"];
        let tok = CharTokenizer::from_corpus(corpus.iter().copied());
        let encoded = tok.encode("abc");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn roundtrip_non_ascii_fallback() {
        let corpus = ["hello"];
        let tok = CharTokenizer::from_corpus(corpus.iter().copied());
        // 'é' is not in vocab -> maps to id 0 -> decodes to replacement char
        let encoded = tok.encode("héllo");
        assert!(encoded.contains(&0));
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "h\u{FFFD}llo");
    }

    #[test]
    fn save_load_roundtrip() {
        let corpus = ["rust", "burn"];
        let tok = CharTokenizer::from_corpus(corpus.iter().copied());
        let tmp = std::env::temp_dir().join("test_tokenizer.json");
        tok.save(&tmp).unwrap();
        let loaded = CharTokenizer::load(&tmp).unwrap();
        assert_eq!(tok.vocab_size(), loaded.vocab_size());
        assert_eq!(tok.encode("rust"), loaded.encode("rust"));
        std::fs::remove_file(&tmp).unwrap();
    }
}
