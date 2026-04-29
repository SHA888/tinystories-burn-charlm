use model_core::Tokenizer;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] != "prepare" {
        eprintln!("Usage: cargo run -p data -- prepare");
        std::process::exit(1);
    }

    let cache_dir = PathBuf::from("data/cache");
    let tokenizer_path = cache_dir.join("tokenizer.json");
    let train_cache = cache_dir.join("train.bin");
    let val_cache = cache_dir.join("val.bin");

    std::fs::create_dir_all(&cache_dir).expect("create cache dir");

    println!("Downloading TinyStories dataset...");
    let (train_path, val_path) =
        data::download::pull_tinystories(&cache_dir).expect("download dataset");
    println!(
        "Train: {}\nVal:   {}",
        train_path.display(),
        val_path.display()
    );

    println!("Building tokenizer from train split...");
    let texts = data::dataset::read_texts_from_parquet(&train_path).expect("read train parquet");
    let tokenizer = data::tokenizer::CharTokenizer::from_corpus(texts.iter().map(String::as_str));
    tokenizer.save(&tokenizer_path).expect("save tokenizer");
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size().0);

    let seq_len = 128usize;
    println!("Tokenizing train split (seq_len={seq_len})...");
    let train_ds = data::dataset::TinyStoriesDataset::from_texts(texts, &tokenizer, seq_len);
    println!("Train chunks: {}", train_ds.len());

    let val_texts = data::dataset::read_texts_from_parquet(&val_path).expect("read val parquet");
    let val_ds = data::dataset::TinyStoriesDataset::from_texts(val_texts, &tokenizer, seq_len);
    println!("Val chunks:   {}", val_ds.len());

    // Persist tokenized chunks as raw u32 bytes for fast loading.
    save_u32_bin(&train_ds, &train_cache).expect("save train cache");
    save_u32_bin(&val_ds, &val_cache).expect("save val cache");
    println!("Done. Cache written to {}", cache_dir.display());
}

fn save_u32_bin(
    ds: &data::dataset::TinyStoriesDataset,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for i in 0..ds.len() {
        let (inp, tgt) = ds.get_pair(i).expect("index within 0..ds.len(); get_pair must return Some");
        for &v in &inp {
            file.write_all(&v.to_le_bytes())?;
        }
        for &v in &tgt {
            file.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}
