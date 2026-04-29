use std::path::PathBuf;

/// Hugging Face repository and pinned revision for TinyStories.
const REPO: &str = "roneneldan/TinyStories";
const REVISION: &str = "refs/convert/parquet"; // Pin to the parquet conversion branch

/// Files we expect in the dataset repository.
const FILES: &[&str] = &["train.parquet", "validation.parquet"];

/// Download TinyStories dataset files into `cache_dir`.
///
/// Returns local paths to the downloaded files (train, validation).
/// Uses `hf-hub` to resolve the HF cache and download if absent.
pub fn pull_tinystories(cache_dir: &std::path::Path) -> Result<(PathBuf, PathBuf), HfError> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(HfError::Api)?;

    let repo = hf_hub::Repo::with_revision(
        REPO.to_string(),
        hf_hub::RepoType::Dataset,
        REVISION.to_string(),
    );

    let dataset = api.repo(repo);

    let train = dataset.get(FILES[0]).map_err(HfError::Download)?;
    let valid = dataset.get(FILES[1]).map_err(HfError::Download)?;

    Ok((train, valid))
}

/// Errors that can occur during dataset pull.
#[derive(Debug)]
pub enum HfError {
    Api(hf_hub::api::sync::ApiError),
    Download(hf_hub::api::sync::ApiError),
}

impl std::fmt::Display for HfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfError::Api(e) => write!(f, "HF API build error: {e}"),
            HfError::Download(e) => write!(f, "HF download error: {e}"),
        }
    }
}

impl std::error::Error for HfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfError::Api(e) | HfError::Download(e) => Some(e),
        }
    }
}
