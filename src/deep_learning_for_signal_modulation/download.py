"""Explicit KaggleHub acquisition; local-file preparation never calls this."""

from pathlib import Path

DATASET_HANDLE = "nolasthitnotomorrow/radioml2016-deepsigcom"
DATASET_FILENAME = "RML2016.10a_dict.pkl"


def download_radioml_dataset() -> Path:
    """Download the latest dataset and resolve exactly one source pickle."""
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("nolasthitnotomorrow/radioml2016-deepsigcom")

    print("Path to dataset files:", path)

    directory = Path(path).resolve()
    if not directory.is_dir():
        raise ValueError(f"KaggleHub returned a non-directory path: {directory}")
    matches = sorted(p for p in directory.rglob(DATASET_FILENAME) if p.is_file())
    if len(matches) != 1:
        candidates = [
            str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
        ]
        raise ValueError(
            f"Expected exactly one {DATASET_FILENAME} beneath {directory}; found {len(matches)}. "
            f"Discovered files: {sorted(candidates)}. Supply --dataset-path FILE to select a local pickle."
        )
    return matches[0]
