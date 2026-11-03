"""Download RadioML and save its 70/15/15 train/validation/test split."""

import argparse
from pathlib import Path

from .data import load_radioml
from .download import download_radioml_dataset
from .splits import SPLIT_NAMES, create_split_indices, save_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Trusted local pickle; skip the KaggleHub download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="New output directory (default: data/splits)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(
                f"Output already exists: {args.output_dir}; choose a new --output-dir"
            )
        if not 0 <= args.seed <= 2**32 - 2:
            raise ValueError("seed must be in [0, 2**32 - 2]")
        source = (
            args.dataset_path
            if args.dataset_path is not None
            else download_radioml_dataset()
        )
        dataset = load_radioml(source)
        splits = create_split_indices(
            dataset.modulation_labels, dataset.snr_db, seed=args.seed
        )
        save_splits(args.output_dir, dataset, splits)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")
    print("Saved splits:", {name: len(getattr(splits, name)) for name in SPLIT_NAMES})
    print("Output directory:", args.output_dir.resolve())
    return 0
