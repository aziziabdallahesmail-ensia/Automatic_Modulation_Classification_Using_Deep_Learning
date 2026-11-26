"""Run Task 1 & Task 2 EDA and signal visualization pipeline for RML2016.10a."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deep_learning_for_signal_modulation.eda import run_eda


def main() -> None:
    summary = run_eda(
        splits_dir=ROOT / "data" / "splits",
        output_dir=ROOT / "artifacts" / "eda",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
