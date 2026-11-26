"""Run Task 3 preprocessing ablation, VT-CNN2 training, and test evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deep_learning_for_signal_modulation.baseline import run_baseline_pipeline


def main() -> None:
    results = run_baseline_pipeline(
        splits_dir=ROOT / "data" / "splits",
        output_dir=ROOT / "artifacts" / "baseline",
        seed=42,
        strategies=("raw", "l2_frame", "rms_frame", "channel_zscore"),
        max_epochs=100,
        patience=10,
        batch_size=1024,
        learning_rate=1e-3,
        dropout=0.6,
    )
    print("Selected normalization:", results["selected_model"]["normalization"])
    print("Training summary:", json.dumps(results["selected_model"]["training_summary"], indent=2))
    test_eval = results["test_evaluation"]
    print(
        f"Test Accuracy: {test_eval['accuracy']*100:.2f}% | "
        f"Macro-F1: {test_eval['macro_f1']:.4f} | "
        f"Macro-Precision: {test_eval['macro_precision']:.4f} | "
        f"Macro-Recall: {test_eval['macro_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
