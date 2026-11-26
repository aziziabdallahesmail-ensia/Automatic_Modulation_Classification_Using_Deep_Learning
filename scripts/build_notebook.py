"""Generate a reproducible Jupyter Notebook covering Tasks 1–3 (EDA, Preprocessing, VT-CNN2 Baseline)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(out_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Automatic Modulation Classification (AMC) — Tasks 1–3: EDA, Preprocessing & VT-CNN2 Baseline

**Course:** Wireless Communication Networks and Systems (ENSIA, 4th Year, Semester 1, 2026–2027)  
**Dataset:** DeepSig RadioML `RML2016.10a` (`RML2016.10a_dict.pkl`)  
**Scope:**
1. **Task 1 — Dataset Verification & Exploratory Data Analysis (EDA)**
2. **Task 2 — Signal Visualization & Physical Interpretation (Time, I/Q Constellation, FFT)**
3. **Task 3 — Leakage-Free Preprocessing Pipeline & Reference VT-CNN2 Baseline**"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Image, display

ROOT = Path("..").resolve() if Path.cwd().name == "notebooks" else Path(".").resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deep_learning_for_signal_modulation.eda import run_eda
from deep_learning_for_signal_modulation.model import VTCNN2, VTCNN2Config
from deep_learning_for_signal_modulation.preprocessing import (
    IQNormalizer,
    LabelMap,
    audit_splits,
    load_all_splits,
    set_seed,
)

set_seed(42)
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Task 1 — Dataset Setup, Verification, and Structure Analysis

To reproduce the raw acquisition and deterministic stratified 70/15/15 split from scratch:
```bash
python -m deep_learning_for_signal_modulation --output-dir data/splits --seed 42
```

### Signal Processing Background
- **What do $I$ and $Q$ represent?** A passband RF signal $s(t) = A(t)\\cos(2\\pi f_c t + \\phi(t))$ is down-converted to complex baseband $z(t) = I(t) + jQ(t) = A(t)e^{j\\phi(t)}$, where $I[n]$ is the in-phase projection ($\\cos$) and $Q[n]$ is the quadrature projection ($-\\sin$). Together they preserve instantaneous **amplitude** $\\sqrt{I^2 + Q^2}$ and **phase** $\\operatorname{atan2}(Q, I)$.
- **Why complex baseband samples for AMC?** Digital modulations encode bits in phase (PSK), amplitude (PAM), both amplitude and phase (QAM), or instantaneous frequency $d\\phi/dt$ (FSK). Using real-only samples or magnitude spectra discards relative phase and negative-frequency asymmetry (e.g., USB vs LSB in AM-SSB).
- **What does SNR represent?** Signal-to-Noise Ratio in decibels, $\\text{SNR}_{\\text{dB}} = 10\\log_{10}(P_{\\text{signal}} / P_{\\text{noise}})$, quantifies channel corruption (AWGN plus simulated fading/clock/carrier-offset impairments in GNU Radio).
- **Why span $-20\\text{ dB}$ to $+18\\text{ dB}$?** Real receivers encounter severe noise floors ($-20\\text{ dB}$, signal buried below noise) through clean links ($+18\\text{ dB}$), allowing us to pinpoint the SNR breakdown threshold of each modulation and model."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """eda_summary_path = ROOT / "artifacts" / "eda" / "eda_summary.json"
if not eda_summary_path.exists():
    eda_summary = run_eda(splits_dir=ROOT / "data" / "splits", output_dir=ROOT / "artifacts" / "eda")
else:
    eda_summary = json.loads(eda_summary_path.read_text(encoding="utf-8"))

raw_meta = eda_summary["raw_pickle_metadata"]
print("File:", raw_meta["file_name"], "| Size (MiB):", raw_meta["size_mib"], "| Bytes:", raw_meta["size_bytes"])
print("Classes (11):", raw_meta["modulation_classes"])
print("SNR levels (20):", raw_meta["snr_levels_db"])
print("Total observations:", raw_meta["total_observations"], "| Shape:", raw_meta["observation_shapes"], "| Dtype:", raw_meta["source_dtypes"])
print("Non-finite values:", raw_meta["nonfinite_values"], "| Exact duplicate groups:", raw_meta["exact_duplicate_groups"])

pd.DataFrame(eda_summary["split_audit"]["splits"]).T"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(ROOT / "artifacts" / "eda" / "figures" / "01_dataset_and_split_balance.png")))
display(Image(filename=str(ROOT / "artifacts" / "eda" / "figures" / "02_signal_statistics_vs_snr.png")))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Task 2 — Time-Domain, Constellation, and FFT Visualizations

We visualize 9 representative modulations covering **PSK** (`BPSK`, `QPSK`, `8PSK`), **QAM** (`QAM16`, `QAM64`), **PAM** (`PAM4`), **FSK** (`GFSK`), **Analog AM** (`AM-DSB`), and **Analog FM** (`WBFM`) at $\\text{SNR} \\in \\{-10, 0, +10, +18\\}\\text{ dB}$, plus a complete 11-class gallery at $+18\\text{ dB}$.
- **FFT settings:** Hann window with coherent power normalization, DC-centered frame, zero-padded to $N_{\\text{fft}} = 512$ (from $N = 128$ samples), `fftshift`, peak-normalized magnitude in dB."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """for fig_name in [
    "03_time_domain_iq.png",
    "04_constellation_diagrams.png",
    "05_frequency_domain_fft.png",
    "06_all_11_classes_at_18db.png",
]:
    display(Image(filename=str(ROOT / "artifacts" / "eda" / "figures" / fig_name)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Task 3 — Leakage-Free Preprocessing & Reference VT-CNN2 Baseline

### Critical EDA Finding Driving Preprocessing
In `RML2016.10a`, raw frames have very small amplitude scale ($\\sigma_I \\approx 0.0058$, $\\sigma_Q \\approx 0.0062$, $\\|x\\|_2 \\approx 0.0966$). Because Glorot/He weight initializers assume $\\mathcal{O}(1)$ input variance, raw unscaled inputs attenuate early-layer gradients and slow convergence. We therefore compare `raw`, `l2_frame`, `rms_frame`, and `channel_zscore` (fitted strictly on `train`) on the **validation split only**, leaving `test.npz` untouched until final evaluation."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """baseline_results_path = ROOT / "artifacts" / "baseline" / "baseline_results.json"
results = json.loads(baseline_results_path.read_text(encoding="utf-8"))

ablation_df = pd.DataFrame(
    [
        {
            "normalization": r["normalization"],
            "epochs_run": r["epochs_run"],
            "best_epoch": r["best_epoch"],
            "best_val_loss": round(r["best_val_loss"], 4),
            "best_val_acc_%": round(r["best_val_acc"] * 100, 2),
            "best_val_macro_f1": round(r["best_val_macro_f1"], 4),
            "train_time_s": r["total_train_time_s"],
        }
        for r in results["normalization_ablation_val"]
    ]
)
display(ablation_df)

te = results["test_evaluation"]
print(
    f"Selected Normalization: {results['selected_model']['normalization']}\\n"
    f"Untouched Test Accuracy: {te['accuracy']*100:.2f}% | "
    f"Macro-F1: {te['macro_f1']:.4f} | "
    f"Macro-Precision: {te['macro_precision']:.4f} | "
    f"Macro-Recall: {te['macro_recall']:.4f}"
)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """for fig_name in [
    "01_training_curves.png",
    "02_normalization_comparison_val.png",
    "03_accuracy_and_f1_vs_snr.png",
    "04_per_class_accuracy_vs_snr.png",
    "05_confusion_matrices_selected_snrs.png",
    "06_confusion_matrix_overall.png",
]:
    display(Image(filename=str(ROOT / "artifacts" / "baseline" / "figures" / fig_name)))"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    print(f"Wrote notebook: {out_path}")


if __name__ == "__main__":
    build_notebook(ROOT / "notebooks" / "01_eda_preprocessing_and_vtcnn2_baseline.ipynb")
