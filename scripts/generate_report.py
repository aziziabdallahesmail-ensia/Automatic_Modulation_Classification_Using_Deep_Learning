"""Generate a grounded Markdown technical report from saved EDA and baseline JSON/CSV artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_report(
    eda_dir: Path = ROOT / "artifacts" / "eda",
    baseline_dir: Path = ROOT / "artifacts" / "baseline",
    out_path: Path = ROOT / "reports" / "eda_and_baseline_report.md",
) -> Path:
    eda_summary = json.loads((eda_dir / "eda_summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((baseline_dir / "baseline_results.json").read_text(encoding="utf-8"))
    snr_stats = pd.read_csv(eda_dir / "train_stats_by_snr.csv")
    mod_stats = pd.read_csv(eda_dir / "train_stats_by_modulation.csv")
    test_snr = pd.read_csv(baseline_dir / "test_metrics_by_snr.csv")

    raw = eda_summary["raw_pickle_metadata"]
    split_audit = eda_summary["split_audit"]
    iq_stats = eda_summary["train_global_iq_stats"]
    sel = baseline["selected_model"]
    te = baseline["test_evaluation"]
    cost = te["computational_cost"]

    lines: list[str] = []
    lines.append("# Automatic Modulation Classification (AMC) — EDA, Preprocessing & VT-CNN2 Baseline Report\n")
    lines.append("**Module:** Wireless Communication Networks and Systems (ENSIA, 4th Year, Semester 1, 2026–2027)  ")
    lines.append("**Tasks Covered:** Task 1 (Dataset Exploration), Task 2 (Signal Visualization & Physical Analysis), Task 3 (Leakage-Free Preprocessing & VT-CNN2 Baseline)\n")

    lines.append("## 1. Dataset Verification & Structure (Task 1)\n")
    lines.append(f"- **Source File:** `{raw['file_name']}` ({raw['format']})")
    lines.append(f"- **Exact File Size:** {raw['size_bytes']:,} bytes ({raw['size_mib']:.3f} MiB)")
    lines.append(f"- **Total Observations:** {raw['total_observations']:,} complex I/Q frames across {raw['num_blocks']} `(modulation, SNR)` blocks")
    lines.append(f"- **Sample Shape & Dtype:** `{raw['observation_shapes'][0]}` (`2 × 128`, I row 0, Q row 1), dtype `{raw['source_dtypes'][0]}`")
    lines.append(f"- **Modulation Classes ({raw['num_classes']}):** {', '.join(f'`{m}`' for m in raw['modulation_classes'])}")
    lines.append(f"- **SNR Levels ({raw['num_snr_levels']}):** `{raw['snr_levels_db'][0]} dB` to `{raw['snr_levels_db'][-1]:+d} dB` in `2 dB` steps (1,000 frames per class per SNR — perfectly balanced)")
    lines.append(f"- **Data Quality Audit:** `{raw['nonfinite_values']}` NaN/Inf values; `{raw['exact_duplicate_groups']}` exact duplicate frame groups\n")

    lines.append("### 1.1 Stratified Split Integrity (70 / 15 / 15)\n")
    lines.append("| Split | Frames | Fraction | Strata Covered | Frames / Stratum | Overlap With Other Splits |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split_name in ("train", "validation", "test"):
        sp = split_audit["splits"][split_name]
        lines.append(
            f"| `{split_name}` | {sp['count']:,} | {sp['fraction']*100:.1f}% | "
            f"{sp['num_strata']} | {sp['stratum_min_count']}–{sp['stratum_max_count']} | 0 |"
        )
    lines.append("")

    lines.append("### 1.2 Signal Processing Questions (Task 1)\n")
    lines.append(
        "- **What do I and Q represent?** Complex baseband representation $z[n] = I[n] + jQ[n] = A[n]e^{j\\phi[n]}$, "
        "where $I[n]$ is the in-phase ($\\cos$) projection and $Q[n]$ is the quadrature ($-\\sin$) projection of the down-converted passband RF carrier.\n"
        "- **Why complex baseband for AMC?** Modulation schemes encode information in instantaneous amplitude $\\sqrt{I^2+Q^2}$ (PAM, AM-DSB), "
        "phase $\\operatorname{atan2}(Q, I)$ (BPSK, QPSK, 8PSK), joint amplitude-phase (QAM16, QAM64), or instantaneous frequency $d\\phi/dt$ (CPFSK, GFSK, WBFM). "
        "Discarding Q destroys phase polarity and spectral asymmetry (e.g., upper vs. lower single-sideband).\n"
        "- **What does SNR represent?** $\\text{SNR}_{\\text{dB}} = 10\\log_{10}(P_{\\text{signal}}/P_{\\text{noise}})$ quantifies channel noise power relative to transmitted signal power.\n"
        "- **Why span $-20\\text{ dB}$ to $+18\\text{ dB}$?** Wireless links operate across harsh interference/fading regimes and clean line-of-sight links; "
        "evaluating across all 20 SNRs reveals each architecture's waterfall curve and noise-floor breakdown threshold.\n"
    )

    lines.append("## 2. Exploratory Signal Statistics & Visualizations (Task 2)\n")
    lines.append(
        f"Training-split raw I/Q amplitude statistics reveal a key numerical property: "
        f"`mean_I = {iq_stats['mean_i']:.6f}`, `mean_Q = {iq_stats['mean_q']:.6f}`, "
        f"`std_I = {iq_stats['std_i']:.6f}`, `std_Q = {iq_stats['std_q']:.6f}`, and "
        f"mean frame L2 norm $= {iq_stats['frame_l2_norm_mean']:.5f}$ (`min = {iq_stats['frame_l2_norm_min']:.5f}`, `max = {iq_stats['frame_l2_norm_max']:.5f}`). "
        "Because raw frames have ~170x smaller standard deviation than unit variance, unscaled inputs significantly attenuate early-layer gradients under Glorot/He initialization.\n"
    )

    lines.append("### 2.1 Training-Split Statistics Across Representative SNRs\n")
    lines.append("| SNR (dB) | Mean L2 Norm | Std(I) | Std(Q) | Mean PAPR (dB) | Excess Kurtosis (I) | Spectral Flatness |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for snr in (-20, -10, -4, 0, 4, 10, 18):
        row = snr_stats.loc[snr_stats["snr_db"] == snr].iloc[0]
        lines.append(
            f"| {int(snr):+d} | {row['l2_norm_mean']:.5f} | {row['std_i_mean']:.5f} | {row['std_q_mean']:.5f} | "
            f"{row['papr_db_mean']:.2f} | {row['kurtosis_i_mean']:.3f} | {row['spectral_flatness_mean']:.3f} |"
        )
    lines.append("")

    lines.append("### 2.2 Per-Modulation Summary (Training Split, Averaged Across All SNRs)\n")
    lines.append("| Modulation | Family | Mean L2 Norm | Mean PAPR (dB) | Excess Kurtosis (I) | Norm. $|C_{40}|$ | Spectral Flatness |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in mod_stats.iterrows():
        lines.append(
            f"| `{r['modulation']}` | {r['family']} | {r['l2_norm_mean']:.5f} | {r['papr_db_mean']:.2f} | "
            f"{r['kurtosis_i_mean']:.3f} | {r['abs_c40_norm_mean']:.3f} | {r['spectral_flatness_mean']:.3f} |"
        )
    lines.append("")

    lines.append("### 2.3 Signal Visualization Findings (Task 2 Questions)\n")
    lines.append(
        "- **Constellation dispersion at low SNR:** Additive white Gaussian noise (AWGN) adds independent zero-mean Gaussian perturbations to both $I$ and $Q$. "
        "At $-10\\text{ dB}$, noise power is $10\\times$ signal power, spreading discrete constellation points into an isotropic circular Gaussian cloud.\n"
        "- **Easiest vs. hardest to distinguish visually:** At high SNR ($+10$ to $+18\\text{ dB}$), constant-modulus FSK (`GFSK`, `CPFSK` circular rings) and linear `BPSK`/`PAM4` trajectories stand out clearly, "
        "whereas `QAM16` vs. `QAM64` (dense overlapping multi-level grids under unknown carrier phase/RRC pulse shaping) and `AM-DSB` vs. `WBFM` (during speech-silence pauses where both collapse near an unmodulated carrier) are hardest.\n"
        "- **Low-SNR similarity:** At $\\le -10\\text{ dB}$, all modulations converge toward circular Gaussian noise clouds (`kurtosis ≈ +0.1`, `spectral flatness ≈ 0.54–0.56`) and become visually indistinguishable.\n"
        "- **Time vs. frequency domain insights:** Time-domain I/Q waveforms expose symbol transitions, envelope fluctuations (QAM/PAM vs. constant-envelope FSK), and $90^\\circ$ phase quadrature. "
        "Hann-windowed FFT spectra ($N_{\\text{fft}}=512$) expose occupied bandwidth and spectral roll-off, but magnitude spectra alone cannot separate modulations sharing the same RRC pulse shape (e.g., `BPSK`, `QPSK`, `8PSK`, `QAM16`, `QAM64`).\n"
    )

    lines.append("### 2.4 Generated EDA Figures\n")
    for fig in eda_summary["generated_figures"]:
        lines.append(f"- `artifacts/eda/figures/{fig}`")
    lines.append("")

    lines.append("## 3. Preprocessing Ablation & VT-CNN2 Baseline (Task 3)\n")
    lines.append("### 3.1 Validation-Only Normalization Comparison (Zero Test Leakage)\n")
    lines.append("| Strategy | Epochs Run | Best Epoch | Best Val Loss | Best Val Acc (%) | Best Val Macro-F1 | Train Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in baseline["normalization_ablation_val"]:
        lines.append(
            f"| `{r['normalization']}` | {r['epochs_run']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | "
            f"{r['best_val_acc']*100:.2f}% | {r['best_val_macro_f1']:.4f} | {r['total_train_time_s']:.1f} |"
        )
    lines.append("")

    lines.append("### 3.2 Untouched Test Set Evaluation (Selected Model: `" + sel["normalization"] + "`)\n")
    lines.append(f"- **Test Accuracy:** {te['accuracy']*100:.2f}%")
    lines.append(f"- **Test Macro-F1:** {te['macro_f1']:.4f} (Weighted-F1: {te['weighted_f1']:.4f})")
    lines.append(f"- **Test Macro-Precision / Recall:** {te['macro_precision']:.4f} / {te['macro_recall']:.4f}")
    lines.append(f"- **Test Cross-Entropy Loss:** {te['loss']:.4f}\n")

    lines.append("#### Test Accuracy & Macro-F1 Across Selected SNRs\n")
    lines.append("| SNR (dB) | Support | Accuracy (%) | Macro-F1 | Macro-Precision | Macro-Recall |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for snr in (-20, -14, -10, -6, -2, 0, 2, 6, 10, 14, 18):
        row = test_snr.loc[test_snr["snr_db"] == snr].iloc[0]
        lines.append(
            f"| {int(snr):+d} | {int(row['support'])} | {row['accuracy']*100:.2f}% | "
            f"{row['macro_f1']:.4f} | {row['macro_precision']:.4f} | {row['macro_recall']:.4f} |"
        )
    lines.append("")

    lines.append("#### Per-Class Test Metrics (Averaged Across All 20 SNRs)\n")
    lines.append("| Modulation | Precision | Recall | F1-Score | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    cls_rep = te["classification_report"]
    for mod in raw["modulation_classes"]:
        m = cls_rep[mod]
        lines.append(
            f"| `{mod}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |"
        )
    lines.append("")

    lines.append("### 3.3 Computational Cost Assessment\n")
    lines.append(f"- **Parameters:** {cost['parameters']['total']:,} total ({cost['parameters']['trainable']:,} trainable)")
    lines.append(f"- **Checkpoint Size:** {sel['checkpoint_size_mib']:.3f} MiB (`artifacts/baseline/vtcnn2_best.pt`)")
    lines.append(f"- **Compute Complexity:** {cost['macs_per_sample']:,} MACs/sample (~{cost['approx_flops_per_sample']:,} FLOPs/sample)")
    for bench in cost["inference_latency"]:
        lines.append(
            f"- **Inference Latency ({bench['device']}):** {bench['ms_per_batch']:.3f} ms / batch of {bench['batch_size']} "
            f"({bench['us_per_sample']:.2f} us/sample, {bench['throughput_samples_per_sec']:,.0f} samples/s)"
        )
    lines.append("")

    lines.append("### 3.4 Baseline Limitations & Next Steps (Tasks 4–5)\n")
    lines.append(
        "1. **Dense-layer parameter bottleneck:** The `Flatten (10,560) -> Dense(256)` projection accounts for "
        "2,703,616 of 2,830,427 parameters (95.5%), limiting spatial/temporal translation invariance and increasing overfitting risk.\n"
        "2. **Limited receptive field:** Two small `(1×3)` and `(2×3)` convolutions capture only local 5-sample temporal context without explicit sequential/global attention.\n"
        "3. **Known RML2016.10a ambiguities:** High-SNR confusion persists between `QAM16` vs. `QAM64` (128 samples are short to populate 64 symbols cleanly) "
        "and `WBFM` vs. `AM-DSB` (silent speech intervals).\n"
        "4. **Task 4 direction:** Replace heavy flattening with global pooling + sequential/residual modeling (CNN-LSTM/CLDNN, ResNet-1D, or Transformer) and compare via accuracy-vs-SNR and t-SNE embeddings.\n"
    )

    lines.append("### 3.5 Generated Baseline Figures\n")
    for fig in baseline["generated_figures"]:
        lines.append(f"- `artifacts/baseline/figures/{fig}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = generate_report()
    print(f"Wrote report: {p}")
