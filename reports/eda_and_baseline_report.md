# Automatic Modulation Classification (AMC) — EDA, Preprocessing & VT-CNN2 Baseline Report

**Module:** Wireless Communication Networks and Systems (ENSIA, 4th Year, Semester 1, 2026–2027)  
**Tasks Covered:** Task 1 (Dataset Exploration), Task 2 (Signal Visualization & Physical Analysis), Task 3 (Leakage-Free Preprocessing & VT-CNN2 Baseline)

## 1. Dataset Verification & Structure (Task 1)

- **Source File:** `RML2016.10a_dict.pkl` (Python Pickle (dict[(modulation, snr_db) -> np.ndarray]))
- **Exact File Size:** 640,919,653 bytes (611.229 MiB)
- **Total Observations:** 220,000 complex I/Q frames across 220 `(modulation, SNR)` blocks
- **Sample Shape & Dtype:** `[2, 128]` (`2 × 128`, I row 0, Q row 1), dtype `float32`
- **Modulation Classes (11):** `8PSK`, `AM-DSB`, `AM-SSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, `WBFM`
- **SNR Levels (20):** `-20 dB` to `+18 dB` in `2 dB` steps (1,000 frames per class per SNR — perfectly balanced)
- **Data Quality Audit:** `0` NaN/Inf values; `0` exact duplicate frame groups

### 1.1 Stratified Split Integrity (70 / 15 / 15)

| Split | Frames | Fraction | Strata Covered | Frames / Stratum | Overlap With Other Splits |
|---|---:|---:|---:|---:|---:|
| `train` | 154,000 | 70.0% | 220 | 700–700 | 0 |
| `validation` | 33,000 | 15.0% | 220 | 150–150 | 0 |
| `test` | 33,000 | 15.0% | 220 | 150–150 | 0 |

### 1.2 Signal Processing Questions (Task 1)

- **What do I and Q represent?** Complex baseband representation $z[n] = I[n] + jQ[n] = A[n]e^{j\phi[n]}$, where $I[n]$ is the in-phase ($\cos$) projection and $Q[n]$ is the quadrature ($-\sin$) projection of the down-converted passband RF carrier.
- **Why complex baseband for AMC?** Modulation schemes encode information in instantaneous amplitude $\sqrt{I^2+Q^2}$ (PAM, AM-DSB), phase $\operatorname{atan2}(Q, I)$ (BPSK, QPSK, 8PSK), joint amplitude-phase (QAM16, QAM64), or instantaneous frequency $d\phi/dt$ (CPFSK, GFSK, WBFM). Discarding Q destroys phase polarity and spectral asymmetry (e.g., upper vs. lower single-sideband).
- **What does SNR represent?** $\text{SNR}_{\text{dB}} = 10\log_{10}(P_{\text{signal}}/P_{\text{noise}})$ quantifies channel noise power relative to transmitted signal power.
- **Why span $-20\text{ dB}$ to $+18\text{ dB}$?** Wireless links operate across harsh interference/fading regimes and clean line-of-sight links; evaluating across all 20 SNRs reveals each architecture's waterfall curve and noise-floor breakdown threshold.

## 2. Exploratory Signal Statistics & Visualizations (Task 2)

Training-split raw I/Q amplitude statistics reveal a key numerical property: `mean_I = 0.000073`, `mean_Q = -0.000618`, `std_I = 0.005811`, `std_Q = 0.006246`, and mean frame L2 norm $= 0.09664$ (`min = 0.08839`, `max = 0.28769`). Because raw frames have ~170x smaller standard deviation than unit variance, unscaled inputs significantly attenuate early-layer gradients under Glorot/He initialization.

### 2.1 Training-Split Statistics Across Representative SNRs

| SNR (dB) | Mean L2 Norm | Std(I) | Std(Q) | Mean PAPR (dB) | Excess Kurtosis (I) | Spectral Flatness |
|---:|---:|---:|---:|---:|---:|---:|
| -20 | 0.10003 | 0.00619 | 0.00625 | 7.53 | 0.127 | 0.564 |
| -10 | 0.09981 | 0.00612 | 0.00621 | 7.40 | 0.084 | 0.538 |
| -4 | 0.09869 | 0.00565 | 0.00595 | 6.79 | -0.092 | 0.417 |
| +0 | 0.09586 | 0.00501 | 0.00535 | 5.58 | -0.373 | 0.274 |
| +4 | 0.09379 | 0.00458 | 0.00494 | 4.23 | -0.561 | 0.181 |
| +10 | 0.09335 | 0.00439 | 0.00480 | 3.26 | -0.649 | 0.139 |
| +18 | 0.09343 | 0.00431 | 0.00479 | 3.02 | -0.678 | 0.123 |

### 2.2 Per-Modulation Summary (Training Split, Averaged Across All SNRs)

| Modulation | Family | Mean L2 Norm | Mean PAPR (dB) | Excess Kurtosis (I) | Norm. $|C_{40}|$ | Spectral Flatness |
|---|---|---:|---:|---:|---:|---:|
| `AM-DSB` | Analog AM | 0.09436 | 4.51 | 0.105 | 0.292 | 0.550 |
| `AM-SSB` | Analog AM | 0.10002 | 7.52 | 0.132 | 0.407 | 0.565 |
| `WBFM` | Analog FM | 0.09438 | 4.51 | -0.023 | 0.433 | 0.469 |
| `CPFSK` | FSK | 0.09461 | 4.76 | -0.490 | 0.307 | 0.294 |
| `GFSK` | FSK | 0.09451 | 4.64 | -0.236 | 0.621 | 0.303 |
| `PAM4` | PAM | 0.10105 | 6.31 | -0.362 | 0.671 | 0.243 |
| `8PSK` | PSK | 0.09616 | 5.51 | -0.418 | 0.360 | 0.294 |
| `BPSK` | PSK | 0.09743 | 5.77 | -0.489 | 0.814 | 0.293 |
| `QPSK` | PSK | 0.09620 | 5.50 | -0.451 | 0.477 | 0.291 |
| `QAM16` | QAM | 0.09713 | 5.68 | -0.408 | 0.453 | 0.222 |
| `QAM64` | QAM | 0.09719 | 5.62 | -0.417 | 0.473 | 0.178 |

### 2.3 Signal Visualization Findings (Task 2 Questions)

- **Constellation dispersion at low SNR:** Additive white Gaussian noise (AWGN) adds independent zero-mean Gaussian perturbations to both $I$ and $Q$. At $-10\text{ dB}$, noise power is $10\times$ signal power, spreading discrete constellation points into an isotropic circular Gaussian cloud.
- **Easiest vs. hardest to distinguish visually:** At high SNR ($+10$ to $+18\text{ dB}$), constant-modulus FSK (`GFSK`, `CPFSK` circular rings) and linear `BPSK`/`PAM4` trajectories stand out clearly, whereas `QAM16` vs. `QAM64` (dense overlapping multi-level grids under unknown carrier phase/RRC pulse shaping) and `AM-DSB` vs. `WBFM` (during speech-silence pauses where both collapse near an unmodulated carrier) are hardest.
- **Low-SNR similarity:** At $\le -10\text{ dB}$, all modulations converge toward circular Gaussian noise clouds (`kurtosis ≈ +0.1`, `spectral flatness ≈ 0.54–0.56`) and become visually indistinguishable.
- **Time vs. frequency domain insights:** Time-domain I/Q waveforms expose symbol transitions, envelope fluctuations (QAM/PAM vs. constant-envelope FSK), and $90^\circ$ phase quadrature. Hann-windowed FFT spectra ($N_{\text{fft}}=512$) expose occupied bandwidth and spectral roll-off, but magnitude spectra alone cannot separate modulations sharing the same RRC pulse shape (e.g., `BPSK`, `QPSK`, `8PSK`, `QAM16`, `QAM64`).

### 2.4 Generated EDA Figures

- `artifacts/eda/figures/01_dataset_and_split_balance.png`
- `artifacts/eda/figures/02_signal_statistics_vs_snr.png`
- `artifacts/eda/figures/03_time_domain_iq.png`
- `artifacts/eda/figures/04_constellation_diagrams.png`
- `artifacts/eda/figures/05_frequency_domain_fft.png`
- `artifacts/eda/figures/06_all_11_classes_at_18db.png`

## 3. Preprocessing Ablation & VT-CNN2 Baseline (Task 3)

### 3.1 Validation-Only Normalization Comparison (Zero Test Leakage)

| Strategy | Epochs Run | Best Epoch | Best Val Loss | Best Val Acc (%) | Best Val Macro-F1 | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| `raw` | 72 | 62 | 1.2514 | 51.05% | 0.5153 | 847.3 |
| `l2_frame` | 69 | 59 | 1.1901 | 55.00% | 0.5551 | 1078.5 |
| `rms_frame` | 52 | 42 | 1.1857 | 55.15% | 0.5654 | 631.2 |
| `channel_zscore` | 44 | 34 | 1.1939 | 54.73% | 0.5445 | 654.6 |

### 3.2 Untouched Test Set Evaluation (Selected Model: `rms_frame`)

- **Test Accuracy:** 54.99%
- **Test Macro-F1:** 0.5641 (Weighted-F1: 0.5641)
- **Test Macro-Precision / Recall:** 0.6656 / 0.5499
- **Test Cross-Entropy Loss:** 1.1844

#### Test Accuracy & Macro-F1 Across Selected SNRs

| SNR (dB) | Support | Accuracy (%) | Macro-F1 | Macro-Precision | Macro-Recall |
|---:|---:|---:|---:|---:|---:|
| -20 | 1650 | 9.64% | 0.0278 | 0.1178 | 0.0964 |
| -14 | 1650 | 10.97% | 0.0524 | 0.2035 | 0.1097 |
| -10 | 1650 | 19.82% | 0.1687 | 0.2721 | 0.1982 |
| -6 | 1650 | 49.64% | 0.4770 | 0.5011 | 0.4964 |
| -2 | 1650 | 70.18% | 0.6850 | 0.6965 | 0.7018 |
| +0 | 1650 | 77.45% | 0.7615 | 0.7682 | 0.7745 |
| +2 | 1650 | 80.12% | 0.7818 | 0.8081 | 0.8012 |
| +6 | 1650 | 81.76% | 0.7923 | 0.8227 | 0.8176 |
| +10 | 1650 | 83.03% | 0.8075 | 0.8327 | 0.8303 |
| +14 | 1650 | 82.00% | 0.7950 | 0.8324 | 0.8200 |
| +18 | 1650 | 82.18% | 0.7976 | 0.8332 | 0.8218 |

#### Per-Class Test Metrics (Averaged Across All 20 SNRs)

| Modulation | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `8PSK` | 0.6202 | 0.5193 | 0.5653 | 3000 |
| `AM-DSB` | 0.5739 | 0.5750 | 0.5744 | 3000 |
| `AM-SSB` | 0.2569 | 0.9640 | 0.4056 | 3000 |
| `BPSK` | 0.8930 | 0.5813 | 0.7042 | 3000 |
| `CPFSK` | 0.8992 | 0.6037 | 0.7224 | 3000 |
| `GFSK` | 0.8572 | 0.6163 | 0.7171 | 3000 |
| `PAM4` | 0.9355 | 0.6667 | 0.7785 | 3000 |
| `QAM16` | 0.3785 | 0.1257 | 0.1887 | 3000 |
| `QAM64` | 0.5080 | 0.6317 | 0.5632 | 3000 |
| `QPSK` | 0.7791 | 0.4727 | 0.5884 | 3000 |
| `WBFM` | 0.6205 | 0.2927 | 0.3977 | 3000 |

### 3.3 Computational Cost Assessment

- **Parameters:** 2,830,427 total (2,830,427 trainable)
- **Checkpoint Size:** 10.801 MiB (`artifacts/baseline/vtcnn2_best.pt`)
- **Compute Complexity:** 19,126,016 MACs/sample (~38,252,032 FLOPs/sample)
- **Inference Latency (cpu):** 169.828 ms / batch of 1024 (165.85 us/sample, 6,030 samples/s)
- **Inference Latency (cuda):** 23.653 ms / batch of 1024 (23.10 us/sample, 43,293 samples/s)

### 3.4 Baseline Limitations & Next Steps (Tasks 4–5)

1. **Dense-layer parameter bottleneck:** The `Flatten (10,560) -> Dense(256)` projection accounts for 2,703,616 of 2,830,427 parameters (95.5%), limiting spatial/temporal translation invariance and increasing overfitting risk.
2. **Limited receptive field:** Two small `(1×3)` and `(2×3)` convolutions capture only local 5-sample temporal context without explicit sequential/global attention.
3. **Known RML2016.10a ambiguities:** High-SNR confusion persists between `QAM16` vs. `QAM64` (128 samples are short to populate 64 symbols cleanly) and `WBFM` vs. `AM-DSB` (silent speech intervals).
4. **Task 4 direction:** Replace heavy flattening with global pooling + sequential/residual modeling (CNN-LSTM/CLDNN, ResNet-1D, or Transformer) and compare via accuracy-vs-SNR and t-SNE embeddings.

### 3.5 Generated Baseline Figures

- `artifacts/baseline/figures/01_training_curves.png`
- `artifacts/baseline/figures/02_normalization_comparison_val.png`
- `artifacts/baseline/figures/03_accuracy_and_f1_vs_snr.png`
- `artifacts/baseline/figures/04_per_class_accuracy_vs_snr.png`
- `artifacts/baseline/figures/05_confusion_matrices_selected_snrs.png`
- `artifacts/baseline/figures/06_confusion_matrix_overall.png`