"""Exploratory data analysis and signal visualization for RadioML RML2016.10a (Tasks 1 & 2)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.stats import kurtosis

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import SplitData, audit_splits, load_all_splits

REPRESENTATIVE_MODS: tuple[str, ...] = (
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "WBFM",
    "AM-DSB",
    "GFSK",
    "PAM4",
)

SELECTED_SNRS: tuple[int, ...] = (-10, 0, 10, 18)

MODULATION_FAMILY: dict[str, str] = {
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "QAM16": "QAM",
    "QAM64": "QAM",
    "PAM4": "PAM",
    "CPFSK": "FSK",
    "GFSK": "FSK",
    "AM-DSB": "Analog AM",
    "AM-SSB": "Analog AM",
    "WBFM": "Analog FM",
}


def _configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def inspect_raw_pickle(pickle_path: str | Path) -> dict:
    """Inspect the raw RML2016.10a pickle for Task 1 dataset verification."""
    path = Path(pickle_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw pickle not found: {path}")

    size_bytes = path.stat().st_size
    with path.open("rb") as fh:
        raw = pickle.load(fh, encoding="latin1")

    blocks_meta = []
    dtypes = set()
    shapes = set()
    mods = set()
    snrs = set()
    total_frames = 0
    nonfinite_count = 0

    for (mod, snr), arr in sorted(raw.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        mod_str = mod.decode("ascii") if isinstance(mod, bytes) else str(mod)
        snr_int = int(snr)
        mods.add(mod_str)
        snrs.add(snr_int)
        dtypes.add(str(arr.dtype))
        shapes.add(tuple(int(d) for d in arr.shape[1:]))
        n_frames = int(arr.shape[0])
        total_frames += n_frames
        nonfinite_count += int(np.size(arr) - np.isfinite(arr).sum())
        blocks_meta.append(
            {
                "modulation": mod_str,
                "family": MODULATION_FAMILY.get(mod_str, "Other"),
                "snr_db": snr_int,
                "num_frames": n_frames,
                "dtype": str(arr.dtype),
                "sample_shape": list(arr.shape[1:]),
            }
        )

    # Check exact duplicate frames across the entire raw dataset
    stacked = np.concatenate(
        [
            raw[k].astype(np.float32)
            for k in sorted(raw.keys(), key=lambda kv: (str(kv[0]), int(kv[1])))
        ],
        axis=0,
    )
    stacked_2d = np.ascontiguousarray(stacked.reshape(len(stacked), -1))
    flat_view = stacked_2d.view(
        np.dtype((np.void, stacked_2d.dtype.itemsize * stacked_2d.shape[1]))
    ).ravel()
    _, unique_counts = np.unique(flat_view, return_counts=True)
    duplicate_frames = int((unique_counts > 1).sum())

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "format": "Python Pickle (dict[(modulation, snr_db) -> np.ndarray])",
        "size_bytes": int(size_bytes),
        "size_mib": round(size_bytes / (1024**2), 3),
        "num_blocks": len(raw),
        "total_observations": int(total_frames),
        "num_classes": len(mods),
        "modulation_classes": sorted(mods),
        "num_snr_levels": len(snrs),
        "snr_levels_db": sorted(snrs),
        "observation_shapes": [list(s) for s in sorted(shapes)],
        "source_dtypes": sorted(dtypes),
        "nonfinite_values": nonfinite_count,
        "exact_duplicate_groups": duplicate_frames,
        "blocks": blocks_meta,
    }


def compute_signal_statistics(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute physical & statistical descriptors per (mod, SNR), per SNR, and per modulation."""
    I = X[:, 0, :].astype(np.float64)
    Q = X[:, 1, :].astype(np.float64)
    z = I + 1j * Q

    frame_l2 = np.sqrt(np.sum(I**2 + Q**2, axis=1))
    frame_power = np.mean(I**2 + Q**2, axis=1)
    amp = np.abs(z)
    peak_power = np.max(amp**2, axis=1)
    papr_db = 10.0 * np.log10(np.maximum(peak_power, 1e-12) / np.maximum(frame_power, 1e-12))

    # Normalized 4th-order cumulant |C40| / sigma^4 for complex baseband signal
    z_centered = z - z.mean(axis=1, keepdims=True)
    var_z = np.mean(np.abs(z_centered) ** 2, axis=1)
    m40 = np.mean(z_centered**4, axis=1)
    m20 = np.mean(z_centered**2, axis=1)
    c40_norm = np.abs(m40 - 3.0 * (m20**2)) / np.maximum(var_z**2, 1e-12)

    # Excess kurtosis of I channel
    kurt_i = kurtosis(I, axis=1, fisher=True, bias=False)

    # Spectral flatness (Wiener entropy) with Hann window
    window = np.hanning(X.shape[2])[None, :]
    spec = np.abs(np.fft.fft(z_centered * window, n=128, axis=1)) ** 2
    spec = np.maximum(spec, 1e-18)
    geo_mean = np.exp(np.mean(np.log(spec), axis=1))
    arith_mean = np.mean(spec, axis=1)
    spectral_flatness = geo_mean / np.maximum(arith_mean, 1e-18)

    frame_df = pd.DataFrame(
        {
            "modulation": labels.astype(str),
            "family": [MODULATION_FAMILY.get(str(m), "Other") for m in labels],
            "snr_db": snrs.astype(int),
            "mean_i": I.mean(axis=1),
            "mean_q": Q.mean(axis=1),
            "std_i": I.std(axis=1),
            "std_q": Q.std(axis=1),
            "frame_l2_norm": frame_l2,
            "frame_mean_power": frame_power,
            "papr_db": papr_db,
            "kurtosis_i": kurt_i,
            "abs_c40_norm": c40_norm,
            "spectral_flatness": spectral_flatness,
        }
    )

    by_mod_snr = (
        frame_df.groupby(["modulation", "family", "snr_db"], as_index=False)
        .agg(
            count=("frame_l2_norm", "size"),
            mean_i=("mean_i", "mean"),
            mean_q=("mean_q", "mean"),
            std_i=("std_i", "mean"),
            std_q=("std_q", "mean"),
            l2_norm_mean=("frame_l2_norm", "mean"),
            l2_norm_std=("frame_l2_norm", "std"),
            mean_power=("frame_mean_power", "mean"),
            papr_db_mean=("papr_db", "mean"),
            kurtosis_i_mean=("kurtosis_i", "mean"),
            abs_c40_norm_mean=("abs_c40_norm", "mean"),
            spectral_flatness_mean=("spectral_flatness", "mean"),
        )
        .sort_values(["modulation", "snr_db"])
    )

    by_snr = (
        frame_df.groupby("snr_db", as_index=False)
        .agg(
            count=("frame_l2_norm", "size"),
            l2_norm_mean=("frame_l2_norm", "mean"),
            l2_norm_std=("frame_l2_norm", "std"),
            std_i_mean=("std_i", "mean"),
            std_q_mean=("std_q", "mean"),
            papr_db_mean=("papr_db", "mean"),
            kurtosis_i_mean=("kurtosis_i", "mean"),
            abs_c40_norm_mean=("abs_c40_norm", "mean"),
            spectral_flatness_mean=("spectral_flatness", "mean"),
        )
        .sort_values("snr_db")
    )

    by_mod = (
        frame_df.groupby(["modulation", "family"], as_index=False)
        .agg(
            count=("frame_l2_norm", "size"),
            l2_norm_mean=("frame_l2_norm", "mean"),
            l2_norm_std=("frame_l2_norm", "std"),
            std_i_mean=("std_i", "mean"),
            std_q_mean=("std_q", "mean"),
            papr_db_mean=("papr_db", "mean"),
            kurtosis_i_mean=("kurtosis_i", "mean"),
            abs_c40_norm_mean=("abs_c40_norm", "mean"),
            spectral_flatness_mean=("spectral_flatness", "mean"),
        )
        .sort_values(["family", "modulation"])
    )

    return by_mod_snr, by_snr, by_mod


def _pick_representative_frame(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
    modulation: str,
    snr_db: int,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Select a deterministic representative frame (closest to median L2 energy) for plotting."""
    mask = (labels == modulation) & (snrs == snr_db)
    candidates = X[mask]
    if len(candidates) == 0:
        raise ValueError(f"No samples for ({modulation}, {snr_db} dB)")
    l2 = np.sqrt(np.sum(candidates**2, axis=(1, 2)))
    median_idx = int(np.argsort(l2)[len(l2) // 2])
    return candidates[median_idx]


def _compute_fft_db(frame: NDArray[np.float32], n_fft: int = 512) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Hann-windowed, DC-centered, peak-normalized magnitude spectrum in dB."""
    z = frame[0].astype(np.float64) + 1j * frame[1].astype(np.float64)
    z = z - z.mean()  # remove DC offset for spectral clarity
    window = np.hanning(len(z))
    window /= np.sqrt(np.mean(window**2))  # coherent power normalization
    spectrum = np.fft.fftshift(np.fft.fft(z * window, n=n_fft))
    mag = np.abs(spectrum) / len(z)
    mag_norm = mag / np.maximum(mag.max(), 1e-12)
    mag_db = 20.0 * np.log10(np.maximum(mag_norm, 1e-6))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0))
    return freqs, mag_db


def plot_dataset_and_split_balance(
    splits: dict[str, SplitData],
    by_mod_snr: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figure 1: Dataset class/SNR balance and stratified 70/15/15 split verification."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    pivot = by_mod_snr.pivot(index="modulation", columns="snr_db", values="count")
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        annot_kws={"size": 6.5},
        cmap="YlGnBu",
        cbar_kws={"label": "Frames per (Modulation, SNR)"},
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("RML2016.10a Class × SNR Distribution (Total = 220,000 Frames)")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Modulation Class")

    # Stacked bar of train / val / test counts per modulation
    mods = sorted(np.unique(splits["train"].modulation_labels).tolist())
    counts = {
        name: [int((splits[name].modulation_labels == m).sum()) for m in mods]
        for name in ("train", "validation", "test")
    }
    x = np.arange(len(mods))
    axes[1].bar(x, counts["train"], label="Train (70%)", color="#0f766e")
    axes[1].bar(x, counts["validation"], bottom=counts["train"], label="Validation (15%)", color="#f59e0b")
    bottom_test = np.array(counts["train"]) + np.array(counts["validation"])
    axes[1].bar(x, counts["test"], bottom=bottom_test, label="Test (15%)", color="#1e293b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mods, rotation=35, ha="right")
    axes[1].set_ylabel("Number of Frames")
    axes[1].set_title("Stratified Split Balance Across All 11 Classes")
    axes[1].legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_signal_statistics_vs_snr(
    by_mod_snr: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figure 2: Signal processing descriptors vs SNR across modulation families."""
    _configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)

    metrics = [
        ("l2_norm_mean", "Frame L2 Norm ($||x||_2$)", "Frame L2 Energy Norm vs SNR"),
        ("papr_db_mean", "PAPR (dB)", "Peak-to-Average Power Ratio (PAPR) vs SNR"),
        ("kurtosis_i_mean", "Excess Kurtosis (I channel)", "In-Phase Excess Kurtosis vs SNR"),
        ("spectral_flatness_mean", "Spectral Flatness (Wiener Entropy)", "Spectral Flatness vs SNR (1 = White Noise)"),
    ]

    palette = sns.color_palette("tab10", n_colors=len(by_mod_snr["modulation"].unique()))
    for ax, (col, ylabel, title) in zip(axes.ravel(), metrics, strict=True):
        sns.lineplot(
            data=by_mod_snr,
            x="snr_db",
            y=col,
            hue="modulation",
            palette=palette,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            ax=ax,
            legend=(ax is axes[0, 1]),
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("SNR (dB)")
        if col == "kurtosis_i_mean":
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)

    axes[0, 1].legend(title="Modulation", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
    fig.suptitle("Signal Processing Characteristics Across SNR Levels (RML2016.10a Training Split)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fmt_snr(snr: int) -> str:
    return "0 dB" if snr == 0 else f"{snr:+d} dB"


def plot_time_domain_grid(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
    mods: tuple[str, ...],
    snr_list: tuple[int, ...],
    out_path: Path,
) -> None:
    """Figure 3: Time-domain I, Q, and overlaid I/Q signals across representative modulations & SNRs."""
    _configure_plot_style()
    n_rows = len(mods)
    n_cols = len(snr_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 1.9 * n_rows), sharex=True)
    t = np.arange(X.shape[2])

    for r, mod in enumerate(mods):
        for c, snr in enumerate(snr_list):
            ax = axes[r, c]
            frame = _pick_representative_frame(X, labels, snrs, mod, snr)
            ax.plot(t, frame[0], color="#0284c7", linewidth=1.0, label="I (In-Phase)")
            ax.plot(t, frame[1], color="#ea580c", linewidth=1.0, alpha=0.88, label="Q (Quadrature)")
            if r == 0:
                ax.set_title(f"SNR = {_fmt_snr(snr)}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mod}\n({MODULATION_FAMILY.get(mod, '')})", fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Sample Index n")
            ax.grid(True, alpha=0.3)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(
        "Time-Domain Baseband I & Q Waveforms Across Representative Modulations and SNRs",
        y=1.035,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_constellation_grid(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
    mods: tuple[str, ...],
    snr_list: tuple[int, ...],
    out_path: Path,
) -> None:
    """Figure 4: I/Q constellation scatter plots across representative modulations and SNRs.

    Combines 8 frames per (mod, SNR) cell (~1,024 I/Q samples) for a clearer constellation density view,
    while marking the single representative frame in darker points.
    """
    _configure_plot_style()
    n_rows = len(mods)
    n_cols = len(snr_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.7 * n_rows))

    for r, mod in enumerate(mods):
        for c, snr in enumerate(snr_list):
            ax = axes[r, c]
            mask = (labels == mod) & (snrs == snr)
            frames = X[mask][:8]
            I_pts = frames[:, 0, :].ravel()
            Q_pts = frames[:, 1, :].ravel()
            ax.scatter(I_pts, Q_pts, s=6, alpha=0.35, color="#0f766e", edgecolors="none")
            single = _pick_representative_frame(X, labels, snrs, mod, snr)
            ax.scatter(single[0], single[1], s=10, alpha=0.85, color="#1e293b", edgecolors="none")
            lim = max(np.percentile(np.abs(np.concatenate([I_pts, Q_pts])), 99.5) * 1.15, 0.02)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0.0, color="gray", linewidth=0.5, alpha=0.6)
            ax.axvline(0.0, color="gray", linewidth=0.5, alpha=0.6)
            ax.set_aspect("equal", adjustable="box")
            if r == 0:
                ax.set_title(f"SNR = {_fmt_snr(snr)}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mod}\nQ", fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("I")

    fig.suptitle(
        "I/Q Constellation Diagrams (Q vs I) Across Representative Modulations and SNRs",
        y=1.015,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_domain_grid(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
    mods: tuple[str, ...],
    snr_list: tuple[int, ...],
    out_path: Path,
    n_fft: int = 512,
) -> None:
    """Figure 5: Hann-windowed, normalized FFT magnitude spectra (N_fft=512, N=128) across SNRs."""
    _configure_plot_style()
    n_rows = len(mods)
    n_cols = len(snr_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 1.9 * n_rows), sharex=True, sharey=True)

    for r, mod in enumerate(mods):
        for c, snr in enumerate(snr_list):
            ax = axes[r, c]
            frame = _pick_representative_frame(X, labels, snrs, mod, snr)
            freqs, mag_db = _compute_fft_db(frame, n_fft=n_fft)
            ax.plot(freqs, mag_db, color="#4f46e5", linewidth=1.0)
            ax.fill_between(freqs, mag_db, -60, color="#4f46e5", alpha=0.12)
            ax.set_ylim(-55, 2)
            ax.set_xlim(-0.5, 0.5)
            if r == 0:
                ax.set_title(f"SNR = {_fmt_snr(snr)}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mod}\nMag (dB)", fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Normalized Freq (cycles/sample)")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Frequency-Domain Magnitude Spectra (Hann Window, Zero-Padded N_fft={n_fft}, Peak-Normalized dB)",
        y=1.015,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_classes_overview_18db(
    X: NDArray[np.float32],
    labels: NDArray[np.str_],
    snrs: NDArray[np.int64],
    out_path: Path,
    snr_db: int = 18,
) -> None:
    """Figure 6: Complete 11-class gallery at +18 dB (Time I/Q, Constellation, and FFT)."""
    _configure_plot_style()
    mods = sorted(np.unique(labels).tolist())
    fig, axes = plt.subplots(len(mods), 3, figsize=(11.5, 2.0 * len(mods)))
    t = np.arange(X.shape[2])

    for r, mod in enumerate(mods):
        frame = _pick_representative_frame(X, labels, snrs, mod, snr_db)
        mask = (labels == mod) & (snrs == snr_db)
        frames = X[mask][:8]

        # Col 0: Time domain I + Q
        ax_t = axes[r, 0]
        ax_t.plot(t, frame[0], color="#0284c7", linewidth=0.95, label="I")
        ax_t.plot(t, frame[1], color="#ea580c", linewidth=0.95, alpha=0.85, label="Q")
        ax_t.set_ylabel(f"{mod}\n({MODULATION_FAMILY.get(mod, '')})", fontweight="bold")
        if r == 0:
            ax_t.set_title("Time-Domain I & Q (+18 dB)", fontweight="bold")
            ax_t.legend(loc="upper right", ncol=2, frameon=True)
        if r == len(mods) - 1:
            ax_t.set_xlabel("Sample Index n")

        # Col 1: Constellation
        ax_c = axes[r, 1]
        I_pts = frames[:, 0, :].ravel()
        Q_pts = frames[:, 1, :].ravel()
        ax_c.scatter(I_pts, Q_pts, s=5, alpha=0.35, color="#0f766e", edgecolors="none")
        ax_c.scatter(frame[0], frame[1], s=9, alpha=0.85, color="#1e293b", edgecolors="none")
        lim = max(np.percentile(np.abs(np.concatenate([I_pts, Q_pts])), 99.5) * 1.15, 0.02)
        ax_c.set_xlim(-lim, lim)
        ax_c.set_ylim(-lim, lim)
        ax_c.axhline(0.0, color="gray", linewidth=0.5, alpha=0.6)
        ax_c.axvline(0.0, color="gray", linewidth=0.5, alpha=0.6)
        ax_c.set_aspect("equal", adjustable="box")
        if r == 0:
            ax_c.set_title("I/Q Constellation (+18 dB)", fontweight="bold")
        if r == len(mods) - 1:
            ax_c.set_xlabel("I")

        # Col 2: FFT Magnitude
        ax_f = axes[r, 2]
        freqs, mag_db = _compute_fft_db(frame, n_fft=512)
        ax_f.plot(freqs, mag_db, color="#4f46e5", linewidth=0.95)
        ax_f.fill_between(freqs, mag_db, -60, color="#4f46e5", alpha=0.12)
        ax_f.set_ylim(-55, 2)
        ax_f.set_xlim(-0.5, 0.5)
        if r == 0:
            ax_f.set_title("FFT Spectrum (N_fft=512, +18 dB)", fontweight="bold")
        if r == len(mods) - 1:
            ax_f.set_xlabel("Normalized Freq (cycles/sample)")

    fig.suptitle("Complete 11-Class Signal Gallery at High SNR (+18 dB)", y=1.008, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_eda(
    splits_dir: str | Path = "data/splits",
    output_dir: str | Path = "artifacts/eda",
    raw_pickle_path: str | Path | None = None,
) -> dict:
    """Execute full Task 1 & Task 2 EDA on RML2016.10a and save tables and figures."""
    out_root = Path(output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Locate raw pickle if not explicitly passed
    if raw_pickle_path is None:
        matches = list(Path("data").rglob("RML2016.10a_dict.pkl"))
        if matches:
            raw_pickle_path = matches[0]

    raw_meta = inspect_raw_pickle(raw_pickle_path) if raw_pickle_path else None
    splits = load_all_splits(splits_dir)
    split_audit = audit_splits(splits)

    # Compute full-dataset statistics from combined splits (labels/SNRs only for Task 1 balance;
    # signal stats from training split + full overview saved separately).
    X_all = np.concatenate([splits[n].X for n in ("train", "validation", "test")], axis=0)
    y_all = np.concatenate([splits[n].modulation_labels for n in ("train", "validation", "test")], axis=0)
    snr_all = np.concatenate([splits[n].snr_db for n in ("train", "validation", "test")], axis=0)

    by_mod_snr_all, _, _ = compute_signal_statistics(X_all, y_all, snr_all)
    train_sp = splits["train"]
    by_mod_snr_train, by_snr_train, by_mod_train = compute_signal_statistics(
        train_sp.X, train_sp.modulation_labels, train_sp.snr_db
    )

    by_mod_snr_all.to_csv(out_root / "dataset_by_mod_snr.csv", index=False)
    by_snr_train.to_csv(out_root / "train_stats_by_snr.csv", index=False)
    by_mod_train.to_csv(out_root / "train_stats_by_modulation.csv", index=False)

    # Generate visualizations strictly from the training split (no test inspection leakage)
    plot_dataset_and_split_balance(
        splits, by_mod_snr_all, fig_dir / "01_dataset_and_split_balance.png"
    )
    plot_signal_statistics_vs_snr(
        by_mod_snr_train, fig_dir / "02_signal_statistics_vs_snr.png"
    )
    plot_time_domain_grid(
        train_sp.X,
        train_sp.modulation_labels,
        train_sp.snr_db,
        mods=REPRESENTATIVE_MODS,
        snr_list=SELECTED_SNRS,
        out_path=fig_dir / "03_time_domain_iq.png",
    )
    plot_constellation_grid(
        train_sp.X,
        train_sp.modulation_labels,
        train_sp.snr_db,
        mods=REPRESENTATIVE_MODS,
        snr_list=SELECTED_SNRS,
        out_path=fig_dir / "04_constellation_diagrams.png",
    )
    plot_frequency_domain_grid(
        train_sp.X,
        train_sp.modulation_labels,
        train_sp.snr_db,
        mods=REPRESENTATIVE_MODS,
        snr_list=SELECTED_SNRS,
        out_path=fig_dir / "05_frequency_domain_fft.png",
    )
    plot_all_classes_overview_18db(
        train_sp.X,
        train_sp.modulation_labels,
        train_sp.snr_db,
        out_path=fig_dir / "06_all_11_classes_at_18db.png",
    )

    summary_payload = {
        "raw_pickle_metadata": raw_meta,
        "split_audit": split_audit,
        "train_global_iq_stats": {
            "mean_i": float(train_sp.X[:, 0, :].mean(dtype=np.float64)),
            "mean_q": float(train_sp.X[:, 1, :].mean(dtype=np.float64)),
            "std_i": float(train_sp.X[:, 0, :].std(dtype=np.float64)),
            "std_q": float(train_sp.X[:, 1, :].std(dtype=np.float64)),
            "min_val": float(train_sp.X.min()),
            "max_val": float(train_sp.X.max()),
            "frame_l2_norm_mean": float(np.sqrt(np.sum(train_sp.X.astype(np.float64) ** 2, axis=(1, 2))).mean()),
            "frame_l2_norm_std": float(np.sqrt(np.sum(train_sp.X.astype(np.float64) ** 2, axis=(1, 2))).std()),
            "frame_l2_norm_min": float(np.sqrt(np.sum(train_sp.X.astype(np.float64) ** 2, axis=(1, 2))).min()),
            "frame_l2_norm_max": float(np.sqrt(np.sum(train_sp.X.astype(np.float64) ** 2, axis=(1, 2))).max()),
        },
        "generated_figures": sorted(p.name for p in fig_dir.glob("*.png")),
    }
    (out_root / "eda_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return summary_payload
