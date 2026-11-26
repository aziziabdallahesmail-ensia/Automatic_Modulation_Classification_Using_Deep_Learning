"""Training, normalization ablation (validation-only), and test evaluation for VT-CNN2 baseline."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import VTCNN2, VTCNN2Config
from .preprocessing import (
    IQNormalizer,
    LabelMap,
    NormalizationStrategy,
    SplitData,
    audit_splits,
    load_all_splits,
    make_dataloader,
    set_seed,
    to_vtcnn2_tensor,
)

CONFUSION_SNRS: tuple[int, ...] = (-10, 0, 10, 18)


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training hyperparameters following Project Specification Section 4.3."""

    seed: int = 42
    batch_size: int = 1024
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    dropout: float = 0.6
    normalization: NormalizationStrategy = "raw"

    def to_dict(self) -> dict:
        return asdict(self)


def _evaluate_tensors(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
) -> tuple[float, float, NDArray[np.int64], NDArray[np.int64]]:
    """Run inference over device-resident tensors, returning (loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    total_count = len(y_tensor)
    preds_all = []

    with torch.no_grad():
        for start in range(0, total_count, batch_size):
            xb = X_tensor[start : start + batch_size]
            yb = y_tensor[start : start + batch_size]
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds_all).astype(np.int64)
    y_true = y_tensor.cpu().numpy().astype(np.int64)
    avg_loss = total_loss / max(total_count, 1)
    acc = float((y_pred == y_true).mean())
    return avg_loss, acc, y_true, y_pred


def train_vtcnn2(
    train_split: SplitData,
    val_split: SplitData,
    label_map: LabelMap,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> tuple[VTCNN2, IQNormalizer, list[dict], dict]:
    """Train VT-CNN2 on train_split with early stopping on val_split (zero test access)."""
    set_seed(config.seed, deterministic=True)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = IQNormalizer(strategy=config.normalization).fit(train_split.X)
    X_train_norm = normalizer.transform(train_split.X)
    X_val_norm = normalizer.transform(val_split.X)

    y_train = label_map.encode(train_split.modulation_labels)
    y_val = label_map.encode(val_split.modulation_labels)

    X_train_t = to_vtcnn2_tensor(X_train_norm).to(dev)
    y_train_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64)).to(dev)
    X_val_t = to_vtcnn2_tensor(X_val_norm).to(dev)
    y_val_t = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.int64)).to(dev)
    shuffle_gen = torch.Generator(device="cpu").manual_seed(config.seed)

    model_cfg = VTCNN2Config(num_classes=len(label_map.classes), dropout=config.dropout)
    model = VTCNN2(model_cfg).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict] = []
    n_train = len(y_train_t)

    t_start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_correct = 0

        perm = torch.randperm(n_train, generator=shuffle_gen).to(dev)
        for start in range(0, n_train, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(yb)
            running_correct += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = running_loss / n_train
        train_acc = running_correct / n_train
        val_loss, val_acc, _, _ = _evaluate_tensors(
            model, X_val_t, y_val_t, criterion, config.batch_size
        )
        ep_time = time.perf_counter() - ep_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "epoch_time_s": round(ep_time, 3),
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    total_train_time_s = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    # Final validation check with restored best weights
    val_loss_best, val_acc_best, val_true, val_pred = _evaluate_tensors(
        model, X_val_t, y_val_t, criterion, config.batch_size
    )
    val_macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))

    val_per_snr = {}
    for snr in sorted(np.unique(val_split.snr_db).tolist()):
        m = val_split.snr_db == snr
        val_per_snr[int(snr)] = float((val_true[m] == val_pred[m]).mean())

    train_meta = {
        "normalization": config.normalization,
        "device": str(dev),
        "epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(val_loss_best),
        "best_val_acc": float(val_acc_best),
        "best_val_macro_f1": val_macro_f1,
        "total_train_time_s": round(total_train_time_s, 3),
        "mean_epoch_time_s": round(total_train_time_s / max(len(history), 1), 3),
        "val_acc_by_snr": val_per_snr,
    }
    print(
        f"[{config.normalization}] epochs={len(history)} best_epoch={best_epoch} "
        f"val_loss={val_loss_best:.4f} val_acc={val_acc_best*100:.2f}% "
        f"time={total_train_time_s:.1f}s",
        flush=True,
    )
    return model, normalizer, history, train_meta


def _benchmark_inference_latency(
    model: VTCNN2,
    sample_batch: NDArray[np.float32],
    device: torch.device,
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> dict:
    """Measure inference latency (ms per batch and us per sample) on target device."""
    model_eval = copy.deepcopy(model).to(device).eval()
    tensor_x = to_vtcnn2_tensor(sample_batch).to(device)
    n_samples = len(sample_batch)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model_eval(tensor_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed_runs):
            _ = model_eval(tensor_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ms_per_batch = (elapsed / timed_runs) * 1e3
    us_per_sample = (elapsed / (timed_runs * n_samples)) * 1e6
    samples_per_sec = (timed_runs * n_samples) / elapsed
    return {
        "device": str(device),
        "batch_size": int(n_samples),
        "ms_per_batch": round(ms_per_batch, 4),
        "us_per_sample": round(us_per_sample, 4),
        "throughput_samples_per_sec": round(samples_per_sec, 1),
    }


def evaluate_on_split(
    model: VTCNN2,
    normalizer: IQNormalizer,
    label_map: LabelMap,
    split: SplitData,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> dict:
    """Evaluate trained model on a held-out split (validation or untouched test)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_norm = normalizer.transform(split.X)
    y_true = label_map.encode(split.modulation_labels)
    X_t = to_vtcnn2_tensor(X_norm).to(dev)
    y_t = torch.from_numpy(np.ascontiguousarray(y_true, dtype=np.int64)).to(dev)

    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true_out, y_pred = _evaluate_tensors(model, X_t, y_t, criterion, batch_size)

    macro_f1 = float(f1_score(y_true_out, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_out, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_out, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_out, y_pred, average="macro", zero_division=0))

    cls_report = classification_report(
        y_true_out,
        y_pred,
        target_names=list(label_map.classes),
        output_dict=True,
        zero_division=0,
    )

    snrs = sorted(np.unique(split.snr_db).tolist())
    per_snr_rows = []
    per_class_snr_rows = []
    for snr in snrs:
        m = split.snr_db == snr
        yt = y_true_out[m]
        yp = y_pred[m]
        per_snr_rows.append(
            {
                "snr_db": int(snr),
                "support": int(m.sum()),
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "macro_precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            }
        )
        for idx, cls_name in label_map.idx_to_class.items():
            cm_mask = yt == idx
            cls_acc = float((yp[cm_mask] == idx).mean()) if cm_mask.any() else 0.0
            per_class_snr_rows.append(
                {
                    "snr_db": int(snr),
                    "modulation": cls_name,
                    "support": int(cm_mask.sum()),
                    "accuracy": cls_acc,
                }
            )

    cm_overall = confusion_matrix(y_true_out, y_pred, labels=list(range(len(label_map.classes))))
    cm_by_snr = {}
    for snr in CONFUSION_SNRS:
        m = split.snr_db == snr
        cm_by_snr[int(snr)] = confusion_matrix(
            y_true_out[m], y_pred[m], labels=list(range(len(label_map.classes)))
        ).tolist()

    # Computational cost metrics
    arch_rows = model.architecture_summary()
    total_macs = int(sum(int(r["macs"]) for r in arch_rows))
    latency_benchmarks = [_benchmark_inference_latency(model, X_norm[:batch_size], torch.device("cpu"))]
    if torch.cuda.is_available():
        latency_benchmarks.append(
            _benchmark_inference_latency(model, X_norm[:batch_size], torch.device("cuda"))
        )

    return {
        "split_name": split.name,
        "num_samples": int(len(y_true_out)),
        "loss": float(loss),
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "classification_report": cls_report,
        "per_snr_metrics": per_snr_rows,
        "per_class_snr_metrics": per_class_snr_rows,
        "confusion_matrix_overall": cm_overall.tolist(),
        "confusion_matrices_selected_snrs": cm_by_snr,
        "computational_cost": {
            "parameters": model.count_parameters(),
            "macs_per_sample": total_macs,
            "approx_flops_per_sample": 2 * total_macs,
            "architecture_layers": arch_rows,
            "inference_latency": latency_benchmarks,
        },
    }


def _plot_training_curves(history_df: pd.DataFrame, best_epoch: int, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#0284c7", linewidth=1.8)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#ea580c", linewidth=1.8)
    axes[0].axvline(best_epoch, color="#0f766e", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[0].set_title("Cross-Entropy Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[0].legend(frameon=True)

    axes[1].plot(history_df["epoch"], history_df["train_acc"] * 100, label="Train Acc", color="#0284c7", linewidth=1.8)
    axes[1].plot(history_df["epoch"], history_df["val_acc"] * 100, label="Val Acc", color="#ea580c", linewidth=1.8)
    axes[1].axvline(best_epoch, color="#0f766e", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[1].set_title("Classification Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=True)

    fig.suptitle("VT-CNN2 Baseline Training & Early Stopping Trajectory", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_norm_comparison(ablation_results: list[dict], out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for res in ablation_results:
        snrs = sorted(int(k) for k in res["val_acc_by_snr"].keys())
        accs = [res["val_acc_by_snr"][str(s)] * 100 if str(s) in res["val_acc_by_snr"] else res["val_acc_by_snr"][s] * 100 for s in snrs]
        ax.plot(
            snrs,
            accs,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"{res['normalization']} (Val Acc={res['best_val_acc']*100:.2f}%, F1={res['best_val_macro_f1']:.3f})",
        )

    ax.set_title("Preprocessing Normalization Comparison on Validation Split (Zero Test Leakage)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_xticks(range(-20, 20, 2))
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_test_accuracy_vs_snr(per_snr_df: pd.DataFrame, overall_acc: float, overall_f1: float, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(
        per_snr_df["snr_db"],
        per_snr_df["accuracy"] * 100,
        marker="o",
        linewidth=2.2,
        color="#0f766e",
        label=f"Overall Accuracy (Mean = {overall_acc*100:.2f}%)",
    )
    ax.plot(
        per_snr_df["snr_db"],
        per_snr_df["macro_f1"] * 100,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="#ea580c",
        label=f"Macro-F1 × 100 (Mean = {overall_f1*100:.2f}%)",
    )
    ax.axhline(100.0 / 11.0, color="gray", linestyle=":", linewidth=1.2, label="Random Chance (9.09%)")

    for snr, acc in zip(per_snr_df["snr_db"], per_snr_df["accuracy"], strict=True):
        if snr in (-20, -10, 0, 10, 18):
            ax.annotate(f"{acc*100:.1f}%", (snr, acc * 100), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_title("VT-CNN2 Baseline Test Performance vs SNR (RML2016.10a)", fontweight="bold")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(per_snr_df["snr_db"].tolist())
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_per_class_snr(per_class_snr_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.15, 1.0]})

    pivot = per_class_snr_df.pivot(index="modulation", columns="snr_db", values="accuracy") * 100.0
    sns.heatmap(
        pivot,
        annot=False,
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        ax=axes[0],
    )
    axes[0].set_title("Per-Modulation Test Accuracy (%) Across SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Modulation")

    palette = sns.color_palette("tab10", n_colors=pivot.shape[0])
    for idx, mod in enumerate(pivot.index):
        axes[1].plot(
            pivot.columns,
            pivot.loc[mod],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=palette[idx],
            label=mod,
        )
    axes[1].set_title("Per-Modulation Accuracy vs SNR Curves")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 102)
    axes[1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True, title="Modulation")

    fig.suptitle("Class-Wise Generalization Across Signal-to-Noise Ratios", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_confusion_matrices(
    cm_by_snr: dict[int, list[list[int]]],
    cm_overall: list[list[int]],
    classes: tuple[str, ...],
    out_grid_path: Path,
    out_overall_path: Path,
) -> None:
    sns.set_theme(style="white", context="paper")

    # 2x2 grid for selected SNRs
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, snr in zip(axes.ravel(), CONFUSION_SNRS, strict=True):
        cm = np.array(cm_by_snr[snr], dtype=np.float64)
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm_norm = (cm / row_sums) * 100.0
        acc = np.trace(cm) / max(cm.sum(), 1.0)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            vmin=0,
            vmax=100,
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"SNR = {snr:+d} dB (Acc = {acc*100:.2f}%)", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=40)

    fig.suptitle("Normalized Confusion Matrices (%) at Critical SNR Levels", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_grid_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Overall confusion matrix across all SNRs
    cm_all = np.array(cm_overall, dtype=np.float64)
    cm_all_norm = (cm_all / np.maximum(cm_all.sum(axis=1, keepdims=True), 1.0)) * 100.0
    fig2, ax2 = plt.subplots(figsize=(9.5, 8.0))
    sns.heatmap(
        cm_all_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Row-Normalized Accuracy (%)"},
        ax=ax2,
    )
    overall_acc = np.trace(cm_all) / max(cm_all.sum(), 1.0)
    ax2.set_title(f"Overall Test Confusion Matrix Across All 20 SNRs (Acc = {overall_acc*100:.2f}%)", fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.tick_params(axis="x", rotation=40)
    fig2.tight_layout()
    fig2.savefig(out_overall_path, bbox_inches="tight", dpi=300)
    plt.close(fig2)


def run_baseline_pipeline(
    splits_dir: str | Path = "data/splits",
    output_dir: str | Path = "artifacts/baseline",
    seed: int = 42,
    strategies: tuple[NormalizationStrategy, ...] = ("raw", "l2_frame", "channel_zscore"),
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    dropout: float = 0.6,
) -> dict:
    """Run validation-only preprocessing ablation, train best VT-CNN2 baseline, and evaluate on test."""
    out_root = Path(output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_splits(splits_dir)
    split_audit = audit_splits(splits)

    # Fit label mapping strictly on training labels and save it
    label_map = LabelMap.from_labels(splits["train"].modulation_labels)
    label_map.save(out_root / "label_mapping.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Validation-only comparison across normalization strategies (test split untouched)
    ablation_summaries: list[dict] = []
    best_candidate: tuple[float, TrainingConfig, VTCNN2, IQNormalizer, list[dict], dict] | None = None

    for strat in strategies:
        cfg = TrainingConfig(
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            dropout=dropout,
            normalization=strat,
        )
        model, normalizer, history, train_meta = train_vtcnn2(
            splits["train"], splits["validation"], label_map, cfg, device=device
        )
        ablation_summaries.append(train_meta)

        # Select lowest validation loss (tie-breaker: prefer raw if virtually identical)
        score = train_meta["best_val_loss"]
        if best_candidate is None or score < best_candidate[0] - 1e-4:
            best_candidate = (score, cfg, model, normalizer, history, train_meta)

    assert best_candidate is not None
    _, best_cfg, best_model, best_normalizer, best_history, best_train_meta = best_candidate

    # Save chosen config, normalizer state, model weights, and training history
    (out_root / "config.json").write_text(
        json.dumps(
            {
                "training": best_cfg.to_dict(),
                "model": best_model.config.to_dict(),
                "selected_by": "minimum validation loss across normalization strategies",
                "candidate_strategies": list(strategies),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best_normalizer.save(out_root / "normalizer.json")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_config": best_model.config.to_dict(),
            "training_config": best_cfg.to_dict(),
            "classes": list(label_map.classes),
        },
        out_root / "vtcnn2_best.pt",
    )
    history_df = pd.DataFrame(best_history)
    history_df.to_csv(out_root / "training_history.csv", index=False)

    # Step 2: Single final evaluation on the untouched test split
    test_eval = evaluate_on_split(
        best_model, best_normalizer, label_map, splits["test"], batch_size=batch_size, device=device
    )

    per_snr_df = pd.DataFrame(test_eval["per_snr_metrics"])
    per_snr_df.to_csv(out_root / "test_metrics_by_snr.csv", index=False)
    per_class_snr_df = pd.DataFrame(test_eval["per_class_snr_metrics"])
    per_class_snr_df.to_csv(out_root / "test_metrics_by_class_snr.csv", index=False)

    # Save figures
    _plot_training_curves(history_df, best_train_meta["best_epoch"], fig_dir / "01_training_curves.png")
    _plot_norm_comparison(ablation_summaries, fig_dir / "02_normalization_comparison_val.png")
    _plot_test_accuracy_vs_snr(
        per_snr_df, test_eval["accuracy"], test_eval["macro_f1"], fig_dir / "03_accuracy_and_f1_vs_snr.png"
    )
    _plot_per_class_snr(per_class_snr_df, fig_dir / "04_per_class_accuracy_vs_snr.png")
    _plot_confusion_matrices(
        test_eval["confusion_matrices_selected_snrs"],
        test_eval["confusion_matrix_overall"],
        label_map.classes,
        fig_dir / "05_confusion_matrices_selected_snrs.png",
        fig_dir / "06_confusion_matrix_overall.png",
    )

    checkpoint_mb = (out_root / "vtcnn2_best.pt").stat().st_size / (1024**2)
    results = {
        "split_audit": split_audit,
        "normalization_ablation_val": ablation_summaries,
        "selected_model": {
            "normalization": best_cfg.normalization,
            "checkpoint_path": str((out_root / "vtcnn2_best.pt").resolve()),
            "checkpoint_size_mib": round(checkpoint_mb, 3),
            "training_summary": best_train_meta,
        },
        "test_evaluation": test_eval,
        "generated_figures": sorted(p.name for p in fig_dir.glob("*.png")),
    }
    (out_root / "baseline_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
