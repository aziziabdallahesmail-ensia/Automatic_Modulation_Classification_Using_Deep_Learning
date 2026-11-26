"""Leakage-free preprocessing, label encoding, split auditing, and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

NormalizationStrategy = Literal[
    "raw",
    "l2_frame",
    "rms_frame",
    "channel_zscore",
    "global_zscore",
]

VALID_NORMALIZATIONS: tuple[NormalizationStrategy, ...] = (
    "raw",
    "l2_frame",
    "rms_frame",
    "channel_zscore",
    "global_zscore",
)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class SplitData:
    """In-memory representation of a saved split NPZ file."""

    name: str
    X: NDArray[np.float32]
    modulation_labels: NDArray[np.str_]
    snr_db: NDArray[np.int64]
    sample_ids: NDArray[np.int64]


def load_split(path: str | Path, name: str | None = None) -> SplitData:
    """Load one saved NPZ split without pickle deserialization."""
    split_path = Path(path)
    split_name = name or split_path.stem
    with np.load(split_path, allow_pickle=False) as archive:
        return SplitData(
            name=split_name,
            X=archive["X"].astype(np.float32, copy=False),
            modulation_labels=archive["modulation_labels"],
            snr_db=archive["snr_db"].astype(np.int64, copy=False),
            sample_ids=archive["sample_ids"].astype(np.int64, copy=False),
        )


def load_all_splits(splits_dir: str | Path) -> dict[str, SplitData]:
    """Load train, validation, and test splits from a directory."""
    root = Path(splits_dir)
    return {
        name: load_split(root / f"{name}.npz", name=name)
        for name in ("train", "validation", "test")
    }


@dataclass(frozen=True)
class LabelMap:
    """Deterministic bijective mapping between modulation names and integer IDs (0..K-1)."""

    classes: tuple[str, ...]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]

    @classmethod
    def from_labels(cls, labels: NDArray[np.str_]) -> LabelMap:
        """Fit class mapping strictly from sorted unique labels (typically training split)."""
        unique_classes = tuple(sorted(str(label) for label in np.unique(labels)))
        if not unique_classes:
            raise ValueError("Cannot build LabelMap from empty label array")
        class_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        idx_to_class = {idx: label for idx, label in enumerate(unique_classes)}
        return cls(
            classes=unique_classes,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
        )

    def encode(self, labels: NDArray[np.str_]) -> NDArray[np.int64]:
        """Convert string modulation labels into integer class indices."""
        unknown = set(str(lbl) for lbl in np.unique(labels)) - set(self.classes)
        if unknown:
            raise ValueError(f"Unknown modulation labels encountered: {sorted(unknown)}")
        return np.array(
            [self.class_to_idx[str(lbl)] for lbl in labels], dtype=np.int64
        )

    def decode(self, indices: NDArray[np.integer]) -> NDArray[np.str_]:
        """Convert integer class indices back into string modulation names."""
        return np.array([self.idx_to_class[int(idx)] for idx in indices])

    def to_dict(self) -> dict:
        """Serialize mapping to JSON-compatible dictionary."""
        return {
            "classes": list(self.classes),
            "class_to_idx": dict(self.class_to_idx),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> LabelMap:
        """Reconstruct LabelMap from a serialized dictionary."""
        classes = tuple(str(c) for c in payload["classes"])
        class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
        idx_to_class = {idx: label for label, idx in class_to_idx.items()}
        return cls(classes=classes, class_to_idx=class_to_idx, idx_to_class=idx_to_class)

    def save(self, path: str | Path) -> None:
        """Persist label mapping as formatted JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LabelMap:
        """Load label mapping from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def audit_splits(splits: dict[str, SplitData]) -> dict:
    """Verify disjointness, non-leakage, shapes, finiteness, and stratification integrity."""
    required = ("train", "validation", "test")
    if tuple(splits.keys()) != required:
        raise ValueError(f"Expected splits {required}, got {tuple(splits.keys())}")

    id_sets = {name: set(splits[name].sample_ids.tolist()) for name in required}
    overlap_tv = len(id_sets["train"] & id_sets["validation"])
    overlap_tt = len(id_sets["train"] & id_sets["test"])
    overlap_vt = len(id_sets["validation"] & id_sets["test"])
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Data leakage detected! Overlapping sample_ids: "
            f"train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}"
        )

    total_samples = sum(len(splits[name].X) for name in required)
    all_ids = np.concatenate([splits[name].sample_ids for name in required])
    exhaustive = bool(np.array_equal(np.sort(all_ids), np.arange(total_samples)))

    summary: dict[str, object] = {
        "total_samples": int(total_samples),
        "exhaustive_sample_ids": exhaustive,
        "overlaps": {
            "train_validation": overlap_tv,
            "train_test": overlap_tt,
            "validation_test": overlap_vt,
        },
        "splits": {},
    }

    train_strata: set[tuple[str, int]] | None = None
    for name in required:
        sp = splits[name]
        if sp.X.ndim != 3 or sp.X.shape[1:] != (2, 128):
            raise ValueError(f"{name}.X has unexpected shape {sp.X.shape}")
        if not np.isfinite(sp.X).all():
            raise ValueError(f"{name}.X contains NaN or Inf values")

        strata_counts: dict[tuple[str, int], int] = {}
        for lbl, snr in zip(sp.modulation_labels, sp.snr_db, strict=True):
            key = (str(lbl), int(snr))
            strata_counts[key] = strata_counts.get(key, 0) + 1

        if train_strata is None:
            train_strata = set(strata_counts.keys())
        elif set(strata_counts.keys()) != train_strata:
            raise ValueError(f"Strata mismatch between train and {name}")

        counts_arr = np.array(list(strata_counts.values()), dtype=np.int64)
        summary["splits"][name] = {
            "count": int(len(sp.X)),
            "fraction": round(len(sp.X) / total_samples, 6),
            "shape": list(sp.X.shape),
            "dtype": str(sp.X.dtype),
            "num_classes": int(len(np.unique(sp.modulation_labels))),
            "num_snrs": int(len(np.unique(sp.snr_db))),
            "num_strata": int(len(strata_counts)),
            "stratum_min_count": int(counts_arr.min()),
            "stratum_max_count": int(counts_arr.max()),
            "finite": True,
        }

    return summary


@dataclass
class IQNormalizer:
    """Leakage-free I/Q frame normalizer fitted strictly on training data."""

    strategy: NormalizationStrategy = "raw"
    eps: float = 1e-8
    mean_: NDArray[np.float32] | None = None
    std_: NDArray[np.float32] | None = None
    fitted_: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization {self.strategy!r}; choose from {VALID_NORMALIZATIONS}"
            )

    def fit(self, X_train: NDArray[np.float32]) -> IQNormalizer:
        """Estimate normalization parameters strictly from training frames."""
        if X_train.ndim != 3 or X_train.shape[1:] != (2, 128):
            raise ValueError(f"Expected X_train shape (N, 2, 128), got {X_train.shape}")

        if self.strategy in ("raw", "l2_frame", "rms_frame"):
            # Stateless per-frame (or identity) strategies do not depend on dataset statistics.
            self.mean_ = None
            self.std_ = None
        elif self.strategy == "channel_zscore":
            mean = X_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
            std = X_train.std(axis=(0, 2), keepdims=True, dtype=np.float64)
            self.mean_ = mean.astype(np.float32)
            self.std_ = np.maximum(std, self.eps).astype(np.float32)
        elif self.strategy == "global_zscore":
            mean = np.array([[[X_train.mean(dtype=np.float64)]]], dtype=np.float64)
            std = np.array([[[X_train.std(dtype=np.float64)]]], dtype=np.float64)
            self.mean_ = mean.astype(np.float32)
            self.std_ = np.maximum(std, self.eps).astype(np.float32)

        self.fitted_ = True
        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply fitted normalization to any split without re-estimating statistics."""
        if not self.fitted_:
            raise RuntimeError("IQNormalizer must be fit on training data before transform()")
        if X.ndim != 3 or X.shape[1:] != (2, 128):
            raise ValueError(f"Expected X shape (N, 2, 128), got {X.shape}")

        X_f32 = X.astype(np.float32, copy=False)
        if self.strategy == "raw":
            return X_f32.copy()
        if self.strategy == "l2_frame":
            norms = np.sqrt(np.sum(X_f32 ** 2, axis=(1, 2), keepdims=True))
            return (X_f32 / np.maximum(norms, self.eps)).astype(np.float32, copy=False)
        if self.strategy == "rms_frame":
            rms = np.sqrt(np.mean(X_f32 ** 2, axis=(1, 2), keepdims=True))
            return (X_f32 / np.maximum(rms, self.eps)).astype(np.float32, copy=False)
        if self.strategy in ("channel_zscore", "global_zscore"):
            assert self.mean_ is not None and self.std_ is not None
            return ((X_f32 - self.mean_) / self.std_).astype(np.float32, copy=False)
        raise AssertionError(f"Unhandled strategy {self.strategy}")

    def fit_transform(self, X_train: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit on training data and return transformed training data."""
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> dict:
        """Serialize normalizer state to JSON-compatible dictionary."""
        return {
            "strategy": self.strategy,
            "eps": float(self.eps),
            "fitted": bool(self.fitted_),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> IQNormalizer:
        """Reconstruct normalizer state from dictionary."""
        norm = cls(strategy=payload["strategy"], eps=float(payload["eps"]))
        norm.fitted_ = bool(payload["fitted"])
        norm.mean_ = (
            None if payload["mean"] is None else np.array(payload["mean"], dtype=np.float32)
        )
        norm.std_ = (
            None if payload["std"] is None else np.array(payload["std"], dtype=np.float32)
        )
        return norm

    def save(self, path: str | Path) -> None:
        """Persist normalizer configuration and fitted training stats to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IQNormalizer:
        """Load fitted normalizer from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def to_vtcnn2_tensor(X: NDArray[np.float32]) -> torch.Tensor:
    """Format (N, 2, 128) I/Q frames as NCHW tensor (N, 1, 2, 128) for VT-CNN2 Conv2D."""
    if X.ndim != 3 or X.shape[1:] != (2, 128):
        raise ValueError(f"Expected X shape (N, 2, 128), got {X.shape}")
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)


def make_dataloader(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    batch_size: int = 1024,
    shuffle: bool = False,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader for VT-CNN2."""
    inputs = to_vtcnn2_tensor(X)
    targets = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    dataset = TensorDataset(inputs, targets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=pin_memory,
    )
