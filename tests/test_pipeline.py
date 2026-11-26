"""Unit tests for preprocessing, leakage checks, and Table 1 VT-CNN2 architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deep_learning_for_signal_modulation.model import VTCNN2, VTCNN2Config
from deep_learning_for_signal_modulation.preprocessing import (
    IQNormalizer,
    LabelMap,
    SplitData,
    audit_splits,
    set_seed,
    to_vtcnn2_tensor,
)


def test_label_map_deterministic_roundtrip(tmp_path: Path) -> None:
    labels = np.array(["QPSK", "BPSK", "16QAM", "QPSK", "AM-DSB"])
    lm = LabelMap.from_labels(labels)
    assert lm.classes == ("16QAM", "AM-DSB", "BPSK", "QPSK")
    encoded = lm.encode(labels)
    assert encoded.tolist() == [3, 2, 0, 3, 1]
    assert lm.decode(encoded).tolist() == labels.tolist()

    save_path = tmp_path / "labels.json"
    lm.save(save_path)
    loaded = LabelMap.load(save_path)
    assert loaded == lm

    with pytest.raises(ValueError, match="Unknown modulation labels"):
        lm.encode(np.array(["64QAM"]))


def test_normalizer_leakage_free_and_l2_unit_norm(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=2.0, scale=0.5, size=(64, 2, 128)).astype(np.float32)
    X_test = rng.normal(loc=100.0, scale=10.0, size=(16, 2, 128)).astype(np.float32)

    # Unfitted normalizer must fail
    unfitted = IQNormalizer(strategy="channel_zscore")
    with pytest.raises(RuntimeError, match="must be fit"):
        unfitted.transform(X_train)

    # Channel z-score must use ONLY X_train statistics, never X_test
    z_norm = IQNormalizer(strategy="channel_zscore").fit(X_train)
    X_train_z = z_norm.transform(X_train)
    X_test_z = z_norm.transform(X_test)
    assert np.allclose(X_train_z.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(X_train_z.std(axis=(0, 2)), 1.0, atol=1e-4)
    # Test set should NOT have zero mean because train stats were applied
    assert abs(X_test_z.mean()) > 50.0

    # Serialization roundtrip
    norm_path = tmp_path / "norm.json"
    z_norm.save(norm_path)
    reloaded = IQNormalizer.load(norm_path)
    assert np.allclose(reloaded.transform(X_test), X_test_z)

    # L2 frame normalization must yield unit L2 norm per frame independently
    l2_norm = IQNormalizer(strategy="l2_frame").fit(X_train)
    X_l2 = l2_norm.transform(X_test)
    norms = np.sqrt(np.sum(X_l2**2, axis=(1, 2)))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_audit_splits_catches_overlap_leakage() -> None:
    rng = np.random.default_rng(0)
    def _make_split(name: str, ids: list[int]) -> SplitData:
        n = len(ids)
        return SplitData(
            name=name,
            X=rng.normal(size=(n, 2, 128)).astype(np.float32),
            modulation_labels=np.array(["BPSK", "QPSK"] * (n // 2)),
            snr_db=np.array([0, 10] * (n // 2), dtype=np.int64),
            sample_ids=np.array(ids, dtype=np.int64),
        )

    valid_splits = {
        "train": _make_split("train", [0, 1, 2, 3]),
        "validation": _make_split("validation", [4, 5]),
        "test": _make_split("test", [6, 7]),
    }
    report = audit_splits(valid_splits)
    assert report["exhaustive_sample_ids"] is True
    assert report["overlaps"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}

    leaked_splits = {
        "train": _make_split("train", [0, 1, 2, 3]),
        "validation": _make_split("validation", [4, 5]),
        "test": _make_split("test", [3, 7]),  # ID 3 leaked from train
    }
    with pytest.raises(ValueError, match="Data leakage detected"):
        audit_splits(leaked_splits)


def test_vtcnn2_table1_architecture_shapes_and_params() -> None:
    set_seed(42)
    model = VTCNN2(VTCNN2Config(num_classes=11, dropout=0.6))
    summary = model.architecture_summary()

    keras_shapes = {row["layer"]: row["output_shape_keras_equiv"] for row in summary}
    assert keras_shapes["Input"] == [2, 128, 1]
    assert keras_shapes["ZeroPadding2D_1"] == [2, 132, 1]
    assert keras_shapes["Conv2D_1"] == [2, 130, 256]
    assert keras_shapes["ZeroPadding2D_2"] == [2, 134, 256]
    assert keras_shapes["Conv2D_2"] == [1, 132, 80]
    assert keras_shapes["Flatten"] == [10560]
    assert keras_shapes["Dense_1 (FC)"] == [256]
    assert keras_shapes["Dense_2 (Output)"] == [11]

    # Exact parameter count for Table 1:
    # Conv1: 256*(1*3) + 256 = 1,024
    # Conv2: 80*(256*2*3) + 80 = 122,960
    # FC1: 10560*256 + 256 = 2,703,616
    # FC2: 256*11 + 11 = 2,827
    # Total = 2,830,427
    assert model.count_parameters()["total"] == 2_830_427

    x_np = np.random.randn(4, 2, 128).astype(np.float32)
    x_t = to_vtcnn2_tensor(x_np)
    feats = model.forward_features(x_t)
    logits = model(x_t)
    assert feats.shape == (4, 256)
    assert logits.shape == (4, 11)
    loss = logits.sum()
    loss.backward()
    assert model.conv1.weight.grad is not None
