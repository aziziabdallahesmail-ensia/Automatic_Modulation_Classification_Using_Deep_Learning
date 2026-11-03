"""Deterministic joint modulation/SNR stratification of canonical row indices."""

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .data import SignalDataset

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitIndices:
    """Global row IDs; validation/test values never index a temporary local pool."""

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]


def _strata(y: NDArray, snr_db: NDArray) -> tuple[NDArray, NDArray]:
    if (
        y.ndim != 1
        or snr_db.shape != y.shape
        or y.dtype.kind not in "iuU"
        or snr_db.dtype.kind not in "iu"
        or len(y) == 0
    ):
        raise ValueError(
            "Stratification requires aligned, nonempty modulation labels and integer snr_db arrays"
        )
    # IDs are temporary strata inputs only; output labels remain strings.
    _, label_ids = np.unique(y, return_inverse=True)
    pairs, inverse = np.unique(
        np.column_stack((label_ids, snr_db)), axis=0, return_inverse=True
    )
    return pairs, inverse


def _sizes(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if (
        len(fractions) != 3
        or any(not math.isfinite(x) or not 0 < x < 1 for x in fractions)
        or not math.isclose(sum(fractions), 1, abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError("Three positive fractions summing to one are required")
    temporary = math.ceil(n * (fractions[1] + fractions[2]))
    test = math.ceil(temporary * (fractions[2] / (fractions[1] + fractions[2])))
    return n - temporary, temporary - test, test


def _check_feasible(
    strata: NDArray, pairs: NDArray, left: int, right: int, stage: str
) -> None:
    counts = np.bincount(strata, minlength=len(pairs))
    affected = [
        {"modulation_id": int(pair[0]), "snr_db": int(pair[1]), "count": int(n)}
        for pair, n in zip(pairs, counts, strict=True)
        if n < 2
    ]
    if affected or min(left, right) < len(pairs):
        raise ValueError(
            f"Infeasible {stage} joint stratification: subset sizes {left}/{right}, {len(pairs)} strata; counts below two: {affected}. Increase stratum counts or revise fractions."
        )


def create_split_indices(
    y: NDArray,
    snr_db: NDArray,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    """Split canonical row IDs twice, preserving the joint class/SNR mixture."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 2:
        raise ValueError("seed must be an integer in [0, 2**32 - 2]")
    pairs, strata = _strata(y, snr_db)
    n_train, n_validation, n_test = _sizes(len(y), fractions)
    _check_feasible(strata, pairs, n_train, n_validation + n_test, "training/temporary")
    train, temporary = train_test_split(
        np.arange(len(y), dtype=np.int64),
        test_size=fractions[1] + fractions[2],
        stratify=strata,
        shuffle=True,
        random_state=seed,
    )
    _check_feasible(strata[temporary], pairs, n_validation, n_test, "validation/test")
    validation, test = train_test_split(
        temporary,
        test_size=fractions[2] / (fractions[1] + fractions[2]),
        stratify=strata[temporary],
        shuffle=True,
        random_state=seed + 1,
    )
    splits = SplitIndices(train, validation, test)
    validate_split_indices(splits, y, snr_db, fractions)
    return splits


def _validate_quota(
    actual: NDArray, population: NDArray, selected: int, total: int, stage: str
) -> None:
    quota = population * (selected / total)
    # Each stage allocates floor/ceil of a proportional quota, not exact decimals.
    if np.any(actual < np.floor(quota + 1e-9)) or np.any(
        actual > np.ceil(quota - 1e-9)
    ):
        raise ValueError(
            f"Per-stratum counts exceed rounding allowances during {stage}"
        )


def validate_split_indices(
    splits: SplitIndices,
    y: NDArray,
    snr_db: NDArray,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> None:
    """Audit disjointness, exhaustive coverage, rounded sizes, and joint coverage."""
    pairs, strata = _strata(y, snr_db)
    expected = _sizes(len(y), fractions)
    arrays = [getattr(splits, name) for name in SPLIT_NAMES]
    for name, values, size in zip(SPLIT_NAMES, arrays, expected, strict=True):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if np.any(values < 0) or np.any(values >= len(y)):
            raise ValueError(f"{name} index out of bounds")
        if len(values) != len(np.unique(values)):
            raise ValueError(f"Duplicate {name} indices")
        if len(values) != size:
            raise ValueError(
                f"{name} size {len(values)} differs from two-stage expected size {size}"
            )
    combined = np.concatenate(arrays)
    if not np.array_equal(np.sort(combined), np.arange(len(y))):
        raise ValueError(
            "Split indices must be pairwise disjoint and exhaust all canonical rows"
        )
    counts = [np.bincount(strata[a], minlength=len(pairs)) for a in arrays]
    for name, values in zip(SPLIT_NAMES, counts, strict=True):
        if np.any(values == 0):
            missing = pairs[values == 0].tolist()
            raise ValueError(
                f"Joint strata missing from {name}: {missing}; per-stratum counts: {values.tolist()}"
            )
    total = np.bincount(strata, minlength=len(pairs))
    _validate_quota(counts[0], total, len(splits.train), len(y), "training/temporary")
    _validate_quota(
        counts[2],
        counts[1] + counts[2],
        len(splits.test),
        len(splits.validation) + len(splits.test),
        "validation/test",
    )


def save_splits(
    output_dir: str | Path, dataset: SignalDataset, splits: SplitIndices
) -> None:
    """Save aligned 70/15/15 NPZs; refuse to replace existing data."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}; choose a new output directory"
        )
    if dataset.X.shape != (len(dataset.modulation_labels), 2, 128):
        raise ValueError("Signals and modulation labels must be aligned (N, 2, 128)")
    validate_split_indices(splits, dataset.modulation_labels, dataset.snr_db)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.splitting-", dir=destination.parent
        )
    )
    try:
        for name in SPLIT_NAMES:
            ids = getattr(splits, name)
            np.savez(
                stage / f"{name}.npz",
                X=dataset.X[ids],
                modulation_labels=dataset.modulation_labels[ids],
                snr_db=dataset.snr_db[ids],
                sample_ids=ids,
            )
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
