"""Load the RadioML dictionary into aligned signals, labels, and SNRs."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignalDataset:
    """Raw I/Q (N, 2, 128), string modulation labels (N,), and SNR in dB (N,)."""

    X: NDArray[np.float32]
    modulation_labels: NDArray[np.str_]
    snr_db: NDArray[np.int64]


def load_radioml(path: str | Path) -> SignalDataset:
    """Read a trusted pickle, sorting blocks by modulation then numeric SNR.

    Within each block, preserve source row order and I/Q channel order. Only
    convert signal dtype to float32; do not normalize the signal amplitudes.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty dictionary of RadioML signal blocks")

    blocks = {}
    for key, block in raw.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid block key {key!r}; expected (modulation, SNR)")
        label, snr = key
        if isinstance(label, bytes):
            label = label.decode("ascii", errors="strict")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid modulation label in block {key!r}")
        if isinstance(snr, (bool, np.bool_)) or not isinstance(
            snr, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"SNR must be an integer in dB in block {key!r}")
        if isinstance(snr, (float, np.floating)) and (
            not np.isfinite(snr) or snr != int(snr)
        ):
            raise ValueError(f"SNR must be a finite integer in dB in block {key!r}")
        snr = int(snr)
        if not np.iinfo(np.int64).min <= snr <= np.iinfo(np.int64).max:
            raise ValueError(f"SNR is outside int64 range in block {key!r}")
        normalized_key = label, snr
        if normalized_key in blocks:
            raise ValueError(
                f"Duplicate block after label decoding: {normalized_key!r}"
            )
        if (
            not isinstance(block, np.ndarray)
            or block.dtype.kind not in "iuf"
            or block.ndim != 3
            or block.shape[1:] != (2, 128)
            or len(block) == 0
            or not np.isfinite(block).all()
        ):
            raise ValueError(
                f"Block {key!r}, shape {getattr(block, 'shape', None)}: expected finite real signals (frames > 0, 2, 128)"
            )
        blocks[normalized_key] = block

    count = sum(len(block) for block in blocks.values())
    X = np.empty((count, 2, 128), dtype=np.float32)
    labels = np.empty(count, dtype=f"U{max(len(label) for label, _ in blocks)}")
    snrs = np.empty(count, dtype=np.int64)
    start = 0
    for (label, snr), block in sorted(blocks.items()):
        stop = start + len(block)
        with np.errstate(over="ignore", invalid="ignore"):
            X[start:stop] = block
        if not np.isfinite(X[start:stop]).all():
            raise ValueError(
                f"Block {(label, snr)!r}: overflow during float32 conversion"
            )
        labels[start:stop] = label
        snrs[start:stop] = snr
        start = stop
    return SignalDataset(X, labels, snrs)
