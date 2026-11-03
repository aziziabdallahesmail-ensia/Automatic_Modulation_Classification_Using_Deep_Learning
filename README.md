# RadioML download and split

code to download RadioML 2016.10A and save a reproducible
**70% training / 15% validation / 15% test** split, stratified jointly by
modulation and SNR.

Run from this repository:

```bash
uv sync --dev --locked
uv run python -m deep_learning_for_signal_modulation
```

The downloader uses exactly:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nolasthitnotomorrow/radioml2016-deepsigcom")

print("Path to dataset files:", path)
```

It locates `RML2016.10a_dict.pkl` inside the returned directory. KaggleHub handles
caching and any required authentication. Set `KAGGLEHUB_CACHE="$PWD/data/raw/kagglehub"`
to keep the cache inside the repository. Resolving the latest version can require
network access even when files are cached.

To use an existing trusted pickle completely offline:

```bash
uv run python -m deep_learning_for_signal_modulation \
  --dataset-path data/raw/RML2016.10a_dict.pkl \
  --output-dir data/splits \
  --seed 42
```

All relative paths resolve from the working directory. Existing output directories
are refused; select a new `--output-dir` to create another split.

The three output files are `train.npz`, `validation.npz`, and `test.npz`. Each has:

| Array | Meaning |
|---|---|
| `X` | float32 I/Q frames `(N, 2, 128)`, with I first and Q second |
| `modulation_labels` | Unicode modulation names `(N,)`, without label encoding |
| `snr_db` | int64 SNR metadata in dB `(N,)` |
| `sample_ids` | int64 global row IDs `(N,)`, identifying the saved partition |

Load a split with:

```python
import numpy as np

with np.load("data/splits/train.npz", allow_pickle=False) as training:
    X = training["X"]
    labels = training["modulation_labels"]
    snr_db = training["snr_db"]
```

Blocks are ordered by modulation name and numeric SNR, retaining row order within
each block. Seeds 42 and 43 drive the two stratified splitting stages. Partitions
must be disjoint, exhaustive, and cover every observed class/SNR pair; insufficient
strata raise an error. Signals retain source amplitude except for float32
conversion.
Disjoint frame IDs alone do not establish independence between related captures.

Only unpickle trusted datasets. Raw data and saved splits are ignored by Git.


