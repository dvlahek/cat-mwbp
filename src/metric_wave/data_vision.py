"""Explicit real-data loaders for optional MNIST and CIFAR-10 probes."""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray
_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _subsample(x: Array, y: Array, count: Optional[int], seed: int):
    if count is None or count >= x.shape[0]:
        return x, y
    rng = np.random.default_rng(seed)
    index = rng.choice(x.shape[0], size=count, replace=False)
    return x[index], y[index]


def _load_mnist(count: Optional[int], seed: int) -> Tuple[Array, Array]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = np.asarray(data.data, dtype=float) / 255.0
    y = np.asarray(data.target, dtype=int)
    return _subsample(x, y, count, seed)


def _safe_extract(archive: tarfile.TarFile, root: Path) -> None:
    resolved_root = root.resolve()
    for member in archive.getmembers():
        destination = (root / member.name).resolve()
        if resolved_root not in destination.parents and destination != resolved_root:
            raise ValueError(f"unsafe CIFAR archive member: {member.name}")
    archive.extractall(root)


def _load_cifar10(root: Path, count: Optional[int], seed: int) -> Tuple[Array, Array]:
    archive_path = root / "cifar-10-python.tar.gz"
    folder = root / "cifar-10-batches-py"
    if not folder.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        if not archive_path.is_file():
            urllib.request.urlretrieve(_CIFAR_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, root)
    xs, ys = [], []
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        with (folder / name).open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        xs.append(np.asarray(batch[b"data"], dtype=float) / 255.0)
        ys.append(np.asarray(batch[b"labels"], dtype=int))
    return _subsample(np.concatenate(xs), np.concatenate(ys), count, seed)


def raw_vision_dataset(
    name: str,
    seed: int = 0,
    subsample: Optional[int] = None,
    cifar_root: str = os.path.expanduser("~/.cache/catmwbp/cifar10"),
) -> Tuple[Array, Array]:
    if name == "mnist":
        return _load_mnist(subsample, seed)
    if name == "cifar10":
        return _load_cifar10(Path(cifar_root), subsample, seed)
    raise ValueError(name)


def vision_widths(name: str) -> list:
    return {
        "mnist": [784, 256, 256, 192, 128, 64, 10],
        "cifar10": [3072, 512, 512, 384, 256, 128, 10],
    }[name]


__all__ = ["raw_vision_dataset", "vision_widths"]

