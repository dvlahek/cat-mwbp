"""Deterministic benchmark datasets."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


def _split_scale(x: Array, y: Array, seed: int, test_size: float = 0.25) -> Tuple[Array, Array, Array, Array]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test), y_train.astype(int), y_test.astype(int)


def _raw_dataset(name: str, seed: int = 0) -> Tuple[Array, Array]:
    if name == "moons":
        x, y = make_moons(n_samples=1200, noise=0.22, random_state=seed)
        return x, y
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    if name == "circles":
        x, y = make_circles(
            n_samples=1200, noise=0.10, factor=0.45, random_state=seed
        )
        return x, y
    if name == "iris":
        data = load_iris()
        return data.data, data.target
    if name == "wine":
        data = load_wine()
        return data.data, data.target
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    if name == "anisotropic":
        rng = np.random.default_rng(seed)
        n, d = 1600, 20
        latent = rng.normal(size=(n, d))
        logits = latent[:, 0] + 0.7 * latent[:, 1] - 0.4 * latent[:, 2] + 0.35 * rng.normal(size=n)
        y = (logits > 0.0).astype(int)
        rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scales = np.geomspace(1.0, 300.0, d)
        transform = rotation @ np.diag(scales) @ rotation.T
        x = latent @ transform
        return x, y
    if name == "synthetic_large":
        x, y = make_classification(
            n_samples=6000,
            n_features=100,
            n_informative=40,
            n_redundant=20,
            n_classes=10,
            n_clusters_per_class=1,
            class_sep=1.25,
            flip_y=0.03,
            random_state=seed,
        )
        return x, y
    raise ValueError(f"unknown dataset: {name}")


def load_dataset(name: str, seed: int = 0) -> Tuple[Array, Array, Array, Array]:
    """Return the legacy stratified 75/25 train/test protocol."""
    x, y = _raw_dataset(name, seed)
    return _split_scale(x, y, seed)


def load_dataset_three_way(
    name: str,
    seed: int = 0,
    test_size: float = 0.25,
    validation_size: float = 0.15,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Return leakage-controlled train/validation/test partitions.

    Fractions are relative to the complete dataset. The scaler is fitted only
    on the optimization-training partition; validation and test observations
    never affect feature normalization.
    """
    if test_size <= 0.0 or validation_size <= 0.0 or test_size + validation_size >= 1.0:
        raise ValueError("test and validation fractions must be positive and sum to less than one")
    x, y = _raw_dataset(name, seed)
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_validation = validation_size / (1.0 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_dev,
        y_dev,
        test_size=relative_validation,
        random_state=seed + 104729,
        stratify=y_dev,
    )
    scaler = StandardScaler().fit(x_train)
    return (
        scaler.transform(x_train),
        scaler.transform(x_validation),
        scaler.transform(x_test),
        y_train.astype(int),
        y_validation.astype(int),
        y_test.astype(int),
    )
