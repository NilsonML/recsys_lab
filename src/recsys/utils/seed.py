"""Reproducibility helpers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a fresh ``Generator``.

    Prefer passing the returned generator explicitly over relying on global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG, still used by some libs
    return np.random.default_rng(seed)
