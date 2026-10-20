"""Train/validation/test splitting strategies."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Split:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def global_temporal_split(
    df: pl.DataFrame,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Split:
    """Split by two global timestamp cutoffs: train < val < test in time.

    TODO: decide which part gets events that share the cutoff timestamp.
    """
    raise NotImplementedError


def leave_last_out(df: pl.DataFrame, *, n_val: int = 1, n_test: int = 1) -> Split:
    """Per user: last n_test events -> test, previous n_val -> val, rest -> train.

    Users with fewer than n_val + n_test + 1 events stay entirely in train.

    TODO: in ML-1M a user often rates many items within the same second.
    Decide which of them is "last"; the tie-break must be deterministic.
    """
    raise NotImplementedError
