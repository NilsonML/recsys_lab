"""Turn raw ratings into clean implicit-feedback interactions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Self

import numpy as np
import polars as pl
import scipy.sparse as sp

from recsys.data.schema import ITEM_ID, RATING, USER_ID

logger = logging.getLogger(__name__)


def to_implicit(df: pl.DataFrame, *, min_rating: int | None = None) -> pl.DataFrame:
    """Keep interactions treated as positive feedback.

    Args:
        min_rating: Keep rows with rating >= min_rating. ``None`` keeps all
            ratings ("watched" signal). See EXPERIMENTS.md for the comparison.
    """
    if min_rating is None:
        return df
    return df.filter(pl.col(RATING) >= min_rating)


def filter_k_core(
    df: pl.DataFrame,
    *,
    min_user_interactions: int,
    min_item_interactions: int,
    max_iter: int = 100,
) -> pl.DataFrame:
    """Iteratively drop users/items below the thresholds until stable.

    Raises:
        RuntimeError: if not converged within ``max_iter`` iterations.
    """
    current_df = df
    for _ in range(max_iter):
        previous_shape = current_df.shape
        current_df = current_df.filter(
            (pl.len().over(USER_ID) >= min_user_interactions)
            & (pl.len().over(ITEM_ID) >= min_item_interactions)
        )
        # Dropping items can push users below the threshold (and vice versa),
        # so we only stop once a full pass removes nothing.
        if current_df.shape == previous_shape:
            return current_df

    raise RuntimeError(f"k-core filtering did not converge within {max_iter} iterations.")


@dataclass(frozen=True)
class IdMapper:
    """Bidirectional mapping between raw ids and contiguous indices 0..n-1.

    Contiguous indices are required to address rows of interaction matrices
    and embedding tables.

    Must be fitted on TRAIN only: test-only users/items are unknown to
    ID-based models, and fitting on all data would hide that fact.
    """

    user_ids: np.ndarray  # index -> raw user id
    item_ids: np.ndarray  # index -> raw item id

    @classmethod
    def fit(cls, train: pl.DataFrame) -> Self:
        """Collect unique user/item ids from ``train``."""
        # Sorting makes the id -> index assignment deterministic across runs.
        unique_users = train.select(USER_ID).unique().sort(USER_ID).to_series().to_numpy()
        unique_items = train.select(ITEM_ID).unique().sort(ITEM_ID).to_series().to_numpy()
        return cls(user_ids=unique_users, item_ids=unique_items)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace raw ids with contiguous indices.

        Unknown ids are dropped with a warning: ID-based models cannot score
        them; cold-start users are evaluated separately.
        """
        initial_row_count = len(df)

        user_map = pl.DataFrame(
            {
                USER_ID: self.user_ids,
                "user_idx": pl.int_range(0, self.n_users, dtype=pl.Int32, eager=True),
            }
        )
        item_map = pl.DataFrame(
            {
                ITEM_ID: self.item_ids,
                "item_idx": pl.int_range(0, self.n_items, dtype=pl.Int32, eager=True),
            }
        )

        # Left join leaves null indices for ids unseen in train.
        transformed_df = df.join(user_map, on=USER_ID, how="left").join(
            item_map, on=ITEM_ID, how="left"
        )
        clean_df = transformed_df.drop_nulls(subset=["user_idx", "item_idx"])

        dropped_rows = initial_row_count - len(clean_df)
        if dropped_rows > 0:
            logger.warning(
                "Dropped %d rows (%.2f%%) with user_id/item_id unseen in train.",
                dropped_rows,
                dropped_rows / initial_row_count * 100,
            )

        return clean_df.drop([USER_ID, ITEM_ID]).rename({"user_idx": USER_ID, "item_idx": ITEM_ID})

    def inverse_items(self, item_idx: np.ndarray) -> np.ndarray:
        """Map item indices back to raw item ids (for showing recommendations)."""
        return np.asarray(self.item_ids[item_idx])


def to_csr(df: pl.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Build a binary user-item matrix from index-mapped interactions."""
    rows = df.select(USER_ID).to_series().to_numpy()
    cols = df.select(ITEM_ID).to_series().to_numpy()
    # float32 rather than int8: models do float linear algebra on this matrix,
    # and float32 halves memory compared to the float64 default.
    data = np.ones(len(df), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
