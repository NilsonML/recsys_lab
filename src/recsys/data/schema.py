"""Column names and dtypes shared across the project.

Single source of truth: no other module hardcodes column names.
"""

from typing import Final

import polars as pl

USER_ID: Final = "user_id"
ITEM_ID: Final = "item_id"
RATING: Final = "rating"
TIMESTAMP: Final = "timestamp"

INTERACTIONS_SCHEMA: Final = pl.Schema(
    {
        USER_ID: pl.Int32,
        ITEM_ID: pl.Int32,
        RATING: pl.Int8,
        # Unix seconds: models only need time deltas, not calendar dates.
        TIMESTAMP: pl.Int64,
    }
)


def validate_interactions(df: pl.DataFrame) -> None:
    """Raise ValueError if ``df`` does not match INTERACTIONS_SCHEMA.

    TODO: decide what else to check besides columns and dtypes (nulls? duplicates?).
    """
    raise NotImplementedError
