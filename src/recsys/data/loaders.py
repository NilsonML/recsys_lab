"""Load raw MovieLens-1M files into typed DataFrames."""

from pathlib import Path

import polars as pl

from recsys.data.schema import INTERACTIONS_SCHEMA, ITEM_ID, RATING, TIMESTAMP, USER_ID

SEP = "::"
# MovieLens-1M .dat files are Latin-1 (some movie titles contain non-ASCII chars).
ENCODING = "latin-1"


def _read_dat(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read a ``::``-separated .dat file into string columns.

    pyarrow and ``pl.read_csv`` only accept single-char separators and polars
    only reads utf8, so lines are read in Python and split with polars.
    """
    lines = path.read_text(encoding=ENCODING).splitlines()
    return (
        pl.DataFrame({"raw": lines})
        .select(pl.col("raw").str.split_exact(SEP, len(columns) - 1).struct.rename_fields(columns))
        .unnest("raw")
    )


def load_ratings(data_dir: Path) -> pl.DataFrame:
    """Load ``ratings.dat`` (format: UserID::MovieID::Rating::Timestamp).

    Returns:
        DataFrame matching ``INTERACTIONS_SCHEMA``.
    """
    df = _read_dat(data_dir / "ratings.dat", [USER_ID, ITEM_ID, RATING, TIMESTAMP])
    return df.cast(INTERACTIONS_SCHEMA)


def load_movies(data_dir: Path) -> pl.DataFrame:
    """Load ``movies.dat`` (format: MovieID::Title::Genres, genres split by '|').

    Returns:
        DataFrame with columns item_id, title, genres (list[str]).
    """
    df = _read_dat(data_dir / "movies.dat", [ITEM_ID, "title", "genres"])
    return df.with_columns(
        pl.col(ITEM_ID).cast(INTERACTIONS_SCHEMA[ITEM_ID]),
        pl.col("genres").str.split("|"),
    )
