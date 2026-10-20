"""Download and extract the MovieLens-1M dataset."""

from __future__ import annotations

import logging
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import truststore

logger = logging.getLogger(__name__)

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(raw_dir: Path, *, force: bool = False) -> Path:
    """Download ML-1M into ``raw_dir`` and return the extracted folder.

    Idempotent: does nothing if the data is already present, unless ``force``.
    """
    target_dir = raw_dir / "ml-1m"
    if target_dir.exists() and not force:
        logger.info("MovieLens-1M already present at %s", target_dir)
        return target_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "ml-1m.zip"

    logger.info("Downloading %s", ML_1M_URL)
    # Verify against the OS trust store: the emSign root used by grouplens is
    # missing from both certifi and Python's OpenSSL store on Windows.
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with (
        urllib.request.urlopen(ML_1M_URL, context=ctx, timeout=60) as resp,
        archive_path.open("wb") as f,
    ):
        shutil.copyfileobj(resp, f)

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(raw_dir)
    archive_path.unlink()

    logger.info("Extracted to %s", target_dir)
    return target_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_movielens_1m(Path("data/raw"))
