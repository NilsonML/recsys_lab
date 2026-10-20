"""Common interface every recommender implements."""

from typing import Protocol, Self

import numpy as np
import scipy.sparse as sp


class Recommender(Protocol):
    """TODO: decide Protocol vs. abc.ABC and document why."""

    def fit(self, train: sp.csr_matrix) -> Self:
        """Train on a binary user-item matrix of shape (n_users, n_items).

        TODO: csr_matrix is enough for Popularity/EASE/MF, but Two-Tower needs
        features and timestamps. Keep it and change at stage 6, or pass a
        richer object now? (YAGNI vs. reworking the interface later.)
        """
        ...

    def recommend(
        self,
        user_idx: np.ndarray,
        k: int,
        *,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        """Return top-k item indices, shape (len(user_idx), k), best first."""
        ...
