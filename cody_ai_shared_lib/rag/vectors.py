"""Vector helpers shared by retrieval, ingestion and the reference store."""
from __future__ import annotations

from typing import Sequence


def is_zero_vector(vector: Sequence[float] | None) -> bool:
    """True for a missing, empty or all-zero vector, which is how a failed embedding is marked."""
    return not vector or all(component == 0.0 for component in vector)
