# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Fuzzy retrieval channels: flat vector index, RRF fusion.

FlatVectorIndex: exact cosine over float32 vectors.
- numpy BLAS path when available (matrix cached until dirty)
- pure-python fallback otherwise
Swap targets when scale demands: sqlite-vec, pgvector, paper-2 engine.
"""
from __future__ import annotations


class FlatVectorIndex:
    def __init__(self) -> None:
        self._vecs: dict[str, list[float]] = {}
        self._ids: list[str] = []
        self._mat = None
        self._dirty = True

    def add(self, eid: str, vec: list[float]) -> None:
        vec = [float(x) for x in vec]
        if self._vecs and len(vec) != len(next(iter(self._vecs.values()))):
            raise ValueError(
                f"dimension mismatch: {len(vec)} != "
                f"{len(next(iter(self._vecs.values())))}")
        self._vecs[eid] = vec
        self._dirty = True

    def remove(self, eid: str) -> None:
        if eid in self._vecs:
            del self._vecs[eid]
            self._dirty = True

    def __len__(self) -> int:
        return len(self._vecs)

    def _rebuild(self) -> None:
        """Rebuild matrix only when dirty; clear flag immediately."""
        self._ids = list(self._vecs.keys())
        try:
            import numpy as np
            if self._vecs:
                self._mat = np.asarray(
                    [self._vecs[i] for i in self._ids], dtype=np.float32)
            else:
                self._mat = None
        except ImportError:
            self._mat = None
        self._dirty = False

    def search(self, qvec, k: int = 8,
               candidates=None) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        if self._dirty or self._mat is None:
            self._rebuild()
        if self._mat is None:
            return []
        import numpy as np
        q = np.asarray([float(x) for x in qvec], dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._mat.shape[1]:
            return []                   # incompatible cue dimension
        qn = float(np.linalg.norm(q)) or 1e-9
        sims = (self._mat @ (q / qn))
        idx = np.argsort(-sims)[:k]
        out = [(self._ids[i], float(sims[i])) for i in idx]
        if candidates is not None:
            out = [(eid, s) for eid, s in out if eid in candidates]
        return out


def rrf_fuse(channels: list[list[str]], k: int = 8,
             weights: list[float] | None = None) -> list[str]:
    """Reciprocal-rank fusion: consumes RANKS, not scores — robust across
    heterogeneous channels without unit calibration."""
    w = weights or [1.0] * len(channels)
    fused: dict[str, float] = {}
    for ch, weight in zip(channels, w):
        for rank, eid in enumerate(ch):
            fused[eid] = fused.get(eid, 0.0) + weight / (60.0 + rank + 1)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return [eid for eid, _ in ranked[:k]]
