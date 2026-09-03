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

"""Telemetry — per-stage timers, counters, percentile reports.

Every new_event records stage durations (predict/gate/encode/reinforce/
store-write) plus counters (events, encoded, reinforced, triggers). The
report gives count/mean/p50/p95/max per stage so bottlenecks name themselves.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager


class Telemetry:
    def __init__(self) -> None:
        self.stages: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name].append((time.perf_counter() - t0) * 1000.0)

    def incr(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def report(self) -> dict:
        out: dict[str, dict] = {}
        for name, vals in sorted(self.stages.items()):
            out[name] = {
                "n": len(vals),
                "mean_ms": round(statistics.fmean(vals), 3),
                "p50_ms": round(statistics.median(vals), 3),
                "p95_ms": round(_pct(vals, 95), 3),
                "max_ms": round(max(vals), 3),
            }
        return {"stages": out, "counters": dict(self.counters)}

    def summary_line(self) -> str:
        r = self.report()["stages"]
        parts = [f"{k}:{v['mean_ms']:.2f}ms(n={v['n']})" for k, v in r.items()]
        c = self.counters
        return " | ".join(parts) + f" | encoded={c.get('encoded', 0)}" \
               f" reinforced={c.get('reinforced', 0)} events={c.get('events', 0)}"


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]
