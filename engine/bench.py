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

"""Throughput benchmark — proves per-event cost at corpus scale.

Run: python3 bench.py [n_events]
Target: the pilot corpus projects to ~100k events; we need >>100 ev/s so a full
mailbox processes in minutes, not days.
"""
import sys
import time

sys.path.insert(0, "src")

from memeng.engine import MemoryEngine
from memeng.models import Event
from memeng.store import SQLiteStore


def main(n: int = 10_000) -> None:
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s)
    e.init_database()
    e.attach_stream("mail:p1", "mail")
    # seed a few rules so the predict stage has real work
    for i, (k, v, ok, ov) in enumerate([
        ("subject_norm", "invoice", "sender", "ap"),
        ("subject_norm", "raidevent", "sender", "officer"),
        ("subject_norm", "newsletter", "sender", "noreply"),
    ]):
        e.store.upsert_rule(domain_id=2, key=k, val=v,
                            outcome_key=ok, outcome_val=ov)

    senders = ["p_boss", "p_wife", "p_ap", "p_stranger1", "p_stranger2"]
    subjects = ["invoice batch", "weekly report", "hello", "raidevent plan",
                "newsletter digest", "quick question", "meeting notes"]
    events = []
    for i in range(n):
        events.append(Event(
            stream="mail:p1", kind="message",
            persons=[senders[i % len(senders)]],
            features={"subject_norm": subjects[i % len(subjects)],
                      "sender": senders[i % len(senders)],
                      "arousal": (0.95 if i % 97 == 0 else 0.0)},
            fuzz_width=None if i % 3 else __import__("datetime").timedelta(hours=2),
        ))

    t0 = time.perf_counter()
    results = e.new_events(events)
    dt = time.perf_counter() - t0
    rate = n / dt

    print(f"ingested {n} events in {dt:.2f}s -> {rate:,.0f} events/s "
          f"({1000*dt/n:.2f} ms/event)")
    print(e.telemetry_line())
    rep = e.telemetry_report()["stages"]
    slowest = sorted(rep.items(), key=lambda kv: -kv[1]["mean_ms"])[:3]
    print("slowest stages:", ", ".join(k for k, _ in slowest))
    print("store rows:", s.count_episodes(), "episodes")

    assert rate > 200, f"throughput {rate:.0f}/s below floor of 200/s"
    print("PASS (>= 200 ev/s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
