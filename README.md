# Positronic-Engram — the polytemporal memory engine

### Deterministic, auditable, tensor-grounded memory for LLM agents

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![SQLite Powered](https://img.shields.io/badge/Storage-SQLite-lightgrey)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Recall](https://img.shields.io/badge/Recall-0.7ms%20store%20%E2%80%94%20real%20%CE%940.44-brightgreen)]()

`positronic-engram` is the engine behind [Positronic for opencode](https://github.com/ShingWong/positronic-opencode-plugin). It's a **polytemporal MemoryEngine** — SQLite-backed, deterministic, auditable, and built to make agent memory actually remember what matters.

Not a vector store wrapper. Not a summarization window. A cognitive loop: **predict → gate → encode | reinforce → consolidate → recall**, running on a typed polytemporal schema where every event carries wall-clock, monotonic, subjective **τ**, and a fuzzy interval.

`import memeng`, point it at a SQLite file, and your agent gets memory that outlives the session.

---

## Table of Contents

- [Why a memory engine?](#why-a-memory-engine)
- [Polytemporal retention — every event carries a time vector](#polytemporal-retention--every-event-carries-a-time-vector)
- [Retention profiles — pick a curve](#retention-profiles--pick-a-curve)
- [Retention profile knobs — tuning reference](#retention-profile-knobs--tuning-reference)
- [Logical time τ](#logical-time-τ)
- [Tensor-grounded objects — why recall is cheap](#tensor-grounded-objects--why-recall-is-cheap)
- [How fast is it?](#how-fast-is-it)
- [100% auditable](#100-auditable)
- [Usage](#usage)
- [Embedding tiers](#embedding-tiers)
- [Benchmarks](#benchmarks)
- [Ecosystem](#ecosystem)
- [License](#license)

---

## Why a memory engine?

Most agent memory is a single-brain design: one model or one embedding index decides what to store, how to consolidate it, and what to retrieve. Human memory works nothing like that — it's a federation of small specialized subsystems, none intelligent alone.

**Curation, not intelligence, is the hard problem of memory.**

`positronic-engram` implements that thesis as a reusable library:

- **Salience gate at encoding** — not everything gets stored. Surprise × goal-weight decides what survives.
- **Polytemporal retention** — decay runs on subjective time (τ), not timestamps.
- **Tensor-grounded objects** — entities become evolving objects that persist across episodes.
- **Deterministic recall** — Activate · Reconstruct · Decay · Fuse, no opaque magic.
- **SQLite-first** — one file per brain, offline, reproducible, auditable.

---

## Polytemporal retention — every event carries a time vector

Most systems stamp an event with one timestamp and call it done.
`positronic-engram` stores **four**:

| Coordinate | Type | What it is |
|---|---|---|
| `wall` | timestamptz | the human calendar time it happened |
| `mono` | bigint | the order it arrived in the stream |
| `tau` | double | the agent's **subjective** time — how much it felt like |
| `fuzz` | tstzrange | the confidence interval around "when" |

Decay doesn't run on wall-clock. It runs on **Δτ** — the subjective distance. That's what makes retention a *curve*, not a TTL counter.

---

## Retention profiles — pick a curve

Same 55 messages, 78 weeks, four policies (`engine.py:48`):

| Profile | S_base | Horizon | @Week 78 |
|---|---|---|---|
| `balanced` | 30 | weeks | 35 / 55 |
| `long_term` | 120 | months | 55 / 55 |
| `archival` | 1e6 | forever | 55 / 55 |
| `short_term` | 6 | days | 7 / 55 |

> **Demotion, not deletion.** Episodes below 0.35 drop to `day_token`; below 0.05 to `week_token` — they are never deleted (`expired` stays 0). The `@Week 78` column counts only `level="event"` episodes remaining after `prune()`.

`balanced` is the everyday default. `archival` never forgets — it grows forever, so it's gated behind a confirm. Pruning follows a weekly ladder: `0.35 → day_token 0.05 → expired` (`engine.py:443`).

---

## Retention profile knobs — tuning reference

Every knob resolves per-domain (via `register_domain(..., retention_profile=...)`), so one brain can mix policies. The authoritative copy of this reference lives in the source docstring at `engine.py:71`; this section is the project-facing version for agents and contributors.

### Episode decay — the verbatim event log

| Knob | Meaning | Units |
|---|---|---|
| `S_base` | baseline strength an episode is born with | τ |
| `S_arousal` | strength added per unit of event arousal (`strength = S_base + S_arousal × arousal`) | τ/arousal |
| `prune_merge` | retain threshold below which an episode demotes to `day_token` | probability |
| `prune_expire` | retain threshold below which an episode expires → residue written | probability |

Retention per episode is `exp(−Δτ/strength)`, so `S_base` is effectively the half-life in τ. Episodes demote, never delete: below `prune_merge` the subject is stripped (FTS kept); below `prune_expire` a residue is written.

### Object (entity) decay — the IP layer

| Knob | Meaning | Units |
|---|---|---|
| `obj_dormant` | τ of no sightings before an entity goes dormant | τ |
| `obj_forget` | τ of no sightings before an entity is forgotten | τ |

Both scale **×3** for entities with ≥3 sightings (repetition protects). Set `None` to make entities immortal (`archival`).

### Load-bearing TTL renewal — spaced repetition for entities

> The mechanism is the **Ebbinghaus forgetting curve** (the principle behind
> Anki and every spaced-repetition system): retention is proportional to
> demonstrated reliability, re-earned each cycle rather than granted once.
> The renewal update itself is the **Rescorla–Wagner** (1972) learning law:
> the association's strength adjusts proportionally to the discrepancy — here,
> the elapsed interval since the last renewal.

| Knob | Meaning | Units |
|---|---|---|
| `renew_ratio` | fraction of the elapsed interval added to an entity's survival clock when it is found still load-bearing at prune time | τ/τ |
| `renew_max` | absolute cap on `last_renewal_tau` — an eternally-mentioned entity cannot become de-facto immortal | τ |

The mechanism: when an entity **would** be forgotten (`Δτ ≥ obj_forget`), prune first asks *"is it still load-bearing?"* — named in a **consolidation** episode since the last prune, **or** re-sighted ≥2 times in real episodes since the last prune. If yes, its clock extends by `renew_ratio × (now − anchor)` (interval-based; the proportional update is the Rescorla–Wagner law) instead of forgetting; if no, it decays on schedule. Retention is **re-earned each cycle** — the moment an entity stops being load-bearing (e.g. a deployment target you've finished using), the check fails and it fades naturally. The asymmetry that motivates this: retaining is one row in SQLite; losing is a full re-derivation.

### Tuning guidance — the two dials that matter

- **τ burns fast under agentic load.** When predictions are empty, novelty is 0.9, so a busy session advances τ at ~0.9 per event — up to ~13k τ/day. Profile numbers are denominated in **τ, not wall-clock**. Size horizons for the workload, or the entity layer flushes within a day — the exact failure long-horizon agentic work hits.
- **For long-horizon projects**, prefer large `obj_forget` + `renew_ratio > 0` over wall-clock decay. Wall-clock penalizes *age*, not *importance*; a load-bearing entity from month 1 of a multi-month project gets flushed at month 6 regardless of how much it still matters. `renew_ratio` lets the agent's own consolidation summaries keep it alive while it matters, and drop it when it doesn't.

---

## Logical time τ

Events accrue **τ** from novelty, prediction error, and arousal. Quiet stretches barely move it; surprises spike it. Decay, reinforcement, and consolidation cadence all run on **Δτ**.

Short horizons: all profiles behave alike. Long horizons: the curves split — exactly what the E7 benchmark shows.

---

## Tensor-grounded objects — why recall is cheap

Each event gets scanned for entities. Those become **tensor-grounded objects** that live across episodes:

- first sighting creates them
- later sightings update them
- edges accumulate
- schemas emerge
- salience adjusts
- identity stabilizes

Recall often hits the object **directly** — no graph walk, no reranking loop, no multi-vector search.

**The tensor does the magic.**

Objects are SQLite rows with embeddings, characteristics, and update history — deterministic, auditable, stable for months or years.

---

## How fast is it?

Two honest numbers, two different layers:

- **The store** — `p95 0.7 ms` on the synthetic harness. SQLite + FTS5 + RRF → the "0.5–2 ms" you see in demos.
- **The full pipeline** — `p95 ~210 s` end-to-end on real LongMemEval `n=50` (ingest ~550 messages, embed, two LLM passes, judge). Recall `0.58` with memory vs `0.14` without → **Δ 0.44**.

Bonus: top-8 retrieval uses a flat **~242 tokens** vs the full haystack (which grows from **~2,249** at 4k to **~18,000** at 32k) — about **1/10th to 1/74th** of the context, with recall@1 1.0.

> The store is instant. The pipeline takes what the LLMs take. Both are real.

---

## 100% auditable

Everything — episodes, objects, τ, salience, edges, embeddings — is a plain SQLite table.

`SELECT * FROM episode` works.
`SELECT * FROM object` works.
`SELECT * FROM object_sighting` works.

Great for debugging, compliance, reproducibility, or simply trusting your agent's memory.

---

## Usage

```python
import sys
sys.path.insert(0, "/path/to/positronic-engram/engine/src")

from memeng.store import SQLiteStore
from memeng.engine import MemoryEngine
from memeng.models import Event

store = SQLiteStore("memory.db")
engine = MemoryEngine(store)
engine.init_database()
engine.register_domain("kairos", retention_profile="balanced")
engine.attach_stream("positronic:kairos", "kairos")

engine.new_event(Event(
    stream="positronic:kairos",
    kind="message",
    persons=["p_0001"],
    features={"subject_norm": "liqui-fire rx", "body_text": "...", "arousal": 0.7},
))

hits = engine.activate({"text": "liqui-fire"}, k=8)
```

Install (from the engine directory):

```bash
pip install -e engine   # or: pip install --break-system-packages -e engine
```

---

## Embedding tiers

| Tier | Method | When to use |
|---|---|---|
| `lexical` | FTS5 + RRF | always works, zero setup, sub-ms |
| `local` | BGE-M3 via `:8090` | semantic recall on your GPU |
| `remote` | API key | when you don't host embeddings |

`lexical` is a real baseline, not a toy — it's what makes the whole system run anywhere.

---

## Benchmarks

- **E7 synthetic** — 55 events → 78 weeks → `55/55/35/7` (`archival/long_term/balanced/short_term`), `profile_order_ok`, 14 unit tests green.
- **LongMemEval real `n=50`** — `archival/local` `0.58` vs `0.14` without → `Δ 0.44`, `fallback 0.0`, `recall 1.0` (clean single run).
- **RULER 4–32k** — top-8 retrieval flat `242 tok` vs growing full haystack (`2.2k–18k tok`) → `1/10`–`1/74` context, recall@1 1.0 preserved.
- **Throughput** — ~6k events/s synthetic batch ingestion (H14 gate economics: ~5% encode rate on realistic mix).

Run them yourself:

```bash
cd engine
pytest -q            # engine unit tests
python3 -m pytest ../consumers/benchmarks/tests/test_synthetic_e7.py -q
```

---

## Ecosystem

- **[positronic-opencode-plugin](https://github.com/ShingWong/positronic-opencode-plugin)** — the opencode plugin that uses this engine (`/positronic:*`, `positronic.*`, CLI).
- **[positronic-research](https://github.com/ShingWong?tab=repositories)** — the paper and benchmark harness: `papers/temporal-perception-in-AI/`, `consumers/benchmarks/`.
- **This engine** — `engine/src/memeng/` (`engine.py`, `store.py`, `models.py`, `triggers.py`, `fuzzy.py`, `vision.py`, `entities.py`, `telemetry.py`).

`ENGRAM_TAG=v0.2.0` pins the version used by benchmarks and the plugin.

---

## License

GPL-3.0-or-later — see `LICENSE`.