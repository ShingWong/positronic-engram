# API Specification v0

All functions live on `MemoryEngine(store, config)`. Store is a protocol;
SQLite implementation ships with v0.

---

## `init_database(path: str | None = None) -> None`

Create schema (episodes, rules, persons, stream_tau, lookups, meta) if absent.
Idempotent; safe to call on every boot. Seeds lookup tables. Sets schema
version in `meta`.

## `new_event(event: Event, *, store: bool = True, handlers: Handlers | None = None) -> EventResult`

The cognitive loop. Full pipeline per DESIGN.md flow chart.

**Event** (input):
- `stream: str` — sensor body id (`"mail:p1"`, `"bot:s10"`, …)
- `kind: str` — `"message"` | `"snapshot"` | … (lookup)
- `wall: datetime` — event's wall-clock time
- `persons: list[str]` — pseudonymous participant ids
- `features: dict[str, str | float]` — extracted features (subject_norm,
  thread_id, body_excerpt, arousal_proxies…)
- `fuzz_width: timedelta | None` — encoding-confidence window; None ⇒ exact

**Returns EventResult**:
- `verdict: GateVerdict` — `{score, novelty, arousal, gain, threshold, encoded}`
- `tau: float` — the event's subjective-time coordinate after processing
- `episode_id: uuid | None` — set iff gated and `store=True`
- `predictions: list[Prediction]` — what schemas expected, and how observed
  matched (per-feature match/violate)
- `trace: list[Trigger]` — every trigger fired, in order, regardless of handlers

**Semantics:**
1. PREDICT against active rules + cadence stats for `(stream, persons)`.
2. Score gate. `arousal ≥ burst_threshold` forces encode + emits
   `escalation_burst`; else threshold compare.
3. If gated and `store=True`: insert episode (level=`event`, fuzz from width,
   tau from stream accumulator advance), detect anchor (salience ≥ anchor_t),
   bind persons, emit `episode_encoded`.
4. If not gated (or `store=False`): apply reinforcement deltas to schema stats
   and rule support counts; emit `schema_reinforced`.
5. Rule confirmations/violations from step-1 predictions fire in both paths
   (H17: learning below the gate).
6. Advance `stream_tau` by surprise-weighted increment.
7. Triggers emitted inline at each step; handlers called synchronously.

**`store=False` guarantee:** no row in `episodes` is created or modified;
schema/rule/τ state still advances. This makes counterfactual replay exact.

## `observe(event, *, handlers=None) -> EventResult`

Convenience alias for `new_event(event, store=False, handlers=handlers)`.

## `prune(tau_now: float | None = None) -> PruneReport`

H15 ladder pass over all streams (or one):
- episodes whose `strength × exp(−Δτ/S)` < merge_floor → merged into their
  day-token (create if absent); day-tokens older than week horizon →
  week-token references
- flashbulb tier skipped entirely
- tombstone residue written: usage_count, person set, anchor edges survive
**PruneReport**: `{day_merged, week_merged, expired, residues, scanned}`.
Deterministic given store state; safe to run repeatedly.

## `activate(cue: Cue, k: int = 8) -> list[Activation]`

*(v0 minimal)* rank episodes by decayed activation = base-level(τ distance,
strength) + embedding similarity (when store supports it) + person boost.
Full spreading-activation arrives with constellation walks.

## Trigger delivery

```python
result = engine.new_event(ev, handlers={
    "gate_decision":      lambda t: log(t.features),
    "escalation_burst":   lambda t: widen_capture(t.until_tau),
})
```
Handlers raise → exception propagates (fail loud in research mode).
`result.trace` always complete.

## Unit-test contract (see tests/)

I1 init idempotent · E1 first-ever event encodes · E2 identical repeat only
reinforces · G1 person_gain flips verdict (H18) · O1 observe never writes
episodes but advances schema+rules+τ · T1 trigger trace completeness/order ·
R1 below-gate rule confirmation (H17) · R2 violation raises score & encodes ·
F1 fuzz widths persisted per precision · A1 anchor detection at threshold ·
P1 prune merges aged → day-token, spares flashbulb, leaves residues ·
T2 τ skips uneventful spans (H14)
