# Object salience: subjective-time reinforcement

**Status:** design in progress, 2026-09-30. **The constants below are not calibrated.**
See "What the corpus measured" — two of them are known to be wrong, and one of the
three premises the design rested on turned out to be false on real data.

## The problem

`object.salience` is written once at insert (0.5) and never updated. `touch_object`
updates only `last_seen_tau` / `last_seen_wall`. The object layer therefore has **no
reinforcement at all** — salience is a constant, and retention depends solely on a
binary `mult = 3.0 if sightings >= 3 else 1.0`.

Defects, in order of how much they matter:

1. **TTL is a step function.** `sightings >= 3` gives a flat 3× on both `obj_dormant`
   and `obj_forget`. "Repetition protects" is a cliff.
2. **No promotion for a return after absence.** There is no mechanism at all, so an
   entity that comes back after a long gap is indistinguishable from one never seen.
3. **The ladder is one-way.** `forming → dormant → forgotten`, never back.
   `touch_object` never writes `status`, and prune filters on
   `WHERE o.status != 'forgotten'`, so a forgotten entity stays forgotten however often
   it is mentioned. `status='stable'` is read by two guards and written by nothing.
4. **TTL renewal is inert.** `_renew_load_bearing` writes an interval extension to
   `last_renewal_tau`, but the survival test reads `dmeas = now − last_seen_tau`, so the
   extension never reaches a decision. It defers forgetting by exactly one prune cycle.

## The brief

- An object reappearing **frequently with no change in salience** must **not** be promoted
  to high salience through additive accumulation.
- An object reappearing **after a long absence should** be promoted.

## Design

Two channels, because persistence and informativeness are different signals and
conflating them fails both ways.

```
w     = clip( max(0, novelty − θ) + γ·(1 − e^{−Δτ/S_gap}),  0, 1 )   surprise weight
P    ← min(p_max, P + β·(1 − P/p_max))                                persistence
I    ← min(i_max, I + α·w·(1 − I/i_max))                              informativeness
S     = min(1, P_eff + I_eff)                                        at read time
P_eff = P · e^{−(τ_now − last_seen_tau)/S_pres}
I_eff = I · e^{−(τ_now − last_seen_tau)/S_info}

ttl_scale      = 1 + (obj_ttl_max − 1)·S_eff
obj_dormant_eff = obj_dormant · ttl_scale
obj_forget_eff  = obj_forget  · ttl_scale
```

- **Volume is capped, not summed.** `P` saturates at `p_max`; the whole reward for
  recurring is that ceiling. `I` has a novelty deadzone, so an unremarkable mention adds
  nothing.
- **A lapse earns a return** through the `γ(1 − e^{−Δτ/S_gap})` term, which describes the
  *reappearance event* as unexpected. `w` is clipped, so the bonus saturates.
- **Nothing ratchets.** Both channels decay with subjective absence and must be
  re-earned — the direct counter to the one-way ladder.
- **High salience extends a TTL, never bypasses one.** Otherwise the layer rebuilds the
  flashbulb pathology, where a saturated entity becomes immortal.

Decay is applied at **read** time only, from `(now − last_seen_tau)`. A daily mention has
a small gap so it effectively never decays; an absent entity decays in one step. Prune
stays read-only, so repeated prunes cannot double-apply.

## Why v1 and v2 were rejected

Both were caught by running the arithmetic, not by reading it.

**v1** put a single saturating term with a `(Δτ/S_ref)` normalizer and a deadzone.
Three failures: an unremarkable-but-frequent entity *decayed toward zero* (the increment
died but the decay term still fired, so volume became a penalty); the unbounded
normalizer let a long lapse slam `S` to 1.0 outright; and the rate-independence check
varied total elapsed time, so it proved nothing.

**v2** capped the normalizer but left it on the increment. Holding total surprise equal,
4 large surprises then scored **higher** than 400 small ones, because the normalizer
favoured sparse delivery — backwards, since 400 near-violations carry more information
than 4. The lapse belongs on the event's surprise weight, not on the accumulator.

## What the corpus measured

Measured on the local email corpus (38,245 messages, 2007-08-09 → 2026-08-25), read from
a local file. Aggregate counts only; no sender, subject, or body is recorded. Reproduce
via the scratch harness, not from this repo — the corpus is private.

**`τ` is very nearly an event counter.** Gate novelty over the full corpus:

| novelty | messages | share |
|---|---|---|
| 1.0 | 6,376 | 16.7% |
| 0.9 | 31,869 | 83.3% |
| 0.6 | 0 | 0% |
| 0.05 | 0 | 0% |

```
tau/message = 0.917   and   0.8333×0.9 + 0.1667×1.0 = 0.9167
```

No message was ever judged routine, so nothing is ever gated out (encode rate 1.000) and
`τ ≈ 0.917 × (messages)`. **The prediction channel keys on `subject_norm`, and email
subjects are near-unique** — 23,689 distinct subjects across 38,245 messages — so 83% of
messages match no antecedent and novelty falls through to its "no prior prediction"
default. 2,056 rules formed, but the antecedent rarely fires.

**`τ` per wall-day is not a constant.** The same corpus gives 61.2 τ/day over a
4,000-message window and 5.0 τ/day over all 38,245. Any constant expressed in τ therefore
depends on the window, which is the `engine.py:139` warning ("a busy session can burn 13k
τ/day") arriving as a live calibration failure.

**Object population is mostly singletons.** 228 objects from a 4,000-message window;
61.4% have exactly one sighting, and only 18.4% clear the old `sightings >= 3` cliff.

## Consequences — the design above is not yet safe to implement

1. **The deadzone is inert on real data.** With novelty ∈ {0.9, 1.0}, every message gets
   `w ≈ 0.4–0.5`, so `I` saturates almost immediately and discriminates nothing. An
   absolute surprise weight cannot work while the novelty estimator does not discriminate.
   The likely fix is a **per-object relative** weight — surprise relative to the entity's
   own inter-arrival distribution (median / MAD) rather than to a global threshold — which
   is self-calibrating and removes the global gap constant entirely.
2. **Every τ constant here is suspect** until the τ rate has a stable definition. If τ
   stays proportional to message count, "500 τ" is really "≈556 events", and expressing
   salience in τ is a presentational choice rather than a temporal one.
3. **Fixing the rule channel may be the prerequisite, not a parallel task.** Repointing
   rule antecedents at features that actually recur in real traffic — sender, thread,
   participant set — would let novelty acquire a 0.05 class, which the gate currently
   reaches only on synthetic streams. Salience built on a degenerate novelty signal
   inherits the degeneracy.

## Open decisions

- **Does the ladder revive?** Reviving forgotten entities means reconsidering them each
  prune, which needs a bound — only entities sighted *after* the forget decision,
  indexable on `object_sighting` — or the scan grows without limit.
- **Should TTL renewal become real?** Measure against `max(last_seen, last_renewal)`.
  Small, but it interacts with revival.
- **Is TTL the right target for salience, or `strength`?** Graded TTL is the smaller
  change; `strength` is more consistent with the episode layer.

## Constants (all UNCALIBRATED)

| knob | meaning | placeholder |
|---|---|---|
| `obj_sal_persist_rate` | β, persistence saturation rate | 0.15 |
| `obj_sal_persist_max` | p_max, ceiling for pure volume | 0.50 |
| `obj_sal_gain` | α, informativeness rate | 0.50 |
| `obj_sal_info_max` | i_max, ceiling for surprise | 1.00 |
| `obj_sal_deadzone` | θ, novelty below which a mention does nothing | 0.50 |
| `obj_sal_lapse_gain` | γ, weight on returning after absence | 0.30 |
| `obj_sal_gap` | S_gap, τ scale of a "lapse" | 500.0 ⚠ |
| `obj_sal_pres_decay` | S_pres, τ decay of persistence | 5000.0 ⚠ |
| `obj_sal_info_decay` | S_info, τ decay of informativeness | 500.0 ⚠ |
| `obj_ttl_max` | TTL multiplier at S = 1 | 3.0 |

⚠ = these are order-of-magnitude guesses invalidated by the measurements above. The three
τ-denominated values must be re-derived once the τ rate is settled; `obj_sal_gap` in
particular should probably disappear in favour of the per-object relative weight.
