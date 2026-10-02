# Theory of Operation — the positronic memory substrate

How the brain decides what to keep, in the order it actually happens, with
every term defined and every constant traceable to a measurement.

Written for two readers: a human who has never seen this code, and an agent
who needs to change it without breaking a property someone spent a month
measuring.

---

## 0. The one-paragraph version

An event arrives. The brain computes **novelty** — how unlike anything it
remembers — and a **score** from novelty, arousal, and relationship gain.
The score is compared against a **cut**, and the cut is not a fixed number:
it is a *quantile of this brain's own recent novelty*. So the brain always
keeps roughly the same fraction of what it sees, whatever that fraction
means in absolute terms. A separate housekeeping pass later collapses
near-duplicates. Two clocks run underneath: **τ**, which advances only when
something happens or enough silence passes, and **wall time**, which does
not exist as far as decay is concerned.

Everything below is elaboration of that paragraph.

---

## 1. Vocabulary

Terms are defined once here and used consistently throughout.

| term | symbol | meaning |
|---|---|---|
| **event** | — | one input record: a message, a log line, an appointment. Immutable once ingested. |
| **episode** | — | an event that was **kept**. Events that fail the gate leave no episode. This is the central asymmetry: the corpus is all events, memory holds only episodes. |
| **stream** | — | an ordered channel of events, e.g. one mail account. Carries one clock. |
| **domain** | — | a policy scope. Owns a stream set and a retention profile. The unit of configuration. |
| **retention profile** | — | per-domain forgetting policy: how fast things fade, how long objects live. Four ship: `short_term`, `balanced`, `long_term`, `archival`. |
| **novelty** | ν | ∈ [0,1]. How unlike anything currently remembered this event is. 1.0 = nothing similar is remembered. |
| **arousal** | α | ∈ [0,1]. Environmental urgency supplied by the caller. The brain does not compute it. |
| **gain** | γ | ∈ [0,1]. How much this event's actor has mattered historically. Starts at the prior **0.5**. |
| **score** | — | the scalar the cut is compared against. |
| **cut** (threshold) | — | the bar. Above it, keep. |
| **bar** | — | the quantile of observed novelty that the cut is built from. Distinct from the cut, which is the bar expressed in score units. |
| **reservoir** | — | a trailing window of the last *W* novelty values, one per event seen. |
| **τ (tau)** | τ | the polytemporal clock. Advances with surprise and with silence. Decay is measured in τ, not seconds. |
| **strength** | S | an episode's decay constant. How much τ must pass before it fades. Larger = longer lived. |
| **retention** | R | fraction of an episode's strength still intact, ∈ (0,1]. |
| **object** | — | an entity the brain tracks across events (a person, a project). Has its own dormancy ladder. |
| **sighting** | — | one observation of an object. Repetition protects: ≥3 sightings earn 3× horizons. |
| **residue** | — | an expired episode's content, preserved after demotion. Nothing is deleted. |
| **dedup** | — | garbage collection: collapsing near-duplicate episodes into the original. |
| **id** | — | an actor's identifier (a sender, a service name, a visitor pseudonym). |

---

## 2. Pipeline

For each event, in this order:

```
  event
    │
    ├─► tokenize ─────────────► content terms
    │
    ├─► novelty  ν  ← IDF-weighted cosine vs the recency window
    │
    ├─► gain  γ  ← the maximum weight among the event's persons
    │
    ├─► score = 0.5·ν + 0.3·α + 0.2·γ
    │
    ├─► cut   = 0.5·bar + 0.2·γ          bar = max(q₀.₇(reservoir), 0.20)
    │
    ├─► encoded = score ≥ cut  OR  α ≥ 0.9
    │
    ├─► τ ← τ + 1.0·ν + 4.2857·(idle_days)
    │
    └─► if encoded: write episode
                    feed the recency window
                    append ν to the reservoir
```

Note the reservoir is appended for **every** event, kept or not. That
detail matters and is explained in §5.

---

## 3. Novelty

### 3.1 Tokenisation

An event's **content terms** are the words of `subject_norm`, `body_text`
and `text`, lowercased, split on non-word characters, keeping a token if

```
 len(token) > 1   OR   token.isdigit()
```

and it is not a stop word.

Two rules with measured reasons:

- **`len > 1` matches `activate`'s own tokenisation**, so the gate and the
  retriever agree on what a word is. Divergence here silently corrupts
  every measurement.
- **Single digits are kept, single letters are not.** Dropping digits made
  `"build 0 green"` and `"build 7 green"` the *same* term set, so an
  operational log collapsed to novelty 0.000 and the novelty window
  deadlocked holding nothing but its own homogeneous entries. Digits carry
  identity in logs; letters are noise.

### 3.2 The similarity measure

Novelty is IDF-weighted cosine against a **recency window** of recently
*stored* events, bounded at `semantic_novelty_window = 64` entries.

Document frequency is computed from the window itself:

```
 n_win  = number of entries in the recency window
 df(t)  = number of those entries containing term t
 w(t)   = ln( (n_win + 1) / (df(t) + 1) ) + 1        (smoothed IDF)

 q̂      = Σ_{t ∈ query} w(t)²           (query norm)
 ô      = Σ_{t ∈ other}  w(t)²           (other norm)
 sim    = Σ_{t ∈ query ∩ other} w(t)² / ( q̂ · ô )

 ν      = 1 − max over window entries of sim
```

**Why IDF and why cosine — both forced by measurement.** Plain Jaccard on
short text reads as similar by chance: on a low-variance stream with 6
content terms per event, mean best overlap was 0.462, so novelty sat near
0.54, score near 0.37, and the gate stored **0.1%** of a perfectly
legitimate stream. Worse, that overlap was mostly *common* words — `alpha`
appeared 75 times in 300 events, so agreeing on `alpha` was weak evidence of
anything. Weighting by rarity fixes both at once: a shared rare term counts,
a shared common one barely registers, and the estimate stops depending on
set size.

**Why IDF is computed from the window rather than kept as state:** the
window *is* the comparison set, so its own frequencies are exactly the
right ones. There is nothing to expire and nothing to resynchronise.

### 3.3 The empty-window case

If there are no terms, or the window is empty, novelty returns
`semantic_novelty_fallback = 0.9`. In practice this fires only on the first
event of a brain's life; after that the window is never empty.

---

## 4. The gate

### 4.1 Score

```
 score = min(1.0,  0.5·ν  +  0.3·α  +  0.2·γ )
```

Weights sum to 1.0, so the score is a convex combination and lands in [0,1].

### 4.2 The cut, and why it is derived

This is the heart of the design. The cut is **not** read from config. Once
the reservoir holds at least `novelty_warmup = 1500` values:

```
 bar = max( q₀.₇(reservoir),  novelty_bar_floor )          floor = 0.20
 cut = 0.5·bar + 0.2·γ
```

`q₀.₇` is the 70th percentile of the last `novelty_reservoir = 5000`
observed novelty values.

**Why a quantile and not a constant.** A fixed cut is reachable only by a
brain whose novelty happens to clear it, and that is not a property you can
design for. Measured: the git corpus has a novelty median of **0.810**, so
a score cut of 0.55 required novelty ≥ 1.32 against a maximum of 1.0 — *no
reachable value existed*. A quantile is reachable by construction, because
it **is** the value that 70% of the stream falls below.

**The consequence, which is the actual claim.** The store rate equals
1 − q₀.₇ almost exactly, on corpora that share nothing:

| q₀.₇ | predicted | mx1 | email | git | pvl |
|---|---|---|---|---|---|
| 0.60 | 40.0% | 39.9% | 37.8% | 36.3% | 39.9% |
| 0.70 | 30.0% | 29.4% | 27.3% | 27.1% | 30.1% |
| 0.80 | 20.0% | 20.3% | 20.4% | 18.0% | 18.3% |

Novelty medians across those four: 0.114, 0.613, 0.807, 0.495. The gate
tracks its prediction within 3 points across that range. One operator-facing
number controls the outcome.

### 4.3 The floor, and why 0.20

`max(bar, 0.20)` exists because **a quantile of a degenerate distribution
*is* the degeneracy.**

Measured on the mx1 `dnsblog` stream — 12 distinct strings in 66,260 lines —
novelty is **98.3% exactly 0.000**. The 70th percentile of a distribution
that is almost entirely zeros is zero. Then:

```
 cut = 0.5·0.000 + 0.2·0.5 = 0.1000
 score = 0.5·ν + 0.2·0.5   ≥ 0.1000   for every ν ≥ 0
```

The cut reaches the floor `0.2·γ` and **every event clears it regardless of
novelty**. Store rate by decile of a 120,000-line run:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stored | 0.1% | 0.2% | 8.1% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

Noise filtering that **decays**: the longer the brain runs, the more
boilerplate it keeps.

**Why 0.20 and not something tuned.** Every corpus's own 70th-percentile bar:

| mx1 | email | git | pvl | dnsblog |
|---|---|---|---|---|
| 0.348 | 0.760 | 0.880 | 0.544 | 0.000 |

Any floor **below 0.348 cannot bind on a varied corpus**. A sweep confirmed
it end to end: every floor in (0, 0.348) takes dnsblog from 100% to 0.0%
while leaving all four varied corpora byte-identical. 0.20 sits mid-range
with ~39% headroom, and the cliff is *predicted* by the lowest observed bar
rather than found by search.

| floor | dnsblog | mx1 | email | git | pvl | spread |
|---|---|---|---|---|---|---|
| 0.00 | 100.0% | 32.2% | 32.4% | 33.4% | 29.7% | 1.12× |
| 0.20 | 0.0% | 32.2% | 32.4% | 33.4% | 29.7% | 1.12× |
| 0.34 | 0.0% | 34.0% | 32.4% | 33.4% | 29.7% | 1.14× |
| 0.50 | 0.0% | 16.1% | 32.4% | 33.4% | 29.7% | 2.07× |

### 4.4 Warmup

Below 1500 events the reservoir cannot support a quantile, so `threshold
= 0.55` applies.

**The cold-start problem this creates.** A fixed cut is reachable only by a
brain whose novelty clears it. A low-novelty corpus stores almost nothing,
so the recency window stays nearly empty, so novelty stays high, so nothing
clears the cut. Traced on mx1:

| events | 100 | 500 | 1000 | **1500** | 2000 | 3000 |
|---|---|---|---|---|---|---|
| stored | 8.0% | 2.2% | 1.2% | **0.9%** | 2.5% | 2.1% |
| recency window | 8 | 11 | 12 | **13** | 50 | 62 |
| cut | 0.55 | 0.55 | 0.55 | **0.55** | 0.476 | 0.443 |

**13 episodes in the first 1500 events.** The brain is blind for its entire
warmup. The opening rate tracks the fallback cut directly — cut 0.55 → 2.2%,
0.40 → 13.4%, 0.30 → 28.2% — so no fixed value serves a corpus whose novelty
sits low.

**This is deliberately not fixed at the gate**, because every warmup
statistic that opens the brain also floods it — see §7. It is fixed
downstream instead, by garbage collection.

### 4.5 Gain cancels, and that is not a bug

`0.2·γ` appears on **both** sides of `score ≥ cut`. It therefore subtracts
out exactly. Measured on one event with the weight forced:

| γ | score | cut | margin |
|---|---|---|---|
| 0.00 | 0.5000 | 0.4696 | +0.0304 |
| 0.50 | 0.6000 | 0.5696 | +0.0304 |
| 1.00 | 0.7000 | 0.6696 | +0.0304 |

The margin is invariant. So `w_gain` **cannot decide any single event**.

It is not inert over a stream. `_remember_terms` feeds the novelty recency
window only for *stored* events, so the store decision changes which events
enter the window, which changes every later novelty value. Forcing γ 0→1
over 8,000 git commits:

| forced γ | episodes | mean novelty |
|---|---|---|
| 0.0 | 1,388 | 0.8136 |
| 0.5 | 2,323 | 0.7814 |
| 1.0 | 3,352 | 0.7739 |

**2.4× more episodes, and novelty falls** because storing more fills the
comparison window. `w_gain` is a *selection* channel, not a score channel.
It also touches only the first 3 of 20 twentieths of a run — from t4 on,
`w_gain` of 0.0, 0.2, 0.4 and 0.8 give byte-identical results — so it sets
how fast the brain *fills*, not where it *settles*.

### 4.6 Burst override

`α ≥ burst_threshold = 0.9` stores regardless of score. Environmental
urgency bypasses novelty entirely. Never exercised in the corpora measured
(they all carry α = 0).

---

## 5. Why the reservoir holds every event

The reservoir is appended for **every** event, kept or not. This is the
difference between this design and an earlier recalibration scheme that
collapsed to ~12% store rate and died.

Fitting the bar to **stored** episodes is a **censored sample**: the store is
selected by the very quantity being fitted, so the fitted distribution is
distorted toward what the gate already accepted. The reservoir holds the
whole population.

Cost: 16 bytes per event, ~21–25% of the text it describes.

---

## 6. Housekeeping: `prune()`

`prune()` is one sweep doing five jobs. It is **never called by the engine
itself** — the caller schedules it (see §9).

### 6.1 Gain decay

```
 weight ← 0.5 + (weight − 0.5)·(1 − gain_decay_rate)      rate = 0.01
```

Regression toward the prior. Runs here so it needs no timer of its own.

### 6.2 Deduplication

Collapses near-copies into the episode that came first.

Two structures, because the two duplicate kinds cost differently:

**Exact duplicates** — the normalised term set is hashed, O(n) and
independent of distance. This is the bulk of the work: **94.1%** of mx1's
duplication is exact.

**Near duplicates** — compared pairwise, bounded by `dedup_window = 200`
recent episodes. A copy arriving more than 200 episodes later escapes.
That is a real, unquantified gap.

Two events are redundant only when the same words describe **the same
actor, close enough in time to be the same occurrence**:

```
 key = ( content_terms , sorted(persons) )

 merge iff   key already seen
        AND  same actors
        AND  wall gap ≤ dedup_span_seconds
        OR    J(terms, other terms) ≥ dedup_sim  within the recency window
```

`dedup_span_seconds = 86400` (one day). All three conditions were added
because measurement demanded them, and each was a real fact-dropping bug
before it was fixed:

- **actor** — content alone destroyed 471 pvl episodes.
- **span** — content + actor still dropped facts. On git, 86 episodes merged
  where the wording recurred across up to **3,972 days**, and one group
  collapsed **31 real commits** into one, including `"initial revision"`
  appearing in 1990 and again in 1995.

The asymmetry this encodes is deliberate: **a few extra duplicates is the
cheap mistake; a dropped fact is not.** Where the two conflict, keep both.

The day-long span is chosen from the observed gap distribution — mx1's
duplicates all fall within one hour, git's recur across years — so the
default sits inside the gap rather than at a guess.

`dedup_sim = 0.6`, from the retention profile so `archival` can set it to
`None` and disable collection entirely.

**Why actor is part of the key — measured.** On pvl, **100%** of same-text
episode groups were a *different person at a different time*; the largest was
6 appointments sharing text across 6 distinct visitors. Content-only dedup
merged **471** of them, destroying the record. Keying on content **and**
actor merges **0**, while mx1 keeps its reclaim because its lines genuinely
are one service emitting identical boilerplate with no distinct actor.

| corpus | episodes kept | merged | net | reclaimed |
|---|---|---|---|---|
| mx1 | 4,339 | 4,088 | 251 | 94.2% |
| git | 6,161 | 94 | 6,067 | 1.5% |
| pvl | 4,159 | 0 | 4,159 | 0.0% |

mx1's reclaim went **up** after the span guard (84.6% → 94.2%), because
within-hour duplicates still collapse freely. git barely moved, because its
merges were precisely the year-spanning occurrences we now correctly keep.

**Merge semantics reuse `prune_merge`'s vocabulary:** demote to `day_token`
and clear `subject_norm`. A copy then occupies no search slot and **nothing
is deleted**.

**Ordering.** Which of a duplicate pair survives is decided by iteration
order, so `iter_episodes(ordered=True)` sorts by τ. The lowest τ survives —
the brain's own experiential clock, so a backdated event is later to the
brain even when earlier on the calendar. Table order is insertion order,
which agrees with τ only for strictly increasing ingest.

### 6.3 Episode ladder

```
 R = exp( −(τ_now − τ_episode) / S )
```

with S from the retention profile. Then:

| condition | action |
|---|---|
| `R < prune_expire` | write a residue, demote to `week_token` |
| `R < prune_merge` | demote to `day_token` |

| profile | S_base | prune_merge | prune_expire |
|---|---|---|---|
| short_term | 6 | 0.20 | 0.02 |
| balanced | 30 | 0.35 | 0.05 |
| long_term | 120 | 0.20 | 0.02 |
| archival | 10⁶ | None | None |

### 6.4 Object ladder

Objects (people, projects) have their own dormancy:

```
 d = τ_now − last_seen_τ
 if d ≥ obj_forget · mult : status → forgotten   (unless load-bearing → renew TTL)
 elif d ≥ obj_dormant · mult : status → dormant
 mult = 3 if sightings ≥ 3 else 1
```

Repetition protects: three sightings earn three times the horizon.
Load-bearing objects — ones still named in recent consolidations — get a
TTL extension instead of being forgotten. Retention is **re-earned each
cycle, never permanent**.

### 6.5 Profile table, complete

| | short_term | balanced | long_term | archival |
|---|---|---|---|---|
| S_base | 6 | 30 | 120 | 10⁶ |
| S_arousal | 4 | 40 | 40 | 0 |
| prune_merge | 0.20 | 0.35 | 0.20 | None |
| prune_expire | 0.02 | 0.05 | 0.02 | None |
| dedup_sim | 0.6 | 0.6 | 0.6 | None |
| obj_dormant | 25 | 200 | 600 | None |
| obj_forget | 150 | 1200 | 5000 | None |
| renew_max | 150 | 12000 | 50000 | None |
| renew_ratio | 0.0 | 0.3 | 0.3 | None |

---

## 7. The cold-start deadlock, and why the gate is left alone

The obvious fix is a self-calibrating warmup — use the median of novelty seen
so far, which is reachable by construction from five observations. It works:
mx1 opens 2.1% → 29.8%, pvl 6.4% → 37.9%.

It is not shipped, because on repetitive input the cold start and duplicate
suppression are **the same knob**:

| warmup statistic | dup stored | mx1 opens |
|---|---|---|
| fixed cut (shipped) | 0.2% | 2.1% |
| median | **99.7%** | 4.6% |
| q₀.₉₀ | 99.8% | 4.6% |
| max novelty | 0.2% | 2.1% ← does not open at all |

Every statistic that opens the brain also floods it, because on a
repetitive stream admitting *any* extra events means admitting copies.

**Garbage collection dissolves it instead of trading along it.** The gate
admits freely during warmup; the housekeeping pass removes the copies
later:

| config | pre-GC episodes | net |
|---|---|---|
| shipped, fixed cut | 4,339 | 4,339 |
| median warmup, no GC | 5,880 | 5,880 |
| **median warmup + GC 0.6** | 5,880 | **2** |

The brain opens (2.1% → 29.7%) *and* ends clean.

Two beliefs of mine were wrong here, both caught by measurement: that the
bar floor made the median safe (it bounds the *bar*; near-duplicate novelty
is 0.272, *above* the 0.20 floor, so it never binds), and that a median is
a safe warmup statistic (a median is the value half the stream exceeds, so
it admits half the stream by construction).

---

## 8. The two clocks

### 8.1 τ (tau), the polytemporal clock

```
 τ ← τ + 1.0·ν + 4.2857·(idle_seconds / 86400)
```

Accrued per event, so each inter-event gap is credited exactly once and the
total idle credited over any window is `rate × span` regardless of how the
events were distributed. That is what makes it a clock component rather than
a function of the final gap.

**Why idle must be credited at all.** Before this, τ advanced only on
surprise. Measured consequence: a 100× difference in event rate produced an
**85×** difference in τ — a tally of events, not elapsed experience. A quiet
agent's clock froze and τ-keyed decay never fired for it.

**Why the rate is 4.2857 and not larger.** It is derived, not tuned: with
balanced `S_base = 30`, a rate of `S_base/7 = 4.2857` makes **one week of
silence deliver exactly one S_base**. Measured on the 5.19-year visitor-log
corpus: idle is ~5% of total accrual, and the largest gap in 5.19 years is
6.92 days = 29.64 τ against S_base = 30. Engaged time dominates; silence
still registers.

### 8.2 Wall time

Wall time exists for *when* things happened, never for *how long they
last*. Decay is denominated in τ. Sizing guidance: τ burn under agentic
load is ~0.9 τ/event when predictions are empty, so a busy session burns
~13k τ/day — size horizons for the workload, or the object layer flushes in
a day.

---

## 9. Who calls `prune()`

The engine never calls it. The agent interface does, on an ingest counter:

```
 auto.prune_every = N      →  every N ingests, run prune()
 auto.consolidate_every = N →  every N ingests, run consolidate()
```

Both default to **0 (off)**. To enable:

```
positronic config set auto.prune_every 100
```

Verified end to end on a live brain:

```
{"scanned": 731, "day_merged": 111, "week_merged": 557,
 "expired": 557, "residues": 557, "deduped": 60, "objects_dormant": 1458}
```

---

## 10. Every knob, in one table

| knob | default | what it does | why that value |
|---|---|---|---|
| `novelty_quantile` | 0.70 | the store-rate target; rate ≈ 1 − q | **the only steady-state parameter.** Tracks prediction within 3 points on 4 corpora |
| `novelty_reservoir` | 5000 | trailing window length, 16 B/event | W ≥ 12000 lets the reservoir span a whole run; mx1 collapses to 16.5% |
| `novelty_bar_floor` | 0.20 | minimum value for the derived bar | free below 0.348 (lowest varied bar); stops a point mass capturing the gate |
| `novelty_warmup` | 1500 | events before the quantile is trusted | below this a quantile is not supportable |
| `dedup_window` | 200 | near-duplicate comparison horizon | **known gap**: a copy arriving later than this escapes |
| `dedup_span_seconds` | 86400 | beyond this gap, same words + same actor is a NEW event | sits inside the observed gap distribution: mx1 duplicates are all within 1h, git's recur across years |
| `threshold` | 0.55 | the warmup fallback cut | not reachable for low-novelty corpora; the cold start, §4.4 |
| `w_novelty` | 0.5 | novelty's share of score | |
| `w_arousal` | 0.3 | arousal's share | never exercised — all corpora carry α = 0 |
| `w_gain` | 0.2 | gain's share | cancels per event; acts by selection. Unmeasurable by store rate |
| `tau_per_surprise` | 1.0 | τ gained per unit novelty | |
| `tau_idle_per_day` | 4.2857 | τ gained per silent day | = S_base/7; one week of silence = one S_base |
| `semantic_novelty_window` | 64 | recency window for novelty | |
| `semantic_novelty_fallback` | 0.9 | novelty when the window is empty | unreachable after the first event |
| `honor_credit` | 0.01 | gain credited per retrieval hit, ÷ rank | 1.0/rank saturated weight 0.5→1.0 on one recall |
| `honor_window` | 8 | retrievals that earn credit | |
| `gain_decay_rate` | 0.01 | regression toward 0.5 per prune | |
| `burst_threshold` | 0.9 | arousal that overrides the gate | never exercised |
| `anchor_salience` | 0.85 | salience at which an episode anchors | |

---

## 11. Known limits, stated plainly

1. **Repetitive input and a point mass.** Fixed by the floor for the
   observed cases. The general form — a quantile landing on *any* mass,
   including a ceiling — is unresolved. Measured on email at q₀.₉₀, where
   the bar reaches 1.000 and the rate floors at 19.2% against a predicted 10%.

2. **Novelty depends on unrelated memory.** With a *fixed* event pair,
   adding window entries that share nothing with it moves novelty 0.2722 →
   0.1920 over 37 entries — a 41% change from memory that has nothing to do
   with the event. Root cause: document frequency is computed over a window
   that contains the event's own near-duplicates. Latent at run scale on all
   four corpora (no bar rises), but real. Five candidate fixes measured and
   rejected; recorded in `engine/KNOWN_DEFECTS.md`.

3. **The dedup window is a hard horizon.** 200 episodes. Copies beyond it
   survive. Unmeasured how much this costs on high-recurrence streams like
   dnsblog.

4. **Gain is inert without retrieval.** 0 of 58,787 identities moved off the
   prior in pure ingest. H18 learns from *retrieval*; a workload that never
   queries its own memory runs with the gain channel dead.

5. **The cold start costs 1500 events.** Real, measured, and traded against
   duplicate suppression rather than solved.

6. **`w_arousal` and `burst_threshold` are untested.** Every corpus measured
   carries α = 0.

---

## 12. For agents changing this

**Properties that took a month to establish. Breaking one costs real work:**

- `novelty_quantile` is the only steady-state parameter. Don't add a second.
- The floor must stay **below the lowest q-bar of any corpus you intend to
  serve**. It is currently 0.20 against a floor of 0.348. Raising it past a
  corpus's bar silently overrides that corpus's own calibration.
- The reservoir must hold **every** event, kept or not. Fitting to stored
  episodes is censored sampling.
- The reservoir must evict by **age**, not by value. A sorted list carries
  order but not age; evicting its head drops the *least* novel instead of
  the *oldest*, and the bias is invisible on short streams.
- `w_gain·γ` must appear on **both** sides of the comparison, or gain
  becomes a score channel with no measured calibration.
- Dedup keys on content, actor **and** a one-day proximity window. Content
  alone destroys records of different people doing the same thing;
  content+actor destroys records of the same words recurring years apart.
  Where dropping a fact and keeping a duplicate are in tension, keep both.
- Tokenisation must match `activate`'s, digits included.

**Testing discipline that worked here:**

- Every claim in this document is a number from a run, not an argument.
- Every test is verified **non-vacuous**: revert the fix and watch it fail.
  Several tests here needed rebuilding because they *could not* fail —
  identical text is scored low-novelty so the gate admits almost none of it,
  and sequential seeding makes insertion order and τ order identical.
- When a measurement contradicts an expectation, the expectation was wrong
  twice as often as the measurement. Both the cold-start and the
  duplicate-suppression results were my errors, not the code's.
- Distrust any result where several cells are byte-identical. That has
  appeared four times this project and was a bug every time — a wrong
  method name, an unreachable code path, a censored sample, a threshold read
  from the wrong config.

---

## 13. Where the evidence lives

| artefact | contents |
|---|---|
| `engine/KNOWN_DEFECTS.md` | measured, unfixed defects with rejected fixes |
| `engine/RULE_CHANNEL.md` | rule-channel findings and a retraction |
| `positronic_ai/docs/USER-GUIDE.md` | user-facing guide |
| `/tmp/opencode/*.md` | the investigation notebooks behind every number here |