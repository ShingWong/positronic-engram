# The rule channel and the gate

**Status:** measured 2026-09-30 against the local email corpus (38,245 messages,
2007-08-09 → 2026-08-25, read from a local file). Every number below was produced
by running the engine or the real gate, not by reading the code. Aggregate
counts only — no sender, address, subject, or body appears here or in any output.
Nothing here is implemented yet.

Companion to `OBJECT_SALIENCE.md`, which depends on conclusions from this file.

## One correction to a previously reported number

An earlier measurement reported "6,376 rule firings, 0 confirmations," and that
was read as evidence that the engine's rules can never confirm. **That number
was an artifact of the measurement harness**, which built events with the sender
in `persons` only. The inducer accepts `sender` from `features` *or* falls back
to `persons[0]` (`engine.py:379`); the evaluator reads `features["sender"]` only
(`engine.py:246`) and sets `p.actual = None` when the key is absent, which can
never equal a string `expected`. Every firing through that path is a guaranteed
violation.

With the sender placed where the evaluator reads it, confirmations appear:

| | novelty 0.05 | 0.9 | 1.0 | confirmations |
|---|---|---|---|---|
| sender in `persons` (old harness) | 0% | 74.2% | 25.8% | 0 of 9,861 |
| sender in `features` | **15.4%** | 74.2% | 10.4% | **5,897 of 9,861** |

So `novelty = 0.05` is reachable, and the rule channel confirms when fed events
in a shape it can read. τ/message on the full corpus is **0.779**, not the 0.917
previously reported. The encoder now gates some events out (encode rate 0.846,
against 1.000 before).

The `persons`/`features` asymmetry is still a real defect — a rule induced
through the fallback path is structurally incapable of confirmation — but it is
a defect in the *engine's* two halves disagreeing, not in the rule idea.

## Finding 1: rule confidence is computed and never gates anything

`grade_rule` maintains `causal_rule.confidence`, but the only reader is the
telemetry payload at `engine.py:412`. `active_rules` (`store.py:511`) selects
every rule in the domain — no threshold, no age limit, no retirement.

Measured: a rule violated 200 times runs to `confidence = 0.000`,
`support_count = 0`, and still fires on every subsequent event contributing
`novelty = 1.0`.

This is the fourth inert mechanism, and the most damaging, because it feeds
τ inflation directly: **a rule that has been disproven produces maximal surprise
forever.**

## Finding 2: the rule antecedent is hardcoded to mail

`engine.py:378-396` reads `subject_norm` as the antecedent and `sender` as the
outcome. There is no config knob; `induce_after` and `induce_top_assoc_min`
change how much repetition is required, not what is predicted. `memeng` is a
generic engine whose one predictive channel is welded to mail features.

## Finding 3: half of real mail cannot be keyed on subject at all

Reply and forward prefixes (`Re:`, `FW:`, and variants) appear on **51.4%** of
messages. Normalising them (strip prefixes, lowercase, collapse whitespace)
collapses antecedent cardinality 24,898 → 19,000 and lifts rule coverage:

| antecedent | distinct | msgs on a rule-eligible subject |
|---|---|---|
| raw subject | 24,898 | 12,736 (33.5%) |
| normalised subject | 19,000 | **18,743 (49.6%)** |

Even normalised, 13,130 of 19,000 subjects still occur exactly once. **Half the
corpus is unreachable by any subject-keyed rule.** This is a property of email,
not a bug.

## Finding 4: coverage is the binding constraint, not precision

Grading novelty by rule confidence — `matched → 1−c`, `violated → c`,
`unpredicted → 1.0` — does produce a real gradient: **1,381 distinct values**
(p10 0.095, p50 0.333, p90 0.750) over the subset where it applies, against the
ladder's two values there.

But it only applies to **16.3% of all traffic**, because 74.2% of events fire no
rule at all and a further 9.5% fire one with too little evidence to trust.
Raising coverage does not rescue it either — see Finding 5.

## Finding 5: τ ≈ 0.8 × messages is arithmetic, not a defect

Pointing the antecedent at a high-recurrence feature raises rule coverage from
25.8% to **91.8%** (a 3.5× gain) and moves τ/message only from 0.779 to 0.840.

| antecedent → outcome | coverage | match rate | τ/message |
|---|---|---|---|
| `subject_norm → sender` | 25.8% | 59.8% | 0.779 |
| `sender → to_primary` | 91.8% | 17.4% | 0.840 |

Measured on the real engine. Coverage and match rate move in opposite
directions, and because the gate prices "unpredicted" at 0.9 and "violated" at
1.0, τ/message is pinned near 0.8 by the gate's own constants:

```
τ/message = 0.05·p(0.05) + 0.9·p(0.9) + 1.0·p(1.0)
```

For any traffic where ~85% of events are unpredicted-or-violated, this is ~0.8.
**The τ degeneracy reported earlier is a true statement about this corpus, not a
broken channel.** Chasing it by swapping antecedents cannot work, and two
attempts to fix it by regrading novelty failed in opposite directions before the
arithmetic was identified.

## Finding 6: the gate has no semantic input, and cannot have one

Novelty is computed from `predictions` alone (`engine.py:254-264`). `gain` is a
stored person weight and `arousal` is injected by the caller. The engine's
semantic similarity (`min_semantic_sim`) and entity extraction run **after** the
gate has already decided (`engine.py:321+`).

So "no rule predicted this" is the engine's only evidence, and it is priced at
0.9 — nearly maximal surprise. A message that is entirely routine but whose
subject has never recurred is scored 0.9, and nothing in the design can lower
it.

This is the actual lever. Novelty needs a second source, and the machinery to
compute one is already in the engine, three steps too late.

## What follows, in dependency order

1. **Fix the inducer/evaluator asymmetry.** One resolution of the sender, used by
   both halves. Small, and it removes a way to induce a rule that can never
   confirm. — **DONE**
2. **Make the antecedent and outcome configurable**, defaulting to today's
   `subject_norm → sender`. Removes the mail dependency from a generic engine.
   Pointing it at `sender → to_primary` is measured above if a deployment wants
   the coverage, but that is a policy choice, not a default. — **DONE**
3. **Use rule confidence.** A floor on evidence before a rule is trusted.
   Fixes Finding 1. — **DONE, but see below**
4. **Give the gate a semantic novelty term**, computed before the decision.
   Fixes Finding 6, and is the only step that can move τ/message materially.
   — **NOT DONE.** This is the one that matters.

## Step 3 landed, and it barely moves anything

Re-measured on the full corpus after the fix:

| arm | novelty 0.05 / 0.9 / 1.0 | τ/message |
|---|---|---|
| `features-raw`, before | 11.9 / 83.3 / 4.8% | 0.8036 |
| `features-raw`, after | 11.9 / 83.5 / 4.6% | 0.8035 |
| `features-norm`, before | 15.4 / 74.2 / 10.4% | 0.7788 |
| `features-norm`, after | 15.4 / 74.8 / 9.8% | 0.7788 |

Novelty 1.0 falls by 0.2 and 0.6 points. τ/message does not move at all — which
is Finding 5 reproducing itself.

The reason is worth recording, because it revises how much Finding 1 mattered.
The inducer rewrites a rule's expected outcome to the current top association
every time it fires, so a rule that starts being violated is *corrected* within
a few events and converges. Measured over 2,835 induced rules:

```
zero violations                 1,923  (68%)
mean violation_count            1.4
support>0 AND violation>0         613  of 2,835
fell below the 0.05 floor          16
```

So on induced traffic the confidence floor is nearly inert — the inducer masks
it. The fix is still right: it matters for stale or externally-seeded rules,
where nothing is rewriting the expectation. But it is a correctness fix, not a
τ fix, and should not be counted as progress on Finding 5.

Two further defects surfaced while implementing it:

- **`grade_rule` computed confidence from pre-update counts.** SQLite evaluates
  every `SET` expression against the pre-update row, so confidence lagged one
  event behind the counts it described.
- **An unproven rule scored confidence 0.0.** Unsmoothed,
  `support/(support+violation+2)` gives 0 for a rule with no evidence, so a
  brand-new belief read as maximally distrusted — and the moment the gate acted
  on confidence, that unproven rule lost its vote. Now add-one smoothed on the
  post-update row: no evidence is 0.5, matching the column default, and a rule
  with 200 violations and no support decays to 1/203 rather than 0.

The second one is a good illustration of why a computed-but-unread field can
hide for a long time: the formula looked reasonable and nobody was checking it
against a belief with no evidence.

## What still has to happen

Step 4 is the whole ballgame. Every mechanical fix above is a correctness fix;
none of them changes what novelty means, and novelty is the only thing that
sets τ. Until the gate can say "this was routine and I did not predict it",
τ will keep tracking message count, and any constant denominated in τ will keep
encoding nothing.

## Step 4: the gate now asks the content, and the gradient appears

When no rule fires, the engine no longer invents surprise. It measures Jaccard
overlap between the event's content terms and a bounded, recency-weighted
window of **recently encoded** events, and uses `1 − best_overlap`.

Measured on the full corpus, sender correctly placed:

| arm | distinct novelty values | largest single value | τ/message | encode rate |
|---|---|---|---|---|
| raw, before | 3 | 83.3% | 0.8036 | 0.881 |
| raw, after | ~130 | 20.2% | **0.6752** | **0.265** |
| norm, before | 3 | 74.2% | 0.7788 | 0.846 |
| norm, after | ~130 | 32.1% | **0.6530** | **0.385** |

The three-value ladder is gone, and the difference between the two arms now
reflects their content rather than the gate's constants.

### Why Jaccard over a recent window, and not the alternatives

Each was measured on the corpus first, because two earlier attempts at fixing
this by regrading the existing signal failed:

* **Word-level novelty against everything ever seen** saturates. A 19-year
  corpus exhausts its vocabulary: **74.6% of messages scored zero novel
  tokens**, and the largest single value did not improve. It is a *type*-level
  test, and types run out.
* **A vector search** needs an embedder at gate time — which is precisely the
  expensive work the gate exists to avoid. Rejected on cost, not quality.
* **FTS5** cannot return per-term hit counts cheaply, so it grades an event
  into three buckets rather than continuously.
* **A recency-weighted Jaccard window** is continuous, needs no embedder, and
  is how `activate`'s own recency fallback already works.

### Cost, honestly

Measured directly rather than end-to-end, because end-to-end timing reported
**−13.6%** — the feature is *faster* with the change, purely because fewer
events encode and encoding is the expensive part. That masked the real number:

```
window= 16   26.7 us/event
window= 32   45.9 us/event
window= 64   75.3 us/event
window=128  167.8 us/event
window=256  221.9 us/event
```

Linear in window size. At the default of 64 this roughly doubles the gate's
measured 43 µs — negligible against a 117 ms event, but it makes
`semantic_novelty_window` a real cost knob, and `semantic_novelty: False`
restores the old behaviour exactly.

### The problem this creates, which is not yet fixed

The encode rate collapsed from 0.85 to 0.27–0.39. Part of that is the feature
working. The rest is that **`threshold` is no longer calibrated**: it was set
when novelty was almost always 0.9 or 1.0, so `score = 0.5·novelty + 0.2·gain`
sat at 0.55–0.70 and only just encoded. With mean novelty now 0.65, `score`
sits near 0.425 and most events are suppressed.

That is a threshold fitted to the old novelty distribution, not a new operating
point chosen on purpose. Recalibrating `w_novelty` and `threshold` against the
measured distribution is the next step, and it should be done deliberately
rather than inherited.

## On calibrating anything in τ

`OBJECT_SALIENCE.md` marks every τ constant uncalibrated, on the grounds that
τ/day varied 61.2 vs 5.0 across windows. Finding 5 explains that: it varies
because τ tracks event count and event density, which is the intended behaviour.
The problem is not that τ is unstable; it is that τ carries no information
beyond "an event happened", so a constant denominated in τ encodes nothing about
elapsed experience. Step 4 is a precondition for any τ constant meaning
anything.