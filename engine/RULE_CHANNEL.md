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
   confirm.
2. **Make the antecedent and outcome configurable**, defaulting to today's
   `subject_norm → sender`. Removes the mail dependency from a generic engine.
   Pointing it at `sender → to_primary` is measured above if a deployment wants
   the coverage, but that is a policy choice, not a default.
3. **Use rule confidence.** A floor on evidence before a rule is trusted, and
   retirement for rules that carry enough evidence to be disproven. Fixes
   Finding 1.
4. **Give the gate a semantic novelty term**, computed before the decision.
   Fixes Finding 6, and is the only step that can move τ/message materially.

Steps 1–3 are mechanical and low-risk. Step 4 changes what Eq. 2 in the paper
describes, so it is a decision rather than an implementation.

## On calibrating anything in τ

`OBJECT_SALIENCE.md` marks every τ constant uncalibrated, on the grounds that
τ/day varied 61.2 vs 5.0 across windows. Finding 5 explains that: it varies
because τ tracks event count and event density, which is the intended behaviour.
The problem is not that τ is unstable; it is that τ carries no information
beyond "an event happened", so a constant denominated in τ encodes nothing about
elapsed experience. Step 4 is a precondition for any τ constant meaning
anything.