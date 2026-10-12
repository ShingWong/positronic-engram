# EG2 prompt sweep on E9 (V1–V6, pre-registered split)

Drivers: `q0_signif.py` (Q0, outcomes `q0_outcomes.npz`), `sweep.py`
(outcomes `sweep_outcomes.npz`). Split: stratified 50/50 of 915 queries
(~457 sel / ~458 eva, alternating within class in seed order).

## Q0: the gap is real

McNemar on acc@1 disagreements: 25 BGE-hit/EG2-miss vs 2 reverse, exact
two-sided p<0.0001. Paired bootstrap (10k) on MRR diff: 0.0173, 95%
CI [0.0106, 0.0248]. Overlap: 870 both-hit, 18 both-miss — not complementary;
BGE strictly dominates the disagreement set.

## V5 vs V1 first (the hypothesis test)

Selection half: V1 MRR 0.9637, V5 (swapped) 0.9699. Swapping the pair costs
nothing — it gains 0.006. The prompt mechanism is not engaged on this corpus.
Per the brief, no variant can then close the gap by prompting, whatever it scores.

## Leaderboard (selection half only)

| id | sel MRR | Δ vs V1 | note |
|---|---|---|---|
| V2 (none/none) | 0.9712 | +0.0075 | winner(sel) — and it is the no-prompt variant |
| V3 | 0.9710 | +0.0073 | |
| V5 (swapped) | 0.9699 | +0.0061 | |
| V6 | 0.9681 | +0.0044 | |
| V1 (control) | 0.9637 | — | |
| V4 (E5-instruct) | 0.9285 | −0.0352 | instruct form actively hurts |

Split-half noise floor: bootstrap SE of V1 sel MRR = 0.0073, 2SE = 0.0147.
Best positive Δ (+0.0075) is under half the noise floor. The prompt is not a
lever here — that is the answer, per the brief's stop rule.

## Gate (evaluation half, reported once)

V2 eva MRR 0.9763 vs BGE eva MRR 0.9860; McNemar 9-vs-2, p=0.0654.
**GATE=FAIL. BGE stays.** Text quality settled twice under pre-registration.

## Harness note

`q0_signif.py` first shipped with a leftover dead-code ternary in `cos()`
(caught on read before running). Per-query outcomes are persisted this time
(`q0_outcomes.npz`, `sweep_outcomes.npz`) so Q0-class questions stay free.
