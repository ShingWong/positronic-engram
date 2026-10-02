# Known defects

Defects that are measured, understood, and NOT fixed, with the reasoning
and the rejected alternatives. Each is pinned by a test — skipped where the
test would otherwise be red, so the claim stays visible rather than being
quietly dropped.

---

## 1. Novelty depends on unrelated memory -- noise filtering DECAYS

**Status:** open, and the PRIMARY defect. Pinned by
`test_novelty_must_not_rise_as_the_window_fills_with_duplicates` (skipped).

**This is the main open failure, not a footnote.** Measured as store rate
by decile of a 120,000-line run on the mx1 corpus:

```
service       d1    d2    d3    d4    d5    d6    d7    d8    d9   d10
dnsblog     14%   13%   15%   13%   20%   49%   44%   73%   64%     -
smtpd       11%   21%   22%   22%   10%    7%   22%   42%   40%   58%
postscreen  21%   18%   20%   18%   29%   63%   54%   66%   69%   92%
```

The boilerplate services are stored MORE over time: `dnsblog` 14% -> 73%,
`postscreen` 21% -> 92%. **Noise filtering decays as the brain fills up.**
mx1 is a noise-filter corpus -- 66,260 dnsblog lines carrying 12 distinct
strings -- so this is precisely what it exists to catch, and it is the only
one of the four corpora with both a repetitive stream and enough length to
show the drift. email, git and PVL have high-variance content, so their
comparison windows stay varied and the defect never fires.

### The claim

`_semantic_novelty` scores an event against a window of recently *stored*
events using IDF-weighted cosine, with document frequency computed from
that window. So an event's novelty depends not only on the event but on
**what else the brain happens to remember.**

Isolated with a fixed event pair — query `{alpha bravo charlie delta echo
z}`, window entry `{alpha bravo charlie delta echo y}` — and window entries
added that share **nothing** with the pair:

```
window size  1      5      9     13     21     37
novelty    0.2722 0.2155 0.2056 0.2010 0.1962 0.1920
```

A 41% change in an event's novelty, caused entirely by unrelated memory.

### The visible symptom

On a repetitive stream the effect compounds, because each stored event
contributes its own unique token. Smoothed IDF is
`log((n+1)/(df+1)) + 1`, so an unseen term weighs `log(n)+1` and grows
**without bound**, while a shared term saturates at 1.0. Two events
differing by one token then have similarity `5 / (5 + (log n + 1)^2)`,
which collapses as the window grows:

```
window  1     2     3     4     5     6     8    10    12
novelty 0.272 0.383 0.455 0.506 0.544 0.574 0.618 0.650 0.674
```

**A brain that remembers more of a repetitive stream decides each new event
is more novel. Duplicate suppression inverts.**

### It is visible at run scale, and hidden at test scale

Over a 600-event test the gate stores so little of a repetitive stream that
`_recent_terms` stays near-empty, so `df` barely moves and the effect looks
small (0.2% stored). Over a 120,000-line run the window fills with copies
and the drift becomes the dominant behaviour. **A short test cannot see
this failure; only a long repetitive corpus can.**

It becomes live the moment the gate is permissive enough to fill the
window with copies. Applying a median-based self-calibrating warmup cut —
and nothing else — takes the same 600 events from **0.2% to 598/600
stored** and fails `test_near_duplicates_are_still_suppressed`.

### Rejected fixes, all measured

| fix | duplicate novelty | mx1 final | git final | verdict |
|---|---|---|---|---|
| shipped (no fix) | 0.383 -> 0.633 rises | 32.1% | 34.5% | defect |
| absolute IDF cap 1.2 | 0.224 -> 0.224 holds | 34.8% | 34.5% | **rejected** |
| absolute IDF cap 2.0 / 3.0 | 0.369 -> 0.444 rises | 32.3% | 32.1% | rejected |
| cap relative to window median | 0.369 -> 0.444 rises | 33.8% | 33.7% | rejected |
| unseen term takes median weight | 0.227 -> 0.439 rises | 32.1% | 32.9% | rejected |
| cosine over the intersection only | 0.000 -> 0.000 | 100.0% | 100.0% | **degenerate** |

- **Cap 1.2** is the only value that stops the rise, but it equalises every
  weight, which **removes IDF entirely**. IDF was adopted because weighting
  by rarity fixed a measured 0.1% store rate, so this trades one bug for
  the one it was meant to fix.
- **Intersection-only norms** make the denominator equal the numerator, so
  similarity is always 1.0 and every event scores novelty 0.000 — the
  degenerate all-1.0 case already documented at the gate's call site, in
  mirror image.

### Why it is not fixed here

The defect is structural, not a matter of picking better weights. Document
frequency is computed over a window that contains the event's own
near-duplicates, so any weighting that respects rarity will overweight the
distinguishing token. A fix has to change *what the window is* — for
instance by keeping document frequency over a lifetime sample independent
of the recency window, or by scoring against stored episodes rather than
term sets. Both are design changes to the novelty estimator, and neither
can be adopted on the strength of a synthetic duplicate stream alone: the
four real corpora must be re-measured, and this is exactly the kind of
change that has already produced two discarded artifacts this project.

### What it blocks

The cold-start fix. A median-based fallback cut removes the cold start on
all four corpora (mx1 opens 2.1% -> 29.8%, pvl 6.4% -> 37.9%, spread 1.12x
-> 1.09x) but depends on the estimator tolerating a duplicate-filled
window. Shipping it before this is fixed would trade a visible cold start
for a silent duplicate flood.