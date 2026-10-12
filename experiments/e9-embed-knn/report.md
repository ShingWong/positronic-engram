# E9 embedding A/B: kNN classification of held-out labelled mail lines

Driver: `e9_knn.py`. Corpus: `mail.db` (E9 run-1791164085): 9,246 unclassified
+ 10,037 labelled message episodes, 47 classes. 915 stratified held-out queries
vs 9,122 labelled candidates. BGE `:8090` raw; EG2 `:8091` Q8 with
`Document`/`SearchQuery` prompts; lexical = FTS5 BM25.

## Metrics (`metrics.json`)

| backend | acc@1 | MRR |
|---|---|---|
| BGE-M3 Q8 | 0.978 | 0.985 |
| EG2 Q8 prompted | 0.953 | 0.968 |
| lexical, like-for-like (labelled only) | 0.191 | 0.307 |
| lexical, deployment-like (+9,246 unlabelled distractors) | 0.119 | 0.208 |

EG2 accuracy on the 806 lexical-miss queries: 0.957. BGE misses ~20/915 total,
so BGE-on-lexmiss is ≥0.97 regardless of overlap: BGE fixes lexical misses
at least as well as EG2.

## Harness bugs found and fixed during the run

1. Singleton-class drain: tiny classes sampled wholly into queries left no
   same-class candidate (`StopIteration`). Now every sampled class keeps ≥1
   candidate; singletons excluded (915 queries, not 1000).
2. Lexical searched group/session episodes (aggregates) while dense searched
   messages only — 915/915 `None`. Now joined to `kind='message'`.

## Gate verdict

FAIL clause 1: EG2 MRR 0.968 < BGE MRR 0.985. **Do not swap; BGE stays.**
Both dense backends crush lexical (vectors matter here), but BGE is the
better vector backend on structured log text. EG2's remaining case is the
multimodal bet (mmproj + PAI client path), not text quality.
