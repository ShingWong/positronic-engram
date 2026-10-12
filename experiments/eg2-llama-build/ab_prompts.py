"""Prompts A/B on the 30-doc / 12-query corpus (all via :8091 Q8 server).
(a) raw/raw  (b) prompted/prompted  (c) prompted docs + raw queries (half-applied).
BGE baseline from ../embed-gemma2-vs-bge/metrics.json: R@1 0.917, MRR 0.958."""
import json, sys, urllib.request
import numpy as np

sys.path.insert(0, "../embed-gemma2-vs-bge")
from run import DOCS, QUERIES, cos  # noqa

QPRE = "task: search result | query: "
DPRE = "title: none | text: "

def srv(texts):
    out = []
    for t in texts:
        req = urllib.request.Request("http://127.0.0.1:8091/embedding",
                                     data=json.dumps({"content": t}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.load(r)
        e = body[0]["embedding"]
        out.append(e[0] if isinstance(e[0], list) else e)
    return np.array(out, dtype=np.float64)

def score(D, QV):
    r1, rr = [], []
    for (_, _, rel), qv in zip(QUERIES, QV):
        s = cos(qv, D)
        ids = [DOCS[i][0] for i in np.argsort(-s)]
        best = min(ids.index(r) for r in rel)
        rr.append(1.0 / (best + 1))
        r1.append(best == 0)
    return round(float(np.mean(r1)), 3), round(float(np.mean(rr)), 3)

D_raw = srv([d for _, d in DOCS])
D_doc = srv([DPRE + d for _, d in DOCS])
Q_raw = srv([q for _, q, _ in QUERIES])
Q_qry = srv([QPRE + q for _, q, _ in QUERIES])

res = {
    "a_raw_raw": score(D_raw, Q_raw),
    "b_prompted_prompted": score(D_doc, Q_qry),
    "c_promptedDocs_rawQueries": score(D_doc, Q_raw),
}
print(json.dumps({k: {"R@1": v[0], "MRR": v[1]} for k, v in res.items()}, indent=1))
json.dump({k: {"R@1": v[0], "MRR": v[1]} for k, v in res.items()},
          open("ab_metrics.json", "w"), indent=1)
