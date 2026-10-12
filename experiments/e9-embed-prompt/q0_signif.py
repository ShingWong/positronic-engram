"""Q0: is the BGE-vs-EG2 gap real? Paired stats on the same 915 queries.
Re-embeds BGE + EG2(ST-default prompts, replicating e9_knn.py) and persists
per-query outcomes to q0_outcomes.npz. McNemar on acc@1 disagreements +
paired bootstrap CI on the MRR difference."""
import json, random, re, sqlite3, urllib.request
import numpy as np

DB = "/usr/local/devel/positronic/positronic-benchmarks/results/log_ingest/run-1791164085/mail/mail.db"
DPRE = "title: none | text: "
QPRE = "task: search result | query: "
random.seed(11)

c = sqlite3.connect(DB)
msgs = []
for (i, fj) in c.execute("select id, features_json from episode where kind='message'"):
    d = json.loads(fj)
    if d.get("event_class") and d.get("message"):
        msgs.append((i, d["message"], d["event_class"]))
by_cls = {}
for m in msgs:
    by_cls.setdefault(m[2], []).append(m)
queries = []
per = max(1, 1000 // len(by_cls))
for cls, items in sorted(by_cls.items()):
    if len(items) < 2:
        continue
    queries += random.sample(items, min(per, len(items) - 1))
queries = queries[:1000]
qids = {q[0] for q in queries}
cands = [m for m in msgs if m[0] not in qids]
print(f"queries={len(queries)} candidates={len(cands)}", flush=True)

def embed(port, texts):
    out = []
    for t in texts:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/embedding",
                                     data=json.dumps({"content": t}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            e = json.load(r)[0]["embedding"]
        out.append(e[0] if isinstance(e[0], list) else e)
    return np.array(out, dtype=np.float64)

def cos(q, D):
    return (D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)) @ (q / (np.linalg.norm(q) + 1e-12))

def per_query(D, Q, cand_cls, true_cls):
    hits = np.zeros(len(Q), dtype=bool)
    rrs = np.zeros(len(Q))
    for i, (qv, tc) in enumerate(zip(Q, true_cls)):
        s = cos(qv, D)
        order = np.argsort(-s)
        hits[i] = cand_cls[order[0]] == tc
        rank = next(j for j, k in enumerate(order) if cand_cls[k] == tc)
        rrs[i] = 1.0 / (rank + 1)
    return hits, rrs

cand_cls = [m[2] for m in cands]
true_cls = [q[2] for q in queries]
print("embed BGE...", flush=True)
hb, rb = per_query(embed(8090, [m[1] for m in cands]), embed(8090, [q[1] for q in queries]), cand_cls, true_cls)
print("embed EG2...", flush=True)
he, re_ = per_query(embed(8091, [DPRE + m[1] for m in cands]), embed(8091, [QPRE + q[1] for q in queries]), cand_cls, true_cls)
np.savez("q0_outcomes.npz", hb=hb, rb=rb, he=he, re=re_, nq=len(queries))

b = hb.mean(); e = he.mean()
print(f"BGE acc={b:.3f} MRR={rb.mean():.3f} | EG2 acc={e:.3f} MRR={re_.mean():.3f}")
# McNemar: b10 = BGE hit/EG2 miss, b01 = EG2 hit/BGE miss; exact two-sided binomial
b10 = int((hb & ~he).sum()); b01 = int((he & ~hb).sum())
from math import comb
n = b10 + b01
p = 2 * sum(comb(n, k) for k in range(0, min(b10, b01) + 1)) / 2**n if n else 1.0
print(f"McNemar: b10={b10} b01={b01} exact two-sided p={min(p,1.0):.4f}")
print(f"overlap: both-hit={(hb&he).sum()} both-miss={(~hb&~he).sum()}")
# paired bootstrap on MRR diff
rng = np.random.default_rng(0)
diffs = rb - re_
boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)])
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"MRR diff (BGE-EG2)={diffs.mean():.4f} 95% CI=[{lo:.4f},{hi:.4f}]")
json.dump({"b10": b10, "b01": b01, "mcnemar_p": round(min(p, 1.0), 4),
           "mrr_diff": round(float(diffs.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
           "n": len(queries)},
          open("q0_metrics.json", "w"), indent=1)
