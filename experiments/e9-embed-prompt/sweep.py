"""Prompt sweep V1-V6 on EG2 :8091. Pre-registered per TASK-eg2-prompt-ablation.
Split: stratified 50/50 of the 915 queries (alternate within class by seed order).
Winner picked on selection half; reported once on eval half vs BGE (from q0 npz).
Gate: eval-half MRR beats BGE eval-half MRR + McNemar significant."""
import json, random, sqlite3, urllib.request
from math import comb
import numpy as np

DB = "/usr/local/devel/positronic/positronic-benchmarks/results/log_ingest/run-1791164085/mail/mail.db"
VARIANTS = {
    "V1": ("Document: ", "SearchQuery: "),
    "V2": ("", ""),
    "V3": ("Document:\n", "SearchQuery:\n"),
    "V4": ("Instruct: Represent this line for searching relevant log lines.\nQuery: ", "passage: "),
    "V5": ("SearchQuery: ", "Document: "),
    "V6": ("Document: ", ""),
}
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

# stratified 50/50: alternate within class, preserving seed order
sel_idx, eva_idx = [], []
per_cls = {}
for i, q in enumerate(queries):
    k = len(per_cls.setdefault(q[2], []))
    per_cls[q[2]].append(i)
    (sel_idx if k % 2 == 0 else eva_idx).append(i)
sel_idx = np.array(sel_idx); eva_idx = np.array(eva_idx)
print(f"queries={len(queries)} sel={len(sel_idx)} eva={len(eva_idx)}", flush=True)

def embed(texts):
    out = []
    for t in texts:
        req = urllib.request.Request("http://127.0.0.1:8091/embedding",
                                     data=json.dumps({"content": t}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            e = json.load(r)[0]["embedding"]
        out.append(e[0] if isinstance(e[0], list) else e)
    return np.array(out, dtype=np.float64)

def cos(q, D):
    return (D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)) @ (q / (np.linalg.norm(q) + 1e-12))

cand_cls = [m[2] for m in cands]
true_all = [q[2] for q in queries]

def per_query(D, Q):
    hits = np.zeros(len(Q), dtype=bool); rrs = np.zeros(len(Q))
    for i, (qv, tc) in enumerate(zip(Q, true_all)):
        s = cos(qv, D)
        order = np.argsort(-s)
        hits[i] = cand_cls[order[0]] == tc
        rank = next(j for j, k in enumerate(order) if cand_cls[k] == tc)
        rrs[i] = 1.0 / (rank + 1)
    return hits, rrs

sel_res, all_hits, all_rr = {}, {}, {}
for vid, (dp, qp) in VARIANTS.items():
    print(f"embed {vid} candidates...", flush=True)
    D = embed([dp + m[1] for m in cands])
    print(f"embed {vid} queries...", flush=True)
    Q = embed([qp + q[1] for q in queries])
    h, r = per_query(D, Q)
    all_hits[vid] = h; all_rr[vid] = r
    sel_res[vid] = (h[sel_idx].mean(), r[sel_idx].mean())
    print(f"{vid} SEL acc={h[sel_idx].mean():.3f} MRR={r[sel_idx].mean():.4f}", flush=True)
np.savez("sweep_outcomes.npz", **{f"h_{v}": all_hits[v] for v in VARIANTS},
         **{f"r_{v}": all_rr[v] for v in VARIANTS},
         sel_idx=sel_idx, eva_idx=eva_idx)

# V5 vs V1 FIRST (hypothesis test, not leaderboard)
h1, r1 = all_hits["V1"], all_rr["V1"]
h5, r5 = all_hits["V5"], all_rr["V5"]
print(f"V5-vs-V1 SEL: V1 MRR={r1[sel_idx].mean():.4f} V5 MRR={r5[sel_idx].mean():.4f} "
      f"EVA: V1 MRR={r1[eva_idx].mean():.4f} V5 MRR={r5[eva_idx].mean():.4f}")

# selection-half noise floor: std of MRR across variants on sel
mrrs_sel = np.array([all_rr[v][sel_idx].mean() for v in VARIANTS])
print(f"variant spread SEL: min={mrrs_sel.min():.4f} max={mrrs_sel.max():.4f} "
      f"range={mrrs_sel.max()-mrrs_sel.min():.4f}")

# winner on selection half only
winner = max(VARIANTS, key=lambda v: all_rr[v][sel_idx].mean())
print(f"WINNER(sel)={winner}", flush=True)

# report once on eval half vs BGE (BGE per-query outcomes from Q0)
q0 = np.load("q0_outcomes.npz")
hb, rb = q0["hb"], q0["rb"]
hw, rw = all_hits[winner], all_rr[winner]
print(f"EVA: {winner} acc={hw[eva_idx].mean():.3f} MRR={rw[eva_idx].mean():.4f} | "
      f"BGE acc={hb[eva_idx].mean():.3f} MRR={rb[eva_idx].mean():.4f}")
b10 = int((hb[eva_idx] & ~hw[eva_idx]).sum()); b01 = int((hw[eva_idx] & ~hb[eva_idx]).sum())
n = b10 + b01
p = 2 * sum(comb(n, k) for k in range(0, min(b10, b01) + 1)) / 2**n if n else 1.0
print(f"EVA McNemar({winner} vs BGE): b10={b10} b01={b01} p={min(p,1.0):.4f}")
gate = bool(rw[eva_idx].mean() > rb[eva_idx].mean() and min(p, 1.0) < 0.05)
print(f"GATE={'PASS swap' if gate else 'FAIL BGE stays'}")

out = {"variants": {v: {"sel_acc": round(float(all_hits[v][sel_idx].mean()), 3),
                       "sel_mrr": round(float(all_rr[v][sel_idx].mean()), 4),
                       "eva_acc": round(float(all_hits[v][eva_idx].mean()), 3),
                       "eva_mrr": round(float(all_rr[v][eva_idx].mean()), 4)}
                   for v in VARIANTS},
       "winner_sel": winner,
       "bge_eva": {"acc": round(float(hb[eva_idx].mean()), 3), "mrr": round(float(rb[eva_idx].mean()), 4)},
       "mcnemar": {"b10": b10, "b01": b01, "p": round(min(p, 1.0), 4)},
       "gate": "PASS swap" if gate else "FAIL BGE stays"}
json.dump(out, open("metrics.json", "w"), indent=1)
