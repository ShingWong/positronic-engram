"""E9 embedding A/B: kNN classification of held-out labelled mail lines.
Queries: 1000 stratified labelled messages (label hidden).
Candidates: all other labelled messages.
Backends: BGE :8090 (raw) vs EG2 :8091 (Document prompt on docs, SearchQuery on queries).
Lexical baseline: FTS5 BM25 over episode_fts.
Metrics: acc@1, MRR (relevant = same event_class); EG2 acc on lexical-miss subset.
"""
import json, random, re, sqlite3, urllib.request
import numpy as np

DB = "/usr/local/devel/positronic/positronic-benchmarks/results/log_ingest/run-1791164085/mail/mail.db"
DPRE = "title: none | text: "
QPRE = "task: search result | query: "
N_Q = 1000
random.seed(11)

c = sqlite3.connect(DB)
msgs = []  # (id, message, class)
for (i, fj) in c.execute("select id, features_json from episode where kind='message'"):
    d = json.loads(fj)
    if d.get("event_class") and d.get("message"):
        msgs.append((i, d["message"], d["event_class"]))
print(f"labelled messages: {len(msgs)}", flush=True)

by_cls = {}
for m in msgs:
    by_cls.setdefault(m[2], []).append(m)
queries = []
per = max(1, N_Q // len(by_cls))
for cls, items in sorted(by_cls.items()):
    if len(items) < 2:
        continue  # singleton: cannot be both query and candidate
    queries += random.sample(items, min(per, len(items) - 1))
queries = queries[:N_Q]
qids = {q[0] for q in queries}
cands = [m for m in msgs if m[0] not in qids]
print(f"queries: {len(queries)}, candidates: {len(cands)}", flush=True)

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

def knn(D, Q, cand_cls, true_cls):
    acc, rrs = [], []
    for qv, tc in zip(Q, true_cls):
        s = cos(qv, D)
        order = np.argsort(-s)
        pred = cand_cls[order[0]]
        acc.append(pred == tc)
        rank = next(i for i, j in enumerate(order) if cand_cls[j] == tc)
        rrs.append(1.0 / (rank + 1))
    return float(np.mean(acc)), float(np.mean(rrs)), [bool(a) for a in acc]

cand_ids = [m[0] for m in cands]
cand_cls = [m[2] for m in cands]
true_cls = [q[2] for q in queries]

print("embedding BGE candidates...", flush=True)
Db = embed(8090, [m[1] for m in cands])
print("embedding BGE queries...", flush=True)
Qb = embed(8090, [q[1] for q in queries])
acc_b, mrr_b, _ = knn(Db, Qb, cand_cls, true_cls)
print(f"BGE acc@1={acc_b:.3f} MRR={mrr_b:.3f}", flush=True)

print("embedding EG2 candidates...", flush=True)
De = embed(8091, [DPRE + m[1] for m in cands])
print("embedding EG2 queries...", flush=True)
Qe = embed(8091, [QPRE + q[1] for q in queries])
acc_e, mrr_e, ok_e = knn(De, Qe, cand_cls, true_cls)
print(f"EG2 acc@1={acc_e:.3f} MRR={mrr_e:.3f}", flush=True)

# lexical baseline: BM25 over message episodes only (same candidate space as
# dense; group/session episodes aggregate many messages and would dominate).
# acc@1 from top hit; MRR from rank of first same-class hit in top-50.
id2cls = {m[0]: m[2] for m in msgs}
lex_hits, lex_rr, lex_miss_idx = [], [], []
tok = re.compile(r"[A-Za-z0-9_.-]+")
for qi, q in enumerate(queries):
    terms = tok.findall(q[1][:400])
    if not terms:
        lex_hits.append(False); lex_rr.append(0.0); lex_miss_idx.append(qi); continue
    mq = " OR ".join(f'"{t}"' for t in terms[:20])
    try:
        rows = c.execute(
            "SELECT fts.id FROM episode_fts fts JOIN episode e ON e.id = fts.id "
            "WHERE e.kind = 'message' AND episode_fts MATCH ? LIMIT 50",
            (mq,)).fetchall()
        classes = [id2cls.get(r[0]) for r in rows]
        hit = bool(classes) and classes[0] == q[2]
        rank = next((i for i, cl in enumerate(classes) if cl == q[2]), None)
    except Exception:
        hit, rank = False, None
    lex_hits.append(hit)
    lex_rr.append(1.0 / (rank + 1) if rank is not None else 0.0)
    if not hit:
        lex_miss_idx.append(qi)
acc_l = float(np.mean(lex_hits))
mrr_l = float(np.mean(lex_rr))
eg2_on_lexmiss = float(np.mean([ok_e[i] for i in lex_miss_idx])) if lex_miss_idx else 0.0
print(f"LEX acc@1={acc_l:.3f} misses={len(lex_miss_idx)} EG2-on-lexmiss={eg2_on_lexmiss:.3f}", flush=True)

res = {
    "n_queries": len(queries), "n_candidates": len(cands),
    "bge": {"acc@1": round(acc_b, 3), "mrr": round(mrr_b, 3)},
    "eg2": {"acc@1": round(acc_e, 3), "mrr": round(mrr_e, 3)},
    "lexical": {"acc@1": round(acc_l, 3), "mrr": round(mrr_l, 3),
                "misses": len(lex_miss_idx),
                "eg2_acc_on_lexmiss": round(eg2_on_lexmiss, 3)},
}
print(json.dumps(res, indent=1))
json.dump(res, open("metrics.json", "w"), indent=1)
