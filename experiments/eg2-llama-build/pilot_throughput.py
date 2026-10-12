"""Pilot: end-to-end embed throughput on 1000 synthetic episodes.
Mirrors positronic_ai.embed: split into ~1200-token chunks, embed each chunk
(EG2 gets the Document prompt per chunk, BGE gets raw text), mean-pool.
Reports records/s per backend and projected full-run (189,401) time."""
import json, random, time, urllib.request
import numpy as np

random.seed(7)
SENTS = [
    "The KV cache streams in block granularity with a shared phase arena.",
    "Prefill throughput scales with batch size until memory bandwidth saturates.",
    "Consolidation distills session episodes into long-term memory markers.",
    "BM25 lexical scores fuse with cosine semantic scores via RRF at weight 1.2.",
    "CVE records carry vendor, product, version ranges and CVSS vectors.",
    "Speculative decoding drafts tokens with a small model for the large one to verify.",
    "Matryoshka truncation slices trailing dims and renormalizes the remainder.",
    "Autovacuum workers reclaim dead tuples without locking the whole table.",
]
DPRE = "title: none | text: "

def make_episode(i):
    n = random.choice([1, 1, 2, 2, 3, 8])  # mostly short, some long (chunked)
    return f"Episode {i}: " + " ".join(random.choice(SENTS) for _ in range(n * 12))

EPS = [make_episode(i) for i in range(1000)]

def chunks(text, chars=4800):
    return [text[i:i + chars] for i in range(0, len(text), chars)] or [text]

def embed(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/embedding",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    e = body[0]["embedding"]
    return np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)

def run(port, prompt):
    t0 = time.time()
    for ep in EPS:
        vecs = [embed(port, (prompt + c) if prompt else c) for c in chunks(ep)]
        np.mean(vecs, axis=0)  # mean-pool like embed.py
    dt = time.time() - t0
    return {"records": len(EPS), "seconds": round(dt, 1),
            "records_per_s": round(len(EPS) / dt, 1),
            "projected_189k_h": round(189401 / (len(EPS) / dt) / 3600, 2)}

res = {"bge_8090": run(8090, ""), "eg2_8091": run(8091, DPRE)}
print(json.dumps(res, indent=1))
json.dump(res, open("pilot_metrics.json", "w"), indent=1)
