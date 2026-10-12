"""EG2 vs BGE-M3 retrieval experiment for Positronic Engram Core.
BGE-M3 via production path (llama-server :8090, Q8 GGUF, pooling cls).
EmbeddingGemma-2 via sentence-transformers (google/embeddinggemma-2, CPU).
Corpus: 30 engram-like episodes. 12 queries with graded relevant sets.
Metrics: recall@1/3/5, MRR, encode latency, dim/storage.
"""
import json, time, urllib.request
import numpy as np

BGE_URL = "http://127.0.0.1:8090/embedding"

DOCS = [
 ("d01", "Tau decay pruning runs nightly. Episodes below salience 0.2 after 30 days are archived to cold storage."),
 ("d02", "The encode gate starves short episodes: with encode_threshold 0.0 all bodies embed, with 0.5 only salient ones."),
 ("d03", "BGE-M3 served by llama-server with --embedding --pooling cls on port 8090, context 8192, Q8 quant, 1024 dims."),
 ("d04", "CVE cascade experiment showed lexical retrieval beats dense vectors for version-string exact matches."),
 ("d05", "RRF fusion combines BM25 lexical scores with cosine semantic scores at weight 1.2 for the semantic channel."),
 ("d06", "Postfix maillog schema: queue_id, status, relay, delay fields parsed with grok patterns for log ingestion."),
 ("d07", "def bind_embedder(fn): hooks the embedding function into the memory engine encode path, best-effort with debug log."),
 ("d08", "Synthetic E7 benchmark: 55 episodes across diverse topics, headline 55/55/37/12 with gate off."),
 ("d09", "Adaptive negative mining improves contrastive recall by sampling hard negatives near the decision boundary."),
 ("d10", "Qwen3 27B draft MTP speculative decoding serves chat on port 8080 with HIP acceleration and cache reuse."),
 ("d11", "FlatVectorIndex stores float32 little-endian blobs in episode.body_embed and rejects dimension mismatches."),
 ("d12", "Consolidation markers summarize sessions into distilled memory; prune runs tau-decay before compacting."),
 ("d13", "Ruler benchmark measures long-context retrieval over 128k tokens with needle-in-haystack probes."),
 ("d14", "The agent interface owns .positronic/config.json and routes embed verbs local, lexical, or remote tiers."),
 ("d15", "Golden corpus for istrix eval: 500 archival episodes with provenance, salience, and wall-span metadata."),
 ("d16", "def chunk_markdown(text): splits bodies into 1200-token chunks, mean-pools embeddings, ceiling 7500 tokens."),
 ("d17", "Doctor command checks tiers lexical, bge, llama, engram health before running brain smoke tests."),
 ("d18", "LongMemEval tests episodic memory over 500-turn conversations with temporal and multi-hop questions."),
 ("d19", "Sparse BM25 index handles keyword queries; dense vectors handle paraphrase and semantic similarity."),
 ("d20", "Brain config per project: brains.{name}.embed with local_url default http://127.0.0.1:8090."),
 ("d21", "Gaussian noise injection during training regularizes the retrieval head without hurting top-1 accuracy."),
 ("d22", "Sourdough starter fed with rye flour rises twice as fast at 24C; fold every 30 minutes for open crumb. [distractor]"),
 ("d23", "Champions League quarter-final second leg kicks off at 21:00 CET; away goals rule abolished. [distractor]"),
 ("d24", "Monstera deliciosa needs indirect light and watering when top 5cm of soil dries out. [distractor]"),
 ("d25", "Postgres vacuum full rewrites the table; autovacuum settings tuned per workload in postgresql.conf."),
 ("d26", "Kubernetes HPA scales pods on CPU at 70 percent; VPA adjusts requests based on historical usage."),
 ("d27", "activate(cue) runs the semantic channel with min_semantic_sim floor 0.35 and weight 1.2."),
 ("d28", "Matryoshka MRL truncation slices 768-dim vectors to 256 or 128 dims with L2 renormalization."),
 ("d29", "systemd unit for bge embedding restarts always, binds port 8090, pooling cls, context 8192."),
 ("d30", "FTS5 full-text search backs the lexical channel with BM25 ranking over episode bodies."),
]

QUERIES = [
 ("q01", "How is the embedding server configured?", ["d03", "d29", "d20"]),
 ("q02", "What happens during memory consolidation and pruning?", ["d12", "d01"]),
 ("q03", "Show me the code that hooks embeddings into the engine", ["d07", "d16"]),
 ("q04", "How does semantic search combine with keyword search?", ["d05", "d19", "d27"]),
 ("q05", "What is the encode gate starvation issue?", ["d02", "d08"]),
 ("q06", "How are vectors stored in the episode table?", ["d11", "d30"]),
 ("q07", "Which benchmark tests long conversations?", ["d18", "d13"]),
 ("q08", "What did the CVE experiment find about exact matches?", ["d04", "d15"]),
 ("q09", "How does chunking work for long episode bodies?", ["d16", "d03"]),
 ("q10", "What checks does the doctor command run?", ["d17", "d14"]),
 ("q11", "How is the chat model served?", ["d10", "d26"]),
 ("q12", "What is the semantic similarity floor in activation?", ["d27", "d05"]),
]

def bge_embed(texts):
    vecs = []
    for t in texts:
        req = urllib.request.Request(BGE_URL, data=json.dumps({"content": t}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
            e = body[0]["embedding"]
            vecs.append(e[0] if isinstance(e[0], list) else e)
    return np.array(vecs, dtype=np.float64)

def cos(q, D):
    qn = q / (np.linalg.norm(q) + 1e-12)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    return Dn @ qn

def evaluate(encode_q, encode_d, name):
    t0 = time.time()
    D = encode_d([d for _, d in DOCS])
    t_enc_d = time.time() - t0
    recalls = {1: [], 3: [], 5: []}
    rrs = []
    t_qlat = []
    for qid, q, rel in QUERIES:
        t1 = time.time()
        qv = encode_q(q)
        t_qlat.append(time.time() - t1)
        s = cos(qv, D)
        rank = np.argsort(-s)
        ids = [DOCS[i][0] for i in rank]
        best = min(ids.index(r) for r in rel if r in ids)
        rrs.append(1.0 / (best + 1))
        for k in recalls:
            recalls[k].append(any(r in ids[:k] for r in rel))
    return {
        "model": name,
        "dim": int(D.shape[1]),
        "recall@1": round(float(np.mean(recalls[1])), 3),
        "recall@3": round(float(np.mean(recalls[3])), 3),
        "recall@5": round(float(np.mean(recalls[5])), 3),
        "mrr": round(float(np.mean(rrs)), 3),
        "encode_corpus_s": round(t_enc_d, 1),
        "ms_per_query": round(float(np.mean(t_qlat)) * 1000, 1),
        "bytes_per_vec_fp32": int(D.shape[1]) * 4,
    }

results = {}
results["bge-m3"] = evaluate(lambda q: bge_embed([q])[0], bge_embed, "BAAI/bge-m3 Q8 llama-server :8090")
print("BGE done:", results["bge-m3"], flush=True)

from sentence_transformers import SentenceTransformer
eg2 = SentenceTransformer("google/embeddinggemma-2", device="cpu")
print("EG2 loaded", flush=True)
def eg2_q(q):
    return eg2.encode([q], prompt_name="SearchQuery", normalize_embeddings=True)[0].astype(np.float64)
def eg2_d(texts):
    return eg2.encode(texts, prompt_name="Document", normalize_embeddings=True).astype(np.float64)
results["embeddinggemma-2"] = evaluate(eg2_q, eg2_d, "google/embeddinggemma-2 fp32 CPU")
print("EG2 done:", results["embeddinggemma-2"], flush=True)

print(json.dumps(results, indent=1))
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=1)
