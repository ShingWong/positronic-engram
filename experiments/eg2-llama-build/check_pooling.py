"""Phase 2 gate v2: isolate pooling/quant from task prompts.
A: server(raw) vs ST(raw, no prompt) -> pooling + Q8 fidelity.
B: server(prompted) vs ST(prompted) -> end-to-end serving path.
Pass: A and B cosine > 0.99 on all probes, dim 768."""
import json, time, urllib.request
import numpy as np

PROBES = [
    "Tau decay pruning runs nightly.",
    "def bind_embedder(fn): hooks the embedding function into the engine.",
    "Postfix maillog schema: queue_id, status, relay, delay fields.",
    "Ruler benchmark measures long-context retrieval over 128k tokens.",
    "Kubernetes HPA scales pods on CPU at 70 percent.",
]
PROMPT = "title: none | text: "  # ST Document-prompt analogue; exact prefix printed below

def server_embed(text):
    req = urllib.request.Request("http://127.0.0.1:8091/embedding",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    dt = (time.time() - t0) * 1000
    e = body[0]["embedding"]
    return (np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64), dt)

from sentence_transformers import SentenceTransformer
ref = SentenceTransformer("google/embeddinggemma-2", device="cpu")
# Show what the Document prompt actually prepends, so the server prefix matches
print("ST Document prompt:", repr(ref.prompts.get("Document", None)))
R_raw = ref.encode(PROBES, normalize_embeddings=True).astype(np.float64)
R_doc = ref.encode(PROBES, prompt_name="Document", normalize_embeddings=True).astype(np.float64)

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

ok = True
for i, p in enumerate(PROBES):
    v_raw, ms = server_embed(p)
    v_doc, _ = server_embed("title: none | text: " + p)
    if i == 0:
        print("server dim:", v_raw.shape[0])
    ca, cb = cos(v_raw, R_raw[i]), cos(v_doc, R_doc[i])
    fa = "OK " if ca > 0.99 else "FAIL"
    fb = "OK " if cb > 0.99 else "FAIL"
    ok = ok and ca > 0.99 and cb > 0.99
    print(f"{fa} A raw      probe{i} cosine={ca:.4f} {ms:.0f}ms")
    print(f"{fb} B prompted probe{i} cosine={cb:.4f}")
print(f"GATE={'PASS' if ok else 'FAIL'}")
