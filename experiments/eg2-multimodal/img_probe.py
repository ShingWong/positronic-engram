"""Cross-modal probe: text queries vs image embeddings via :8091 Q8+mmproj.
Working shape: {'content': {'prompt_string': '<__media__>', 'multimodal_data': [b64]}}.
Server needs LLAMA_MEDIA_MARKER='<__media__>' (default is a random marker).
Reports full cosine matrix; pass = diagonal argmax."""
import base64, json, urllib.request
import numpy as np

IMGS = ["sq_red.png", "ci_blue.png", "tr_green.png", "cat.png"]
QUERIES = ["a red square", "a blue circle", "a green triangle", "a cat"]
QPRE = "task: search result | query: "

def post(payload):
    req = urllib.request.Request("http://127.0.0.1:8091/embedding",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    e = body[0]["embedding"]
    v = np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)
    return v / np.linalg.norm(v)

def img(f, txt="<__media__>"):
    b64 = base64.b64encode(open(f, "rb").read()).decode()
    return post({"content": {"prompt_string": txt, "multimodal_data": [b64]}})

print("embedding images...", flush=True)
I = [img(f) for f in IMGS]
print("image-image cos(red,blue)=%.3f" % float(I[0] @ I[1]), flush=True)
print("embedding queries...", flush=True)
Qr = [post({"content": q}) for q in QUERIES]
Qp = [post({"content": QPRE + q}) for q in QUERIES]

res = {}
for name, Q in [("raw", Qr), ("prompted", Qp)]:
    print(f"--- queries {name} ---")
    ok = 0
    sims_all = []
    for i, q in enumerate(Q):
        sims = [round(float(q @ v), 3) for v in I]
        sims_all.append(sims)
        best = int(np.argmax(sims))
        ok += best == i
        print(f"Q'{QUERIES[i]}': sims={sims} best={IMGS[best]} {'OK' if best == i else 'MISS'}")
    print(f"{name}: {ok}/4 diagonal")
    res[name] = {"diagonal": ok, "sims": sims_all}
json.dump(res, open("img_metrics.json", "w"), indent=1)
