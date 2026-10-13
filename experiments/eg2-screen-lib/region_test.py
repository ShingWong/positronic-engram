"""Region-crop tutorial detector test: which crop separates tutorial frames?
Regions: full, right half, center band, top band. Metric: LOO prototype accuracy.
"""
import base64, io, json, urllib.request, glob
import numpy as np
from PIL import Image

REGIONS = {
    "full": None,
    "right": (0.5, 0.0, 1.0, 1.0),
    "centerband": (0.2, 0.3, 1.0, 0.7),
    "top": (0.0, 0.0, 1.0, 0.35),
}

def post(payload):
    req = urllib.request.Request("http://127.0.0.1:8091/embedding",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    e = body[0]["embedding"]
    v = np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)
    return v / np.linalg.norm(v)

def emb(img):
    img.thumbnail((400, 400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return post({"content": {"prompt_string": "<__media__>",
                             "multimodal_data": [base64.b64encode(buf.getvalue()).decode()]}})

pos_files = sorted(glob.glob("tut_frames/*.png"))
neg_files = ["frames/f000.png", "frames/f014.png", "walk_frames/w000.png",
             "walk_frames/w001_e0.png", "after_click.png", "hybrid_frame.png"] + sorted(glob.glob("auto_frames/*.png"))[:4]
cache = {}
for rname, box in REGIONS.items():
    for f in pos_files + neg_files:
        img = Image.open(f)
        if box:
            W, H = img.size
            img = img.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
        cache[(rname, f)] = emb(img)
    print(f"embedded region {rname}", flush=True)

for rname in REGIONS:
    P = [cache[(rname, f)] for f in pos_files]
    N = [cache[(rname, f)] for f in neg_files]
    def centroid(vs):
        c = np.mean(vs, axis=0)
        return c / np.linalg.norm(c)
    ok = tot = 0
    for i in range(len(P)):
        cp = centroid(P[:i] + P[i + 1:]); cn = centroid(N)
        ok += float(P[i] @ cp) > float(P[i] @ cn); tot += 1
    for i in range(len(N)):
        cn = centroid(N[:i] + N[i + 1:]); cp = centroid(P)
        ok += float(N[i] @ cn) > float(N[i] @ cp); tot += 1
    cp, cn = centroid(P), centroid(N)
    print(f"{rname}: LOO {ok}/{tot}, centroid cos={float(cp @ cn):.3f}", flush=True)
