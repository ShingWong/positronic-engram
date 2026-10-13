"""Split the afk_popup bucket: embed all 84 saved frames, greedy cluster at 0.97,
report distinct visual screens vs DB labels from survey/autodiscover/walker logs.
"""
import base64, glob, io, json
import numpy as np
from PIL import Image
import urllib.request

def embed(buf):
    req = urllib.request.Request(
        "http://127.0.0.1:8091/embedding",
        data=json.dumps({"content": {"prompt_string": "<__media__>",
                                      "multimodal_data": [base64.b64encode(buf).decode()]}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        e = json.load(r)[0]["embedding"]
    v = np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)
    return v / np.linalg.norm(v)

labels = {}
for row in open("survey.jsonl"):
    d = json.loads(row)
    labels[f"frames/f{d['i']:03d}.png"] = d["screen"]
n = 0
files = sorted(glob.glob("frames/*.png") + glob.glob("auto_frames/*.png") +
               glob.glob("walk_frames/*.png") + ["after_click.png", "hybrid_frame.png"])
vecs = {}
for f in files:
    img = Image.open(f)
    img.thumbnail((400, 400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    vecs[f] = embed(buf.getvalue())
    n += 1
    if n % 20 == 0:
        print(f"{n}/{len(files)}", flush=True)

clusters = []
for f, v in vecs.items():
    placed = False
    for cl in clusters:
        if max(float(v @ vecs[o]) for o in cl) > 0.97:
            cl.append(f)
            placed = True
            break
    if not placed:
        clusters.append([f])
for i, cl in enumerate(clusters):
    labs = sorted(set(labels.get(f, "?") for f in cl))
    print(f"cluster {i}: n={len(cl)} labels={labs} e.g. {cl[0]}")
json.dump({"clusters": clusters}, open("bucket_split.json", "w"), indent=1)
