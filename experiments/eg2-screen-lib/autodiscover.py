"""Autodiscovery pass 1 (passive): capture bot17 repeatedly, identify + EG2-embed.
For screens identify() misses: cluster by vector, derive stable-pixel
fingerprint probes (pixels with ~zero variance across captures of the same
screen), and emit proposed game_world.yaml screen entries. No game input.
"""
import base64, io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")
from gt_spector import Session
from gt_spector.game_world import GameWorld

N, PERIOD = 8, 30

def embed_png(buf):
    b64 = base64.b64encode(buf).decode()
    req = urllib.request.Request(
        "http://127.0.0.1:8091/embedding",
        data=json.dumps({"content": {"prompt_string": "<__media__>",
                                      "multimodal_data": [b64]}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        e = json.load(r)[0]["embedding"]
    return np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)

s = Session(":26", source="shm://26")
gw = GameWorld()
import os
os.makedirs("auto_frames", exist_ok=True)
caps = []
for i in range(N):
    s.refresh()
    frame = s.frame.copy()
    try:
        screen = gw.identify(s)
    except Exception as ex:
        screen = f"identify-error:{ex}"
    Image.fromarray(frame).save(f"auto_frames/a{i:02d}.png")
    small = Image.fromarray(frame)
    small.thumbnail((400, 400))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    vec = embed_png(buf.getvalue())
    caps.append({"i": i, "screen": screen, "vec": vec, "frame": frame})
    print(f"{i}: screen={screen}", flush=True)
    time.sleep(PERIOD)

# cluster unknowns by cosine (threshold 0.97, greedy)
unk = [c for c in caps if not c["screen"] or "unknown" in str(c["screen"]).lower()]
print(f"unknown captures: {len(unk)}/{len(caps)}")
clusters = []
for c in unk:
    placed = False
    for cl in clusters:
        sims = [float(c["vec"] @ o["vec"] / (np.linalg.norm(c["vec"]) * np.linalg.norm(o["vec"]))) for o in cl]
        if max(sims) > 0.97:
            cl.append(c)
            placed = True
            break
    if not placed:
        clusters.append([c])
print(f"unknown clusters: {len(clusters)} sizes={[len(c) for c in clusters]}")

# stable-pixel fingerprint per cluster with >=2 captures
out = []
for ci, cl in enumerate(clusters):
    if len(cl) < 2:
        out.append({"cluster": ci, "n": 1, "note": "single capture, no fingerprint yet"})
        continue
    arr = np.stack([c["frame"].astype(float) for c in cl])
    std = arr.std(axis=0).mean(axis=2)
    ys, xs = np.where(std < 3.0)
    # spread picks across the frame: grid-sample the most stable
    picks = []
    H, W = std.shape
    for gy in range(6):
        for gx in range(8):
            cell = std[gy * H // 6:(gy + 1) * H // 6, gx * W // 8:(gx + 1) * W // 8]
            j, k = np.unravel_index(np.argmin(cell), cell.shape)
            y, x = gy * H // 6 + j, gx * W // 8 + k
            col = tuple(int(v) for v in arr[0, y, x, :3])
            picks.append({"x": int(x), "y": int(y), "rgb": col,
                          "std": round(float(std[y, x]), 2)})
    out.append({"cluster": ci, "n": len(cl), "fingerprint_probes": picks})
json.dump(out, open("proposals.json", "w"), indent=1)
print("proposals written")
