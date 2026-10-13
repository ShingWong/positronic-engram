"""Passive screen survey on bot17 (:26 SHM).
Every 10s: capture, identify() via game_world, save PNG, EG2-embed (400px).
Output: frames/*.png + survey.jsonl {t, screen, vec}.
Unknown/None screens get embedded for later clustering into new entries.
"""
import base64, io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")
from gt_spector import Session
from gt_spector.game_world import GameWorld

N, PERIOD = 60, 10

def embed_png(buf):
    b64 = base64.b64encode(buf).decode()
    req = urllib.request.Request(
        "http://127.0.0.1:8091/embedding",
        data=json.dumps({"content": {"prompt_string": "<__media__>",
                                      "multimodal_data": [b64]}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        e = json.load(r)[0]["embedding"]
    return e[0] if isinstance(e[0], list) else e

s = Session(":26", source="shm://26")
gw = GameWorld()
import os
os.makedirs("frames", exist_ok=True)
out = open("survey.jsonl", "w")
for i in range(N):
    s.refresh()
    frame = s.frame
    try:
        screen = gw.identify(s)
    except Exception as ex:
        screen = f"identify-error:{ex}"
    img = Image.fromarray(frame)
    img.save(f"frames/f{i:03d}.png")
    small = img.copy()
    small.thumbnail((400, 400))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    try:
        vec = embed_png(buf.getvalue())
        ok = True
    except Exception as ex:
        vec, ok = [], False
    out.write(json.dumps({"i": i, "t": time.time(), "screen": screen,
                          "ok": ok, "vec": vec}) + "\n")
    out.flush()
    print(f"{i}: screen={screen} vec={'ok' if ok else 'FAIL'}", flush=True)
    time.sleep(PERIOD)
print("done")
