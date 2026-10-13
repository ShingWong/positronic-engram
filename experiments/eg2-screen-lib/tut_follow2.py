"""Tutorial follower v2: full-frame TAP proposal -> crop-zoom confirm ->
tap -> verify. Fixes v1's fraction-grounding failure (hit SKIP twice).
Usage: python3 tut_follow2.py :25
"""
import base64, io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")
from gt_spector import Session
from gt_spector.game_world import GameWorld

DISP = sys.argv[1] if len(sys.argv) > 1 else ":26"
MAX_ITERS = 15
import re
TAP_RE = re.compile(r"TAP\s+(\d+)\s+(\d+)", re.IGNORECASE)

def chat(img, ask, mt=500):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"model": "qwen", "messages": [{"role": "user", "content": [
        {"type": "text", "text": ask},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

s = Session(DISP, source=f"shm://{DISP.lstrip(':')}")
gw = GameWorld()
log = []
for i in range(MAX_ITERS):
    s.refresh()
    img = Image.fromarray(s.frame)
    img.save(f"tut2_{DISP.strip(':')}_{i:02d}.png")
    try:
        screen = gw.identify(s)
    except Exception as ex:
        screen = f"err:{ex}"
    W, H = img.size
    a1 = chat(img, ("Tutorial active. Which ONE element to tap next? Reply TAP x y REASON "
                    "with 0-1000 fractions. Never SKIP/Cancel/exit. If no tutorial visible, reply DONE."))
    log.append({"i": i, "screen": screen, "ask1": a1})
    print(f"iter {i} screen={screen}\n  {a1[:200]}", flush=True)
    if a1.strip().upper().startswith("DONE"):
        print("tutorial clear", flush=True)
        break
    m = TAP_RE.search(a1)
    if not m:
        print("no TAP, stop", flush=True)
        break
    cx, cy = int(m.group(1)) / 1000 * W, int(m.group(2)) / 1000 * H
    x0, y0 = max(0, int(cx - 200)), max(0, int(cy - 150))
    crop = img.crop((x0, y0, min(W, x0 + 400), min(H, y0 + 300)))
    cw, ch = crop.size
    a2 = chat(crop, ("Confirm tap target inside THIS crop for the game tutorial. Reply TAP x y "
                     "with 0-1000 fractions inside this crop, or NOTFOUND. Never SKIP."))
    m2 = TAP_RE.search(a2)
    log.append({"i": i, "confirm": a2[:200]})
    if not m2:
        print("  crop confirm failed, stop", flush=True)
        break
    x = x0 + int(m2.group(1)) / 1000 * cw
    y = y0 + int(m2.group(2)) / 1000 * ch
    print(f"  tap {(int(x), int(y))} (crop @ {(x0, y0)})", flush=True)
    s.click(int(x), int(y))
    time.sleep(7)
json.dump(log, open(f"tut2_{DISP.strip(':')}_log.json", "w"), indent=1)
print("log written")
