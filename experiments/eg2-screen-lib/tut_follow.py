"""Tutorial follower: capture -> Qwen (where to tap, generous budget) ->
tap -> verify. Stops when banner gone / screen leaves tutorial / N iters.
Logs every step with frames. Disposable account: free clicking.
"""
import base64, io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")
from gt_spector import Session
from gt_spector.game_world import GameWorld

MAX_ITERS = 15

def shot(s):
    s.refresh()
    return Image.fromarray(s.frame)

def ask_qwen(img, ask):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"model": "qwen", "messages": [{"role": "user", "content": [
        {"type": "text", "text": ask},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": 500, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

s = Session(":26", source="shm://26")
gw = GameWorld()
import os
os.makedirs("tut_frames", exist_ok=True)
log = []
for i in range(MAX_ITERS):
    img = shot(s)
    img.save(f"tut_frames/t{i:02d}.png")
    try:
        screen = gw.identify(s)
    except Exception as ex:
        screen = f"err:{ex}"
    W, H = img.size
    ans = ask_qwen(img, (
        "You are driving a game tutorial. Look at this screenshot. "
        "If a tutorial instruction/banner is visible, tell me the ONE UI element to tap next "
        "as: TAP x y REASON, where x,y are 0-1000 fractions of width/height, then one line why. "
        "Never choose SKIP/Cancel/exit unless nothing else can advance. "
        "If no tutorial is visible, reply DONE."))
    log.append({"i": i, "screen": screen, "qwen": ans})
    print(f"iter {i} screen={screen}\nQWEN: {ans}", flush=True)
    if ans.strip().upper().startswith("DONE"):
        print("tutorial clear", flush=True)
        break
    x = y = None
    for tok in ans.replace(",", " ").split():
        pass
    import re
    m = re.search(r"TAP\s+(\d+)\s+(\d+)", ans)
    if not m:
        print("no TAP coords, stop", flush=True)
        break
    x, y = int(m.group(1)) / 1000 * W, int(m.group(2)) / 1000 * H
    print(f"tap {(int(x), int(y))}", flush=True)
    s.click(int(x), int(y))
    time.sleep(7)
json.dump(log, open("tut_log.json", "w"), indent=1)
print("log written")
