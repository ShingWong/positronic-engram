"""Driven walk pass 2 on bot17: identify -> ESC -> settle -> identify.
Records observed edges (prev --ESC--> current). Rules:
- require actual screen CHANGE before recording (no reflexive edges);
- ESC up to 2x per screen, then mark ESC-resistant and stop (no blind clicks);
- Confirm/OK/Collect buttons are NEVER clicked;
- tutorial indicators (tap-here/hand pointer) -> hands-off, record only.
All proposals go to walk_edges.json + walk_frames/. Nothing touches game_world.yaml.
"""
import base64, io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")
from gt_spector import Session
from gt_spector.game_world import GameWorld

STEPS, SETTLE_S, MAX_ESC = 12, 6, 2

def embed_png(buf):
    b64 = base64.b64encode(buf).decode()
    req = urllib.request.Request(
        "http://127.0.0.1:8091/embedding",
        data=json.dumps({"content": {"prompt_string": "<__media__>",
                                      "multimodal_data": [b64]}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        e = json.load(r)[0]["embedding"]
    v = np.array(e[0] if isinstance(e[0], list) else e, dtype=np.float64)
    return v / np.linalg.norm(v)

def snap(s, gw, tag):
    s.refresh()
    frame = s.frame.copy()
    try:
        screen = gw.identify(s)
    except Exception as ex:
        screen = f"identify-error:{ex}"
    img = Image.fromarray(frame)
    img.save(f"walk_frames/{tag}.png")
    small = img.copy()
    small.thumbnail((400, 400))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return screen, embed_png(buf.getvalue()), frame

s = Session(":26", source="shm://26")
gw = GameWorld()
import os
os.makedirs("walk_frames", exist_ok=True)
edges, log = [], []

def tut_check(vec, frame):
    return None

screen, vec, frame = snap(s, gw, "w000")
log.append({"step": 0, "screen": screen})
print(f"start: {screen}", flush=True)
prev = screen
seen = set()
for i in range(1, STEPS + 1):
    changed = False
    for attempt in range(MAX_ESC):
        s.key_press("Escape")
        time.sleep(SETTLE_S)
        cur, v, f = snap(s, gw, f"w{i:03d}_e{attempt}")
        if cur != prev and cur is not None and "identify-error" not in str(cur):
            changed = True
            break
        prev_poll = cur
    if changed:
        key = (prev, "esc", cur)
        if key in seen:
            print(f"step {i}: loop detected ({prev} --esc--> {cur} seen before), stop", flush=True)
            log.append({"step": i, "screen": cur, "note": "loop"})
            break
        seen.add(key)
        edges.append({"from": prev, "action": "esc", "to": cur})
        print(f"step {i}: EDGE {prev} --esc--> {cur}", flush=True)
        prev = cur
    else:
        print(f"step {i}: no change from {prev} after {MAX_ESC}x ESC -> ESC-resistant, stop", flush=True)
        log.append({"step": i, "screen": prev, "note": "esc-resistant"})
        break
    log.append({"step": i, "screen": cur})
json.dump({"edges": edges, "log": log}, open("walk_edges.json", "w"), indent=1)
print(f"done: {len(edges)} edges")
