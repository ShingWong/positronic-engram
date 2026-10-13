"""Make a descriptive screen entry from a cluster: stable-pixel fingerprint,
EG2 reference vector, Qwen one-line description. Appends to new_screens.yaml
(same schema as game_world.yaml screens:). Never touches gt-spector tree."""
import base64, io, json, sys, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/swong/dls/src/gt-spector")

def qwen_describe(png_path):
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()
    payload = {"model": "qwen", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Name this game screen in 3 words snake_case, then one line saying what it is and its main button. Format: NAME | description"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": 80, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

data = json.load(open("bucket_split.json"))
clusters = data["clusters"]
# cluster 0 = AFK resources popup (after_click + kin)
cl = clusters[0]
print(f"cluster 0: {len(cl)} frames", flush=True)
arrs = [np.array(Image.open(f).convert("RGB")).astype(float) for f in cl]
arr = np.stack(arrs)
std = arr.std(axis=0).mean(axis=2)
H, W = std.shape
picks = []
for gy in range(6):
    for gx in range(8):
        cell = std[gy * H // 6:(gy + 1) * H // 6, gx * W // 8:(gx + 1) * W // 8]
        j, k = np.unravel_index(np.argmin(cell), cell.shape)
        y, x = gy * H // 6 + j, gx * W // 8 + k
        picks.append({"x": int(x), "y": int(y),
                      "rgb": [int(v) for v in arr[0, y, x, :3]],
                      "std": round(float(std[y, x]), 2)})
print("Qwen naming...", flush=True)
name_desc = qwen_describe(cl[0])
print("QWEN:", name_desc, flush=True)
entry = {
    "proposed_name": "afk_resources_popup",
    "qwen": name_desc,
    "source_frames": cl,
    "fingerprint_probes": picks,
    "advance_observed": "Confirm(741,676) -> shelter_view (DB claims world_map; flagged stale)",
}
json.dump(entry, open("entry_afk_resources.json", "w"), indent=1)
print("entry written")
