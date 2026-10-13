"""Prototype: bright-banner contour detector for tutorial overlays.
Tutorial instruction banners are light/cream rounded bars; dialogs/panels
are dark. Feature: largest bright quad-like contour (area fraction, aspect,
mean brightness). Reports separation on banked frames.
"""
import glob
import cv2
import numpy as np

def features(path):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    _, th = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        frac = area / (W * H)
        if frac < 0.01 or frac > 0.6:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        bright = cv2.mean(v, mask=mask)[0]
        quad = len(approx) <= 8
        score = (frac, bright if quad else 0)
        if best is None or score > best[0]:
            best = (score, {"frac": round(frac, 3), "aspect": round(aspect, 2),
                            "bright": round(bright, 1), "quad": quad,
                            "rect": (x, y, w, h)})
    return best[1] if best else None

pos = sorted(glob.glob("tut_frames/*.png"))
neg = ["frames/f000.png", "frames/f014.png", "walk_frames/w000.png",
       "walk_frames/w001_e0.png", "after_click.png", "hybrid_frame.png"] + sorted(glob.glob("auto_frames/*.png"))[:4]
print("=== tutorial ===")
for f in pos:
    print(f.split("/")[-1], features(f))
print("=== normal ===")
for f in neg:
    print(f.split("/")[-1], features(f))
