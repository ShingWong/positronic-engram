# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Vision gate — perceptual recognition so the retina tier never re-analyzes
the same signature twice (user insight: 'small images are just signatures').

Two-tier hashing:
  bhash  sha256(bytes)          exact duplicates (forwarded signature files)
  phash  dHash 64-bit           near-duplicates (resized/re-encoded variants)

Classification (single repeated-branding class):
  'signature' — recurring imagery: personal signatures AND corporate logos.
                 Repetition defines the class, not size. Small images are
                 provisionally signature at first sight; anything repeated
                 (exact or perceptual-variant) is promoted to signature
                 permanently. Non-repeating large images are 'content' and
                 get exactly one VLM triage.
Recognition accumulates corpus-wide in `image_registry`.
"""
from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash_hex(img_bytes: bytes, size: int = 8) -> str | None:
    """Difference hash: grayscale, resize to (size+1, size), compare adjacent.
    Returns 16-hex-char string (64 bits) or None if not decodable as image."""
    try:
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("L").resize((size + 1, size))
        px = list(img.getdata())
        bits = []
        for row in range(size):
            for col in range(size):
                left = px[row * (size + 1) + col]
                right = px[row * (size + 1) + col + 1]
                bits.append("1" if left > right else "0")
        return f"{int(''.join(bits), 2):0{size * size // 4}x}"
    except (OSError, ValueError, TypeError):
        return None


def hamming(hex_a: str, hex_b: str) -> int:
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 64
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def classify_image(store, img_bytes: bytes,
                   sender_pid: str | None = None) -> dict:
    """Recognize or register an image. Returns decision dict; callers act:
       action='skip'            -> already recognized, schema-reinforce only
       action='triage'          -> novel content; escalate to VLM analysis
       action='schema-reinforce'-> new but trivially small (signature-like)
    """
    bhash = sha256_hex(img_bytes)
    known = store.get_image_by_bhash(bhash)
    if known:
        store.bump_image_seen(known["bhash"])
        # repetition promotes ANY image to the signature class permanently
        if known["classification"] != "signature" \
                and known["seen_count"] + 1 >= 2:
            store.set_image_classification(bhash, "signature")
            known["classification"] = "signature"
        return {"verdict": "known", "action": "skip",
                "classification": known["classification"],
                "bhash": bhash, "phash": known["phash"],
                "seen_count": known["seen_count"] + 1}

    phash = dhash_hex(img_bytes)
    w = h = None
    try:
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
    except (OSError, ValueError):
        pass  # non-decodable -> phash is None below -> "not-image" verdict

    if phash is None:
        return {"verdict": "not-image", "action": "ignore",
                "bhash": bhash, "phash": None}

    # near-duplicate search over registered phashes
    variant_of = None
    for row in store.iter_image_phashes():
        dist = hamming(phash, row["phash"] or "")
        if dist <= 6:
            variant_of = row
            break

    small = (w is not None and max(w, h) <= 350 and len(img_bytes) <= 120_000)

    if variant_of:
        store.bump_image_seen(variant_of["bhash"])
        store.set_image_classification(variant_of["bhash"], "signature")
        return {"verdict": "variant-known", "action": "skip",
                "classification": "signature", "bhash": bhash,
                "phash": phash, "variant_distance": dist,
                "seen_count": variant_of["seen_count"] + 1}

    classification = "signature" if small else "content"
    store.register_image(bhash=bhash, phash=phash, width=w, height=h,
                         nbytes=len(img_bytes), sender_pid=sender_pid,
                         classification=classification)
    return {"verdict": f"new-{classification}",
            "classification": classification,
            "action":
            ("triage" if classification == "content" else "schema-reinforce"),
            "bhash": bhash, "phash": phash, "width": w, "height": h}
