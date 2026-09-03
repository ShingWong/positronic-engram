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

"""Vision gate tests — signature recognition lifecycle (H14 at perception layer)."""
import io

import pytest

pytest.importorskip("PIL")

from memeng.engine import MemoryEngine
from memeng.store import SQLiteStore
from memeng.vision import classify_image, dhash_hex, hamming
from PIL import Image, ImageDraw


def mk():
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s)
    e.init_database()
    return e, s


def png_bytes(w=300, h=100, draw_logo=False):
    img = Image.new("RGB", (w, h), (240, 240, 245))
    d = ImageDraw.Draw(img)
    if draw_logo:                       # distinctive mark → stable dhash bits
        d.ellipse([10, 10, 80, 80], fill=(200, 30, 30))
        d.rectangle([100, 40, 220, 60], fill=(30, 30, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# exact duplicate → recognized, seen_count accumulates
def test_exact_duplicate_skips():
    e, s = mk()
    b = png_bytes(draw_logo=True)
    r1 = classify_image(s, b, "p_0001")
    assert r1["verdict"].startswith("new-")
    r2 = classify_image(s, b, "p_0001")
    assert r2["verdict"] == "known"
    assert r2["action"] == "skip"
    assert r2["seen_count"] == 2


# resized + re-encoded variant → dHash catches it
def test_resized_variant_recognized():
    e, s = mk()
    orig = png_bytes(w=300, h=100, draw_logo=True)
    r1 = classify_image(s, orig, "p_0001")
    assert r1["verdict"] != "known"

    img = Image.open(io.BytesIO(orig)).resize((150, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")          # re-encoded smaller variant
    variant = buf.getvalue()

    r2 = classify_image(s, variant, "p_0001")
    assert r2["verdict"] == "variant-known"
    assert r2["action"] == "skip"


# genuinely different image → triaged as new content
def test_distinct_image_triages():
    e, s = mk()
    classify_image(s, png_bytes(draw_logo=True), "p_0001")
    other = png_bytes(w=640, h=480)      # large blank — different structure
    r = classify_image(s, other, "p_0002")
    assert r["verdict"] in ("new-content", "new-signature-like")
    if r["classification"] == "content":
        assert r["action"] == "triage"


# small images classify signature-like; action schema-reinforce not triage
def test_small_signature_classification():
    e, s = mk()
    tiny = png_bytes(w=120, h=40)
    r = classify_image(s, tiny, "p_0001")
    assert r["classification"] == "signature"
    assert r["action"] == "schema-reinforce"


def test_repetition_promotes_content_to_signature():
    e, s = mk()
    big = png_bytes(w=640, h=480, draw_logo=True)   # large → triaged once
    r1 = classify_image(s, big, "p_0001")
    assert r1["classification"] == "content"
    assert r1["action"] == "triage"                 # analyzed exactly once
    r2 = classify_image(s, big, "p_0002")           # repeats from another sender
    assert r2["verdict"] == "known"
    assert r2["classification"] == "signature"      # promoted: repeated branding
    assert r2["action"] == "skip"


# non-image bytes rejected cleanly
def test_not_image():
    e, s = mk()
    r = classify_image(s, b"this is definitely not an image " * 10)
    assert r["verdict"] == "not-image"
    assert r["action"] == "ignore"


# dhash helpers sane
def test_dhash_hamming_basics():
    h1 = dhash_hex(png_bytes(draw_logo=True))
    assert h1 is not None and len(h1) == 16
    assert hamming(h1, h1) == 0
