# EG2 cross-modal probe: text queries retrieve images (2026-10-11)

Server: `:8091`, EG2 Q8 + mmproj Q8, `LLAMA_MEDIA_MARKER='<__media__>'`.
Driver: `img_probe.py`. Images in this dir (3 PIL shapes + 1 natural photo).

## Result: 4/4 diagonal, raw and prompted

| query | red² | blue○ | green△ | cat photo |
|---|---|---|---|---|
| a red square | **0.749** | 0.624 | 0.613 | 0.613 |
| a blue circle | 0.586 | **0.722** | 0.564 | 0.525 |
| a green triangle | 0.590 | 0.569 | **0.764** | 0.591 |
| a cat | 0.600 | 0.596 | 0.596 | **0.688** |

Margins 0.09–0.16 over distractors. Prompted queries score slightly higher
absolute (0.78–0.81) with the same ranking. Works on synthetic shapes and a
natural photo.

## Serving contract (for the PAI client path)

- Shape: `{"content": {"prompt_string": "<text incl. marker>", "multimodal_data":
  ["<raw base64>"]}}`. Legacy `image_data` key is silently ignored (text-only
  vectors back, identical across images — a silent false negative).
  OAI content-array shape returns 400.
- `LLAMA_MEDIA_MARKER='<__media__>'` at server start; default is a random
  marker per boot and every image request 500s. Belongs in the systemd unit.
- Missing marker in `prompt_string` → 500 "Failed to tokenize prompt"
  (markers/bitmaps count mismatch). Also loud, also fine.

## Standing

Text verdict unchanged (BGE stays for text). This is EG2's open case working:
one space, text queries, image retrieval, no text bridge. Next decider is the
mail-attachment testbed once the PAI client path lands.

## Scanned documents (2026-10-12): 5/5

Three PIL-rendered doc pages with scan feel (slight rotation, noise):
invoice (Acme, $12,480, net 30), memo (elevator B out, north stairwell),
safety (fire exits, drill Thursday). Text queries against image embeddings:

| query | invoice | memo | safety |
|---|---|---|---|
| invoice amount due | **0.733** | 0.594 | 0.552 |
| elevator out of service | 0.495 | **0.689** | 0.547 |
| fire drill thursday | 0.530 | 0.610 | **0.736** |
| invoice acme payment terms | **0.785** | 0.561 | 0.509 |
| north stairwell after six | 0.554 | **0.697** | 0.666 |

The last row is the telling one: memo and safety both mention stairwells
(doc-doc cosine 0.816), yet the query still lands on the right page. The
vision encoder reads the words, not just the layout. Numbers in
`img_metrics.json` under `scanned_docs`.
