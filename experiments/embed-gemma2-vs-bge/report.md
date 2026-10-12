# Embedding Gemma 2 vs BGE-M3 for Engram Core

Date: 2026-10-11. Corpus: 30 engram-like episodes (24 topical + 3 distractors + 3 config/code),
12 queries with graded relevant sets. Driver: `run.py`. Full numbers: `metrics.json`.

## Model facts (researched)

- Embedding Gemma 2: `google/embeddinggemma-2`, Apache 2.0 ungated, 740M total
  (270M text-only path), native 768-dim with Matryoshka truncation to 512/256/128,
  8192 context, 100+ languages, text+image+video+audio in one space, task prompts
  required (`SearchQuery`/`Document`). Use bf16/fp32, never fp16 (NaN risk).
  Sources: https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/,
  https://huggingface.co/google/embeddinggemma-2
- Incumbent: `BAAI/bge-m3` Q8 GGUF (606MB) via llama-server `:8090`, pooling cls,
  1024-dim, 8192 context. See `positronic-agent-interface/docs/llama.md`.

## Results

| model | dim | R@1 | R@3 | R@5 | MRR | corpus enc | ms/query |
|---|---|---|---|---|---|---|---|
| BGE-M3 (Q8, llama-server/HIP) | 1024 | 0.917 | 1.000 | 1.000 | 0.958 | 1.0s | 27 |
| EmbeddingGemma-2 (fp32, torch CPU) | 768 | 1.000 | 1.000 | 1.000 | 1.000 | 23.1s | 2751 |

## Interpretation

1. Quality: EG2 is marginally better here (12/12 vs 11/12 top-1). Small corpus —
   treat as a signal, not a proof. Both are strong; neither fails.
2. Latency gap is a runtime confound, not a model property: BGE ran quantized on
   HIP via llama-server; EG2 ran fp32 torch on CPU (llama.cpp build predates the
   `gemma-embedding2` arch and refused the GGUF, so no quantized EG2 server test).
3. Footprint: EG2 text-only is 270M params / 768-dim (25% smaller vectors than
   BGE-M3's 1024-dim); with MRL-256 truncation it would use 4x less storage.

## Recommendation

Do not swap yet. EG2 looks like a genuine upgrade candidate (better small-test
recall, smaller vectors, Apache 2.0, code-retrieval gains +9.9 reported), but
adopting it needs: (a) a llama.cpp or Ollama build that serves the EG2 GGUF so
encode latency matches today's 27ms path, (b) a larger eval (200+ episodes,
real brain queries), (c) a migration plan for existing 1024-dim `body_embed`
blobs (re-embed or dual-index). Revisit when (a) lands.
