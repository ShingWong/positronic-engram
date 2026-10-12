# EG2 llama-server build + serving validation (Phase 1-2)

## Build

- Tree: `/home/swong/dls/.tmp/eg2-build`, fresh clone at `f2cc7282`
  (upstream master, first commit with `gemma-embedding2` support is `4fbc76dec`).
- Binary: `/home/swong/dls/.tmp/eg2-build/build-eg2/bin/llama-server`.
- Flags: `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx906`, `FA_ALL_QUANTS=OFF`,
  `KVARN=OFF`, `hipcc`, Release — same as production `build-hip`.
- Runtime: `GGML_CUDA_DISABLE_GRAPHS=1` (required on gfx906/ROCm 5.7).

### Experiment-tree-only compat patches (4 files, all portable upstream)

1. `ggml/src/ggml-hip/CMakeLists.txt` — HIP floor 6.1 → 5.7.
2. `ggml/src/ggml-cuda/allreduce.cu` — 5× explicit `, 0` flags on
   `cudaStreamWaitEvent` (HIP 5.7 lacks the 6.1 default arg).
3. `ggml/src/ggml-cuda/ggml-cuda.cu` — 2× same (concurrent-stream graph code).
4. `ggml/src/ggml-cuda/solve_tri.cu` — `(float * const *)` cast: HIP 5.7's
   `hipblasStrsmBatched` wants non-const batch pointers, 6.x relaxed it.

Prior art: `/usr/local/devel/llama.cpp` working tree already held fixes 1
(equivalent), 3, 4 against an older master — left untouched, approach reused.

## Serving (Phase 2 gate: PASS)

- Server: `:8091`, `--embedding --pooling mean -c 8192`, Q8 GGUF at
  `/home/swong/dls/.tmp/eg2-build/models/embeddinggemma-2-Q8_0.gguf` (310 MB).
  (`/usr/local/devel/models/embedding/` is root-owned; co-location needs sudo.)
- Pooling: `cls` FAILS (cosine 0.65-0.71 vs fp32); `mean` PASSES (0.9998-0.9999
  on all 5 probes, raw and prompted). Model default is mean — matches.
- Task prompts explain the residual: server(raw) vs ST(Document) scored only
  0.90-0.93; with matched prompts both sides agree to 0.9998. F3 confirmed
  empirically: prompts are mandatory, not optional.
- Latency: 20-40 ms/query on HIP — at parity with BGE-M3's 27 ms.
- Dim 768 confirmed. Gate script: `check_pooling.py` in this dir.

## Next (Phase 3)

Prompts A/B on the 30-doc corpus, then 1,000-record pilot, per the build plan.
Pre-registered swap criterion still owed before any model claim.

## Pilot results (2026-10-11, updated)

1,000 synthetic episodes, chunk + mean-pool per record, one HTTP POST per chunk:

| backend | rec/s | projected 189k |
|---|---|---|
| BGE `:8090` | 8.6 | 6.1 h |
| EG2 `:8091` (Document prompt per chunk) | 9.5 | 5.5 h |

EG2 embeds slightly faster than BGE. But both projections are worst-case: the
real 2026-10-11 ingest ran at 106 records/s (189,401 in 1,776 s), ~12x faster,
because most CVE records skip embedding (short/duplicate) or batch. The pilot
embeds every record individually, so ~6 h is a ceiling, not an estimate.
Criterion (b) is recalibrated: measure a real 1,000-record CVE sample through
the actual PAI ingest path before projecting the full run — the synthetic
number cannot gate the decision.
