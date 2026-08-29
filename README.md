# LLMem — Cognitive-Architecture-Inspired Memory for LLM Agents

> *"The brain doesn't remember everything. It remembers what mattered, distilled
> by specialized systems, consolidated during rest, and reconstructed — never
> replayed — on demand."*

## Thesis

Current agent-memory projects are **single-brain designs**: one frontier model
(or one embedding index) decides what to store, how to consolidate it, and what
to retrieve. Human memory works nothing like that. It is a federation of small
specialized subsystems — salience gating, episodic encoding, semantic
distillation, domain-specific cortices, and a slow consolidation cycle — none
of which is intelligent alone.

**Hypothesis:** a federation of *small, cheap, specialized* models managing
domain-partitioned memory will outperform single-model memory on relevance,
latency, cost, and scalability — because curation, not intelligence, is the
hard problem of memory.

## Why now

This project builds directly on the MI50 inference-rig work (`dls` repo):
we proved multiple quantized models can share one GPU efficiently
(FastMTP-class speculation makes sub-second responses from billion-parameter
curators routine). A 32 GB card can host the main agent *plus* an entire
memory-management staff simultaneously.

## Research tracks

| Track | Question | Directory |
|---|---|---|
| **Cognitive foundations** | How does human memory actually work? Encoding gates, systems consolidation, reconstructive retrieval, adaptive forgetting | `research/cognitive/` |
| **LLM landscape** | What exists (MemGPT/Letta, Mem0, Zep/Graphiti, HippoRAG, Cognee, A-MEM…), what each gets wrong | `research/llm-landscape/` |
| **Architecture** | The multi-specialist design: salience gate → domain curators → sleep consolidation → retrieval contract | `research/architecture/` |

## Seed architecture sketch

```
                    ┌────────────────────────┐
   agent turn ────▶ │  SALIENCE GATE (0.5–1.5B) │  store / discard / defer?
                    └───────────┬────────────┘
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │ CURATOR:     │  │ CURATOR:     │  │ CURATOR:     │
      │ project-state│  │ user-model   │  │ tech-facts   │    (3B-class each)
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             └─────────────────┼──────────────────┘
                               ▼
                    ┌────────────────────────┐
                    │ CONSOLIDATOR ("sleep") │  off-cycle:
                    │ distill episodic→      │  merge contradictions,
                    │ semantic; decay stale  │  promote/reinforce/discard
                    └────────────────────────┘
```

## Core research questions

1. **Taxonomy** — what domains partition memory? Fixed ontology vs learned boundaries? Overlap contracts between curators?
2. **Promotion** — when does an episode become semantic fact? (Repetition? Contradiction resolution? Age + access count?)
3. **Decay** — recency vs frequency vs importance weighting; what *should* be forgotten?
4. **Reconstruction** — human recall rebuilds rather than replays. What storage format supports faithful-enough reconstruction without hoarding transcripts?
5. **Curation reliability** — can 0.5–8B models gate/distill/consolidate at production quality? Benchmark design needed.
6. **Evaluation** — how do we measure memory quality at all? (Need task suites: contradiction detection, temporal reasoning, preference drift.)

## Method

Same loop that produced the MI50 rig: human judgment generates hypotheses and
steers; AI measures, implements, falsifies. Every claim lands in these notes
with evidence attached or marked as speculation.

## Status (updated 2026-08-25)

- [x] Project initiated, thesis drafted
- [x] Cognitive foundations seed docs (`research/cognitive/01–02`)
- [x] LLM memory landscape survey + gap analysis (`research/llm-landscape/`)
- [x] Architecture v0: three-tier cognition, episode schema, curator contracts
      (`research/architecture/`)
- [x] **Paper #1 scaffold**: *Temporal Perception in Artificial Intelligence*
      — outline, case corpus (C1–C8), hypotheses **H1–H18**, annotated
      bibliography (30+ verified sources), experiment designs E1–E6
      (`papers/temporal-perception-in-AI/`)
- [x] **Fuzzy-memory engine monograph** (paper #2 seed): primitives
      Activate·Reconstruct·Decay·Fuse formalized; 20 verified citations;
      τ-as-Lamport-clock result (`orchestration/TASK-001-fuzzy-memory-engine/`)
- [x] **MemoryEngine v0 implementation** (`engine/`): cognitive loop
      predict→gate→encode|reinforce; open-world person/domain registries;
      H14–H18 executable; trigger bus (12 types); per-stage telemetry;
      batch ingestion at **~6k events/s** (H14 gate economics verified:
      5% encode rate on realistic mix); 14 unit tests green
- [x] Orchestration protocol for research subagents
      (`orchestration/check.sh` + mission-folder pattern)
- [x] Infrastructure assessed: business mail archive (636 GB, 223 accounts,
      2016→2026) as long-timescale corpus; pilot scoped to one domain
      (8 accounts, ~27 GB, Sent 2007→2026); audio senses deployed on ai2;
      dev tooling stack on web2
- [ ] Phase-0 extractor: principal-account chronological reader with live
      gate review
- [ ] E1 headline experiment: τ-keyed vs wall-clock-keyed memory
- [ ] Privacy/spam policy finalization for mail pilot

## Hardware tiering (added 2026-08)

Sensory processing is tiered onto dedicated cheap silicon, mirroring
retina-before-cortex:

- **Tier 0 · retina**: OpenCV + Coral TPU (~5 W) — capture, diff,
  classify, embed. Always-on.
- **Tier 1 · attention**: salience gate on detected events only.
- **Tier 2 · cortex**: VLM distillation on MI50 for salient episodes.
- **Tier 3 · sleep**: consolidator, off-cycle.

See `research/architecture/hardware-tiering.md`.
