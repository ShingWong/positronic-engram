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

"""MemoryEngine — the cognitive loop. Open-world: persons/domains register at runtime.

v0 perf: per-stream caches + batch commits; per-stage telemetry so bottlenecks
name themselves (see telemetry.py).
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from .entities import extract_entities
from .fuzzy import FlatVectorIndex, rrf_fuse
from .models import Event, EventResult, GateVerdict, Prediction, PruneReport, Tier
from .store import SQLiteStore
from .telemetry import Telemetry
from .triggers import TriggerBus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryEngine:
    # FTS5 OR-term bag excludes high-frequency noise words so a sentence-level
    # cue's OR retry ranks on content terms, not filler.
    _FTS_STOP = frozenset(
        ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"])

    def __init__(self, store: SQLiteStore, config: dict | None = None) -> None:
        self.store = store
        cfg = {
            "threshold": 0.55,
            "burst_threshold": 0.90,      # H16 escalation
            "w_novelty": 0.5,
            "w_arousal": 0.3,
            "w_gain": 0.2,
            "anchor_salience": 0.85,
            "tau_per_surprise": 1.0,      # H14/H15: tau advances on surprise only
            "induce_after": 3,            # H17: repeats needed to distill a rule
            "induce_top_assoc_min": 2,
            "entity_extraction": True,
            "min_semantic_sim": 0.35,
        }
        cfg.update(config or {})
        self.base_cfg = cfg
        self.tel = Telemetry()

        # ── retention profiles: application-controlled forgetting ─────────
        # Same brain hardware; the POLICY differs per deployment/domain.
        self.retention_profiles = {
            "balanced":   {"S_base": 30.0,  "S_arousal": 40.0,
                           "prune_merge": 0.35, "prune_expire": 0.05,
                           "obj_dormant": 200.0, "obj_forget": 1200.0},
            "archival":   {"S_base": 1e6,   "S_arousal": 0.0,
                           "prune_merge": None, "prune_expire": None,
                           "obj_dormant": None, "obj_forget": None},
            "long_term":  {"S_base": 120.0, "S_arousal": 40.0,
                           "prune_merge": 0.20, "prune_expire": 0.02,
                           "obj_dormant": 600.0, "obj_forget": 5000.0},
            "short_term": {"S_base": 6.0,   "S_arousal": 4.0,
                           "prune_merge": 0.20, "prune_expire": 0.02,
                           "obj_dormant": 25.0, "obj_forget": 150.0},
        }
        self.vec_index = FlatVectorIndex()
        for eid, vec in store.iter_embeddings():
            self.vec_index.add(eid, vec)

    # -- foundation ---------------------------------------------------------
    def init_database(self) -> None:
        self.store.ensure_default_domain()
        self.store.conn.commit()

    def register_domain(self, name: str, *, threshold: float | None = None,
                        burst_threshold: float | None = None,
                        retention_profile: str | None = None) -> int:
        if retention_profile and retention_profile not in self.retention_profiles:
            raise ValueError(f"unknown retention profile: {retention_profile}")
        did = self.store.register_domain(name, threshold, burst_threshold)
        if retention_profile:
            self.store.set_domain_retention(did, retention_profile)
        return did

    def attach_stream(self, stream: str, domain: str) -> int:
        did = self.store.register_domain(domain)
        self.store.attach_stream(stream, did)
        return did

    def register_person(self, pid: str, *, key_ref: str | None = None) -> dict:
        return self.store.register_person(pid, auto=False, tau=0.0,
                                          wall=utcnow(), key_ref=key_ref)

    # -- resolution -----------------------------------------------------------
    def _resolve(self, event: Event) -> tuple[dict, float, int]:
        """Returns (effective_cfg, tau, mono); domain overrides merged."""
        srow = self.store.get_stream(event.stream)
        if not srow:
            did = self.store.ensure_default_domain()
            self.store.attach_stream(event.stream, did)
            srow = self.store.get_stream(event.stream) or {
                "stream": event.stream, "domain_id": did,
                "tau": 0.0, "mono": 0}
        dom = self.store.get_domain(int(srow["domain_id"])) or {}
        cfg = dict(self.base_cfg)
        dom_cfg = {k: v for k, v in dom.items()
                   if v is not None and k not in ("id","name","created_wall")}
        for k, v in dom_cfg.items():
            if k in cfg:
                cfg[k] = v
        if dom.get("threshold") is not None:
            cfg["threshold"] = float(dom["threshold"])
        if dom.get("burst_threshold") is not None:
            cfg["burst_threshold"] = float(dom["burst_threshold"])
        if dom.get("retention_profile"):
            cfg["retention_profile"] = dom["retention_profile"]
        return cfg, float(srow["tau"]), int(srow["mono"])

    # -- the cognitive loop -----------------------------------------------------
    def new_event(self, event: Event, *, store: bool = True,
                  handlers: dict | None = None) -> EventResult:
        t0 = time.perf_counter()
        bus = TriggerBus()
        self.tel.incr("events")

        with self.tel.stage("resolve"):
            cfg, tau_stream, mono = self._resolve(event)
        srow = self.store.get_stream(event.stream) or {}
        domain_id = int(srow.get("domain_id") or
                        self.store.ensure_default_domain())

        # persons auto-register on first encounter (open world)
        with self.tel.stage("persons"):
            for pid in [p for p in event.persons if p]:
                if self.store.get_person(pid) is None:
                    self.store.register_person(pid, auto=True, tau=tau_stream,
                                               wall=event.wall)
                    bus.emit("person_registered", tau_stream, pid=pid)

        # ---- PREDICT (H17: schemas run forward) --------------------------
        with self.tel.stage("predict"):
            predictions: list[Prediction] = []
            for rule in self.store.active_rules(domain_id):
                ak, av = rule["antecedent_key"], rule["antecedent_val"]
                actual_kv = event.features.get(ak)
                if actual_key_val_ok(actual_kv, av):
                    p = Prediction(source=str(rule["id"]),
                                   feature=rule["outcome_key"],
                                   expected=rule["outcome_val"])
                    p.actual = (str(event.features[rule["outcome_key"]])
                                if rule["outcome_key"] in event.features
                                else None)
                    p.matched = (p.actual == str(p.expected))
                    predictions.append(p)
        self.tel.incr("predictions", len(predictions))
        bus.emit("prediction_made", tau_stream, n=len(predictions))

        # ---- GATE (H14 novelty / H18 gain / H16 arousal) ------------------
        with self.tel.stage("gate"):
            violations = [p for p in predictions if p.matched is False]
            if violations:
                novelty = 1.0
            elif not predictions:
                novelty = 0.9
            elif all(p.matched for p in predictions):
                novelty = 0.05
            else:
                novelty = 0.6

            gains = [float((self.store.get_person(p) or
                            {"weight": 0.5})["weight"])
                     for p in event.persons]
            gain = max(gains, default=0.5)

            arousal = float(event.features.get("arousal", 0.0))
            score = min(1.0, cfg["w_novelty"] * novelty
                        + cfg["w_arousal"] * arousal
                        + cfg["w_gain"] * gain)
            encoded = (score >= cfg["threshold"]
                       or arousal >= cfg["burst_threshold"])

            verdict = GateVerdict(score=round(score, 4), novelty=novelty,
                                  arousal=arousal, gain=gain,
                                  threshold=float(cfg["threshold"]),
                                  encoded=bool(encoded))
            prof = self.retention_profiles[
                cfg.get("retention_profile", "balanced")]

        for p in violations:
            bus.emit("prediction_violation", tau_stream, source=p.source,
                     expected=p.expected, actual=p.actual)

        episode_id = None
        tier = Tier.NORMAL

        # tau advances on surprise only (H14): uneventful ≈ zero
        tau_advance = cfg["tau_per_surprise"] * novelty
        new_tau = tau_stream + tau_advance

        # ---- ENCODE --------------------------------------------------------
        if encoded:
            salience = score
            burst = arousal >= cfg["burst_threshold"]
            if burst:
                bus.emit("escalation_burst", new_tau,
                         window="until-arousal-subsides")
                tier = Tier.ESCALATED
            strength = (math.inf if tier is Tier.FLASHBULB else
                        prof["S_base"] + prof["S_arousal"] * arousal)

        if encoded and store:
            with self.tel.stage("encode"):
                from .models import EpisodeRecord, Provenance
                is_anchor = salience >= cfg["anchor_salience"]
                fuzz_lo = fuzz_hi = None
                precision = "exact"
                if event.fuzz_width is not None:
                    fuzz_lo = event.wall - event.fuzz_width
                    fuzz_hi = event.wall + event.fuzz_width
                    precision = "phase_of_day"
                rec = EpisodeRecord(
                    id=uuid.uuid4(), stream=event.stream, kind=event.kind,
                    wall=event.wall, mono=mono, tau=new_tau,
                    persons=list(event.persons),
                    subject_norm=event.features.get("subject_norm"),
                    salience=salience, tier=tier, strength=strength,
                    provenance=Provenance.WITNESSED, fuzz_lo=fuzz_lo,
                    fuzz_hi=fuzz_hi, precision_src=precision,
                    is_anchor=is_anchor, features=dict(event.features))
                self.store.insert_episode(rec, domain_id)
                episode_id = rec.id
                verdict.encoded = True
                self.tel.incr("encoded")
                # lexical + semantic indexing (FTS5 + optional embedder)
                bt = ((event.features.get("subject_norm") or "") + " " +
                      (event.features.get("body_text") or "")).strip()
                if bt:
                    self.store.fts_upsert(str(rec.id), bt)
                    emb = getattr(self, "_embedder", None)
                    if emb:
                        try:
                            vec = emb(bt)
                            self.store.set_embedding(str(rec.id), vec)
                            self.vec_index.add(str(rec.id), vec)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("embedding failed (best-effort): %s", e)
                if cfg.get("entity_extraction", True):
                    try:
                        etext = " ".join(
                            x for x in (event.features.get("subject_norm"),
                                        event.features.get("body_text"))
                            if x)
                        for name in sorted(extract_entities(etext)):
                            oid_ = self.store.get_or_create_object(
                                domain_id=domain_id, kind="entity",
                                canonical_name=name,
                                wall=event.wall, tau=new_tau)
                            self.store.link_sighting(
                                str(rec.id), oid_, channel="text")
                            self.store.touch_object(
                                oid_, new_tau, event.wall)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("entity extraction failed (best-effort): %s", e)
            bus.emit("episode_encoded", new_tau, episode_id=str(rec.id),
                     level="event", tier=tier.value)
            if is_anchor:
                bus.emit("anchor_detected", new_tau, anchor_id=str(rec.id))
        else:
            tier = Tier.NORMAL

        # ---- REINFORCE (below gate OR observe-mode) --------------------------
        if not (encoded and store):
            with self.tel.stage("reinforce"):
                for pid in event.persons:
                    self.store.bump_schema_stat(event.stream, f"sender:{pid}")
                self.store.bump_schema_stat(
                    event.stream, "cadence:" + event.wall.strftime("%H"))
                self.tel.incr("reinforced")

            bus.emit("schema_reinforced", tau_stream)
        # ---- H17 INDUCER: repetition -> causal-rule proposal -------
        subj = event.features.get("subject_norm")
        sender = (event.features.get("sender")
                  or (event.persons[0] if event.persons else None))
        if subj and sender:
            pair = f"subject_norm={subj}"
            count, top_assoc, assoc_n =                         self.store.bump_cooccurrence(
                    event.stream, pair, f"sender={sender}")
            if (count >= cfg["induce_after"]
                    and assoc_n >= cfg["induce_top_assoc_min"]):
                rid = "ind-" + uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    event.stream + "|" + pair).hex[:10]
                existed = any(r["id"] == rid for r in
                              self.store.active_rules(domain_id))
                self.store.upsert_rule(
                    domain_id, key="subject_norm", val=subj,
                    outcome_key="sender",
                    outcome_val=top_assoc.split("=", 1)[-1],
                    rule_id=rid)
                if not existed:
                    bus.emit("rule_proposed", new_tau, rule_id=rid,
                             pattern=pair, predicts=top_assoc,
                             support=count)

        # ---- H17: rules learn in BOTH paths -----------------------------------
        with self.tel.stage("rules"):
            for p in predictions:
                rid = str(p.source)
                graded = self.store.grade_rule(rid, matched=bool(p.matched))
                if p.matched:
                    bus.emit("rule_confirmed", tau_stream, rule_id=rid,
                             support=graded["support_count"])
                else:
                    bus.emit("rule_violated", tau_stream, rule_id=rid,
                             confidence=graded["confidence"])

        for pid in event.persons:
            self.store.touch_person(pid, new_tau)

        with self.tel.stage("commit"):
            bus.emit("gate_decision", new_tau, score=verdict.score,
                     encoded=verdict.encoded)
            self.store.set_stream_time(event.stream, new_tau, mono + 1)

        result = EventResult(verdict=verdict, tau=new_tau,
                             episode_id=episode_id, predictions=predictions,
                             trace=bus.trace)
        for t in result.trace:
            h = (handlers or {}).get(t.type)
            if h:
                h(t)
        self.tel.stages["total_ms"].append(
            (time.perf_counter() - t0) * 1000.0)
        return result

    # -- fuzzy recall (H5/H8: reconstruction channels) -----------------------
    def activate(self, cue: dict, k: int = 8,
                 consolidation: str | None = None,
                 context_window: int = 0) -> list[dict]:
        """Fused fuzzy recall. Cue channels (all optional):
          text:      -> lexical FTS5 + semantic embedding (when bound)
          persons:   -> person boost on fusion
          stream:    -> restrict candidates
          tau_now:   -> unused (kept for caller compat)
        Relevance channels are RRF-fused; the recency channel is a FALLBACK
        consulted only when both relevance channels come back empty.

        consolidation mode is a post-fuse view, not a fusion change:
          None           -> current behavior (freshness/score wins)
          'only'         -> keep consolidation episodes, drop the rest
          'first'        -> consolidations first (RRF order), then live, trim k

        context_window=N expands each hit's snippet to the ±N τ-adjacent
        episodes in the same stream, reuniting premise and answer messages
        that per-message chunking split apart (0 = unchanged).
        """
        with self.tel.stage("activate"):
            channels: list[list[str]] = []
            weights: list[float] = []
            fallback = False

            if cue.get("embedding"):
                # pick up embeddings persisted outside the encode path
                # (direct store writes / other processes) before searching
                for eid, vec in self.store.iter_embeddings():
                    self.vec_index.add(eid, vec)
                sim_floor = float(self.base_cfg["min_semantic_sim"])
                hits = [(eid, s) for eid, s in
                        self.vec_index.search(cue["embedding"], k=k * 3)
                        if s >= sim_floor]
                if hits:
                    channels.append([eid for eid, _ in hits])
                    weights.append(1.2)

            text = cue.get("text")
            if text:
                fts_ids = self.store.fts_search(text, k=k * 3)
                or_ids: list[str] = []
                if consolidation:
                    # consolidation mode searches directly among
                    # kind='consolidation' episodes with a term-bag OR, so the
                    # pool is exact — no guessing how deep a matching
                    # consolidation ranks against live chatter.
                    terms = [t for t in re.split(r"[^\w]+", text.lower())
                             if len(t) > 1 and t not in self._FTS_STOP]
                    if terms:
                        or_ids = self.store.fts_search(
                            " OR ".join(f'"{t}"' for t in terms),
                            k=k * 4, kind="consolidation")
                elif not fts_ids:
                    # FTS5 implicit-AND: every space-separated token must
                    # co-occur, so a sentence-level cue matches nothing and
                    # the recency fallback swallows the recall with self-echo.
                    # Retry with a term-bag OR so partial matches surface.
                    terms = [t for t in re.split(r"[^\w]+", text.lower())
                             if len(t) > 1 and t not in self._FTS_STOP]
                    if terms:
                        or_ids = self.store.fts_search(
                            " OR ".join(f'"{t}"' for t in terms), k=k * 3)
                if fts_ids:
                    channels.append(fts_ids)
                    weights.append(1.0)
                if or_ids:
                    channels.append(or_ids)
                    weights.append(1.0)

            if not channels:
                fallback = True
                rec_ids = self.store.recent_candidates(k * 3,
                                                       stream=cue.get("stream"))
                if rec_ids:
                    channels.append(rec_ids)
                    weights.append(1.0)

            eps_by_id: dict[str, Any] = {}
            for ch in channels:
                for eid in ch:
                    if eid not in eps_by_id:
                        full = self.store.get_episode(eid)
                        if full:
                            eps_by_id[eid] = full

            fused = rrf_fuse(channels, k=k if not consolidation else k * 4,
                         weights=weights)
            persons = set(cue.get("persons") or [])
            out = []
            for rank, eid in enumerate(fused, 1):
                ep = eps_by_id.get(eid)
                pboost = (1.3 if ep and persons & set(ep.persons or [])
                          else 1.0)
                base = sum(w / (60 + i + 1)
                           for ch, w in zip(channels, weights)
                           for i, x in enumerate(ch) if x == eid)
                feats = ep.features if ep else {}
                bt = feats.get("body_text") or ""
                out.append({
                    "episode_id": eid,
                    "rrf_score": round(base * pboost, 4),
                    "subject": ep.subject_norm if ep else "?",
                    "snippet": bt[:200],
                    "wall": ep.wall.isoformat() if ep else None,
                    "tau": ep.tau if ep else None,
                    "stream": ep.stream if ep else None,
                    "salience": ep.salience if ep else None,
                    "kind": ep.kind if ep else None,
                    "person_boost": pboost,
                    "fallback": fallback})
            out.sort(key=lambda d: -d["rrf_score"])
            if consolidation == "only":
                out = [d for d in out if d["kind"] == "consolidation"][:k]
            elif consolidation == "first":
                cons = [d for d in out if d["kind"] == "consolidation"]
                live = [d for d in out if d["kind"] != "consolidation"]
                out = (cons + live)[:k]
            if context_window and out:
                for d in out:
                    eid = d["episode_id"]
                    window = self.store.stream_neighbors(eid, context_window)
                    frags = []
                    seen_local: set[str] = set()
                    for wid in window:
                        if wid in seen_local:
                            continue
                        seen_local.add(wid)
                        wep = eps_by_id.get(wid) or self.store.get_episode(wid)
                        if wep:
                            frags.append(
                                (wep.features.get("body_text") or "")[:400])
                    d["snippet"] = "\n".join(
                        f for f in frags if f)[:3000]
            return out

    def bind_embedder(self, fn):
        """fn(text) -> list[float]. Called at encode time when configured."""
        self._embedder = fn

    # -- batch ingestion (single transaction, cache-warm) ----------------------
    def new_events(self, events: list[Event], *,
                   commit_every: int = 1000) -> list[EventResult]:
        out: list[EventResult] = []
        self.store.begin_batch()
        try:
            for i, ev in enumerate(events, 1):
                out.append(self.new_event(ev))
                if i % commit_every == 0:
                    self.store.commit_batch()
                    self.store.begin_batch()
            self.store.commit_batch()
        except Exception:
            self.store.commit_batch()   # flush what's done; research-mode semantics
            raise
        return out

    def observe(self, event: Event, *,
                handlers: dict | None = None) -> EventResult:
        return self.new_event(event, store=False, handlers=handlers)

    def telemetry_report(self) -> dict:
        return self.tel.report()

    def telemetry_line(self) -> str:
        return self.tel.summary_line()

    # -----------------------------------------------------------------------
    def prune(self, tau_now: float | None = None, *,
              domain: str | None = None) -> PruneReport:
        """Selective pruning. domain='kairos' scopes the pass to one domain —
        the foundation for per-origin memory hygiene."""
        rep = {"scanned": 0, "day_merged": 0, "week_merged": 0,
               "expired": 0, "residues": 0}
        eps = self.store.iter_episodes()
        did = self.store.get_domain_id(domain) if domain else None
        if domain and did is not None:
            eps = [e for e in eps if e.domain_id == did]
        rep["scanned"] = len(eps)
        dom_cache: dict[int, dict] = {}
        for ep in eps:
            now = tau_now if tau_now is not None else \
                self.store.stream_time(ep.stream)[0]
            if ep.tier is Tier.FLASHBULB or math.isinf(ep.strength):
                continue
            d_id = ep.domain_id if ep.domain_id is not None else \
                (self.store.get_stream(ep.stream) or {}).get("domain_id")
            if d_id not in dom_cache:
                row = self.store.get_domain(d_id) if d_id is not None else None
                pname = (row or {}).get("retention_profile") or \
                    self.base_cfg.get("retention_profile", "balanced")
                dom_cache[d_id] = self.retention_profiles[pname]
            P = dom_cache[d_id]
            p_merge, p_expire = P["prune_merge"], P["prune_expire"]
            if p_merge is None or p_expire is None:
                continue                        # archival: episodes immortal
            retain = math.exp(-((now - ep.tau) / max(ep.strength, 1e-9)))
            if retain < p_expire:
                self.store.write_residue(str(ep.id), usage_count=1,
                                         persons=ep.persons,
                                         anchor_edges=int(ep.is_anchor))
                self.store.update_episode(str(ep.id), level="week_token",
                                          subject_norm=None)
                rep["expired"] += 1
                rep["residues"] += 1
                rep["week_merged"] += 1
            elif retain < p_merge:
                self.store.update_episode(str(ep.id), level="day_token",
                                          subject_norm=None)
                rep["day_merged"] += 1

        # ---- H15 applied to OBJECTS: dormancy -> forgetting ----------------
        # Per-DOMAIN profile resolves each object's horizons (mixed brains
        # supported). Repetition protects: >=3 sightings earn 3x horizons.
        rep["objects_dormant"] = 0
        rep["objects_forgotten"] = 0
        dom_rows = self.store.conn.execute(
            "SELECT o.id, o.status, o.domain_id, "
            "COALESCE(o.last_seen_tau, o.first_seen_tau) lst, "
            "COALESCE(s.sightings,0) sx FROM object o LEFT JOIN "
            "(SELECT object_id, COUNT(*) sightings FROM object_sighting "
            "GROUP BY object_id) s ON s.object_id=o.id "
            "WHERE o.status != 'forgotten'").fetchall()
        dom_cache = {}
        now_cache: dict[int, float] = {}
        for r in dom_rows:
            d_id = int(r["domain_id"])
            if d_id not in dom_cache:
                row = self.store.get_domain(d_id) or {}
                pname = row.get("retention_profile") or \
                    self.base_cfg.get("retention_profile", "balanced")
                dom_cache[d_id] = self.retention_profiles[pname]
                if tau_now is not None:
                    now_cache[d_id] = tau_now
                else:
                    _row = self.store.conn.execute(
                        "SELECT COALESCE(MAX(tau),0.0) t FROM stream "
                        "WHERE domain_id=?", (d_id,)).fetchone()
                    now_cache[d_id] = float(_row["t"] or 0.0)
            P = dom_cache[d_id]
            od, of_ = P["obj_dormant"], P["obj_forget"]
            if od is None or of_ is None:
                continue                        # archival: objects immortal
            dtau = now_cache[d_id] - float(r["lst"] or 0)
            mult = 3.0 if r["sx"] >= 3 else 1.0
            if dtau >= of_ * mult and (
                    r["status"] != "stable" or of_ * mult >= 75 * mult):
                self.store.set_object_status(str(r["id"]), "forgotten")
                rep["objects_forgotten"] += 1
            elif dtau >= od * mult and r["status"] not in ("stable", "dormant"):
                self.store.set_object_status(str(r["id"]), "dormant")
                rep["objects_dormant"] += 1

        return PruneReport(**rep)


def actual_key_val_ok(actual, expected) -> bool:
    return actual is not None and str(actual) == str(expected)


import time
