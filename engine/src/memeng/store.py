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

"""Store protocol + SQLite v0 implementation.

Open-world ontology: persons and domains are runtime entities (registered via
API or auto-registered on first encounter), never baked-in constants.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
import struct
import uuid
from datetime import datetime

from .models import EpisodeRecord, Provenance, Tier

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
INSERT OR IGNORE INTO meta(k,v) VALUES('schema_version','2');

CREATE TABLE IF NOT EXISTS lu_kind (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
INSERT OR IGNORE INTO lu_kind(name) VALUES
 ('message'),('snapshot'),('day_token'),('week_token'),('period_ref');

CREATE TABLE IF NOT EXISTS lu_rel (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
INSERT OR IGNORE INTO lu_rel(name) VALUES
 ('before'),('during'),('after'),('within');

-- ── open-world registries ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS domain (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  created_wall TEXT NOT NULL,
  threshold REAL,               -- optional per-domain gate override
  burst_threshold REAL,
  retention_profile TEXT
);

CREATE TABLE IF NOT EXISTS person (
  pid TEXT PRIMARY KEY,
  key_ref TEXT,                          -- external name-key (optional)
  weight REAL NOT NULL DEFAULT 0.5,      -- H18 relationship gain
  miss_costs INTEGER NOT NULL DEFAULT 0, -- ignored-signal harmful outcomes
  honors INTEGER NOT NULL DEFAULT 0,
  first_seen_wall TEXT,
  first_seen_tau REAL,
  last_seen_tau REAL,
  auto_registered INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stream (
  stream TEXT PRIMARY KEY,
  domain_id INTEGER REFERENCES domain(id),
  tau REAL NOT NULL DEFAULT 0.0,
  mono INTEGER NOT NULL DEFAULT 0
);

-- ── memory ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episode (
  id TEXT PRIMARY KEY,
  stream TEXT NOT NULL,
  domain_id INTEGER REFERENCES domain(id),
  kind TEXT NOT NULL,
  wall TEXT NOT NULL,
  mono INTEGER NOT NULL,
  tau REAL NOT NULL,
  persons_json TEXT NOT NULL DEFAULT '[]',
  subject_norm TEXT,
  salience REAL NOT NULL,
  tier TEXT NOT NULL DEFAULT 'normal',
  strength REAL NOT NULL,
  provenance TEXT NOT NULL DEFAULT 'witnessed',
  fuzz_lo TEXT, fuzz_hi TEXT,
  precision_src TEXT NOT NULL DEFAULT 'exact',
  is_anchor INTEGER NOT NULL DEFAULT 0,
  features_json TEXT NOT NULL DEFAULT '{}',
  level TEXT NOT NULL DEFAULT 'event',
  body_embed BLOB,                     -- float32[] little-endian
  body_text TEXT
);
CREATE INDEX IF NOT EXISTS ep_tau ON episode(tau);
CREATE INDEX IF NOT EXISTS ep_stream_level ON episode(stream, level);
CREATE INDEX IF NOT EXISTS ep_stream_tau ON episode(stream, tau DESC);

CREATE TABLE IF NOT EXISTS anchor_edge (
  episode_id TEXT REFERENCES episode(id) ON DELETE CASCADE,
  anchor_id TEXT REFERENCES episode(id),
  rel TEXT REFERENCES lu_rel(name),
  qualifier TEXT,
  PRIMARY KEY (episode_id, anchor_id)
);

CREATE TABLE IF NOT EXISTS causal_rule (
  id TEXT PRIMARY KEY,
  domain_id INTEGER REFERENCES domain(id),
  antecedent_key TEXT NOT NULL,
  antecedent_val TEXT NOT NULL,
  outcome_key TEXT NOT NULL,
  outcome_val TEXT NOT NULL,
  support_count INTEGER NOT NULL DEFAULT 0,
  violation_count INTEGER NOT NULL DEFAULT 0,
  confidence REAL NOT NULL DEFAULT 0.5,
  derived_from TEXT NOT NULL DEFAULT '[]',
  last_confirmed_tau REAL
);

CREATE TABLE IF NOT EXISTS schema_stat (
  stream TEXT NOT NULL,
  stat_key TEXT NOT NULL,
  stat_val REAL NOT NULL,
  updates INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (stream, stat_key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
  id UNINDEXED, text_content
);

CREATE TABLE IF NOT EXISTS cooccurrence (
  stream TEXT NOT NULL,
  pair_key TEXT NOT NULL,          -- 'subject_norm=<normalized subject>'
  count INTEGER NOT NULL DEFAULT 0,
  top_assoc TEXT NOT NULL DEFAULT '',   -- most-seen associated feature value
  assoc_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (stream, pair_key)
);

CREATE TABLE IF NOT EXISTS image_registry (
  bhash TEXT PRIMARY KEY,          -- sha256 of exact bytes
  phash TEXT,                      -- 64-bit dhash hex
  width INTEGER, height INTEGER,
  nbytes INTEGER,
  first_sender TEXT,
  classification TEXT DEFAULT 'pending',
  seen_count INTEGER NOT NULL DEFAULT 1,
  first_seen_wall TEXT
);

CREATE TABLE IF NOT EXISTS lu_objrel (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
INSERT OR IGNORE INTO lu_objrel(name) VALUES
 ('contains'),('part-of'),('brand-of'),('advertises'),('belongs-to'),
 ('references'),('variant-of'),('made-of'),('located-in'),('associated-with'),
 ('created-by'),('preceded-by');

-- ── object layer (H19/H20 + characteristic grounding hierarchy) ──────
CREATE TABLE IF NOT EXISTS lu_kind_object (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
INSERT OR IGNORE INTO lu_kind_object(name) VALUES
 ('logo'),('signature'),('document'),('product'),('photo'),('scene'),
 ('chart'),('screenshot'),('device'),('place'),('other');

CREATE TABLE IF NOT EXISTS lu_charlevel (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
INSERT OR IGNORE INTO lu_charlevel(id,name) VALUES
 (1,'physical'),(2,'qualitative'),(3,'abstract');

CREATE TABLE IF NOT EXISTS object (
  id TEXT PRIMARY KEY,
  domain_id INTEGER REFERENCES domain(id),
  kind TEXT REFERENCES lu_kind_object(name),
  canonical_name TEXT,
  visual_phash TEXT,
  text_embed BLOB,
  first_seen_wall TEXT,
  last_seen_wall TEXT,
  first_seen_tau REAL,
  last_seen_tau REAL,
  salience REAL DEFAULT 0.5,
  status TEXT DEFAULT 'forming',    -- forming | stable | dormant
  -- multi-axial classification (borrowed: SUMO split + Wikidata multi-axis)
  class_materiality TEXT,           -- physical | abstract
  class_animacy     TEXT,           -- living | non-living (null if abstract)
  class_origin      TEXT,           -- natural | artifact (null if abstract)
  basic_level_name  TEXT            -- Rosch basic-level label (free text)
);

CREATE TABLE IF NOT EXISTS object_relation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id TEXT REFERENCES object(id),
  predicate TEXT REFERENCES lu_objrel(name),
  object_id TEXT REFERENCES object(id),
  confidence REAL DEFAULT 0.5,
  provenance TEXT NOT NULL,          -- vlm-visual | deduced | stated
  evidence_episode TEXT,
  first_seen_wall TEXT,
  UNIQUE(subject_id, predicate, object_id, provenance)
);
CREATE INDEX IF NOT EXISTS objrel_sub ON object_relation(subject_id);
CREATE INDEX IF NOT EXISTS objrel_obj ON object_relation(object_id);

CREATE TABLE IF NOT EXISTS object_sighting (
  episode_id TEXT,
  object_id TEXT REFERENCES object(id),
  channel TEXT,                     -- visual | textual | doc-hash | audio
  confidence REAL,
  PRIMARY KEY (episode_id, object_id, channel)
);

CREATE TABLE IF NOT EXISTS object_characteristic (
  object_id TEXT REFERENCES object(id),
  level INTEGER REFERENCES lu_charlevel(id),
  dimension TEXT NOT NULL,
  val_scalar REAL,
  val_range_lo REAL, val_range_hi REAL,
  val_ordinal INTEGER,
  val_symbol TEXT,
  val_embed BLOB,
  comparator TEXT DEFAULT 'is-a',
  confidence REAL DEFAULT 0.5,
  provenance TEXT DEFAULT 'vlm-estimated',
  valid_tau REAL,
  PRIMARY KEY (object_id, level, dimension, val_symbol, val_scalar)
);

CREATE TABLE IF NOT EXISTS cooccur_assoc (
  stream TEXT NOT NULL,
  pair_key TEXT NOT NULL,
  assoc_val TEXT NOT NULL,
  n INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (stream, pair_key, assoc_val)
);

CREATE TABLE IF NOT EXISTS residue (
  episode_id TEXT PRIMARY KEY,
  usage_count INTEGER NOT NULL,
  persons_json TEXT NOT NULL,
  anchor_edges INTEGER NOT NULL
);
"""


class SQLiteStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._in_batch = False
        self._rules_cache: dict[int, list[dict]] = {}
        self._person_cache: dict[str, dict | None] = {}
        self._stream_cache: dict[str, dict | None] = {}
        self.conn.executescript(_DDL)
        self._migrate()
        self._commit()

    def _migrate(self) -> None:
        """Lightweight additive migrations for pre-existing databases."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(episode)")}
        ocols = {r[1] for r in self.conn.execute("PRAGMA table_info(object)")}
        dcols = {r[1] for r in self.conn.execute("PRAGMA table_info(domain)")}
        if "retention_profile" not in dcols:
            self.conn.execute(
                "ALTER TABLE domain ADD COLUMN retention_profile TEXT")
        for col, decl in (("class_materiality", "TEXT"),
                          ("class_animacy", "TEXT"),
                          ("class_origin", "TEXT"),
                          ("basic_level_name", "TEXT"),
                          ("last_seen_tau", "REAL")):
            if col not in ocols:
                self.conn.execute(f"ALTER TABLE object ADD COLUMN {col} {decl}")
        if "body_embed" not in cols:
            self.conn.execute("ALTER TABLE episode ADD COLUMN body_embed BLOB")
            self.conn.execute("ALTER TABLE episode ADD COLUMN body_text TEXT")
        has_fts = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='episode_fts'").fetchone()
        if not has_fts:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5("
                "id UNINDEXED, text_content)")

    def _commit(self) -> None:
        if not self._in_batch:
            self.conn.commit()

    def begin_batch(self) -> None:
        self._in_batch = True
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN")

    def commit_batch(self) -> None:
        self._in_batch = False
        self.conn.commit()

    def invalidate_caches(self) -> None:
        self._rules_cache.clear()
        self._person_cache.clear()
        self._stream_cache.clear()

    # -- registry ----------------------------------------------------------
    def ensure_default_domain(self) -> int:
        r = self.conn.execute(
            "SELECT id FROM domain WHERE name='default'").fetchone()
        if r:
            return int(r["id"])
        cur = self.conn.execute(
            "INSERT INTO domain(name,created_wall) VALUES('default',?)",
            (datetime.utcnow().isoformat(),))
        self._commit()
        return int(cur.lastrowid)

    def register_domain(self, name: str, threshold: float | None = None,
                        burst_threshold: float | None = None) -> int:
        self.conn.execute(
            "INSERT INTO domain(name,created_wall,threshold,burst_threshold) "
            "VALUES(?,?,?,?) ON CONFLICT(name) DO NOTHING",
            (name, datetime.utcnow().isoformat(), threshold, burst_threshold))
        self._commit()
        r = self.conn.execute("SELECT id FROM domain WHERE name=?",
                              (name,)).fetchone()
        return int(r["id"])

    def set_domain_retention(self, domain_id: int, profile: str) -> None:
        self.conn.execute(
            "UPDATE domain SET retention_profile=? WHERE id=?",
            (profile, domain_id))
        self._commit()
        self._stream_cache.clear()
        self._domain_cache = getattr(self, "_domain_cache", {})
        self._domain_cache.pop(domain_id, None)

    def get_domain_id(self, name: str):
        r = self.conn.execute("SELECT id FROM domain WHERE name=?",
                              (name,)).fetchone()
        return int(r["id"]) if r else None

    def get_domain(self, domain_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM domain WHERE id=?",
                              (domain_id,)).fetchone()
        return dict(r) if r else None

    def attach_stream(self, stream: str, domain_id: int) -> None:
        self.conn.execute(
            "INSERT INTO stream(stream,domain_id) VALUES(?,?) "
            "ON CONFLICT(stream) DO UPDATE SET domain_id=excluded.domain_id",
            (stream, domain_id))
        self._commit()
        self._stream_cache.pop(stream, None)

    def get_stream(self, stream: str) -> dict | None:
        if stream in self._stream_cache:
            return self._stream_cache[stream]
        r = self.conn.execute("SELECT * FROM stream WHERE stream=?",
                              (stream,)).fetchone()
        row = dict(r) if r else None
        self._stream_cache[stream] = row
        return row

    # -- person lifecycle ---------------------------------------------------
    def get_person(self, pid: str) -> dict | None:
        if pid in self._person_cache:
            return self._person_cache[pid]
        r = self.conn.execute("SELECT * FROM person WHERE pid=?",
                              (pid,)).fetchone()
        row = dict(r) if r else None
        self._person_cache[pid] = row
        return row

    def register_person(self, pid: str, *, auto: bool,
                        tau: float, wall: datetime,
                        key_ref: str | None = None) -> dict:
        self.conn.execute(
            "INSERT OR IGNORE INTO person(pid,key_ref,first_seen_wall,"
            "first_seen_tau,last_seen_tau,auto_registered) VALUES(?,?,?,?,?,?)",
            (pid, key_ref, wall.isoformat(), tau, tau, int(auto)))
        self._commit()
        self._person_cache.pop(pid, None)
        return self.get_person(pid)  # type: ignore[return-value]

    def touch_person(self, pid: str, tau: float) -> None:
        self.conn.execute(
            "UPDATE person SET last_seen_tau=? WHERE pid=?", (tau, pid))
        self._commit()

    def set_person_weight(self, pid: str, weight: float) -> None:
        self.conn.execute("UPDATE person SET weight=? WHERE pid=?",
                          (weight, pid))
        self._commit()
        self._person_cache.pop(pid, None)

    def bump_miss_cost(self, pid: str) -> None:
        self.conn.execute(
            "UPDATE person SET miss_costs=miss_costs+1, "
            "weight=MIN(1.0, weight+0.05) WHERE pid=?", (pid,))
        self._commit()

    # -- tau / mono ---------------------------------------------------------
    def stream_time(self, stream: str) -> tuple[float, int]:
        r = self.conn.execute("SELECT tau,mono FROM stream WHERE stream=?",
                              (stream,)).fetchone()
        return (float(r["tau"]), int(r["mono"])) if r else (0.0, 0)

    def set_stream_time(self, stream: str, tau: float, mono: int) -> None:
        self.conn.execute("UPDATE stream SET tau=?, mono=? WHERE stream=?",
                          (tau, mono, stream))
        self._commit()
        if stream in self._stream_cache:
            self._stream_cache[stream]["tau"] = tau
            self._stream_cache[stream]["mono"] = mono

    # -- episodes ------------------------------------------------------------
    def insert_episode(self, rec: EpisodeRecord, domain_id: int) -> None:
        self.conn.execute(
            """INSERT INTO episode(id,stream,domain_id,kind,wall,mono,tau,
               persons_json,subject_norm,salience,tier,strength,provenance,
               fuzz_lo,fuzz_hi,precision_src,is_anchor,features_json,level)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(rec.id), rec.stream, domain_id, rec.kind, rec.wall.isoformat(),
             rec.mono, rec.tau, json.dumps(rec.persons), rec.subject_norm,
             rec.salience, rec.tier.value, rec.strength, rec.provenance.value,
             rec.fuzz_lo.isoformat() if rec.fuzz_lo else None,
             rec.fuzz_hi.isoformat() if rec.fuzz_hi else None,
             rec.precision_src, int(rec.is_anchor),
             json.dumps(rec.features), "event"))
        self._commit()

    def update_episode(self, episode_id: str, **cols: object) -> None:
        sets = ", ".join(f"{k}=?" for k in cols)
        self.conn.execute(f"UPDATE episode SET {sets} WHERE id=?",
                          (*cols.values(), episode_id))
        self._commit()

    def count_episodes(self) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) c FROM episode WHERE level='event'").fetchone()
        return int(r["c"])

    def get_episode(self, episode_id: str | uuid.UUID) -> EpisodeRecord | None:
        r = self.conn.execute("SELECT * FROM episode WHERE id=?",
                              (str(episode_id),)).fetchone()
        if not r:
            return None
        return EpisodeRecord(
            id=uuid.UUID(r["id"]), stream=r["stream"], kind=r["kind"],
            wall=datetime.fromisoformat(r["wall"]), mono=r["mono"],
            tau=r["tau"], persons=json.loads(r["persons_json"]),
            subject_norm=r["subject_norm"], salience=r["salience"],
            tier=Tier(r["tier"]), strength=r["strength"],
            provenance=Provenance(r["provenance"]),
            fuzz_lo=_dt(r["fuzz_lo"]), fuzz_hi=_dt(r["fuzz_hi"]),
            precision_src=r["precision_src"], is_anchor=bool(r["is_anchor"]),
            features=json.loads(r["features_json"]))


    def iter_episodes(self, level: str = "event") -> list[EpisodeRecord]:
        rows = self.conn.execute("SELECT * FROM episode WHERE level=?",
                                 (level,)).fetchall()
        out = []
        for r in rows:
            out.append(EpisodeRecord(
                id=uuid.UUID(r["id"]), stream=r["stream"],
                domain_id=r["domain_id"], kind=r["kind"],
                wall=datetime.fromisoformat(r["wall"]), mono=r["mono"],
                tau=r["tau"], persons=json.loads(r["persons_json"]),
                subject_norm=r["subject_norm"], salience=r["salience"],
                tier=Tier(r["tier"]), strength=r["strength"],
                provenance=Provenance(r["provenance"]),
                fuzz_lo=_dt(r["fuzz_lo"]), fuzz_hi=_dt(r["fuzz_hi"]),
                precision_src=r["precision_src"],
                is_anchor=bool(r["is_anchor"]),
                features=json.loads(r["features_json"])))
        return out

    # -- rules ---------------------------------------------------------------
    def upsert_rule(self, domain_id: int, key: str, val: str,
                    outcome_key: str, outcome_val: str,
                    rule_id: str | None = None) -> str:
        rid = rule_id or uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT OR IGNORE INTO causal_rule(id,domain_id,antecedent_key,"
            "antecedent_val,outcome_key,outcome_val) VALUES(?,?,?,?,?,?)",
            (rid, domain_id, key, val, outcome_key, outcome_val))
        self._commit()
        self._rules_cache.pop(domain_id, None)
        return rid

    def active_rules(self, domain_id: int) -> list[dict]:
        cached = self._rules_cache.get(domain_id)
        if cached is not None:
            return cached
        rows = self.conn.execute(
            "SELECT * FROM causal_rule WHERE domain_id=?",
            (domain_id,)).fetchall()
        out = [dict(r) for r in rows]
        self._rules_cache[domain_id] = out
        return out

    def grade_rule(self, rule_id: str, *, matched: bool) -> dict:
        col = ("support_count", "violation_count")[not matched]
        self.conn.execute(
            f"UPDATE causal_rule SET {col}={col}+1, "
            "confidence=CAST(support_count AS REAL)/"
            "(support_count+violation_count+2.0) WHERE id=?", (rule_id,))
        self._commit()
        r = self.conn.execute("SELECT * FROM causal_rule WHERE id=?",
                              (rule_id,)).fetchone()
        if r:
            self._rules_cache.pop(int(r["domain_id"]), None)
        return dict(r)

    # -- misc ------------------------------------------------------------------
    def bump_schema_stat(self, stream: str, key: str, val: float = 1.0) -> None:
        self.conn.execute(
            "INSERT INTO schema_stat(stream,stat_key,stat_val,updates) "
            "VALUES(?,?,?,1) ON CONFLICT(stream,stat_key) DO UPDATE SET "
            "stat_val=stat_val+excluded.stat_val, updates=updates+1",
            (stream, key, val))
        self._commit()

    def set_embedding(self, episode_id: str, vec: list[float]) -> None:
        import struct
        self.conn.execute(
            "UPDATE episode SET body_embed=? WHERE id=?",
            (struct.pack(f"<{len(vec)}f", *vec), episode_id))
        self._commit()

    def iter_embeddings(self) -> list[tuple[str, list[float]]]:
        out = []
        for r in self.conn.execute(
                "SELECT id, body_embed FROM episode "
                "WHERE body_embed IS NOT NULL"):
            n = len(r["body_embed"]) // 4
            vec = list(struct.unpack(f"<{n}f", r["body_embed"]))
            out.append((r["id"], vec))
        return out

    def fts_upsert(self, episode_id: str, text: str) -> None:
        self.conn.execute(
            "INSERT INTO episode_fts(id,text_content) VALUES(?,?)",
            (episode_id, text))
        self._commit()

    def recent_candidates(self, k: int = 24,
                          stream: str | None = None) -> list[str]:
        """Recent non-empty episodes for the recency channel (indexed)."""
        q = ("SELECT id FROM episode WHERE level='event' "
             "AND COALESCE(subject_norm,'') != '' ")
        args: list = []
        if stream:
            q += "AND stream=? "
            args.append(stream)
        q += "ORDER BY tau DESC LIMIT ?"
        rows = self.conn.execute(q, (*args, k)).fetchall()
        return [r[0] if isinstance(r[0], str) else r[0][0] if False else
                (r[0]) for r in rows]

    def fts_search(self, query: str, k: int = 8) -> list[str]:
        try:
            rows = self.conn.execute(
                "SELECT id FROM episode_fts WHERE episode_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (query, k)).fetchall()
        except Exception:
            return []
        return [r["id"] for r in rows]

    def bump_cooccurrence(self, stream: str, pair_key: str,
                          assoc_val: str) -> tuple[int, str, int]:
        """Count pair occurrences and per-association votes.
        Returns (total_pair_count, best_assoc_val, best_assoc_n)."""
        self.conn.execute(
            "INSERT INTO cooccurrence(stream,pair_key,count,top_assoc,"
            "assoc_count) VALUES(?,?,1,'',0) "
            "ON CONFLICT(stream,pair_key) DO UPDATE SET count=count+1",
            (stream, pair_key))
        self.conn.execute(
            "INSERT INTO cooccur_assoc(stream,pair_key,assoc_val,n) "
            "VALUES(?,?,?,1) ON CONFLICT(stream,pair_key,assoc_val) "
            "DO UPDATE SET n=n+1", (stream, pair_key, assoc_val))
        r = self.conn.execute(
            "SELECT count FROM cooccurrence WHERE stream=? AND pair_key=?",
            (stream, pair_key)).fetchone()
        best = self.conn.execute(
            "SELECT assoc_val, n FROM cooccur_assoc WHERE stream=? AND "
            "pair_key=? ORDER BY n DESC LIMIT 1",
            (stream, pair_key)).fetchone()
        self._commit()
        return (int(r["count"]),
                best["assoc_val"] if best else "",
                int(best["n"]) if best else 0)

    def iter_cooccurrences(self, stream: str, min_count: int):
        rows = self.conn.execute(
            "SELECT pair_key, count FROM cooccurrence "
            "WHERE stream=? AND count>=? ORDER BY count DESC",
            (stream, min_count)).fetchall()
        return [dict(r) for r in rows]

    # -- object layer --------------------------------------------------------
    def upsert_object(self, *, domain_id: int, kind: str,
                      canonical_name: str | None, visual_phash: str | None,
                      wall: datetime, tau: float,
                      class_materiality: str | None = None,
                      class_animacy: str | None = None,
                      class_origin: str | None = None,
                      basic_level_name: str | None = None) -> str:
        oid = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO object(id,domain_id,kind,canonical_name,"
            "visual_phash,first_seen_wall,last_seen_wall,first_seen_tau,"
            "salience,status,class_materiality,class_animacy,class_origin,"
            "basic_level_name) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, domain_id, kind, canonical_name, visual_phash,
             wall.isoformat(), wall.isoformat(), tau, 0.5, "forming",
             class_materiality, class_animacy, class_origin,
             basic_level_name))
        self._commit()
        return oid

    def add_object_characteristic(self, object_id: str, level: int,
                                  dimension: str, *,
                                  val_scalar: float | None = None,
                                  val_symbol: str | None = None,
                                  confidence: float = 0.5,
                                  provenance: str = "vlm-estimated",
                                  valid_tau: float | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO object_characteristic(object_id,level,"
            "dimension,val_scalar,val_symbol,confidence,provenance,valid_tau)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (object_id, level, dimension, val_scalar, val_symbol,
             confidence, provenance, valid_tau))
        self._commit()

    def get_object_full(self, object_id: str) -> dict | None:
        o = self.conn.execute("SELECT * FROM object WHERE id=?",
                              (object_id,)).fetchone()
        if not o:
            return None
        chars = [dict(r) for r in self.conn.execute(
            "SELECT * FROM object_characteristic WHERE object_id=?",
            (object_id,)).fetchall()]
        return {"object": dict(o), "characteristics": chars}

    def get_or_create_object(self, *, domain_id: int, kind: str,
                             canonical_name: str | None, wall: datetime,
                             tau: float, phash: str | None = None) -> str:
        """Dedupe by (kind, canonical_name) so relations converge on one node."""
        row = self.conn.execute(
            "SELECT id FROM object WHERE domain_id=? AND kind=? AND "
            "canonical_name IS ?", (domain_id, kind,
                                    canonical_name)).fetchone()
        if row:
            return str(row["id"])
        oid = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO object(id,domain_id,kind,canonical_name,"
            "visual_phash,first_seen_wall,last_seen_wall,first_seen_tau,"
            "salience,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (oid, domain_id, kind, canonical_name, phash,
             wall.isoformat(), wall.isoformat(), tau, 0.5, "forming"))
        self._commit()
        return oid

    def add_object_relation(self, *, subject_id: str, predicate: str,
                            object_id: str, provenance: str,
                            confidence: float = 0.5,
                            evidence_episode: str | None = None) -> int | None:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO object_relation(subject_id,predicate,"
            "object_id,confidence,provenance,evidence_episode,"
            "first_seen_wall) VALUES(?,?,?,?,?,?,?)",
            (subject_id, predicate, object_id, confidence, provenance,
             evidence_episode, datetime.now(timezone.utc).isoformat()))
        self._commit()
        return cur.lastrowid if cur.rowcount else None

    def object_graph(self, root_id: str, depth: int = 2) -> dict:
        """Neighborhood walk: nodes + typed edges up to depth."""
        nodes, edges = {}, []
        frontier = [root_id]
        seen_edges = set()
        for _ in range(depth):
            nxt = []
            for nid in frontier:
                if nid in nodes:
                    continue
                r = self.conn.execute("SELECT * FROM object WHERE id=?",
                                      (nid,)).fetchone()
                if r:
                    nodes[str(r["id"])] = dict(r)
                rows = self.conn.execute(
                    "SELECT * FROM object_relation WHERE subject_id=? "
                    "OR object_id=?", (nid, nid)).fetchall()
                for rel in rows:
                    key = (rel["subject_id"], rel["predicate"],
                           rel["object_id"])
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(dict(rel))
                    other = (rel["object_id"]
                             if rel["subject_id"] == nid
                             else rel["subject_id"])
                    if other not in nodes:
                        nxt.append(other)
            frontier = nxt
        return {"nodes": list(nodes.values()), "edges": edges}

    def link_sighting(self, episode_id: str, object_id: str,
                      channel: str, confidence: float = 0.9) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO object_sighting(episode_id,object_id,"
            "channel,confidence) VALUES(?,?,?,?)",
            (episode_id, object_id, channel, confidence))
        self._commit()

    def touch_object(self, object_id: str, tau: float,
                     wall: datetime) -> None:
        self.conn.execute(
            "UPDATE object SET last_seen_tau=?, last_seen_wall=? WHERE id=?",
            (tau, wall.isoformat(), object_id))
        self._commit()

    # -- image recognition registry ---------------------------------------
    def get_image_by_bhash(self, bhash: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM image_registry WHERE bhash=?",
                              (bhash,)).fetchone()
        return dict(r) if r else None

    def iter_image_phashes(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT bhash, phash, classification, seen_count "
            "FROM image_registry WHERE phash IS NOT NULL")]

    def set_image_classification(self, bhash: str, classification: str) -> None:
        self.conn.execute(
            "UPDATE image_registry SET classification=? WHERE bhash=?",
            (classification, bhash))
        self._commit()

    def get_object(self, object_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM object WHERE id=?",
                              (object_id,)).fetchone()
        return dict(r) if r else None

    def set_object_status(self, object_id: str, status: str) -> None:
        self.conn.execute("UPDATE object SET status=? WHERE id=?",
                          (status, object_id))
        self._commit()

    def bump_image_seen(self, bhash: str) -> None:
        self.conn.execute(
            "UPDATE image_registry SET seen_count=seen_count+1 WHERE bhash=?",
            (bhash,))
        self._commit()

    def register_image(self, *, bhash: str, phash: str | None,
                       width: int | None, height: int | None,
                       nbytes: int, sender_pid: str | None,
                       classification: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO image_registry(bhash,phash,width,height,"
            "nbytes,first_sender,classification,seen_count,first_seen_wall) "
            "VALUES(?,?,?,?,?,?,?,?,datetime('now'))",
            (bhash, phash, width, height, nbytes, sender_pid,
             classification, 1))
        self._commit()

    def write_residue(self, episode_id: str, usage_count: int,
                      persons: list[str], anchor_edges: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO residue(episode_id,usage_count,"
            "persons_json,anchor_edges) VALUES(?,?,?,?)",
            (episode_id, usage_count, json.dumps(persons), anchor_edges))
        self._commit()


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None
