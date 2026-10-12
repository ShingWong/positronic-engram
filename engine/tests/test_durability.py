"""Durability is a contract, so it is asserted rather than assumed.

`PRAGMA synchronous` is per-connection and is NOT persisted in the file,
unlike `journal_mode`. Setting it once via a CLI therefore evaporates the
moment any process opens the store -- a brain can silently sit at the SQLite
default while its operator believes it is tuned. These tests exist so that
regression is caught here instead of at the next bulk ingest.
"""
import sqlite3

import pytest

from memeng.store import SQLiteStore


def _pragmas(store):
    return {
        "synchronous": store.conn.execute("PRAGMA synchronous").fetchone()[0],
        "journal_mode": str(
            store.conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
    }


def test_default_durability_is_full(tmp_path):
    """Default must stay `full`: archival brains promise never to forget."""
    store = SQLiteStore(str(tmp_path / "a.db"))
    assert store.durability == "full"
    assert _pragmas(store)["synchronous"] == 2


def test_in_memory_default_is_full():
    store = SQLiteStore()
    assert store.durability == "full"
    assert _pragmas(store)["synchronous"] == 2


def test_normal_implies_wal(tmp_path):
    """`normal` is only survivable under WAL, so opting in enables it."""
    store = SQLiteStore(str(tmp_path / "b.db"), durability="normal")
    p = _pragmas(store)
    assert store.durability == "normal"
    assert p["journal_mode"] == "wal"
    assert p["synchronous"] == 1


def test_durability_survives_reopen(tmp_path):
    """Full is re-applied at every open, because synchronous is not persisted."""
    path = str(tmp_path / "c.db")
    SQLiteStore(path, durability="normal").conn.close()
    reopened = SQLiteStore(path)  # default is full, regardless of last time
    assert _pragmas(reopened)["synchronous"] == 2


def test_unknown_durability_raises(tmp_path):
    with pytest.raises(ValueError, match="durability must be one of"):
        SQLiteStore(str(tmp_path / "d.db"), durability="fast")


def test_normal_engages_wal_on_a_rollback_journal(tmp_path):
    """`normal` implies WAL.

    On a rollback-journal DB it engages WAL rather than refusing. Refusing
    would be the worse outcome: it would leave the caller at
    synchronous=NORMAL *without* WAL, which is the pairing the setting is only
    survivable under. The raise in _set_durability stays for the case where
    WAL genuinely cannot be engaged.
    """
    path = str(tmp_path / "e.db")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    store = SQLiteStore(path, durability="normal")
    p = _pragmas(store)
    assert p["journal_mode"] == "wal"
    assert p["synchronous"] == 1


def test_durability_survives_writes(tmp_path):
    """The pragma holds after real writes, not just at open."""
    store = SQLiteStore(str(tmp_path / "f.db"), durability="normal")
    for i in range(50):
        store.conn.execute(
            "INSERT OR IGNORE INTO stream(stream, tau) VALUES(?, ?)",
            (f"s{i}", float(i)))
        store._commit()
    assert _pragmas(store)["synchronous"] == 1
