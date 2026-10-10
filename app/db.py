"""SQLite audit log of screenings (stdlib only)."""
import json
import sqlite3
from contextlib import closing
import time
from .config import DB_PATH

SCHEMA = """CREATE TABLE IF NOT EXISTS screenings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL NOT NULL,
  patient_ref TEXT,
  kind TEXT NOT NULL,
  inputs TEXT NOT NULL,
  risk REAL NOT NULL,
  band TEXT,
  mode TEXT,
  result TEXT NOT NULL)"""


class _Conn:
    """Context manager: commits on success, always closes."""
    def __enter__(self):
        self.c = sqlite3.connect(DB_PATH)
        self.c.row_factory = sqlite3.Row
        self.c.execute(SCHEMA)
        return self.c

    def __exit__(self, et, ev, tb):
        (self.c.rollback if et else self.c.commit)()
        self.c.close()


def _conn():
    return _Conn()


def save(kind, patient_ref, inputs, result) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO screenings(created_at,patient_ref,kind,inputs,risk,band,mode,result) VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), patient_ref, kind, json.dumps(inputs), result["risk"],
             result.get("band"), result["gate"]["mode"], json.dumps(result)))
        return cur.lastrowid


def _row(r, full=False):
    d = {"id": r["id"], "created_at": r["created_at"], "patient_ref": r["patient_ref"],
         "kind": r["kind"], "inputs": json.loads(r["inputs"]), "risk": r["risk"],
         "band": r["band"], "mode": r["mode"]}
    if full:
        d["result"] = json.loads(r["result"])
    return d


def history(limit=50, patient_ref=None):
    q, a = "SELECT * FROM screenings", []
    if patient_ref:
        q += " WHERE patient_ref=?"; a.append(patient_ref)
    q += " ORDER BY id DESC LIMIT ?"; a.append(limit)
    with _conn() as c:
        return [_row(r) for r in c.execute(q, a)]


def get(i):
    with _conn() as c:
        r = c.execute("SELECT * FROM screenings WHERE id=?", (i,)).fetchone()
        return _row(r, True) if r else None


def delete(i) -> bool:
    with _conn() as c:
        return c.execute("DELETE FROM screenings WHERE id=?", (i,)).rowcount > 0


def trend(patient_ref):
    """Risk over time for one patient (the 'trajectory' idea from the deck)."""
    with _conn() as c:
        rows = c.execute("SELECT created_at,risk,band FROM screenings WHERE patient_ref=? AND kind='demo' ORDER BY created_at",
                         (patient_ref,)).fetchall()
    pts = [{"t": r["created_at"], "risk": r["risk"], "band": r["band"]} for r in rows]
    drift = (pts[-1]["risk"] - pts[0]["risk"]) if len(pts) > 1 else 0.0
    return {"patient_ref": patient_ref, "points": pts, "drift": drift,
            "alert": drift > 0.10}
