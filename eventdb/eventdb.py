"""SENTINEL-EDGE event store: every detection becomes a structured event.

The analyst LLM reads this database — never raw video. Privacy by
architecture: footage stays on the device; only events leave it.

Commits are batched: log() stages rows, commit() flushes. The pipeline
commits every N events and once more on shutdown.

Metric honesty notes (see shift_summary):
- transition_compliance_rate is the share of logged state-TRANSITION events
  that are compliant. It is not time-in-state and not per-worker.
- violations_per_hour divides by the OBSERVED time span in the window,
  not the window length.
- track_id is scoped to a run_id: IDs restart every run, so repeat-offender
  analysis is per-run unless runs are joined explicitly.
"""

import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,   -- unix epoch
    run_id      TEXT    NOT NULL DEFAULT '',  -- one pipeline run
    zone        TEXT    NOT NULL DEFAULT 'default',
    track_id    INTEGER NOT NULL,
    event       TEXT    NOT NULL,   -- 'violation' | 'compliant' | 'escalation'
    class_name  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    crop_path   TEXT                    -- evidence crop, on-device only
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone);
"""


class EventStore:
    def __init__(self, path: str | Path = "sentinel.db",
                 run_id: str | None = None):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        # migrate databases created before run_id existed
        cols = [r[1] for r in
                self.db.execute("PRAGMA table_info(events)").fetchall()]
        if "run_id" not in cols:
            self.db.execute("ALTER TABLE events ADD COLUMN run_id TEXT "
                            "NOT NULL DEFAULT ''")
            self.db.commit()
        self.run_id = run_id or uuid.uuid4().hex[:8]

    def log(self, run_id: str, zone: str, track_id: int, event: str,
            class_name: str, confidence: float,
            crop_path: str | None = None) -> int:
        """Stage a row. Call commit() to flush (batched by the caller)."""
        cur = self.db.execute(
            "INSERT INTO events (ts, run_id, zone, track_id, event, "
            "class_name, confidence, crop_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), run_id, zone, track_id, event, class_name,
             confidence, crop_path))
        return cur.lastrowid

    def commit(self) -> None:
        self.db.commit()

    def shift_summary(self, hours: float = 8.0) -> dict:
        """Aggregate compliance stats for the trailing window — the
        analyst's input. Rates use the observed time span, so a 5-minute
        demo doesn't report violations/hour diluted over 8 hours."""
        span = self.db.execute(
            "SELECT MIN(ts), MAX(ts) FROM events "
            "WHERE ts > strftime('%s','now') - ?",
            (hours * 3600,)).fetchone()
        observed_hours = max(((span[1] or 0) - (span[0] or 0)) / 3600.0,
                             1.0 / 3600.0)  # at least 1 second
        cur = self.db.execute(
            """SELECT zone,
                      COUNT(*) AS n,
                      SUM(event = 'violation') AS violations,
                      SUM(event = 'compliant') AS compliant,
                      SUM(event = 'escalation') AS escalations
               FROM events WHERE ts > strftime('%s','now') - ?
               GROUP BY zone""", (hours * 3600,))
        zones = []
        for z, n, v, c, e in cur.fetchall():
            v, c, e = v or 0, c or 0, e or 0
            zones.append({"zone": z, "events": n, "violations": v,
                          "compliant": c, "escalations": e,
                          "transition_compliance_rate": c / max(1, n),
                          "violations_per_hour": v / observed_hours})
        zones.sort(key=lambda r: r["violations_per_hour"], reverse=True)
        reps = self.db.execute(
            """SELECT run_id, track_id, COUNT(*) AS v FROM events
               WHERE event = 'violation'
                 AND ts > strftime('%s','now') - ?
               GROUP BY run_id, track_id HAVING v >= 2
               ORDER BY v DESC LIMIT 5""", (hours * 3600,)).fetchall()
        return {"window_hours": hours,
                "observed_hours": round(observed_hours, 3),
                "zones": zones,
                "repeat_tracks": [
                    {"run_id": r, "track_id": t, "violations": v}
                    for r, t, v in reps]}

    def close(self):
        self.commit()
        self.db.close()
