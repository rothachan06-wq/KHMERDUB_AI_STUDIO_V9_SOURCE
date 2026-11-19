import sqlite3
import os
import json
import threading
from datetime import datetime

class ProjectStore:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        os.makedirs(self.project_dir, exist_ok=True)
        self.db_path = os.path.join(self.project_dir, "project_state.sqlite")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS speakers (
            speaker_id TEXT PRIMARY KEY,
            display_name TEXT,
            ref_clip_path TEXT DEFAULT '',
            voice_mode TEXT DEFAULT 'VoxCPM2 Clone'
        );
        CREATE TABLE IF NOT EXISTS lines (
            idx INTEGER PRIMARY KEY,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            speaker_id TEXT NOT NULL,
            source_text TEXT NOT NULL,
            khmer_text TEXT DEFAULT '',
            dub_wav_path TEXT DEFAULT '',
            stage_done INTEGER DEFAULT 0
        );
        """)
        self.conn.commit()

    def set_meta(self, key, value):
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value))
            )
            self.conn.commit()

    def get_meta(self, key, default=""):
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def upsert_speaker(self, speaker_id, display_name=None, ref_clip_path="", voice_mode="VoxCPM2 Clone"):
        with self._lock:
            self.conn.execute(
                """INSERT INTO speakers(speaker_id, display_name, ref_clip_path, voice_mode)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(speaker_id) DO UPDATE SET
                   display_name=COALESCE(excluded.display_name, speakers.display_name),
                   ref_clip_path=CASE WHEN excluded.ref_clip_path != '' THEN excluded.ref_clip_path ELSE speakers.ref_clip_path END,
                   voice_mode=excluded.voice_mode""",
                (speaker_id, display_name or speaker_id, ref_clip_path, voice_mode)
            )
            self.conn.commit()

    def get_speakers(self):
        with self._lock:
            return {r["speaker_id"]: dict(r) for r in self.conn.execute("SELECT * FROM speakers").fetchall()}

    def replace_all_lines(self, rows):
        with self._lock:
            with self.conn:
                self.conn.execute("DELETE FROM lines")
                for r in rows:
                    self.upsert_speaker(r["speaker_id"])
                    self.conn.execute(
                        "INSERT INTO lines(idx, start_sec, end_sec, speaker_id, source_text, khmer_text, dub_wav_path, stage_done) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                        (r["idx"], r["start_sec"], r["end_sec"], r["speaker_id"], r["source_text"],
                         r.get("khmer_text", ""), r.get("dub_wav_path", ""), r.get("stage_done", 0))
                    )

    def update_line(self, idx, **fields):
        if not fields:
            return
        allowed = {"start_sec", "end_sec", "speaker_id", "source_text", "khmer_text", "dub_wav_path", "stage_done"}
        if not set(fields).issubset(allowed):
            raise ValueError("Unsupported project line field update.")
        with self._lock:
            cols = ", ".join([f"{k}=?" for k in fields.keys()])
            vals = list(fields.values()) + [idx]
            self.conn.execute(f"UPDATE lines SET {cols} WHERE idx=?", vals)
            self.conn.commit()

    def get_lines(self):
        with self._lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM lines ORDER BY idx ASC").fetchall()]

    def close(self):
        with self._lock:
            self.conn.close()
