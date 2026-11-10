import json
import sqlite3
import time
from pathlib import Path


class Memory:
    """Request state lives in LangGraph; episodes and explicit facts live here."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS facts (
              user_id INTEGER, key TEXT, value TEXT, source TEXT,
              valid_from REAL, expires_at REAL, version INTEGER,
              PRIMARY KEY(user_id,key));
            CREATE TABLE IF NOT EXISTS episodes (
              id INTEGER PRIMARY KEY, user_id INTEGER, session_id TEXT,
              summary TEXT, trace_id TEXT, created_at REAL, expires_at REAL);
            CREATE INDEX IF NOT EXISTS episode_user ON episodes(user_id,created_at);
            CREATE TABLE IF NOT EXISTS alerts (
              id INTEGER PRIMARY KEY, received_at REAL, payload TEXT);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def recall(self, user_id):
        now = time.time()
        with self.connect() as db:
            db.execute('DELETE FROM facts WHERE expires_at < ?', (now,))
            db.execute('DELETE FROM episodes WHERE expires_at < ?', (now,))
            facts = db.execute('SELECT key,value,version FROM facts WHERE user_id=?',
                               (user_id,)).fetchall()
            episodes = db.execute('SELECT summary FROM episodes WHERE user_id=? '
                                  'ORDER BY created_at DESC LIMIT 3', (user_id,)).fetchall()
        return {"facts": {k: {"value": json.loads(v), "version": ver} for k,v,ver in facts},
                "episodes": [json.loads(e[0]) for e in episodes]}

    def explicit_preference(self, user_id, key, value):
        if key not in {"exclude_genres", "prefer_genres"}:
            raise ValueError("Unsupported preference")
        now = time.time()
        with self.connect() as db:
            db.execute('INSERT INTO facts VALUES(?,?,?,?,?,?,1) '
                       'ON CONFLICT(user_id,key) DO UPDATE SET '
                       'value=excluded.value,source=excluded.source,'
                       'valid_from=excluded.valid_from,expires_at=excluded.expires_at,'
                       'version=facts.version+1',
                       (user_id,key,json.dumps(value),"explicit_user_action",now,now+90*86400))

    def remember_episode(self, user_id, session_id, movie_ids, trace_id):
        now=time.time()
        summary=json.dumps({"movie_ids":movie_ids,"catalog_version":"ml25m-2017-train"})
        with self.connect() as db:
            db.execute('INSERT INTO episodes(user_id,session_id,summary,trace_id,created_at,expires_at) '
                       'VALUES(?,?,?,?,?,?)',(user_id,session_id,summary,trace_id,now,now+30*86400))

    def forget(self, user_id):
        with self.connect() as db:
            db.execute('DELETE FROM facts WHERE user_id=?',(user_id,))
            db.execute('DELETE FROM episodes WHERE user_id=?',(user_id,))

    def alert(self, payload):
        with self.connect() as db:
            db.execute('INSERT INTO alerts(received_at,payload) VALUES(?,?)',
                       (time.time(),json.dumps(payload,ensure_ascii=False)))

    def recent_alerts(self):
        with self.connect() as db:
            rows=db.execute('SELECT received_at,payload FROM alerts ORDER BY id DESC LIMIT 10').fetchall()
        return [{"received_at":t,"payload":json.loads(p)} for t,p in rows]
