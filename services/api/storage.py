import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "complaints.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_ref TEXT NOT NULL,
    text TEXT NOT NULL,
    channel TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def insert(conn, *, customer_ref, text, channel, category, priority, created_at):
    cur = conn.execute(
        "INSERT INTO complaints (customer_ref, text, channel, category, priority, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (customer_ref, text, channel, category, priority, created_at),
    )
    conn.commit()
    return conn.execute("SELECT * FROM complaints WHERE id = ?", (cur.lastrowid,)).fetchone()


def fetch_all(conn):
    return conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()


def fetch_one(conn, complaint_id):
    return conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()


def stats(conn):
    total = conn.execute("SELECT COUNT(*) AS n FROM complaints").fetchone()["n"]
    rows = conn.execute(
        "SELECT category, COUNT(*) AS n FROM complaints GROUP BY category ORDER BY n DESC"
    ).fetchall()
    high = conn.execute("SELECT COUNT(*) AS n FROM complaints WHERE priority = 'high'").fetchone()["n"]
    return total, rows, high
