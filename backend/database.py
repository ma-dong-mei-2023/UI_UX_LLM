import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Saved configurations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            module TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Training run history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT PRIMARY KEY,
            module TEXT NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL,
            metrics_json TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_config(name: str, module: str, config: dict) -> int:
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO saved_configs (name, module, config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (name, module, json.dumps(config), now, now)
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def list_configs(module: str | None = None) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    if module:
        cursor.execute("SELECT * FROM saved_configs WHERE module = ? ORDER BY updated_at DESC", (module,))
    else:
        cursor.execute("SELECT * FROM saved_configs ORDER BY updated_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    for r in rows:
        r["config"] = json.loads(r["config_json"])
    return rows


def delete_config(config_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_configs WHERE id = ?", (config_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def save_run(run_id: str, module: str, config: dict, status: str):
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO training_runs (id, module, config_json, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, module, json.dumps(config), status, now)
    )
    conn.commit()
    conn.close()


def update_run_status(run_id: str, status: str, metrics: list | None = None):
    conn = get_connection()
    cursor = conn.cursor()
    if status in ("completed", "cancelled", "error"):
        completed_at = datetime.utcnow().isoformat()
        cursor.execute(
            "UPDATE training_runs SET status = ?, metrics_json = ?, completed_at = ? WHERE id = ?",
            (status, json.dumps(metrics) if metrics else None, completed_at, run_id)
        )
    else:
        cursor.execute(
            "UPDATE training_runs SET status = ? WHERE id = ?",
            (status, run_id)
        )
    conn.commit()
    conn.close()


def list_runs(module: str | None = None) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    if module:
        cursor.execute("SELECT * FROM training_runs WHERE module = ? ORDER BY created_at DESC LIMIT 50", (module,))
    else:
        cursor.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 50")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
