"""SQLite persistence for user-saved graphs and experiment runs (cloud-safe with /tmp fallback)."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _resolve_db_path() -> Path:
    env_path = os.environ.get("RWF_DB_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    default_path = Path(__file__).resolve().parent.parent / "rwf_studio.db"
    try:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        test_file = default_path.parent / ".rwf_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return default_path
    except OSError:
        return Path(tempfile.gettempdir()) / "rwf_studio.db"


DB_PATH = _resolve_db_path()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                mode TEXT NOT NULL,
                k INTEGER NOT NULL,
                tau_c INTEGER NOT NULL,
                fingerprint_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_graph(
    name: str,
    nodes: List[Dict[str, Any]],
    edges: List[List[str]],
    mode: str,
    k: int,
    tau_c: int,
    fingerprint: List[float],
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO saved_graphs (name, nodes_json, edges_json, mode, k, tau_c, fingerprint_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, json.dumps(nodes), json.dumps(edges), mode, k, tau_c, json.dumps(fingerprint), now),
        )
        conn.commit()
        gid = cur.lastrowid
    return {
        "id": gid,
        "name": name,
        "n": len(nodes),
        "m": len(edges),
        "mode": mode,
        "k": k,
        "tau_c": tau_c,
        "created_at": now,
    }


def list_graphs() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM saved_graphs ORDER BY id DESC LIMIT 50").fetchall()
    out = []
    for r in rows:
        nodes = json.loads(r["nodes_json"])
        edges = json.loads(r["edges_json"])
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "nodes": nodes,
                "edges": edges,
                "n": len(nodes),
                "m": len(edges),
                "mode": r["mode"],
                "k": r["k"],
                "tau_c": r["tau_c"],
                "fingerprint": json.loads(r["fingerprint_json"]),
                "created_at": r["created_at"],
            }
        )
    return out


def delete_graph(graph_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM saved_graphs WHERE id = ?", (graph_id,))
        conn.commit()
        return cur.rowcount > 0


def log_experiment(kind: str, summary: str, payload: Dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO experiment_logs (kind, summary, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (kind, summary, json.dumps(payload), now),
        )
        conn.commit()


def list_experiments(limit: int = 20) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, kind, summary, created_at FROM experiment_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
