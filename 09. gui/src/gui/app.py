"""Flask web GUI — browse indexed resources.

Run from repo root:
    python -m gui

Or with Flask's dev server:
    flask --app "09. gui/src/gui/app.py" run --debug
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flask import Flask, Response, abort, redirect, render_template, request, url_for

from compartido.rutas import INDICE_DB

app = Flask(__name__)

PAGE_SIZE = 24  # resources per page


@app.template_filter("hhmmss")
def hhmmss(seconds):
    """Format a duration in seconds as h:mm:ss or m:ss."""
    if seconds is None:
        return ""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(INDICE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _recursos_paginados(page: int, q: str = "") -> tuple[list[dict], int]:
    """Return (rows, total_count) for the given page and optional title filter."""
    offset = (page - 1) * PAGE_SIZE
    like = f"%{q}%"

    with _db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM recursos WHERE titulo LIKE ?", (like,)
        ).fetchone()[0]

        rows = conn.execute(
            """
            SELECT r.hash, r.titulo, r.uri, r.fuente, r.duracion, r.tags,
                   (r.thumbnail IS NOT NULL) AS has_thumbnail,
                   COUNT(c.id) AS num_chunks
            FROM recursos r
            LEFT JOIN chunks c ON c.hash = r.hash
            WHERE r.titulo LIKE ?
            GROUP BY r.hash
            ORDER BY r.titulo COLLATE NOCASE
            LIMIT ? OFFSET ?
            """,
            (like, PAGE_SIZE, offset),
        ).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        try:
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
        items.append(d)

    return items, total


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("browse"))


@app.route("/browse")
def browse():
    q = request.args.get("q", "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    if not INDICE_DB.exists():
        recursos, total = [], 0
    else:
        recursos, total = _recursos_paginados(page, q)

    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)

    return render_template(
        "browse.html",
        recursos=recursos,
        page=page,
        total_pages=total_pages,
        total=total,
        q=q,
        page_size=PAGE_SIZE,
    )


@app.route("/thumbnail/<hash_id>")
def thumbnail(hash_id: str):
    if not INDICE_DB.exists():
        abort(404)

    with _db() as conn:
        row = conn.execute(
            "SELECT thumbnail FROM recursos WHERE hash = ?", (hash_id,)
        ).fetchone()

    if row is None or row["thumbnail"] is None:
        abort(404)

    # Detect image format from magic bytes
    data: bytes = bytes(row["thumbnail"])
    if data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/jpeg"  # fallback

    resp = Response(data, mimetype=mime)
    resp.cache_control.max_age = 3600
    resp.cache_control.public = True
    return resp
