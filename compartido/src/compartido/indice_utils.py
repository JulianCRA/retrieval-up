"""SQLite store for index-level metadata: recursos and chunks.

Schema
------
recursos
    hash          TEXT  PK
    titulo        TEXT
    uri           TEXT
    fuente        TEXT
    tiempos_json  TEXT  – JSON dict of pipeline timing info (per-resource)
    thumbnail     BLOB  – raw bytes of the thumbnail image (nullable)

chunks
    id            TEXT  PK  – "{hash}:{chunk_idx}"
    hash          TEXT  FK → recursos.hash
    chunk_idx     INTEGER
    texto         TEXT
    inicio        REAL
    fin           REAL
    segmentos_json TEXT  – JSON array of {inicio, fin, texto}
"""

import sqlite3

from compartido.rutas import INDICE_DB


def _conectar() -> sqlite3.Connection:
    INDICE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDICE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def crear_tablas() -> None:
    """Crea las tablas si no existen."""
    conn = _conectar()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recursos (
                hash         TEXT PRIMARY KEY,
                titulo       TEXT,
                uri          TEXT,
                fuente       TEXT,
                tiempos_json TEXT,
                thumbnail    BLOB
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id             TEXT    PRIMARY KEY,
                hash           TEXT    NOT NULL REFERENCES recursos(hash) ON DELETE CASCADE,
                chunk_idx      INTEGER NOT NULL,
                texto          TEXT,
                inicio         REAL,
                fin            REAL,
                segmentos_json TEXT
            );
        """)
        conn.commit()
    finally:
        conn.close()


def escribir_recurso(
    hash_id: str,
    titulo: str,
    uri: str,
    fuente: str,
    tiempos_json: str = "{}",
    thumbnail: bytes | None = None,
) -> None:
    conn = _conectar()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO recursos (hash, titulo, uri, fuente, tiempos_json, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (hash_id, titulo, uri, fuente, tiempos_json, thumbnail),
        )
        conn.commit()
    finally:
        conn.close()


def escribir_chunks(filas: list[dict]) -> None:
    rows = [
        (
            fila["id"],
            fila["hash"],
            fila["chunk_idx"],
            fila["texto"],
            fila["inicio"],
            fila["fin"],
            fila.get("segmentos_json", "[]"),
        )
        for fila in filas
    ]
    conn = _conectar()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (id, hash, chunk_idx, texto, inicio, fin, segmentos_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def enriquecer(filas: list[dict]) -> list[dict]:
    """Merge chunk + resource metadata into LanceDB result rows (joined by id)."""
    if not filas or not INDICE_DB.exists():
        return filas
    ids = [f["id"] for f in filas if f.get("id")]
    if not ids:
        return filas

    conn = _conectar()
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.chunk_idx, c.texto, c.inicio, c.fin, c.segmentos_json,
                   r.titulo, r.uri, r.fuente
            FROM chunks c
            JOIN recursos r ON c.hash = r.hash
            WHERE c.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    finally:
        conn.close()

    meta = {row["id"]: dict(row) for row in rows}
    return [{**dict(fila), **meta.get(fila.get("id"), {})} for fila in filas]


def borrar_hash(hash_id: str) -> None:
    """Remove a resource and all its chunks (cascade)."""
    conn = _conectar()
    try:
        conn.execute("DELETE FROM recursos WHERE hash = ?", (hash_id,))
        conn.commit()
    finally:
        conn.close()
