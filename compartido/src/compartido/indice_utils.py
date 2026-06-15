"""SQLite store for index-level metadata: recursos and chunks.

Schema
------
recursos
    hash          TEXT  PK
    titulo        TEXT
    uri           TEXT
    fuente        TEXT
    duracion      REAL  – duration in seconds (nullable)
    tags          TEXT  – JSON array of tag strings, e.g. '["science","math"]' (nullable)
    tiempos_json  TEXT  – JSON dict of pipeline timing info (per-resource)
    thumbnail     BLOB  – raw bytes of the thumbnail image (nullable)

Filtering by tag (SQLite json_each):
    SELECT * FROM recursos
    WHERE EXISTS (SELECT 1 FROM json_each(recursos.tags) WHERE value = 'science')

chunks_{modelo}  (one table per embedding model)
    id            TEXT  PK  – "{hash}:{chunk_idx}"
    hash          TEXT  FK → recursos.hash
    chunk_idx     INTEGER
    texto         TEXT
    inicio        REAL
    fin           REAL
    duracion      REAL  – fin - inicio (seconds)
    segmentos_json TEXT  – JSON array of {inicio, fin, texto}
"""

import re
import sqlite3

from compartido.rutas import INDICE_DB


def _conectar() -> sqlite3.Connection:
    INDICE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDICE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _nombre_chunks(tabla: str) -> str:
    """Devuelve el nombre de la tabla SQLite para los chunks del modelo dado."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", tabla)
    return f"chunks_{safe}"


def crear_tablas(tabla: str) -> None:
    """Crea las tablas si no existen y migra columnas nuevas."""
    chunks_tbl = _nombre_chunks(tabla)
    conn = _conectar()
    try:
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS recursos (
                hash         TEXT PRIMARY KEY,
                titulo       TEXT,
                uri          TEXT,
                fuente       TEXT,
                duracion     REAL,
                tags         TEXT,
                tiempos_json TEXT,
                thumbnail    BLOB
            );

            CREATE TABLE IF NOT EXISTS {chunks_tbl} (
                id             TEXT    PRIMARY KEY,
                hash           TEXT    NOT NULL REFERENCES recursos(hash) ON DELETE CASCADE,
                chunk_idx      INTEGER NOT NULL,
                texto          TEXT,
                inicio         REAL,
                fin            REAL,
                duracion       REAL,
                segmentos_json TEXT
            );
        """)
        conn.commit()

        # Migrate: add new columns to pre-existing databases (ALTER TABLE ADD
        # COLUMN IF NOT EXISTS is not supported in SQLite, so we ignore the
        # error if the column already exists).
        existing_r = {row[1] for row in conn.execute("PRAGMA table_info(recursos)")}
        for col, typedef in [("duracion", "REAL"), ("tags", "TEXT")]:
            if col not in existing_r:
                conn.execute(f"ALTER TABLE recursos ADD COLUMN {col} {typedef}")

        existing_c = {row[1] for row in conn.execute(f"PRAGMA table_info({chunks_tbl})")}
        if "duracion" not in existing_c:
            conn.execute(f"ALTER TABLE {chunks_tbl} ADD COLUMN duracion REAL")
        conn.commit()
    finally:
        conn.close()


def escribir_recurso(
    hash_id: str,
    titulo: str,
    uri: str,
    fuente: str,
    duracion: float | None = None,
    tags: str | None = None,
    tiempos_json: str = "{}",
    thumbnail: bytes | None = None,
) -> None:
    conn = _conectar()
    try:
        conn.execute(
            """
            INSERT INTO recursos (hash, titulo, uri, fuente, duracion, tags, tiempos_json, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
                titulo       = excluded.titulo,
                uri          = excluded.uri,
                fuente       = excluded.fuente,
                duracion     = excluded.duracion,
                tags         = excluded.tags,
                tiempos_json = excluded.tiempos_json,
                thumbnail    = excluded.thumbnail
            """,
            (hash_id, titulo, uri, fuente, duracion, tags, tiempos_json, thumbnail),
        )
        conn.commit()
    finally:
        conn.close()


def escribir_chunks(filas: list[dict], tabla: str) -> None:
    chunks_tbl = _nombre_chunks(tabla)
    rows = [
        (
            fila["id"],
            fila["hash"],
            fila["chunk_idx"],
            fila["texto"],
            fila["inicio"],
            fila["fin"],
            fila["fin"] - fila["inicio"] if fila.get("fin") is not None and fila.get("inicio") is not None else None,
            fila.get("segmentos_json", "[]"),
        )
        for fila in filas
    ]
    conn = _conectar()
    try:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {chunks_tbl}
                (id, hash, chunk_idx, texto, inicio, fin, duracion, segmentos_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def enriquecer(filas: list[dict], tabla: str) -> list[dict]:
    """Merge chunk + resource metadata into LanceDB result rows (joined by id)."""
    if not filas or not INDICE_DB.exists():
        return filas
    ids = [f["id"] for f in filas if f.get("id")]
    if not ids:
        return filas

    chunks_tbl = _nombre_chunks(tabla)
    conn = _conectar()
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.chunk_idx, c.texto, c.inicio, c.fin, c.segmentos_json,
                   r.titulo, r.uri, r.fuente
            FROM {chunks_tbl} c
            JOIN recursos r ON c.hash = r.hash
            WHERE c.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    finally:
        conn.close()

    meta = {row["id"]: dict(row) for row in rows}
    return [{**dict(fila), **meta.get(fila.get("id"), {})} for fila in filas]


def borrar_hash(hash_id: str, tabla: str) -> None:
    """Remove a resource's chunks for the given model table, and the resource itself."""
    chunks_tbl = _nombre_chunks(tabla)
    conn = _conectar()
    try:
        conn.execute(f"DELETE FROM {chunks_tbl} WHERE hash = ?", (hash_id,))
        conn.execute("DELETE FROM recursos WHERE hash = ?", (hash_id,))
        conn.commit()
    finally:
        conn.close()
