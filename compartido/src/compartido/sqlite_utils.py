"""Utilidades para la base de datos SQLite compartida.

Schema
------
busquedas
    id            INTEGER PK AUTOINCREMENT
    timestamp     TEXT   – ISO-8601 del momento de la búsqueda
    inicio        TEXT   – hora de inicio (ISO-8601)
    fin           TEXT   – hora de fin   (ISO-8601)
    query         TEXT   – texto de la consulta
    query_bm25    TEXT   – tokens BM25 de la query (se rellena en un paso posterior)
    query_vector  BLOB   – embedding de la query (se rellena en un paso posterior)
    embedder      TEXT   – id del modelo de embeddings usado para indexar
    modo          TEXT   – rrf / wrrf / denso / bm25
    top_k         INTEGER
    reranker      TEXT   – nombre del reranker, o NULL
    peso_semantica REAL  – peso del recuperador semántico en wrrf, o NULL
    tiempos       TEXT   – JSON con desglose de tiempos

resultados
    id           INTEGER PK AUTOINCREMENT
    busqueda_id  INTEGER FK → busquedas.id
    rank         INTEGER
    video_id     TEXT
    titulo       TEXT
    chunk_idx    INTEGER
    score        REAL
    score_rerank REAL
    texto        TEXT
    score_denso  REAL   – score coseno original (NULL si modo bm25)
    score_bm25   REAL   – score BM25 original   (NULL si modo denso)
    rank_denso   INTEGER
    rank_bm25    INTEGER

selecciones
    id           INTEGER PK AUTOINCREMENT
    busqueda_id  INTEGER FK → busquedas.id
    resultado_id INTEGER FK → resultados.id (NULL si solo se conoce el video)
    rank         INTEGER  – posición del resultado elegido en el ranking mostrado
    video_id     TEXT     – hash del recurso seleccionado
    chunk_idx    INTEGER
    timestamp    TEXT     – ISO-8601 del momento de la selección

La tabla *selecciones* registra la interacción posterior a la búsqueda (el
resultado que el usuario usó como punto de acceso al contenido). Es la fuente
de la salida «vídeos más seleccionados» del módulo de analítica docente.
"""

import json
import sqlite3
from datetime import datetime

from compartido.rutas import RESULTADOS_DB


def _conectar() -> sqlite3.Connection:
    RESULTADOS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RESULTADOS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def crear_tablas() -> None:
    """Crea las tablas si no existen."""
    conn = _conectar()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS busquedas (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                inicio         TEXT    NOT NULL,
                fin            TEXT    NOT NULL,
                query          TEXT    NOT NULL,
                query_bm25     TEXT,
                query_vector   BLOB,
                embedder       TEXT    NOT NULL,
                modo           TEXT    NOT NULL,
                top_k          INTEGER NOT NULL,
                reranker       TEXT,
                peso_semantica REAL,
                tiempos        TEXT
            );

            CREATE TABLE IF NOT EXISTS resultados (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                busqueda_id  INTEGER NOT NULL REFERENCES busquedas(id) ON DELETE CASCADE,
                rank         INTEGER NOT NULL,
                video_id     TEXT,
                titulo       TEXT,
                chunk_idx    INTEGER,
                score        REAL,
                score_rerank REAL,
                texto        TEXT,
                score_denso  REAL,
                score_bm25   REAL,
                rank_denso   INTEGER,
                rank_bm25    INTEGER
            );

            CREATE TABLE IF NOT EXISTS selecciones (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                busqueda_id  INTEGER NOT NULL REFERENCES busquedas(id) ON DELETE CASCADE,
                resultado_id INTEGER REFERENCES resultados(id) ON DELETE SET NULL,
                rank         INTEGER,
                video_id     TEXT,
                chunk_idx    INTEGER,
                timestamp    TEXT
            );
        """)
        conn.commit()
        # Migraciones: añade columnas nuevas si la tabla ya existía sin ellas
        columnas = {row[1] for row in conn.execute("PRAGMA table_info(busquedas)")}
        if "query_bm25" not in columnas:
            conn.execute("ALTER TABLE busquedas ADD COLUMN query_bm25 TEXT")
            conn.commit()
        if "query_vector" not in columnas:
            conn.execute("ALTER TABLE busquedas ADD COLUMN query_vector BLOB")
            conn.commit()
        cols_res = {row[1] for row in conn.execute("PRAGMA table_info(resultados)")}
        for col, defn in [
            ("score_rerank", "REAL"),
            ("score_denso",  "REAL"),
            ("score_bm25",   "REAL"),
            ("rank_denso",   "INTEGER"),
            ("rank_bm25",    "INTEGER"),
        ]:
            if col not in cols_res:
                conn.execute(f"ALTER TABLE resultados ADD COLUMN {col} {defn}")
        conn.commit()
    finally:
        conn.close()


def insertar_busqueda(datos: dict) -> int:
    """Inserta una fila en *busquedas* y devuelve el id generado."""
    conn = _conectar()
    try:
        cur = conn.execute(
            """
            INSERT INTO busquedas
                (timestamp, inicio, fin, query, query_bm25, query_vector,
                 embedder, modo, top_k, reranker, peso_semantica, tiempos)
            VALUES
                (:timestamp, :inicio, :fin, :query, :query_bm25, :query_vector,
                 :embedder, :modo, :top_k, :reranker, :peso_semantica, :tiempos)
            """,
            {
                "timestamp":      datos["timestamp"],
                "inicio":         datos["inicio"],
                "fin":            datos["fin"],
                "query":          datos["query"],
                "query_bm25":     datos.get("query_bm25"),
                "query_vector":   datos.get("query_vector"),
                "embedder":       datos["embedder"],
                "modo":           datos["modo"],
                "top_k":          datos["top_k"],
                "reranker":       datos.get("reranker"),
                "peso_semantica": datos.get("peso_semantica"),
                "tiempos":        json.dumps(datos.get("tiempos") or {}, ensure_ascii=False),
            },
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def insertar_resultados(busqueda_id: int, filas: list[dict]) -> None:
    """Inserta las filas de resultados asociadas a una búsqueda."""
    rows = [
        (
            busqueda_id,
            fila.get("rank"),
            fila.get("video_id") or fila.get("hash"),
            fila.get("titulo") or fila.get("fuente"),
            fila.get("chunk_idx"),
            fila.get("score"),
            fila.get("score_rerank"),
            fila.get("texto", ""),
            (fila.get("scores_origen") or {}).get("denso"),
            (fila.get("scores_origen") or {}).get("bm25"),
            (fila.get("ranks_origen") or {}).get("denso"),
            (fila.get("ranks_origen") or {}).get("bm25"),
        )
        for fila in filas
    ]
    conn = _conectar()
    try:
        conn.executemany(
            """
            INSERT INTO resultados
                (busqueda_id, rank, video_id, titulo, chunk_idx,
                 score, score_rerank, texto,
                 score_denso, score_bm25, rank_denso, rank_bm25)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def guardar_busqueda_completa(datos: dict, filas: list[dict]) -> int:
    """Crea las tablas (si hacen falta), inserta la búsqueda y sus resultados.

    Devuelve el id de la fila en *busquedas*.
    """
    crear_tablas()
    busqueda_id = insertar_busqueda(datos)
    if filas:
        insertar_resultados(busqueda_id, filas)
    return busqueda_id


def actualizar_query_vector(busqueda_id: int, vector: bytes) -> None:
    """Rellena el campo *query_vector* de una búsqueda existente."""
    conn = _conectar()
    try:
        conn.execute(
            "UPDATE busquedas SET query_vector = ? WHERE id = ?",
            (vector, busqueda_id),
        )
        conn.commit()
    finally:
        conn.close()


def actualizar_query_bm25(busqueda_id: int, tokens: str) -> None:
    """Rellena el campo *query_bm25* de una búsqueda existente."""
    conn = _conectar()
    try:
        conn.execute(
            "UPDATE busquedas SET query_bm25 = ? WHERE id = ?",
            (tokens, busqueda_id),
        )
        conn.commit()
    finally:
        conn.close()


def registrar_seleccion(
    busqueda_id: int,
    video_id: str | None = None,
    chunk_idx: int | None = None,
    rank: int | None = None,
    resultado_id: int | None = None,
    timestamp: str | None = None,
) -> int:
    """Registra la interacción posterior a una búsqueda (resultado elegido).

    Vincula el resultado seleccionado por el usuario con la búsqueda que lo
    produjo. Si solo se conoce el *rank* del resultado mostrado, se resuelven
    *resultado_id*, *video_id* y *chunk_idx* a partir de la tabla *resultados*.

    Devuelve el id de la fila insertada en *selecciones*.
    """
    crear_tablas()
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    conn = _conectar()
    try:
        # Completar datos faltantes desde la fila de resultados correspondiente.
        if resultado_id is None and rank is not None:
            fila = conn.execute(
                "SELECT id, video_id, chunk_idx FROM resultados "
                "WHERE busqueda_id = ? AND rank = ?",
                (busqueda_id, rank),
            ).fetchone()
            if fila is not None:
                resultado_id = fila["id"]
                video_id = video_id or fila["video_id"]
                chunk_idx = chunk_idx if chunk_idx is not None else fila["chunk_idx"]

        cur = conn.execute(
            """
            INSERT INTO selecciones
                (busqueda_id, resultado_id, rank, video_id, chunk_idx, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (busqueda_id, resultado_id, rank, video_id, chunk_idx, timestamp),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def leer_busquedas() -> list[dict]:
    """Devuelve todas las búsquedas registradas (incluido *query_vector* en bytes)."""
    crear_tablas()
    conn = _conectar()
    try:
        filas = conn.execute("SELECT * FROM busquedas ORDER BY id").fetchall()
        return [dict(f) for f in filas]
    finally:
        conn.close()


def leer_resultados() -> list[dict]:
    """Devuelve todas las filas de resultados mostrados."""
    crear_tablas()
    conn = _conectar()
    try:
        filas = conn.execute("SELECT * FROM resultados ORDER BY busqueda_id, rank").fetchall()
        return [dict(f) for f in filas]
    finally:
        conn.close()


def leer_selecciones() -> list[dict]:
    """Devuelve todas las selecciones registradas."""
    crear_tablas()
    conn = _conectar()
    try:
        filas = conn.execute("SELECT * FROM selecciones ORDER BY id").fetchall()
        return [dict(f) for f in filas]
    finally:
        conn.close()


def reset_database() -> None:
    """Elimina y recrea las tablas, dejando la DB vacía."""
    conn = _conectar()
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS selecciones;
            DROP TABLE IF EXISTS resultados;
            DROP TABLE IF EXISTS busquedas;
        """)
        conn.commit()
    finally:
        conn.close()
    crear_tablas()
