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
    id            INTEGER PK AUTOINCREMENT
    busqueda_id   INTEGER FK → busquedas.id
    rank          INTEGER
    video_id      TEXT
    titulo        TEXT
    chunk_idx     INTEGER
    score         REAL
    score_rerank  REAL
    texto         TEXT
    scores_origen TEXT   – JSON {"denso": float, "bm25": float}
    ranks_origen  TEXT   – JSON {"denso": int,   "bm25": int}
"""

import json
import sqlite3
from pathlib import Path


def conectar(ruta: Path) -> sqlite3.Connection:
    """Abre (o crea) la base de datos en *ruta* y devuelve la conexión."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ruta)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def crear_tablas(conn: sqlite3.Connection) -> None:
    """Crea las tablas si no existen."""
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
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            busqueda_id   INTEGER NOT NULL REFERENCES busquedas(id) ON DELETE CASCADE,
            rank          INTEGER NOT NULL,
            video_id      TEXT,
            titulo        TEXT,
            chunk_idx     INTEGER,
            score         REAL,
            score_rerank  REAL,
            texto         TEXT,
            scores_origen TEXT,
            ranks_origen  TEXT
        );
    """)
    conn.commit()


def insertar_busqueda(conn: sqlite3.Connection, datos: dict) -> int:
    """Inserta una fila en *busquedas* y devuelve el id generado."""
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
            "timestamp":     datos["timestamp"],
            "inicio":        datos["inicio"],
            "fin":           datos["fin"],
            "query":         datos["query"],
            "query_bm25":    datos.get("query_bm25"),
            "query_vector":  datos.get("query_vector"),
            "embedder":      datos["embedder"],
            "modo":          datos["modo"],
            "top_k":         datos["top_k"],
            "reranker":      datos.get("reranker"),
            "peso_semantica": datos.get("peso_semantica"),
            "tiempos":       json.dumps(datos.get("tiempos") or {}, ensure_ascii=False),
        },
    )
    conn.commit()
    return cur.lastrowid


def insertar_resultados(conn: sqlite3.Connection, busqueda_id: int, filas: list[dict]) -> None:
    """Inserta las filas de resultados asociadas a una búsqueda."""
    rows = [
        (
            busqueda_id,
            fila.get("rank"),
            fila.get("video_id"),
            fila.get("titulo") or fila.get("fuente"),
            fila.get("chunk_idx"),
            fila.get("score"),
            fila.get("score_rerank"),
            fila.get("texto", ""),
            json.dumps(fila.get("scores_origen") or {}, ensure_ascii=False),
            json.dumps(fila.get("ranks_origen") or {}, ensure_ascii=False),
        )
        for fila in filas
    ]
    conn.executemany(
        """
        INSERT INTO resultados
            (busqueda_id, rank, video_id, titulo, chunk_idx,
             score, score_rerank, texto, scores_origen, ranks_origen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def guardar_busqueda_completa(
    conn: sqlite3.Connection,
    datos: dict,
    filas: list[dict],
) -> int:
    """Crea las tablas (si hacen falta), inserta la búsqueda y sus resultados.

    Devuelve el id de la fila en *busquedas*.
    """
    crear_tablas(conn)
    busqueda_id = insertar_busqueda(conn, datos)
    if filas:
        insertar_resultados(conn, busqueda_id, filas)
    return busqueda_id


def actualizar_query_vector(conn: sqlite3.Connection, busqueda_id: int, vector: bytes) -> None:
    """Rellena el campo *query_vector* de una búsqueda existente."""
    conn.execute(
        "UPDATE busquedas SET query_vector = ? WHERE id = ?",
        (vector, busqueda_id),
    )
    conn.commit()


def reset_database(ruta: Path, remove_file: bool = False) -> None:
    """Resetea la base de datos indicada por *ruta*.

    Si *remove_file* es True, el archivo de la base de datos se elimina.
    En caso contrario, se intentan eliminar las tablas `resultados` y
    `busquedas` dejando el archivo intacto.
    """
    ruta = Path(ruta)
    if remove_file:
        try:
            if ruta.exists():
                ruta.unlink()
                return
        except Exception as e:
            raise RuntimeError(f"No se pudo eliminar el archivo de la DB: {e}")

    # Si no eliminamos el archivo, abrimos la conexión y borramos las tablas.
    conn = conectar(ruta)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS resultados;
            DROP TABLE IF EXISTS busquedas;
            """
        )
        conn.commit()
    finally:
        conn.close()
