"""Flask web GUI — browse resources + pipeline runner.

Run:
    python -m gui
"""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid

from flask import Flask, Response, abort, redirect, render_template, request, url_for

from compartido.rutas import ARCHIVO_REGISTRO, INDICE_DB
from recuperador.rerank import RERANKERS as RERANKERS_DISPONIBLES

app = Flask(__name__)

PAGE_SIZE = 24
EMBEDDERS = ["qwen3-0.6b", "bge-m3", "e5-large-instruct", "granite-107m", "jina-v3"]
MODOS_BUSQUEDA = ["rrf", "wrrf", "denso", "bm25"]
_ANSI = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJhs]")

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()
_search_jobs: dict[str, queue.Queue] = {}
_search_jobs_lock = threading.Lock()

# ─── Search helpers ──────────────────────────────────────────────────────────

def _tablas_lancedb() -> list[str]:
    """Devuelve la lista de tablas disponibles en el índice LanceDB."""
    try:
        import lancedb
        indice_dir = INDICE_DB.parent
        if not indice_dir.exists():
            return []
        db = lancedb.connect(indice_dir)
        return sorted(db.list_tables().tables)
    except Exception:
        return []


def _enriquecer_recursos(resultados: list[dict]) -> list[dict]:
    """Añade duracion_recurso y has_thumbnail a los items de resultados."""
    if not resultados:
        return resultados
    if not INDICE_DB.exists():
        for item in resultados:
            item.setdefault("duracion_recurso", None)
            item.setdefault("has_thumbnail", False)
        return resultados
    hashes = list({r["hash"] for r in resultados if r.get("hash")})
    if not hashes:
        return resultados
    placeholders = ",".join("?" * len(hashes))
    with _db() as conn:
        rows = conn.execute(
            f"SELECT hash, duracion, (thumbnail IS NOT NULL) AS has_thumbnail "
            f"FROM recursos WHERE hash IN ({placeholders})",
            hashes,
        ).fetchall()
    meta = {
        r["hash"]: {
            "duracion_recurso": r["duracion"],
            "has_thumbnail": bool(r["has_thumbnail"]),
        }
        for r in rows
    }
    for item in resultados:
        h = item.get("hash")
        if h and h in meta:
            item.update(meta[h])
        else:
            item.setdefault("duracion_recurso", None)
            item.setdefault("has_thumbnail", False)
    return resultados


def _run_search(params: dict, sq: queue.Queue) -> None:
    """Background thread: runs search pipeline emitting SSE-ready dicts."""
    import time as _time
    import json as _json

    def emit(**kwargs: object) -> None:
        sq.put(kwargs)

    t_total = _time.perf_counter()

    try:
        from recuperador.busqueda import (
            abrir_tabla, vectorizar_query,
            tokenizar_query_bm25, busqueda_semantica, busqueda_sintactica,
        )
        from recuperador.fusionado import rrf, wrrf
        from recuperador.rerank import rerank as _rerank
        import compartido.indice_utils as iu

        modo     = params["modo"]
        embedder = params["embedder"]
        query    = params["query"]
        top_k    = params["top_k"]
        reranker = params["reranker"]
        peso_sem = params["peso_semantica"]
        device   = params["device"]

        # ── Phase: table ──────────────────────────────────────────────────────
        emit(type="phase_start", phase="table", label="Abriendo índice…")
        t0 = _time.perf_counter()
        try:
            tabla = abrir_tabla(embedder)
        except SystemExit as exc:
            raise RuntimeError(
                f"No se encontró una tabla para el embedder \u00ab{embedder}\u00bb."
            ) from exc
        if tabla is None:
            raise RuntimeError(f"Tabla \u00ab{embedder}\u00bb vacía o no encontrada.")
        emit(type="phase_done", phase="table", elapsed=round(_time.perf_counter() - t0, 3))

        vector_query = None
        tokens_query = None

        # ── Phase: encode query ───────────────────────────────────────────────
        if modo in ("denso", "rrf", "wrrf", "hibrido"):
            emit(type="phase_start", phase="encode_query", label="Vectorizando consulta…")
            t0 = _time.perf_counter()
            vector_query = vectorizar_query(query, tabla.name, device=device)
            emit(type="phase_done", phase="encode_query", elapsed=round(_time.perf_counter() - t0, 3))

        # ── Phase: bm25 tokenize ──────────────────────────────────────────────
        if modo in ("bm25", "rrf", "wrrf", "hibrido"):
            emit(type="phase_start", phase="bm25_tokenize", label="Tokenizando BM25…")
            t0 = _time.perf_counter()
            tokens_query = tokenizar_query_bm25(query)
            emit(type="phase_done", phase="bm25_tokenize", elapsed=round(_time.perf_counter() - t0, 3))

        semantica  = None
        sintactica = None

        # ── Phase: dense search ───────────────────────────────────────────────
        if modo in ("denso", "rrf", "wrrf", "hibrido"):
            emit(type="phase_start", phase="search_dense", label="Búsqueda semántica…")
            t0 = _time.perf_counter()
            semantica = iu.enriquecer(busqueda_semantica(tabla, vector_query, top_k), embedder)
            emit(type="phase_done", phase="search_dense", elapsed=round(_time.perf_counter() - t0, 3),
                 n=len(semantica))

        # ── Phase: bm25 search ────────────────────────────────────────────────
        if modo in ("bm25", "rrf", "wrrf", "hibrido"):
            emit(type="phase_start", phase="search_bm25", label="Búsqueda BM25…")
            t0 = _time.perf_counter()
            sintactica = iu.enriquecer(busqueda_sintactica(tabla, tokens_query, top_k), embedder)
            emit(type="phase_done", phase="search_bm25", elapsed=round(_time.perf_counter() - t0, 3),
                 n=len(sintactica))

        # ── Phase: fusion ─────────────────────────────────────────────────────
        if modo in ("rrf", "hibrido"):
            emit(type="phase_start", phase="fusion", label="Fusión RRF…")
            t0 = _time.perf_counter()
            filas = rrf(semantica, sintactica)
            emit(type="phase_done", phase="fusion", elapsed=round(_time.perf_counter() - t0, 3))
        elif modo == "wrrf":
            emit(type="phase_start", phase="fusion", label="Fusión wRRF…")
            t0 = _time.perf_counter()
            filas = wrrf(semantica, sintactica, peso_semantica=peso_sem)
            emit(type="phase_done", phase="fusion", elapsed=round(_time.perf_counter() - t0, 3))
        elif modo == "bm25":
            filas = list(sintactica)
        else:  # denso
            filas = list(semantica)

        for rank, fila in enumerate(filas, 1):
            fila["rank_fusion"] = rank

        # ── Phase: rerank ─────────────────────────────────────────────────────
        if reranker:
            emit(type="phase_start", phase="rerank", label=f"Reranking ({reranker})…")
            t0 = _time.perf_counter()
            filas = _rerank(query, filas, reranker, device=device)
            emit(type="phase_done", phase="rerank", elapsed=round(_time.perf_counter() - t0, 3))
            for rank, fila in enumerate(filas, 1):
                fila["rank_reranker"] = rank

        filas = filas[:top_k]

        sort_options: list[str] = []
        if reranker:
            sort_options.append("reranker")
        if modo in ("rrf", "wrrf", "hibrido"):
            sort_options.extend(["rrf", "semantico", "sintactico"])
        elif modo == "denso":
            sort_options.append("semantico")
        elif modo == "bm25":
            sort_options.append("sintactico")

        _EXCLUIR = {"vector", "texto_bm25"}
        resultados: list[dict] = []
        for fila in filas:
            item = {k: v for k, v in fila.items() if k not in _EXCLUIR}
            so = item.pop("scores_origen", None) or {}
            ro = item.pop("ranks_origen", None) or {}
            if modo in ("rrf", "wrrf", "hibrido"):
                item["score_denso"] = so.get("denso")
                item["score_bm25"]  = so.get("bm25")
                item["rank_denso"]  = ro.get("denso")
                item["rank_bm25"]   = ro.get("bm25")
            else:
                item["score_denso"] = fila.get("score") if modo == "denso" else None
                item["score_bm25"]  = fila.get("score") if modo == "bm25"  else None
                item["rank_denso"]  = None
                item["rank_bm25"]   = None
            segs_raw = item.pop("segmentos_json", None) or "[]"
            try:
                item["segmentos"] = _json.loads(segs_raw) if isinstance(segs_raw, str) else (segs_raw or [])
            except (TypeError, ValueError):
                item["segmentos"] = []
            resultados.append(item)

        resultados = _enriquecer_recursos(resultados)
        total_elapsed = round(_time.perf_counter() - t_total, 3)

        emit(type="results", data={
            "query":          query,
            "embedder":       embedder,
            "modo":           modo,
            "reranker":       reranker,
            "sort_options":   sort_options,
            "num_resultados": len(resultados),
            "resultados":     resultados,
            "total_elapsed":  total_elapsed,
        })

    except Exception as exc:
        emit(type="error", message=str(exc))
    finally:
        sq.put(None)  # sentinel → SSE generator exits


# ─── Browse helpers ───────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(INDICE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _recursos_paginados(page: int, q: str = "") -> tuple[list[dict], int]:
    offset = (page - 1) * PAGE_SIZE
    like = f"%{q}%"
    with _db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM recursos WHERE titulo LIKE ?", (like,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT hash, titulo, uri, fuente, duracion, tags,
                   (thumbnail IS NOT NULL) AS has_thumbnail
            FROM recursos
            WHERE titulo LIKE ?
            ORDER BY titulo COLLATE NOCASE
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


# ─── Pipeline helpers ─────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _valid_uris(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "://" not in line:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _snapshot_hashes() -> set[str]:
    try:
        if not ARCHIVO_REGISTRO.exists():
            return set()
        return set(json.loads(ARCHIVO_REGISTRO.read_text(encoding="utf-8")).keys())
    except Exception:
        return set()


def _run_pipeline(params: dict, uris: list[str], q: queue.Queue) -> None:
    """Background thread: runs each pipeline stage and emits SSE-ready dicts."""
    import pathlib
    _log_path = pathlib.Path(__file__).resolve().parents[3] / "pipeline_gui.log"
    _log_fh = open(_log_path, "w", encoding="utf-8", buffering=1)
    _log_fh.write(f"uris: {uris}\nparams: {params}\n\n")

    def emit(**kwargs: object) -> None:
        q.put(kwargs)

    def emit_log(step: str, text: str) -> None:
        text = _strip_ansi(text).rstrip()
        if text:
            emit(type="log", step=step, text=text)

    def run_step(step: str, cmd: list[str]) -> int:
        emit(type="step_start", step=step)
        _log_fh.write(f"\n=== {step} ===\ncmd: {' '.join(cmd)}\n\n")
        t0 = time.perf_counter()
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except FileNotFoundError:
            emit_log(step, f"[ERROR] Comando no encontrado: {cmd[0]}")
            emit(type="step_done", step=step, rc=127, elapsed=0.0)
            return 127
        for line in proc.stdout:
            emit_log(step, line)
            _log_fh.write(line)
        proc.wait()
        elapsed = time.perf_counter() - t0
        _log_fh.write(f"\n--- rc={proc.returncode} elapsed={elapsed:.2f}s ---\n")
        emit(type="step_done", step=step, rc=proc.returncode, elapsed=round(elapsed, 2))
        return proc.returncode

    tmp_path: str | None = None
    try:
        embedders = EMBEDDERS if params["frag_todos"] else [params["frag_embedder"]]
        total_steps = 4 + len(embedders) * 3
        emit(type="run_started", total_steps=total_steps, uris=uris)

        cpu_flag = ["--forzar-cpu"] if params["forzar_cpu"] else []

        # Write URIs to a temporary .txt for the descargador
        fd, tmp_path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(uris) + "\n")

        # ── 1. descargar ──────────────────────────────────────────────────────
        pre = _snapshot_hashes()
        if (rc := run_step("descargar", ["descargar", "-s", tmp_path])) != 0:
            emit(type="pipeline_error", step="descargar", rc=rc)
            return

        os.unlink(tmp_path)
        tmp_path = None

        post = _snapshot_hashes()
        nuevos = sorted(post - pre)
        if not nuevos:
            emit(type="log", step="descargar",
                 text="[AVISO] No se detectaron hashes nuevos — el contenido puede ya estar descargado.")
            emit(type="pipeline_error", step="descargar", rc=0,
                 message="Sin hashes nuevos. Verifica la sección Recursos.")
            return

        emit(type="hashes", hashes=nuevos)
        hflags: list[str] = [flag for h in nuevos for flag in ("--hash", h)]

        # ── 2. procesar ───────────────────────────────────────────────────────
        cmd: list[str] = ["procesar", *hflags]
        if params["vad_metodo"] != "ninguno":
            cmd += ["-m", params["vad_metodo"]]
        if (rc := run_step("procesar", cmd)) != 0:
            emit(type="pipeline_error", step="procesar", rc=rc); return

        # ── 3. asr ────────────────────────────────────────────────────────────
        cmd = ["asr", *hflags, "-m", params["asr_modelo"], *cpu_flag]
        if params.get("asr_batch_size") and params["asr_modelo"] == "cohere":
            cmd += ["--batch-size", str(params["asr_batch_size"])]
        if (rc := run_step("asr", cmd)) != 0:
            emit(type="pipeline_error", step="asr", rc=rc); return

        # ── 4. corr ───────────────────────────────────────────────────────────
        cmd = ["corr", *hflags, "--m", params["corr_backend"], *cpu_flag]
        if (rc := run_step("corr", cmd)) != 0:
            emit(type="pipeline_error", step="corr", rc=rc); return

        # ── 5–7. frag + vect + indexar (per embedder) ─────────────────────────
        for emb in embedders:
            cmd = ["frag", *hflags, "--embedder", emb,
                   "--estrategia", params["frag_estrategia"]]
            if params.get("frag_chunk_tokens"):
                cmd += ["--chunk-tokens", str(params["frag_chunk_tokens"])]
            if params["frag_estrategia"] == "tamano_fijo":
                cmd += ["--overlap", str(params.get("frag_overlap", 20))]
            else:
                cmd += [
                    "--umbral", str(params.get("frag_umbral", 0.5)),
                    "--min-tokens", str(params.get("frag_min_tokens", 64)),
                    "--boundary-embedder", params.get("frag_boundary_embedder") or emb,
                ]
            cmd += cpu_flag
            if (rc := run_step(f"frag[{emb}]", cmd)) != 0:
                emit(type="pipeline_error", step=f"frag[{emb}]", rc=rc); return

            cmd = ["vect", *hflags, "--embedder", emb,
                   "--batch-size", str(params.get("vect_batch_size", 16))]
            if not params.get("vect_normalizar", True):
                cmd += ["--sin-normalizar"]
            cmd += cpu_flag
            if (rc := run_step(f"vect[{emb}]", cmd)) != 0:
                emit(type="pipeline_error", step=f"vect[{emb}]", rc=rc); return

            cmd = ["indexar", *hflags, "--embedder", emb,
                   "--backend", params.get("idx_backend", "lance")]
            if params.get("idx_db"):
                cmd += ["--db", params["idx_db"]]
            if params.get("idx_tabla"):
                cmd += ["--tabla", params["idx_tabla"]]
            if params.get("idx_recrear"):
                cmd += ["--recrear"]
            for tag in params.get("idx_tags", []):
                cmd += ["--tag", tag]
            if (rc := run_step(f"indexar[{emb}]", cmd)) != 0:
                emit(type="pipeline_error", step=f"indexar[{emb}]", rc=rc); return

        emit(type="pipeline_done", hashes=nuevos)

    except Exception as exc:
        emit(type="pipeline_error", step="?", rc=-1, message=str(exc))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        _log_fh.close()
        q.put(None)  # sentinel → SSE generator exits


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("browse"))


# ── Browse ────────────────────────────────────────────────────────────────────

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
        mime = "image/jpeg"

    resp = Response(data, mimetype=mime)
    resp.cache_control.max_age = 3600
    resp.cache_control.public = True
    return resp


# ── Pipeline ──────────────────────────────────────────────────────────────────

@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html", embedders=EMBEDDERS)


@app.route("/pipeline/submit", methods=["POST"])
def pipeline_submit():
    data = request.get_json(force=True)
    if not data:
        return {"error": "JSON body requerido"}, 400

    uris = _valid_uris(data.get("urls", ""))
    if not uris:
        return {"error": "No se encontraron URIs válidas (deben contener '://')"}, 400

    params = {
        "forzar_cpu":             bool(data.get("forzar_cpu", False)),
        "vad_metodo":             str(data.get("vad_metodo", "silero")),
        "asr_modelo":             str(data.get("asr_modelo", "whisper:turbo")),
        "asr_batch_size":         int(data["asr_batch_size"]) if data.get("asr_batch_size") else None,
        "corr_backend":           str(data.get("corr_backend", "silero")),
        "frag_todos":             bool(data.get("frag_todos", False)),
        "frag_embedder":          str(data.get("frag_embedder", "jina-v3")),
        "frag_estrategia":        str(data.get("frag_estrategia", "semantico")),
        "frag_chunk_tokens":      int(data["frag_chunk_tokens"]) if data.get("frag_chunk_tokens") else None,
        "frag_overlap":           int(data.get("frag_overlap", 20)),
        "frag_umbral":            float(data.get("frag_umbral", 0.5)),
        "frag_min_tokens":        int(data.get("frag_min_tokens", 64)),
        "frag_boundary_embedder": data.get("frag_boundary_embedder") or None,
        "vect_batch_size":        int(data.get("vect_batch_size", 16)),
        "vect_normalizar":        bool(data.get("vect_normalizar", True)),
        "idx_backend":            str(data.get("idx_backend", "lance")),
        "idx_db":                 data.get("idx_db") or None,
        "idx_tabla":              data.get("idx_tabla") or None,
        "idx_recrear":            bool(data.get("idx_recrear", False)),
        "idx_tags":               [t for t in data.get("idx_tags", []) if "=" in t],
    }

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = q

    threading.Thread(
        target=_run_pipeline, args=(params, uris, q), daemon=True
    ).start()

    return {"job_id": job_id}


# ── Search ───────────────────────────────────────────────────────────────────

@app.route("/search")
def search():
    tablas = _tablas_lancedb()
    return render_template(
        "search.html",
        tablas=tablas,
        rerankers=list(RERANKERS_DISPONIBLES.keys()),
    )


@app.route("/search/submit", methods=["POST"])
def search_submit():
    data = request.get_json(force=True)
    if not data:
        return {"error": "JSON body requerido"}, 400

    query = str(data.get("query", "")).strip()
    if not query:
        return {"error": "La consulta no puede estar vacía"}, 400

    embedder = str(data.get("embedder", "")).strip()
    if not embedder:
        return {"error": "Se requiere el embedder"}, 400

    modo = str(data.get("modo", "rrf"))
    if modo not in MODOS_BUSQUEDA:
        return {"error": f"Modo inválido: {modo!r}"}, 400

    try:
        top_k = max(1, min(50, int(data.get("top_k", 10))))
    except (ValueError, TypeError):
        top_k = 10

    reranker = data.get("reranker") or None
    if reranker and reranker not in RERANKERS_DISPONIBLES:
        return {"error": f"Reranker inválido: {reranker!r}"}, 400

    try:
        peso = float(data.get("peso_semantica", 0.7))
        peso = max(0.0, min(1.0, peso))
    except (ValueError, TypeError):
        peso = 0.7

    forzar_cpu = bool(data.get("forzar_cpu", False))

    from compartido.utils import crear_perfil_hardware
    device = crear_perfil_hardware(
        forzado={"device": "cpu"} if forzar_cpu else None
    )["device"]

    params = {
        "query":          query,
        "embedder":       embedder,
        "modo":           modo,
        "top_k":          top_k,
        "reranker":       reranker,
        "peso_semantica": peso,
        "device":         device,
    }

    job_id = str(uuid.uuid4())
    sq: queue.Queue = queue.Queue()
    with _search_jobs_lock:
        _search_jobs[job_id] = sq

    threading.Thread(
        target=_run_search, args=(params, sq), daemon=True
    ).start()

    return {"job_id": job_id}


@app.route("/search/stream/<job_id>")
def search_stream(job_id: str):
    with _search_jobs_lock:
        sq = _search_jobs.get(job_id)
    if sq is None:
        abort(404)

    def generate():
        while True:
            try:
                msg = sq.get(timeout=300)  # 5 min — model load can be slow
                if msg is None:
                    with _search_jobs_lock:
                        _search_jobs.pop(job_id, None)
                    yield 'data: {"type":"stream_end"}\n\n'
                    break
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/pipeline/stream/<job_id>")
def pipeline_stream(job_id: str):
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        abort(404)

    def generate():
        while True:
            try:
                msg = q.get(timeout=30)
                if msg is None:           # sentinel → pipeline thread finished
                    with _jobs_lock:
                        _jobs.pop(job_id, None)
                    yield 'data: {"type":"stream_end"}\n\n'
                    break
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
