#!/usr/bin/env python3
"""
Continúa el pipeline para uno o más hashes desde el punto en que quedaron,
sin borrar nada de las bases de datos.

Lee el status actual de cada hash en descargas/registros.json y solo ejecuta
las etapas pendientes.

Uso:
    python continuar.py                           # todos los hashes del registro
    python continuar.py --hash abc123 --hash def456
    python continuar.py --desde asr               # fuerza inicio desde esa etapa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
REGISTROS = ROOT / "descargas" / "registros.json"
EMBEDDERS = ["qwen3-0.6b", "bge-m3", "e5-large-instruct", "granite-107m", "jina-v3"]

# Mapa de nombre de etapa → status mínimo que debe tener un hash para necesitarla
# (es decir: el hash está en status X y todavía no llegó a X+1)
ETAPAS_ORDEN = ["procesar", "asr", "corr", "frag", "vect", "indexar"]

# Status que produce cada etapa al completarse
STATUS_COMPLETADO = {
    "procesar": 2,
    "asr":      3,
    "corr":     4,
    "frag":     5,
    "vect":     6,
    "indexar":  7,
}

# Status mínimo que un hash debe tener para ser elegible para una etapa
STATUS_REQUERIDO = {
    "procesar": 1,
    "asr":      2,
    "corr":     3,
    "frag":     4,
    "vect":     5,
    "indexar":  6,
}

NOMBRES_LEGIBLES = {
    0: "iniciado (sin descargar)",
    1: "descargado",
    2: "audio procesado",
    3: "transcrito",
    4: "corregido",
    5: "fragmentado",
    6: "vectorizado",
    7: "indexado (completo)",
}


# ─── Utilidades de prompt ────────────────────────────────────────────────────

def _print_seccion(titulo: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {titulo}")
    print(f"{'═' * 70}")


def preguntar(
    pregunta: str,
    *,
    default: str | None = None,
    opciones: list[str] | None = None,
    descripcion: str | None = None,
) -> str:
    if descripcion:
        print(f"  · {descripcion}")
    if opciones:
        print(f"  · opciones: {', '.join(opciones)}")
    sufijo = f" [{default}]" if default is not None else ""
    while True:
        try:
            resp = input(f"  > {pregunta}{sufijo}: ").strip()
        except EOFError:
            resp = ""
        if not resp:
            if default is None:
                print("    (respuesta vacía no permitida)")
                continue
            resp = default
        if opciones and resp not in opciones:
            print(f"    valor inválido. usar una de: {', '.join(opciones)}")
            continue
        return resp


def preguntar_bool(pregunta: str, *, default: bool = False, descripcion: str | None = None) -> bool:
    if descripcion:
        print(f"  · {descripcion}")
    d = "s" if default else "n"
    while True:
        try:
            resp = input(f"  > {pregunta} (s/n) [{d}]: ").strip().lower()
        except EOFError:
            resp = ""
        if not resp:
            return default
        if resp in ("s", "si", "sí", "y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("    responder s/n")


def preguntar_int(pregunta: str, *, default: int | None = None, descripcion: str | None = None) -> int | None:
    if descripcion:
        print(f"  · {descripcion}")
    sufijo = f" [{default}]" if default is not None else " [auto]"
    while True:
        try:
            resp = input(f"  > {pregunta}{sufijo}: ").strip()
        except EOFError:
            resp = ""
        if not resp:
            return default
        try:
            return int(resp)
        except ValueError:
            print("    debe ser un entero")


def preguntar_float(pregunta: str, *, default: float, descripcion: str | None = None) -> float:
    if descripcion:
        print(f"  · {descripcion}")
    while True:
        try:
            resp = input(f"  > {pregunta} [{default}]: ").strip()
        except EOFError:
            resp = ""
        if not resp:
            return default
        try:
            return float(resp)
        except ValueError:
            print("    debe ser un número")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def leer_registros() -> dict:
    if not REGISTROS.exists():
        print(f"[ERROR] No se encontró {REGISTROS}", file=sys.stderr)
        sys.exit(1)
    return json.loads(REGISTROS.read_text(encoding="utf-8"))


def hash_flags(hashes: list[str]) -> list[str]:
    out: list[str] = []
    for h in hashes:
        out += ["--hash", h]
    return out


def fmt_tiempo(seg: float) -> str:
    if seg < 60:
        return f"{seg:.2f}s"
    m, s = divmod(int(seg), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def ejecutar(cmd: list[str]) -> tuple[int, float]:
    print(f"\n  $ {' '.join(_quote(c) for c in cmd)}\n")
    t0 = time.perf_counter()
    res = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    return res.returncode, elapsed


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s else s


# ─── Configuración de etapas ─────────────────────────────────────────────────

def cfg_procesador() -> dict:
    _print_seccion("Config: procesar (VAD)")
    print("  · Reduce ruido y segmenta el audio en intervalos de voz.")
    metodo = preguntar(
        "Método VAD (-m)",
        default="silero",
        opciones=["energia", "webrtc", "silero", "ninguno"],
        descripcion="energia=rápido/básico, webrtc=medio, silero=preciso.",
    )
    return {"metodo": None if metodo == "ninguno" else metodo}


def cfg_transcriptor() -> dict:
    _print_seccion("Config: asr (transcripción)")
    modelo = preguntar(
        "Modelo (-m)",
        default="whisper:turbo",
        opciones=["vosk", "whisper:base", "whisper:small", "whisper:turbo", "cohere"],
        descripcion="vosk=CPU sin puntuación, whisper:*=GPU/CPU, cohere=alta calidad GPU.",
    )
    batch_size: int | None = None
    if modelo == "cohere":
        batch_size = preguntar_int(
            "Batch size (--batch-size, enter=automático)",
            default=None,
            descripcion="Dejar vacío para calcular automáticamente.",
        )
    return {"modelo": modelo, "batch_size": batch_size}


def cfg_corrector() -> dict:
    _print_seccion("Config: corr (corrector de puntuación)")
    backend = preguntar(
        "Backend (--m)",
        default="silero",
        opciones=["silero", "p-all"],
        descripcion="silero=liviano y rápido (CPU), p-all=punctuate-all+spaCy.",
    )
    return {"backend": backend}


def cfg_fragmentador() -> dict:
    _print_seccion("Config: frag (fragmentador)")
    todos = preguntar_bool(
        "Ejecutar para todos los embedders",
        default=False,
        descripcion="Corre frag+vect+indexar con cada embedder.",
    )
    if not todos:
        embedder: str | None = preguntar(
            "Embedder objetivo (--embedder)",
            default="jina-v3",
            opciones=EMBEDDERS,
        )
    else:
        embedder = None
    estrategia = preguntar(
        "Estrategia (--estrategia)",
        default="semantico",
        opciones=["tamano_fijo", "semantico"],
    )
    chunk_tokens = preguntar_int(
        "Tokens máximos por chunk (enter=recomendado por modelo)",
        default=None,
    )
    extra: dict = {"chunk_tokens": chunk_tokens}
    if estrategia == "tamano_fijo":
        extra["overlap"] = preguntar_int("Overlap % (--overlap)", default=20)
    else:
        extra["umbral"] = preguntar_float("Umbral semántico (--umbral)", default=0.5)
        extra["min_tokens"] = preguntar_int("Mínimo de tokens por chunk (--min-tokens)", default=64)
        if not todos:
            extra["boundary_embedder"] = preguntar(
                "Boundary embedder (enter=mismo embedder)",
                default=embedder,
                opciones=EMBEDDERS,
            )
        else:
            extra["boundary_embedder"] = None
    return {"todos_embedders": todos, "embedder": embedder, "estrategia": estrategia, **extra}


def cfg_vectorizador() -> dict:
    _print_seccion("Config: vect (vectorizador)")
    batch_size = preguntar_int("Batch size (--batch-size)", default=16)
    normalizar = preguntar_bool("Normalizar embeddings a norma 1", default=True)
    return {"batch_size": batch_size, "normalizar": normalizar}


def cfg_indexador() -> dict:
    _print_seccion("Config: indexar")
    print("  · NOTA: no se usará --recrear para no borrar datos existentes.")
    backend = preguntar(
        "Backend (--backend)",
        default="lance",
        opciones=["lance", "qdrant", "milvus"],
    )
    db = preguntar("Ruta o URI de la base (enter=default)", default="")
    tabla = preguntar("Nombre de tabla (enter=id del embedder)", default="")
    tags: list[str] = []
    if preguntar_bool("¿Agregar tags (--tag clave=valor)?", default=False):
        print("    (enter vacío para terminar)")
        while True:
            t = input("    tag clave=valor: ").strip()
            if not t:
                break
            if "=" not in t:
                print("    formato inválido, usar clave=valor")
                continue
            tags.append(t)
    return {"backend": backend, "db": db or None, "tabla": tabla or None, "tags": tags}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Continúa el pipeline para hashes que no llegaron al final, "
            "sin borrar datos existentes."
        )
    )
    parser.add_argument(
        "--hash",
        action="append",
        dest="hashes",
        metavar="HASH",
        default=None,
        help="Hash a continuar. Se puede repetir. Si se omite, usa todos los del registro.",
    )
    parser.add_argument(
        "--desde",
        choices=ETAPAS_ORDEN,
        default=None,
        help="Forzar inicio desde esta etapa (ignora el status actual).",
    )
    parser.add_argument(
        "--forzar-cpu",
        action="store_true",
        dest="forzar_cpu",
        help="Pasar --forzar-cpu a los módulos que lo soportan.",
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  continuar.py — retoma el pipeline desde el último status conocido  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    registros = leer_registros()

    # Determinar el conjunto de hashes a procesar
    if args.hashes:
        hashes_pedidos = args.hashes
        desconocidos = [h for h in hashes_pedidos if h not in registros]
        if desconocidos:
            print(f"[WARN] Hashes no encontrados en registros.json: {desconocidos}", file=sys.stderr)
        hashes_pedidos = [h for h in hashes_pedidos if h in registros]
    else:
        hashes_pedidos = list(registros.keys())

    if not hashes_pedidos:
        print("[ERROR] No hay hashes para procesar.", file=sys.stderr)
        return 1

    # ── Mostrar resumen de estados ────────────────────────────────────────
    _print_seccion("Estado actual de los hashes")
    conteo_por_status: dict[int, list[str]] = {}
    for h in hashes_pedidos:
        st = registros[h].get("status", 0)
        conteo_por_status.setdefault(st, []).append(h)

    for st in sorted(conteo_por_status):
        lbl = NOMBRES_LEGIBLES.get(st, f"status {st}")
        hs = conteo_por_status[st]
        print(f"  status {st} ({lbl}): {len(hs)} hash(es)")
        for h in hs:
            titulo = registros[h].get("title", "")
            print(f"      · {h}  {titulo}")

    # ── Determinar qué etapas son necesarias ─────────────────────────────
    if args.desde:
        # El usuario fuerza un punto de partida: todas las etapas desde ahí
        idx_desde = ETAPAS_ORDEN.index(args.desde)
        etapas_necesarias = ETAPAS_ORDEN[idx_desde:]
        # Para cada etapa forzada, los hashes elegibles son los que tienen el
        # status requerido (o mayor) — no retrotraemos a estados anteriores.
        hashes_por_etapa: dict[str, list[str]] = {}
        for etapa in etapas_necesarias:
            req = STATUS_REQUERIDO[etapa]
            elegibles = [h for h in hashes_pedidos if registros[h].get("status", 0) >= req]
            if elegibles:
                hashes_por_etapa[etapa] = elegibles
    else:
        # Modo normal: cada hash solo ejecuta las etapas que aún le faltan
        hashes_por_etapa = {}
        for etapa in ETAPAS_ORDEN:
            completado = STATUS_COMPLETADO[etapa]
            req = STATUS_REQUERIDO[etapa]
            pendientes = [
                h for h in hashes_pedidos
                if registros[h].get("status", 0) >= req
                and registros[h].get("status", 0) < completado
            ]
            if pendientes:
                hashes_por_etapa[etapa] = pendientes

    if not hashes_por_etapa:
        print("\n  Todos los hashes ya tienen status 7 (completo). Nada que hacer.")
        return 0

    print(f"\n  Etapas a ejecutar:")
    for etapa in ETAPAS_ORDEN:
        if etapa in hashes_por_etapa:
            print(f"    · {etapa}: {len(hashes_por_etapa[etapa])} hash(es)")

    if not preguntar_bool("\n¿Continuar con la configuración de parámetros?", default=True):
        print("Cancelado.")
        return 0

    # ── Recoger parámetros solo para las etapas necesarias ───────────────
    cpu_flag = ["--forzar-cpu"] if args.forzar_cpu else []

    c_proc  = cfg_procesador()   if "procesar" in hashes_por_etapa else None
    c_asr   = cfg_transcriptor() if "asr"      in hashes_por_etapa else None
    c_corr  = cfg_corrector()    if "corr"     in hashes_por_etapa else None
    c_frag  = cfg_fragmentador() if "frag"     in hashes_por_etapa else None
    c_vect  = cfg_vectorizador() if "vect"     in hashes_por_etapa else None
    c_idx   = cfg_indexador()    if "indexar"  in hashes_por_etapa else None

    # Si frag fue configurado, también necesitamos vect e indexar aunque no
    # estuvieran en hashes_por_etapa (frag/vect/indexar son una unidad por embedder).
    if c_frag and c_vect is None:
        c_vect = cfg_vectorizador()
    if c_frag and c_idx is None:
        c_idx = cfg_indexador()

    # ── Resumen ───────────────────────────────────────────────────────────
    _print_seccion("Resumen de configuración")
    if c_proc:
        print(f"  procesar : vad={c_proc['metodo']}")
    if c_asr:
        bs = f" (batch={c_asr['batch_size']})" if c_asr.get("batch_size") else ""
        print(f"  asr      : modelo={c_asr['modelo']}{bs}")
    if c_corr:
        print(f"  corr     : backend={c_corr['backend']}")
    if c_frag:
        e_lbl = "TODOS" if c_frag["todos_embedders"] else c_frag["embedder"]
        print(f"  frag     : estrategia={c_frag['estrategia']} embedder={e_lbl}")
    if c_vect:
        print(f"  vect     : batch={c_vect['batch_size']} norm={c_vect['normalizar']}")
    if c_idx:
        print(f"  indexar  : backend={c_idx['backend']} tags={c_idx['tags']}")
    print(f"  forzar-cpu: {args.forzar_cpu}")

    if not preguntar_bool("\n¿Ejecutar?", default=True):
        print("Cancelado.")
        return 0

    tiempos: dict[str, float] = {}
    t_total_0 = time.perf_counter()

    # ── procesar ─────────────────────────────────────────────────────────
    if c_proc and "procesar" in hashes_por_etapa:
        _print_seccion(f"Ejecutando: procesar ({len(hashes_por_etapa['procesar'])} hashes)")
        hflags = hash_flags(hashes_por_etapa["procesar"])
        cmd = ["procesar", *hflags]
        if c_proc["metodo"]:
            cmd += ["-m", c_proc["metodo"]]
        rc, dt = ejecutar(cmd)
        tiempos["procesar"] = dt
        if rc != 0:
            print(f"[ERROR] procesar falló (código {rc})", file=sys.stderr)
            return rc

    # ── asr ──────────────────────────────────────────────────────────────
    if c_asr and "asr" in hashes_por_etapa:
        _print_seccion(f"Ejecutando: asr ({len(hashes_por_etapa['asr'])} hashes)")
        hflags = hash_flags(hashes_por_etapa["asr"])
        cmd = ["asr", *hflags, "-m", c_asr["modelo"], *cpu_flag]
        if c_asr.get("batch_size") is not None:
            cmd += ["--batch-size", str(c_asr["batch_size"])]
        rc, dt = ejecutar(cmd)
        tiempos["asr"] = dt
        if rc != 0:
            print(f"[ERROR] asr falló (código {rc})", file=sys.stderr)
            return rc

    # ── corr ─────────────────────────────────────────────────────────────
    if c_corr and "corr" in hashes_por_etapa:
        _print_seccion(f"Ejecutando: corr ({len(hashes_por_etapa['corr'])} hashes)")
        hflags = hash_flags(hashes_por_etapa["corr"])
        cmd = ["corr", *hflags, "--m", c_corr["backend"], *cpu_flag]
        rc, dt = ejecutar(cmd)
        tiempos["corr"] = dt
        if rc != 0:
            print(f"[ERROR] corr falló (código {rc})", file=sys.stderr)
            return rc

    # ── frag → vect → indexar (por embedder) ─────────────────────────────
    # Determinar el conjunto de hashes para el bloque frag/vect/indexar:
    # usamos el superset de los hashes necesarios en cualquiera de las tres.
    hashes_fvi: list[str] = []
    for etapa in ("frag", "vect", "indexar"):
        for h in hashes_por_etapa.get(etapa, []):
            if h not in hashes_fvi:
                hashes_fvi.append(h)

    if hashes_fvi and c_frag:
        embedders_a_procesar = EMBEDDERS if c_frag["todos_embedders"] else [c_frag["embedder"]]
        tiempos.setdefault("frag", 0.0)
        tiempos.setdefault("vect", 0.0)
        tiempos.setdefault("indexar", 0.0)
        hflags = hash_flags(hashes_fvi)

        for embedder in embedders_a_procesar:
            sufijo = f" [{embedder}]"

            # frag
            if "frag" in hashes_por_etapa:
                _print_seccion(f"Ejecutando: frag{sufijo}")
                cmd = [
                    "frag", *hflags,
                    "--embedder", embedder,
                    "--estrategia", c_frag["estrategia"],
                ]
                if c_frag.get("chunk_tokens") is not None:
                    cmd += ["--chunk-tokens", str(c_frag["chunk_tokens"])]
                if c_frag["estrategia"] == "tamano_fijo":
                    cmd += ["--overlap", str(c_frag["overlap"])]
                else:
                    cmd += [
                        "--umbral", str(c_frag["umbral"]),
                        "--min-tokens", str(c_frag["min_tokens"]),
                        "--boundary-embedder", c_frag.get("boundary_embedder") or embedder,
                    ]
                cmd += cpu_flag
                rc, dt = ejecutar(cmd)
                tiempos["frag"] += dt
                if rc != 0:
                    print(f"[ERROR] frag{sufijo} falló (código {rc})", file=sys.stderr)
                    return rc

            # vect
            if c_vect:
                _print_seccion(f"Ejecutando: vect{sufijo}")
                cmd = [
                    "vect", *hflags,
                    "--embedder", embedder,
                    "--batch-size", str(c_vect["batch_size"]),
                ]
                if not c_vect["normalizar"]:
                    cmd += ["--sin-normalizar"]
                cmd += cpu_flag
                rc, dt = ejecutar(cmd)
                tiempos["vect"] += dt
                if rc != 0:
                    print(f"[ERROR] vect{sufijo} falló (código {rc})", file=sys.stderr)
                    return rc

            # indexar — sin --recrear
            if c_idx:
                _print_seccion(f"Ejecutando: indexar{sufijo}")
                cmd = [
                    "indexar", *hflags,
                    "--embedder", embedder,
                    "--backend", c_idx["backend"],
                ]
                if c_idx["db"]:
                    cmd += ["--db", c_idx["db"]]
                if c_idx["tabla"]:
                    cmd += ["--tabla", c_idx["tabla"]]
                for t in c_idx["tags"]:
                    cmd += ["--tag", t]
                rc, dt = ejecutar(cmd)
                tiempos["indexar"] += dt
                if rc != 0:
                    print(f"[ERROR] indexar{sufijo} falló (código {rc})", file=sys.stderr)
                    return rc

    total = time.perf_counter() - t_total_0

    # ── Resumen final ─────────────────────────────────────────────────────
    _print_seccion("Pipeline completado")
    print("  Tiempos por módulo:")
    ancho = max((len(k) for k in tiempos), default=8)
    for k, v in tiempos.items():
        print(f"    {k.ljust(ancho)}  {fmt_tiempo(v):>12}")
    print(f"\n  TIEMPO TOTAL: {fmt_tiempo(total)}")
    print(f"\n  Hashes procesados ({len(hashes_pedidos)}):")
    for h in hashes_pedidos:
        titulo = registros[h].get("title", "")
        print(f"    · {h}  {titulo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
