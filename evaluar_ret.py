"""Evalúa el pipeline de recuperación con un dataset de consultas con anotación temporal gold.

El gold se define ANTES del chunking como un intervalo temporal (gold_inicio, gold_fin)
en segundos dentro de un audio específico. Esto permite comparar cualquier combinación de
estrategia de fragmentación, embedder, modo de recuperación y reranker con el mismo dataset.

Por cada consulta se evalúan todas las posiciones del pool no truncado contra el gold span
usando solapamiento temporal como proxy de relevancia.

Métricas computadas a k ∈ {5, 10, 20, 40}:
  - Recall@k    → 1 si algún chunk en top-k tiene overlap >= umbral, si no 0
  - MRR@k       → 1/rango del primer chunk con overlap >= umbral
  - nDCG@k      → DCG/IDCG usando el overlap bruto como grado de relevancia
  - Coverage@k  → fracción del gold span cubierta por la unión temporal de los top-k chunks

Métricas adicionales (diagnóstico):
  - pool_recall   → Recall calculado sobre el pool completo antes de truncar (techo del pipeline)
  - pool_coverage → Coverage sobre el pool completo

Uso:
    python evaluar_ret.py \\
        --dataset eval_ds_ejemplo.json \\
        --embedder bge-m3 \\
        --modo rrf \\
        --top-k 10 \\
        --reranker bge
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTADOS_DIR = ROOT / "resultados" / "ret_eval"

# Debe coincidir con FACTOR_OVERSAMPLING en recuperador/busqueda.py.
_FACTOR_OVERSAMPLING = 4

# k-valores fijos para los que se reportan métricas en todos los runs.
_EVAL_KS: tuple[int, ...] = (5, 10, 20, 40)

# Campos de alto peso que no aportan valor en la salida JSON.
_HEAVY_FIELDS = frozenset({"vector", "texto_bm25"})


def _safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "run"


# ─────────────────────── solapamiento temporal ───────────────────────────────

def _overlap(chunk_inicio: float, chunk_fin: float, gold_inicio: float, gold_fin: float) -> float:
    """Fracción del gold span cubierta por el chunk (0–1)."""
    denom = gold_fin - gold_inicio
    if denom <= 0:
        return 0.0
    return max(0.0, min(chunk_fin, gold_fin) - max(chunk_inicio, gold_inicio)) / denom


def _union_coverage(chunks: list[dict[str, Any]], gold_inicio: float, gold_fin: float) -> float:
    """Fracción del gold span cubierta por la UNIÓN temporal de los chunks."""
    denom = gold_fin - gold_inicio
    if denom <= 0 or not chunks:
        return 0.0
    intervals: list[list[float]] = []
    for c in chunks:
        s = max(float(c.get("inicio") or 0.0), gold_inicio)
        e = min(float(c.get("fin") or 0.0), gold_fin)
        if e > s:
            intervals.append([s, e])
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return min(1.0, sum(e - s for s, e in merged) / denom)


# ─────────────────────── funciones de métricas ───────────────────────────────

def _ndcg(overlaps: list[float]) -> float:
    """nDCG usando el overlap como grado de relevancia continuo (0–1).
    El IDCG se calcula sobre el propio conjunto recuperado (juicios incompletos).
    """
    dcg = sum(ov / math.log2(i + 2) for i, ov in enumerate(overlaps))
    idcg = sum(ov / math.log2(i + 2) for i, ov in enumerate(sorted(overlaps, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0


def _metrics_at_k(
    pool: list[dict[str, Any]],
    gold_inicio: float,
    gold_fin: float,
    k: int,
    threshold: float,
) -> dict[str, float]:
    """Recall, MRR, nDCG y Coverage para los primeros k chunks del pool."""
    window = pool[:k]
    overlaps = [
        _overlap(
            float(c.get("inicio") or 0.0),
            float(c.get("fin") or 0.0),
            gold_inicio,
            gold_fin,
        )
        for c in window
    ]
    recall = 1.0 if any(ov >= threshold for ov in overlaps) else 0.0
    mrr = next((1.0 / (i + 1) for i, ov in enumerate(overlaps) if ov >= threshold), 0.0)
    cov = _union_coverage(window, gold_inicio, gold_fin)
    # hit@k: 1 if the UNION of top-k chunks covers >= threshold of the gold span.
    # Unlike recall (single-chunk), this fires when the answer is spread across chunks.
    hit = 1.0 if cov >= threshold else 0.0
    return {
        "recall": round(recall, 4),
        "hit": round(hit, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(_ndcg(overlaps), 4),
        "coverage": round(cov, 4),
    }


def _all_metrics(
    pool: list[dict[str, Any]],
    gold_inicio: float,
    gold_fin: float,
    eval_ks: tuple[int, ...],
    threshold: float,
) -> dict[str, Any]:
    """Todas las métricas a todos los k-valores más las métricas de techo (pool completo)."""
    out: dict[str, Any] = {"pool_size": len(pool)}
    for k in eval_ks:
        for metric, val in _metrics_at_k(pool, gold_inicio, gold_fin, k, threshold).items():
            out[f"{metric}_at_{k}"] = val
    # Techo: calculado a k = tamaño real del pool, independientemente de eval_ks.
    ceil_m = _metrics_at_k(pool, gold_inicio, gold_fin, len(pool), threshold)
    out["pool_recall"] = ceil_m["recall"]
    out["pool_hit"] = ceil_m["hit"]
    out["pool_coverage"] = ceil_m["coverage"]
    return out


def _aggregate(all_query_metrics: list[dict[str, Any]], eval_ks: tuple[int, ...]) -> dict[str, Any]:
    """Media aritmética de cada métrica sobre todas las consultas con status=ok."""
    if not all_query_metrics:
        return {}
    keys = (
        [f"{m}_at_{k}" for k in eval_ks for m in ("recall", "hit", "mrr", "ndcg", "coverage")]
        + ["pool_recall", "pool_hit", "pool_coverage"]
    )
    result: dict[str, Any] = {}
    for key in keys:
        vals = [q[key] for q in all_query_metrics if key in q]
        result[key] = round(sum(vals) / len(vals), 4) if vals else None
    return result


# ─────────────────────── carga del dataset ───────────────────────────────────

def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = raw if isinstance(raw, list) else raw.get("queries", [])
    required = {"query_id", "query", "hash", "gold_inicio", "gold_fin"}
    for i, q in enumerate(queries):
        missing = required - q.keys()
        if missing:
            raise ValueError(f"Consulta #{i} ({q.get('query_id', '?')!r}): faltan campos {missing}")
    return queries


# ─────────────────────── pipeline de recuperación ────────────────────────────

def _retrieve(
    query: str,
    embedder: str,
    modo: str,
    top_k: int,
    reranker: str | None,
    peso_semantica: float,
    device: str,
) -> list[dict[str, Any]]:
    """Ejecuta la consulta y devuelve el pool completo NO truncado.

    El pool incluye score_rerank si se aplicó reranker, y scores_origen / ranks_origen
    si se aplicó fusión. Los campos vector y texto_bm25 se eliminan al anotar la salida,
    no aquí, para mantener el tipo original intacto.
    """
    from recuperador.busqueda import buscar
    from recuperador.fusionado import rrf as _rrf, wrrf as _wrrf
    from recuperador.rerank import rerank as _rerank

    try:
        semantica, sintactica = buscar(embedder, query, modo, top_k, device=device)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc

    if modo in ("rrf", "hibrido"):
        pool = _rrf(semantica, sintactica)
    elif modo == "wrrf":
        pool = _wrrf(semantica, sintactica, peso_semantica=peso_semantica)
    elif modo == "bm25":
        pool = list(sintactica or [])
    else:  # denso
        pool = list(semantica or [])

    if reranker:
        pool = _rerank(query, pool, reranker, device=device)

    return pool  # intencionalemente NO truncado a top_k


# ─────────────────────── anotación de salida ─────────────────────────────────

def _annotate_pool(
    pool: list[dict[str, Any]],
    gold_inicio: float,
    gold_fin: float,
) -> list[dict[str, Any]]:
    """Añade rank y overlap a cada chunk del pool, elimina campos pesados y
    parsea segmentos_json → segmentos para legibilidad."""
    result: list[dict[str, Any]] = []
    for rank, chunk in enumerate(pool, 1):
        ov = _overlap(
            float(chunk.get("inicio") or 0.0),
            float(chunk.get("fin") or 0.0),
            gold_inicio,
            gold_fin,
        )
        entry: dict[str, Any] = {k: v for k, v in chunk.items() if k not in _HEAVY_FIELDS}
        seg_raw = entry.pop("segmentos_json", None)
        if seg_raw:
            try:
                entry["segmentos"] = json.loads(seg_raw) if isinstance(seg_raw, str) else seg_raw
            except (TypeError, ValueError):
                entry["segmentos"] = []
        entry["rank"] = rank
        entry["overlap"] = round(ov, 4)
        result.append(entry)
    return result


# ─────────────────────── resumen en consola ──────────────────────────────────

def _print_summary(summary: dict[str, Any], eval_ks: tuple[int, ...]) -> None:
    n_ok = summary.get("num_ok", 0)
    n_total = summary.get("num_queries", 0)
    n_fail = summary.get("num_failed", 0)
    col = 9
    sep = "─" * (14 + col * len(eval_ks))
    print(f"\n{sep}")
    print(f"  {n_ok}/{n_total} consultas evaluadas  |  {n_fail} error(es)")
    print(f"  {'métrica':<12}  " + "  ".join(f"k={k:<{col - 3}}" for k in eval_ks))
    print(f"  {'─' * 12}  " + "  ".join("─" * (col - 1) for _ in eval_ks))
    for metric in ("recall", "hit", "mrr", "ndcg", "coverage"):
        row = f"  {metric:<12}  "
        for k in eval_ks:
            val = summary.get(f"{metric}_at_{k}")
            row += f"{val:.4f}   " if isinstance(val, float) else f"{'—':>{col - 3}}   "
        print(row)
    pr = summary.get("pool_recall")
    ph = summary.get("pool_hit")
    pc = summary.get("pool_coverage")
    if pr is not None:
        print(f"\n  pool_recall   (techo, 1 chunk >= threshold): {pr:.4f}")
    if ph is not None:
        print(f"  pool_hit      (techo, union >= threshold):   {ph:.4f}")
    if pc is not None:
        print(f"  pool_coverage (techo, fraccion cubierta):    {pc:.4f}")
    print(f"{sep}\n")


# ─────────────────────── bucle principal ─────────────────────────────────────

def run_eval(args: argparse.Namespace) -> Path:
    from compartido.utils import crear_perfil_hardware

    forzado = {"device": "cpu"} if args.forzar_cpu else None
    perfil = crear_perfil_hardware(forzado=forzado)
    device = perfil["device"]
    out_root = Path(args.output_root)
    dataset_path = Path(args.dataset).resolve()

    queries = _load_queries(dataset_path)
    if args.limit:
        queries = queries[: args.limit]

    config: dict[str, Any] = {
        "embedder": args.embedder,
        "modo": args.modo,
        "top_k": args.top_k,
        "reranker": args.reranker,
        "peso_semantica": args.peso_semantica if args.modo == "wrrf" else None,
        "overlap_threshold": args.overlap_threshold,
        "eval_ks": list(_EVAL_KS),
        "dataset": str(dataset_path),
        "chunking": {
            "estrategia": args.chunk_estrategia,
            "max_tokens": args.chunk_max_tokens,
            "overlap_pct": args.chunk_overlap_pct,
            "umbral": args.chunk_umbral,
            "min_tokens": args.chunk_min_tokens,
        },
        "hardware": perfil,
    }

    print(
        f"[INFO] Embedder={args.embedder} | Modo={args.modo} | "
        f"Reranker={args.reranker or '—'} | top_k={args.top_k}"
    )
    print(f"[INFO] eval_ks={list(_EVAL_KS)} | overlap_threshold={args.overlap_threshold}")
    print(f"[INFO] Consultas: {len(queries)}")

    query_results: list[dict[str, Any]] = []
    ok_metrics: list[dict[str, Any]] = []
    n_failed = 0

    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        query_text = q["query"]
        gold_inicio = float(q["gold_inicio"])
        gold_fin = float(q["gold_fin"])

        print(f"  [{i}/{len(queries)}] {qid}: «{query_text[:72]}»")

        try:
            pool = _retrieve(
                query=query_text,
                embedder=args.embedder,
                modo=args.modo,
                top_k=args.top_k,
                reranker=args.reranker,
                peso_semantica=args.peso_semantica,
                device=device,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            n_failed += 1
            query_results.append({
                "query_id": qid,
                "query": query_text,
                "hash": q["hash"],
                "gold_inicio": gold_inicio,
                "gold_fin": gold_fin,
                "status": "error",
                "error": str(exc),
            })
            continue

        metrics = _all_metrics(pool, gold_inicio, gold_fin, _EVAL_KS, args.overlap_threshold)
        ok_metrics.append(metrics)

        query_results.append({
            "query_id": qid,
            "query": query_text,
            "hash": q["hash"],
            "gold_inicio": gold_inicio,
            "gold_fin": gold_fin,
            "status": "ok",
            "metrics": metrics,
            "pool": _annotate_pool(pool, gold_inicio, gold_fin),
        })

        print(
            f"    pool={len(pool)} | pool_recall={metrics['pool_recall']:.2f} | "
            f"recall@10={metrics['recall_at_10']:.2f} | "
            f"nDCG@10={metrics['ndcg_at_10']:.3f}"
        )

    aggregate = _aggregate(ok_metrics, _EVAL_KS)
    ts = datetime.now()
    run_id = args.run_name or _safe_slug(
        f"{args.embedder}-{args.modo}-{args.reranker or 'norerank'}"
    )
    stamp = ts.strftime("%Y%m%d_%H%M%S")

    output: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": ts.isoformat(timespec="seconds"),
        "config": config,
        "summary": {
            "num_queries": len(queries),
            "num_ok": len(ok_metrics),
            "num_failed": n_failed,
            **aggregate,
        },
        "queries": query_results,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stamp}_{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(output["summary"], _EVAL_KS)
    print(f"[OK] Guardado en: {out_path}")
    return out_path


# ─────────────────────── CLI / prompts interactivos ──────────────────────────

def _ask(prompt: str, default: str | None = None, choices: list[str] | None = None) -> str:
    """Pregunta una línea al usuario con prompt, default y validación opcional."""
    if choices:
        options = "  " + " | ".join(
            f"[{c}]" if c == default else c for c in choices
        )
        print(options)
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        value = raw if raw else (default or "")
        if choices and value not in choices:
            print(f"  Opción inválida. Elige entre: {', '.join(choices)}")
            continue
        return value


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Ingresa un número entero.")


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            return v
        except ValueError:
            print("  Ingresa un número decimal.")


def _ask_optional_int(prompt: str) -> int | None:
    raw = input(f"  {prompt} [Enter para omitir]: ").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _ask_optional_str(prompt: str) -> str | None:
    raw = input(f"  {prompt} [Enter para omitir]: ").strip()
    return raw if raw else None


def _ask_dataset_path() -> str:
    while True:
        raw = input("  Ruta al dataset JSON de consultas gold: ").strip().strip('"').strip("'")
        if not raw:
            print("  La ruta no puede estar vacía.")
            continue
        p = Path(raw)
        if not p.exists():
            print(f"  No se encontró el archivo '{p}'. Comprueba la ruta.")
            continue
        return str(p)


def _interactive_fill(args: argparse.Namespace) -> None:
    """Rellena interactivamente cualquier parámetro que no haya sido pasado por flag."""
    try:
        from compartido.embedders import listar_ids
        embedder_choices = listar_ids()
    except ImportError:
        embedder_choices = []

    try:
        from recuperador.rerank import RERANKERS
        reranker_choices = list(RERANKERS)
    except ImportError:
        reranker_choices = []

    modos = ["rrf", "denso", "bm25", "wrrf"]

    needs_prompt = (
        args.dataset is None
        or args.embedder is None
    )
    if not needs_prompt:
        return

    print("\n── Configuración del run ────────────────────────────────────────────────")

    if args.dataset is None:
        args.dataset = _ask_dataset_path()

    if args.embedder is None:
        print("\n  Embedder a evaluar:")
        args.embedder = _ask("Embedder", default=embedder_choices[0] if embedder_choices else None, choices=embedder_choices or None)

    if args.modo is None:
        print("\n  Modo de recuperación:")
        args.modo = _ask("Modo", default="rrf", choices=modos)

    if args.top_k is None:
        print()
        args.top_k = _ask_int("top_k (resultados finales al usuario)", default=10)

    if args.reranker is None and reranker_choices:
        opts = ["ninguno"] + reranker_choices
        print("\n  Reranker:")
        val = _ask("Reranker", default="ninguno", choices=opts)
        args.reranker = None if val == "ninguno" else val

    if args.modo == "wrrf" and args.peso_semantica is None:
        print()
        args.peso_semantica = _ask_float("Peso semántico para wrrf (0–1)", default=0.7)

    if args.overlap_threshold is None:
        print()
        args.overlap_threshold = _ask_float("Umbral de overlap para Recall/MRR (0–1)", default=0.8)

    # ── Chunking (solo registro, el índice ya fue construido manualmente) ──
    print("\n── Configuración de chunking (solo para registro) ─────────────────────")
    if args.chunk_estrategia is None:
        args.chunk_estrategia = _ask(
            "Estrategia de fragmentación",
            default="semantico",
            choices=["semantico", "tamano_fijo"],
        )
    if args.chunk_estrategia == "tamano_fijo":
        if args.chunk_max_tokens is None:
            args.chunk_max_tokens = _ask_int("max_tokens", default=512)
        if args.chunk_overlap_pct is None:
            args.chunk_overlap_pct = _ask_int("overlap_pct (%)", default=20)
    else:  # semantico
        if args.chunk_umbral is None:
            args.chunk_umbral = _ask_float("umbral (similitud coseno 0–1)", default=0.65)
        if args.chunk_min_tokens is None:
            args.chunk_min_tokens = _ask_int("min_tokens", default=100)
        if args.chunk_max_tokens is None:
            args.chunk_max_tokens = _ask_int("max_tokens (límite de seguridad)", default=512)

    if not args.forzar_cpu:
        raw = input("\n  ¿Forzar CPU aunque haya GPU? (s/n) [n]: ").strip().lower()
        args.forzar_cpu = raw in ("s", "si", "sí", "y", "yes")

    print()
    args.limit = _ask_optional_int("Limitar a las primeras N consultas (diagnóstico rápido)")
    args.run_name = _ask_optional_str("Nombre del run (para el archivo de salida)")

    print("─────────────────────────────────────────────────────────────────────────\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluar_ret.py",
        description=(
            "Evalúa el pipeline de recuperación con un dataset de consultas con anotación temporal gold.\n"
            "Métricas: Recall@k, MRR@k, nDCG@k, Coverage@k  a  k ∈ {5, 10, 20, 40}.\n\n"
            "Todos los parámetros son opcionales: si se omite cualquiera de los dos\n"
            "obligatorios (--dataset, --embedder), el script los pregunta interactivamente."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=None, help="Ruta al JSON de consultas gold.")
    parser.add_argument("--embedder", default=None, help="Embedder con el que está indexado el contenido.")
    parser.add_argument("--modo", default=None, choices=["denso", "bm25", "rrf", "wrrf"], help="Modo de recuperación. Default: rrf.")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k", help="k final. Default: 10.")
    parser.add_argument("--reranker", default=None, help="Reranker a aplicar. Default: ninguno.")
    parser.add_argument("--peso-semantica", type=float, default=None, dest="peso_semantica", help="Peso semántico para wrrf. Default: 0.7.")
    parser.add_argument("--overlap-threshold", type=float, default=None, dest="overlap_threshold", help="Umbral overlap para Recall/MRR. Default: 0.8.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar a las primeras N consultas.")
    parser.add_argument("--forzar-cpu", action="store_true", dest="forzar_cpu", help="Forzar CPU.")
    parser.add_argument("--run-name", default=None, dest="run_name", help="Nombre del run.")
    parser.add_argument("--output-root", default=str(RESULTADOS_DIR), dest="output_root", help=f"Directorio de salida. Default: {RESULTADOS_DIR}.")
    # Chunking — solo para registro en el JSON de resultados.
    parser.add_argument("--chunk-estrategia", default=None, dest="chunk_estrategia", choices=["semantico", "tamano_fijo"], help="Estrategia de fragmentación usada al indexar.")
    parser.add_argument("--chunk-max-tokens", type=int, default=None, dest="chunk_max_tokens", help="max_tokens del chunker.")
    parser.add_argument("--chunk-overlap-pct", type=int, default=None, dest="chunk_overlap_pct", help="overlap_pct para tamano_fijo.")
    parser.add_argument("--chunk-umbral", type=float, default=None, dest="chunk_umbral", help="Umbral coseno para estrategia semántica.")
    parser.add_argument("--chunk-min-tokens", type=int, default=None, dest="chunk_min_tokens", help="min_tokens para estrategia semántica.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Aplicar defaults para parámetros opcionales que no requieren prompt.
    if args.modo is None:
        args.modo = "rrf"
    if args.top_k is None:
        args.top_k = 10
    if args.peso_semantica is None:
        args.peso_semantica = 0.7
    if args.overlap_threshold is None:
        args.overlap_threshold = 0.8
    # Chunking defaults — None signals "ask interactively".
    # chunk_estrategia, chunk_max_tokens, etc. stay None until _interactive_fill.

    _interactive_fill(args)
    run_eval(args)


if __name__ == "__main__":
    main()
