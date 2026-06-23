"""Evalúa el pipeline de recuperación con un dataset de consultas con anotación temporal gold.

El gold se define ANTES del chunking como uno o varios intervalos temporales dentro de un
audio específico. El formato legacy usa (gold_inicio, gold_fin); el formato nuevo usa
gold_spans=[{"inicio": ..., "fin": ...}, ...]. Esto permite comparar cualquier combinación
de estrategia de fragmentación, embedder, modo de recuperación y reranker con el mismo dataset.

Por cada consulta se evalúan todas las posiciones del pool no truncado contra la unión de los
gold spans usando solapamiento temporal como proxy de relevancia.

Métricas computadas a k ∈ {5, 10, 20, 40}:
  - Hit@k       → 1 si la UNIÓN de los top-k chunks cubre >= umbral del gold
  - MRR@k       → 1/rango del primer chunk individual con overlap >= umbral
  - nDCG@k      → DCG/IDCG usando el overlap bruto como grado de relevancia
  - Coverage@k  → fracción del gold span cubierta por la unión temporal de los top-k chunks
  - EntryHit@k  → fracción de gold spans con al menos un chunk cuyo inicio caiga dentro del span
                  (± entry_tolerance s antes del inicio). Modela el UX real: el usuario hace
                  clic y empieza a ver desde ese timestamp, cubriendo el resto secuencialmente.
  - EntryMRR@k  → 1/rango del primer chunk con inicio dentro de algún gold span (± tolerancia)

Métricas adicionales (diagnóstico):
  - pool_recall     → 1 si algún chunk individual del pool cubre >= umbral (diagnóstico chunking)
  - pool_hit        → 1 si la unión del pool completo cubre >= umbral del gold (techo pipeline)
  - pool_coverage   → cobertura del gold por la unión del pool completo
  - pool_entry_hit  → fracción de gold spans con buen punto de entrada en el pool completo

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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows regardless of console encoding or pipe mode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RESULTADOS_DIR = ROOT / "resultados" / "ret_eval"

# Debe coincidir con FACTOR_OVERSAMPLING en recuperador/busqueda.py.
_FACTOR_OVERSAMPLING = 4

# k-valores fijos para los que se reportan métricas en todos los runs.
_EVAL_KS: tuple[int, ...] = (5, 10, 20, 40)

# Campos de alto peso que no aportan valor en la salida JSON.
_HEAVY_FIELDS = frozenset({"vector", "texto_bm25"})

GoldSpan = tuple[float, float]


def _safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "run"


# ─────────────────────── solapamiento temporal ───────────────────────────────

def _merge_intervals(intervals: list[GoldSpan]) -> list[GoldSpan]:
    if not intervals:
        return []
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _gold_duration(gold_spans: list[GoldSpan]) -> float:
    return sum(end - start for start, end in gold_spans)


def _gold_bounds(gold_spans: list[GoldSpan]) -> GoldSpan:
    if not gold_spans:
        return 0.0, 0.0
    return gold_spans[0][0], gold_spans[-1][1]


def _serialize_gold_spans(gold_spans: list[GoldSpan]) -> list[dict[str, float]]:
    return [
        {"inicio": round(start, 3), "fin": round(end, 3)}
        for start, end in gold_spans
    ]


def _extract_gold_spans(query: dict[str, Any]) -> list[GoldSpan]:
    qid = query.get("query_id", "?")
    raw_spans = query.get("gold_spans")
    spans: list[GoldSpan] = []

    if raw_spans is not None:
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError(f"Consulta {qid!r}: gold_spans debe ser una lista no vacía")
        for idx, span in enumerate(raw_spans, 1):
            if not isinstance(span, dict):
                raise ValueError(f"Consulta {qid!r}: gold_spans[{idx}] debe ser un objeto")
            if "inicio" not in span or "fin" not in span:
                raise ValueError(
                    f"Consulta {qid!r}: gold_spans[{idx}] debe incluir 'inicio' y 'fin'"
                )
            start = float(span["inicio"])
            end = float(span["fin"])
            if end <= start:
                raise ValueError(
                    f"Consulta {qid!r}: gold_spans[{idx}] tiene un intervalo inválido"
                )
            spans.append((start, end))
    else:
        if "gold_inicio" not in query or "gold_fin" not in query:
            raise ValueError(
                f"Consulta {qid!r}: debe definir gold_inicio/gold_fin o gold_spans"
            )
        start = float(query["gold_inicio"])
        end = float(query["gold_fin"])
        if end <= start:
            raise ValueError(f"Consulta {qid!r}: gold_fin debe ser mayor que gold_inicio")
        spans.append((start, end))

    merged = _merge_intervals(spans)
    if not merged:
        raise ValueError(f"Consulta {qid!r}: no hay gold spans válidos")
    return merged


def _overlap(chunk_inicio: float, chunk_fin: float, gold_spans: list[GoldSpan]) -> float:
    """Fracción del gold cubierta por el chunk (0–1)."""
    denom = _gold_duration(gold_spans)
    if denom <= 0 or chunk_fin <= chunk_inicio:
        return 0.0
    covered = 0.0
    for gold_inicio, gold_fin in gold_spans:
        covered += max(0.0, min(chunk_fin, gold_fin) - max(chunk_inicio, gold_inicio))
    return covered / denom


def _union_coverage(chunks: list[dict[str, Any]], gold_spans: list[GoldSpan]) -> float:
    """Fracción del gold cubierta por la UNIÓN temporal de los chunks."""
    denom = _gold_duration(gold_spans)
    if denom <= 0 or not chunks:
        return 0.0
    intervals: list[GoldSpan] = []
    for c in chunks:
        chunk_inicio = float(c.get("inicio") or 0.0)
        chunk_fin = float(c.get("fin") or 0.0)
        if chunk_fin <= chunk_inicio:
            continue
        for gold_inicio, gold_fin in gold_spans:
            s = max(chunk_inicio, gold_inicio)
            e = min(chunk_fin, gold_fin)
            if e > s:
                intervals.append((s, e))
    if not intervals:
        return 0.0
    merged = _merge_intervals(intervals)
    return min(1.0, sum(e - s for s, e in merged) / denom)


# ─────────────────────── funciones de métricas ───────────────────────────────

def _ndcg(overlaps: list[float]) -> float:
    """nDCG usando el overlap como grado de relevancia continuo (0–1).
    El IDCG se calcula sobre el propio conjunto recuperado (juicios incompletos).
    """
    dcg = sum(ov / math.log2(i + 2) for i, ov in enumerate(overlaps))
    idcg = sum(ov / math.log2(i + 2) for i, ov in enumerate(sorted(overlaps, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0


def _first_hit_rank(pool: list[dict[str, Any]], gold_spans: list[GoldSpan], threshold: float) -> int | None:
    """Rango 1-based del primer chunk que por sí solo supera el umbral."""
    for rank, chunk in enumerate(pool, 1):
        ov = _overlap(
            float(chunk.get("inicio") or 0.0),
            float(chunk.get("fin") or 0.0),
            gold_spans,
        )
        if ov >= threshold:
            return rank
    return None


def _k_needed_for_hit(pool: list[dict[str, Any]], gold_spans: list[GoldSpan], threshold: float) -> int | None:
    """Mínimo k 1-based cuya cobertura acumulada alcanza el umbral."""
    if threshold <= 0:
        return 1 if pool else None
    for k in range(1, len(pool) + 1):
        if _union_coverage(pool[:k], gold_spans) >= threshold:
            return k
    return None


def _entry_spans_covered(
    window: list[dict[str, Any]],
    gold_spans: list[GoldSpan],
    tolerance: float,
) -> float:
    """Fracción de gold spans para los que algún chunk de window es un buen punto de entrada.

    Un chunk es buen punto de entrada para un span si su inicio cae en
    [span_inicio - tolerance, span_fin]. Modela el UX real: el usuario hace clic y empieza
    a ver desde ese timestamp, cubriendo el resto de forma secuencial.
    """
    if not gold_spans:
        return 0.0
    covered = sum(
        1 for (g_start, g_end) in gold_spans
        if any(
            g_start - tolerance <= float(c.get("inicio") or 0.0) <= g_end
            for c in window
        )
    )
    return covered / len(gold_spans)


def _entry_first_rank(
    pool: list[dict[str, Any]],
    gold_spans: list[GoldSpan],
    tolerance: float,
) -> int | None:
    """Rango 1-based del primer chunk que es punto de entrada para algún gold span."""
    for rank, chunk in enumerate(pool, 1):
        chunk_inicio = float(chunk.get("inicio") or 0.0)
        if any(g_start - tolerance <= chunk_inicio <= g_end for (g_start, g_end) in gold_spans):
            return rank
    return None


def _metrics_at_k(
    pool: list[dict[str, Any]],
    gold_spans: list[GoldSpan],
    k: int,
    threshold: float,
    entry_tolerance: float,
) -> dict[str, float]:
    """Hit, MRR, nDCG, Coverage y métricas de entrada para los primeros k chunks del pool."""
    window = pool[:k]
    overlaps = [
        _overlap(
            float(c.get("inicio") or 0.0),
            float(c.get("fin") or 0.0),
            gold_spans,
        )
        for c in window
    ]
    mrr = next((1.0 / (i + 1) for i, ov in enumerate(overlaps) if ov >= threshold), 0.0)
    cov = _union_coverage(window, gold_spans)
    hit = 1.0 if cov >= threshold else 0.0
    entry_hit = _entry_spans_covered(window, gold_spans, entry_tolerance)
    entry_mrr = next(
        (
            1.0 / (i + 1)
            for i, c in enumerate(window)
            if any(
                g_start - entry_tolerance <= float(c.get("inicio") or 0.0) <= g_end
                for (g_start, g_end) in gold_spans
            )
        ),
        0.0,
    )
    return {
        "hit": round(hit, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(_ndcg(overlaps), 4),
        "coverage": round(cov, 4),
        "entry_hit": round(entry_hit, 4),
        "entry_mrr": round(entry_mrr, 4),
    }


def _all_metrics(
    pool: list[dict[str, Any]],
    gold_spans: list[GoldSpan],
    eval_ks: tuple[int, ...],
    threshold: float,
    entry_tolerance: float,
) -> dict[str, Any]:
    """Todas las métricas a todos los k-valores más las métricas de techo (pool completo)."""
    out: dict[str, Any] = {"pool_size": len(pool)}
    for k in eval_ks:
        for metric, val in _metrics_at_k(pool, gold_spans, k, threshold, entry_tolerance).items():
            out[f"{metric}_at_{k}"] = val
    # Techo: calculado a k = tamaño real del pool, independientemente de eval_ks.
    ceil_m = _metrics_at_k(pool, gold_spans, len(pool), threshold, entry_tolerance)
    out["pool_recall"] = 1.0 if _first_hit_rank(pool, gold_spans, threshold) is not None else 0.0
    out["pool_hit"] = ceil_m["hit"]
    out["pool_coverage"] = ceil_m["coverage"]
    out["pool_entry_hit"] = round(_entry_spans_covered(pool, gold_spans, entry_tolerance), 4)
    out["first_hit_rank"] = _first_hit_rank(pool, gold_spans, threshold)
    out["entry_first_rank"] = _entry_first_rank(pool, gold_spans, entry_tolerance)
    out["k_needed_for_hit"] = _k_needed_for_hit(pool, gold_spans, threshold)
    return out


def _aggregate(all_query_metrics: list[dict[str, Any]], eval_ks: tuple[int, ...]) -> dict[str, Any]:
    """Media aritmética de cada métrica sobre todas las consultas con status=ok."""
    if not all_query_metrics:
        return {}
    keys = (
        [f"{m}_at_{k}" for k in eval_ks for m in ("hit", "mrr", "ndcg", "coverage", "entry_hit", "entry_mrr")]
        + ["pool_recall", "pool_hit", "pool_coverage", "pool_entry_hit"]
    )
    result: dict[str, Any] = {}
    for key in keys:
        vals = [q[key] for q in all_query_metrics if key in q]
        result[key] = round(sum(vals) / len(vals), 4) if vals else None
    sizes = [q["pool_size"] for q in all_query_metrics if "pool_size" in q]
    result["avg_pool_size"] = round(sum(sizes) / len(sizes), 1) if sizes else None
    return result


# ─────────────────────── carga del dataset ───────────────────────────────────

def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = raw if isinstance(raw, list) else raw.get("queries", [])
    required = {"query_id", "query", "hash"}
    for i, q in enumerate(queries):
        missing = required - q.keys()
        if missing:
            raise ValueError(f"Consulta #{i} ({q.get('query_id', '?')!r}): faltan campos {missing}")
        _extract_gold_spans(q)
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Ejecuta la consulta y devuelve (pool_reranked, pool_pre_rerank).

    pool_pre_rerank es None si no se usó reranker.
    Ambos pools son NO truncados a top_k.
    Los campos vector y texto_bm25 se eliminan al anotar la salida, no aquí.
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
        # Preserve pre-rerank order (sorted by RRF/dense score, which stays on each item).
        pool_pre_rerank = sorted(pool, key=lambda x: float(x.get("score") or 0.0), reverse=True)
        pool = _rerank(query, pool, reranker, device=device)
        return pool, pool_pre_rerank

    return pool, None


# ─────────────────────── resumen en consola ──────────────────────────────────

def _print_summary(summary: dict[str, Any], config: dict[str, Any], eval_ks: tuple[int, ...]) -> None:
    n_ok = summary.get("num_ok", 0)
    n_total = summary.get("num_queries", 0)
    n_fail = summary.get("num_failed", 0)
    pre = summary.get("pre_rerank")
    col = 16 if pre else 9
    sep = "─" * (14 + col * len(eval_ks))
    print(f"\n{sep}")
    print(f"  Embedder: {config.get('embedder')}   |   Reranker: {config.get('reranker') or '—'}")
    chunk_cfg = config.get("chunking", {})
    print(f"  Umbral semántico: {chunk_cfg.get('umbral', '—')}   |   min_tokens: {chunk_cfg.get('min_tokens', '—')}")
    print(f"  {n_ok}/{n_total} consultas evaluadas  |  {n_fail} error(es)")

    def _print_table(title: str, data: dict[str, Any], reference: dict[str, Any] | None = None) -> None:
        print(f"\n  {title:<12}  " + "".join(f"k={k:<{col}}" for k in eval_ks))
        print(f"  {'─' * 12}  " + "".join("─" * col for _ in eval_ks))
        for metric in ("hit", "mrr", "ndcg", "coverage", "entry_hit", "entry_mrr"):
            row = f"  {metric:<12}  "
            for k in eval_ks:
                val = data.get(f"{metric}_at_{k}")
                if isinstance(val, float):
                    if reference:
                        ref_val = reference.get(f"{metric}_at_{k}")
                        if isinstance(ref_val, float):
                            delta = val - ref_val
                            txt = f"{val:.4f} ({delta:+.3f})"
                            row += f"{txt:<{col}}"
                            continue
                    txt = f"{val:.4f}"
                    row += f"{txt:<{col}}"
                else:
                    row += f"{'—':<{col}}"
            print(row)

    if pre:
        _print_table("(pre-rerank)", pre)
        _print_table("(reranked)", summary, reference=pre)
    else:
        _print_table("métrica", summary)

    pr = summary.get("pool_recall")
    ph = summary.get("pool_hit")
    pc = summary.get("pool_coverage")
    ps = summary.get("avg_pool_size")
    if ps is not None:
        print(f"\n  avg_pool_size (chunks antes de truncar a top_k):     {ps:.1f}")
    if pr is not None:
        print(f"  pool_recall   (diagnóstico chunking, 1 chunk >= threshold): {pr:.4f}")
    peh = summary.get("pool_entry_hit")
    if ph is not None:
        print(f"  pool_hit      (techo, union >= threshold):   {ph:.4f}")
    if peh is not None:
        print(f"  pool_entry_hit(techo, fraccion spans con entrada):  {peh:.4f}")
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
        "entry_tolerance": args.entry_tolerance,
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
    print(f"[INFO] eval_ks={list(_EVAL_KS)} | overlap_threshold={args.overlap_threshold} | entry_tolerance={args.entry_tolerance}s")
    print(f"[INFO] Consultas: {len(queries)}")

    query_results: list[dict[str, Any]] = []
    ok_metrics: list[dict[str, Any]] = []
    ok_metrics_pre: list[dict[str, Any]] = []
    n_failed = 0

    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        query_text = q["query"]
        gold_spans = _extract_gold_spans(q)
        gold_payload: dict[str, Any] = {"gold_spans": q.get("gold_spans") or _serialize_gold_spans(gold_spans)}
        if "gold_inicio" in q and "gold_fin" in q:
            gold_payload["gold_inicio"] = float(q["gold_inicio"])
            gold_payload["gold_fin"] = float(q["gold_fin"])

        print(f"  [{i}/{len(queries)}] {qid}: «{query_text[:72]}»")

        try:
            pool, pool_pre = _retrieve(
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
                "status": "error",
                "error": str(exc),
                **gold_payload,
            })
            continue

        metrics = _all_metrics(pool, gold_spans, _EVAL_KS, args.overlap_threshold, args.entry_tolerance)
        metrics_pre = _all_metrics(pool_pre, gold_spans, _EVAL_KS, args.overlap_threshold, args.entry_tolerance) if pool_pre is not None else None
        ok_metrics.append(metrics)
        if metrics_pre is not None:
            ok_metrics_pre.append(metrics_pre)

        result_entry = {
            "query_id": qid,
            "query": query_text,
            "hash": q["hash"],
            "status": "ok",
            "metrics": metrics,
            **({"metrics_pre_rerank": metrics_pre} if metrics_pre is not None else {}),
            **gold_payload,
        }
        if args.guardar_pool:
            result_entry["pool"] = pool
        query_results.append(result_entry)

        ndcg_delta = f" | Δndcg@10={metrics['ndcg_at_10'] - metrics_pre['ndcg_at_10']:+.3f}" if metrics_pre else ""
        print(
            f"    pool={len(pool)} | pool_recall={metrics['pool_recall']:.2f} | "
            f"entry_hit@10={metrics['entry_hit_at_10']:.2f} | "
            f"nDCG@10={metrics['ndcg_at_10']:.3f}{ndcg_delta}"
        )

    aggregate = _aggregate(ok_metrics, _EVAL_KS)
    aggregate_pre = _aggregate(ok_metrics_pre, _EVAL_KS) if ok_metrics_pre else None
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
            **({"pre_rerank": aggregate_pre} if aggregate_pre is not None else {}),
        },
        "queries": query_results,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stamp}_{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(output["summary"], config, _EVAL_KS)
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
        args.overlap_threshold = _ask_float("Umbral de overlap para hit@k y mrr@k (0–1)", default=0.5)

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
            "Cada consulta puede usar gold_inicio/gold_fin o gold_spans=[{inicio, fin}, ...].\n"
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
    parser.add_argument("--overlap-threshold", type=float, default=None, dest="overlap_threshold", help="Umbral overlap para hit@k y mrr@k. Default: 0.5.")
    parser.add_argument("--entry-tolerance", type=float, default=None, dest="entry_tolerance", help="Tolerancia en segundos antes del inicio del gold para entry_hit/entry_mrr. Default: 30.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar a las primeras N consultas.")
    parser.add_argument("--forzar-cpu", action="store_true", dest="forzar_cpu", help="Forzar CPU.")
    parser.add_argument("--run-name", default=None, dest="run_name", help="Nombre del run.")
    parser.add_argument("--output-root", default=str(RESULTADOS_DIR), dest="output_root", help=f"Directorio de salida. Default: {RESULTADOS_DIR}.")
    parser.add_argument(
        "--guardar-pool",
        action="store_true",
        dest="guardar_pool",
        help="Incluir el pool completo por consulta en el JSON de salida. Default: no.",
    )
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
        args.overlap_threshold = 0.5
    if args.entry_tolerance is None:
        args.entry_tolerance = 30.0
    # Chunking defaults — None signals "ask interactively".
    # chunk_estrategia, chunk_max_tokens, etc. stay None until _interactive_fill.

    _interactive_fill(args)
    run_eval(args)


if __name__ == "__main__":
    main()
