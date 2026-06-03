import json
from datetime import datetime

from compartido.rutas import RESULTADOS_DB, RESULTADOS_DIR
from compartido.sqlite_utils import conectar, guardar_busqueda_completa

_EXCLUIR = {"vector", "texto_bm25"}


def _filas_limpias(filas) -> list[dict]:
	resultado = []
	for rank, fila in enumerate(filas or [], start=1):
		item = {k: v for k, v in fila.items() if k not in _EXCLUIR}
		item["rank"] = rank
		resultado.append(item)
	return resultado


def guardar_resultado_json(args, filas, tiempos, inicio: datetime, fin: datetime) -> None:
	"""Persiste el resultado de una búsqueda como archivo JSON (depuración/testeo)."""
	RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
	stamp = inicio.strftime("%Y%m%d_%H%M%S")
	nombre = f"{stamp}_{args.modo}_{args.embedder}.json"

	resultados_limpios = _filas_limpias(filas)
	peso = {"peso_semantica": args.peso_semantica} if args.modo == "wrrf" else {}

	documento = {
		"timestamp": inicio.isoformat(timespec="seconds"),
		"inicio": inicio.isoformat(timespec="seconds"),
		"fin": fin.isoformat(timespec="seconds"),
		"query": args.query,
		"embedder": args.embedder,
		"modo": args.modo,
		"top_k": args.top_k,
		**peso,
		"reranker": args.reranker,
		"num_resultados": len(resultados_limpios),
		"tiempos": tiempos or {},
		"resultados": resultados_limpios,
	}

	ruta = RESULTADOS_DIR / nombre
	try:
		with open(ruta, "w", encoding="utf-8") as f:
			json.dump(documento, f, indent=2, ensure_ascii=False)
		print(f"[OK] Resultado guardado en '{ruta}'.")
	except IOError as e:
		print(f"[ERROR] No se pudo guardar el resultado: {e}")


def guardar_resultado_db(args, filas, tiempos, inicio: datetime, fin: datetime) -> int | None:
	"""Persiste la búsqueda y sus resultados en la base de datos SQLite.

	El campo *query_vector* queda en NULL; un paso posterior puede rellenarlo
	con :func:`compartido.sqlite_utils.actualizar_query_vector`.

	Devuelve el id de la búsqueda insertada, o None si hubo error.
	"""
	resultados_limpios = _filas_limpias(filas)
	peso = args.peso_semantica if args.modo == "wrrf" else None

	datos = {
		"timestamp":     inicio.isoformat(timespec="seconds"),
		"inicio":        inicio.isoformat(timespec="seconds"),
		"fin":           fin.isoformat(timespec="seconds"),
		"query":         args.query,
		"query_vector":  None,
		"embedder":      args.embedder,
		"modo":          args.modo,
		"top_k":         args.top_k,
		"reranker":      args.reranker,
		"peso_semantica": peso,
		"tiempos":       tiempos or {},
	}

	try:
		conn = conectar(RESULTADOS_DB)
		busqueda_id = guardar_busqueda_completa(conn, datos, resultados_limpios)
		conn.close()
		print(f"[OK] Resultado guardado en base de datos (id={busqueda_id}).")
		return busqueda_id
	except Exception as e:
		print(f"[ERROR] No se pudo guardar en base de datos: {e}")
		return None


def imprimir_resultados(
	query: str,
	modo: str,
	filas,
	reranker: str | None = None,
	inicio: datetime | None = None,
	fin: datetime | None = None,
) -> None:
	reranker_txt = f", reranker: {reranker}" if reranker else ""
	inicio_txt = inicio.strftime("%H:%M:%S") if inicio else "?"
	fin_txt = fin.strftime("%H:%M:%S") if fin else "?"
	print(f"\nResultados para la consulta: '{query}' (modo: {modo}{reranker_txt})")
	print(f"Inicio: {inicio_txt}  |  Fin: {fin_txt}")

	if not filas:
		print("Sin resultados.")
		return

	for i, fila in enumerate(filas, start=1):
		titulo = fila.get("titulo") or fila.get("fuente") or "(sin titulo)"
		chunk_idx = fila.get("chunk_idx")
		chunk_txt = f" [chunk {chunk_idx}]" if chunk_idx is not None else ""
		score = fila.get("score")
		score_rerank = fila.get("score_rerank")

		if modo in ("rrf", "wrrf"):
			score_txt = f"RRF: {score:.6f}" if isinstance(score, (float, int)) else "RRF: n/a"
			scores_origen = fila.get("scores_origen", {})
			ranks_origen = fila.get("ranks_origen", {})
			score_denso = scores_origen.get("denso")
			score_bm25 = scores_origen.get("bm25")
			rank_denso = ranks_origen.get("denso")
			rank_bm25 = ranks_origen.get("bm25")
			origen_denso = f"coseno={score_denso:.4f} (rank {rank_denso})" if score_denso is not None else "ausente"
			origen_bm25 = f"bm25={score_bm25:.4f} (rank {rank_bm25})" if score_bm25 is not None else "ausente"
			detalle = f"   denso: {origen_denso} | sintactica: {origen_bm25}"
		elif modo == "bm25":
			score_txt = f"BM25: {score:.4f}" if isinstance(score, (float, int)) else "BM25: n/a"
			detalle = None
		else:
			score_txt = f"Coseno: {score:.4f}" if isinstance(score, (float, int)) else "Coseno: n/a"
			detalle = None

		rerank_txt = f" | rerank: {score_rerank:.4f}" if score_rerank is not None else ""
		texto = fila.get("texto", "")
		preview = texto[:500] + ("..." if len(texto) > 500 else "")
		print(f"{i}. {titulo}{chunk_txt} — {score_txt}{rerank_txt}")
		if detalle:
			print(detalle)
		print(f"   {preview}\n")
