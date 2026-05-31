import argparse
import json
from datetime import datetime

from compartido.embedders import listar_ids
from compartido.rutas import DESCARGAS_DIR, RESULTADOS_DIR
from compartido.utils import cronometro_activo

from recuperador.busqueda import buscar
from recuperador.fusionado import rrf, wrrf
from recuperador.rerank import RERANKERS, rerank

MODOS = ["rrf", "wrrf", "denso", "bm25"]


def main():
	parser = argparse.ArgumentParser(
		prog="recuperador",
		description="Busqueda hibrida (densa + BM25) sobre el indice vectorial.",
	)
	parser.add_argument("--query", required=True, help="Texto de la consulta.")
	parser.add_argument(
		"--embedder",
		required=True,
		choices=listar_ids(),
		help="Embedder con el que se indexo el contenido.",
	)
	parser.add_argument(
		"--modo",
		default="denso",
		choices=MODOS,
		help="rrf (fusion de ranks densa+BM25), wrrf (rrf ponderado), denso (solo coseno) o bm25 (solo keywords). Default: denso.",
	)
	parser.add_argument(
		"--peso-semantica",
		type=float,
		default=0.7,
		dest="peso_semantica",
		help="Peso del recuperador semantico en wrrf (0.0-1.0, default: 0.7).",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=5,
		dest="top_k",
		help="Numero de resultados a devolver (default: 5).",
	)
	parser.add_argument(
		"--backend",
		default="lance",
		choices=["lance", "qdrant", "milvus"],
		help="Backend de busqueda (default: lance).",
	)
	parser.add_argument(
		"--reranker",
		default=None,
		choices=list(RERANKERS),
		help=f"Reranker a aplicar tras la recuperacion: {', '.join(RERANKERS)}. Default: ninguno.",
	)

	args = parser.parse_args()

	if args.backend != "lance":
		print(f"Backend '{args.backend}' no soportado en esta version. haciendo fallback a 'lance'.")
		args.backend = "lance"

	with cronometro_activo() as crono:
		semantica, sintactica = buscar(args.embedder, args.query, args.modo, args.top_k)

		if args.modo in ("rrf", "hibrido"):
			filas = rrf(semantica, sintactica)
		elif args.modo == "wrrf":
			filas = wrrf(semantica, sintactica, peso_semantica=args.peso_semantica)
		elif args.modo == "bm25":
			filas = sintactica
		else:
			filas = semantica

		if args.reranker:
			filas = rerank(args.query, filas, args.reranker)

		filas = filas[:args.top_k]

		imprimir_resultados(args.query, args.modo, filas, reranker=args.reranker)
		guardar_resultado(args, filas, tiempos=crono.resumen())


def guardar_resultado(args, filas, tiempos=None):
	RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
	ahora = datetime.now()
	stamp = ahora.strftime("%Y%m%d_%H%M%S")
	nombre = f"{stamp}_{args.modo}_{args.embedder}.json"

	_EXCLUIR = {"vector", "texto_bm25"}

	resultados_limpios = []
	for rank, fila in enumerate(filas or [], start=1):
		item = {k: v for k, v in fila.items() if k not in _EXCLUIR}
		item["rank"] = rank
		resultados_limpios.append(item)

	peso = {"peso_semantica": args.peso_semantica} if args.modo == "wrrf" else {}

	documento = {
		"timestamp": ahora.isoformat(timespec="seconds"),
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


def imprimir_resultados(query, modo, filas, reranker=None):
	reranker_txt = f", reranker: {reranker}" if reranker else ""
	print(f"\nResultados para la consulta: '{query}' (modo: {modo}{reranker_txt}):")
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

	

if __name__ == "__main__":
	main()