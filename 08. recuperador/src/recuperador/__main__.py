import argparse
from datetime import datetime

import numpy as np

from compartido.embedders import listar_ids
from compartido.sqlite_utils import registrar_seleccion
from compartido.utils import crear_perfil_hardware, cronometro_activo

from recuperador.busqueda import buscar, tokenizar_query_bm25, vectorizar_query
from recuperador.fusionado import rrf, wrrf
from recuperador.rerank import RERANKERS, rerank
from recuperador.resultados import guardar_resultado_json, guardar_resultado_db, imprimir_resultados

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
	parser.add_argument(
		"--forzar-cpu",
		action="store_true",
		dest="forzar_cpu",
		help="Forzar uso de CPU aunque haya GPU disponible (afecta al encoder de la query y al reranker).",
	)
	parser.add_argument(
		"--seleccionar",
		type=int,
		default=None,
		dest="seleccionar",
		metavar="RANK",
		help=(
			"Registra el resultado en la posicion RANK (1-based) como el elegido por "
			"el usuario. Alimenta la salida 'videos mas seleccionados' de analitica."
		),
	)

	args = parser.parse_args()

	if args.backend != "lance":
		print(f"Backend '{args.backend}' no soportado en esta version. haciendo fallback a 'lance'.")
		args.backend = "lance"

	forzado = {"device": "cpu"} if args.forzar_cpu else None
	device = crear_perfil_hardware(forzado=forzado)["device"]

	inicio = datetime.now()
	with cronometro_activo() as crono:
		semantica, sintactica = buscar(args.embedder, args.query, args.modo, args.top_k, device=device)

		if args.modo in ("rrf", "hibrido"):
			filas = rrf(semantica, sintactica)
		elif args.modo == "wrrf":
			filas = wrrf(semantica, sintactica, peso_semantica=args.peso_semantica)
		elif args.modo == "bm25":
			filas = sintactica
		else:
			filas = semantica

		if args.reranker:
			filas = rerank(args.query, filas, args.reranker, device=device)

		filas = filas[:args.top_k]
		tiempos = crono.resumen()

	fin = datetime.now()
	imprimir_resultados(args.query, args.modo, filas, reranker=args.reranker, inicio=inicio, fin=fin)
	guardar_resultado_json(args, filas, tiempos, inicio=inicio, fin=fin)

	# Enriquecer el registro de actividad para el modulo de analitica docente:
	# el vector de la query permite agrupar consultas similares sin recodificarlas.
	query_vector = None
	query_bm25 = None
	if args.modo in ("rrf", "wrrf", "hibrido", "denso"):
		vec = vectorizar_query(args.query, args.embedder, device=device)
		query_vector = np.asarray(vec, dtype=np.float32).tobytes()
	if args.modo in ("rrf", "wrrf", "hibrido", "bm25"):
		query_bm25 = tokenizar_query_bm25(args.query)

	busqueda_id = guardar_resultado_db(
		args, filas, tiempos, inicio=inicio, fin=fin,
		query_vector=query_vector, query_bm25=query_bm25,
	)

	if args.seleccionar is not None and busqueda_id is not None:
		if 1 <= args.seleccionar <= len(filas):
			sel_id = registrar_seleccion(busqueda_id, rank=args.seleccionar)
			print(f"[OK] Seleccion registrada (rank={args.seleccionar}, id={sel_id}).")
		else:
			print(f"[AVISO] --seleccionar {args.seleccionar} fuera de rango (1..{len(filas)}). Ignorado.")


if __name__ == "__main__":
	main()