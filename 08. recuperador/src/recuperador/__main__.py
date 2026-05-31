import argparse

from compartido.embedders import listar_ids
from compartido.rutas import DESCARGAS_DIR

from recuperador.busqueda import buscar

RESULTADOS_DIR = DESCARGAS_DIR / "resultados"

MODOS = ["rrf", "denso", "bm25"]


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
		help="rrf (fusion de ranks densa+BM25), denso (solo coseno) o bm25 (solo keywords). Default: denso.",
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

	args = parser.parse_args()

	if args.backend != "lance":
		print(f"Backend '{args.backend}' no soportado en esta version. haciendo fallback a 'lance'.")
		args.backend = "lance"

	
	semantica, sintactica = buscar(args.embedder, args.query, args.modo, args.top_k)
	filas = sintactica if args.modo == "bm25" else semantica
	imprimir_resultados(args.query, args.modo, filas)

	

def imprimir_resultados(query, modo, filas):
	print(f"\nResultados para la consulta: '{query}' (modo: {modo}):")
	if not filas:
		print("Sin resultados.")
		return

	score_label = "BM25" if modo == "bm25" else "Coseno"

	for i, fila in enumerate(filas, start=1):
		titulo = fila.get("titulo") or fila.get("fuente") or "(sin titulo)"
		chunk_idx = fila.get("chunk_idx")
		chunk_txt = f" [chunk {chunk_idx}]" if chunk_idx is not None else ""
		score = fila.get("score")
		score_txt = f"{score_label}: {score:.4f}" if isinstance(score, (float, int)) else f"{score_label}: n/a"
		texto = fila.get("texto", "")
		preview = texto[:500] + ("..." if len(texto) > 500 else "")
		print(f"{i}. {titulo}{chunk_txt} — {score_txt}")
		print(f"   {preview}\n")

	

if __name__ == "__main__":
	main()