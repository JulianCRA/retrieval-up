import argparse

from compartido.embedders import listar_ids, cargar_sentence_transformer, listar_ids, get_spec
from compartido.rutas import DESCARGAS_DIR

from indexador.bm25 import tokenizar, tokens_a_texto

import lancedb

RESULTADOS_DIR = DESCARGAS_DIR / "resultados"


INDICE_DIR = DESCARGAS_DIR / "indice"
MODOS = ["hibrido", "denso", "bm25"]
UMBRAL_CONFIANZA = 0.20


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
		default="hibrido",
		choices=MODOS,
		help="hibrido (weighted fusion densa+BM25), denso (solo coseno) o bm25 (solo keywords). Default: hibrido.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=5,
		dest="top_k",
		help="Numero de resultados a devolver (default: 5).",
	)
	parser.add_argument(
		"--peso-denso",
		type=float,
		default=0.5,
		dest="peso_denso",
		help="Peso de la busqueda densa en modo hibrido (0.0-1.0). El resto va a BM25. Default: 0.5.",
	)
	parser.add_argument(
		"--backend",
		default="lance",
		choices=["lance", "qdrant", "milvus"],
		help="Backend de busqueda (default: lance).",
	)

	args = parser.parse_args()

	if args.modo == "hibrido" and not (0.0 <= args.peso_denso <= 1.0):
		print("Error: --peso-denso debe estar entre 0.0 y 1.0.")
		return

	if args.backend != "lance":
		print(f"Backend '{args.backend}' no soportado en esta version. haciendo fallback a 'lance'.")
		args.backend = "lance"

	
	vector_query = vectorizar_query(args.query, args.embedder)
	tokens_query = tokenizar_query_bm25(args.query)
	tabla = abrir_tabla(args.embedder)
	res = buscar(tabla, tokens_query)
	imprimir_resultados(args.query, args.modo, res)

		
def vectorizar_query(query, embedder_id):
	modelo = cargar_sentence_transformer(embedder_id)
	config = get_spec(embedder_id)
	query = query.strip()
	if config.prefijo_query:
		query = config.prefijo_query + query
	
	if config.tarea_query:
		vector = modelo.encode(query, normalize_embeddings=True, task=config.tarea_query)
	else:
		vector = modelo.encode(query, normalize_embeddings=True)

	vector = vector.astype("float32").tolist()
	return vector

def tokenizar_query_bm25(query: str) -> str:
	tokens = tokenizar(query)
	return tokens_a_texto(tokens)

def abrir_tabla(nombre):
	db = lancedb.connect(INDICE_DIR)
	tablas = list(db.list_tables().tables)

	print(f"Tablas disponibles en el indice: {tablas}")
	if nombre not in tablas:
		print(f"Error: No se encontro una tabla para el embedder '{nombre}' en el indice.")
		exit(1)
		return None

	tabla = db.open_table(nombre)
	if tabla.count_rows() == 0:
		print(f"Error: La tabla para el embedder '{nombre}' esta vacia.")
		exit(1)
		return None
	
	return tabla

def buscar(tabla, tokens_query, top_k=5):
	filas = (
		tabla.search(tokens_query, query_type="fts")
		.limit(top_k)
		.to_list()
	)

	resultados = []
	for fila in filas:
		item = dict(fila)
		score = item.get("_score")
		item["score"] = None if score is None else float(score)
		resultados.append(item)
	return resultados

def busqueda_semantica(tabla, embed_query, top_k=5):
	filas = (
		tabla.search(embed_query)
		.distance_type("cosine")
		.limit(top_k)
		.to_list()
	)

	resultados = []
	for fila in filas:
		item = dict(fila)
		distancia = item.get("_distance")
		item["score"] = None if distancia is None else 1.0 - float(distancia)
		resultados.append(item)
	return resultados

def imprimir_resultados(query, modo, filas):
	print(f"\nResultados para la consulta: '{query}' (modo: {modo}):")
	if not filas:
		print("Sin resultados.")
		return

	for i, fila in enumerate(filas, start=1):
		texto = fila.get("texto", "")
		score = fila.get("score")
		score_txt = f"Score: {score:.4f}" if isinstance(score, (float, int)) else "Score: n/a"

		
		preview = texto[:500] + ("..." if len(texto) > 500 else "")
		print(f"{i}. ({score_txt}) {preview}\n")

	



if __name__ == "__main__":
	main()