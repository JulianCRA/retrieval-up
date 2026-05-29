import argparse

from compartido.embedders import listar_ids, cargar_sentence_transformer, listar_ids, get_spec
from compartido.rutas import DESCARGAS_DIR

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

	# embed_query = obtener_embed_query(args.query, args.embedder)
	# print(f"Embed de la query (size): {len(embed_query)}.")

	buscar(args, "embed_query")


		
def obtener_embed_query(query, embedder_id):
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

def buscar(args, embed_query):
	query = args.query.strip()
	modo = args.modo
	peso = args.peso_denso
	nombre_modelo = args.embedder

	db = lancedb.connect(INDICE_DIR)
	tablas = list(db.list_tables().tables)

	print(f"Tablas disponibles en el indice: {tablas}")
	if nombre_modelo not in tablas:
		print(f"Error: No se encontro una tabla para el embedder '{nombre_modelo}' en el indice.")
		return

	tabla = db.open_table(nombre_modelo)
	if tabla.count_rows() == 0:
		print(f"Error: La tabla para el embedder '{nombre_modelo}' esta vacia.")
		return
	
	print(tabla.schema)
	



if __name__ == "__main__":
	main()