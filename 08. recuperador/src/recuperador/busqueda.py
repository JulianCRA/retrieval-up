import lancedb

from compartido.embedders import cargar_sentence_transformer, get_spec
from indexador.bm25 import tokenizar, tokens_a_texto

from compartido.rutas import DESCARGAS_DIR

INDICE_DIR = DESCARGAS_DIR / "indice"

FACTOR_OVERSAMPLING = 4
	
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

def busqueda_sintactica(tabla, tokens_query, top_k=5):
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

def buscar(nombre_embedder, query, modo, top_k=5):
	tabla = abrir_tabla(nombre_embedder)
	
	if modo == "bm25":
		tokens_query = tokenizar_query_bm25(query)
		return None, busqueda_sintactica(tabla, tokens_query, top_k)
	elif modo == "denso":
		vector_query = vectorizar_query(query, tabla.name)
		return busqueda_semantica(tabla, vector_query, top_k), None
	elif modo == "rrf" or modo == "hibrido":
		tokens_query = tokenizar_query_bm25(query)
		vector_query = vectorizar_query(query, tabla.name)
		return busqueda_semantica(tabla, vector_query, top_k*FACTOR_OVERSAMPLING), busqueda_sintactica(tabla, tokens_query, top_k*FACTOR_OVERSAMPLING)
        
	return None, None