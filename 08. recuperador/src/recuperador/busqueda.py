import lancedb

import compartido.indice_utils as iu
from compartido.embedders import cargar_sentence_transformer, get_spec
from compartido.bm25 import tokenizar, tokens_a_texto
from compartido.utils import cronometrar, medir

from compartido.rutas import INDICE_DIR

FACTOR_OVERSAMPLING = 4
	
@cronometrar(etiqueta="apertura_tabla")
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

def vectorizar_query(query, embedder_id, device: str = "cpu"):
	with medir("carga_modelo_query"):
		modelo = cargar_sentence_transformer(embedder_id, device=device)
	config = get_spec(embedder_id)
	query = query.strip()
	if config.prefijo_query:
		query = config.prefijo_query + query
	
	with medir("encode_query"):
		if config.tarea_query:
			vector = modelo.encode(query, normalize_embeddings=True, task=config.tarea_query)
		else:
			vector = modelo.encode(query, normalize_embeddings=True)

	vector = vector.astype("float32").tolist()
	return vector

@cronometrar(etiqueta="tokenizacion_query_bm25")
def tokenizar_query_bm25(query: str) -> str:
	tokens = tokenizar(query)
	return tokens_a_texto(tokens)

@cronometrar(etiqueta="busqueda_bm25")
def busqueda_sintactica(tabla, tokens_query, top_k=5):
	filas = (
		tabla.search(tokens_query, query_type="fts")
		.limit(top_k * FACTOR_OVERSAMPLING)
		.to_list()
	)

	resultados = []
	for fila in filas:
		item = dict(fila)
		score = item.get("_score")
		item["score"] = None if score is None else float(score)
		resultados.append(item)
	return resultados

@cronometrar(etiqueta="busqueda_densa")
def busqueda_semantica(tabla, embed_query, top_k=5):
	filas = (
		tabla.search(embed_query)
		.distance_type("cosine")
		.limit(top_k * FACTOR_OVERSAMPLING)
		.to_list()
	)

	resultados = []
	for fila in filas:
		item = dict(fila)
		distancia = item.get("_distance")
		item["score"] = None if distancia is None else 1.0 - float(distancia)
		resultados.append(item)
	return resultados

def buscar(nombre_embedder, query, modo, top_k=5, device: str = "cpu"):
	tabla = abrir_tabla(nombre_embedder)
	
	if modo == "bm25":
		tokens_query = tokenizar_query_bm25(query)
		resultados = iu.enriquecer(busqueda_sintactica(tabla, tokens_query, top_k), nombre_embedder)
		return None, resultados
	elif modo == "denso":
		vector_query = vectorizar_query(query, tabla.name, device=device)
		resultados = iu.enriquecer(busqueda_semantica(tabla, vector_query, top_k), nombre_embedder)
		return resultados, None
	elif modo in ("rrf", "wrrf", "hibrido"):
		tokens_query = tokenizar_query_bm25(query)
		vector_query = vectorizar_query(query, tabla.name, device=device)
		semantica = iu.enriquecer(busqueda_semantica(tabla, vector_query, top_k), nombre_embedder)
		sintactica = iu.enriquecer(busqueda_sintactica(tabla, tokens_query, top_k), nombre_embedder)
		return semantica, sintactica
        
	return None, None