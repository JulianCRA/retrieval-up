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


def buscar_completo(
	query: str,
	embedder: str,
	modo: str,
	top_k: int = 10,
	reranker: str | None = None,
	peso_semantica: float = 0.7,
	device: str = "cpu",
) -> dict:
	"""Búsqueda completa con todos los scores necesarios para la GUI.

	Devuelve un dict con:
	  - resultados: lista de chunks con scores aplanados y segmentos parseados
	  - sort_options: modos de ordenación disponibles según los parámetros usados

	Raises:
		RuntimeError: si no existe una tabla para el embedder indicado.
	"""
	import json as _json
	from recuperador.fusionado import rrf, wrrf
	from recuperador.rerank import rerank as _rerank

	try:
		semantica, sintactica = buscar(embedder, query, modo, top_k, device=device)
	except SystemExit as exc:
		raise RuntimeError(
			f"No se pudo acceder al índice para el embedder «{embedder}». "
			"Comprueba que existe una tabla indexada con ese nombre."
		) from exc

	if modo in ("rrf", "hibrido"):
		filas = rrf(semantica, sintactica)
	elif modo == "wrrf":
		filas = wrrf(semantica, sintactica, peso_semantica=peso_semantica)
	elif modo == "bm25":
		filas = list(sintactica)
	else:  # denso
		filas = list(semantica)

	# Asignar rank pre-rerank (posición en la lista de fusión/recuperación)
	for rank, fila in enumerate(filas, 1):
		fila["rank_fusion"] = rank

	# Aplicar reranker si se solicitó
	if reranker:
		filas = _rerank(query, filas, reranker, device=device)
		for rank, fila in enumerate(filas, 1):
			fila["rank_reranker"] = rank

	filas = filas[:top_k]

	# Opciones de ordenación disponibles según modo+reranker
	sort_options: list[str] = []
	if reranker:
		sort_options.append("reranker")
	if modo in ("rrf", "wrrf", "hibrido"):
		sort_options.extend(["rrf", "semantico", "sintactico"])
	elif modo == "denso":
		sort_options.append("semantico")
	elif modo == "bm25":
		sort_options.append("sintactico")

	_EXCLUIR = {"vector", "texto_bm25"}
	resultados: list[dict] = []
	for fila in filas:
		item = {k: v for k, v in fila.items() if k not in _EXCLUIR}
		so = item.pop("scores_origen", None) or {}
		ro = item.pop("ranks_origen", None) or {}
		if modo in ("rrf", "wrrf", "hibrido"):
			item["score_denso"] = so.get("denso")
			item["score_bm25"]  = so.get("bm25")
			item["rank_denso"]  = ro.get("denso")
			item["rank_bm25"]   = ro.get("bm25")
		else:
			item["score_denso"] = fila.get("score") if modo == "denso" else None
			item["score_bm25"]  = fila.get("score") if modo == "bm25"  else None
			item["rank_denso"]  = None
			item["rank_bm25"]   = None
		# Parsear segmentos_json → lista
		segs_raw = item.pop("segmentos_json", None) or "[]"
		try:
			item["segmentos"] = _json.loads(segs_raw) if isinstance(segs_raw, str) else (segs_raw or [])
		except (TypeError, ValueError):
			item["segmentos"] = []
		resultados.append(item)

	return {
		"query":          query,
		"embedder":       embedder,
		"modo":           modo,
		"reranker":       reranker,
		"sort_options":   sort_options,
		"num_resultados": len(resultados),
		"resultados":     resultados,
	}