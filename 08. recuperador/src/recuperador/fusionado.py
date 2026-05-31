def rrf(resultados_semantica, resultados_sintactica, top_k = 5, k = 60):
	# indexar por id
	indices_semantica: dict[str, dict] = {}
	for fila in resultados_semantica:
		doc_id = fila.get("id")
		if doc_id is not None and doc_id not in indices_semantica:
			indices_semantica[doc_id] = fila

	indices_sintactica: dict[str, dict] = {}
	for fila in resultados_sintactica:
		doc_id = fila.get("id")
		if doc_id is not None and doc_id not in indices_sintactica:
			indices_sintactica[doc_id] = fila

	# asignar rank regun posicion en cada lista
	ranks_semantica = {id_documento: rank for rank, id_documento in enumerate(indices_semantica, start=1)}
	ranks_sintactica = {id_documento: rank for rank, id_documento in enumerate(indices_sintactica, start=1)}

	# acumular score RRF por documento (union de ambas listas)
	combinado: dict[str, float] = {}
	for id_documento, rank in ranks_semantica.items():
		combinado[id_documento] = 1.0 / (k + rank)
	for id_documento, rank in ranks_sintactica.items():
		combinado[id_documento] = combinado.get(id_documento, 0.0) + 1.0 / (k + rank)

	# ordenar desc y truncar a top_k
	ordenados = sorted(combinado.items(), key=lambda x: x[1], reverse=True)[:top_k]

	# construir la lista de salida con metadatos de trazabilidad
	salida: list[dict] = []
	for doc_id, score in ordenados:
		# copiar los campos del documento (semantico tiene prioridad si aparece en ambos)
		base = dict(indices_semantica.get(doc_id) or indices_sintactica.get(doc_id) or {})
		base["score"] = float(score)
		# scores originales de cada recuperador (None si el doc no aparecio en esa lista)
		base["scores_origen"] = {
			"denso": (indices_semantica[doc_id].get("score") if doc_id in indices_semantica else None),
			"bm25": (indices_sintactica[doc_id].get("score") if doc_id in indices_sintactica else None),
		}
		# posicion en el ranking original de cada recuperador
		base["ranks_origen"] = {
			"denso": ranks_semantica.get(doc_id),
			"bm25": ranks_sintactica.get(doc_id),
		}
		salida.append(base)
	return salida

def wrrf(resultados_semantica, resultados_sintactica, top_k = 5, k = 60, peso_semantica = 0.7):
	# indexar por id
	indices_semantica: dict[str, dict] = {}
	for fila in resultados_semantica:
		doc_id = fila.get("id")
		if doc_id is not None and doc_id not in indices_semantica:
			indices_semantica[doc_id] = fila

	indices_sintactica: dict[str, dict] = {}
	for fila in resultados_sintactica:
		doc_id = fila.get("id")
		if doc_id is not None and doc_id not in indices_sintactica:
			indices_sintactica[doc_id] = fila

	# asignar rank regun posicion en cada lista
	ranks_semantica = {id_documento: rank for rank, id_documento in enumerate(indices_semantica, start=1)}
	ranks_sintactica = {id_documento: rank for rank, id_documento in enumerate(indices_sintactica, start=1)}

	# acumular score RRF por documento (union de ambas listas)
	combinado: dict[str, float] = {}
	for id_documento, rank in ranks_semantica.items():
		combinado[id_documento] = peso_semantica * (1.0 / (k + rank))
	for id_documento, rank in ranks_sintactica.items():
		combinado[id_documento] = combinado.get(id_documento, 0.0) + (1.0 - peso_semantica) * (1.0 / (k + rank))

	# ordenar desc y truncar a top_k
	ordenados = sorted(combinado.items(), key=lambda x: x[1], reverse=True)[:top_k]

	# construir la lista de salida con metadatos de trazabilidad
	salida: list[dict] = []
	for doc_id, score in ordenados:
		# copiar los campos del documento (semantico tiene prioridad si aparece en ambos)
		base = dict(indices_semantica.get(doc_id) or indices_sintactica.get(doc_id) or {})
		base["score"] = float(score)
		# scores originales de cada recuperador (None si el doc no aparecio en esa lista)
		base["scores_origen"] = {
			"denso": (indices_semantica[doc_id].get("score") if doc_id in indices_semantica else None),
			"bm25": (indices_sintactica[doc_id].get("score") if doc_id in indices_sintactica else None),
		}
		# posicion en el ranking original de cada recuperador
		base["ranks_origen"] = {
			"denso": ranks_semantica.get(doc_id),
			"bm25": ranks_sintactica.get(doc_id),
		}
		salida.append(base)
	return salida