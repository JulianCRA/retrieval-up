"""Salida 3: agrupacion de consultas semanticamente similares.

Las consultas registradas se codifican con uno de los modelos de embedding
disponibles y se normalizan en el espacio vectorial. Sobre estos vectores se
aplica HDBSCAN, que identifica grupos densos sin fijar de antemano el numero de
grupos y marca como ruido las consultas aisladas o poco frecuentes. Despues se
aplica KeyBERT sobre las consultas de cada grupo para extraer terminos
representativos que sirven como etiquetas interpretables.

El objetivo es que el docente reconozca temas recurrentes, dudas formuladas con
vocabularios distintos y posibles vacios del corpus (un grupo que acumula
busquedas sin resultados seleccionados).
"""
from __future__ import annotations

import numpy as np

from compartido.embedders import cargar_sentence_transformer, get_spec, listar_ids

from insights.registro import Actividad, decodificar_vector


def _normalizar(matriz: np.ndarray) -> np.ndarray:
	"""L2-normaliza por fila (vectores en la hiperesfera unidad)."""
	normas = np.linalg.norm(matriz, axis=1, keepdims=True)
	normas[normas == 0] = 1.0
	return matriz / normas


def _consultas_unicas(actividad: Actividad) -> list[dict]:
	"""Consultas distintas con su frecuencia, vector almacenado y selecciones.

	Se agrupa por (texto, embedder) para no contar dos veces la misma consulta.
	"""
	por_texto: dict[tuple[str, str], dict] = {}
	sel_por_busqueda: dict[int, int] = {}
	for s in actividad.selecciones:
		sel_por_busqueda[s["busqueda_id"]] = sel_por_busqueda.get(s["busqueda_id"], 0) + 1

	for b in actividad.busquedas:
		query = (b.get("query") or "").strip()
		if not query:
			continue
		clave = (query, b.get("embedder") or "")
		entrada = por_texto.get(clave)
		if entrada is None:
			entrada = {
				"query": query,
				"embedder": b.get("embedder") or "",
				"frecuencia": 0,
				"selecciones": 0,
				"vector": decodificar_vector(b.get("query_vector")),
			}
			por_texto[clave] = entrada
		entrada["frecuencia"] += 1
		entrada["selecciones"] += sel_por_busqueda.get(b["id"], 0)
		if entrada["vector"] is None:
			entrada["vector"] = decodificar_vector(b.get("query_vector"))

	return list(por_texto.values())


def _resolver_embedder(consultas: list[dict], embedder: str | None) -> str:
	if embedder:
		return embedder
	# Embedder mas usado entre las consultas; fallback al primero disponible.
	conteo: dict[str, int] = {}
	for c in consultas:
		e = c.get("embedder")
		if e:
			conteo[e] = conteo.get(e, 0) + c["frecuencia"]
	if conteo:
		return max(conteo, key=conteo.get)
	return listar_ids()[0]


def _matriz_embeddings(consultas: list[dict], embedder: str, device: str) -> np.ndarray:
	"""Devuelve la matriz de embeddings, reusando vectores almacenados cuando existan."""
	dim = get_spec(embedder).dim
	faltan_idx = [
		i for i, c in enumerate(consultas)
		if c.get("vector") is None or len(c["vector"]) != dim
	]

	if faltan_idx:
		modelo = cargar_sentence_transformer(embedder, device=device)
		spec = get_spec(embedder)
		textos = []
		for i in faltan_idx:
			q = consultas[i]["query"]
			textos.append(spec.prefijo_query + q if spec.prefijo_query else q)
		kwargs = {"normalize_embeddings": True}
		if spec.tarea_query:
			kwargs["task"] = spec.tarea_query
		nuevos = modelo.encode(textos, **kwargs)
		nuevos = np.asarray(nuevos, dtype=np.float32)
		for pos, i in enumerate(faltan_idx):
			consultas[i]["vector"] = nuevos[pos]

	matriz = np.vstack([np.asarray(c["vector"], dtype=np.float32) for c in consultas])
	return _normalizar(matriz)


def _etiquetas_keybert(textos: list[str], device: str, top_n: int, embedder: str) -> list[str]:
	"""Extrae terminos representativos de un grupo de consultas con KeyBERT."""
	try:
		from keybert import KeyBERT
	except ImportError:
		return []
	if not textos:
		return []

	modelo_st = cargar_sentence_transformer(embedder, device=device)
	kw = KeyBERT(model=modelo_st)
	documento = ". ".join(textos)
	pares = kw.extract_keywords(
		documento,
		keyphrase_ngram_range=(1, 2),
		stop_words=None,
		top_n=top_n,
	)
	return [termino for termino, _score in pares]


def agrupar_consultas(
	actividad: Actividad,
	embedder: str | None = None,
	device: str = "cpu",
	min_cluster_size: int = 3,
	min_samples: int | None = None,
	top_terminos: int = 5,
) -> dict:
	"""Agrupa consultas similares con HDBSCAN y las etiqueta con KeyBERT.

	Devuelve un dict con la lista de grupos (cada uno con sus consultas, tamano,
	terminos representativos y selecciones acumuladas) y las consultas marcadas
	como ruido. Si no hay suficientes consultas, devuelve una estructura vacia.
	"""
	consultas = _consultas_unicas(actividad)
	resultado: dict = {
		"embedder": None,
		"num_consultas": len(consultas),
		"num_grupos": 0,
		"grupos": [],
		"ruido": [],
	}

	if len(consultas) < max(min_cluster_size, 2):
		# No hay material suficiente para una agrupacion densa significativa.
		resultado["ruido"] = [
			{"query": c["query"], "frecuencia": c["frecuencia"], "selecciones": c["selecciones"]}
			for c in consultas
		]
		return resultado

	embedder = _resolver_embedder(consultas, embedder)
	resultado["embedder"] = embedder

	matriz = _matriz_embeddings(consultas, embedder, device)

	try:
		import hdbscan
	except ImportError as exc:
		raise RuntimeError(
			"hdbscan no esta instalado. Instala el modulo '09. insights' "
			"(pip install -e '09. insights')."
		) from exc

	clusterer = hdbscan.HDBSCAN(
		min_cluster_size=min_cluster_size,
		min_samples=min_samples,
		metric="euclidean",  # sobre vectores L2-normalizados ~ distancia coseno
	)
	etiquetas = clusterer.fit_predict(matriz.astype(np.float64))

	grupos: dict[int, list[int]] = {}
	ruido: list[int] = []
	for idx, lab in enumerate(etiquetas):
		if lab == -1:
			ruido.append(idx)
		else:
			grupos.setdefault(int(lab), []).append(idx)

	grupos_salida = []
	for lab, indices in grupos.items():
		textos = [consultas[i]["query"] for i in indices]
		frecuencia = sum(consultas[i]["frecuencia"] for i in indices)
		selecciones = sum(consultas[i]["selecciones"] for i in indices)
		terminos = _etiquetas_keybert(textos, device, top_terminos, embedder)
		grupos_salida.append({
			"grupo": lab,
			"tamano": len(indices),
			"frecuencia_total": frecuencia,
			"selecciones_total": selecciones,
			"sin_seleccion": selecciones == 0,
			"terminos": terminos,
			"etiqueta": ", ".join(terminos) if terminos else f"grupo {lab}",
			"consultas": textos,
		})

	grupos_salida.sort(key=lambda g: g["frecuencia_total"], reverse=True)

	resultado["num_grupos"] = len(grupos_salida)
	resultado["grupos"] = grupos_salida
	resultado["ruido"] = [
		{
			"query": consultas[i]["query"],
			"frecuencia": consultas[i]["frecuencia"],
			"selecciones": consultas[i]["selecciones"],
		}
		for i in ruido
	]
	return resultado
