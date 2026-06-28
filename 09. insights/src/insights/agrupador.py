"""Salida 3: agrupacion de consultas semanticamente similares.

Las consultas registradas se codifican con qwen3-0.6b, se reducen con UMAP
(metrica cosine, min_dist=0.0) y se agrupan con HDBSCAN (leaf + soft
assignment). Las etiquetas de cada grupo se extraen con KeyBERT.

El objetivo es que el docente reconozca temas recurrentes, dudas formuladas
con vocabularios distintos y posibles vacios del corpus (un grupo que acumula
busquedas sin resultados seleccionados).

Configuracion derivada del benchmark cluster_qwen.py:
  qwen3-0.6b  +  UMAP (cosine, min_dist=0.0)  +  HDBSCAN leaf + soft assignment
  ARI=0.90  NMI=0.92  purity=0.93 en el dataset de evaluacion (305 queries, 10 grupos).
"""
from __future__ import annotations

import warnings
import numpy as np

from compartido.embedders import cargar_sentence_transformer

from insights.registro import Actividad

_EMBEDDER_ID   = "qwen3-0.6b"
_TOP_TERMINOS  = 3       # terminos KeyBERT por grupo (MMR asegura diversidad)
_UMAP_MIN_DIST = 0.0     # 0.0 = clusters compactos

# Cache de embeddings qwen por texto de consulta (vive mientras el proceso este activo).
# mcs/min_samples no afectan los vectores: se puede reutilizar entre llamadas.
_embed_cache: dict[str, list[float]] = {}


def _normalizar(matriz: np.ndarray) -> np.ndarray:
	normas = np.linalg.norm(matriz, axis=1, keepdims=True)
	normas[normas == 0] = 1.0
	return matriz / normas


def _compute_umap_params(n: int) -> tuple[int, int]:
	"""Devuelve (n_neighbors, n_components) — misma logica que cluster_qwen.py.

	n_neighbors: ~10% de N, min 15, max 100 (capado a n-1 para que UMAP no falle).
	n_components: mitad de n_neighbors, min 5 (capado a n-1).
	"""
	nn = min(n - 1, max(15, min(n // 10, 100)))
	nc = min(n - 1, max(5, nn // 2))
	return nn, nc


def _soft_assign(cl, pred: np.ndarray) -> np.ndarray:
	"""Reasigna puntos de ruido al cluster de mayor probabilidad de pertenencia."""
	import hdbscan as _hdbscan
	if (pred == -1).sum() == 0:
		return pred
	soft = _hdbscan.all_points_membership_vectors(cl)
	soft = np.atleast_2d(soft)
	if soft.shape[0] == 1:
		soft = soft.T
	pred_soft = pred.copy()
	for i, lab in enumerate(pred):
		if lab == -1:
			best = int(np.argmax(soft[i]))
			if soft[i][best] > 0.0:
				pred_soft[i] = best
	return pred_soft


def _consultas_unicas(actividad: Actividad) -> list[dict]:
	"""Consultas distintas (por texto) con frecuencia y selecciones acumuladas."""
	por_texto: dict[str, dict] = {}
	sel_por_busqueda: dict[int, int] = {}
	for s in actividad.selecciones:
		sel_por_busqueda[s["busqueda_id"]] = sel_por_busqueda.get(s["busqueda_id"], 0) + 1

	for b in actividad.busquedas:
		query = (b.get("query") or "").strip()
		if not query:
			continue
		entrada = por_texto.get(query)
		if entrada is None:
			entrada = {"query": query, "frecuencia": 0, "selecciones": 0}
			por_texto[query] = entrada
		entrada["frecuencia"] += 1
		entrada["selecciones"] += sel_por_busqueda.get(b["id"], 0)

	return list(por_texto.values())


def _embed_consultas(consultas: list[dict], device: str) -> np.ndarray:
	"""Codifica consultas con qwen3-0.6b, reutilizando vectores ya calculados."""
	faltan = [c["query"] for c in consultas if c["query"] not in _embed_cache]
	if faltan:
		modelo = cargar_sentence_transformer(_EMBEDDER_ID, device=device)
		vecs = modelo.encode(faltan, normalize_embeddings=True)
		for texto, vec in zip(faltan, vecs):
			_embed_cache[texto] = vec.tolist()
	matriz = np.array([_embed_cache[c["query"]] for c in consultas], dtype=np.float32)
	return _normalizar(matriz)


def _etiquetas_keybert(textos: list[str], device: str) -> list[str]:
	"""Extrae terminos representativos de un grupo de consultas con KeyBERT."""
	try:
		from keybert import KeyBERT
	except ImportError:
		return []
	if not textos:
		return []
	modelo_st = cargar_sentence_transformer(_EMBEDDER_ID, device=device)
	kw = KeyBERT(model=modelo_st)
	documento = ". ".join(textos)
	pares = kw.extract_keywords(
		documento,
		keyphrase_ngram_range=(1, 3),
		stop_words=None,
		top_n=_TOP_TERMINOS,
	)
	return [termino for termino, _score in pares]


def agrupar_consultas(
	actividad: Actividad,
	device: str = "cpu",
	min_cluster_size: int = 5,
	min_samples: int = 2,
) -> dict:
	"""Agrupa consultas con UMAP + HDBSCAN (leaf + soft) y etiqueta con KeyBERT.

	Devuelve un dict con la lista de grupos (cada uno con sus consultas, tamano,
	terminos representativos y selecciones acumuladas) y las consultas marcadas
	como ruido.
	"""
	consultas = _consultas_unicas(actividad)
	resultado: dict = {
		"embedder": _EMBEDDER_ID,
		"num_consultas": len(consultas),
		"num_grupos": 0,
		"grupos": [],
		"ruido": [],
	}

	n = len(consultas)
	if n < max(min_cluster_size, 2):
		resultado["ruido"] = [
			{"query": c["query"], "frecuencia": c["frecuencia"], "selecciones": c["selecciones"]}
			for c in consultas
		]
		return resultado

	# ── Embeddings ────────────────────────────────────────────────────────────
	matriz = _embed_consultas(consultas, device)

	# ── UMAP ─────────────────────────────────────────────────────────────────
	try:
		import umap as umap_lib
	except ImportError as exc:
		raise RuntimeError(
			"umap-learn no esta instalado. Instala el modulo '09. insights'."
		) from exc

	nn, nc = _compute_umap_params(n)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
		reducer = umap_lib.UMAP(
			n_components=nc,
			metric="cosine",
			n_neighbors=nn,
			min_dist=_UMAP_MIN_DIST,
			random_state=42,
			n_jobs=1,
		)
		matriz_low = reducer.fit_transform(matriz.astype(np.float64))

	# ── HDBSCAN + soft assignment ─────────────────────────────────────────────
	try:
		import hdbscan
	except ImportError as exc:
		raise RuntimeError(
			"hdbscan no esta instalado. Instala el modulo '09. insights'."
		) from exc

	mcs_efectivo = max(2, min_cluster_size)
	ms_efectivo  = max(1, min_samples)

	clusterer = hdbscan.HDBSCAN(
		min_cluster_size=mcs_efectivo,
		min_samples=ms_efectivo,
		metric="euclidean",
		cluster_selection_method="leaf",
		prediction_data=True,
	)
	pred_hard = clusterer.fit_predict(matriz_low.astype(np.float64))
	pred = _soft_assign(clusterer, pred_hard)

	# ── Construir grupos ──────────────────────────────────────────────────────
	grupos: dict[int, list[int]] = {}
	ruido: list[int] = []
	for idx, lab in enumerate(pred):
		if lab == -1:
			ruido.append(idx)
		else:
			grupos.setdefault(int(lab), []).append(idx)

	grupos_salida = []
	for lab, indices in grupos.items():
		textos = [consultas[i]["query"] for i in indices]
		frecuencia  = sum(consultas[i]["frecuencia"]  for i in indices)
		selecciones = sum(consultas[i]["selecciones"] for i in indices)
		# Consulta mas cercana al centroide del cluster como etiqueta principal.
		vecs_cluster = np.array([_embed_cache[consultas[i]["query"]] for i in indices], dtype=np.float32)
		centroide = vecs_cluster.mean(axis=0)
		dists = np.linalg.norm(vecs_cluster - centroide, axis=1)
		etiqueta = textos[int(np.argmin(dists))]
		terminos = _etiquetas_keybert(textos, device)
		grupos_salida.append({
			"grupo": lab,
			"tamano": len(indices),
			"frecuencia_total": frecuencia,
			"selecciones_total": selecciones,
			"sin_seleccion": selecciones == 0,
			"terminos": terminos,
			"etiqueta": etiqueta,
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
