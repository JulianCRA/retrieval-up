"""Estrategia de fragmentacion semantica: corta donde cae la similitud
coseno entre segmentos vecinos, usando el MISMO modelo de embedding objetivo
(opcionalmente un modelo `boundary` mas liviano).
"""
from __future__ import annotations

import numpy as np

from compartido.embedders import Sizer, cargar_sentence_transformer, get_spec
from compartido.utils import cronometrar, medir

from fragmentador._comun import construir_fragmento, preparar_segmentos


@cronometrar(etiqueta="fragmentacion")
def fragmentar(
	transcripciones: list[dict],
	sizer: Sizer,
	umbral: float = 0.5,
	min_tokens: int = 64,
	boundary_embedder: str | None = None,
	device: str = "cpu",
	model=None,
) -> dict:
	"""Devuelve un dict con:
	  - fragmentos: list[dict]
	  - boundary_hf_id: str
	"""
	vacio = {"fragmentos": [], "boundary_hf_id": ""}
	if not transcripciones:
		return vacio

	segmentos, tokens_seg = preparar_segmentos(transcripciones, sizer)
	if not segmentos:
		return vacio

	boundary_id = boundary_embedder or sizer.spec.id_corto
	boundary_spec = get_spec(boundary_id)

	# 1) Carga del modelo (solo si no se paso uno pre-cargado)
	if model is None:
		with medir(f"carga_modelo_boundary ({boundary_id})"):
			modelo = cargar_sentence_transformer(boundary_id, device=device)
	else:
		modelo = model

	# 2) Inferencia (encode de todos los segmentos)
	textos_emb = [boundary_spec.prefijo_passage + s.get("texto", "") for s in segmentos]
	encode_kwargs: dict = {
		"show_progress_bar": True,
		"convert_to_numpy": True,
		"normalize_embeddings": True,
	}
	if boundary_spec.tarea_passage:
		encode_kwargs["task"] = boundary_spec.tarea_passage
	with medir(f"inferencia_boundary ({len(segmentos)} segmentos)"):
		embeddings = modelo.encode(textos_emb, **encode_kwargs)

	# Con embeddings normalizados el coseno es el dot directo.
	similitudes = [
		float(np.dot(embeddings[i], embeddings[i + 1]))
		for i in range(len(embeddings) - 1)
	]

	chunk_max = sizer.chunk_max
	fragmentos: list[dict] = []
	buf_seg: list[dict] = []
	buf_tok: list[int] = []
	buf_total = 0

	for i, (seg, t) in enumerate(zip(segmentos, tokens_seg)):
		if buf_seg and buf_total + t > chunk_max:
			fragmentos.append(construir_fragmento(buf_seg, sizer))
			buf_seg, buf_tok, buf_total = [], [], 0

		buf_seg.append(seg)
		buf_tok.append(t)
		buf_total += t

		es_ultimo = i == len(segmentos) - 1
		if not es_ultimo and similitudes[i] < umbral:
			fragmentos.append(construir_fragmento(buf_seg, sizer))
			buf_seg, buf_tok, buf_total = [], [], 0

	if buf_seg:
		fragmentos.append(construir_fragmento(buf_seg, sizer))

	if min_tokens > 0 and len(fragmentos) > 1:
		fragmentos = _fusionar_pequenos(fragmentos, sizer, min_tokens)

	return {
		"fragmentos": fragmentos,
		"boundary_hf_id": boundary_spec.hf_id,
	}


def _fusionar_pequenos(
	fragmentos: list[dict], sizer: Sizer, min_tokens: int
) -> list[dict]:
	"""Fusiona hacia atras (o hacia adelante si no entra) cualquier fragmento
	con `num_tokens` < min_tokens, manteniendo `num_tokens <= chunk_max`.
	"""
	chunk_max = sizer.chunk_max
	resultado: list[list[dict]] = [list(f["segmentos"]) for f in fragmentos]
	tokens = [f["num_tokens"] for f in fragmentos]

	i = 0
	while i < len(resultado):
		if tokens[i] >= min_tokens or len(resultado) == 1:
			i += 1
			continue
		if i > 0 and tokens[i - 1] + tokens[i] <= chunk_max:
			resultado[i - 1].extend(resultado[i])
			tokens[i - 1] += tokens[i]
			resultado.pop(i)
			tokens.pop(i)
			continue
		if i + 1 < len(resultado) and tokens[i] + tokens[i + 1] <= chunk_max:
			resultado[i].extend(resultado[i + 1])
			tokens[i] += tokens[i + 1]
			resultado.pop(i + 1)
			tokens.pop(i + 1)
			continue
		i += 1

	return [construir_fragmento(segs, sizer) for segs in resultado]
