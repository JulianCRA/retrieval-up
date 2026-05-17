import numpy as np
from sentence_transformers import SentenceTransformer

from compartido.utils import cronometrar

_CHARS_POR_TOKEN = 3.5
MODELO_ID = "paraphrase-multilingual-MiniLM-L12-v2"


def _estimar_tokens(texto: str) -> int:
	return int(len(texto) / _CHARS_POR_TOKEN)


def _construir_fragmento(segmentos: list[dict]) -> dict:
	texto = " ".join(seg.get("texto", "") for seg in segmentos).strip()
	return {
		"inicio": segmentos[0]["inicio"],
		"fin": segmentos[-1]["fin"],
		"duracion": round(segmentos[-1]["fin"] - segmentos[0]["inicio"], 3),
		"num_caracteres": len(texto),
		"num_palabras": len(texto.split()),
		"texto": texto,
		"segmentos": [
			{"inicio": s["inicio"], "fin": s["fin"], "texto": s.get("texto", "")}
			for s in segmentos
		],
	}


def _similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
	dot = float(np.dot(a, b))
	norm = float(np.linalg.norm(a) * np.linalg.norm(b))
	return dot / norm if norm > 0 else 0.0


@cronometrar(etiqueta="Fragmentacion semantica")
def fragmentar(
	transcripciones: list[dict],
	umbral: float = 0.5,
	min_tokens: int = 64,
	max_tokens: int = 512,
) -> list[dict]:
	"""
	Fragmenta segmentos usando similitud semantica entre segmentos consecutivos.

	Corta donde la similitud coseno cae por debajo de `umbral` (cambio tematico)
	o cuando el chunk acumula mas de `max_tokens`. Fusiona fragmentos finales
	menores a `min_tokens` con el anterior para evitar chunks muy pequenos.
	"""
	if not transcripciones:
		return []

	modelo = SentenceTransformer(MODELO_ID)
	textos = [seg.get("texto", "") for seg in transcripciones]
	embeddings = modelo.encode(textos, show_progress_bar=False, convert_to_numpy=True)

	# Similitud coseno entre cada par de segmentos consecutivos
	similitudes = [
		_similitud_coseno(embeddings[i], embeddings[i + 1])
		for i in range(len(embeddings) - 1)
	]

	fragmentos: list[dict] = []
	grupo: list[dict] = []
	tokens_acumulados = 0

	for i, seg in enumerate(transcripciones):
		grupo.append(seg)
		tokens_acumulados += _estimar_tokens(seg.get("texto", ""))

		es_ultimo = i == len(transcripciones) - 1
		corte_semantico = not es_ultimo and similitudes[i] < umbral
		corte_max = not es_ultimo and tokens_acumulados >= max_tokens

		if corte_semantico or corte_max:
			fragmentos.append(_construir_fragmento(grupo))
			grupo = []
			tokens_acumulados = 0

	if grupo:
		fragmentos.append(_construir_fragmento(grupo))

	# Fusionar fragmentos finales demasiado pequeños con el anterior
	if min_tokens > 0 and len(fragmentos) > 1:
		fusionados: list[list[dict]] = []
		for frag in fragmentos:
			if fusionados and _estimar_tokens(frag["texto"]) < min_tokens:
				fusionados[-1] = fusionados[-1] + frag["segmentos"]
			else:
				fusionados.append(frag["segmentos"])
		fragmentos = [_construir_fragmento(segs) for segs in fusionados]

	return fragmentos
