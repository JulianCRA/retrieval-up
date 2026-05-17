from compartido.utils import cronometrar


# Estimacion conservadora para espanol (~3.5 caracteres por token)
_CHARS_POR_TOKEN = 3.5


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
	}


def _segmentos_overlap(segmentos: list[dict], overlap_tokens: int) -> list[dict]:
	"""Retorna los ultimos segmentos del chunk cuya suma de tokens no supere overlap_tokens."""
	cola: list[dict] = []
	tokens = 0
	for seg in reversed(segmentos):
		t = _estimar_tokens(seg.get("texto", ""))
		if tokens + t > overlap_tokens:
			break
		cola.insert(0, seg)
		tokens += t
	return cola


@cronometrar(etiqueta="Fragmentacion tamano fijo")
def fragmentar(
	transcripciones: list[dict],
	max_tokens: int = 512,
	overlap_pct: int = 10,
) -> tuple[list[dict], int]:
	"""
	Retorna (fragmentos, overlap_tokens_aprox).
	overlap_pct: porcentaje de max_tokens que se solapa entre chunks consecutivos (0-50).
	"""
	overlap_tokens = int(max_tokens * overlap_pct / 100)

	fragmentos = []
	segmentos_acumulados: list[dict] = []
	tokens_acumulados = 0

	for seg in transcripciones:
		texto_seg = seg.get("texto", "")
		tokens_seg = _estimar_tokens(texto_seg)

		if segmentos_acumulados and tokens_acumulados + tokens_seg > max_tokens:
			fragmentos.append(_construir_fragmento(segmentos_acumulados))
			segmentos_acumulados = _segmentos_overlap(segmentos_acumulados, overlap_tokens)
			tokens_acumulados = sum(_estimar_tokens(s.get("texto", "")) for s in segmentos_acumulados)

		segmentos_acumulados.append(seg)
		tokens_acumulados += tokens_seg

	if segmentos_acumulados:
		fragmentos.append(_construir_fragmento(segmentos_acumulados))

	return fragmentos, overlap_tokens
