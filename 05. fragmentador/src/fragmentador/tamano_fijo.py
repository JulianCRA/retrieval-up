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


@cronometrar(etiqueta="Fragmentacion tamano fijo")
def fragmentar(transcripciones: list[dict], max_tokens: int = 512) -> list[dict]:
	fragmentos = []
	segmentos_acumulados = []
	tokens_acumulados = 0

	for seg in transcripciones:
		texto_seg = seg.get("texto", "")
		tokens_seg = _estimar_tokens(texto_seg)

		if segmentos_acumulados and tokens_acumulados + tokens_seg > max_tokens:
			fragmentos.append(_construir_fragmento(segmentos_acumulados))
			segmentos_acumulados = []
			tokens_acumulados = 0

		segmentos_acumulados.append(seg)
		tokens_acumulados += tokens_seg

	if segmentos_acumulados:
		fragmentos.append(_construir_fragmento(segmentos_acumulados))

	return fragmentos
