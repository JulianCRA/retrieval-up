"""Utilidades comunes a las estrategias de fragmentacion."""
from __future__ import annotations

import re

from compartido.embedders import Sizer


_SEP_ORACIONES = re.compile(r"(?<=[.!?])\s+")


def construir_fragmento(segmentos: list[dict], sizer: Sizer) -> dict:
	"""Arma el dict final de un fragmento a partir de sus segmentos.
	Calcula tokens reales con el `Sizer` del modelo objetivo.
	"""
	texto = " ".join(seg.get("texto", "") for seg in segmentos).strip()
	return {
		"inicio": segmentos[0]["inicio"],
		"fin": segmentos[-1]["fin"],
		"duracion": round(segmentos[-1]["fin"] - segmentos[0]["inicio"], 3),
		"num_tokens": sizer.count(texto),
		"num_caracteres": len(texto),
		"num_palabras": len(texto.split()),
		"texto": texto,
		"segmentos": [
			{"inicio": s["inicio"], "fin": s["fin"], "texto": s.get("texto", "")}
			for s in segmentos
		],
	}


def _interpolar_tiempos(seg: dict, parte_idx: int, total_partes: int, total_chars: int, chars_acum: int, chars_pieza: int) -> tuple[float, float]:
	"""Interpola inicio/fin proporcionalmente a la posicion de caracter."""
	dur = seg["fin"] - seg["inicio"]
	if total_chars <= 0:
		# fallback uniforme
		paso = dur / total_partes
		return (
			seg["inicio"] + paso * parte_idx,
			seg["inicio"] + paso * (parte_idx + 1),
		)
	t0 = seg["inicio"] + dur * (chars_acum / total_chars)
	t1 = seg["inicio"] + dur * ((chars_acum + chars_pieza) / total_chars)
	return t0, t1


def trocear_segmento_largo(seg: dict, sizer: Sizer) -> list[dict]:
	"""Si un segmento ASR mide mas que chunk_max, lo divide en sub-segmentos
	respetando el limite del modelo. Estrategia:
	  1. Cortar por oraciones (`.!?`).
	  2. Si una oracion sigue siendo demasiado larga, trocear por tokens.
	Los tiempos `inicio`/`fin` se interpolan por proporcion de caracteres.
	"""
	texto = seg.get("texto", "").strip()
	if not texto or sizer.count(texto) <= sizer.chunk_max:
		return [seg]

	# 1. Por oraciones
	oraciones = [o for o in _SEP_ORACIONES.split(texto) if o.strip()]
	if not oraciones:
		oraciones = [texto]

	# 2. Reagrupar oraciones en piezas <= chunk_max. Si una oracion no entra,
	#    trocear a nivel de tokens.
	piezas_texto: list[str] = []
	buffer = ""
	for oracion in oraciones:
		candidato = (buffer + " " + oracion).strip() if buffer else oracion
		if sizer.count(candidato) <= sizer.chunk_max:
			buffer = candidato
			continue
		if buffer:
			piezas_texto.append(buffer)
			buffer = ""
		# La oracion sola supera el limite -> trocear por tokens
		if sizer.count(oracion) > sizer.chunk_max:
			piezas_texto.extend(sizer.trocear_texto(oracion))
		else:
			buffer = oracion
	if buffer:
		piezas_texto.append(buffer)

	# Reconstruir sub-segmentos con tiempos interpolados por caracteres
	total_chars = sum(len(p) for p in piezas_texto)
	sub_segmentos: list[dict] = []
	chars_acum = 0
	for i, pieza in enumerate(piezas_texto):
		t0, t1 = _interpolar_tiempos(
			seg, i, len(piezas_texto), total_chars, chars_acum, len(pieza)
		)
		sub_segmentos.append({
			"inicio": round(t0, 3),
			"fin": round(t1, 3),
			"texto": pieza,
		})
		chars_acum += len(pieza)
	return sub_segmentos


def preparar_segmentos(transcripciones: list[dict], sizer: Sizer) -> tuple[list[dict], list[int]]:
	"""Devuelve segmentos (con oversize ya troceados) y la lista paralela de
	conteo de tokens reales para cada uno.
	"""
	segmentos: list[dict] = []
	for seg in transcripciones:
		segmentos.extend(trocear_segmento_largo(seg, sizer))
	tokens_por_seg = [sizer.count(s.get("texto", "")) for s in segmentos]
	return segmentos, tokens_por_seg
