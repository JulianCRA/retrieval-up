"""Estrategia de fragmentacion por tamano fijo con tokens reales del modelo."""
from __future__ import annotations

from compartido.embedders import Sizer
from compartido.utils import cronometrar

from fragmentador._comun import construir_fragmento, preparar_segmentos


def _segmentos_overlap(
	segmentos: list[dict], tokens_segs: list[int], overlap_tokens: int
) -> tuple[list[dict], list[int]]:
	"""Toma desde el final los segmentos cuyo total <= overlap_tokens.
	Nunca incluye un segmento que por si solo supere overlap_tokens.
	"""
	if overlap_tokens <= 0:
		return [], []
	cola_seg: list[dict] = []
	cola_tok: list[int] = []
	acum = 0
	for seg, t in zip(reversed(segmentos), reversed(tokens_segs)):
		if acum + t > overlap_tokens:
			break
		cola_seg.insert(0, seg)
		cola_tok.insert(0, t)
		acum += t
	return cola_seg, cola_tok


@cronometrar(etiqueta="fragmentacion")
def fragmentar(
	transcripciones: list[dict],
	sizer: Sizer,
	overlap_pct: int = 20,
) -> tuple[list[dict], int]:
	"""Retorna (fragmentos, overlap_tokens) usando el tokenizador real del modelo.

	`overlap_pct` se aplica sobre `sizer.chunk_max` (0-50).
	"""
	overlap_pct = max(0, min(50, overlap_pct))
	chunk_max = sizer.chunk_max
	overlap_tokens = int(chunk_max * overlap_pct / 100)

	segmentos, tokens_seg = preparar_segmentos(transcripciones, sizer)

	fragmentos: list[dict] = []
	buf_seg: list[dict] = []
	buf_tok: list[int] = []
	buf_total = 0

	for seg, t in zip(segmentos, tokens_seg):
		# Si anadir este segmento excederia el limite, cerrar el fragmento.
		if buf_seg and buf_total + t > chunk_max:
			fragmentos.append(construir_fragmento(buf_seg, sizer))
			buf_seg, buf_tok = _segmentos_overlap(buf_seg, buf_tok, overlap_tokens)
			buf_total = sum(buf_tok)
			# Si tras el overlap aun no entra, descartar overlap.
			if buf_total + t > chunk_max:
				buf_seg, buf_tok, buf_total = [], [], 0
		buf_seg.append(seg)
		buf_tok.append(t)
		buf_total += t

	if buf_seg:
		fragmentos.append(construir_fragmento(buf_seg, sizer))

	return fragmentos, overlap_tokens
