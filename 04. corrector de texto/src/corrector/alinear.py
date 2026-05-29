import re


def _normalizar(token: str) -> str:
	return re.sub(r"[^\w]", "", token, flags=re.UNICODE).lower()


def alinear_segmentos(segmentos: list[dict], texto_corregido: str) -> list[dict]:
	tokens_corr = texto_corregido.split()
	pos = 0
	resultado: list[dict] = []

	for seg in segmentos:
		words = seg.get("texto", "").split()
		nuevo_seg = dict(seg)

		if not words:
			nuevo_seg["texto_corregido"] = ""
			resultado.append(nuevo_seg)
			continue

		slice_tokens: list[str] = []
		ok = True
		for word in words:
			if pos >= len(tokens_corr):
				print(f"[WARN] alinear: tokens agotados en segmento {seg.get('inicio'):.2f}s")
				ok = False
				break
			if _normalizar(word) != _normalizar(tokens_corr[pos]):
				print(f"[WARN] alinear: desajuste en {seg.get('inicio'):.2f}s: {word!r} vs {tokens_corr[pos]!r}")
				ok = False
				break
			slice_tokens.append(tokens_corr[pos])
			pos += 1

		nuevo_seg["texto_corregido"] = " ".join(slice_tokens) if ok else seg.get("texto", "")
		resultado.append(nuevo_seg)

	return resultado
