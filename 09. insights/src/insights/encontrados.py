"""Salida 1: videos mas encontrados por el sistema.

Contabiliza la frecuencia con la que cada recurso aparece entre los primeros
resultados de busqueda. Aproxima que materiales son recuperados de forma
recurrente ante las consultas de los usuarios.
"""
from __future__ import annotations

from insights.corpus import titulos_por_hash
from insights.registro import Actividad


def videos_mas_encontrados(
	actividad: Actividad,
	top_n: int = 20,
	rank_max: int | None = None,
) -> list[dict]:
	"""Frecuencia de aparicion de cada recurso en los resultados mostrados.

	Args:
		actividad: registro de actividad cargado.
		top_n: numero de recursos a devolver (0 = todos).
		rank_max: si se indica, solo cuenta apariciones con rank <= rank_max
			(por ejemplo, "primeros 3 resultados").
	"""
	titulos = titulos_por_hash()

	conteo: dict[str, int] = {}
	busquedas_distintas: dict[str, set[int]] = {}
	for r in actividad.resultados:
		if rank_max is not None and (r.get("rank") or 0) > rank_max:
			continue
		video_id = r.get("video_id")
		if not video_id:
			continue
		conteo[video_id] = conteo.get(video_id, 0) + 1
		busquedas_distintas.setdefault(video_id, set()).add(r["busqueda_id"])

	filas = [
		{
			"video_id": vid,
			"titulo": titulos.get(vid, ""),
			"apariciones": n,
			"busquedas_distintas": len(busquedas_distintas.get(vid, ())),
		}
		for vid, n in conteo.items()
	]
	filas.sort(key=lambda f: (f["apariciones"], f["busquedas_distintas"]), reverse=True)
	if top_n and top_n > 0:
		filas = filas[:top_n]
	for i, f in enumerate(filas, 1):
		f["rank"] = i
	return filas
