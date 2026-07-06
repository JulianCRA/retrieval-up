"""Salida 2: videos mas seleccionados por los usuarios.

Usa el resultado elegido tras la busqueda para estimar que recursos no solo
aparecen en el ranking, sino que ademas son utilizados como punto de acceso al
contenido audiovisual. La comparacion con los videos mas encontrados permite
distinguir material recuperado con frecuencia de material usado efectivamente.
"""
from __future__ import annotations

from insights.corpus import titulos_por_hash
from insights.encontrados import videos_mas_encontrados
from insights.registro import Actividad


def videos_mas_seleccionados(actividad: Actividad, top_n: int = 20) -> list[dict]:
	"""Frecuencia con la que cada recurso fue el resultado seleccionado."""
	titulos = titulos_por_hash()

	conteo: dict[str, int] = {}
	for s in actividad.selecciones:
		video_id = s.get("video_id")
		if not video_id:
			continue
		conteo[video_id] = conteo.get(video_id, 0) + 1

	filas = [
		{"video_id": vid, "titulo": titulos.get(vid, ""), "selecciones": n}
		for vid, n in conteo.items()
	]
	filas.sort(key=lambda f: f["selecciones"], reverse=True)
	if top_n and top_n > 0:
		filas = filas[:top_n]
	for i, f in enumerate(filas, 1):
		f["rank"] = i
	return filas


def comparar_encontrados_seleccionados(
	actividad: Actividad,
	top_n: int = 20,
) -> list[dict]:
	"""Tabla comparada: apariciones en resultados vs. selecciones por recurso.

	Una tasa de seleccion baja con muchas apariciones sugiere material recuperado
	pero poco usado; lo contrario sugiere un buen punto de acceso a la coleccion.
	"""
	titulos = titulos_por_hash()

	encontrados = {f["video_id"]: f["apariciones"] for f in videos_mas_encontrados(actividad, top_n=0)}
	seleccionados = {f["video_id"]: f["selecciones"] for f in videos_mas_seleccionados(actividad, top_n=0)}

	video_ids = set(encontrados) | set(seleccionados)
	filas = []
	for vid in video_ids:
		apariciones = encontrados.get(vid, 0)
		selecciones = seleccionados.get(vid, 0)
		tasa = round(selecciones / apariciones, 4) if apariciones else None
		filas.append({
			"video_id": vid,
			"titulo": titulos.get(vid, ""),
			"apariciones": apariciones,
			"selecciones": selecciones,
			"tasa_seleccion": tasa,
		})

	filas.sort(key=lambda f: (f["apariciones"], f["selecciones"]), reverse=True)
	if top_n and top_n > 0:
		filas = filas[:top_n]
	return filas
