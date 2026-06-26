"""Vista agregada del corpus indexado (`indice.db`).

Resume el material disponible en el indice: numero de recursos, duracion total,
distribucion de etiquetas y conteo de chunks por modelo de embedding. Es una
foto del corpus, independiente del comportamiento de busqueda.
"""
from __future__ import annotations

import json
import re
import sqlite3

from compartido.rutas import INDICE_DB


def _conectar() -> sqlite3.Connection:
	conn = sqlite3.connect(INDICE_DB)
	conn.row_factory = sqlite3.Row
	return conn


def _tablas(conn: sqlite3.Connection) -> set[str]:
	filas = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
	return {f[0] for f in filas}


def resumen_corpus() -> dict:
	"""Devuelve un resumen agregado del corpus indexado.

	Si el indice no existe todavia, devuelve una estructura vacia coherente.
	"""
	if not INDICE_DB.exists():
		return {
			"num_recursos": 0,
			"duracion_total_seg": 0.0,
			"duracion_total_horas": 0.0,
			"num_fuentes": 0,
			"fuentes": {},
			"tags": {},
			"chunks_por_modelo": {},
			"recursos_sin_tags": 0,
		}

	conn = _conectar()
	try:
		tablas = _tablas(conn)
		if "recursos" not in tablas:
			return {
				"num_recursos": 0,
				"duracion_total_seg": 0.0,
				"duracion_total_horas": 0.0,
				"num_fuentes": 0,
				"fuentes": {},
				"tags": {},
				"chunks_por_modelo": {},
				"recursos_sin_tags": 0,
			}

		recursos = conn.execute(
			"SELECT hash, fuente, duracion, tags FROM recursos"
		).fetchall()

		num_recursos = len(recursos)
		duracion_total = 0.0
		fuentes: dict[str, int] = {}
		tags: dict[str, int] = {}
		sin_tags = 0

		for r in recursos:
			if r["duracion"]:
				duracion_total += float(r["duracion"])
			fuente = r["fuente"] or "(desconocida)"
			fuentes[fuente] = fuentes.get(fuente, 0) + 1
			etiquetas = _parse_tags(r["tags"])
			if not etiquetas:
				sin_tags += 1
			for t in etiquetas:
				tags[t] = tags.get(t, 0) + 1

		chunks_por_modelo: dict[str, int] = {}
		for tabla in sorted(tablas):
			if tabla.startswith("chunks_"):
				modelo = tabla[len("chunks_"):]
				n = conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
				chunks_por_modelo[modelo] = int(n)

		return {
			"num_recursos": num_recursos,
			"duracion_total_seg": round(duracion_total, 2),
			"duracion_total_horas": round(duracion_total / 3600.0, 2),
			"num_fuentes": len(fuentes),
			"fuentes": dict(sorted(fuentes.items(), key=lambda kv: kv[1], reverse=True)),
			"tags": dict(sorted(tags.items(), key=lambda kv: kv[1], reverse=True)),
			"chunks_por_modelo": chunks_por_modelo,
			"recursos_sin_tags": sin_tags,
		}
	finally:
		conn.close()


def titulos_por_hash() -> dict[str, str]:
	"""Mapa hash -> titulo del recurso, para enriquecer las salidas."""
	if not INDICE_DB.exists():
		return {}
	conn = _conectar()
	try:
		if "recursos" not in _tablas(conn):
			return {}
		filas = conn.execute("SELECT hash, titulo FROM recursos").fetchall()
		return {f["hash"]: (f["titulo"] or "") for f in filas}
	finally:
		conn.close()


def _parse_tags(raw: str | None) -> list[str]:
	if not raw:
		return []
	try:
		val = json.loads(raw)
	except (TypeError, ValueError):
		return [t for t in re.split(r"[,\s]+", raw) if t]
	if isinstance(val, list):
		return [str(t) for t in val if str(t).strip()]
	if isinstance(val, dict):
		return [f"{k}={v}" for k, v in val.items()]
	return []
