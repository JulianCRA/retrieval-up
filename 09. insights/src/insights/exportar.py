"""Exportacion de tablas de inspeccion (CSV + JSON).

Las salidas se calculan offline / bajo demanda y se escriben en
`resultados/insights/`. Se generan en CSV (para hojas de calculo) y JSON (para
inspeccion programatica), sin afectar las consultas interactivas.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from compartido.rutas import RESULTADOS_DIR

INSIGHTS_DIR = RESULTADOS_DIR / "insights"


def _asegurar_dir() -> Path:
	INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
	return INSIGHTS_DIR


def _stamp() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S")


def exportar_tabla(nombre: str, filas: list[dict], campos: list[str] | None = None) -> dict:
	"""Exporta una lista de filas como CSV y JSON. Devuelve las rutas escritas."""
	carpeta = _asegurar_dir()
	stamp = _stamp()
	base = f"{stamp}_{nombre}"

	ruta_json = carpeta / f"{base}.json"
	ruta_json.write_text(
		json.dumps(filas, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	ruta_csv = carpeta / f"{base}.csv"
	if filas:
		if campos is None:
			campos = list(filas[0].keys())
		with ruta_csv.open("w", encoding="utf-8", newline="") as fh:
			writer = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
			writer.writeheader()
			for fila in filas:
				writer.writerow({k: _escalar(fila.get(k)) for k in campos})
	else:
		ruta_csv.write_text("", encoding="utf-8")

	print(f"[OK] '{nombre}': {ruta_csv.name}, {ruta_json.name}")
	return {"csv": str(ruta_csv), "json": str(ruta_json)}


def exportar_json(nombre: str, datos) -> str:
	"""Exporta una estructura arbitraria solo como JSON (p. ej. el resumen del corpus)."""
	carpeta = _asegurar_dir()
	ruta = carpeta / f"{_stamp()}_{nombre}.json"
	ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[OK] '{nombre}': {ruta.name}")
	return str(ruta)


def _escalar(valor):
	"""Aplana valores compuestos para que entren en una celda CSV."""
	if isinstance(valor, (list, tuple)):
		return "; ".join(str(v) for v in valor)
	if isinstance(valor, dict):
		return json.dumps(valor, ensure_ascii=False)
	return valor
