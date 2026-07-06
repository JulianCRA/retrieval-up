"""Lectura del registro de actividad de busqueda (`resultados.db`).

El registro minimo vincula cada busqueda con los resultados mostrados y, cuando
existe interaccion posterior, con el resultado seleccionado por el usuario. Este
modulo centraliza el acceso de solo lectura a esos datos y deja los vectores de
query decodificados como numpy para el agrupador.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compartido.sqlite_utils import leer_busquedas, leer_resultados, leer_selecciones


@dataclass
class Actividad:
	"""Vista en memoria del registro de actividad."""

	busquedas: list[dict] = field(default_factory=list)
	resultados: list[dict] = field(default_factory=list)
	selecciones: list[dict] = field(default_factory=list)

	# Indices auxiliares
	resultados_por_busqueda: dict[int, list[dict]] = field(default_factory=dict)
	busqueda_por_id: dict[int, dict] = field(default_factory=dict)

	def __post_init__(self) -> None:
		for b in self.busquedas:
			self.busqueda_por_id[b["id"]] = b
		for r in self.resultados:
			self.resultados_por_busqueda.setdefault(r["busqueda_id"], []).append(r)

	@property
	def num_busquedas(self) -> int:
		return len(self.busquedas)

	@property
	def num_selecciones(self) -> int:
		return len(self.selecciones)

	def filtrar_embedder(self, embedder: str | None) -> "Actividad":
		"""Devuelve una copia restringida a un embedder (o tal cual si es None)."""
		if not embedder:
			return self
		busquedas = [b for b in self.busquedas if b.get("embedder") == embedder]
		ids = {b["id"] for b in busquedas}
		resultados = [r for r in self.resultados if r["busqueda_id"] in ids]
		selecciones = [s for s in self.selecciones if s["busqueda_id"] in ids]
		return Actividad(busquedas=busquedas, resultados=resultados, selecciones=selecciones)


def cargar_actividad() -> Actividad:
	"""Carga el registro completo de actividad desde la base SQLite compartida."""
	return Actividad(
		busquedas=leer_busquedas(),
		resultados=leer_resultados(),
		selecciones=leer_selecciones(),
	)


def decodificar_vector(blob: bytes | None) -> np.ndarray | None:
	"""Decodifica un *query_vector* almacenado como bytes float32."""
	if not blob:
		return None
	try:
		return np.frombuffer(blob, dtype=np.float32).copy()
	except (TypeError, ValueError):
		return None
