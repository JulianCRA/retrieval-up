"""Backend Qdrant: implementa la interfaz de indexacion para un servidor Qdrant.

Interfaz implementada:
  abrir(db_ruta: str) -> Any
  hash_indexado(db, nombre: str, hash_id: str) -> bool
  escribir_tabla(db, nombre: str, filas: list[dict], dim: int, reclear: bool) -> None

db_ruta se interpreta como la URL del servidor, ej. 'http://localhost:6333'.

Dependencias necesarias (no incluidas en pyproject.toml todavia):
  qdrant-client>=1.9
"""


def abrir(db_ruta: str):
	"""Conecta al servidor Qdrant en db_ruta y devuelve el cliente."""
	raise NotImplementedError


def hash_indexado(db, nombre: str, hash_id: str) -> bool:
	"""Devuelve True si ya existen puntos con payload hash == hash_id en la coleccion."""
	raise NotImplementedError


def escribir_tabla(
	db,
	nombre: str,
	filas: list[dict],
	dim: int,
	reclear: bool,
) -> None:
	"""Crea o actualiza la coleccion `nombre` con las filas dadas."""
	raise NotImplementedError
