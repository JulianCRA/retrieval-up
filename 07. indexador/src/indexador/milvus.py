"""Backend Milvus: implementa la interfaz de indexacion para un servidor Milvus.

Interfaz implementada:
  abrir(db_ruta: str) -> Any
  hash_indexado(db, nombre: str, hash_id: str) -> bool
  escribir_tabla(db, nombre: str, filas: list[dict], dim: int, reclear: bool) -> None

db_ruta se interpreta como 'host:port', ej. 'localhost:19530'.

Dependencias necesarias (no incluidas en pyproject.toml todavia):
  pymilvus>=2.4
"""


def abrir(db_ruta: str):
	"""Conecta a Milvus en db_ruta y devuelve el cliente/conexion."""
	raise NotImplementedError


def hash_indexado(db, nombre: str, hash_id: str) -> bool:
	"""Devuelve True si ya existen entidades con campo hash == hash_id en la coleccion."""
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
