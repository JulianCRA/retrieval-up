"""Backend LanceDB: implementa la interfaz de indexacion para una base local embebida.

Interfaz que todo backend debe exponer:
  abrir(db_ruta: str) -> Any
  hash_indexado(db, nombre: str, hash_id: str) -> bool
  escribir_tabla(db, nombre: str, filas: list[dict], dim: int, reclear: bool) -> None
"""
import sys

import lancedb
import pyarrow as pa


def abrir(db_ruta: str) -> lancedb.DBConnection:
	db = lancedb.connect(db_ruta)
	print(f"[OK] LanceDB conectado en '{db_ruta}'. Tablas existentes: {_listar(db)}")
	return db


def _listar(db: lancedb.DBConnection) -> list[str]:
	return list(db.list_tables().tables)


def esquema(dim: int) -> pa.Schema:
	return pa.schema([
		pa.field("id", pa.string()),
		pa.field("hash", pa.string()),
		pa.field("texto_bm25", pa.string()),
		pa.field("tags", pa.string()),
		pa.field("vector", pa.list_(pa.float32(), dim)),
	])


def hash_indexado(db: lancedb.DBConnection, nombre: str, hash_id: str) -> bool:
	"""Devuelve True si el hash ya tiene filas en la tabla."""
	if nombre not in _listar(db):
		return False
	tabla = db.open_table(nombre)
	return tabla.count_rows(f"hash = '{hash_id}'") > 0


def escribir_tabla(
	db: lancedb.DBConnection,
	nombre: str,
	filas: list[dict],
	dim: int,
	reclear: bool,
):
	existe = nombre in _listar(db)
	if existe and reclear:
		db.drop_table(nombre)
		print(f"[OK] Tabla '{nombre}' eliminada (--recrear).")
		existe = False

	if not existe:
		tabla = db.create_table(nombre, data=filas, schema=esquema(dim))
		print(f"[OK] Tabla '{nombre}' creada con {len(filas)} filas (dim={dim}).")
		_crear_indice_ann(tabla)
		_crear_indice_fts(tabla)
		return

	tabla = db.open_table(nombre)
	dim_tabla = tabla.schema.field("vector").type.list_size
	if dim_tabla != dim:
		print(
			f"[ERROR] Dim de la tabla '{nombre}' ({dim_tabla}) "
			f"no coincide con dim del lote ({dim})."
		)
		sys.exit(1)
	tabla.add(filas)
	print(f"[OK] Tabla '{nombre}': agregadas {len(filas)} filas (total={tabla.count_rows()}).")
	_crear_indice_ann(tabla)
	_crear_indice_fts(tabla)


ANN_MIN_FILAS = 50000


def _crear_indice_ann(tabla):
	n = tabla.count_rows()
	if n < ANN_MIN_FILAS:
		print(f"[INFO] Indice ANN omitido ({n} filas < {ANN_MIN_FILAS} requeridas). Se usara full scan.")
		return
	tabla.create_index(metric="cosine", replace=True)
	print(f"[OK] Indice ANN (IVF-PQ, cosine) creado ({n} filas).")


def _crear_indice_fts(tabla):
	tabla.create_fts_index(
		"texto_bm25",
		tokenizer_name="whitespace",
		with_position=False,
		replace=True,
	)
	print(f"[OK] Indice FTS (BM25) creado sobre 'texto_bm25'.")
