"""Capa fina sobre LanceDB: abrir base, definir esquema y escribir filas."""
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
		pa.field("chunk_idx", pa.int32()),
		pa.field("texto", pa.string()),
		pa.field("texto_bm25", pa.string()),
		pa.field("inicio", pa.float32()),
		pa.field("fin", pa.float32()),
		pa.field("tags", pa.string()),
		pa.field("vector", pa.list_(pa.float32(), dim)),
	])


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
		db.create_table(nombre, data=filas, schema=esquema(dim))
		print(f"[OK] Tabla '{nombre}' creada con {len(filas)} filas (dim={dim}).")
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
